#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.clients import (  # noqa: E402
    JsonCall,
    ModelContentFilterError,
    ModelResponseParseError,
    build_endpoint_pool_runtime,
    build_json_client,
)
from stage_narrative.io import atomic_write_json, load_config, load_json, sha256_file  # noqa: E402
from stage_narrative.prompt_loader import PROMPTS  # noqa: E402
from stage_narrative.task2_hybrid import (  # noqa: E402
    abstract_sensitive_screenplay_terms,
    artifact,
    read_jsonl,
    summarize_hidden_reasoning_usage,
    validate_direct_prediction,
    validate_reasoning_disabled,
    write_jsonl,
)


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Run closed-book Task 2 predictions from question text only."
    )
    parser.add_argument("--questions-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-provider-forced-reasoning",
        action="store_true",
        help=(
            "Allow a required model that ignores the explicit disable request to return "
            "hidden reasoning; the manifest remains noncompliant and records exact usage."
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Worker count and limit must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not (
        args.resume or args.preflight_only
    ):
        raise FileExistsError(f"Refusing to overwrite Task 2 direct predictions: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    questions_manifest = _object(args.questions_manifest.resolve())
    if questions_manifest.get("status") != "completed":
        raise ValueError("Direct-question manifest is not complete")
    contract = questions_manifest.get("input_contract") or {}
    expected_contract = {
        "prediction_visible_fields": ["question"],
        "screenplay_provided": False,
        "retrieved_passages_provided": False,
        "memory_provided": False,
        "scene_metadata_provided": False,
        "reference_answer_provided": False,
        "reference_evidence_provided": False,
        "rag_used": False,
    }
    if contract != expected_contract:
        raise ValueError(f"Direct-question input contract drift: {contract}")
    jobs = []
    for variant in ("ordinary", "anonymous"):
        source = questions_manifest["questions"][variant]
        _verify_artifact(source)
        jobs.extend(read_jsonl(Path(source["path"])))
    jobs.sort(key=lambda row: (row["variant"], row["movie_id"], row["question_id"]))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    config = load_config(args.config.resolve())
    prediction_config = dict(config["prediction_llm"])
    reasoning_policy = validate_reasoning_disabled(prediction_config)
    prompt_paths = config["task2"]["direct_prediction_prompt_paths"]
    materialized = [_materialize(job, prompt_paths) for job in jobs]
    preflight = {
        "schema_version": "stage_task2_direct_prediction_preflight",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed_zero_call",
        "prediction_model": prediction_config["model"],
        "model_role": "prediction_llm",
        "questions_manifest": artifact(args.questions_manifest.resolve()),
        "config": artifact(args.config.resolve()),
        "job_count": len(materialized),
        "workers": args.workers,
        "prompt_paths": prompt_paths,
        "prediction_visible_fields": ["question"],
        "screenplay_provided": False,
        "retrieved_passages_provided": False,
        "memory_provided": False,
        "scene_metadata_provided": False,
        "gold_answer_exposed_to_prediction": False,
        "rag_used": False,
        "reasoning_policy": reasoning_policy,
        "provider_forced_reasoning_allowed": args.allow_provider_forced_reasoning,
    }
    atomic_write_json(output_root / "preflight.json", preflight)
    if args.preflight_only:
        print(output_root / "preflight.json")
        return

    endpoint_runtime = build_endpoint_pool_runtime(prediction_config)
    client = build_json_client(prediction_config, endpoint_runtime=endpoint_runtime)
    semaphore = asyncio.Semaphore(args.workers)
    semantic_attempts = max(
        1, int(config["task2"].get("prediction_semantic_attempts", 2))
    )
    partial_root = output_root / "partial"

    async def run(job: dict[str, Any]) -> dict[str, Any]:
        path = _partial_path(partial_root, job)
        if args.resume and path.is_file():
            payload = _object(path)
            validate_direct_prediction(payload["prediction"])
            return payload
        errors = []
        effective_user_prompt = job["user_prompt"]
        async with semaphore:
            for attempt in range(1, semantic_attempts + 1):
                try:
                    call = await _generate_with_content_filter_fallback(
                        client=client,
                        system_prompt=job["system_prompt"],
                        user_prompt=effective_user_prompt,
                        stage=(
                            f"task2_direct_prediction:{job['pairing_id']}:"
                            f"{job['variant']}"
                        ),
                    )
                    prediction = validate_direct_prediction(call.data)
                    payload = {
                        "schema_version": "stage_task2_direct_prediction",
                        "pairing_id": job["pairing_id"],
                        "question_id": job["question_id"],
                        "movie_id": job["movie_id"],
                        "language": job["language"],
                        "variant": job["variant"],
                        "question": job["question"],
                        "question_type": job["question_type"],
                        "prediction": prediction,
                        "call_metadata": call.metadata,
                        "semantic_attempt": attempt,
                        "prediction_visible_fields": ["question"],
                    }
                    atomic_write_json(path, payload)
                    return payload
                except (ModelResponseParseError, ValueError) as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    prior = (
                        exc.raw_text
                        if isinstance(exc, ModelResponseParseError)
                        else json.dumps(
                            call.data if "call" in locals() else {}, ensure_ascii=False
                        )
                    )
                    effective_user_prompt = (
                        job["user_prompt"]
                        + "\n\nThe previous response failed strict JSON validation: "
                        + str(exc)
                        + "\nReturn a corrected JSON object with only the answer field."
                        + "\n<invalid_previous_response>"
                        + prior
                        + "</invalid_previous_response>"
                    )
        raise RuntimeError(
            f"Direct prediction failed after {semantic_attempts} attempts for "
            f"{job['pairing_id']} {job['variant']}: {' | '.join(errors)}"
        )

    settled = await asyncio.gather(*(run(job) for job in materialized), return_exceptions=True)
    failures = [row for row in settled if isinstance(row, BaseException)]
    if failures:
        atomic_write_json(
            output_root / "failure_manifest.json",
            {
                "status": "failed",
                "failure_count": len(failures),
                "first_failure": f"{type(failures[0]).__name__}: {failures[0]}",
            },
        )
        raise RuntimeError(
            f"Task 2 direct predictions failed for {len(failures)} jobs; first: {failures[0]}"
        )
    rows = [row for row in settled if isinstance(row, dict)]
    rows.sort(key=lambda row: (row["variant"], row["movie_id"], row["question_id"]))
    reasoning = summarize_hidden_reasoning_usage(rows)
    if not reasoning["verified_zero"] and not args.allow_provider_forced_reasoning:
        atomic_write_json(
            output_root / "failure_manifest.json",
            {
                "status": "failed_nonzero_reasoning",
                "failure_count": len(rows),
                "prediction_model": prediction_config["model"],
                "reasoning_policy": reasoning_policy,
                "returned_reasoning": reasoning,
            },
        )
        raise RuntimeError(f"Prediction endpoint returned hidden reasoning: {reasoning}")
    predictions_path = output_root / "predictions.jsonl"
    write_jsonl(predictions_path, rows)
    endpoint_snapshot = await endpoint_runtime.snapshot() if endpoint_runtime else None
    manifest_path = output_root / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task2_direct_prediction_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "completed",
            "prediction_model": prediction_config["model"],
            "model_role": "prediction_llm",
            "questions_manifest": artifact(args.questions_manifest.resolve()),
            "config": artifact(args.config.resolve()),
            "input_contract": expected_contract,
            "counts": {
                "expected": len(materialized),
                "completed": len(rows),
                "ordinary": sum(row["variant"] == "ordinary" for row in rows),
                "anonymous": sum(row["variant"] == "anonymous" for row in rows),
                "failure_count": 0,
            },
            "reasoning": {
                **reasoning_policy,
                "returned": reasoning,
                "verified_disabled": reasoning["verified_zero"],
                "provider_forced_reasoning_allowed": (
                    args.allow_provider_forced_reasoning
                ),
                "compliance_status": (
                    "verified_disabled"
                    if reasoning["verified_zero"]
                    else "provider_forced_reasoning_observed"
                ),
            },
            "endpoint_pool": endpoint_snapshot,
            "usage": _usage(rows),
            "predictions": artifact(predictions_path),
        },
    )
    print(manifest_path)


async def _generate_with_content_filter_fallback(
    *, client: Any, system_prompt: str, user_prompt: str, stage: str
) -> JsonCall:
    primary_metadata = None
    try:
        call = await client.generate_json(
            system_prompt=system_prompt, user_prompt=user_prompt, stage=stage
        )
        if call.metadata.get("finish_reason") == "content_filter":
            primary_metadata = call.metadata
            raise ModelContentFilterError(f"{stage} returned a filtered completion")
        return call
    except ModelContentFilterError:
        fallback_system = (
            system_prompt
            + "\nThis is a safety-conscious, closed-book analysis of a question about "
            "already-written fictional material. Return only the requested concise "
            "answer abstraction; do not recreate sensitive material."
        )
        try:
            fallback = await client.generate_json(
                system_prompt=fallback_system,
                user_prompt=user_prompt,
                stage=f"{stage}:safety_framed_content_filter_fallback",
            )
            mode = "safety_framed_question_only"
            withheld_fields: list[str] = []
        except ModelContentFilterError:
            fallback = await client.generate_json(
                system_prompt=fallback_system,
                user_prompt=abstract_sensitive_screenplay_terms(user_prompt),
                stage=f"{stage}:safety_abstracted_content_filter_fallback",
            )
            mode = "safety_abstracted_question_only"
            withheld_fields = ["filter_prone_surface_wording"]
        if fallback.metadata.get("finish_reason") == "content_filter":
            raise ModelContentFilterError(f"{stage} fallback was filtered")
        return JsonCall(
            data=fallback.data,
            metadata={
                **fallback.metadata,
                "content_filter_fallback": {
                    "trigger": "provider_content_filter",
                    "primary_response_metadata": primary_metadata,
                    "fallback_input_mode": mode,
                    "withheld_fields": withheld_fields,
                    "rag_context_added": False,
                },
            },
        )


def _materialize(row: dict[str, Any], prompt_paths: dict[str, str]) -> dict[str, Any]:
    system, user = PROMPTS.render(prompt_paths[row["language"]], question=row["question"])
    return {
        "pairing_id": row["pairing_id"],
        "question_id": row["question_id"],
        "movie_id": row["movie_id"],
        "language": row["language"],
        "variant": row["variant"],
        "question": row["question"],
        "question_type": row["question_type"],
        "system_prompt": system,
        "user_prompt": user,
    }


def _partial_path(root: Path, job: dict[str, Any]) -> Path:
    question = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(job["question_id"]))
    return root / job["variant"] / job["movie_id"] / f"{question}.json"


def _usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(row["call_metadata"].get(field) or 0) for row in rows)
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "provider_thought_tokens",
        )
    }


def _verify_artifact(row: dict[str, Any]) -> None:
    path = Path(row["path"])
    if not path.is_file() or sha256_file(path) != row["sha256"]:
        raise ValueError(f"Artifact missing or hash mismatch: {path}")


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    asyncio.run(main_async())
