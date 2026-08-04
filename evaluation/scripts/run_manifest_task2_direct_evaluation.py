#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.clients import (  # noqa: E402
    JsonCall,
    ModelContentFilterError,
    ModelResponseParseError,
    build_json_client,
)
from stage_narrative.io import atomic_write_json, load_config, load_json, sha256_file  # noqa: E402
from stage_narrative.prompt_loader import PROMPTS  # noqa: E402
from stage_narrative.task2_hybrid import (  # noqa: E402
    abstract_sensitive_screenplay_terms,
    artifact,
    read_jsonl,
    summarize_hidden_reasoning_usage,
    validate_direct_judgment,
    validate_reasoning_disabled,
    write_jsonl,
)


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate closed-book Task 2 direct answers with an independent judge."
    )
    parser.add_argument("--predictions-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Worker count and limit must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not (
        args.resume or args.preflight_only
    ):
        raise FileExistsError(f"Refusing to overwrite Task 2 direct evaluation: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_manifest = _object(args.predictions_manifest.resolve())
    if prediction_manifest.get("status") != "completed":
        raise ValueError("Direct predictions are not complete")
    _verify_artifact(prediction_manifest["predictions"])
    predictions = read_jsonl(Path(prediction_manifest["predictions"]["path"]))
    if args.limit is not None:
        predictions = predictions[: args.limit]
    questions_manifest_path = Path(prediction_manifest["questions_manifest"]["path"])
    _verify_artifact(prediction_manifest["questions_manifest"])
    questions_manifest = _object(questions_manifest_path)
    references = {}
    for variant in ("ordinary", "anonymous"):
        source = questions_manifest["questions"][variant]
        _verify_artifact(source)
        for row in read_jsonl(Path(source["path"])):
            references[(variant, row["pairing_id"])] = row

    config = load_config(args.config.resolve())
    evaluation_config = dict(config["evaluation_llm"])
    reasoning_policy = validate_reasoning_disabled(evaluation_config)
    prompt_paths = config["task2"]["direct_judge_prompt_paths"]
    compact_prompt_paths = {
        "en": "en/evaluation/task2_direct_compact_judge",
        "zh": "zh/evaluation/task2_direct_compact_judge",
    }
    materialized = [
        _materialize(
            row,
            references[(row["variant"], row["pairing_id"])],
            prompt_paths,
            compact_prompt_paths,
        )
        for row in predictions
    ]
    preflight = {
        "schema_version": "stage_task2_direct_evaluation_preflight",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed_zero_call",
        "prediction_model": prediction_manifest["prediction_model"],
        "evaluation_model": evaluation_config["model"],
        "model_variables_are_separate": True,
        "condition_and_prediction_model_hidden_from_judge_prompt": True,
        "evaluation_mode": "formal_independent_direct_answer_evaluation",
        "retrieved_passages_used_by_judge": False,
        "job_count": len(materialized),
        "workers": args.workers,
        "predictions_manifest": artifact(args.predictions_manifest.resolve()),
        "config": artifact(args.config.resolve()),
        "prompt_paths": prompt_paths,
        "reasoning_policy": reasoning_policy,
    }
    atomic_write_json(output_root / "preflight.json", preflight)
    if args.preflight_only:
        print(output_root / "preflight.json")
        return

    client = build_json_client(evaluation_config)
    semaphore = asyncio.Semaphore(args.workers)
    semantic_attempts = max(1, int(config["task2"].get("judge_semantic_attempts", 4)))
    partial_root = output_root / "partial"

    async def run(job: dict[str, Any]) -> dict[str, Any]:
        path = _partial_path(partial_root, job)
        if args.resume and path.is_file():
            payload = _object(path)
            validate_direct_judgment(payload["judgment"])
            return payload
        errors = []
        effective_user_prompt = job["user_prompt"]
        effective_system_prompt = job["system_prompt"]
        evaluation_input_mode = "full_direct_answer_judge"
        async with semaphore:
            for attempt in range(1, semantic_attempts + 1):
                try:
                    call = await _generate_with_provider_filter_fallback(
                        client=client,
                        system_prompt=effective_system_prompt,
                        user_prompt=effective_user_prompt,
                        stage=(
                            f"task2_direct_evaluation:{job['pairing_id']}:"
                            f"{job['variant']}"
                        ),
                    )
                    judgment = validate_direct_judgment(call.data)
                    payload = {
                        "schema_version": "stage_task2_direct_evaluation",
                        "pairing_id": job["pairing_id"],
                        "question_id": job["question_id"],
                        "movie_id": job["movie_id"],
                        "language": job["language"],
                        "variant": job["variant"],
                        "question_type": job["question_type"],
                        "candidate_answer": job["candidate_answer"],
                        "judgment": judgment,
                        "answer_correct": judgment["answer_correct"],
                        "call_metadata": call.metadata,
                        "semantic_attempt": attempt,
                        "evaluation_input_mode": evaluation_input_mode,
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
                    if attempt == 1:
                        effective_user_prompt = abstract_sensitive_screenplay_terms(
                            job["user_prompt"]
                        )
                        evaluation_input_mode = "safety_abstracted_full_judge"
                    else:
                        effective_system_prompt = job["compact_system_prompt"]
                        effective_user_prompt = abstract_sensitive_screenplay_terms(
                            job["compact_user_prompt"]
                        )
                        evaluation_input_mode = "compact_answer_equivalence"
        raise RuntimeError(
            f"Direct evaluation failed after {semantic_attempts} attempts for "
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
            f"Task 2 direct evaluation failed for {len(failures)} jobs; first: {failures[0]}"
        )
    rows = [row for row in settled if isinstance(row, dict)]
    rows.sort(key=lambda row: (row["variant"], row["movie_id"], row["question_id"]))
    reasoning = summarize_hidden_reasoning_usage(rows)
    if not reasoning["verified_zero"]:
        atomic_write_json(
            output_root / "failure_manifest.json",
            {
                "status": "failed_nonzero_reasoning",
                "failure_count": len(rows),
                "evaluation_model": evaluation_config["model"],
                "reasoning_policy": reasoning_policy,
                "returned_reasoning": reasoning,
            },
        )
        raise RuntimeError(f"Evaluation endpoint returned hidden reasoning: {reasoning}")

    evaluations_path = output_root / "evaluations.jsonl"
    write_jsonl(evaluations_path, rows)
    results_path = output_root / "results.json"
    atomic_write_json(results_path, _summarize(rows))
    manifest_path = output_root / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task2_direct_evaluation_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "completed",
            "prediction_model": prediction_manifest["prediction_model"],
            "evaluation_model": evaluation_config["model"],
            "model_variables_are_separate": True,
            "predictions_manifest": artifact(args.predictions_manifest.resolve()),
            "config": artifact(args.config.resolve()),
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
                "verified_disabled": True,
            },
            "usage": _usage(rows),
            "evaluations": artifact(evaluations_path),
            "results": artifact(results_path),
        },
    )
    print(manifest_path)


async def _generate_with_provider_filter_fallback(
    *, client: Any, system_prompt: str, user_prompt: str, stage: str
) -> JsonCall:
    try:
        call = await client.generate_json(
            system_prompt=system_prompt, user_prompt=user_prompt, stage=stage
        )
        if call.metadata.get("finish_reason") == "content_filter":
            raise ModelContentFilterError(f"{stage} returned a filtered completion")
        return call
    except ModelContentFilterError:
        fallback = await client.generate_json(
            system_prompt=(
                system_prompt
                + "\nThis is safety-conscious correctness classification of concise "
                "text about already-written fiction. Do not recreate sensitive content."
            ),
            user_prompt=abstract_sensitive_screenplay_terms(user_prompt),
            stage=f"{stage}:safety_abstracted_provider_filter_fallback",
        )
        if fallback.metadata.get("finish_reason") == "content_filter":
            raise ModelContentFilterError(f"{stage} fallback was filtered")
        return JsonCall(
            data=fallback.data,
            metadata={
                **fallback.metadata,
                "provider_refusal_fallback": {
                    "trigger": "provider_content_filter",
                    "fallback_input_mode": "safety_abstracted_answer_judge",
                    "judge_model_unchanged": True,
                    "retrieved_passages_added": False,
                },
            },
        )


def _materialize(
    prediction: dict[str, Any],
    reference: dict[str, Any],
    prompt_paths: dict[str, str],
    compact_prompt_paths: dict[str, str],
) -> dict[str, Any]:
    if prediction["question"] != reference["question"]:
        raise ValueError(f"Prediction question drift: {prediction['pairing_id']}")
    candidate = prediction["prediction"]["answer"]
    system, user = PROMPTS.render(
        prompt_paths[prediction["language"]],
        question=reference["question"],
        reference_answer=reference["reference_answer"],
        reference_evidence_or_reason=reference["reference_evidence_or_reason"],
        candidate_answer=candidate,
    )
    compact_system, compact_user = PROMPTS.render(
        compact_prompt_paths[prediction["language"]],
        reference_answer=reference["reference_answer"],
        candidate_answer=candidate,
    )
    return {
        "pairing_id": prediction["pairing_id"],
        "question_id": prediction["question_id"],
        "movie_id": prediction["movie_id"],
        "language": prediction["language"],
        "variant": prediction["variant"],
        "question_type": prediction["question_type"],
        "candidate_answer": candidate,
        "system_prompt": system,
        "user_prompt": user,
        "compact_system_prompt": compact_system,
        "compact_user_prompt": compact_user,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    return {
        "schema_version": "stage_task2_direct_results",
        "status": "completed",
        "variants": {
            variant: {
                "count": len(items),
                "correct": sum(row["answer_correct"] for row in items),
                "accuracy": sum(row["answer_correct"] for row in items) / len(items),
            }
            for variant, items in sorted(by_variant.items())
        },
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
