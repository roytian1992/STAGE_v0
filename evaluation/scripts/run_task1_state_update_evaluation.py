#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.clients import (  # noqa: E402
    ModelResponseParseError,
    build_endpoint_pool_runtime,
    build_json_client,
)
from stage_narrative.chunking import chunk_scene  # noqa: E402
from stage_narrative.evaluation.aggregation import mean_defined  # noqa: E402
from stage_narrative.evaluation.task1_metrics import (  # noqa: E402
    localize_task1_prediction,
)
from stage_narrative.io import (  # noqa: E402
    atomic_write_json,
    load_json,
    load_scenes,
    sha256_file,
    sha256_json,
)
from stage_narrative.models import stable_id  # noqa: E402
from stage_narrative.prompt_loader import PROMPTS  # noqa: E402
from stage_narrative.task2_hybrid import keyword_tokens  # noqa: E402
from stage_narrative.temporal.benchmark_protocol import (  # noqa: E402
    BenchmarkRuntimeConfig,
)


COVERAGE_PROMPT_NAME = "task1_state_coverage_judge"
EVIDENCE_PROMPT_NAME = "task1_prediction_validity_judge"
EVIDENCE_EXCERPT_TOKENS = 100
EVIDENCE_EXCERPT_TOP_K = 1
EVIDENCE_POLICY = "claim_bm25_top1_100_tokens_stopwords_removed"
ENGLISH_STOPWORDS = frozenset(
    "a an the and or but if then else of to in on at for from by with as is are was "
    "were be been being it its this that these those he she they them his her their "
    "who whom which what when where why how not no do does did doing have has had "
    "having can could should would may might must will just even such about into over "
    "after before during while than so very right".split()
)


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate paired Task 1 state-update predictions with an independent judge."
    )
    parser.add_argument("--reference-asset", type=Path, required=True)
    parser.add_argument("--prediction-run", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("--workers must be in 1..64")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite evaluation run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    reference_path = args.reference_asset.resolve()
    prediction_run = args.prediction_run.resolve()
    script_path = args.script.resolve()
    reference_asset = _object(reference_path)
    language = str(reference_asset.get("language") or "")
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported Task 1 language: {language}")
    if sha256_file(script_path) != reference_asset["script"]["sha256"]:
        raise ValueError("Evaluation screenplay differs from the reference asset")
    predictions = {
        setting: _object(prediction_run / f"{setting}_predictions.json")
        for setting in ("reference_state_update", "autoregressive_state_update")
    }
    prediction_manifest = _object(prediction_run / "manifest.json")
    if prediction_manifest.get("previous_state_projection") != "claim_text_only":
        raise ValueError("Formal evaluation requires claim-only previous-state prompts")
    memory_index = _memory_index(_object(prediction_run / "shared_entity_memory.json"))
    reference_index = _reference_index(reference_asset)
    for setting, payload in predictions.items():
        if payload.get("setting") != setting:
            raise ValueError(f"Prediction setting mismatch: {setting}")
        if set(_prediction_index(payload)) != set(reference_index):
            raise ValueError(f"Prediction coverage differs from reference: {setting}")
    if set(memory_index) != set(reference_index):
        raise ValueError("Shared entity memory coverage differs from reference")

    config = BenchmarkRuntimeConfig.load(args.config.resolve())
    if config.evaluation_llm["model"] == prediction_manifest["prediction_model"]:
        raise ValueError("Formal state_update evaluation requires an independent judge model")
    counter = config.build_token_counter()
    scenes = {scene.order: scene for scene in load_scenes(script_path)}
    preflight = _preflight(
        reference_index=reference_index,
        predictions=predictions,
        scenes=scenes,
        config=config,
        counter=counter,
    )
    if args.dry_run:
        fallback_config = _fallback_llm_config(config)
        payload = {
            "schema": "stage_task1_state_update_evaluation_preflight",
            "status": "passed_zero_call",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "prediction_model": prediction_manifest["prediction_model"],
            "evaluation_model": config.evaluation_llm["model"],
            "evaluation_fallback_model": fallback_config["model"],
            "inputs": {
                "reference_asset": _artifact(reference_path),
                "prediction_manifest": _artifact(prediction_run / "manifest.json"),
                "shared_entity_memory": _artifact(
                    prediction_run / "shared_entity_memory.json"
                ),
                "script": _artifact(script_path),
                "config": _artifact(args.config.resolve()),
            },
            "prompt_artifacts": _prompt_artifacts(language),
            **preflight,
            "semantic_calls": 0,
        }
        atomic_write_json(output_root / "preflight.json", payload)
        print(output_root / "preflight.json")
        print(payload["counts"])
        return

    model_config = {
        **config.evaluation_llm,
        "json_response_format": True,
        "max_tokens": config.call_budgets["task1_judge"].max_output_tokens,
    }
    endpoint_runtime = build_endpoint_pool_runtime(model_config)
    client = build_json_client(model_config, endpoint_runtime=endpoint_runtime)
    fallback_config = {
        **_fallback_llm_config(config),
        "json_response_format": True,
        "max_tokens": config.call_budgets["task1_judge"].max_output_tokens,
    }
    fallback_endpoint_runtime = build_endpoint_pool_runtime(fallback_config)
    fallback_client = build_json_client(
        fallback_config, endpoint_runtime=fallback_endpoint_runtime
    )
    semaphore = asyncio.Semaphore(args.workers)
    setting_results = {}
    for setting, payload in predictions.items():
        prediction_index = _prediction_index(payload)
        settled = await asyncio.gather(
            *[
                _evaluate_checkpoint(
                    setting=setting,
                    reference=reference_index[checkpoint_id],
                    prediction=prediction_index[checkpoint_id],
                    scenes=scenes,
                    client=client,
                    fallback_client=fallback_client,
                    semaphore=semaphore,
                    config=config,
                    counter=counter,
                    output_root=output_root,
                    resume=args.resume,
                )
                for checkpoint_id in sorted(reference_index)
            ]
        )
        evaluation = {
            "schema": "stage_task1_state_update_evaluation",
            "setting": setting,
            "movie_id": reference_asset["movie_id"],
            "checkpoint_count": len(settled),
            "aggregate": _aggregate(settled),
            "checkpoints": settled,
        }
        path = output_root / f"{setting}_evaluation.json"
        atomic_write_json(path, evaluation)
        setting_results[setting] = evaluation

    reference_headline = setting_results["reference_state_update"]["aggregate"]["movie_macro"]
    prediction_headline = setting_results["autoregressive_state_update"]["aggregate"]["movie_macro"]
    summary = {
        "schema": "stage_task1_state_update_comparison",
        "movie_id": reference_asset["movie_id"],
        "reference_state_update": reference_headline,
        "autoregressive_state_update": prediction_headline,
        "accumulation_gap": {
            "current_state_reference_coverage": _difference(
                reference_headline["current_state"]["reference_coverage"],
                prediction_headline["current_state"]["reference_coverage"],
            ),
            "development_reference_coverage": _difference(
                reference_headline["development"]["reference_coverage"],
                prediction_headline["development"]["reference_coverage"],
            ),
        },
        "metric_policy": {
            "primary": "reference_centric_set_coverage",
            "coverage_weights": {"full": 1.0, "partial": 0.5},
            "extra_predictions": "audited_separately_not_counted_as_false_positives",
            "f1_reported": False,
        },
    }
    summary_path = output_root / "summary.json"
    atomic_write_json(summary_path, summary)
    endpoint_snapshot = (
        await endpoint_runtime.snapshot() if endpoint_runtime is not None else None
    )
    fallback_endpoint_snapshot = (
        await fallback_endpoint_runtime.snapshot()
        if fallback_endpoint_runtime is not None
        else None
    )
    semantic_usage = _semantic_usage(setting_results)
    manifest = {
        "schema": "stage_task1_state_update_evaluation_run",
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prediction_model": prediction_manifest["prediction_model"],
        "evaluation_model": config.evaluation_llm["model"],
        "evaluation_fallback_model": fallback_config["model"],
        "model_variables_are_separate": True,
        "thinking_disabled": True,
        "metric_policy": "reference_centric_coverage_plus_prediction_safety_rates",
        "workers": args.workers,
        "counts": {
            **preflight["counts"],
            "completed_coverage_calls": preflight["counts"]["coverage_calls"],
            "completed_evidence_calls": preflight["counts"]["evidence_calls"],
            **semantic_usage,
            "failure_count": 0,
        },
        "evidence_policy": preflight["evidence_policy"],
        "inputs": {
            "reference_asset": _artifact(reference_path),
            "prediction_manifest": _artifact(prediction_run / "manifest.json"),
            "shared_entity_memory": _artifact(prediction_run / "shared_entity_memory.json"),
            "script": _artifact(script_path),
            "config": _artifact(args.config.resolve()),
        },
        "outputs": {
            "reference_state_update": _artifact(output_root / "reference_state_update_evaluation.json"),
            "autoregressive_state_update": _artifact(output_root / "autoregressive_state_update_evaluation.json"),
            "summary": _artifact(summary_path),
        },
        "prompt_artifacts": _prompt_artifacts(language),
        "endpoint_pool": endpoint_snapshot,
        "fallback_endpoint_pool": fallback_endpoint_snapshot,
        "fallbacks": {
            "content_filter": "Qwen3 fallback only after exhausted DeepSeek semantic retries",
            "semantic": "same-prompt retries are recorded per judgment",
        },
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(output_root / "manifest.json")
    print(summary)


async def _evaluate_checkpoint(
    *, setting: str, reference: dict[str, Any], prediction: dict[str, Any],
    scenes: dict[int, Any], client: Any, fallback_client: Any,
    semaphore: asyncio.Semaphore,
    config: BenchmarkRuntimeConfig, counter: Any, output_root: Path, resume: bool,
) -> dict[str, Any]:
    localized_reference = _localize_reference(reference["evaluation_target"])
    localized_prediction = localize_task1_prediction(prediction["prediction"])
    coverage_prompt = _render_coverage(
        reference=reference, localized_reference=localized_reference,
        localized_prediction=localized_prediction, config=config, counter=counter,
    )
    record_id = stable_id("task1-state_update-evaluation", setting, reference["checkpoint_id"])
    checkpoint_path = output_root / "checkpoints" / setting / f"{record_id}.json"
    if resume and checkpoint_path.is_file():
        record = _object(checkpoint_path)
        if record.get("coverage_prompt_sha256") != coverage_prompt["prompt_sha256"]:
            raise ValueError(f"Evaluation checkpoint prompt drift: {checkpoint_path}")
        if record.get("evidence_policy") != EVIDENCE_POLICY:
            raise ValueError(f"Evaluation checkpoint evidence-policy drift: {checkpoint_path}")
        return record
    coverage_judgments, coverage_metadata = await _generate_validated(
        client=client,
        semaphore=semaphore,
        system_prompt=coverage_prompt["system_prompt"],
        user_prompt=coverage_prompt["user_prompt"],
        stage=f"task1_state_update_coverage:{setting}:{record_id}",
        semantic_attempts=_semantic_attempts(config),
        fallback_client=fallback_client,
        fallback_semantic_attempts=_fallback_semantic_attempts(config),
        validator=lambda payload: _validate_coverage(
            payload,
            reference_ids={row["local_id"] for row in localized_reference},
            prediction_ids={row["local_id"] for row in localized_prediction},
        ),
    )
    checks = await asyncio.gather(
        *[
            _judge_evidence(
                setting=setting, reference=reference, prediction=prediction,
                row=row, scenes=scenes,
                client=client, fallback_client=fallback_client,
                semaphore=semaphore, config=config, counter=counter,
            )
            for row in localized_prediction
        ]
    )
    scoring = _score(
        localized_reference, localized_prediction, coverage_judgments, checks
    )
    record = {
        "trajectory_id": reference["trajectory_id"],
        "character": reference["character"],
        "checkpoint_id": reference["checkpoint_id"],
        "instance_id": reference["instance_id"],
        "localized_reference": localized_reference,
        "localized_prediction": localized_prediction,
        "reference_coverage_judgments": coverage_judgments,
        "prediction_checks": checks,
        "scoring": scoring,
        "evidence_policy": EVIDENCE_POLICY,
        "coverage_prompt_sha256": coverage_prompt["prompt_sha256"],
        "coverage_generator_metadata": coverage_metadata,
    }
    atomic_write_json(checkpoint_path, record)
    return record


async def _judge_evidence(
    *, setting: str, reference: dict[str, Any], prediction: dict[str, Any],
    row: dict[str, Any], scenes: dict[int, Any],
    client: Any, fallback_client: Any, semaphore: asyncio.Semaphore,
    config: BenchmarkRuntimeConfig, counter: Any,
) -> dict[str, Any]:
    evidence = _evidence_excerpts(
        reference=reference,
        prediction=prediction,
        row=row,
        scenes=scenes,
        counter=counter,
    )
    interval = reference["screenplay_interval"]
    system, user = PROMPTS.render(
        _prompt_key(reference["language"], EVIDENCE_PROMPT_NAME),
        character=reference["character"],
        previous_scene_order=interval["start_scene_order_exclusive"],
        current_scene_order=interval["end_scene_order_inclusive"],
        predicted_claim={
            "type": row["prediction_type"], "claim": row["claim"],
            "evidence_scene_orders": row["evidence_scene_orders"],
        },
        evidence_scenes=evidence, prediction_id=row["local_id"],
    )
    rendered = _budgeted(system, user, config=config, counter=counter)
    check, metadata = await _generate_validated(
        client=client,
        semaphore=semaphore,
        system_prompt=system,
        user_prompt=user,
        stage=f"task1_state_update_evidence:{setting}:{reference['checkpoint_id']}:{row['local_id']}",
        semantic_attempts=_semantic_attempts(config),
        fallback_client=fallback_client,
        fallback_semantic_attempts=_fallback_semantic_attempts(config),
        validator=lambda payload: _validate_check(payload, expected_id=row["local_id"]),
    )
    check["prompt_sha256"] = rendered["prompt_sha256"]
    check["generator_metadata"] = metadata
    return check


async def _generate_validated(
    *, client: Any, semaphore: asyncio.Semaphore, system_prompt: str,
    user_prompt: str, stage: str, semantic_attempts: int, validator: Any,
    fallback_client: Any | None = None, fallback_semantic_attempts: int = 0,
) -> tuple[Any, dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    primary = await _attempt_validated_route(
        client=client,
        semaphore=semaphore,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage=stage,
        semantic_attempts=semantic_attempts,
        validator=validator,
        route="deepseek_primary",
        rejected=rejected,
    )
    if primary is not None:
        validated, metadata = primary
        return validated, {
            **metadata,
            "judge_route": "deepseek_primary",
            "received_semantic_calls": int(metadata["semantic_attempt"]),
            "primary_rejected_semantic_calls": len(rejected),
            "fallback_received_semantic_calls": 0,
            "rejected_semantic_attempts": rejected,
        }
    if fallback_client is None or fallback_semantic_attempts <= 0:
        raise ValueError(
            f"{stage} failed semantic validation after {semantic_attempts} attempts: "
            f"{rejected}"
        )
    primary_rejected = len(rejected)
    fallback = await _attempt_validated_route(
        client=fallback_client,
        semaphore=semaphore,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage=f"{stage}:qwen_fallback",
        semantic_attempts=fallback_semantic_attempts,
        validator=validator,
        route="qwen_fallback",
        rejected=rejected,
    )
    if fallback is None:
        raise ValueError(
            f"{stage} failed DeepSeek primary and Qwen fallback semantic validation: "
            f"{rejected}"
        )
    validated, metadata = fallback
    fallback_received = int(metadata["semantic_attempt"])
    return validated, {
        **metadata,
        "judge_route": "qwen_fallback",
        "received_semantic_calls": primary_rejected + fallback_received,
        "primary_rejected_semantic_calls": primary_rejected,
        "fallback_received_semantic_calls": fallback_received,
        "rejected_semantic_attempts": rejected,
    }


async def _attempt_validated_route(
    *, client: Any, semaphore: asyncio.Semaphore, system_prompt: str,
    user_prompt: str, stage: str, semantic_attempts: int, validator: Any,
    route: str, rejected: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]] | None:
    for semantic_attempt in range(1, semantic_attempts + 1):
        try:
            async with semaphore:
                response = await client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    stage=stage,
                )
        except ModelResponseParseError as exc:
            _assert_no_reasoning(exc.metadata)
            rejected.append(
                {
                    "judge_route": route,
                    "semantic_attempt": semantic_attempt,
                    "error_type": type(exc).__name__,
                    "model": exc.metadata.get("model"),
                    "finish_reason": exc.metadata.get("finish_reason"),
                    "response_parse_error": exc.metadata.get("response_parse_error"),
                    "raw_response_chars": exc.metadata.get("raw_response_chars"),
                    "raw_response_sha256": exc.metadata.get("raw_response_sha256"),
                }
            )
            continue
        _assert_no_reasoning(response.metadata)
        try:
            validated = validator(response.data)
        except ValueError as exc:
            rejected.append(
                {
                    "judge_route": route,
                    "semantic_attempt": semantic_attempt,
                    "error_type": type(exc).__name__,
                    "model": response.metadata.get("model"),
                    "validation_error": str(exc),
                }
            )
        else:
            return validated, {
                **response.metadata,
                "semantic_attempt": semantic_attempt,
                "semantic_attempts_configured": semantic_attempts,
            }
    return None


def _semantic_attempts(config: BenchmarkRuntimeConfig) -> int:
    return max(1, int(config.evaluation_llm.get("semantic_attempts", 3)))


def _fallback_llm_config(config: BenchmarkRuntimeConfig) -> dict[str, Any]:
    payload = config.evaluation_llm.get("fallback_llm")
    if not isinstance(payload, dict) or not payload.get("model"):
        raise ValueError("Task 1 evaluation requires evaluation_llm.fallback_llm")
    return dict(payload)


def _fallback_semantic_attempts(config: BenchmarkRuntimeConfig) -> int:
    return max(1, int(_fallback_llm_config(config).get("semantic_attempts", 3)))


def _evidence_excerpts(
    *, reference: dict[str, Any], prediction: dict[str, Any], row: dict[str, Any],
    scenes: dict[int, Any], counter: Any,
) -> list[dict[str, Any]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("Install rank-bm25 to select Task 1 evidence excerpts") from exc
    query_tokens = _evidence_tokens(row["claim"], reference["language"])
    evidence = []
    for order in _candidate_evidence_scene_orders(prediction=prediction, row=row):
        scene = scenes.get(order)
        if scene is None:
            continue
        chunks = chunk_scene(
            movie_id=reference["trajectory_id"],
            scene=scene,
            token_counter=counter,
            max_content_tokens=EVIDENCE_EXCERPT_TOKENS,
        )
        corpus = [_evidence_tokens(chunk.text, reference["language"]) for chunk in chunks]
        scores = BM25Okapi(corpus).get_scores(query_tokens)
        selected = sorted(
            sorted(range(len(chunks)), key=lambda index: (-float(scores[index]), index))[
                :EVIDENCE_EXCERPT_TOP_K
            ]
        )
        excerpts = []
        for index in selected:
            excerpts.append(f"[Excerpt {index + 1}]\n{chunks[index].text}")
        evidence.append(
            {
                "scene_order": order,
                "title": scene.title,
                "content": "\n\n".join(excerpts),
            }
        )
    return evidence


def _evidence_tokens(text: str, language: str) -> list[str]:
    tokens = keyword_tokens(text, language)
    if language == "en":
        return [token for token in tokens if token not in ENGLISH_STOPWORDS]
    return tokens


def _preflight(
    *, reference_index: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]],
    scenes: dict[int, Any], config: BenchmarkRuntimeConfig, counter: Any,
) -> dict[str, Any]:
    coverage_calls = evidence_calls = 0
    maximum = {"tokens": -1}
    for setting, payload in predictions.items():
        prediction_index = _prediction_index(payload)
        for checkpoint_id, reference in reference_index.items():
            localized_reference = _localize_reference(reference["evaluation_target"])
            prediction_record = prediction_index[checkpoint_id]
            localized_prediction = localize_task1_prediction(prediction_record["prediction"])
            rendered = _render_coverage(
                reference=reference, localized_reference=localized_reference,
                localized_prediction=localized_prediction, config=config, counter=counter,
            )
            coverage_calls += 1
            maximum = _maximum(maximum, rendered, setting, checkpoint_id, "coverage")
            for row in localized_prediction:
                evidence_calls += 1
                evidence = _evidence_excerpts(
                    reference=reference,
                    prediction=prediction_record,
                    row=row,
                    scenes=scenes,
                    counter=counter,
                )
                interval = reference["screenplay_interval"]
                system, user = PROMPTS.render(
                    _prompt_key(reference["language"], EVIDENCE_PROMPT_NAME),
                    character=reference["character"],
                    previous_scene_order=interval["start_scene_order_exclusive"],
                    current_scene_order=interval["end_scene_order_inclusive"],
                    predicted_claim={"type": row["prediction_type"], "claim": row["claim"], "evidence_scene_orders": row["evidence_scene_orders"]},
                    evidence_scenes=evidence, prediction_id=row["local_id"],
                )
                check = _budgeted(system, user, config=config, counter=counter)
                maximum = _maximum(maximum, check, setting, checkpoint_id, row["local_id"])
    return {
        "counts": {
            "settings": 2,
            "checkpoints_per_setting": len(reference_index),
            "coverage_calls": coverage_calls,
            "evidence_calls": evidence_calls,
            "planned_semantic_calls": coverage_calls + evidence_calls,
        },
        "maximum_prompt": maximum,
        "evidence_policy": {
            "name": EVIDENCE_POLICY,
            "excerpt_content_tokens": EVIDENCE_EXCERPT_TOKENS,
            "excerpts_per_scene": EVIDENCE_EXCERPT_TOP_K,
            "ranking": "BM25 against the predicted claim",
            "content_omission": "none",
        },
    }


def _reference_index(asset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for trajectory in asset["trajectories"]:
        for checkpoint in trajectory["checkpoints"]:
            output[checkpoint["checkpoint_id"]] = {
                **checkpoint,
                "trajectory_id": trajectory["trajectory_id"],
                "character": trajectory["character"],
                "language": asset["language"],
            }
    return output


def _prediction_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        checkpoint["checkpoint_id"]: checkpoint
        for trajectory in payload["trajectories"]
        for checkpoint in trajectory["checkpoints"]
    }


def _memory_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("status") != "completed" or not isinstance(payload.get("slots"), list):
        raise ValueError("Shared entity memory is incomplete")
    output = {row["checkpoint_id"]: row for row in payload["slots"]}
    if len(output) != len(payload["slots"]):
        raise ValueError("Shared entity memory contains duplicate checkpoints")
    return output


def _candidate_evidence_scene_orders(
    *, prediction: dict[str, Any], row: dict[str, Any]
) -> list[int]:
    values = set(row.get("evidence_scene_orders", []))
    if not values and row.get("prediction_type") == "current_state":
        values.update(prediction.get("previous_state_evidence_scene_orders", []))
    return sorted(
        value
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    )


def _localize_reference(target: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for field, kind in (
        ("current_state", "current_state"),
        ("developments_since_previous_checkpoint", "development"),
    ):
        for row in target[field]:
            output.append(
                {
                    "local_id": f"R{len(output) + 1}",
                    "claim_type": kind,
                    "claim": row["claim"],
                    "evidence_scene_orders": row["evidence_scene_orders"],
                }
            )
    return output


def _render_coverage(
    *, reference: dict[str, Any], localized_reference: list[dict[str, Any]],
    localized_prediction: list[dict[str, Any]], config: BenchmarkRuntimeConfig, counter: Any,
) -> dict[str, Any]:
    interval = reference["screenplay_interval"]
    compact_reference = [
        [row["local_id"], "s" if row["claim_type"] == "current_state" else "d", row["claim"]]
        for row in localized_reference
    ]
    compact_prediction = [
        [row["local_id"], "s" if row["prediction_type"] == "current_state" else "d", row["claim"]]
        for row in localized_prediction
    ]
    system, user = PROMPTS.render(
        _prompt_key(reference["language"], COVERAGE_PROMPT_NAME),
        character=reference["character"],
        previous_scene_order=interval["start_scene_order_exclusive"],
        current_scene_order=interval["end_scene_order_inclusive"],
        reference_claims=compact_reference, predicted_claims=compact_prediction,
        allowed_reference_ids=[row["local_id"] for row in localized_reference],
        allowed_prediction_ids=[row["local_id"] for row in localized_prediction],
    )
    return _budgeted(system, user, config=config, counter=counter)


def _validate_coverage(
    payload: dict[str, Any], *, reference_ids: set[str], prediction_ids: set[str]
) -> list[dict[str, Any]]:
    if set(payload) != {"reference_results"} or not isinstance(payload["reference_results"], list):
        raise ValueError("Coverage judge must return reference_results only")
    output = []
    seen = set()
    for row in payload["reference_results"]:
        if not isinstance(row, dict) or set(row) != {"reference_id", "status", "prediction_ids"}:
            raise ValueError("Coverage judgment schema mismatch")
        reference_id = row["reference_id"]
        status = row["status"]
        predicted = row["prediction_ids"]
        if reference_id not in reference_ids or reference_id in seen:
            raise ValueError("Coverage judgment has an unknown or duplicate reference ID")
        if status not in {"full", "partial", "missing", "contradiction"}:
            raise ValueError("Coverage judgment status is invalid")
        if not isinstance(predicted, list) or not set(predicted) <= prediction_ids:
            raise ValueError("Coverage judgment contains an unknown prediction ID")
        if len(predicted) != len(set(predicted)):
            raise ValueError("Coverage judgment repeats a prediction ID")
        if status == "missing" and predicted:
            raise ValueError("Missing reference coverage cannot cite predictions")
        if status != "missing" and not predicted:
            raise ValueError("Non-missing reference coverage must cite predictions")
        output.append(
            {
                "reference_local_id": reference_id,
                "status": status,
                "prediction_local_ids": sorted(predicted),
            }
        )
        seen.add(reference_id)
    if seen != reference_ids:
        raise ValueError("Coverage judgment requires exact reference coverage")
    return output


def _validate_check(payload: dict[str, Any], *, expected_id: str) -> dict[str, Any]:
    fields = {
        "prediction_id",
        "supported",
        "evidence_grounded",
        "currently_valid",
        "salient",
        "development_coherent",
        "contradiction",
    }
    if set(payload) != fields or payload["prediction_id"] != expected_id:
        raise ValueError("Evidence judgment schema or prediction ID mismatch")
    for field in fields - {"prediction_id"}:
        if not isinstance(payload[field], bool):
            raise ValueError(f"Evidence judgment flag must be boolean: {field}")
    return payload


def _score(
    reference: list[dict[str, Any]], prediction: list[dict[str, Any]],
    coverage_judgments: list[dict[str, Any]], checks: list[dict[str, Any]],
) -> dict[str, Any]:
    checks_by_id = {row["prediction_id"]: row for row in checks}
    output = {}
    for name, reference_kind, prediction_kind in (
        ("current_state", "current_state", "current_state"),
        ("development", "development", "development"),
    ):
        reference_ids = [row["local_id"] for row in reference if row["claim_type"] == reference_kind]
        prediction_ids = [row["local_id"] for row in prediction if row["prediction_type"] == prediction_kind]
        relevant_coverage = [
            row
            for row in coverage_judgments
            if row["reference_local_id"] in set(reference_ids)
        ]
        coverage = _reference_centric_coverage(reference_ids, relevant_coverage)
        contradictory_ids = {
            value for value in prediction_ids if checks_by_id[value]["contradiction"]
        }
        unsupported_ids = {
            value
            for value in prediction_ids
            if not checks_by_id[value]["supported"]
            or not checks_by_id[value]["evidence_grounded"]
        }
        temporally_invalid_ids = {
            value
            for value in prediction_ids
            if not checks_by_id[value]["currently_valid"]
            or (name == "development" and not checks_by_id[value]["development_coherent"])
        }
        grounded_valid_ids = {
            value
            for value in prediction_ids
            if value not in unsupported_ids
            and value not in temporally_invalid_ids
            and value not in contradictory_ids
        }
        salient_ids = {
            value for value in prediction_ids if checks_by_id[value]["salient"]
        }
        output[name] = {
            **coverage,
            "reference_count": len(reference_ids),
            "reference_coverage": _ratio(coverage["weighted_covered_reference"], len(reference_ids)),
            "full_coverage": _ratio(coverage["fully_covered_reference"], len(reference_ids)),
            "prediction_count": len(prediction_ids),
            "grounded_valid_predictions": len(grounded_valid_ids),
            "grounded_validity_rate": _ratio(len(grounded_valid_ids), len(prediction_ids)),
            "unsupported_predictions": len(unsupported_ids),
            "unsupported_rate": _ratio(len(unsupported_ids), len(prediction_ids)),
            "contradiction_predictions": len(contradictory_ids),
            "contradiction_rate": _ratio(len(contradictory_ids), len(prediction_ids)),
            "temporally_invalid_predictions": len(temporally_invalid_ids),
            "temporally_invalid_rate": _ratio(
                len(temporally_invalid_ids), len(prediction_ids)
            ),
            "salient_predictions": len(salient_ids),
            "salience_rate": _ratio(len(salient_ids), len(prediction_ids)),
        }
    return output


def _reference_centric_coverage(
    reference_ids: list[str], judgments: list[dict[str, Any]]
) -> dict[str, Any]:
    by_reference = {row["reference_local_id"]: row for row in judgments}
    claims = []
    for reference_id in reference_ids:
        judgment = by_reference[reference_id]
        status = judgment["status"]
        weight = 1.0 if status == "full" else 0.5 if status == "partial" else 0.0
        claims.append(
            {
                "reference_local_id": reference_id,
                "status": status,
                "weight": weight,
                "prediction_local_ids": judgment["prediction_local_ids"],
            }
        )
    return {
        "weighted_covered_reference": sum(row["weight"] for row in claims),
        "fully_covered_reference": sum(row["status"] == "full" for row in claims),
        "partially_covered_reference": sum(row["status"] == "partial" for row in claims),
        "contradicted_reference": sum(row["status"] == "contradiction" for row in claims),
        "missing_reference": sum(row["status"] == "missing" for row in claims),
        "reference_claim_results": claims,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trajectory[row["trajectory_id"]].append(row)
    trajectories = []
    for trajectory_id, members in sorted(by_trajectory.items()):
        metrics = {}
        for pool in ("current_state", "development"):
            counts = {
                key: sum(row["scoring"][pool][key] for row in members)
                for key in (
                    "weighted_covered_reference",
                    "fully_covered_reference",
                    "partially_covered_reference",
                    "contradicted_reference",
                    "missing_reference",
                    "reference_count",
                    "prediction_count",
                    "grounded_valid_predictions",
                    "unsupported_predictions",
                    "contradiction_predictions",
                    "temporally_invalid_predictions",
                    "salient_predictions",
                )
            }
            metrics[pool] = {
                **counts,
                "reference_coverage": _ratio(
                    counts["weighted_covered_reference"], counts["reference_count"]
                ),
                "full_coverage": _ratio(
                    counts["fully_covered_reference"], counts["reference_count"]
                ),
                "grounded_validity_rate": _ratio(
                    counts["grounded_valid_predictions"], counts["prediction_count"]
                ),
                "unsupported_rate": _ratio(
                    counts["unsupported_predictions"], counts["prediction_count"]
                ),
                "contradiction_rate": _ratio(
                    counts["contradiction_predictions"], counts["prediction_count"]
                ),
                "temporally_invalid_rate": _ratio(
                    counts["temporally_invalid_predictions"], counts["prediction_count"]
                ),
                "salience_rate": _ratio(
                    counts["salient_predictions"], counts["prediction_count"]
                ),
            }
        trajectories.append({"trajectory_id": trajectory_id, "character": members[0]["character"], **metrics})
    movie_macro = {
        pool: {
            field: mean_defined(row[pool][field] for row in trajectories)
            for field in (
                "reference_coverage",
                "full_coverage",
                "grounded_validity_rate",
                "unsupported_rate",
                "contradiction_rate",
                "temporally_invalid_rate",
                "salience_rate",
            )
        }
        for pool in ("current_state", "development")
    }
    return {"aggregation_order": "checkpoint counts within trajectory, then character macro", "trajectory_count": len(trajectories), "movie_macro": movie_macro, "trajectories": trajectories}


def _semantic_usage(setting_results: dict[str, dict[str, Any]]) -> dict[str, int]:
    metadata = []
    for payload in setting_results.values():
        for checkpoint in payload["checkpoints"]:
            metadata.append(checkpoint["coverage_generator_metadata"])
            metadata.extend(
                row["generator_metadata"] for row in checkpoint["prediction_checks"]
            )
    semantic_calls = sum(int(row.get("received_semantic_calls", 1)) for row in metadata)
    rejected = sum(len(row.get("rejected_semantic_attempts", [])) for row in metadata)
    fallback_judgments = sum(row.get("judge_route") == "qwen_fallback" for row in metadata)
    fallback_calls = sum(
        int(row.get("fallback_received_semantic_calls", 0)) for row in metadata
    )
    primary_rejections = sum(
        int(row.get("primary_rejected_semantic_calls", 0)) for row in metadata
    )
    return {
        "successful_judgments": len(metadata),
        "received_semantic_calls": semantic_calls,
        "rejected_semantic_calls_retried": rejected,
        "deepseek_primary_successful_judgments": len(metadata) - fallback_judgments,
        "deepseek_rejected_semantic_calls": primary_rejections,
        "qwen_fallback_successful_judgments": fallback_judgments,
        "qwen_fallback_received_semantic_calls": fallback_calls,
    }


def _budgeted(system: str, user: str, *, config: BenchmarkRuntimeConfig, counter: Any) -> dict[str, Any]:
    accounted = counter.count(system) + counter.count(user) + config.reserved_chat_template_tokens
    maximum = config.call_budgets["task1_judge"].max_input_tokens
    if accounted > maximum:
        raise ValueError(f"State update evaluation prompt exceeds budget: {accounted}>{maximum}")
    return {"system_prompt": system, "user_prompt": user, "prompt_sha256": sha256_json({"system": system, "user": user}), "accounted_input_tokens": accounted}


def _maximum(current: dict[str, Any], rendered: dict[str, Any], setting: str, checkpoint_id: str, stage: str) -> dict[str, Any]:
    if rendered["accounted_input_tokens"] <= current["tokens"]:
        return current
    return {"tokens": rendered["accounted_input_tokens"], "setting": setting, "checkpoint_id": checkpoint_id, "stage": stage}


def _ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _assert_no_reasoning(metadata: dict[str, Any]) -> None:
    for key in ("reasoning_tokens", "provider_thought_tokens", "thinking_chars"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and value > 0:
            raise ValueError(f"Reasoning was not disabled: {key}={value}")


def _prompt_key(language: str, name: str) -> str:
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported prompt language: {language}")
    return f"{language}/evaluation/{name}"


def _prompt_artifacts(language: str) -> list[dict[str, Any]]:
    paths = set(
        PROMPTS.get(_prompt_key(language, COVERAGE_PROMPT_NAME)).source_paths
    ) | set(
        PROMPTS.get(_prompt_key(language, EVIDENCE_PROMPT_NAME)).source_paths
    )
    return [_artifact(path) for path in sorted(paths)]


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path.resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
