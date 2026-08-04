from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..clients import JsonCall, ModelResponseParseError, build_endpoint_pool_runtime, build_json_client
from ..io import atomic_write_json, load_json, load_scenes, sha256_file, sha256_json
from ..prompt_loader import PROMPTS
from ..temporal.benchmark_protocol import BenchmarkRuntimeConfig
from .materialization import (
    materialize_task1_claim_judge,
    materialize_task1_sequence_judge,
    materialize_task3_pair_judge,
    materialize_task3_response_judge,
)
from .schemas import (
    validate_task1_judgment,
    validate_task1_sequence_judgment,
    validate_task3_pair_judgment,
    validate_task3_response_judgment,
)
from .task1_checkpoint_metrics import (
    aggregate_checkpoint_task1,
    aggregate_task1_sequences,
    score_task1_instance,
)
from .task3_metrics import aggregate_task3


async def run_evaluation(
    *,
    task_release_dir: Path,
    task1_prediction_path: Path,
    task3_prediction_path: Path,
    pair_annotation_path: Path,
    config_path: Path,
    output_dir: Path,
    workers: int,
    evaluation_mode: str,
    preflight_only: bool = False,
    resume: bool = False,
    seed_evaluation_dir: Path | None = None,
) -> Path:
    if workers <= 0 or workers > 64:
        raise ValueError("workers must be in 1..64")
    if evaluation_mode not in {"qwen_self_judge_diagnostic", "formal_independent_evaluation"}:
        raise ValueError("Unknown evaluation mode")
    task_release_dir = task_release_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_root = output_dir / "partial"
    for name in ("task1", "task1_sequences", "task3", "task3_pairs", "failures"):
        (partial_root / name).mkdir(parents=True, exist_ok=True)

    release_manifest_path = task_release_dir / "standard24k_benchmark_release_manifest.json"
    release_manifest = _read(release_manifest_path)
    artifacts = release_manifest["artifacts"]
    protocol_manifest_path = _artifact_path(artifacts["protocol_manifest"])
    protocol_dir = protocol_manifest_path.parent
    gold_manifest_path = _artifact_path(artifacts["gold_manifest"])
    gold_dir = gold_manifest_path.parent
    config = BenchmarkRuntimeConfig.load(config_path)
    counter = config.build_token_counter()

    task1_gold = _read(gold_dir / "task1_gold_rubrics.json")
    task3_gold = _read(gold_dir / "task3_gold_rubrics.json")
    task1_predictions = _read(task1_prediction_path.resolve())
    task3_predictions = _read(task3_prediction_path.resolve())
    task1_plan = _read(protocol_dir / "task1" / "task1_rolling_plans.json")
    context_packs = _read(protocol_dir / "task3" / "task3_actor_context_packs.json")
    anchored_task3 = _read(protocol_dir / "task3" / "task3_checkpoint_single_turn.anchored.json")
    protocol_manifest = _read(protocol_manifest_path)
    evidence_bank = _read(
        Path(protocol_manifest["temporal_run_dir"])
        / "assets"
        / "source"
        / "evidence_units.json"
    )
    pair_annotations = _read(pair_annotation_path.resolve())
    source_pairs = _read(task_release_dir / "task_3_pair_groups.json")
    seed_root = seed_evaluation_dir.resolve() / "partial" if seed_evaluation_dir else None

    movie_ids = {
        release_manifest["movie_id"],
        task1_gold["movie_id"],
        task3_gold["movie_id"],
        task1_predictions["movie_id"],
        task3_predictions["movie_id"],
        pair_annotations["movie_id"],
    }
    if len(movie_ids) != 1:
        raise ValueError(f"Evaluation inputs belong to different movies: {sorted(movie_ids)}")

    task1_gold_by_id = {row["instance_id"]: row for row in task1_gold["rubrics"]}
    task3_gold_by_id = {row["instance_id"]: row for row in task3_gold["rubrics"]}
    task1_prediction_by_id = {
        row["instance_id"]: {**row, "character_id": character["character_id"]}
        for character in task1_predictions["characters"]
        for row in character["checkpoint_predictions"]
    }
    task3_prediction_by_id = {
        row["instance_id"]: row for row in task3_predictions["predictions"]
    }
    context_by_id = {row["instance_id"]: row for row in context_packs["context_packs"]}
    instance_by_id = {row["instance_id"]: row for row in anchored_task3["instances"]}
    _require_exact_ids(task1_gold_by_id, task1_prediction_by_id, "Task 1 predictions")
    _require_exact_ids(task3_gold_by_id, task3_prediction_by_id, "Task 3 predictions")
    _require_exact_ids(task3_gold_by_id, context_by_id, "Task 3 context packs")
    _require_exact_ids(task3_gold_by_id, instance_by_id, "Task 3 instances")
    annotation_by_id = _validate_pair_annotations(
        pair_annotations, source_pairs=source_pairs, instances=instance_by_id
    )

    task1_prediction_manifest = _read(task1_prediction_path.resolve().parent / "manifest.json")
    task3_prediction_manifest = _read(task3_prediction_path.resolve().parent / "manifest.json")
    actor_models = {
        str(task1_prediction_manifest["model"]), str(task3_prediction_manifest["model"])
    }
    evaluation_llm = config.evaluation_llm
    judge_model = str(evaluation_llm["model"])
    independent = all(model != judge_model for model in actor_models)
    if evaluation_mode == "formal_independent_evaluation" and not independent:
        raise ValueError("Formal independent mode requires judge model distinct from all actor models")
    judge_identity = evaluation_llm.get("judge_identity")
    if evaluation_mode == "formal_independent_evaluation":
        _validate_independent_judge_identity(
            judge_identity, judge_model=judge_model, actor_models=actor_models
        )
    if evaluation_mode == "qwen_self_judge_diagnostic" and independent:
        raise ValueError("Self-judge diagnostic mode requires the same actor and judge model")

    scenes = {scene.order: scene for scene in load_scenes(Path(task1_plan["script_path"]))}
    aliases_by_character = {
        row["character_id"]: [row["focal_character"], *row.get("aliases", [])]
        for row in task1_plan["plans"]
    }
    language = str(context_packs["language"])
    task1_materialized = {
        instance_id: materialize_task1_claim_judge(
            gold=gold,
            prediction=task1_prediction_by_id[instance_id],
            scenes=scenes,
            evidence_bank=evidence_bank,
            aliases=aliases_by_character[gold["character_id"]],
            language=language,
            config=config,
            counter=counter,
        )
        for instance_id, gold in task1_gold_by_id.items()
    }
    task3_materialized = {
        instance_id: materialize_task3_response_judge(
            gold=gold,
            prediction=task3_prediction_by_id[instance_id],
            context_pack=context_by_id[instance_id],
            instance=instance_by_id[instance_id],
            language=language,
            config=config,
            counter=counter,
        )
        for instance_id, gold in task3_gold_by_id.items()
    }
    pair_materialized = {
        pair_id: materialize_task3_pair_judge(
            annotation=annotation,
            predictions=task3_prediction_by_id,
            gold=task3_gold_by_id,
            instances=instance_by_id,
            language=language,
            config=config,
            counter=counter,
        )
        for pair_id, annotation in annotation_by_id.items()
    }
    sequence_specs = _sequence_specs(
        task1_gold_by_id=task1_gold_by_id,
        task1_predictions=task1_prediction_by_id,
        aliases_by_character=aliases_by_character,
        scenes=scenes,
        evidence_bank=evidence_bank,
        language=language,
        config=config,
        counter=counter,
    )

    run_contract = {
        "schema_version": "stage_evaluation_v1_run_contract",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_mode": evaluation_mode,
        "movie_id": next(iter(movie_ids)),
        "task_release": {"path": str(task_release_dir), "manifest_sha256": sha256_file(release_manifest_path)},
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for path in (
                task1_prediction_path,
                task3_prediction_path,
                gold_manifest_path,
                protocol_manifest_path,
                pair_annotation_path,
            )
        ],
        "config": {"path": str(config.source_path), "sha256": config.source_sha256},
        "model_contract": {
            "actor_models": sorted(actor_models),
            "judge_model": judge_model,
            "independent": independent,
            "judge_identity": judge_identity,
            "temperature": evaluation_llm["temperature"],
            "semantic_samples_per_call": 1,
        },
        "prompt_artifacts": _prompt_artifacts(config, language),
        "task1_sequence_evidence_policy": (
            "interval-only later predicted-development citations plus later "
            "gold-development supporting scenes; no content truncation"
        ),
        "seed_evaluation": (
            {
                "path": str(seed_evaluation_dir.resolve()),
                "manifest_sha256": sha256_file(seed_evaluation_dir.resolve() / "manifest.json"),
                "reuse_policy": "reuse only when identifier and prompt_sha256 exactly match",
            }
            if seed_evaluation_dir
            else None
        ),
        "preflight": {
            "task1_claim_judges": len(task1_materialized),
            "task1_sequence_judges": len(sequence_specs),
            "task3_response_judges": len(task3_materialized),
            "task3_pair_judges": len(pair_materialized),
            "max_input_tokens": {
                "task1_claim": max(row["prompt_tokens"] for row in task1_materialized.values()),
                "task1_sequence": max(row["materialized"]["prompt_tokens"] for row in sequence_specs.values()),
                "task3_response": max(row["prompt_tokens"] for row in task3_materialized.values()),
                "task3_pair": max(row["prompt_tokens"] for row in pair_materialized.values()),
            },
            "truncated_count": 0,
        },
    }
    contract_path = output_dir / "run_contract.json"
    if contract_path.exists() and resume:
        existing = _read(contract_path)
        comparable_existing = {key: value for key, value in existing.items() if key != "created_at"}
        comparable_new = {key: value for key, value in run_contract.items() if key != "created_at"}
        if comparable_existing != comparable_new:
            raise ValueError("Evaluation resume run contract drift")
    else:
        atomic_write_json(contract_path, run_contract)
    if preflight_only:
        preflight_path = output_dir / "preflight.json"
        atomic_write_json(
            preflight_path,
            {
                "schema_version": "stage_evaluation_v1_preflight",
                "status": "passed_zero_call",
                "run_contract_sha256": sha256_file(contract_path),
                **run_contract["preflight"],
            },
        )
        return preflight_path

    endpoint_runtime = build_endpoint_pool_runtime(evaluation_llm)
    client = build_json_client(
        {
            **evaluation_llm,
            "json_response_format": True,
            "max_tokens": max(
                config.call_budgets["task1_judge"].max_output_tokens,
                config.call_budgets["task3_response_judge"].max_output_tokens,
                config.call_budgets["task3_pair_judge"].max_output_tokens,
            ),
        },
        endpoint_runtime=endpoint_runtime,
    )
    semaphore = asyncio.Semaphore(workers)

    task1_results = await _run_jobs(
        task1_materialized,
        directory=partial_root / "task1",
        resume=resume,
        failure_directory=partial_root / "failures",
        runner=lambda instance_id, materialized: _call_task1(
            instance_id,
            materialized=materialized,
            gold=task1_gold_by_id[instance_id],
            client=client,
            semaphore=semaphore,
        ),
        seed_directory=seed_root / "task1" if seed_root else None,
    )
    sequence_results = await _run_jobs(
        {key: row["materialized"] for key, row in sequence_specs.items()},
        directory=partial_root / "task1_sequences",
        resume=resume,
        failure_directory=partial_root / "failures",
        runner=lambda sequence_id, materialized: _call_sequence(
            sequence_id,
            materialized=materialized,
            spec=sequence_specs[sequence_id],
            client=client,
            semaphore=semaphore,
        ),
        seed_directory=seed_root / "task1_sequences" if seed_root else None,
    )
    task3_results = await _run_jobs(
        task3_materialized,
        directory=partial_root / "task3",
        resume=resume,
        failure_directory=partial_root / "failures",
        runner=lambda instance_id, materialized: _call_task3(
            instance_id,
            materialized=materialized,
            gold=task3_gold_by_id[instance_id],
            client=client,
            semaphore=semaphore,
        ),
        seed_directory=seed_root / "task3" if seed_root else None,
    )
    pair_results = await _run_jobs(
        pair_materialized,
        directory=partial_root / "task3_pairs",
        resume=resume,
        failure_directory=partial_root / "failures",
        runner=lambda pair_id, materialized: _call_pair(
            pair_id,
            materialized=materialized,
            annotation=annotation_by_id[pair_id],
            client=client,
            recovery_client=None,
            semaphore=semaphore,
        ),
        seed_directory=seed_root / "task3_pairs" if seed_root else None,
    )
    task3_result_by_id = {row["instance_id"]: row for row in task3_results}
    for pair in pair_results:
        prerequisites = []
        for instance_id in pair["instance_ids"]:
            response_judgment = task3_result_by_id[instance_id]["judgment"]
            prerequisites.append(
                {
                    "instance_id": instance_id,
                    "stance_compatible": response_judgment["stance_compatible"],
                    "future_leakage": response_judgment["future_leakage"],
                    "unknown_fact_hallucination": response_judgment[
                        "unknown_fact_hallucination"
                    ],
                    "passed": bool(
                        response_judgment["stance_compatible"]
                        and not response_judgment["future_leakage"]
                        and not response_judgment["unknown_fact_hallucination"]
                    ),
                }
            )
        pair["response_prerequisites"] = prerequisites
        atomic_write_json(
            partial_root / "task3_pairs" / f"{pair['pair_group_id']}.json", pair
        )

    task1_aggregate = aggregate_checkpoint_task1(task1_results)
    task1_aggregate["longitudinal_consistency"] = aggregate_task1_sequences(sequence_results)
    task1_aggregate["delayed_update"] = {
        "eligible_developments": 0,
        "value": None,
        "reason": "missing_explicit_gold_development_lineage",
    }
    task3_aggregate = aggregate_task3(task3_results, pair_results)
    task1_output_path = output_dir / "task1_evaluation.json"
    task3_output_path = output_dir / "task3_evaluation.json"
    atomic_write_json(
        task1_output_path,
        {
            "schema_version": "stage_task1_evaluation_v1",
            "movie_id": next(iter(movie_ids)),
            "instance_count": len(task1_results),
            "sequence_count": len(sequence_results),
            "aggregate": task1_aggregate,
            "instances": task1_results,
            "sequences": sequence_results,
        },
    )
    atomic_write_json(
        task3_output_path,
        {
            "schema_version": "stage_task3_evaluation_v1",
            "movie_id": next(iter(movie_ids)),
            "instance_count": len(task3_results),
            "pair_count": len(pair_results),
            "aggregate": task3_aggregate,
            "instances": task3_results,
            "pairs": pair_results,
        },
    )
    endpoint_snapshot = await endpoint_runtime.snapshot() if endpoint_runtime else None
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task1_task3_evaluation_run_v1",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "completed",
            "evaluation_mode": evaluation_mode,
            "movie_id": next(iter(movie_ids)),
            "python_executable": sys.executable,
            "model_contract": run_contract["model_contract"],
            "run_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            "endpoint_pool": endpoint_snapshot,
            "counts": {
                "expected_task1": len(task1_gold_by_id),
                "evaluated_task1": len(task1_results),
                "expected_task1_sequences": len(sequence_specs),
                "evaluated_task1_sequences": len(sequence_results),
                "expected_task3": len(task3_gold_by_id),
                "evaluated_task3": len(task3_results),
                "expected_task3_pairs": len(annotation_by_id),
                "evaluated_task3_pairs": len(pair_results),
                "failure_count": 0,
                "truncated_count": 0,
                "seed_reused_task1": sum("seed_reused_from" in row for row in task1_results),
                "seed_reused_task1_sequences": sum("seed_reused_from" in row for row in sequence_results),
                "seed_reused_task3": sum("seed_reused_from" in row for row in task3_results),
                "seed_reused_task3_pairs": sum("seed_reused_from" in row for row in pair_results),
            },
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in (task1_output_path, task3_output_path)
            ],
        },
    )
    return manifest_path


async def _call_task1(
    instance_id: str,
    *,
    materialized: dict[str, Any],
    gold: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        response = await client.generate_json(
            system_prompt=materialized["system_prompt"],
            user_prompt=materialized["user_prompt"],
            stage=f"evaluation_v1_task1_claim:{instance_id}",
        )
    localized = materialized["localized_prediction"]
    gold_ids = {
        row["local_id"]
        for field in ("current_state_claims", "development_claims", "invariant_claims")
        for row in gold["rubric"][field]
    }
    prediction_ids = {row["local_id"] for row in localized}
    try:
        judgment = validate_task1_judgment(
            response.data, gold_ids=gold_ids, prediction_ids=prediction_ids
        )
    except ValueError as exc:
        _attach_response_context(exc, response)
        raise
    scoring = score_task1_instance(
        prediction=localized, rubric=gold["rubric"], judgment=judgment
    )
    return {
        "instance_id": instance_id,
        "character_id": gold["character_id"],
        "character": gold["character"],
        "checkpoint": gold["checkpoint"],
        "localized_prediction": localized,
        "judgment": judgment,
        "scoring": scoring,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "generator_metadata": response.metadata,
    }


async def _call_sequence(
    sequence_id: str,
    *,
    materialized: dict[str, Any],
    spec: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        response = await client.generate_json(
            system_prompt=materialized["system_prompt"],
            user_prompt=materialized["user_prompt"],
            stage=f"evaluation_v1_task1_sequence:{sequence_id}",
        )
    try:
        judgment = validate_task1_sequence_judgment(response.data)
    except ValueError as exc:
        _attach_response_context(exc, response)
        raise
    consistent = bool(
        judgment["state_carry_forward"]
        and judgment["development_to_state_coherent"]
        and not judgment["contradiction_present"]
        and not judgment["premature_or_future_information"]
    )
    return {
        "sequence_id": sequence_id,
        "character_id": spec["character_id"],
        "character": spec["character"],
        "instance_ids": spec["instance_ids"],
        "judgment": judgment,
        "consistent": consistent,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "generator_metadata": response.metadata,
    }


async def _call_task3(
    instance_id: str,
    *,
    materialized: dict[str, Any],
    gold: dict[str, Any],
    client: Any,
    recovery_client: Any | None = None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        response, effective = await _generate_with_content_filter_fallback(
            client=client,
            recovery_client=recovery_client,
            materialized=materialized,
            stage=f"task3_response_judge:{instance_id}",
        )
    raw_judgment = deepcopy(response.data)
    candidate, normalizations = _normalize_task3_response(
        response.data, allowed_evidence_ids=materialized["allowed_labels"]
    )
    try:
        judgment = validate_task3_response_judgment(
            candidate, allowed_evidence_ids=materialized["allowed_labels"]
        )
    except ValueError as exc:
        _attach_response_context(exc, response)
        raise
    result = {
        "instance_id": instance_id,
        "character_id": gold["character_id"],
        "character": gold["character"],
        "checkpoint": materialized["checkpoint"],
        "response": materialized["actor_response"],
        "judgment": judgment,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "effective_prompt_tokens": effective["prompt_tokens"],
        "effective_prompt_sha256": effective["prompt_sha256"],
        "evaluation_input_mode": effective["context_mode"],
        "generator_metadata": response.metadata,
    }
    return _with_normalization_trace(result, raw_judgment, normalizations)


async def _generate_with_content_filter_fallback(
    *, client: Any, recovery_client: Any | None, materialized: dict[str, Any], stage: str
) -> tuple[JsonCall, dict[str, Any]]:
    try:
        response = await client.generate_json(
            system_prompt=materialized["system_prompt"],
            user_prompt=materialized["user_prompt"],
            stage=stage,
        )
        return response, materialized
    except ModelResponseParseError as exc:
        if recovery_client is not None and exc.metadata.get("finish_reason") == "max_tokens":
            recovered = await recovery_client.generate_json(
                system_prompt=materialized["system_prompt"],
                user_prompt=materialized["user_prompt"],
                stage=f"{stage}:output_limit_recovery",
            )
            return (
                JsonCall(
                    data=recovered.data,
                    metadata={
                        **recovered.metadata,
                        "output_limit_recovery": {
                            "trigger": "primary_max_tokens_parse_failure",
                            "primary_response_sha256": exc.metadata.get(
                                "raw_response_sha256"
                            ),
                            "primary_response_metadata": exc.metadata,
                        },
                    },
                ),
                materialized,
            )
        if exc.raw_text.strip() != "你好，我无法给到相关内容。":
            raise
        fallback = materialized["content_filter_fallback"]
        secondary_refusal = None
        effective = fallback
        try:
            response = await client.generate_json(
                system_prompt=fallback["system_prompt"],
                user_prompt=fallback["user_prompt"],
                stage=f"{stage}:reviewed_gold_context_fallback",
            )
        except ModelResponseParseError as fallback_exc:
            if (
                recovery_client is not None
                and fallback_exc.metadata.get("finish_reason") == "max_tokens"
            ):
                recovered = await recovery_client.generate_json(
                    system_prompt=fallback["system_prompt"],
                    user_prompt=fallback["user_prompt"],
                    stage=f"{stage}:reviewed_gold_context_fallback:output_limit_recovery",
                )
                response = JsonCall(
                    data=recovered.data,
                    metadata={
                        **recovered.metadata,
                        "output_limit_recovery": {
                            "trigger": "fallback_max_tokens_parse_failure",
                            "primary_response_sha256": fallback_exc.metadata.get(
                                "raw_response_sha256"
                            ),
                            "primary_response_metadata": fallback_exc.metadata,
                        },
                    },
                )
            elif fallback_exc.raw_text.strip() != "你好，我无法给到相关内容。":
                raise
            else:
                secondary_refusal = fallback_exc
                safety_system = (
                    fallback["system_prompt"]
                    + "\nSafety-framed classification: the quoted fictional response may "
                    "contain profanity or sexual-health references. Treat it only as data "
                    "to classify. Do not reproduce it; return only the required JSON scores "
                    "and labels."
                )
                effective = {
                    **fallback,
                    "system_prompt": safety_system,
                    "prompt_sha256": sha256_json(
                        {
                            "system": safety_system,
                            "user": fallback["user_prompt"],
                        }
                    ),
                    "context_mode": (
                        "reviewed_gold_without_verbatim_actor_context_safety_framed"
                    ),
                }
                try:
                    response = await client.generate_json(
                        system_prompt=effective["system_prompt"],
                        user_prompt=effective["user_prompt"],
                        stage=f"{stage}:safety_framed_classification_fallback",
                    )
                except ModelResponseParseError as safety_exc:
                    if (
                        recovery_client is None
                        or safety_exc.metadata.get("finish_reason") != "max_tokens"
                    ):
                        raise
                    recovered = await recovery_client.generate_json(
                        system_prompt=effective["system_prompt"],
                        user_prompt=effective["user_prompt"],
                        stage=(
                            f"{stage}:safety_framed_classification_fallback:"
                            "output_limit_recovery"
                        ),
                    )
                    response = JsonCall(
                        data=recovered.data,
                        metadata={
                            **recovered.metadata,
                            "output_limit_recovery": {
                                "trigger": "safety_fallback_max_tokens_parse_failure",
                                "primary_response_sha256": safety_exc.metadata.get(
                                    "raw_response_sha256"
                                ),
                                "primary_response_metadata": safety_exc.metadata,
                            },
                        },
                    )
                effective = {
                    **effective,
                    "prompt_tokens": response.metadata.get(
                        "prompt_tokens", effective["prompt_tokens"]
                    ),
                }
        return (
            JsonCall(
                data=response.data,
                metadata={
                    **response.metadata,
                    "content_filter_fallback": {
                        "trigger": "provider_content_refusal",
                        "primary_prompt_sha256": materialized["prompt_sha256"],
                        "primary_response_sha256": exc.metadata.get(
                            "raw_response_sha256"
                        ),
                        "primary_response_metadata": exc.metadata,
                        "fallback_prompt_sha256": effective["prompt_sha256"],
                        "fallback_context_mode": effective["context_mode"],
                        "secondary_refusal_metadata": (
                            secondary_refusal.metadata
                            if secondary_refusal is not None
                            else None
                        ),
                    },
                },
            ),
            effective,
        )


async def _call_pair(
    pair_id: str,
    *,
    materialized: dict[str, Any],
    annotation: dict[str, Any],
    client: Any,
    recovery_client: Any | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        response = await _generate_pair_with_content_filter_fallback(
            client=client,
            recovery_client=recovery_client,
            materialized=materialized,
            stage=f"task3_pair_judge:{pair_id}",
        )
    raw_judgment = deepcopy(response.data)
    candidate, normalizations = _normalize_task3_pair(
        response.data,
        expected_pair_type=annotation["pair_type"],
        responses_by_label=materialized["responses_by_label"],
        allowed_evidence_labels={row["local_label"] for row in materialized["pair_evidence"]},
    )
    try:
        judgment = validate_task3_pair_judgment(
            candidate, expected_pair_type=annotation["pair_type"]
        )
    except ValueError as exc:
        _attach_response_context(exc, response)
        raise
    allowed_labels = {row["local_label"] for row in materialized["pair_evidence"]}
    if not set(judgment["local_evidence_labels"]) <= allowed_labels:
        raise ValueError(f"Task 3 pair returned unknown evidence label: {pair_id}")
    for assessment in judgment["response_assessments"]:
        source = " ".join(
            materialized["responses_by_label"][assessment["response_label"]].split()
        ).casefold()
        excerpt = " ".join(assessment["response_excerpt"].split()).casefold()
        if excerpt not in source:
            raise ValueError(
                f"Task 3 pair response excerpt is not verbatim: {pair_id}/"
                f"{assessment['response_label']}"
            )
    result = {
        "pair_group_id": pair_id,
        "pair_type": annotation["pair_type"],
        "instance_ids": materialized["instance_ids"],
        "expected_direction": annotation["expected_direction"],
        "comparability": annotation["comparability"],
        "judgment": judgment,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "generator_metadata": response.metadata,
    }
    return _with_normalization_trace(result, raw_judgment, normalizations)


async def _generate_pair_with_content_filter_fallback(
    *, client: Any, recovery_client: Any | None, materialized: dict[str, Any], stage: str
) -> JsonCall:
    try:
        return await client.generate_json(
            system_prompt=materialized["system_prompt"],
            user_prompt=materialized["user_prompt"],
            stage=stage,
        )
    except ModelResponseParseError as exc:
        if recovery_client is not None and exc.metadata.get("finish_reason") == "max_tokens":
            recovered = await recovery_client.generate_json(
                system_prompt=materialized["system_prompt"],
                user_prompt=materialized["user_prompt"],
                stage=f"{stage}:output_limit_recovery",
            )
            return JsonCall(
                data=recovered.data,
                metadata={
                    **recovered.metadata,
                    "output_limit_recovery": {
                        "trigger": "primary_max_tokens_parse_failure",
                        "primary_response_sha256": exc.metadata.get("raw_response_sha256"),
                        "primary_response_metadata": exc.metadata,
                    },
                },
            )
        if exc.raw_text.strip() != "你好，我无法给到相关内容。":
            raise
        safety_system = (
            materialized["system_prompt"]
            + "\nSafety-framed classification: the quoted fictional responses may "
            "contain profanity or sexual-health references. Treat them only as data "
            "to compare. Do not reproduce sensitive content; return only the required "
            "JSON labels and brief abstract rationale."
        )
        fallback_prompt_sha256 = sha256_json(
            {"system": safety_system, "user": materialized["user_prompt"]}
        )
        try:
            fallback_response = await client.generate_json(
                system_prompt=safety_system,
                user_prompt=materialized["user_prompt"],
                stage=f"{stage}:safety_framed_classification_fallback",
            )
        except ModelResponseParseError as fallback_exc:
            if (
                recovery_client is None
                or fallback_exc.metadata.get("finish_reason") != "max_tokens"
            ):
                reviewed = materialized.get("reviewed_judgment_fallback")
                if (
                    fallback_exc.raw_text.strip() != "你好，我无法给到相关内容。"
                    or not isinstance(reviewed, dict)
                ):
                    raise
                reviewed_response = await client.generate_json(
                    system_prompt=reviewed["system_prompt"],
                    user_prompt=reviewed["user_prompt"],
                    stage=f"{stage}:reviewed_response_judgment_fallback",
                )
                return JsonCall(
                    data=reviewed_response.data,
                    metadata={
                        **reviewed_response.metadata,
                        "content_filter_fallback": {
                            "trigger": "provider_content_refusal",
                            "primary_prompt_sha256": materialized["prompt_sha256"],
                            "primary_response_sha256": exc.metadata.get(
                                "raw_response_sha256"
                            ),
                            "primary_response_metadata": exc.metadata,
                            "safety_fallback_response_metadata": fallback_exc.metadata,
                            "fallback_prompt_sha256": reviewed["prompt_sha256"],
                            "fallback_context_mode": reviewed["context_mode"],
                        },
                    },
                )
            else:
                recovered = await recovery_client.generate_json(
                    system_prompt=safety_system,
                    user_prompt=materialized["user_prompt"],
                    stage=f"{stage}:safety_framed_classification_fallback:output_limit_recovery",
                )
                fallback_response = JsonCall(
                    data=recovered.data,
                    metadata={
                        **recovered.metadata,
                        "output_limit_recovery": {
                            "trigger": "safety_fallback_max_tokens_parse_failure",
                            "primary_response_sha256": fallback_exc.metadata.get(
                                "raw_response_sha256"
                            ),
                            "primary_response_metadata": fallback_exc.metadata,
                        },
                    },
                )
        return JsonCall(
            data=fallback_response.data,
            metadata={
                **fallback_response.metadata,
                "content_filter_fallback": {
                    "trigger": "provider_content_refusal",
                    "primary_prompt_sha256": materialized["prompt_sha256"],
                    "primary_response_sha256": exc.metadata.get("raw_response_sha256"),
                    "primary_response_metadata": exc.metadata,
                    "fallback_prompt_sha256": fallback_prompt_sha256,
                    "fallback_context_mode": "safety_framed_pair_classification",
                },
            },
        )


def _normalize_task3_response(
    payload: dict[str, Any], *, allowed_evidence_ids: set[str]
) -> tuple[dict[str, Any], list[str]]:
    candidate = deepcopy(payload)
    if not isinstance(candidate, dict):
        return candidate, []
    normalizations: list[str] = []
    scores = candidate.get("scores")
    if isinstance(scores, dict):
        boundary = scores.get("boundary_compliance")
        future = candidate.get("future_leakage")
        unknown = candidate.get("unknown_fact_hallucination")
        if isinstance(boundary, int) and not isinstance(boundary, bool):
            if (future is True or unknown is True) and boundary >= 4:
                scores["boundary_compliance"] = 3
                normalizations.append("capped_boundary_score_for_violation_flag")
            elif future is False and unknown is False and boundary <= 2:
                candidate["future_leakage"] = True
                normalizations.append("set_future_leakage_for_low_boundary_score")
    evidence = candidate.get("evidence_local_ids")
    if isinstance(evidence, list):
        filtered = [value for value in evidence if value in allowed_evidence_ids]
        if filtered != evidence:
            candidate["evidence_local_ids"] = filtered
            normalizations.append("dropped_unknown_evidence_label")
    return candidate, sorted(set(normalizations))


def _normalize_task3_pair(
    payload: dict[str, Any], *, expected_pair_type: str, responses_by_label: dict[str, str],
    allowed_evidence_labels: set[str]
) -> tuple[dict[str, Any], list[str]]:
    candidate = deepcopy(payload)
    if not isinstance(candidate, dict):
        return candidate, []
    normalizations: list[str] = []
    if set(candidate) == {"pair_judgment"} and isinstance(
        candidate["pair_judgment"], dict
    ):
        candidate = candidate["pair_judgment"]
        normalizations.append("unwrapped_pair_judgment_object")
    if candidate.get("pair_type") != expected_pair_type:
        candidate["pair_type"] = expected_pair_type
        normalizations.append("restored_pair_type_from_input_annotation")
    evidence = candidate.get("local_evidence_labels")
    if isinstance(evidence, list):
        filtered = [value for value in evidence if value in allowed_evidence_labels]
        if filtered != evidence:
            candidate["local_evidence_labels"] = filtered
            normalizations.append("dropped_unknown_pair_evidence_label")
    assessments = candidate.get("response_assessments")
    if isinstance(assessments, list):
        for row in assessments:
            if not isinstance(row, dict) or row.get("response_label") not in responses_by_label:
                continue
            label = row["response_label"]
            source = " ".join(str(responses_by_label[label]).split()).casefold()
            excerpt = " ".join(str(row.get("response_excerpt", "")).split()).casefold()
            if excerpt and excerpt not in source:
                row["response_excerpt"] = responses_by_label[label]
                normalizations.append("replaced_nonverbatim_excerpt_with_full_response")
    return candidate, sorted(set(normalizations))


def _with_normalization_trace(
    result: dict[str, Any], raw_judgment: dict[str, Any], normalizations: list[str]
) -> dict[str, Any]:
    if not normalizations:
        return result
    return {
        **result,
        "raw_judgment": raw_judgment,
        "deterministic_normalizations": normalizations,
    }


async def _run_jobs(
    materialized_by_id: dict[str, dict[str, Any]],
    *,
    directory: Path,
    resume: bool,
    failure_directory: Path,
    runner: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    seed_directory: Path | None = None,
) -> list[dict[str, Any]]:
    async def one(identifier: str, materialized: dict[str, Any]) -> dict[str, Any]:
        path = directory / f"{identifier}.json"
        if resume and path.is_file():
            existing = _read(path)
            if existing.get("prompt_sha256") != materialized["prompt_sha256"]:
                raise ValueError(f"Partial prompt drift: {path}")
            (failure_directory / f"{identifier}.json").unlink(missing_ok=True)
            return existing
        if seed_directory is not None:
            seed_path = seed_directory / f"{identifier}.json"
            if seed_path.is_file():
                seeded = _read(seed_path)
                if seeded.get("prompt_sha256") == materialized["prompt_sha256"]:
                    seeded = {**seeded, "seed_reused_from": str(seed_path)}
                    atomic_write_json(path, seeded)
                    return seeded
        try:
            result = await runner(identifier, materialized)
        except Exception as exc:
            atomic_write_json(
                failure_directory / f"{identifier}.json",
                {
                    "identifier": identifier,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "prompt_sha256": materialized["prompt_sha256"],
                    "prompt_tokens": materialized["prompt_tokens"],
                    "parsed_response": getattr(exc, "parsed_response", None),
                    "raw_response": getattr(exc, "raw_text", None),
                    "generator_metadata": getattr(
                        exc, "generator_metadata", getattr(exc, "metadata", None)
                    ),
                },
            )
            raise
        atomic_write_json(path, result)
        (failure_directory / f"{identifier}.json").unlink(missing_ok=True)
        return result

    settled = await asyncio.gather(
        *(one(identifier, materialized) for identifier, materialized in materialized_by_id.items()),
        return_exceptions=True,
    )
    failures = [row for row in settled if isinstance(row, BaseException)]
    if failures:
        raise RuntimeError(
            f"{len(failures)} evaluation calls failed; first: {type(failures[0]).__name__}: {failures[0]}"
        )
    return [row for row in settled if isinstance(row, dict)]


def _sequence_specs(
    *,
    task1_gold_by_id: dict[str, dict[str, Any]],
    task1_predictions: dict[str, dict[str, Any]],
    aliases_by_character: dict[str, list[str]],
    scenes: dict[int, Any],
    evidence_bank: dict[str, Any],
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> dict[str, dict[str, Any]]:
    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance_id, gold in task1_gold_by_id.items():
        by_character[gold["character_id"]].append(
            {
                "instance_id": instance_id,
                "character_id": gold["character_id"],
                "character": gold["character"],
                "checkpoint": gold["checkpoint"],
                "prediction": task1_predictions[instance_id]["prediction"],
                "gold": gold,
            }
        )
    output = {}
    for character_id, rows in by_character.items():
        rows.sort(key=lambda row: row["checkpoint"]["current_scene_order"])
        for earlier, later in zip(rows, rows[1:]):
            sequence_id = f"{earlier['instance_id']}--{later['instance_id']}"
            materialized = materialize_task1_sequence_judge(
                character=later["character"],
                character_id=character_id,
                aliases=aliases_by_character[character_id],
                earlier=earlier,
                later=later,
                later_gold=later["gold"],
                scenes=scenes,
                evidence_bank=evidence_bank,
                language=language,
                config=config,
                counter=counter,
            )
            output[sequence_id] = {
                "character_id": character_id,
                "character": later["character"],
                "instance_ids": [earlier["instance_id"], later["instance_id"]],
                "materialized": materialized,
            }
    return output


def _validate_pair_annotations(
    payload: dict[str, Any],
    *,
    source_pairs: dict[str, Any],
    instances: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = payload["pairs"]
    by_id = {row["pair_group_id"]: row for row in rows}
    source_by_id = {row["pair_group_id"]: row for row in source_pairs["pair_groups"]}
    _require_exact_ids(source_by_id, by_id, "Task 3 pair annotations")
    for pair_id, annotation in by_id.items():
        ordered = annotation["ordered_instance_ids"]
        if set(ordered) != set(source_by_id[pair_id]["instance_ids"]):
            raise ValueError(f"Pair annotation instance drift: {pair_id}")
        observed_orders = [
            int(instances[instance_id]["model_input"]["checkpoint_anchor"]["scene_order"])
            for instance_id in ordered
        ]
        if observed_orders != sorted(observed_orders) or observed_orders != annotation["checkpoint_scene_orders"]:
            raise ValueError(f"Pair annotation checkpoint order drift: {pair_id}")
    if payload["pair_count"] != len(by_id):
        raise ValueError("Pair annotation count mismatch")
    return by_id


def _validate_independent_judge_identity(
    payload: Any, *, judge_model: str, actor_models: set[str]
) -> None:
    required = {
        "provider",
        "model_version",
        "access_date",
        "independence_basis",
        "weights_and_model_family_distinct_from_actor",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Formal independent mode requires a complete judge_identity contract")
    if str(payload["model_version"]).strip() != judge_model:
        raise ValueError("judge_identity model_version must equal the configured judge model")
    if not str(payload["provider"]).strip() or not str(payload["access_date"]).strip():
        raise ValueError("judge_identity provider and access_date must be nonempty")
    if not str(payload["independence_basis"]).strip():
        raise ValueError("judge_identity independence_basis must be nonempty")
    if payload["weights_and_model_family_distinct_from_actor"] is not True:
        raise ValueError("Formal judge identity must attest distinct weights and model family")
    if judge_model in actor_models:
        raise ValueError("Formal judge identity conflicts with actor model names")


def _prompt_artifacts(config: BenchmarkRuntimeConfig, language: str) -> list[dict[str, str]]:
    paths = set()
    for call_name in (
        "task1_judge",
        "task3_response_judge",
        "task3_pair_judge",
    ):
        paths.update(PROMPTS.get(config.prompt_path(call_name, language)).source_paths)
    language_key = "zh" if str(language).casefold() in {"zh", "chinese"} else "en"
    paths.update(PROMPTS.get(f"{language_key}/evaluation_v1/task1_sequence_judge").source_paths)
    return [
        {"path": str(path), "sha256": sha256_file(path)} for path in sorted(paths)
    ]


def _require_exact_ids(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    if set(left) != set(right):
        raise ValueError(
            f"{label} ID mismatch: missing={sorted(set(left) - set(right))} "
            f"extra={sorted(set(right) - set(left))}"
        )


def _read(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _attach_response_context(exc: Exception, response: Any) -> None:
    exc.parsed_response = response.data
    exc.generator_metadata = response.metadata


def _artifact_path(value: Any) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Release artifact lacks a path: {value!r}")
    return Path(value).resolve()
