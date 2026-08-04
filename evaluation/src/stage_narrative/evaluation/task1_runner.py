from __future__ import annotations

import asyncio
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..clients import JsonCall, ModelResponseParseError, build_endpoint_pool_runtime, build_json_client
from ..io import atomic_write_json, load_json, load_scenes, sha256_file
from ..prompt_loader import PROMPTS
from ..temporal.benchmark_protocol import BenchmarkRuntimeConfig
from .task1_materialization import (
    materialize_task1_adjacent_judge,
    materialize_task1_staged_checkpoint_judges,
    materialize_task1_development_cluster_judge,
)
from .task1_metrics import aggregate_task1, localize_task1_prediction, score_task1_checkpoint, score_task1_trajectory
from .task1_schemas import (
    validate_task1_adjacent_judgment,
    validate_task1_alignment_judgment,
    validate_task1_checkpoint_judgment,
    validate_task1_development_cluster_judgment,
    validate_task1_evidence_judgment,
    validate_task1_prediction,
    validate_task1_private_assets,
)


EVALUATION_MODES = {"formal_independent_evaluation", "self_judge_diagnostic"}


async def run_task1_evaluation(
    *,
    public_instances_path: Path,
    private_assets_path: Path,
    rolling_plans_path: Path,
    evidence_bank_path: Path,
    prediction_path: Path,
    config_path: Path,
    output_dir: Path,
    workers: int = 8,
    evaluation_mode: str = "formal_independent_evaluation",
    resume: bool = False,
    preflight_only: bool = False,
    client: Any | None = None,
) -> Path:
    if workers <= 0:
        raise ValueError("Task 1 evaluation workers must be positive")
    if evaluation_mode not in EVALUATION_MODES:
        raise ValueError(f"Unsupported Task 1 evaluation mode: {evaluation_mode}")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite Task 1 evaluation: {output_dir}")
    partial_root = output_dir / "partials"
    failure_root = partial_root / "failures"
    for path in (
        output_dir,
        partial_root / "checkpoints",
        partial_root / "checkpoint_alignments",
        partial_root / "checkpoint_evidence",
        partial_root / "development_clusters",
        partial_root / "adjacent_checks",
        failure_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    public = _object(public_instances_path)
    private = validate_task1_private_assets(_object(private_assets_path))
    plans = _object(rolling_plans_path)
    evidence_bank = _object(evidence_bank_path)
    predictions = _object(prediction_path)
    config = BenchmarkRuntimeConfig.load(config_path)
    counter = config.build_token_counter()
    _validate_input_contracts(
        public=public,
        public_path=public_instances_path,
        private=private,
        plans=plans,
        predictions=predictions,
    )

    prediction_manifest_path = prediction_path.resolve().parent / "manifest.json"
    prediction_manifest = _object(prediction_manifest_path)
    actor_model = str(prediction_manifest.get("model") or "").strip()
    evaluation_llm = config.evaluation_llm
    judge_model = str(evaluation_llm.get("model") or "").strip()
    if not actor_model or not judge_model:
        raise ValueError("Task 1 evaluation requires actor and judge model identities")
    independent = actor_model != judge_model
    if evaluation_mode == "formal_independent_evaluation":
        if not independent:
            raise ValueError("Formal Task 1 evaluation requires a judge distinct from the actor")
        _validate_independent_judge_identity(
            evaluation_llm.get("judge_identity"), judge_model=judge_model, actor_model=actor_model
        )
    elif independent:
        raise ValueError("Self-judge diagnostic mode requires the actor and judge model to match")

    prediction_by_id = _predictions_by_instance(predictions)
    aliases_by_character = {
        row["character_id"]: [row["focal_character"], *row.get("aliases", [])]
        for row in plans["plans"]
    }
    scenes = {scene.order: scene for scene in load_scenes(Path(plans["script_path"]))}
    language = str(private["language"])
    checkpoint_specs: dict[str, dict[str, Any]] = {}
    alignment_specs: dict[str, dict[str, Any]] = {}
    evidence_specs: dict[str, dict[str, Any]] = {}
    cluster_specs: dict[str, dict[str, Any]] = {}
    adjacent_specs: dict[str, dict[str, Any]] = {}
    trajectory_by_character = {
        row["character_id"]: row for row in private["trajectories"]
    }
    for trajectory in private["trajectories"]:
        character_id = trajectory["character_id"]
        aliases = aliases_by_character[character_id]
        for rubric in trajectory["checkpoint_rubrics"]:
            instance_id = rubric["instance_id"]
            staged = materialize_task1_staged_checkpoint_judges(
                rubric=rubric,
                prediction=prediction_by_id[instance_id],
                scenes=scenes,
                evidence_bank=evidence_bank,
                aliases=aliases,
                character_id=character_id,
                language=language,
                config=config,
                counter=counter,
            )
            checkpoint_specs[instance_id] = {
                "movie_id": private["movie_id"],
                "trajectory": trajectory,
                "rubric": rubric,
                "prediction": prediction_by_id[instance_id],
                "localized_prediction": staged["localized_prediction"],
            }
            alignment_specs[instance_id] = {
                **checkpoint_specs[instance_id],
                "materialized": staged["alignment"],
            }
            for index, materialized in enumerate(staged["evidence_batches"], start=1):
                batch_id = f"{instance_id}--E{index}"
                evidence_specs[batch_id] = {
                    **checkpoint_specs[instance_id],
                    "instance_id": instance_id,
                    "batch_index": index,
                    "materialized": materialized,
                }
        cluster_specs[character_id] = {
            "trajectory": trajectory,
            "materialized": materialize_task1_development_cluster_judge(
                trajectory=trajectory,
                predictions_by_instance=prediction_by_id,
                language=language,
                config=config,
                counter=counter,
            ),
        }
        for earlier, later in zip(
            trajectory["checkpoint_rubrics"], trajectory["checkpoint_rubrics"][1:]
        ):
            pair_id = f"{earlier['instance_id']}--{later['instance_id']}"
            adjacent_specs[pair_id] = {
                "trajectory": trajectory,
                "earlier": earlier,
                "later": later,
                "materialized": materialize_task1_adjacent_judge(
                    trajectory=trajectory,
                    earlier_rubric=earlier,
                    later_rubric=later,
                    predictions_by_instance=prediction_by_id,
                    scenes=scenes,
                    evidence_bank=evidence_bank,
                    aliases=aliases,
                    language=language,
                    config=config,
                    counter=counter,
                ),
            }

    contract = {
        "schema_version": "stage_task1_evaluation_run_contract",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_mode": evaluation_mode,
        "movie_id": private["movie_id"],
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for path in (
                public_instances_path,
                private_assets_path,
                rolling_plans_path,
                evidence_bank_path,
                prediction_path,
                prediction_manifest_path,
            )
        ],
        "config": {"path": str(config.source_path), "sha256": config.source_sha256},
        "model_contract": {
            "actor_model": actor_model,
            "judge_model": judge_model,
            "independent": independent,
            "judge_identity": evaluation_llm.get("judge_identity"),
            "temperature": evaluation_llm.get("temperature"),
            "semantic_samples_per_call": 1,
        },
        "prompt_artifacts": _prompt_artifacts(language),
        "preflight": {
            "checkpoint_instances": len(checkpoint_specs),
            "checkpoint_alignment_judges": len(alignment_specs),
            "checkpoint_evidence_judges": len(evidence_specs),
            "development_cluster_judges": len(cluster_specs),
            "adjacent_judges": len(adjacent_specs),
            "max_input_tokens": {
                "checkpoint_alignment": _maximum_prompt_tokens(alignment_specs),
                "checkpoint_evidence": _maximum_prompt_tokens(evidence_specs),
                "development_cluster": _maximum_prompt_tokens(cluster_specs),
                "adjacent": _maximum_prompt_tokens(adjacent_specs),
            },
            "truncated_count": 0,
        },
    }
    contract_path = output_dir / "run_contract.json"
    if resume and contract_path.is_file():
        existing = _object(contract_path)
        if _without_created_at(existing) != _without_created_at(contract):
            raise ValueError("Task 1 evaluation resume contract drift")
    else:
        atomic_write_json(contract_path, contract)
    if preflight_only:
        path = output_dir / "preflight.json"
        atomic_write_json(
            path,
            {
                "schema_version": "stage_task1_evaluation_preflight",
                "status": "passed_zero_call",
                "run_contract_sha256": sha256_file(contract_path),
                **contract["preflight"],
            },
        )
        return path

    endpoint_runtime = None
    if client is None:
        endpoint_runtime = build_endpoint_pool_runtime(evaluation_llm)
        task1_max_tokens = config.call_budgets["task1_judge"].max_output_tokens
        client = build_json_client(
            {
                **evaluation_llm,
                "json_response_format": True,
                "max_tokens": task1_max_tokens,
            },
            endpoint_runtime=endpoint_runtime,
        )
        alignment_recovery_client = build_json_client(
            {
                **evaluation_llm,
                "json_response_format": True,
                "max_tokens": task1_max_tokens * 2,
            },
            endpoint_runtime=endpoint_runtime,
        )
    else:
        alignment_recovery_client = None
    semaphore = asyncio.Semaphore(workers)
    alignments = await _run_jobs(
        alignment_specs,
        directory=partial_root / "checkpoint_alignments",
        failure_directory=failure_root,
        resume=resume,
        call=lambda identifier, spec: _call_checkpoint_alignment(
            identifier,
            spec=spec,
            client=client,
            recovery_client=alignment_recovery_client,
            semaphore=semaphore,
        ),
    )
    evidence_batches = await _run_jobs(
        evidence_specs,
        directory=partial_root / "checkpoint_evidence",
        failure_directory=failure_root,
        resume=resume,
        call=lambda identifier, spec: _call_checkpoint_evidence(
            identifier,
            spec=spec,
            client=client,
            recovery_client=alignment_recovery_client,
            semaphore=semaphore,
        ),
    )
    checkpoints = _compose_checkpoint_results(
        checkpoint_specs=checkpoint_specs,
        alignments=alignments,
        evidence_batches=evidence_batches,
    )
    clusters = await _run_jobs(
        cluster_specs,
        directory=partial_root / "development_clusters",
        failure_directory=failure_root,
        resume=resume,
        recover_failure=_recover_cluster_json_sequence_failure,
        call=lambda identifier, spec: _call_cluster(
            identifier,
            spec=spec,
            client=client,
            recovery_client=alignment_recovery_client,
            semaphore=semaphore,
        ),
    )
    adjacent = await _run_jobs(
        adjacent_specs,
        directory=partial_root / "adjacent_checks",
        failure_directory=failure_root,
        resume=resume,
        call=lambda identifier, spec: _call_adjacent(
            identifier,
            spec=spec,
            client=client,
            recovery_client=alignment_recovery_client,
            semaphore=semaphore,
        ),
    )

    checkpoints_by_character: dict[str, list[dict[str, Any]]] = {
        character_id: [] for character_id in trajectory_by_character
    }
    for row in checkpoints:
        checkpoints_by_character[row["character_id"]].append(row)
    clusters_by_character = {row["character_id"]: row for row in clusters}
    adjacent_by_character: dict[str, list[dict[str, Any]]] = {
        character_id: [] for character_id in trajectory_by_character
    }
    for row in adjacent:
        adjacent_by_character[row["character_id"]].append(row["judgment"])
    trajectory_results = []
    trajectory_judgments = []
    for character_id, trajectory in trajectory_by_character.items():
        order = {value: index for index, value in enumerate(trajectory["checkpoint_ids"])}
        checkpoint_rows = sorted(
            checkpoints_by_character[character_id], key=lambda row: order[row["checkpoint_id"]]
        )
        adjacent_rows = sorted(
            adjacent_by_character[character_id],
            key=lambda row: order[
                next(
                    rubric["checkpoint_id"]
                    for rubric in trajectory["checkpoint_rubrics"]
                    if rubric["instance_id"] == row["earlier_instance_id"]
                )
            ],
        )
        judgment = {
            "development_clusters": clusters_by_character[character_id]["judgment"][
                "development_clusters"
            ],
            "adjacent_checks": adjacent_rows,
        }
        trajectory_judgments.append(
            {"character_id": character_id, "character": trajectory["character"], **judgment}
        )
        trajectory_results.append(
            score_task1_trajectory(
                trajectory=trajectory,
                checkpoint_results=checkpoint_rows,
                trajectory_judgment=judgment,
            )
        )
    aggregate = aggregate_task1(trajectory_results)
    output_path = output_dir / "task1_evaluation.json"
    atomic_write_json(
        output_path,
        {
            "schema_version": "stage_task1_evaluation",
            "movie_id": private["movie_id"],
            "trajectory_count": len(trajectory_results),
            "checkpoint_count": len(checkpoints),
            "aggregate": aggregate,
            "checkpoint_results": checkpoints,
            "trajectory_judgments": trajectory_judgments,
        },
    )
    endpoint_snapshot = await endpoint_runtime.snapshot() if endpoint_runtime else None
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task1_evaluation_run_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "completed",
            "evaluation_mode": evaluation_mode,
            "movie_id": private["movie_id"],
            "python_executable": sys.executable,
            "model_contract": contract["model_contract"],
            "run_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            "endpoint_pool": endpoint_snapshot,
            "counts": {
                "expected_checkpoints": len(checkpoint_specs),
                "evaluated_checkpoints": len(checkpoints),
                "expected_checkpoint_alignment_judges": len(alignment_specs),
                "evaluated_checkpoint_alignment_judges": len(alignments),
                "expected_checkpoint_evidence_judges": len(evidence_specs),
                "evaluated_checkpoint_evidence_judges": len(evidence_batches),
                "expected_trajectories": len(cluster_specs),
                "evaluated_trajectories": len(trajectory_results),
                "expected_adjacent_pairs": len(adjacent_specs),
                "evaluated_adjacent_pairs": len(adjacent),
                "failure_count": 0,
                "truncated_count": 0,
            },
            "outputs": [{"path": str(output_path), "sha256": sha256_file(output_path)}],
        },
    )
    return manifest_path


async def _call_checkpoint_alignment(
    instance_id: str,
    *,
    spec: dict[str, Any],
    client: Any,
    recovery_client: Any | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    materialized = spec["materialized"]
    async with semaphore:
        response = await _generate_alignment_with_output_limit_recovery(
            client=client,
            recovery_client=recovery_client,
            materialized=materialized,
            stage=f"task1_alignment_judge:{instance_id}",
        )
    localized = spec["localized_prediction"]
    rubric = spec["rubric"]
    gold_ids = {
            row["local_id"]
            for field in ("current_state_claims", "development_claims")
            for row in rubric[field]
        }
    prediction_ids = {row["local_id"] for row in localized}
    inactive_state_ids = {row["local_id"] for row in rubric["inactive_state_claims"]}
    raw_judgment = deepcopy(response.data)
    candidate, normalizations = _normalize_task1_alignment(
        response.data,
        gold_ids=gold_ids,
        prediction_ids=prediction_ids,
        inactive_state_ids=inactive_state_ids,
    )
    try:
        judgment = validate_task1_alignment_judgment(
            candidate,
            gold_ids=gold_ids,
            prediction_ids=prediction_ids,
            inactive_state_ids=inactive_state_ids,
        )
    except ValueError as exc:
        _attach_response_context(exc, response)
        raise
    result = {
        "instance_id": instance_id,
        "judgment": judgment,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "generator_metadata": response.metadata,
    }
    return _with_normalization_trace(result, raw_judgment, normalizations)


async def _generate_alignment_with_output_limit_recovery(
    *, client: Any, recovery_client: Any | None, materialized: dict[str, Any], stage: str
) -> JsonCall:
    try:
        return await client.generate_json(
            system_prompt=materialized["system_prompt"],
            user_prompt=materialized["user_prompt"],
            stage=stage,
        )
    except ModelResponseParseError as exc:
        if recovery_client is None or exc.metadata.get("finish_reason") != "max_tokens":
            raise
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


async def _call_checkpoint_evidence(
    batch_id: str,
    *,
    spec: dict[str, Any],
    client: Any,
    recovery_client: Any | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    materialized = spec["materialized"]
    prediction_ids = {
        row["local_id"] for row in materialized["localized_prediction"]
    }
    async with semaphore:
        response, effective = await _generate_evidence_judgment(
            client=client,
            recovery_client=recovery_client,
            materialized=materialized,
            stage=f"task1_evidence_judge:{batch_id}",
        )
    try:
        judgment = validate_task1_evidence_judgment(
            response.data, prediction_ids=prediction_ids
        )
    except ValueError as exc:
        _attach_response_context(exc, response)
        raise
    return {
        "batch_id": batch_id,
        "instance_id": spec["instance_id"],
        "batch_index": spec["batch_index"],
        "prediction_ids": sorted(prediction_ids),
        "judgment": judgment,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "effective_prompt_tokens": effective["prompt_tokens"],
        "effective_prompt_sha256": effective["prompt_sha256"],
        "evaluation_input_mode": effective["evidence_mode"],
        "generator_metadata": response.metadata,
    }


async def _generate_evidence_judgment(
    *,
    client: Any,
    recovery_client: Any | None,
    materialized: dict[str, Any],
    stage: str,
) -> tuple[JsonCall, dict[str, Any]]:
    try:
        response = await client.generate_json(
            system_prompt=materialized["system_prompt"],
            user_prompt=materialized["user_prompt"],
            stage=stage,
        )
        response = await _recover_valid_output_limit_finish(
            response=response,
            recovery_client=recovery_client,
            materialized=materialized,
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
        try:
            response = await client.generate_json(
                system_prompt=fallback["system_prompt"],
                user_prompt=fallback["user_prompt"],
                stage=f"{stage}:reviewed_gold_scene_index_fallback",
            )
            response = await _recover_valid_output_limit_finish(
                response=response,
                recovery_client=recovery_client,
                materialized=fallback,
                stage=f"{stage}:reviewed_gold_scene_index_fallback",
            )
        except ModelResponseParseError as fallback_exc:
            if (
                recovery_client is None
                or fallback_exc.metadata.get("finish_reason") != "max_tokens"
            ):
                raise
            recovered = await recovery_client.generate_json(
                system_prompt=fallback["system_prompt"],
                user_prompt=fallback["user_prompt"],
                stage=(
                    f"{stage}:reviewed_gold_scene_index_fallback:"
                    "output_limit_recovery"
                ),
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
                        "fallback_prompt_sha256": fallback["prompt_sha256"],
                        "fallback_evidence_mode": fallback["evidence_mode"],
                    },
                },
            ),
            fallback,
        )


async def _recover_valid_output_limit_finish(
    *, response: JsonCall, recovery_client: Any | None,
    materialized: dict[str, Any], stage: str
) -> JsonCall:
    if response.metadata.get("finish_reason") != "max_tokens":
        return response
    if recovery_client is None:
        raise ValueError(f"{stage} returned valid JSON with a max_tokens finish")
    recovered = await recovery_client.generate_json(
        system_prompt=materialized["system_prompt"],
        user_prompt=materialized["user_prompt"],
        stage=f"{stage}:valid_json_output_limit_recovery",
    )
    return JsonCall(
        data=recovered.data,
        metadata={
            **recovered.metadata,
            "output_limit_recovery": {
                "trigger": "valid_json_max_tokens_finish",
                "primary_response_metadata": response.metadata,
            },
        },
    )


def _compose_checkpoint_results(
    *, checkpoint_specs: dict[str, dict[str, Any]], alignments: list[dict[str, Any]],
    evidence_batches: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    alignment_by_id = {row["instance_id"]: row for row in alignments}
    evidence_by_id: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in checkpoint_specs
    }
    for row in evidence_batches:
        evidence_by_id[row["instance_id"]].append(row)
    if set(alignment_by_id) != set(checkpoint_specs):
        raise ValueError("Task 1 alignment results lack exact checkpoint coverage")
    output = []
    for instance_id, spec in checkpoint_specs.items():
        localized = spec["localized_prediction"]
        alignment = alignment_by_id[instance_id]
        batches = sorted(evidence_by_id[instance_id], key=lambda row: row["batch_index"])
        observed_prediction_ids = [
            value for row in batches for value in row["prediction_ids"]
        ]
        expected_prediction_ids = [row["local_id"] for row in localized]
        _exact_ids(expected_prediction_ids, observed_prediction_ids, "evidence batch predictions")
        judgment = {
            "claim_pair_judgments": alignment["judgment"]["claim_pair_judgments"],
            "prediction_checks": [
                check
                for row in batches
                for check in row["judgment"]["prediction_checks"]
            ],
            "future_leak_prediction_ids": sorted(
                {
                    value
                    for row in batches
                    for value in row["judgment"]["future_leak_prediction_ids"]
                }
            ),
            "premature_update_prediction_ids": sorted(
                {
                    value
                    for row in batches
                    for value in row["judgment"]["premature_update_prediction_ids"]
                }
            ),
            "false_persistence_pairs": alignment["judgment"]["false_persistence_pairs"],
            "no_change_false_update": alignment["judgment"]["no_change_false_update"],
        }
        rubric = spec["rubric"]
        validate_task1_checkpoint_judgment(
            judgment,
            gold_ids={
                row["local_id"]
                for field in ("current_state_claims", "development_claims")
                for row in rubric[field]
            },
            prediction_ids=set(expected_prediction_ids),
            inactive_state_ids={
                row["local_id"] for row in rubric["inactive_state_claims"]
            },
        )
        output.append(
            {
                "movie_id": spec["movie_id"],
                "character_id": spec["trajectory"]["character_id"],
                "character": spec["trajectory"]["character"],
                "instance_id": instance_id,
                "checkpoint_id": rubric["checkpoint_id"],
                "localized_prediction": localized,
                "judgment": judgment,
                "scoring": score_task1_checkpoint(
                    prediction=localized, rubric=rubric, judgment=judgment
                ),
                "staged_judgments": {
                    "alignment": alignment,
                    "evidence_batches": batches,
                },
            }
        )
    return output


async def _call_cluster(
    character_id: str,
    *,
    spec: dict[str, Any],
    client: Any,
    recovery_client: Any | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    materialized = spec["materialized"]
    async with semaphore:
        response, parse_normalizations = await _generate_cluster_judgment(
            client=client,
            recovery_client=recovery_client,
            materialized=materialized,
            stage=f"task1_development_cluster_judge:{character_id}",
        )
    allowed_refs = set(materialized["development_prediction_refs"])
    raw_judgment = deepcopy(response.data)
    candidate, normalizations = _normalize_task1_clusters(response.data, allowed_refs)
    normalizations.extend(parse_normalizations)
    try:
        judgment = validate_task1_development_cluster_judgment(
            candidate,
            development_prediction_refs=allowed_refs,
        )
    except ValueError as exc:
        _attach_response_context(exc, response)
        raise
    result = {
        "character_id": character_id,
        "judgment": judgment,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "generator_metadata": response.metadata,
    }
    return _with_normalization_trace(result, raw_judgment, normalizations)


async def _generate_cluster_judgment(
    *, client: Any, recovery_client: Any | None, materialized: dict[str, Any], stage: str
) -> tuple[JsonCall, list[str]]:
    try:
        response = await client.generate_json(
            system_prompt=materialized["system_prompt"],
            user_prompt=materialized["user_prompt"],
            stage=stage,
        )
        return response, []
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
                [],
            )
        if exc.metadata.get("finish_reason") != "end_turn":
            raise
        try:
            objects = _decode_json_sequence(exc.raw_text)
        except (json.JSONDecodeError, ValueError):
            raise exc
        if len(objects) < 2:
            raise exc
        clusters: list[dict[str, Any]] = []
        for payload in objects:
            if set(payload) != {"development_clusters"} or not isinstance(
                payload["development_clusters"], list
            ):
                raise exc
            clusters.extend(payload["development_clusters"])
        return (
            JsonCall(
                data={"development_clusters": clusters},
                metadata={
                    **exc.metadata,
                    "json_sequence_recovery": {
                        "object_count": len(objects),
                        "raw_response_sha256": exc.metadata.get(
                            "raw_response_sha256"
                        ),
                    },
                },
            ),
            ["merged_json_sequence"],
        )


def _decode_json_sequence(raw_text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(raw_text):
        while cursor < len(raw_text) and raw_text[cursor].isspace():
            cursor += 1
        if cursor == len(raw_text):
            break
        payload, cursor = decoder.raw_decode(raw_text, cursor)
        if not isinstance(payload, dict):
            raise ValueError("JSON sequence member is not an object")
        objects.append(payload)
    return objects


def _recover_cluster_json_sequence_failure(
    identifier: str, spec: dict[str, Any], failure_path: Path
) -> dict[str, Any] | None:
    failure = _object(failure_path)
    materialized = spec["materialized"]
    metadata = failure.get("generator_metadata")
    if (
        failure.get("error_type") != "ModelResponseParseError"
        or failure.get("prompt_sha256") != materialized["prompt_sha256"]
        or not isinstance(metadata, dict)
        or metadata.get("finish_reason") != "end_turn"
    ):
        return None
    try:
        objects = _decode_json_sequence(str(failure.get("raw_response") or ""))
    except (json.JSONDecodeError, ValueError):
        return None
    if len(objects) < 2:
        return None
    clusters: list[dict[str, Any]] = []
    for payload in objects:
        if set(payload) != {"development_clusters"} or not isinstance(
            payload["development_clusters"], list
        ):
            return None
        clusters.extend(payload["development_clusters"])
    raw_judgment = {"development_clusters": clusters}
    allowed_refs = set(materialized["development_prediction_refs"])
    candidate, normalizations = _normalize_task1_clusters(raw_judgment, allowed_refs)
    normalizations.append("merged_json_sequence")
    judgment = validate_task1_development_cluster_judgment(
        candidate,
        development_prediction_refs=allowed_refs,
    )
    result = {
        "character_id": identifier,
        "judgment": judgment,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "generator_metadata": {
            **metadata,
            "json_sequence_recovery": {
                "object_count": len(objects),
                "raw_response_sha256": metadata.get("raw_response_sha256"),
                "recovered_from_failure_artifact": str(failure_path),
                "failure_artifact_sha256": sha256_file(failure_path),
            },
        },
    }
    return _with_normalization_trace(result, raw_judgment, normalizations)


async def _call_adjacent(
    pair_id: str,
    *,
    spec: dict[str, Any],
    client: Any,
    recovery_client: Any | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    materialized = spec["materialized"]
    async with semaphore:
        response, effective = await _generate_evidence_judgment(
            client=client,
            recovery_client=recovery_client,
            materialized=materialized,
            stage=f"task1_adjacent_judge:{pair_id}",
        )
    raw_judgment = deepcopy(response.data)
    candidate, normalizations = _normalize_task1_adjacent(
        response.data,
        earlier_instance_id=materialized["earlier_instance_id"],
        later_instance_id=materialized["later_instance_id"],
    )
    try:
        judgment = validate_task1_adjacent_judgment(
            candidate,
            earlier_instance_id=materialized["earlier_instance_id"],
            later_instance_id=materialized["later_instance_id"],
        )
    except ValueError as exc:
        _attach_response_context(exc, response)
        raise
    result = {
        "character_id": spec["trajectory"]["character_id"],
        "pair_id": pair_id,
        "judgment": judgment,
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "effective_prompt_tokens": effective["prompt_tokens"],
        "effective_prompt_sha256": effective["prompt_sha256"],
        "evaluation_input_mode": effective["evidence_mode"],
        "generator_metadata": response.metadata,
    }
    return _with_normalization_trace(result, raw_judgment, normalizations)


def _normalize_task1_adjacent(
    payload: dict[str, Any], *, earlier_instance_id: str, later_instance_id: str
) -> tuple[dict[str, Any], list[str]]:
    candidate = deepcopy(payload)
    if not isinstance(candidate, dict):
        return candidate, []
    normalizations: list[str] = []
    expected = {
        "earlier_instance_id": earlier_instance_id,
        "later_instance_id": later_instance_id,
    }
    for field, value in expected.items():
        if field in candidate and candidate[field] != value:
            candidate[field] = value
            normalizations.append("restored_expected_adjacent_instance_ids")
    return candidate, sorted(set(normalizations))


def _normalize_task1_alignment(
    payload: dict[str, Any], *, gold_ids: set[str], prediction_ids: set[str],
    inactive_state_ids: set[str]
) -> tuple[dict[str, Any], list[str]]:
    candidate = deepcopy(payload)
    normalizations: list[str] = []
    if not isinstance(candidate, dict):
        return candidate, normalizations
    pairs = candidate.get("claim_pair_judgments")
    if isinstance(pairs, list):
        seen = set()
        kept = []
        for row in pairs:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            key = (row.get("gold_local_id"), row.get("prediction_local_id"))
            if (
                key[0] not in gold_ids
                or key[1] not in prediction_ids
                or key in seen
                or row.get("label") not in {"full", "partial", "contradiction"}
            ):
                normalizations.append("dropped_unknown_or_duplicate_claim_pair")
                continue
            seen.add(key)
            kept.append(row)
        candidate["claim_pair_judgments"] = kept
    false_pairs = candidate.get("false_persistence_pairs")
    if isinstance(false_pairs, list):
        seen = set()
        kept = []
        for row in false_pairs:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            key = (row.get("inactive_state_local_id"), row.get("prediction_local_id"))
            if key[0] not in inactive_state_ids or key[1] not in prediction_ids or key in seen:
                normalizations.append("dropped_unknown_or_duplicate_false_persistence_pair")
                continue
            seen.add(key)
            kept.append(row)
        candidate["false_persistence_pairs"] = kept
    return candidate, sorted(set(normalizations))


def _normalize_task1_clusters(
    payload: dict[str, Any], allowed_refs: set[str]
) -> tuple[dict[str, Any], list[str]]:
    candidate = deepcopy(payload)
    if not isinstance(candidate, dict) or not isinstance(candidate.get("development_clusters"), list):
        return candidate, []
    normalizations: list[str] = []
    observed = set()
    clusters = []
    for row in candidate["development_clusters"]:
        if not isinstance(row, dict) or not isinstance(row.get("members"), list):
            clusters.append(row)
            continue
        members = []
        for ref in row["members"]:
            if ref not in allowed_refs or ref in observed:
                normalizations.append("dropped_unknown_or_duplicate_cluster_member")
                continue
            observed.add(ref)
            members.append(ref)
        if members:
            clusters.append({**row, "members": members})
        else:
            normalizations.append("dropped_empty_cluster")
    for ref in sorted(allowed_refs - observed):
        clusters.append(
            {
                "cluster_id": "",
                "members": [ref],
                "brief_rationale": "Conservative singleton for an omitted prediction reference.",
            }
        )
        normalizations.append("added_singleton_for_omitted_cluster_member")
    expected_ids = [f"PD{index}" for index in range(1, len(clusters) + 1)]
    actual_ids = [row.get("cluster_id") if isinstance(row, dict) else None for row in clusters]
    if actual_ids != expected_ids:
        for index, row in enumerate(clusters, start=1):
            if isinstance(row, dict):
                row["cluster_id"] = f"PD{index}"
        normalizations.append("renumbered_cluster_ids")
    candidate["development_clusters"] = clusters
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


def _attach_response_context(exc: Exception, response: Any) -> None:
    setattr(exc, "parsed_response", getattr(response, "data", None))
    setattr(exc, "generator_metadata", getattr(response, "metadata", None))


async def _run_jobs(
    specs: dict[str, dict[str, Any]],
    *,
    directory: Path,
    failure_directory: Path,
    resume: bool,
    call: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    recover_failure: Callable[
        [str, dict[str, Any], Path], dict[str, Any] | None
    ]
    | None = None,
) -> list[dict[str, Any]]:
    async def one(identifier: str, spec: dict[str, Any]) -> dict[str, Any]:
        path = directory / f"{identifier}.json"
        materialized = spec["materialized"]
        if resume and path.is_file():
            existing = _object(path)
            if existing.get("prompt_sha256") != materialized["prompt_sha256"]:
                raise ValueError(f"Task 1 partial prompt drift: {path}")
            (failure_directory / f"{identifier}.json").unlink(missing_ok=True)
            return existing
        failure_path = failure_directory / f"{identifier}.json"
        if resume and recover_failure is not None and failure_path.is_file():
            recovered = recover_failure(identifier, spec, failure_path)
            if recovered is not None:
                atomic_write_json(path, recovered)
                failure_path.unlink()
                return recovered
        try:
            result = await call(identifier, spec)
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
        *(one(identifier, spec) for identifier, spec in specs.items()),
        return_exceptions=True,
    )
    failures = [row for row in settled if isinstance(row, BaseException)]
    if failures:
        raise RuntimeError(
            f"{len(failures)} Task 1 evaluation calls failed; first: "
            f"{type(failures[0]).__name__}: {failures[0]}"
        )
    return [row for row in settled if isinstance(row, dict)]


def _validate_input_contracts(
    *,
    public: dict[str, Any],
    public_path: Path,
    private: dict[str, Any],
    plans: dict[str, Any],
    predictions: dict[str, Any],
) -> None:
    movie_ids = {
        str(public.get("movie_id")),
        str(private.get("movie_id")),
        str(plans.get("movie_id")),
        str(predictions.get("movie_id")),
    }
    if len(movie_ids) != 1:
        raise ValueError(f"Task 1 evaluation movie IDs differ: {sorted(movie_ids)}")
    if sha256_file(public_path.resolve()) != private["public_instances_sha256"]:
        raise ValueError("Task 1 private assets do not match the public instance release")
    if public.get("language") != private.get("language"):
        raise ValueError("Task 1 public/private language differs")
    public_ids = [row["instance_id"] for row in public.get("instances", [])]
    private_ids = [
        rubric["instance_id"]
        for trajectory in private["trajectories"]
        for rubric in trajectory["checkpoint_rubrics"]
    ]
    prediction_ids = [
        row["instance_id"]
        for character in predictions.get("characters", [])
        for row in character.get("checkpoint_predictions", [])
    ]
    _exact_ids(public_ids, private_ids, "private assets")
    _exact_ids(public_ids, prediction_ids, "predictions")
    if int(public.get("instance_count", -1)) != len(public_ids):
        raise ValueError("Task 1 public instance count drift")
    if int(predictions.get("checkpoint_prediction_count", -1)) != len(prediction_ids):
        raise ValueError("Task 1 prediction count drift")
    private_characters = {row["character_id"] for row in private["trajectories"]}
    plan_characters = {row["character_id"] for row in plans.get("plans", [])}
    if private_characters != plan_characters:
        raise ValueError("Task 1 private assets and rolling plans differ by character")


def _predictions_by_instance(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for character in payload["characters"]:
        for row in character["checkpoint_predictions"]:
            validate_task1_prediction(row["prediction"])
            if row["instance_id"] in output:
                raise ValueError("Task 1 predictions contain duplicate instance IDs")
            output[row["instance_id"]] = {**row, "character_id": character["character_id"]}
    return output


def _validate_independent_judge_identity(
    payload: Any, *, judge_model: str, actor_model: str
) -> None:
    required = {
        "provider",
        "model_version",
        "access_date",
        "independence_basis",
        "weights_and_model_family_distinct_from_actor",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Formal Task 1 evaluation requires a complete judge_identity contract")
    if str(payload["model_version"]).strip() != judge_model:
        raise ValueError("Task 1 judge_identity model_version differs from the judge model")
    if not str(payload["provider"]).strip() or not str(payload["access_date"]).strip():
        raise ValueError("Task 1 judge_identity provider and access_date must be nonempty")
    if not str(payload["independence_basis"]).strip():
        raise ValueError("Task 1 judge_identity independence_basis must be nonempty")
    if payload["weights_and_model_family_distinct_from_actor"] is not True:
        raise ValueError("Task 1 formal judge must attest distinct weights and model family")
    if actor_model == judge_model:
        raise ValueError("Task 1 judge identity conflicts with the actor model")


def _prompt_artifacts(language: str) -> list[dict[str, str]]:
    language_key = "zh" if language.casefold() in {"zh", "chinese"} else "en"
    paths = set()
    for name in (
        "task1_alignment_judge",
        "task1_evidence_judge",
        "task1_development_cluster_judge",
        "task1_adjacent_judge",
    ):
        paths.update(PROMPTS.get(f"{language_key}/evaluation/{name}").source_paths)
    return [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(paths)]


def _maximum_prompt_tokens(specs: dict[str, dict[str, Any]]) -> int:
    return max((row["materialized"]["accounted_input_tokens"] for row in specs.values()), default=0)


def _exact_ids(expected: list[str], observed: list[str], label: str) -> None:
    if len(expected) != len(set(expected)) or len(observed) != len(set(observed)):
        raise ValueError(f"Task 1 {label} contains duplicate IDs")
    if set(expected) != set(observed):
        raise ValueError(
            f"Task 1 {label} ID mismatch: missing={sorted(set(expected) - set(observed))} "
            f"extra={sorted(set(observed) - set(expected))}"
        )


def _without_created_at(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at"}


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path.resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
