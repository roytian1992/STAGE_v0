from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..clients import build_endpoint_pool_runtime, build_json_client
from ..io import atomic_write_json, sha256_file, sha256_json
from ..prompt_loader import PROMPTS
from ..temporal.benchmark_protocol import BenchmarkRuntimeConfig
from .materialization import materialize_task3_pair_judge, materialize_task3_response_judge
from .runner import (
    _call_pair,
    _call_task3,
    _read,
    _require_exact_ids,
    _run_jobs,
    _validate_independent_judge_identity,
    _validate_pair_annotations,
)
from .task3_metrics import aggregate_task3


EVALUATION_MODES = {"formal_independent_evaluation", "self_judge_diagnostic"}


async def run_task3_evaluation(
    *,
    instances_path: Path,
    context_packs_path: Path,
    gold_rubrics_path: Path,
    pair_groups_path: Path,
    pair_annotations_path: Path,
    prediction_path: Path,
    config_path: Path,
    output_dir: Path,
    workers: int = 32,
    evaluation_mode: str = "formal_independent_evaluation",
    resume: bool = False,
    preflight_only: bool = False,
    client: Any | None = None,
) -> Path:
    if workers <= 0 or workers > 64:
        raise ValueError("Task 3 evaluation workers must be in 1..64")
    if evaluation_mode not in EVALUATION_MODES:
        raise ValueError(f"Unsupported Task 3 evaluation mode: {evaluation_mode}")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite Task 3 evaluation: {output_dir}")
    partial_root = output_dir / "partials"
    failure_root = partial_root / "failures"
    for path in (output_dir, partial_root / "responses", partial_root / "pairs", failure_root):
        path.mkdir(parents=True, exist_ok=True)

    instances = _read(instances_path)
    context_packs = _read(context_packs_path)
    gold = _read(gold_rubrics_path)
    pair_groups = _read(pair_groups_path)
    annotations = _read(pair_annotations_path)
    predictions = _read(prediction_path)
    config = BenchmarkRuntimeConfig.load(config_path)
    counter = config.build_token_counter()
    movie_ids = {
        str(payload["movie_id"])
        for payload in (instances, context_packs, gold, pair_groups, annotations, predictions)
    }
    if len(movie_ids) != 1:
        raise ValueError(f"Task 3 inputs belong to different movies: {sorted(movie_ids)}")
    movie_id = next(iter(movie_ids))

    instance_by_id = {
        row["instance_id"]: _normalized_instance(row) for row in instances["instances"]
    }
    context_by_id = {row["instance_id"]: row for row in context_packs["context_packs"]}
    gold_by_id = {row["instance_id"]: row for row in gold["rubrics"]}
    prediction_by_id = {row["instance_id"]: row for row in predictions["predictions"]}
    _require_exact_ids(instance_by_id, context_by_id, "Task 3 context packs")
    _require_exact_ids(instance_by_id, gold_by_id, "Task 3 gold rubrics")
    _require_exact_ids(instance_by_id, prediction_by_id, "Task 3 predictions")
    annotation_by_id = _validate_pair_annotations(
        annotations, source_pairs=pair_groups, instances=instance_by_id
    )

    prediction_manifest_path = prediction_path.resolve().parent / "manifest.json"
    prediction_manifest = _read(prediction_manifest_path)
    actor_model = str(prediction_manifest.get("model") or "").strip()
    evaluation_llm = config.evaluation_llm
    judge_model = str(evaluation_llm.get("model") or "").strip()
    if not actor_model or not judge_model:
        raise ValueError("Task 3 evaluation requires actor and judge model identities")
    independent = actor_model != judge_model
    if evaluation_mode == "formal_independent_evaluation":
        if not independent:
            raise ValueError("Formal Task 3 evaluation requires a judge distinct from the actor")
        _validate_independent_judge_identity(
            evaluation_llm.get("judge_identity"),
            judge_model=judge_model,
            actor_models={actor_model},
        )
    elif independent:
        raise ValueError("Self-judge diagnostic mode requires actor and judge models to match")

    language = str(instances["instances"][0]["language"])
    response_specs = {
        instance_id: materialize_task3_response_judge(
            gold=row,
            prediction=prediction_by_id[instance_id],
            context_pack=context_by_id[instance_id],
            instance=instance_by_id[instance_id],
            language=language,
            config=config,
            counter=counter,
        )
        for instance_id, row in gold_by_id.items()
    }
    pair_specs = {
        pair_id: materialize_task3_pair_judge(
            annotation=annotation,
            predictions=prediction_by_id,
            gold=gold_by_id,
            instances=instance_by_id,
            language=language,
            config=config,
            counter=counter,
        )
        for pair_id, annotation in annotation_by_id.items()
    }
    prompt_paths = {
        path
        for task in ("task3_response_judge", "task3_pair_judge")
        for path in PROMPTS.get(config.prompt_path(task, language)).source_paths
    }
    contract = {
        "schema_version": "stage_task3_evaluation_run_contract",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_mode": evaluation_mode,
        "movie_id": movie_id,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for path in (
                instances_path,
                context_packs_path,
                gold_rubrics_path,
                pair_groups_path,
                pair_annotations_path,
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
        "prompt_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in sorted(prompt_paths)
        ],
        "preflight": {
            "response_judges": len(response_specs),
            "pair_judges": len(pair_specs),
            "max_input_tokens": {
                "response": max(row["prompt_tokens"] for row in response_specs.values()),
                "pair": max(row["prompt_tokens"] for row in pair_specs.values()),
            },
            "truncated_count": 0,
        },
    }
    contract_path = output_dir / "run_contract.json"
    if resume and contract_path.is_file():
        existing = _read(contract_path)
        if _without_created_at(existing) != _without_created_at(contract):
            raise ValueError("Task 3 evaluation resume contract drift")
    else:
        atomic_write_json(contract_path, contract)
    if preflight_only:
        path = output_dir / "preflight.json"
        atomic_write_json(
            path,
            {
                "schema_version": "stage_task3_evaluation_preflight",
                "status": "passed_zero_call",
                "run_contract_sha256": sha256_file(contract_path),
                **contract["preflight"],
            },
        )
        return path

    endpoint_runtime = None
    if client is None:
        endpoint_runtime = build_endpoint_pool_runtime(evaluation_llm)
        response_max_tokens = config.call_budgets[
            "task3_response_judge"
        ].max_output_tokens
        response_client = build_json_client(
            {
                **evaluation_llm,
                "json_response_format": True,
                "max_tokens": response_max_tokens,
            },
            endpoint_runtime=endpoint_runtime,
        )
        response_recovery_client = build_json_client(
            {
                **evaluation_llm,
                "json_response_format": True,
                "max_tokens": response_max_tokens * 2,
            },
            endpoint_runtime=endpoint_runtime,
        )
        pair_client = build_json_client(
            {
                **evaluation_llm,
                "json_response_format": True,
                "max_tokens": config.call_budgets["task3_pair_judge"].max_output_tokens,
            },
            endpoint_runtime=endpoint_runtime,
        )
        pair_recovery_client = build_json_client(
            {
                **evaluation_llm,
                "json_response_format": True,
                "max_tokens": config.call_budgets["task3_pair_judge"].max_output_tokens * 2,
            },
            endpoint_runtime=endpoint_runtime,
        )
    else:
        response_client = client
        response_recovery_client = None
        pair_client = client
        pair_recovery_client = None
    semaphore = asyncio.Semaphore(workers)
    responses = await _run_jobs(
        response_specs,
        directory=partial_root / "responses",
        resume=resume,
        failure_directory=failure_root,
        runner=lambda identifier, materialized: _call_task3(
            identifier,
            materialized=materialized,
            gold=gold_by_id[identifier],
            client=response_client,
            recovery_client=response_recovery_client,
            semaphore=semaphore,
        ),
    )
    response_by_id = {row["instance_id"]: row for row in responses}
    for pair_id, materialized in pair_specs.items():
        annotation = annotation_by_id[pair_id]
        reviewed = [
            {
                "response_label": f"T{index}",
                "instance_id": instance_id,
                "reviewed_response_judgment": response_by_id[instance_id]["judgment"],
            }
            for index, instance_id in enumerate(annotation["ordered_instance_ids"], 1)
        ]
        fallback_system = (
            materialized["system_prompt"]
            + "\nProvider-filter fallback: compare only the completed independent response "
            "audit summaries below. The original response strings were withheld after a "
            "provider refusal. Use abstract placeholder excerpts T1 and T2; they will be "
            "rebound to the source responses deterministically after validation."
        )
        fallback_user = (
            "Compare the reviewed response-judgment summaries in chronological order.\n"
            f"<pair_type>{annotation['pair_type']}</pair_type>"
            f"<expected_direction>{annotation['expected_direction']}</expected_direction>"
            f"<reviewed_response_judgments>{json.dumps(reviewed, ensure_ascii=False)}</reviewed_response_judgments>\n"
            "Use response_excerpt values T1 and T2 respectively, and cite only pair-local "
            "evidence labels. Return only this exact JSON schema with no wrapper or extra keys: "
            '{"pair_type":"expected_change|expected_stability|knowledge_acquisition|relationship_change",'
            '"response_assessments":[{"response_label":"T1","response_excerpt":"T1",'
            '"observed_behavior":"abstract observed behavior","supports_expected_component":true},'
            '{"response_label":"T2","response_excerpt":"T2",'
            '"observed_behavior":"abstract observed behavior","supports_expected_component":true}],'
            '"expected_direction_present":true,"unsupported_drift":false,'
            '"knowledge_boundaries_preserved":true,"local_evidence_labels":[],'
            '"brief_rationale":"short audit rationale"}'
        )
        fallback_prompt_tokens = (
            counter.count(fallback_system)
            + counter.count(fallback_user)
            + config.reserved_chat_template_tokens
        )
        fallback_maximum = config.call_budgets["task3_pair_judge"].max_input_tokens
        if fallback_prompt_tokens > fallback_maximum:
            raise ValueError(
                f"Task 3 reviewed pair fallback exceeds budget: "
                f"{fallback_prompt_tokens}>{fallback_maximum}"
            )
        materialized["reviewed_judgment_fallback"] = {
            "system_prompt": fallback_system,
            "user_prompt": fallback_user,
            "prompt_tokens": fallback_prompt_tokens,
            "prompt_sha256": sha256_json(
                {"system": fallback_system, "user": fallback_user}
            ),
            "context_mode": "reviewed_response_judgments_without_actor_responses",
        }
    pairs = await _run_jobs(
        pair_specs,
        directory=partial_root / "pairs",
        resume=resume,
        failure_directory=failure_root,
        runner=lambda identifier, materialized: _call_pair(
            identifier,
            materialized=materialized,
            annotation=annotation_by_id[identifier],
            client=pair_client,
            recovery_client=pair_recovery_client,
            semaphore=semaphore,
        ),
    )
    for pair in pairs:
        pair["response_prerequisites"] = [
            {
                "instance_id": instance_id,
                "stance_compatible": response_by_id[instance_id]["judgment"]["stance_compatible"],
                "future_leakage": response_by_id[instance_id]["judgment"]["future_leakage"],
                "unknown_fact_hallucination": response_by_id[instance_id]["judgment"]["unknown_fact_hallucination"],
                "passed": bool(
                    response_by_id[instance_id]["judgment"]["stance_compatible"]
                    and not response_by_id[instance_id]["judgment"]["future_leakage"]
                    and not response_by_id[instance_id]["judgment"]["unknown_fact_hallucination"]
                ),
            }
            for instance_id in pair["instance_ids"]
        ]
        atomic_write_json(partial_root / "pairs" / f"{pair['pair_group_id']}.json", pair)

    output_path = output_dir / "task3_evaluation.json"
    atomic_write_json(
        output_path,
        {
            "schema_version": "stage_task3_evaluation",
            "movie_id": movie_id,
            "instance_count": len(responses),
            "pair_count": len(pairs),
            "aggregate": aggregate_task3(responses, pairs),
            "instances": responses,
            "pairs": pairs,
        },
    )
    endpoint_snapshot = await endpoint_runtime.snapshot() if endpoint_runtime else None
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task3_evaluation_run_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "completed",
            "evaluation_mode": evaluation_mode,
            "movie_id": movie_id,
            "python_executable": sys.executable,
            "model_contract": contract["model_contract"],
            "run_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            "endpoint_pool": endpoint_snapshot,
            "counts": {
                "expected_responses": len(response_specs),
                "evaluated_responses": len(responses),
                "expected_pairs": len(pair_specs),
                "evaluated_pairs": len(pairs),
                "failure_count": 0,
                "truncated_count": 0,
            },
            "outputs": [{"path": str(output_path), "sha256": sha256_file(output_path)}],
        },
    )
    return manifest_path


def _without_created_at(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at"}


def _normalized_instance(row: dict[str, Any]) -> dict[str, Any]:
    model_input = dict(row["model_input"])
    model_input.setdefault("checkpoint_anchor", row["checkpoint_boundary"])
    return {**row, "model_input": model_input}
