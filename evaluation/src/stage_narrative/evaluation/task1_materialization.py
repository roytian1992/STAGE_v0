from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..io import load_scenes, sha256_file, sha256_json
from ..prompt_loader import PROMPTS, YamlPromptRegistry
from ..temporal.benchmark_protocol import BenchmarkRuntimeConfig
from .materialization import task1_evidence_excerpts
from .task1_metrics import localize_task1_prediction
from .task1_schemas import validate_task1_prediction


def materialize_task1_rolling_prompt(
    *,
    character_plan: dict[str, Any],
    call: dict[str, Any],
    prior_scored_memory: dict[str, Any],
    private_unscored_memory: dict[str, Any],
    script_path: Path,
    token_manifest: dict[str, Any],
    language: str,
    token_counter: Any,
    config: BenchmarkRuntimeConfig,
    prompt_registry: YamlPromptRegistry = PROMPTS,
) -> dict[str, Any]:
    validate_task1_prediction(prior_scored_memory)
    if not isinstance(private_unscored_memory, dict) or set(private_unscored_memory) != {
        "unresolved_threads"
    }:
        raise ValueError("Task 1 private rolling memory must contain unresolved_threads only")
    resolved_script = script_path.resolve()
    if sha256_file(resolved_script) != token_manifest.get("script_sha256"):
        raise ValueError("Task 1 runtime script differs from its token manifest")
    scenes = {scene.scene_id: scene for scene in load_scenes(resolved_script)}
    units = {row["unit_id"]: row for row in token_manifest.get("scene_units", [])}
    refs = list(call.get("scene_unit_refs", []))
    if not refs or any(value not in units for value in refs):
        raise ValueError("Task 1 rolling call contains unknown or empty scene-unit refs")
    block = []
    for ref in refs:
        unit = units[ref]
        scene = scenes[unit["scene_id"]]
        block.append(
            {
                "scene_order": scene.order,
                "source_scene_id": scene.source_scene_id,
                "part_order": unit["part_order"],
                "part_count": unit["part_count"],
                "title": scene.title,
                "subtitle": scene.subtitle,
                "content": scene.content[int(unit["char_start"]) : int(unit["char_end"])],
            }
        )
    language_key = _language_key(language)
    prompt_path = f"{language_key}/evaluation/task1_prediction"
    system, user = prompt_registry.render(
        prompt_path,
        focal_character=character_plan["focal_character"],
        aliases=character_plan.get("aliases", []),
        previous_checkpoint_scene_order=call["previous_checkpoint_scene_order"],
        block_end_scene_order=call["block_end_scene_order"],
        checkpoint_at_block_end=call["checkpoint_at_block_end"],
        prior_scored_memory=prior_scored_memory,
        private_unscored_memory=private_unscored_memory,
        screenplay_block=block,
    )
    prompt_tokens = token_counter.count(system) + token_counter.count(user)
    accounted = prompt_tokens + config.reserved_chat_template_tokens
    maximum = config.call_budgets["task1_prediction"].max_input_tokens
    if accounted > maximum:
        raise ValueError(f"Task 1 rolling prompt exceeds input budget: {accounted}>{maximum}")
    return {
        "call_id": call["call_id"],
        "system_prompt": system,
        "user_prompt": user,
        "prompt_path": prompt_path,
        "prompt_sha256": sha256_json({"system": system, "user": user}),
        "raw_prompt_tokens": prompt_tokens,
        "accounted_input_tokens": accounted,
        "max_input_tokens": maximum,
        "checkpoint_at_block_end": bool(call["checkpoint_at_block_end"]),
        "checkpoint_id": call.get("checkpoint_id", ""),
        "task_instance_id": call.get("task_instance_id", ""),
    }


def materialize_task1_checkpoint_judge(
    *,
    rubric: dict[str, Any],
    prediction: dict[str, Any],
    scenes: dict[int, Any],
    evidence_bank: dict[str, Any],
    aliases: list[str],
    character_id: str,
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
    prompt_registry: YamlPromptRegistry = PROMPTS,
) -> dict[str, Any]:
    localized = localize_task1_prediction(prediction["prediction"])
    scene_orders = {
        value for row in localized for value in row["evidence_scene_orders"]
    } | {
        value
        for field in (
            "current_state_claims",
            "development_claims",
        )
        for row in rubric[field]
        for value in row.get("supporting_scene_orders", [])
    }
    evidence = task1_evidence_excerpts(
        scene_orders,
        character_id=character_id,
        aliases=aliases,
        evidence_bank=evidence_bank,
        scenes=scenes,
    )
    compact_prediction = [
        [
            row["local_id"],
            "s" if row["prediction_type"] == "current_state" else "d",
            row["evidence_scene_orders"],
            row["claim"],
        ]
        for row in localized
    ]
    compact_active = [
        [row["local_id"], code, row["supporting_scene_orders"], row["claim"]]
        for field, code in (("current_state_claims", "s"), ("development_claims", "d"))
        for row in rubric[field]
    ]
    compact_inactive = [
        [row["local_id"], "x", row["supporting_scene_orders"], row["claim"]]
        for row in rubric["inactive_state_claims"]
    ]
    compact_future = [
        [row["local_id"], "f", row.get("supporting_scene_orders", []), row["claim"]]
        for row in rubric["salient_future_negatives"]
    ]
    prompt_path = f"{_language_key(language)}/evaluation/task1_checkpoint_judge"
    system, user = prompt_registry.render(
        prompt_path,
        checkpoint=rubric["checkpoint"],
        prediction=compact_prediction,
        active_gold_rubric=compact_active,
        inactive_state_rubric=compact_inactive,
        future_negatives=compact_future,
        evidence_scenes=evidence,
        allowed_gold_pair_ids=[row[0] for row in compact_active],
        allowed_inactive_state_ids=[row[0] for row in compact_inactive],
        allowed_prediction_ids=[row[0] for row in compact_prediction],
    )
    result = _budgeted(
        system=system,
        user=user,
        prompt_path=prompt_path,
        budget_name="task1_judge",
        counter=counter,
        config=config,
    ) | {"localized_prediction": localized, "evidence_scenes": evidence}
    return result


def materialize_task1_staged_checkpoint_judges(
    *,
    rubric: dict[str, Any],
    prediction: dict[str, Any],
    scenes: dict[int, Any],
    evidence_bank: dict[str, Any],
    aliases: list[str],
    character_id: str,
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
    prompt_registry: YamlPromptRegistry = PROMPTS,
) -> dict[str, Any]:
    localized = localize_task1_prediction(prediction["prediction"])
    compact_prediction = [
        [
            row["local_id"],
            "s" if row["prediction_type"] == "current_state" else "d",
            row["evidence_scene_orders"],
            row["claim"],
        ]
        for row in localized
    ]
    compact_active = [
        [row["local_id"], code, row["supporting_scene_orders"], row["claim"]]
        for field, code in (("current_state_claims", "s"), ("development_claims", "d"))
        for row in rubric[field]
    ]
    compact_inactive = [
        [row["local_id"], "x", row["supporting_scene_orders"], row["claim"]]
        for row in rubric["inactive_state_claims"]
    ]
    compact_future = [
        [row["local_id"], "f", row.get("supporting_scene_orders", []), row["claim"]]
        for row in rubric["salient_future_negatives"]
    ]
    language_key = _language_key(language)
    alignment_path = f"{language_key}/evaluation/task1_alignment_judge"
    system, user = prompt_registry.render(
        alignment_path,
        checkpoint=rubric["checkpoint"],
        prediction=compact_prediction,
        active_gold_rubric=compact_active,
        inactive_state_rubric=compact_inactive,
        future_negatives=compact_future,
        allowed_gold_pair_ids=[row[0] for row in compact_active],
        allowed_inactive_state_ids=[row[0] for row in compact_inactive],
        allowed_prediction_ids=[row[0] for row in compact_prediction],
    )
    alignment = _budgeted(
        system=system,
        user=user,
        prompt_path=alignment_path,
        budget_name="task1_judge",
        counter=counter,
        config=config,
    )
    batches = []
    current: list[dict[str, Any]] = []
    for row in localized:
        candidate = [*current, row]
        try:
            materialized = _materialize_task1_evidence_batch(
                rows=candidate,
                rubric=rubric,
                compact_active=compact_active,
                compact_future=compact_future,
                scenes=scenes,
                evidence_bank=evidence_bank,
                aliases=aliases,
                character_id=character_id,
                language_key=language_key,
                config=config,
                counter=counter,
                prompt_registry=prompt_registry,
            )
            _validate_evidence_batch_limit(materialized, config)
        except ValueError as exc:
            if not current or "exceeds budget" not in str(exc):
                raise
            batches.append(
                _materialize_task1_evidence_batch(
                    rows=current,
                    rubric=rubric,
                    compact_active=compact_active,
                    compact_future=compact_future,
                    scenes=scenes,
                    evidence_bank=evidence_bank,
                    aliases=aliases,
                    character_id=character_id,
                    language_key=language_key,
                    config=config,
                    counter=counter,
                    prompt_registry=prompt_registry,
                )
            )
            current = [row]
            materialized = _materialize_task1_evidence_batch(
                rows=current,
                rubric=rubric,
                compact_active=compact_active,
                compact_future=compact_future,
                scenes=scenes,
                evidence_bank=evidence_bank,
                aliases=aliases,
                character_id=character_id,
                language_key=language_key,
                config=config,
                counter=counter,
                prompt_registry=prompt_registry,
            )
            _validate_evidence_batch_limit(materialized, config)
        current = candidate if len(candidate) == len(materialized["localized_prediction"]) else current
    if current:
        batches.append(
            _materialize_task1_evidence_batch(
                rows=current,
                rubric=rubric,
                compact_active=compact_active,
                compact_future=compact_future,
                scenes=scenes,
                evidence_bank=evidence_bank,
                aliases=aliases,
                character_id=character_id,
                language_key=language_key,
                config=config,
                counter=counter,
                prompt_registry=prompt_registry,
            )
        )
    return {
        "localized_prediction": localized,
        "alignment": alignment,
        "evidence_batches": batches,
    }


def _validate_evidence_batch_limit(
    materialized: dict[str, Any], config: BenchmarkRuntimeConfig
) -> None:
    if len(materialized["localized_prediction"]) == 1:
        return
    maximum = int(
        config.task1.get(
            "max_evidence_batch_input_tokens",
            config.call_budgets["task1_judge"].max_input_tokens,
        )
    )
    accounted = int(materialized["accounted_input_tokens"])
    if accounted > maximum:
        raise ValueError(
            f"Task 1 evidence batch exceeds budget: {accounted}>{maximum}"
        )


def _materialize_task1_evidence_batch(
    *, rows: list[dict[str, Any]], rubric: dict[str, Any], compact_active: list[Any],
    compact_future: list[Any], scenes: dict[int, Any], evidence_bank: dict[str, Any],
    aliases: list[str], character_id: str, language_key: str,
    config: BenchmarkRuntimeConfig, counter: Any,
    prompt_registry: YamlPromptRegistry, evidence_mode: str = "verbatim_evidence",
) -> dict[str, Any]:
    scene_orders = {value for row in rows for value in row["evidence_scene_orders"]}
    if evidence_mode == "verbatim_evidence":
        evidence = task1_evidence_excerpts(
            scene_orders,
            character_id=character_id,
            aliases=aliases,
            evidence_bank=evidence_bank,
            scenes=scenes,
        )
    elif evidence_mode == "reviewed_gold_scene_index":
        evidence = [
            {
                "scene_order": scene_order,
                "selection_policy": evidence_mode,
                "reviewed_gold_local_ids": [
                    row[0] for row in compact_active if scene_order in row[2]
                ],
                "verbatim_excerpt_withheld": True,
            }
            for scene_order in sorted(scene_orders)
        ]
    else:
        raise ValueError(f"Unknown Task 1 evidence mode: {evidence_mode}")
    compact = [
        [
            row["local_id"],
            "s" if row["prediction_type"] == "current_state" else "d",
            row["evidence_scene_orders"],
            row["claim"],
        ]
        for row in rows
    ]
    prompt_path = f"{language_key}/evaluation/task1_evidence_judge"
    system, user = prompt_registry.render(
        prompt_path,
        checkpoint=rubric["checkpoint"],
        prediction_batch=compact,
        active_gold_rubric=compact_active,
        future_negatives=compact_future,
        evidence_scenes=evidence,
        allowed_prediction_ids=[row[0] for row in compact],
    )
    try:
        budgeted = _budgeted(
            system=system,
            user=user,
            prompt_path=prompt_path,
            budget_name="task1_judge",
            counter=counter,
            config=config,
        )
    except ValueError as exc:
        if evidence_mode != "verbatim_evidence" or "exceeds budget" not in str(exc):
            raise
        fallback = _materialize_task1_evidence_batch(
            rows=rows,
            rubric=rubric,
            compact_active=compact_active,
            compact_future=compact_future,
            scenes=scenes,
            evidence_bank=evidence_bank,
            aliases=aliases,
            character_id=character_id,
            language_key=language_key,
            config=config,
            counter=counter,
            prompt_registry=prompt_registry,
            evidence_mode="reviewed_gold_scene_index",
        )
        return {
            **fallback,
            "content_filter_fallback": dict(fallback),
            "budget_fallback": {
                "trigger": "verbatim_evidence_prompt_exceeded_context_budget",
                "original_error": str(exc),
            },
        }
    result = budgeted | {
        "localized_prediction": rows,
        "evidence_scenes": evidence,
        "evidence_mode": evidence_mode,
    }
    if evidence_mode == "verbatim_evidence":
        result["content_filter_fallback"] = _materialize_task1_evidence_batch(
            rows=rows,
            rubric=rubric,
            compact_active=compact_active,
            compact_future=compact_future,
            scenes=scenes,
            evidence_bank=evidence_bank,
            aliases=aliases,
            character_id=character_id,
            language_key=language_key,
            config=config,
            counter=counter,
            prompt_registry=prompt_registry,
            evidence_mode="reviewed_gold_scene_index",
        )
    return result


def materialize_task1_development_cluster_judge(
    *,
    trajectory: dict[str, Any],
    predictions_by_instance: dict[str, dict[str, Any]],
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
    prompt_registry: YamlPromptRegistry = PROMPTS,
) -> dict[str, Any]:
    ordered = []
    refs = []
    for rubric in trajectory["checkpoint_rubrics"]:
        instance_id = rubric["instance_id"]
        localized = localize_task1_prediction(
            predictions_by_instance[instance_id]["prediction"]
        )
        developments = [
            {
                "prediction_ref": f"{instance_id}|{row['local_id']}",
                "claim": row["claim"],
                "evidence_scene_orders": row["evidence_scene_orders"],
            }
            for row in localized
            if row["prediction_type"] == "development"
        ]
        refs.extend(row["prediction_ref"] for row in developments)
        ordered.append(
            {
                "instance_id": instance_id,
                "checkpoint_id": rubric["checkpoint_id"],
                "checkpoint_scene_order": rubric["checkpoint"]["current_scene_order"],
                "developments": developments,
            }
        )
    prompt_path = f"{_language_key(language)}/evaluation/task1_development_cluster_judge"
    system, user = prompt_registry.render(
        prompt_path,
        focal_character=trajectory["character"],
        ordered_developments=ordered,
        allowed_development_refs=refs,
    )
    result = _budgeted(
        system=system,
        user=user,
        prompt_path=prompt_path,
        budget_name="task1_judge",
        counter=counter,
        config=config,
    ) | {"development_prediction_refs": refs}
    return result


def materialize_task1_adjacent_judge(
    *,
    trajectory: dict[str, Any],
    earlier_rubric: dict[str, Any],
    later_rubric: dict[str, Any],
    predictions_by_instance: dict[str, dict[str, Any]],
    scenes: dict[int, Any],
    evidence_bank: dict[str, Any],
    aliases: list[str],
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
    prompt_registry: YamlPromptRegistry = PROMPTS,
) -> dict[str, Any]:
    earlier_id = earlier_rubric["instance_id"]
    later_id = later_rubric["instance_id"]
    earlier_localized = localize_task1_prediction(
        predictions_by_instance[earlier_id]["prediction"]
    )
    later_localized = localize_task1_prediction(
        predictions_by_instance[later_id]["prediction"]
    )

    def compact(rows: list[dict[str, Any]]) -> list[list[Any]]:
        return [
            [
                row["local_id"],
                "s" if row["prediction_type"] == "current_state" else "d",
                row["evidence_scene_orders"],
                row["claim"],
            ]
            for row in rows
        ]

    earlier = {
        "instance_id": earlier_id,
        "checkpoint_id": earlier_rubric["checkpoint_id"],
        "checkpoint_scene_order": earlier_rubric["checkpoint"]["current_scene_order"],
        "prediction": compact(earlier_localized),
    }
    later = {
        "instance_id": later_id,
        "checkpoint_id": later_rubric["checkpoint_id"],
        "checkpoint_scene_order": later_rubric["checkpoint"]["current_scene_order"],
        "prediction": compact(later_localized),
    }
    earlier_order = int(earlier_rubric["checkpoint"]["current_scene_order"])
    later_order = int(later_rubric["checkpoint"]["current_scene_order"])
    cited_scene_orders = {
        int(scene_order)
        for row in later_localized
        if row["prediction_type"] == "development"
        for scene_order in row["evidence_scene_orders"]
        if earlier_order < int(scene_order) <= later_order
    }
    evidence = task1_evidence_excerpts(
        cited_scene_orders,
        character_id=trajectory["character_id"],
        aliases=aliases,
        evidence_bank=evidence_bank,
        scenes=scenes,
    )
    prompt_path = f"{_language_key(language)}/evaluation/task1_adjacent_judge"
    system, user = prompt_registry.render(
        prompt_path,
        focal_character=trajectory["character"],
        earlier_prediction=earlier,
        later_prediction=later,
        interval_evidence=evidence,
    )
    fallback_evidence = [
        {
            "scene_order": scene_order,
            "selection_policy": "checkpoint_scene_index",
            "verbatim_excerpt_withheld": True,
        }
        for scene_order in sorted(cited_scene_orders)
    ]
    fallback_system, fallback_user = prompt_registry.render(
        prompt_path,
        focal_character=trajectory["character"],
        earlier_prediction=earlier,
        later_prediction=later,
        interval_evidence=fallback_evidence,
    )
    fallback = _budgeted(
        system=fallback_system,
        user=fallback_user,
        prompt_path=prompt_path,
        budget_name="task1_judge",
        counter=counter,
        config=config,
    ) | {
        "earlier_instance_id": earlier_id,
        "later_instance_id": later_id,
        "interval_evidence": fallback_evidence,
        "evidence_mode": "checkpoint_scene_index",
    }
    try:
        result = _budgeted(
            system=system,
            user=user,
            prompt_path=prompt_path,
            budget_name="task1_judge",
            counter=counter,
            config=config,
        ) | {
            "earlier_instance_id": earlier_id,
            "later_instance_id": later_id,
            "interval_evidence": evidence,
            "evidence_mode": "verbatim_interval_evidence",
        }
    except ValueError as exc:
        if "exceeds budget" not in str(exc):
            raise
        return {
            **fallback,
            "content_filter_fallback": dict(fallback),
            "budget_fallback": {
                "trigger": "verbatim_interval_prompt_exceeded_context_budget",
                "original_error": str(exc),
            },
        }
    result["content_filter_fallback"] = fallback
    return result


def _budgeted(
    *,
    system: str,
    user: str,
    prompt_path: str,
    budget_name: str,
    counter: Any,
    config: BenchmarkRuntimeConfig,
) -> dict[str, Any]:
    prompt_tokens = counter.count(system) + counter.count(user)
    accounted = prompt_tokens + config.reserved_chat_template_tokens
    maximum = config.call_budgets[budget_name].max_input_tokens
    if accounted > maximum:
        raise ValueError(f"Task 1 prompt exceeds budget: {accounted}>{maximum}")
    return {
        "system_prompt": system,
        "user_prompt": user,
        "prompt_path": prompt_path,
        "prompt_tokens": prompt_tokens,
        "accounted_input_tokens": accounted,
        "max_input_tokens": maximum,
        "prompt_sha256": sha256_json({"system": system, "user": user}),
    }


def _language_key(language: str) -> str:
    return "zh" if str(language).casefold() in {"zh", "chinese"} else "en"
