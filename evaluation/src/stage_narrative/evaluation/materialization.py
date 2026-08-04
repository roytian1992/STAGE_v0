from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ..io import sha256_json
from ..prompt_loader import PROMPTS
from ..temporal.benchmark_protocol import BenchmarkRuntimeConfig


def localize_task1_prediction(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field, prediction_type in (
        ("current_state", "current_state"),
        ("developments_since_previous_checkpoint", "development"),
        ("unresolved_threads", "unresolved_thread"),
    ):
        values = payload.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"Task 1 prediction field must be an array: {field}")
        for item in values:
            rows.append(
                {
                    "local_id": f"P{len(rows) + 1}",
                    "prediction_type": prediction_type,
                    "claim": str(item["claim"]),
                    "evidence_scene_orders": sorted(
                        {int(value) for value in item["evidence_scene_orders"]}
                    ),
                }
            )
    return rows


def task1_evidence_excerpts(
    scene_orders: set[int],
    *,
    character_id: str,
    aliases: list[str],
    evidence_bank: dict[str, Any],
    scenes: dict[int, Any],
) -> list[dict[str, Any]]:
    selected_by_scene: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    normalized_aliases = {
        str(alias).strip().casefold() for alias in aliases if str(alias).strip()
    }
    for item in evidence_bank.get("evidence_units", []):
        scene_order = int(item.get("scene_order", 0))
        if scene_order not in scene_orders:
            continue
        speaker = item.get("speaker_character_id", "")
        direct = (
            speaker == character_id
            or character_id in item.get("addressee_character_ids", [])
            or character_id in item.get("direct_observer_character_ids", [])
            or (not speaker and character_id in item.get("participant_character_ids", []))
        )
        scene = scenes.get(scene_order)
        start = int(item["char_start"])
        end = int(item["char_end"])
        excerpt = scene.content[start:end].casefold() if scene is not None else ""
        alias_mention = any(alias in excerpt for alias in normalized_aliases)
        if direct or alias_mention:
            selected_by_scene[scene_order].append(
                (start, end, "direct" if direct else "alias_mention")
            )
    output = []
    for scene_order in sorted(scene_orders):
        intervals = sorted(set(selected_by_scene.get(scene_order, [])))
        merged: list[list[int]] = []
        policies = set()
        for start, end, policy in intervals:
            policies.add(policy)
            if not merged or start > merged[-1][1] + 2:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        if merged and scene_order in scenes:
            content = scenes[scene_order].content
            output.append(
                {
                    "scene_order": scene_order,
                    "selection_policy": "+".join(sorted(policies)),
                    "excerpts": [content[start:end] for start, end in merged],
                }
            )
        elif scene_order in scenes:
            output.append(
                {
                    "scene_order": scene_order,
                    "selection_policy": "unavailable_at_evidence_unit_granularity",
                    "evidence_unavailable": True,
                }
            )
    return output


def materialize_task1_claim_judge(
    *,
    gold: dict[str, Any],
    prediction: dict[str, Any],
    scenes: dict[int, Any],
    evidence_bank: dict[str, Any],
    aliases: list[str],
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> dict[str, Any]:
    localized = localize_task1_prediction(prediction["prediction"])
    cited_orders = {
        order for item in localized for order in item["evidence_scene_orders"]
    }
    for field in ("current_state_claims", "development_claims", "invariant_claims"):
        for item in gold["rubric"][field]:
            cited_orders.update(item["supporting_scene_orders"])
    evidence = task1_evidence_excerpts(
        cited_orders,
        character_id=gold["character_id"],
        aliases=aliases,
        evidence_bank=evidence_bank,
        scenes=scenes,
    )
    system, user = PROMPTS.render(
        config.prompt_path("task1_judge", language),
        checkpoint=_compact(gold.get("checkpoint", {})),
        prediction=_compact_task1_prediction(localized),
        gold_rubric=_compact_task1_rubric(gold["rubric"]),
        evidence_scenes=_compact_task1_evidence(evidence),
        allowed_gold_pair_ids=sorted(
            row["local_id"]
            for field in ("current_state_claims", "development_claims", "invariant_claims")
            for row in gold["rubric"][field]
        ),
        allowed_prediction_ids=sorted(row["local_id"] for row in localized),
    )
    return _prompt_record(
        system, user, budget_name="task1_judge", config=config, counter=counter,
        extra={"localized_prediction": localized, "evidence": evidence},
    )


def materialize_task1_sequence_judge(
    *,
    character: str,
    character_id: str,
    aliases: list[str],
    earlier: dict[str, Any],
    later: dict[str, Any],
    later_gold: dict[str, Any],
    scenes: dict[int, Any],
    evidence_bank: dict[str, Any],
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> dict[str, Any]:
    earlier_order = int(earlier["checkpoint"]["current_scene_order"])
    later_order = int(later["checkpoint"]["current_scene_order"])
    cited_orders = {
        int(order)
        for row in later["prediction"].get("developments_since_previous_checkpoint", [])
        for order in row.get("evidence_scene_orders", [])
        if earlier_order < int(order) <= later_order
    }
    for row in later_gold["rubric"]["development_claims"]:
        cited_orders.update(
            int(order)
            for order in row["supporting_scene_orders"]
            if earlier_order < int(order) <= later_order
        )
    evidence = task1_evidence_excerpts(
        cited_orders,
        character_id=character_id,
        aliases=aliases,
        evidence_bank=evidence_bank,
        scenes=scenes,
    )
    system, user = PROMPTS.render(
        f"{'zh' if str(language).casefold() in {'zh', 'chinese'} else 'en'}/evaluation_v1/task1_sequence_judge",
        focal_character=character,
        earlier_checkpoint=earlier["checkpoint"],
        later_checkpoint=later["checkpoint"],
        earlier_prediction=earlier["prediction"],
        later_prediction=later["prediction"],
        interval_evidence=evidence,
    )
    return _prompt_record(
        system, user, budget_name="task1_judge", config=config, counter=counter,
        extra={
            "evidence": evidence,
            "evidence_policy": "interval-only later predicted-development citations plus later gold-development supporting scenes",
        },
    )


def localize_actor_context(actor_input: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    labels: set[str] = set()
    counter = 0

    def labeled(kind: str, value: Any) -> list[Any]:
        nonlocal counter
        counter += 1
        local_id = f"C{counter}"
        labels.add(local_id)
        return [local_id, kind, value]

    role = actor_input["role_context"]
    records = [labeled("identity", role["identity"])]
    for field in (
        "persona_evidence",
        "dialogue_exemplars",
        "visible_memories",
        "relation_evidence",
    ):
        records.extend(labeled(field, item) for item in role[field])
    return {
        "character": actor_input["character"],
        "role_context_records": records,
        "interaction_context": actor_input["interaction_context"],
    }, labels


def materialize_task3_response_judge(
    *,
    gold: dict[str, Any],
    prediction: dict[str, Any],
    context_pack: dict[str, Any],
    instance: dict[str, Any],
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> dict[str, Any]:
    actor_input = context_pack["actor_input"]
    judge_context, allowed_labels = localize_actor_context(actor_input)
    allowed_labels.update(
        row["local_id"] for row in gold["rubric"]["salient_future_negatives"]
    )
    checkpoint = instance["model_input"]["checkpoint_anchor"]
    system, user = PROMPTS.render(
        config.prompt_path("task3_response_judge", language),
        character=actor_input["character"],
        checkpoint=checkpoint,
        current_user_turn=actor_input["current_user_turn"],
        actor_context_pack=judge_context,
        gold_rubric=gold["rubric"],
        actor_response=prediction["response"],
        allowed_evidence_ids=sorted(allowed_labels),
    )
    result = _prompt_record(
        system, user, budget_name="task3_response_judge", config=config, counter=counter,
        extra={
            "allowed_labels": allowed_labels,
            "checkpoint": checkpoint,
            "actor_response": prediction["response"],
            "character": actor_input["character"],
            "context_mode": "verbatim_actor_context",
        },
    )
    fallback_context = {
        "character": judge_context["character"],
        "role_context_records": [
            [row[0], row[1], "withheld_due_to_provider_content_filter"]
            for row in judge_context["role_context_records"]
        ],
        "interaction_context": {"verbatim_context_withheld": True},
    }
    fallback_system, fallback_user = PROMPTS.render(
        config.prompt_path("task3_response_judge", language),
        character=actor_input["character"],
        checkpoint=checkpoint,
        current_user_turn=actor_input["current_user_turn"],
        actor_context_pack=fallback_context,
        gold_rubric=gold["rubric"],
        actor_response=prediction["response"],
        allowed_evidence_ids=sorted(allowed_labels),
    )
    result["content_filter_fallback"] = _prompt_record(
        fallback_system,
        fallback_user,
        budget_name="task3_response_judge",
        config=config,
        counter=counter,
        extra={
            "allowed_labels": allowed_labels,
            "checkpoint": checkpoint,
            "actor_response": prediction["response"],
            "character": actor_input["character"],
            "context_mode": "reviewed_gold_without_verbatim_actor_context",
        },
    )
    return result


def materialize_task3_pair_judge(
    *,
    annotation: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    instances: dict[str, dict[str, Any]],
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> dict[str, Any]:
    ordered = annotation["ordered_instance_ids"]
    paired_responses = [
        {
            "instance_id": instance_id,
            "checkpoint": instances[instance_id]["model_input"]["checkpoint_anchor"],
            "current_user_turn": instances[instance_id]["model_input"]["current_user_turn"],
            "response": predictions[instance_id]["response"],
        }
        for instance_id in ordered
    ]
    paired_rubrics = [
        {"instance_id": instance_id, "rubric": gold[instance_id]["rubric"]}
        for instance_id in ordered
    ]
    pair_evidence = [
        {
            "local_label": f"T{index}",
            "instance_id": instance_id,
            "checkpoint": instances[instance_id]["model_input"]["checkpoint_anchor"],
            "interaction_context": instances[instance_id]["model_input"]["interaction_context"],
        }
        for index, instance_id in enumerate(ordered, 1)
    ]
    system, user = PROMPTS.render(
        config.prompt_path("task3_pair_judge", language),
        pair_type=annotation["pair_type"],
        expected_direction=annotation["expected_direction"],
        paired_responses=paired_responses,
        paired_rubrics=paired_rubrics,
        pair_evidence=pair_evidence,
    )
    return _prompt_record(
        system, user, budget_name="task3_pair_judge", config=config, counter=counter,
        extra={
            "instance_ids": ordered,
            "pair_evidence": pair_evidence,
            "responses_by_label": {
                f"T{index}": row["response"]
                for index, row in enumerate(paired_responses, 1)
            },
        },
    )


def _prompt_record(
    system: str,
    user: str,
    *,
    budget_name: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
    extra: dict[str, Any],
) -> dict[str, Any]:
    prompt_tokens = (
        counter.count(system) + counter.count(user) + config.reserved_chat_template_tokens
    )
    maximum = config.call_budgets[budget_name].max_input_tokens
    if prompt_tokens > maximum:
        raise ValueError(f"{budget_name} prompt exceeds budget: {prompt_tokens}>{maximum}")
    return {
        "system_prompt": system,
        "user_prompt": user,
        "prompt_tokens": prompt_tokens,
        "max_input_tokens": maximum,
        "prompt_sha256": sha256_json({"system": system, "user": user}),
        **extra,
    }


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _compact_task1_prediction(rows: list[dict[str, Any]]) -> str:
    code = {"current_state": "s", "development": "d", "unresolved_thread": "u"}
    return _compact(
        [
            [row["local_id"], code[row["prediction_type"]], row["evidence_scene_orders"], row["claim"]]
            for row in rows
        ]
    )


def _compact_task1_rubric(rubric: dict[str, Any]) -> str:
    rows = []
    for field, code in (
        ("current_state_claims", "s"),
        ("development_claims", "d"),
        ("invariant_claims", "i"),
    ):
        rows.extend(
            [row["local_id"], code, row["supporting_scene_orders"], row["claim"]]
            for row in rubric[field]
        )
    rows.extend(
        [row["local_id"], "f", row.get("source_future_local_ids", []), row["claim"]]
        for row in rubric["salient_future_negatives"]
    )
    return _compact(rows)


def _compact_task1_evidence(rows: list[dict[str, Any]]) -> str:
    return _compact(
        [
            [row["scene_order"], None if row.get("evidence_unavailable") else row["excerpts"]]
            for row in rows
        ]
    )
