from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any

from ..models import clean_text
from .visibility import boundary_key


def build_memory_migration_map(
    *, legacy_role_assets: dict[str, Any], role_assets: dict[str, Any]
) -> dict[str, Any]:
    _one_movie_id(legacy_role_assets, role_assets)
    entries: list[dict[str, Any]] = []
    mapped_new_ids: set[str] = set()
    for legacy_role in legacy_role_assets.get("roles", []):
        role = _role_for_name(legacy_role.get("character_name"), role_assets["roles"])
        if role is None:
            raise ValueError(f"Legacy role has no current match: {legacy_role.get('character_name')}")
        for legacy_memory in legacy_role.get("memories", []):
            scene_order = int(legacy_memory.get("scene_order") or 0)
            candidates = [
                memory
                for memory in role["memories"]
                if int(memory["fact_origin"]["scene_order"]) == scene_order
            ]
            scored = sorted(
                ((_memory_similarity(legacy_memory, item), item) for item in candidates),
                key=lambda item: (-item[0], item[1]["memory_id"]),
            )
            best_score = scored[0][0] if scored else 0.0
            selected = [
                item
                for score, item in scored
                if score >= max(0.24, best_score * 0.72)
            ][:4]
            if best_score < 0.24:
                selected = []
            new_ids = [item["memory_id"] for item in selected]
            mapped_new_ids.update(new_ids)
            if not new_ids:
                classification = "dropped_unsupported"
            elif len(new_ids) > 1:
                classification = "split_or_merged"
            elif best_score >= 0.66:
                classification = "retained_with_new_provenance"
            else:
                classification = "rewritten"
            entries.append(
                {
                    "character_id": role["character_id"],
                    "legacy_character": legacy_role.get("character_name"),
                    "legacy_memory_id": legacy_memory["memory_id"],
                    "legacy_scene_order": scene_order,
                    "classification": classification,
                    "matched_memory_ids": new_ids,
                    "similarity": round(best_score, 6),
                    "requires_review": not new_ids or best_score < 0.45,
                }
            )
    for role in role_assets["roles"]:
        for memory in role["memories"]:
            if memory["memory_id"] not in mapped_new_ids:
                entries.append(
                    {
                        "character_id": role["character_id"],
                        "legacy_character": None,
                        "legacy_memory_id": None,
                        "legacy_scene_order": None,
                        "classification": "new_from_current_kg",
                        "matched_memory_ids": [memory["memory_id"]],
                        "similarity": None,
                        "requires_review": False,
                    }
                )
    counts: dict[str, int] = {}
    for entry in entries:
        label = entry["classification"]
        counts[label] = counts.get(label, 0) + 1
    return {
        "schema_version": "stage_task3_memory_migration_map",
        "movie_id": role_assets["movie_id"],
        "status": "deterministic_candidates_require_agent_review",
        "entry_count": len(entries),
        "classification_counts": dict(sorted(counts.items())),
        "entries": entries,
    }


def build_single_turn_release(
    *,
    released_single: dict[str, Any],
    temporal_single: dict[str, Any],
    gold_rubrics: dict[str, Any],
    pair_groups: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    role_assets: dict[str, Any],
) -> dict[str, Any]:
    movie_id = _one_movie_id(
        released_single,
        temporal_single,
        gold_rubrics,
        pair_groups,
        checkpoint_manifest,
        role_assets,
    )
    temporal_by_id = _unique_by(temporal_single["instances"], "instance_id")
    rubric_by_id = _unique_by(gold_rubrics["rubrics"], "instance_id")
    checkpoint_by_id = _unique_by(checkpoint_manifest["checkpoints"], "checkpoint_id")
    roles = _unique_by(role_assets["roles"], "character_id")
    pair_by_instance: dict[str, dict[str, Any]] = {}
    for group in pair_groups["pair_groups"]:
        for instance_id in group["instance_ids"]:
            if instance_id in pair_by_instance:
                raise ValueError(f"Instance belongs to multiple pair groups: {instance_id}")
            pair_by_instance[instance_id] = group
    memory_by_fact = {
        fact_id: (role["role_id"], memory["memory_id"])
        for role in role_assets["roles"]
        for memory in role["memories"]
        for fact_id in memory["source_fact_ids"]
    }
    released_ids = {item["instance_id"] for item in released_single["instances"]}
    if not released_ids.issubset(temporal_by_id) or released_ids != set(rubric_by_id):
        raise ValueError("Released single-turn IDs do not match temporal refs and gold rubrics")
    instances = []
    for released in released_single["instances"]:
        instance_id = released["instance_id"]
        temporal = temporal_by_id[instance_id]
        rubric = rubric_by_id[instance_id]
        role = roles.get(temporal["character_id"])
        if role is None:
            raise ValueError(f"Single-turn character lacks a role: {instance_id}")
        evaluator = deepcopy(temporal["evaluator_reference"])
        checkpoint_source = checkpoint_by_id.get(evaluator["checkpoint_id"])
        if checkpoint_source is None:
            raise ValueError(f"Unknown checkpoint: {evaluator['checkpoint_id']}")
        old_boundary = released["checkpoint_boundary"]
        if int(old_boundary["scene_order"]) != int(checkpoint_source["scene_order"]):
            raise ValueError(f"Checkpoint scene mismatch: {instance_id}")
        checkpoint = {
            "checkpoint_id": evaluator["checkpoint_id"],
            "scene_id": checkpoint_source["scene_id"],
            "scene_order": int(old_boundary["scene_order"]),
            "char_end": int(old_boundary["char_end"]),
        }
        source_memory_ids = []
        for fact_id in evaluator.get("required_memory_fact_ids", []):
            mapped = memory_by_fact.get(fact_id)
            if mapped is None or mapped[0] != role["role_id"]:
                raise ValueError(f"Required single-turn fact lacks role memory: {fact_id}")
            source_memory_ids.append(mapped[1])
        pair = pair_by_instance.get(instance_id)
        model_input = released["model_input"]
        instances.append(
            {
                "instance_id": instance_id,
                "movie_id": movie_id,
                "language": released["language"],
                "interaction_format": "single_turn",
                "role_ref": role["role_id"],
                "checkpoint": checkpoint,
                "interaction_context": model_input["interaction_context"],
                "dialogue_history": deepcopy(model_input.get("dialogue_history", [])),
                "current_user_turn": model_input["current_user_turn"],
                "evaluator_reference": {
                    "source_memory_ids": sorted(set(source_memory_ids)),
                    "pair_group_id": pair.get("pair_group_id") if pair else None,
                    "pair_type": pair.get("pair_type") if pair else None,
                    "temporal_reference": evaluator,
                    "rubric": deepcopy(rubric["rubric"]),
                    "gold_review_status": rubric["review_status"],
                    "gold_review_note": rubric.get("review_note"),
                },
            }
        )
    return {
        "schema_version": "stage_task3_single_turn",
        "task": "task_3_character_role_play",
        "movie_id": movie_id,
        "language": instances[0]["language"] if instances else None,
        "instance_count": len(instances),
        "instances": instances,
    }


def build_multi_turn_release(
    *,
    legacy_multi: dict[str, Any],
    role_assets: dict[str, Any],
    migration_map: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    movie_id = _one_movie_id(legacy_multi, role_assets, migration_map)
    memory_by_id = {
        memory["memory_id"]: memory
        for role in role_assets["roles"]
        for memory in role["memories"]
    }
    legacy_map = {
        (item["character_id"], item["legacy_memory_id"]): item["matched_memory_ids"]
        for item in migration_map["entries"]
        if item.get("legacy_memory_id")
    }
    override_by_turn = {
        (item["episode_id"], item["question_id"]): item
        for item in (overrides or {}).get("overrides", [])
    }
    episodes = []
    for source_episode in legacy_multi["episodes"]:
        role = _role_for_name(source_episode["character"], role_assets["roles"])
        if role is None:
            raise ValueError(f"Multi-turn character lacks a role: {source_episode['character']}")
        role_memory_ids = {item["memory_id"] for item in role["memories"]}
        episode_memory_ids: set[str] = set()
        turns = []
        for expected_index, source_turn in enumerate(source_episode["turns"], start=1):
            if int(source_turn.get("turn_index", -1)) != expected_index:
                raise ValueError(f"Non-contiguous multi-turn indices: {source_episode['episode_id']}")
            reference = deepcopy(source_turn["reference"])
            legacy_ids = reference.pop("source_memory_ids", [])
            override = override_by_turn.get(
                (source_episode["episode_id"], source_turn["question_id"])
            )
            if override:
                mapped_ids = list(override["replacement_source_memory_ids"])
                reference = deepcopy(override["replacement_reference"])
                current_user_turn = override["current_user_turn"]
            else:
                mapped_ids = []
                for legacy_id in legacy_ids:
                    mapped = legacy_map.get((role["character_id"], legacy_id), [])
                    if not mapped:
                        raise ValueError(
                            f"Referenced legacy memory is unmapped: {role['canonical_name']}/{legacy_id}"
                        )
                    mapped_ids.extend(mapped)
                current_user_turn = source_turn["input"]["current_user_turn"]
            mapped_ids = sorted(set(mapped_ids))
            if not mapped_ids or any(item not in role_memory_ids for item in mapped_ids):
                raise ValueError(f"Multi-turn source memories are invalid: {mapped_ids}")
            episode_memory_ids.update(mapped_ids)
            turns.append(
                {
                    "turn_index": expected_index,
                    "question_id": source_turn["question_id"],
                    "dialogue_history_template": deepcopy(
                        source_turn["input"].get("dialogue_history_template", [])
                    ),
                    "current_user_turn": current_user_turn,
                    "evaluator_reference": {
                        "source_memory_ids": mapped_ids,
                        "reference": reference,
                        "migration_provenance": {
                            "legacy_source_memory_ids": legacy_ids,
                            "targeted_override": override is not None,
                            "runtime_dependency": "none",
                        },
                    },
                }
            )
        effective_questions = {item["turn_index"]: item["current_user_turn"] for item in turns}
        for turn in turns:
            for history_item in turn["dialogue_history_template"]:
                if history_item.get("speaker") == "user":
                    history_item["text"] = effective_questions[int(history_item["source_turn_index"])]
        latest = max(
            (memory_by_id[item] for item in episode_memory_ids),
            key=lambda item: boundary_key(item["available_from"]),
        )
        episodes.append(
            {
                "episode_id": source_episode["episode_id"],
                "episode_theme": source_episode.get("episode_theme"),
                "role_ref": role["role_id"],
                "checkpoint": {
                    "scene_id": latest["fact_origin"]["scene_id"],
                    **deepcopy(latest["available_from"]),
                    "derivation": "latest_mapped_source_memory_available_from",
                },
                "turns": turns,
            }
        )
    return {
        "schema_version": "stage_task3_multi_turn",
        "task": "task_3_character_role_play",
        "movie_id": movie_id,
        "language": legacy_multi.get("language"),
        "interaction_format": "multi_turn",
        "construction_policy": "legacy_questions_with_current_grounded_role_memory",
        "episode_count": len(episodes),
        "turn_count": sum(len(item["turns"]) for item in episodes),
        "episodes": episodes,
    }


def _memory_similarity(legacy: dict[str, Any], current: dict[str, Any]) -> float:
    left = [legacy.get("memory_text", ""), *legacy.get("grounded_facts", [])]
    right = [current.get("memory_text", ""), *current.get("grounded_facts", [])]
    return max((_text_similarity(a, b) for a in left for b in right), default=0.0)


def _text_similarity(left: Any, right: Any) -> float:
    a, b = _normalize_text(left), _normalize_text(right)
    if not a or not b:
        return 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    a2 = {a[index : index + 2] for index in range(len(a) - 1)}
    b2 = {b[index : index + 2] for index in range(len(b) - 1)}
    union = a2 | b2
    return max(sequence, len(a2 & b2) / len(union) if union else 0.0)


def _unique_by(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    output = {}
    for item in items:
        value = clean_text(item.get(key))
        if not value or value in output:
            raise ValueError(f"Missing or duplicate {key}: {value}")
        output[value] = item
    return output


def _one_movie_id(*payloads: dict[str, Any]) -> str:
    movie_ids = {clean_text(item.get("movie_id")) for item in payloads}
    movie_ids.discard("")
    if len(movie_ids) != 1:
        raise ValueError(f"Task 3 inputs disagree on movie_id: {sorted(movie_ids)}")
    return next(iter(movie_ids))


def _role_for_name(name: Any, roles: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized = _normalize_text(name)
    matches = []
    for role in roles:
        names = {
            _normalize_text(role.get("canonical_name")),
            *(_normalize_text(item) for item in role.get("aliases", [])),
            *(
                _normalize_text(item.get("name"))
                for item in role.get("identity_phases", [])
                if isinstance(item, dict)
            ),
        }
        if normalized and normalized in names:
            matches.append(role)
    if len(matches) > 1:
        raise ValueError(f"Character matches multiple role assets: {name}")
    return matches[0] if matches else None


def _normalize_text(value: Any) -> str:
    return "".join(character for character in clean_text(value).casefold() if character.isalnum())
