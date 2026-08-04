from __future__ import annotations

from typing import Any

from ..models import clean_text
from .memory_visibility import boundary_key, materialize_role_at_boundary


def validate_role_assets(payload: dict[str, Any]) -> dict[str, int]:
    if payload.get("schema_version") != "stage_task3_role_assets":
        raise ValueError("Unsupported Task 3 role asset schema")
    roles = payload.get("roles")
    if not isinstance(roles, list) or int(payload.get("role_count", -1)) != len(roles):
        raise ValueError("role_count must equal the roles array length")
    role_ids: set[str] = set()
    memory_ids: set[str] = set()
    total_source_fact_count = 0
    for role in roles:
        character_id = clean_text(role.get("character_id"))
        if not character_id or character_id in role_ids:
            raise ValueError(f"Missing or duplicate character_id: {character_id}")
        role_ids.add(character_id)
        role_source_fact_ids: set[str] = set()
        previous = (-1, -1)
        for memory in role.get("memories", []):
            memory_id = clean_text(memory.get("memory_id"))
            if not memory_id or memory_id in memory_ids:
                raise ValueError(f"Missing or duplicate memory_id: {memory_id}")
            memory_ids.add(memory_id)
            if not clean_text(memory.get("memory_text")):
                raise ValueError(f"Memory text is empty: {memory_id}")
            current = boundary_key(memory["available_from"])
            if current < previous:
                raise ValueError(f"Memories are not sorted at {memory_id}")
            previous = current
            fact_ids = memory.get("source_fact_ids")
            evidence_ids = memory.get("source_evidence_ids")
            if not isinstance(fact_ids, list) or not fact_ids:
                raise ValueError(f"Memory lacks source facts: {memory_id}")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError(f"Memory lacks source evidence: {memory_id}")
            overlap = role_source_fact_ids & set(fact_ids)
            if overlap:
                raise ValueError(
                    f"Known facts were duplicated across memories: {sorted(overlap)}"
                )
            role_source_fact_ids.update(fact_ids)
            accesses = memory.get("knowledge_access")
            if not isinstance(accesses, list) or {
                item["fact_id"] for item in accesses
            } != set(fact_ids):
                raise ValueError(
                    f"knowledge_access does not cover memory facts: {memory_id}"
                )
            for access in accesses:
                if access.get("access_type") not in {
                    "witnessed",
                    "involved",
                    "told",
                    "inferred",
                }:
                    raise ValueError(f"Illegal knowledge access in {memory_id}")
                if int(access["acquired_at_scene"]) > current[0]:
                    raise ValueError(
                        f"Memory is visible before a source fact is acquired: {memory_id}"
                    )
        total_source_fact_count += len(role_source_fact_ids)
        _assert_actor_view_hides_provenance(role)
    return {
        "roles": len(role_ids),
        "memories": len(memory_ids),
        "source_facts": total_source_fact_count,
    }


def validate_task3_release(
    *,
    role_assets: dict[str, Any],
    single_turn: dict[str, Any],
    multi_turn: dict[str, Any],
) -> dict[str, int]:
    role_counts = validate_role_assets(role_assets)
    movie_ids = {
        clean_text(item.get("movie_id"))
        for item in (role_assets, single_turn, multi_turn)
    }
    if len(movie_ids) != 1:
        raise ValueError(f"Task 3 release movie IDs disagree: {sorted(movie_ids)}")
    role_by_id = {item["character_id"]: item for item in role_assets["roles"]}
    instances = single_turn.get("instances")
    if single_turn.get("schema_version") != "stage_task3_single_turn" or not isinstance(
        instances, list
    ):
        raise ValueError("Unsupported Task 3 single-turn schema")
    if int(single_turn.get("instance_count", -1)) != len(instances):
        raise ValueError("Task 3 single-turn instance_count mismatch")
    instance_ids: set[str] = set()
    for item in instances:
        instance_id = clean_text(item.get("instance_id"))
        if not instance_id or instance_id in instance_ids:
            raise ValueError(f"Missing or duplicate single-turn instance_id: {instance_id}")
        instance_ids.add(instance_id)
        role = _validate_task_role_ref(item, role_by_id)
        materialize_role_at_boundary(role, item["checkpoint_boundary"])
        if not isinstance(item.get("evaluator_reference"), dict):
            raise ValueError(f"Single-turn evaluator reference is missing: {instance_id}")

    episodes = multi_turn.get("episodes")
    if multi_turn.get("schema_version") != "stage_task3_multi_turn" or not isinstance(
        episodes, list
    ):
        raise ValueError("Unsupported Task 3 multi-turn schema")
    if int(multi_turn.get("episode_count", -1)) != len(episodes):
        raise ValueError("Task 3 multi-turn episode_count mismatch")
    turn_count = 0
    for episode in episodes:
        _validate_task_role_ref(episode, role_by_id)
        policy = episode.get("memory_context_policy")
        if policy != {"mode": "all_role_memories", "retrieval": "none"}:
            raise ValueError(f"Unexpected multi-turn memory policy: {policy}")
        for expected_index, turn in enumerate(episode.get("turns", []), start=1):
            turn_count += 1
            if int(turn.get("turn_index", -1)) != expected_index:
                raise ValueError(
                    f"Multi-turn indices are not contiguous: {episode['episode_id']}"
                )
            if not isinstance(turn.get("evaluator_reference"), dict):
                raise ValueError(
                    f"Multi-turn evaluator reference is missing: {episode['episode_id']}"
                )
    if int(multi_turn.get("turn_count", -1)) != turn_count:
        raise ValueError("Task 3 multi-turn turn_count mismatch")
    return {
        **role_counts,
        "single_turn_instances": len(instance_ids),
        "multi_turn_episodes": len(episodes),
        "multi_turn_turns": turn_count,
    }


def validate_multi_turn_release(
    *, role_assets: dict[str, Any], multi_turn: dict[str, Any]
) -> dict[str, int]:
    """Validate the independently retained legacy multi-turn track."""
    role_counts = validate_role_assets(role_assets)
    if clean_text(role_assets.get("movie_id")) != clean_text(multi_turn.get("movie_id")):
        raise ValueError("Task 3 role assets and multi-turn movie IDs disagree")
    role_by_id = {item["character_id"]: item for item in role_assets["roles"]}
    episodes = multi_turn.get("episodes")
    if multi_turn.get("schema_version") != "stage_task3_multi_turn" or not isinstance(
        episodes, list
    ):
        raise ValueError("Unsupported Task 3 multi-turn schema")
    if int(multi_turn.get("episode_count", -1)) != len(episodes):
        raise ValueError("Task 3 multi-turn episode_count mismatch")
    turn_count = 0
    for episode in episodes:
        _validate_task_role_ref(episode, role_by_id)
        if episode.get("memory_context_policy") != {
            "mode": "all_role_memories",
            "retrieval": "none",
        }:
            raise ValueError(
                f"Unexpected multi-turn memory policy: {episode.get('episode_id')}"
            )
        for expected_index, turn in enumerate(episode.get("turns", []), start=1):
            turn_count += 1
            if int(turn.get("turn_index", -1)) != expected_index:
                raise ValueError(
                    f"Multi-turn indices are not contiguous: {episode['episode_id']}"
                )
            if not isinstance(turn.get("evaluator_reference"), dict):
                raise ValueError(
                    f"Multi-turn evaluator reference is missing: {episode['episode_id']}"
                )
    if int(multi_turn.get("turn_count", -1)) != turn_count:
        raise ValueError("Task 3 multi-turn turn_count mismatch")
    return {
        **role_counts,
        "multi_turn_episodes": len(episodes),
        "multi_turn_turns": turn_count,
    }


def _validate_task_role_ref(
    item: dict[str, Any], role_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    character_id = clean_text(item.get("character_id"))
    ref = item.get("role_asset_ref")
    if (
        character_id not in role_by_id
        or not isinstance(ref, dict)
        or ref.get("asset_file") != "task_3_role_assets.json"
        or ref.get("character_id") != character_id
    ):
        raise ValueError(f"Invalid Task 3 role asset reference: {item}")
    return role_by_id[character_id]


def _assert_actor_view_hides_provenance(role: dict[str, Any]) -> None:
    actor = materialize_role_at_boundary(
        role, {"scene_order": 10**9, "char_end": 10**9}
    )
    forbidden = {
        "source_fact_ids",
        "source_episode_ids",
        "source_evidence_ids",
        "knowledge_access",
        "available_from",
        "valid_from",
        "valid_until_scene",
        "evaluator_reference",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            overlap = forbidden & set(value)
            if overlap:
                raise ValueError(f"Actor role view leaks hidden fields: {sorted(overlap)}")
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(actor)
