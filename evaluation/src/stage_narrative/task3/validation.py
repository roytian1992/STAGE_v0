from __future__ import annotations

from typing import Any

from ..models import clean_text
from .visibility import boundary_key, materialize_role_at_boundary, phase_is_active


def validate_role_assets(payload: dict[str, Any]) -> dict[str, int]:
    if payload.get("schema_version") != "stage_task3_role_assets":
        raise ValueError("Unsupported Task 3 role asset schema")
    roles = payload.get("roles")
    if not isinstance(roles, list) or int(payload.get("role_count", -1)) != len(roles):
        raise ValueError("role_count must equal the roles array length")
    role_ids: set[str] = set()
    character_ids: set[str] = set()
    memory_ids: set[str] = set()
    total_source_fact_count = 0
    for role in roles:
        required_role_keys = {
            "role_id",
            "character_id",
            "canonical_name",
            "aliases",
            "identity_phases",
            "persona_phases",
            "relationship_phases",
            "memories",
        }
        if not isinstance(role, dict) or set(role) != required_role_keys:
            raise ValueError(
                f"Formal role keys must be exactly {sorted(required_role_keys)}"
            )
        role_id = clean_text(role.get("role_id"))
        character_id = clean_text(role.get("character_id"))
        if not role_id or role_id in role_ids:
            raise ValueError(f"Missing or duplicate role_id: {role_id}")
        if not character_id or character_id in character_ids:
            raise ValueError(f"Missing or duplicate character_id: {character_id}")
        if not clean_text(role.get("canonical_name")):
            raise ValueError(f"Role lacks canonical_name: {role_id}")
        role_ids.add(role_id)
        character_ids.add(character_id)
        identity_phase_ids = _validate_phases(
            role.get("identity_phases"),
            id_key="identity_phase_id",
            label=f"identity phases for {role_id}",
            require_evidence=False,
        )
        _validate_phases(
            role.get("persona_phases"),
            id_key="persona_phase_id",
            label=f"persona phases for {role_id}",
            require_evidence=True,
            identity_phase_ids=identity_phase_ids,
        )
        _validate_phases(
            role.get("relationship_phases"),
            id_key="relationship_phase_id",
            label=f"relationship phases for {role_id}",
            require_evidence=True,
            identity_phase_ids=identity_phase_ids,
            partition_key="target",
        )
        role_source_fact_ids: set[str] = set()
        previous = (-1, -1)
        for memory in role.get("memories", []):
            memory_id = clean_text(memory.get("memory_id"))
            if not memory_id or memory_id in memory_ids:
                raise ValueError(f"Missing or duplicate memory_id: {memory_id}")
            memory_ids.add(memory_id)
            if not clean_text(memory.get("memory_text")):
                raise ValueError(f"Memory text is empty: {memory_id}")
            required_memory_keys = {
                "memory_id",
                "memory_text",
                "importance",
                "fact_origin",
                "available_from",
                "access_type",
                "source_episode_ids",
                "source_fact_ids",
                "source_evidence_ids",
                "grounded_facts",
                "identity_phase",
                "tags",
                "knowledge_access",
            }
            if set(memory) != required_memory_keys:
                raise ValueError(
                    f"Formal memory keys must be exactly {sorted(required_memory_keys)}: "
                    f"{memory_id}"
                )
            fact_origin = memory.get("fact_origin")
            if (
                not isinstance(fact_origin, dict)
                or set(fact_origin) != {"scene_id", "scene_order"}
                or not clean_text(fact_origin.get("scene_id"))
                or int(fact_origin.get("scene_order", 0)) <= 0
            ):
                raise ValueError(f"Memory has invalid fact_origin: {memory_id}")
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
            grounded_facts = memory.get("grounded_facts")
            if (
                not isinstance(grounded_facts, list)
                or len(grounded_facts) != len(fact_ids)
                or not all(clean_text(item) for item in grounded_facts)
            ):
                raise ValueError(f"grounded_facts do not match source facts: {memory_id}")
            access_type = clean_text(memory.get("access_type"))
            if access_type not in {"witnessed", "involved", "told", "inferred"}:
                raise ValueError(f"Illegal top-level access_type in {memory_id}")
            identity_phase_id = clean_text(memory.get("identity_phase"))
            if identity_phase_id not in identity_phase_ids:
                raise ValueError(f"Memory references an unknown identity phase: {memory_id}")
            identity_phase = next(
                item
                for item in role["identity_phases"]
                if item["identity_phase_id"] == identity_phase_id
            )
            if not phase_is_active(identity_phase, memory["available_from"]):
                raise ValueError(f"Memory identity phase is inactive: {memory_id}")
            overlap = role_source_fact_ids & set(fact_ids)
            if overlap:
                raise ValueError(f"Known facts were duplicated across memories: {sorted(overlap)}")
            role_source_fact_ids.update(fact_ids)
            accesses = memory.get("knowledge_access")
            if not isinstance(accesses, list) or {item["fact_id"] for item in accesses} != set(fact_ids):
                raise ValueError(f"knowledge_access does not cover memory facts: {memory_id}")
            for access in accesses:
                if access.get("access_type") != access_type:
                    raise ValueError(f"Illegal knowledge access in {memory_id}")
                if int(access["acquired_at_scene"]) > current[0]:
                    raise ValueError(f"Memory is visible before a source fact is acquired: {memory_id}")
        total_source_fact_count += len(role_source_fact_ids)
        _assert_actor_view_hides_provenance(role)
    return {
        "roles": len(role_ids),
        "memories": len(memory_ids),
        "source_facts": total_source_fact_count,
    }


def _validate_phases(
    phases: Any,
    *,
    id_key: str,
    label: str,
    require_evidence: bool,
    identity_phase_ids: set[str] | None = None,
    partition_key: str | None = None,
) -> set[str]:
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"{label} must be a non-empty array")
    identifiers: set[str] = set()
    partitions: dict[str, list[dict[str, Any]]] = {}
    for phase in phases:
        identifier = clean_text(phase.get(id_key)) if isinstance(phase, dict) else ""
        if not identifier or identifier in identifiers:
            raise ValueError(f"Missing or duplicate {id_key} in {label}: {identifier}")
        identifiers.add(identifier)
        start = boundary_key(phase["valid_from"])
        end = (
            boundary_key(phase["valid_until"])
            if phase.get("valid_until") is not None
            else None
        )
        if end is not None and end <= start:
            raise ValueError(f"Non-positive phase interval in {label}: {identifier}")
        if require_evidence:
            evidence_ids = phase.get("source_evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError(f"Phase lacks source evidence in {label}: {identifier}")
        if identity_phase_ids is not None and clean_text(
            phase.get("identity_phase")
        ) not in identity_phase_ids:
            raise ValueError(f"Phase references unknown identity phase: {identifier}")
        partition = clean_text(phase.get(partition_key)) if partition_key else "__all__"
        partitions.setdefault(partition, []).append(phase)
    for partition, items in partitions.items():
        ordered = sorted(items, key=lambda item: boundary_key(item["valid_from"]))
        for left, right in zip(ordered, ordered[1:]):
            left_until = left.get("valid_until")
            if left_until is None or boundary_key(left_until) > boundary_key(
                right["valid_from"]
            ):
                raise ValueError(f"Overlapping {label} in partition {partition}")
    return identifiers


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
    role_by_id = {item["role_id"]: item for item in role_assets["roles"]}
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
        required_keys = {
            "instance_id",
            "movie_id",
            "language",
            "interaction_format",
            "role_ref",
            "checkpoint",
            "interaction_context",
            "dialogue_history",
            "current_user_turn",
            "evaluator_reference",
        }
        if set(item) != required_keys:
            raise ValueError(f"Unexpected formal single-turn fields: {instance_id}")
        checkpoint = item["checkpoint"]
        materialize_role_at_boundary(role, checkpoint)
        evaluator = item.get("evaluator_reference")
        if not isinstance(evaluator, dict):
            raise ValueError(f"Single-turn evaluator reference is missing: {instance_id}")
        _validate_source_memories(
            evaluator.get("source_memory_ids"), role, checkpoint, instance_id
        )
        if not clean_text(item.get("current_user_turn")):
            raise ValueError(f"Single-turn user turn is empty: {instance_id}")

    episodes = multi_turn.get("episodes")
    if multi_turn.get("schema_version") != "stage_task3_multi_turn" or not isinstance(
        episodes, list
    ):
        raise ValueError("Unsupported Task 3 multi-turn schema")
    if int(multi_turn.get("episode_count", -1)) != len(episodes):
        raise ValueError("Task 3 multi-turn episode_count mismatch")
    turn_count = 0
    for episode in episodes:
        if set(episode) != {
            "episode_id",
            "episode_theme",
            "role_ref",
            "checkpoint",
            "turns",
        }:
            raise ValueError(f"Unexpected formal multi-turn episode fields: {episode}")
        role = _validate_task_role_ref(episode, role_by_id)
        checkpoint = episode["checkpoint"]
        materialize_role_at_boundary(role, checkpoint)
        for expected_index, turn in enumerate(episode.get("turns", []), start=1):
            turn_count += 1
            if set(turn) != {
                "turn_index",
                "question_id",
                "dialogue_history_template",
                "current_user_turn",
                "evaluator_reference",
            }:
                raise ValueError(
                    f"Unexpected formal multi-turn fields: {episode['episode_id']}"
                )
            if int(turn.get("turn_index", -1)) != expected_index:
                raise ValueError(f"Multi-turn indices are not contiguous: {episode['episode_id']}")
            evaluator = turn.get("evaluator_reference")
            if not isinstance(evaluator, dict):
                raise ValueError(
                    f"Multi-turn evaluator reference is missing: {episode['episode_id']}"
                )
            _validate_source_memories(
                evaluator.get("source_memory_ids"),
                role,
                checkpoint,
                f"{episode['episode_id']} turn {expected_index}",
            )
            _validate_dialogue_history_template(
                turn.get("dialogue_history_template"),
                expected_index=expected_index,
                episode_id=episode["episode_id"],
            )
            if not clean_text(turn.get("current_user_turn")):
                raise ValueError(
                    f"Multi-turn user turn is empty: {episode['episode_id']}"
                )
    if int(multi_turn.get("turn_count", -1)) != turn_count:
        raise ValueError("Task 3 multi-turn turn_count mismatch")
    return {
        **role_counts,
        "single_turn_instances": len(instance_ids),
        "multi_turn_episodes": len(episodes),
        "multi_turn_turns": turn_count,
    }


def _validate_task_role_ref(
    item: dict[str, Any], role_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    role_id = clean_text(item.get("role_ref"))
    if role_id not in role_by_id:
        raise ValueError(f"Invalid Task 3 role reference: {role_id}")
    return role_by_id[role_id]


def _validate_source_memories(
    source_memory_ids: Any,
    role: dict[str, Any],
    checkpoint: dict[str, Any],
    label: str,
) -> None:
    if not isinstance(source_memory_ids, list) or len(source_memory_ids) != len(
        set(source_memory_ids)
    ):
        raise ValueError(f"Source memory IDs must be a unique array: {label}")
    memory_by_id = {item["memory_id"]: item for item in role["memories"]}
    for memory_id in source_memory_ids:
        memory = memory_by_id.get(memory_id)
        if memory is None:
            raise ValueError(f"Source memory does not belong to role: {label}/{memory_id}")
        if boundary_key(memory["available_from"]) > boundary_key(checkpoint):
            raise ValueError(f"Source memory is not visible at checkpoint: {label}/{memory_id}")


def _validate_dialogue_history_template(
    history: Any, *, expected_index: int, episode_id: str
) -> None:
    if not isinstance(history, list):
        raise ValueError(f"Dialogue history must be an array: {episode_id}")
    for item in history:
        if not isinstance(item, dict) or item.get("speaker") not in {"user", "assistant"}:
            raise ValueError(f"Invalid dialogue history item: {episode_id}")
        source_index = int(item.get("source_turn_index", 0))
        if source_index <= 0 or source_index >= expected_index:
            raise ValueError(f"Dialogue history references a future turn: {episode_id}")
        if item["speaker"] == "assistant" and item.get("fill_with") != "previous_model_response":
            raise ValueError(f"Assistant history must use a real prior response: {episode_id}")
        if item["speaker"] == "assistant" and "text" in item:
            raise ValueError(f"Assistant history leaks fixed reference text: {episode_id}")


def _assert_actor_view_hides_provenance(role: dict[str, Any]) -> None:
    visible_boundaries = [
        item["available_from"] for item in role.get("memories", [])
    ] + [
        item["valid_from"]
        for key in ("identity_phases", "persona_phases", "relationship_phases")
        for item in role.get(key, [])
    ]
    boundary = max(visible_boundaries, key=boundary_key)
    actor = materialize_role_at_boundary(role, boundary)
    forbidden = {
        "role_id",
        "character_id",
        "memory_id",
        "identity_phase_id",
        "persona_phase_id",
        "relationship_phase_id",
        "source_fact_ids",
        "source_episode_ids",
        "source_evidence_ids",
        "source_state_ids",
        "source_development_ids",
        "source_persona_evidence_ids",
        "knowledge_access",
        "available_from",
        "valid_from",
        "valid_until",
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
