from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import load_scenes, sha256_file


TASK3_MEMORY_MODES = {
    "persona_only",
    "full_visible_memory",
    "retrieval_visible_memory",
}


def materialize_task1_model_input(
    instance: dict[str, Any], *, script_path: Path
) -> dict[str, Any]:
    """Resolve one frozen Task 1 prefix reference without truncating the screenplay."""
    model_input = instance.get("model_input")
    if not isinstance(model_input, dict):
        raise ValueError("Task 1 instance requires model_input")
    prefix_ref = model_input.get("screenplay_prefix_ref")
    if not isinstance(prefix_ref, dict):
        raise ValueError("Task 1 model_input requires screenplay_prefix_ref")
    resolved_path = script_path.resolve()
    if sha256_file(resolved_path) != prefix_ref.get("script_sha256"):
        raise ValueError("Task 1 screenplay checksum differs from the frozen prefix reference")
    scenes = load_scenes(resolved_path)
    start = int(prefix_ref.get("start_scene_order", 0))
    end = int(prefix_ref.get("end_scene_order", 0))
    if start != 1 or end < start or end > len(scenes):
        raise ValueError("Task 1 screenplay prefix boundary is invalid")
    selected = [scene for scene in scenes if start <= scene.order <= end]
    if len(selected) != int(prefix_ref.get("scene_count", -1)):
        raise ValueError("Task 1 screenplay prefix count is inconsistent")
    return {
        "movie_id": instance["movie_id"],
        "focal_character": model_input["focal_character"],
        "aliases": list(model_input.get("aliases", [])),
        "previous_checkpoint_scene_order": model_input[
            "previous_checkpoint_scene_order"
        ],
        "current_checkpoint_scene_order": model_input[
            "current_checkpoint_scene_order"
        ],
        "screenplay_prefix": [
            {
                "scene_id": scene.scene_id,
                "scene_order": scene.order,
                "title": scene.title,
                "subtitle": scene.subtitle,
                "content": scene.content,
            }
            for scene in selected
        ],
        "output_schema": model_input["output_schema"],
        "input_policy": {
            "prefix_access": "complete",
            "truncation": "none",
        },
    }


def materialize_task3_actor_input(
    instance: dict[str, Any],
    *,
    role_snapshots: dict[str, Any],
    evidence_bank: dict[str, Any],
    persona_bank: dict[str, Any],
    graph: dict[str, Any],
    memory_mode: str = "full_visible_memory",
    retrieved_fact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve one role snapshot while keeping evaluator references out of actor input."""
    if memory_mode not in TASK3_MEMORY_MODES:
        raise ValueError(f"Unsupported Task 3 memory mode: {memory_mode}")
    model_input = instance.get("model_input")
    if not isinstance(model_input, dict):
        raise ValueError("Task 3 instance requires model_input")
    snapshots = {
        item["role_snapshot_id"]: item
        for item in role_snapshots.get("role_snapshots", [])
    }
    snapshot_id = model_input.get("role_snapshot_ref")
    snapshot = snapshots.get(snapshot_id)
    if snapshot is None:
        raise ValueError("Task 3 instance references an unknown role snapshot")
    if snapshot.get("character_id") != instance.get("character_id"):
        raise ValueError("Task 3 role snapshot belongs to another character")
    anchor = model_input.get("checkpoint_anchor")
    if anchor is None:
        anchor = {
            "scene_order": int(snapshot.get("scene_order", 0)),
            "char_end": 10**18,
            "boundary_policy": "legacy_scene_snapshot_unanchored",
            "review_status": "not_benchmark_ready",
        }
    elif not isinstance(anchor, dict):
        raise ValueError("Task 3 checkpoint anchor must be an object")
    anchor_scene_order = int(anchor.get("scene_order", 0))
    anchor_char_end = int(anchor.get("char_end", -1))
    if anchor_scene_order <= 0 or anchor_char_end < 0:
        raise ValueError("Task 3 checkpoint anchor is invalid")
    if int(snapshot.get("scene_order", 0)) != anchor_scene_order:
        raise ValueError("Task 3 role snapshot and checkpoint anchor disagree")

    evidence = {
        item["evidence_id"]: item for item in evidence_bank.get("evidence_units", [])
    }
    persona = {
        item["persona_evidence_id"]: item
        for item in persona_bank.get("persona_evidence", [])
    }
    facts = {
        item["id"]: item
        for item in graph.get("nodes", [])
        if item.get("node_type") in {"event", "occasion", "interaction", "atomic_fact"}
    }
    _require_known(
        snapshot.get("visible_persona_evidence_ids", []), persona, "persona evidence"
    )
    _require_known(
        snapshot.get("visible_dialogue_exemplar_ids", []),
        evidence,
        "dialogue exemplar",
    )
    _require_known(
        snapshot.get("visible_relation_evidence_ids", []),
        evidence,
        "relation evidence",
    )
    visible_fact_ids = list(snapshot.get("visible_memory_fact_ids", []))
    _require_known(visible_fact_ids, facts, "visible memory fact")
    visible_persona_ids = [
        item_id
        for item_id in snapshot.get("visible_persona_evidence_ids", [])
        if _persona_visible_at_anchor(
            persona[item_id],
            evidence,
            scene_order=anchor_scene_order,
            char_end=anchor_char_end,
        )
    ]
    visible_dialogue_ids = [
        item_id
        for item_id in snapshot.get("visible_dialogue_exemplar_ids", [])
        if _evidence_visible_at_anchor(
            evidence[item_id], scene_order=anchor_scene_order, char_end=anchor_char_end
        )
    ]
    visible_relation_ids = [
        item_id
        for item_id in snapshot.get("visible_relation_evidence_ids", [])
        if _evidence_visible_at_anchor(
            evidence[item_id], scene_order=anchor_scene_order, char_end=anchor_char_end
        )
    ]
    # Narrative graph nodes currently have scene-level source spans. Same-scene
    # nodes are therefore withheld instead of pretending they have a precise order.
    visible_fact_ids = [
        item_id
        for item_id in visible_fact_ids
        if _fact_visible_before_anchor_scene(
            facts[item_id], scene_order=anchor_scene_order
        )
    ]
    requested = list(retrieved_fact_ids or [])
    if memory_mode == "persona_only":
        selected_fact_ids: list[str] = []
        if requested:
            raise ValueError("persona_only mode does not accept retrieved facts")
    elif memory_mode == "full_visible_memory":
        selected_fact_ids = visible_fact_ids
        if requested:
            raise ValueError("retrieved_fact_ids are valid only in retrieval_visible_memory mode")
    else:
        if not set(requested) <= set(visible_fact_ids):
            raise ValueError("Task 3 retrieval requested a fact outside the role snapshot")
        selected_fact_ids = requested

    return {
        "movie_id": instance["movie_id"],
        "character": instance["character"],
        "interaction_format": "single_turn",
        "role_context": {
            "identity": _identity_visible_at_anchor(
                snapshot["identity_context"], scene_order=anchor_scene_order
            ),
            "persona_evidence": [
                {
                    "kind": persona[item_id]["evidence_kind"],
                    "value": persona[item_id]["value"],
                }
                for item_id in visible_persona_ids
                if item_id in persona
            ],
            "dialogue_exemplars": [
                {
                    "scene_order": evidence[item_id]["scene_order"],
                    "text": evidence[item_id]["evidence_text"],
                }
                for item_id in visible_dialogue_ids
                if item_id in evidence
            ],
            "visible_memories": [
                {
                    "fact_type": facts[item_id]["node_type"],
                    "fact": _fact_text(facts[item_id]),
                }
                for item_id in selected_fact_ids
                if item_id in facts
            ],
            "relation_evidence": [
                {
                    "scene_order": evidence[item_id]["scene_order"],
                    "text": evidence[item_id]["evidence_text"],
                }
                for item_id in visible_relation_ids
                if item_id in evidence
            ],
        },
        "interaction_context": model_input.get("interaction_context", ""),
        "dialogue_history": [],
        "current_user_turn": model_input["current_user_turn"],
        "input_policy": {
            "memory_mode": memory_mode,
            "checkpoint_anchor": anchor,
            "same_scene_graph_memory": "withheld_due_to_scene_level_provenance",
            "filtered_counts": {
                "persona_evidence": len(snapshot.get("visible_persona_evidence_ids", []))
                - len(visible_persona_ids),
                "dialogue_exemplars": len(snapshot.get("visible_dialogue_exemplar_ids", []))
                - len(visible_dialogue_ids),
                "visible_memories": len(snapshot.get("visible_memory_fact_ids", []))
                - len(visible_fact_ids),
                "relation_evidence": len(snapshot.get("visible_relation_evidence_ids", []))
                - len(visible_relation_ids),
            },
        },
    }


def _fact_text(node: dict[str, Any]) -> str:
    return str(node.get("fact") or node.get("description") or node.get("name") or "").strip()


def _require_known(values: Any, known: dict[str, Any], label: str) -> None:
    if not isinstance(values, list) or any(value not in known for value in values):
        raise ValueError(f"Task 3 role snapshot contains an unknown {label}")


def _evidence_visible_at_anchor(
    item: dict[str, Any], *, scene_order: int, char_end: int
) -> bool:
    item_scene = int(item.get("scene_order", 0))
    return item_scene < scene_order or (
        item_scene == scene_order and int(item.get("char_end", 10**18)) <= char_end
    )


def _persona_visible_at_anchor(
    item: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    *,
    scene_order: int,
    char_end: int,
) -> bool:
    established = int(item.get("established_from_scene", 10**18))
    if established > scene_order:
        return False
    support_ids = item.get("supporting_evidence_ids", [])
    if established < scene_order:
        return True
    return bool(support_ids) and all(
        evidence_id in evidence
        and _evidence_visible_at_anchor(
            evidence[evidence_id], scene_order=scene_order, char_end=char_end
        )
        for evidence_id in support_ids
    )


def _fact_visible_before_anchor_scene(
    item: dict[str, Any], *, scene_order: int
) -> bool:
    source_scene = item.get("source_scene_order")
    return source_scene is not None and int(source_scene) < scene_order


def _identity_visible_at_anchor(
    identity: dict[str, Any], *, scene_order: int
) -> dict[str, Any]:
    phases = [
        phase
        for phase in identity.get("identity_phases", [])
        if int(phase.get("valid_from_scene", 10**18)) <= scene_order
    ]
    active = [
        phase
        for phase in phases
        if phase.get("valid_until_scene") is None
        or int(phase["valid_until_scene"]) >= scene_order
    ]
    active_name = (
        str(active[-1].get("name", "")).strip()
        if active
        else str(identity.get("canonical_name", "")).strip()
    )
    aliases = []
    for phase in phases:
        text = str(phase.get("name", "")).strip()
        if text and text not in aliases:
            aliases.append(text)
    if not phases:
        canonical = str(identity.get("canonical_name", "")).strip()
        aliases = [canonical] if canonical else []
    return {
        "canonical_name": active_name,
        "aliases": aliases,
        "identity_phases_through_checkpoint": [
            {
                "name": phase.get("name", ""),
                "valid_from_scene": phase.get("valid_from_scene"),
                "valid_until_scene": phase.get("valid_until_scene"),
            }
            for phase in phases
        ],
    }
