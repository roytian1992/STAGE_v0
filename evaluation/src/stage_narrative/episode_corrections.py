from __future__ import annotations

import copy
from typing import Any

from .hierarchy import _episode_anchor_assets
from .models import clean_text, unique_text


_UPDATABLE_EPISODE_FIELDS = frozenset(
    {
        "name",
        "description",
        "setting",
        "initial_situation",
        "progression_steps",
        "outcome",
        "state_changes",
        "causal_links",
        "open_threads",
        "closed_threads",
        "key_evidence",
        "main_participants",
        "order",
        "scene_episode_order",
    }
)


def apply_episode_correction_patch(
    episode_stage: dict[str, Any],
    scene_records: list[dict[str, Any]],
    patch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if patch.get("schema_version") != "stage_episode_correction_patch_v1":
        raise ValueError("Unsupported episode correction patch schema")
    output = copy.deepcopy(episode_stage)
    unit_by_id = {
        clean_text(unit.get("unit_id")): unit
        for record in scene_records
        for unit in record.get("narrative_units", [])
        if clean_text(unit.get("unit_id"))
    }
    episode_by_id = {
        episode["episode_id"]: episode for episode in output.get("episodes", [])
    }
    for episode in output.get("episodes", []):
        unit_ids = [
            *episode.get("child_unit_ids", []),
            *episode.get("context_occasion_unit_ids", []),
        ]
        missing = [unit_id for unit_id in unit_ids if unit_id not in unit_by_id]
        if missing:
            raise ValueError(
                f"Episode {episode['episode_id']} references missing units: {missing}"
            )
        anchors = _episode_anchor_assets([unit_by_id[unit_id] for unit_id in unit_ids])
        child_modalities = unique_text(
            clean_text(unit_by_id[unit_id].get("modality")) or "asserted"
            for unit_id in episode.get("child_unit_ids", [])
        )
        episode.update(
            {
                "participants": [
                    item["canonical_name"] for item in anchors["participant_entities"]
                ],
                "modality": (
                    child_modalities[0]
                    if len(child_modalities) == 1
                    else "uncertain"
                ),
                "source_modalities": child_modalities,
                **anchors,
            }
        )

    applied: list[dict[str, Any]] = []
    seen: set[str] = set()
    for decision in patch.get("decisions", []):
        episode_id = clean_text(decision.get("episode_id"))
        if episode_id not in episode_by_id:
            raise ValueError(f"Unknown episode correction target: {episode_id}")
        if episode_id in seen:
            raise ValueError(f"Duplicate episode correction target: {episode_id}")
        seen.add(episode_id)
        updates = decision.get("field_updates")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("Episode correction requires non-empty field_updates")
        unsupported = set(updates) - _UPDATABLE_EPISODE_FIELDS
        if unsupported:
            raise ValueError(f"Unsupported episode update fields: {sorted(unsupported)}")
        episode = episode_by_id[episode_id]
        before = {key: copy.deepcopy(episode.get(key)) for key in updates}
        episode.update(copy.deepcopy(updates))
        applied.append(
            {
                "decision_id": decision.get("decision_id"),
                "episode_id": episode_id,
                "source_scene_ids": episode.get("source_scene_ids", []),
                "reason": decision.get("reason"),
                "evidence": decision.get("evidence", []),
                "before": before,
                "after": {key: copy.deepcopy(episode.get(key)) for key in updates},
            }
        )
    output["episodes"].sort(key=lambda item: int(item["order"]))
    orders = [int(item["order"]) for item in output["episodes"]]
    if orders != list(range(1, len(output["episodes"]) + 1)):
        raise ValueError("Corrected Episode orders must be unique and contiguous")
    next_scene_order: dict[str, int] = {}
    scene_order_renumber_count = 0
    for episode in output["episodes"]:
        scene_id = episode["source_scene_ids"][0]
        normalized_order = next_scene_order.get(scene_id, 0) + 1
        next_scene_order[scene_id] = normalized_order
        if int(episode["scene_episode_order"]) != normalized_order:
            episode["scene_episode_order"] = normalized_order
            scene_order_renumber_count += 1
    scene_orders: dict[str, list[int]] = {}
    for episode in output["episodes"]:
        scene_id = episode["source_scene_ids"][0]
        scene_orders.setdefault(scene_id, []).append(int(episode["scene_episode_order"]))
    invalid_scene_orders = {
        scene_id: values
        for scene_id, values in scene_orders.items()
        if sorted(values) != list(range(1, len(values) + 1))
    }
    if invalid_scene_orders:
        raise ValueError(f"Corrected scene Episode orders are invalid: {invalid_scene_orders}")
    output.setdefault("audit", {})["agent_correction"] = {
        "schema_version": "stage_episode_correction_audit_v1",
        "reviewer": patch.get("reviewer"),
        "release_tier": patch.get("release_tier", "agent_reviewed_silver"),
        "model_calls": 0,
        "episode_count": len(episode_by_id),
        "structurally_rematerialized_episode_count": len(episode_by_id),
        "semantic_correction_count": len(applied),
        "scene_episode_order_renumber_count": scene_order_renumber_count,
        "decisions": applied,
    }
    return output, output["audit"]["agent_correction"]
