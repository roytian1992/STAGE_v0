from __future__ import annotations

import copy
from typing import Any

from .models import normalize_name


_UPDATABLE_UNIT_FIELDS = frozenset(
    {
        "name",
        "description",
        "participants",
        "locations",
        "times",
        "setting",
        "evidence",
        "modality",
        "event_subtype",
        "state_before",
        "state_after",
        "intent",
        "cause_hints",
        "effect_hints",
        "related_occasion",
        "subject",
        "object",
        "interaction_type",
        "polarity",
        "tags",
        "outcome",
        "related_event",
        "occasion_type",
        "institutional_context",
    }
)


def apply_narrative_correction_patch(
    scene_records: list[dict[str, Any]], patch: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if patch.get("schema_version") != "stage_narrative_correction_patch_v1":
        raise ValueError("Unsupported narrative correction patch schema")
    output = copy.deepcopy(scene_records)
    scene_by_id = {
        record.get("scene", {}).get("scene_id"): record for record in output
    }
    reference_drop_audit: list[dict[str, Any]] = []
    for decision in patch.get("entity_reference_drops", []):
        scene_id = decision.get("scene_id")
        record = scene_by_id.get(scene_id)
        if record is None:
            raise ValueError(
                f"Entity reference drop references missing scene: {scene_id}"
            )
        names = {
            normalize_name(value)
            for value in (decision.get("entity_names") or [])
            if normalize_name(value)
        }
        if not names:
            raise ValueError("entity_reference_drops requires entity_names")
        before_entities = len(record.get("entities", []))
        record["entities"] = [
            item
            for item in record.get("entities", [])
            if normalize_name(item.get("name")) not in names
        ]
        before_relations = len(record.get("entity_relations", []))
        record["entity_relations"] = [
            item
            for item in record.get("entity_relations", [])
            if normalize_name(item.get("subject")) not in names
            and normalize_name(item.get("object")) not in names
        ]
        dropped_participants = 0
        for unit in record.get("narrative_units", []):
            participants = list(unit.get("participants", []))
            unit["participants"] = [
                value for value in participants if normalize_name(value) not in names
            ]
            dropped_participants += len(participants) - len(unit["participants"])
        reference_drop_audit.append(
            {
                "decision_id": decision.get("decision_id"),
                "action": "drop_entity_references",
                "scene_id": scene_id,
                "entity_names": sorted(names),
                "dropped_scene_entity_count": before_entities - len(record["entities"]),
                "dropped_scene_relation_count": before_relations
                - len(record["entity_relations"]),
                "dropped_participant_count": dropped_participants,
                "reason": decision.get("reason"),
                "evidence": decision.get("evidence"),
            }
        )
    unit_locations = {
        unit.get("unit_id"): (record, unit)
        for record in output
        for unit in record.get("narrative_units", [])
    }
    applied: list[dict[str, Any]] = []
    for decision in patch.get("decisions", []):
        action = decision.get("action")
        if action not in {
            "drop_narrative_unit",
            "drop_occasion_artifact",
            "update_narrative_unit",
        }:
            raise ValueError(
                f"Unsupported narrative correction action: {decision.get('action')}"
            )
        unit_id = decision.get("unit_id")
        if unit_id not in unit_locations:
            raise ValueError(f"Narrative correction references missing unit: {unit_id}")
        record, unit = unit_locations.pop(unit_id)
        if decision.get("expected_kind") and unit.get("kind") != decision["expected_kind"]:
            raise ValueError(f"Narrative unit kind mismatch for {unit_id}")
        if action == "update_narrative_unit":
            updates = decision.get("field_updates")
            if not isinstance(updates, dict) or not updates:
                raise ValueError("update_narrative_unit requires non-empty field_updates")
            unsupported = set(updates) - _UPDATABLE_UNIT_FIELDS
            if unsupported:
                raise ValueError(
                    f"Unsupported narrative unit update fields: {sorted(unsupported)}"
                )
            before = {key: copy.deepcopy(unit.get(key)) for key in updates}
            unit.update(copy.deepcopy(updates))
            applied.append(
                {
                    "decision_id": decision.get("decision_id"),
                    "action": action,
                    "unit_id": unit_id,
                    "scene_id": record.get("scene", {}).get("scene_id"),
                    "kind": unit.get("kind"),
                    "reason": decision.get("reason"),
                    "evidence": decision.get("evidence"),
                    "before": before,
                    "after": {key: copy.deepcopy(unit.get(key)) for key in updates},
                    "dropped_scene_entity_count": 0,
                    "dropped_scene_relation_count": 0,
                }
            )
            continue
        record["narrative_units"] = [
            item for item in record.get("narrative_units", []) if item.get("unit_id") != unit_id
        ]
        dropped_entity_count = 0
        dropped_relation_count = 0
        if action == "drop_occasion_artifact":
            unit_name = normalize_name(
                decision.get("occasion_name") or unit.get("name")
            )
            if not unit_name:
                raise ValueError(
                    f"drop_occasion_artifact requires a unit or explicit Occasion name: {unit_id}"
                )
            before_entities = len(record.get("entities", []))
            record["entities"] = [
                item
                for item in record.get("entities", [])
                if not (
                    item.get("entity_type") == "Occasion"
                    and normalize_name(item.get("name")) == unit_name
                )
            ]
            dropped_entity_count = before_entities - len(record["entities"])
            before_relations = len(record.get("entity_relations", []))
            record["entity_relations"] = [
                item
                for item in record.get("entity_relations", [])
                if normalize_name(item.get("subject")) != unit_name
                and normalize_name(item.get("object")) != unit_name
            ]
            dropped_relation_count = before_relations - len(
                record["entity_relations"]
            )
        applied.append(
            {
                "decision_id": decision.get("decision_id"),
                "action": action,
                "unit_id": unit_id,
                "scene_id": record.get("scene", {}).get("scene_id"),
                "kind": unit.get("kind"),
                "name": unit.get("name"),
                "occasion_name": decision.get("occasion_name") or unit.get("name"),
                "reason": decision.get("reason"),
                "evidence": decision.get("evidence"),
                "dropped_scene_entity_count": dropped_entity_count,
                "dropped_scene_relation_count": dropped_relation_count,
            }
        )
    return output, {
        "schema_version": "stage_narrative_correction_audit_v1",
        "reviewer": patch.get("reviewer"),
        "release_tier": patch.get("release_tier", "agent_reviewed_silver"),
        "decision_count": len(applied),
        "model_calls": 0,
        "decisions": [*reference_drop_audit, *applied],
        "entity_reference_drop_count": len(reference_drop_audit),
    }
