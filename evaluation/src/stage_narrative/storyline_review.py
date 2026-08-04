from __future__ import annotations

from typing import Any

from .models import clean_text


def build_storyline_agent_audit(
    storylines: list[dict[str, Any]],
    review: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if review.get("schema_version") != "stage_storyline_agent_review_v1":
        raise ValueError("Unsupported Storyline agent review schema")
    storyline_by_id = {item["storyline_id"]: item for item in storylines}
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Storyline review decisions must be an array")
    decision_by_id: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        storyline_id = clean_text(decision.get("storyline_id"))
        if storyline_id in decision_by_id:
            raise ValueError(f"Duplicate Storyline review decision: {storyline_id}")
        if storyline_id not in storyline_by_id:
            raise ValueError(f"Unknown Storyline review target: {storyline_id}")
        status = clean_text(decision.get("status"))
        if status not in {"approved", "approved_after_correction"}:
            raise ValueError(f"Unsupported Storyline review status: {status}")
        if not clean_text(decision.get("reason")):
            raise ValueError("Storyline review decisions require a reason")
        decision_by_id[storyline_id] = decision
    missing = set(storyline_by_id) - set(decision_by_id)
    if missing:
        raise ValueError(f"Storyline review does not cover all outputs: {sorted(missing)}")

    audit_items: list[dict[str, Any]] = []
    for storyline in sorted(storylines, key=lambda item: int(item["order"])):
        decision = decision_by_id[storyline["storyline_id"]]
        is_backbone = storyline["focus_type"] == "chronological_backbone"
        audit_items.append(
            {
                "schema_version": "stage_storyline_agent_audit_item_v1",
                "reviewer": review.get("reviewer"),
                "decision_id": decision.get("decision_id"),
                "storyline_id": storyline["storyline_id"],
                "name": storyline["name"],
                "status": decision["status"],
                "reason": decision["reason"],
                "evidence": decision.get("evidence", []),
                "checks": {
                    "role": "chronological_index" if is_backbone else "evolution_storyline",
                    "cross_scene": is_backbone or len(storyline["source_scene_ids"]) >= 2,
                    "relation_connected": is_backbone
                    or bool(storyline["supporting_relation_ids"]),
                    "focus_grounded": is_backbone or bool(storyline["focus_entity_ids"]),
                    "ordered_state_transition": is_backbone
                    or bool(storyline["ordered_transitions"]),
                    "modality_qualified_when_needed": bool(
                        decision.get("modality_qualified_when_needed", True)
                    ),
                },
            }
        )
    summary = {
        "schema_version": "stage_storyline_agent_review_summary_v1",
        "reviewer": review.get("reviewer"),
        "release_tier": review.get("release_tier", "agent_reviewed_silver"),
        "model_calls": 0,
        "storyline_count": len(storylines),
        "evolution_storyline_count": sum(
            item["focus_type"] != "chronological_backbone" for item in storylines
        ),
        "approved_count": sum(
            item["status"] == "approved" for item in audit_items
        ),
        "approved_after_correction_count": sum(
            item["status"] == "approved_after_correction" for item in audit_items
        ),
        "full_coverage": True,
    }
    return audit_items, summary
