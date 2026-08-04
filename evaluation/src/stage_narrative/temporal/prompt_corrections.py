from __future__ import annotations

import copy
from typing import Any

from stage_narrative.models import clean_text


UPDATABLE_PROMPT_FIELDS = frozenset(
    {
        "interaction_context",
        "current_user_turn",
        "expected_stances",
        "boundary_risk_type",
        "unknown_fact_ids",
        "future_forbidden_fact_ids",
        "contradicting_fact_ids",
    }
)


def apply_temporal_prompt_correction_patch(
    prompt_stage: dict[str, Any], patch: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if patch.get("schema_version") != "stage_temporal_prompt_correction_patch_v1":
        raise ValueError("Unsupported temporal prompt correction patch")
    output = copy.deepcopy(prompt_stage)
    prompt_by_id = {item["prompt_id"]: item for item in output.get("prompts", [])}
    rejected_ids: set[str] = set()
    decisions = []
    for rejection in patch.get("rejections", []):
        prompt_id = clean_text(rejection.get("prompt_id"))
        if prompt_id not in prompt_by_id:
            raise ValueError(f"Unknown temporal prompt rejection: {prompt_id}")
        if prompt_id in rejected_ids:
            raise ValueError(f"Duplicate temporal prompt rejection: {prompt_id}")
        rejected_ids.add(prompt_id)
        decisions.append(
            {
                "decision_id": rejection.get("decision_id"),
                "prompt_id": prompt_id,
                "action": "reject",
                "reason_code": rejection.get("reason_code"),
                "reason": rejection.get("reason"),
                "evidence": rejection.get("evidence", []),
            }
        )

    corrected_ids: set[str] = set()
    for correction in patch.get("decisions", []):
        prompt_id = clean_text(correction.get("prompt_id"))
        if prompt_id not in prompt_by_id:
            raise ValueError(f"Unknown temporal prompt correction: {prompt_id}")
        if prompt_id in rejected_ids or prompt_id in corrected_ids:
            raise ValueError(f"Prompt has conflicting correction decisions: {prompt_id}")
        updates = correction.get("field_updates")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("Temporal prompt correction requires field_updates")
        unsupported = set(updates) - UPDATABLE_PROMPT_FIELDS
        if unsupported:
            raise ValueError(
                f"Unsupported temporal prompt fields: {sorted(unsupported)}"
            )
        prompt = prompt_by_id[prompt_id]
        before = {key: copy.deepcopy(prompt.get(key)) for key in updates}
        prompt.update(copy.deepcopy(updates))
        corrected_ids.add(prompt_id)
        decisions.append(
            {
                "decision_id": correction.get("decision_id"),
                "prompt_id": prompt_id,
                "action": "update",
                "reason_code": correction.get("reason_code"),
                "reason": correction.get("reason"),
                "evidence": correction.get("evidence", []),
                "before": before,
                "after": {key: copy.deepcopy(prompt.get(key)) for key in updates},
            }
        )

    output["prompts"] = [
        item for item in output.get("prompts", []) if item["prompt_id"] not in rejected_ids
    ]
    audit = {
        "schema_version": "stage_temporal_prompt_correction_audit_v1",
        "reviewer": patch.get("reviewer"),
        "release_tier": patch.get("release_tier", "agent_reviewed_silver"),
        "model_calls": 0,
        "input_prompt_count": len(prompt_by_id),
        "output_prompt_count": len(output["prompts"]),
        "rejection_count": len(rejected_ids),
        "correction_count": len(corrected_ids),
        "decisions": decisions,
    }
    output.setdefault("audit", {}).setdefault("agent_corrections", []).append(audit)
    return output, audit
