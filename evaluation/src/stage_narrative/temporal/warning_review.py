from __future__ import annotations

from typing import Any

from stage_narrative.models import clean_text


WARNING_STATUSES = frozenset({"accepted_nonblocking", "resolved"})


def validate_temporal_warning_review(
    warnings: list[str], review: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if review.get("schema_version") != "stage_temporal_warning_review_v1":
        raise ValueError("Unsupported temporal warning review schema")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Temporal warning review decisions must be an array")
    by_warning: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        warning = clean_text(decision.get("warning"))
        if warning in by_warning:
            raise ValueError(f"Duplicate temporal warning review: {warning}")
        if warning not in warnings:
            raise ValueError(f"Unknown temporal warning review target: {warning}")
        status = clean_text(decision.get("status"))
        if status not in WARNING_STATUSES:
            raise ValueError(f"Unsupported temporal warning status: {status}")
        if not clean_text(decision.get("reason")):
            raise ValueError("Temporal warning decisions require a reason")
        by_warning[warning] = decision
    missing = set(warnings) - set(by_warning)
    if missing:
        raise ValueError(
            f"Temporal warning review does not cover all warnings: {sorted(missing)}"
        )
    ordered = [by_warning[warning] for warning in warnings]
    summary = {
        "schema_version": "stage_temporal_warning_review_summary_v1",
        "reviewer": review.get("reviewer"),
        "release_tier": review.get("release_tier", "agent_reviewed_silver"),
        "warning_count": len(warnings),
        "accepted_nonblocking_count": sum(
            item["status"] == "accepted_nonblocking" for item in ordered
        ),
        "resolved_count": sum(item["status"] == "resolved" for item in ordered),
        "released_scope_warning_count": sum(
            bool(item.get("referenced_by_released_tasks")) for item in ordered
        ),
        "full_coverage": True,
    }
    return ordered, summary
