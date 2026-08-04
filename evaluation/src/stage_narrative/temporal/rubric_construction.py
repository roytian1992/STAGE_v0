from __future__ import annotations

from typing import Any

from ..models import clean_text


TASK1_RUBRIC_KEYS = {
    "current_state_claims",
    "development_claims",
    "invariant_claims",
    "salient_future_negatives",
}
TASK3_RUBRIC_KEYS = {
    "expected_stances",
    "acceptable_variations",
    "required_or_relevant_memories",
    "contradictions",
    "unknown_at_checkpoint",
    "salient_future_negatives",
    "style_requirements",
}
TASK1_RUBRIC_LIMITS = {
    "current_state_claims": 8,
    "development_claims": 4,
    "invariant_claims": 4,
    "salient_future_negatives": 4,
}


def validate_and_localize_task1_rubric(
    payload: dict[str, Any],
    *,
    checkpoint_scene_order: int,
    previous_checkpoint_scene_order: int = 0,
    allowed_future_local_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != TASK1_RUBRIC_KEYS:
        raise ValueError("Task 1 rubric draft has invalid top-level fields")
    for field, maximum in TASK1_RUBRIC_LIMITS.items():
        if not isinstance(payload[field], list) or len(payload[field]) > maximum:
            raise ValueError(
                f"Task 1 rubric draft exceeds compactness limit: "
                f"{field}/{len(payload[field]) if isinstance(payload[field], list) else 'non-list'}>{maximum}"
            )
    output: dict[str, Any] = {}
    for field, prefix in (
        ("current_state_claims", "S"),
        ("development_claims", "D"),
        ("invariant_claims", "I"),
    ):
        output[field] = _claim_rows(
            payload[field],
            prefix=prefix,
            maximum_scene_order=checkpoint_scene_order,
        )
    if any(
        not any(
            scene > previous_checkpoint_scene_order
            for scene in row["supporting_scene_orders"]
        )
        for row in output["development_claims"]
    ):
        raise ValueError(
            "Task 1 development lacks evidence after the previous checkpoint"
        )
    output["salient_future_negatives"] = _future_rows(
        payload["salient_future_negatives"],
        allowed_future_local_ids=allowed_future_local_ids,
    )
    return output


def validate_and_localize_task3_rubric(
    payload: dict[str, Any], *, allowed_future_local_ids: set[str]
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != TASK3_RUBRIC_KEYS:
        raise ValueError("Task 3 rubric draft has invalid top-level fields")
    output = {
        field: _unique_nonempty_strings(payload[field], label=field)
        for field in TASK3_RUBRIC_KEYS - {"salient_future_negatives"}
    }
    output["salient_future_negatives"] = _future_rows(
        payload["salient_future_negatives"],
        allowed_future_local_ids=allowed_future_local_ids,
    )
    return output


def _claim_rows(
    raw: Any, *, prefix: str, maximum_scene_order: int
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("Rubric claims must be a list")
    output = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "claim",
            "supporting_scene_orders",
        }:
            raise ValueError("Rubric claim has invalid fields")
        claim = clean_text(item["claim"])
        normalized = claim.casefold()
        scenes = item["supporting_scene_orders"]
        if not claim or normalized in seen:
            continue
        if (
            not isinstance(scenes, list)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in scenes)
            or any(value <= 0 or value > maximum_scene_order for value in scenes)
        ):
            raise ValueError("Rubric claim cites an invalid scene order")
        seen.add(normalized)
        output.append(
            {
                "local_id": f"{prefix}{len(output) + 1}",
                "claim": claim,
                "supporting_scene_orders": sorted(set(scenes)),
            }
        )
    return output


def _future_rows(
    raw: Any, *, allowed_future_local_ids: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("Future-negative rubric field must be a list")
    output = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "claim",
            "source_future_local_ids",
        }:
            raise ValueError("Future-negative rubric row has invalid fields")
        claim = clean_text(item["claim"])
        source_ids = item["source_future_local_ids"]
        if (
            not claim
            or not isinstance(source_ids, list)
            or not source_ids
            or any(value not in allowed_future_local_ids for value in source_ids)
        ):
            raise ValueError("Future-negative rubric row is not grounded in supplied candidates")
        normalized = claim.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(
            {
                "local_id": f"F{len(output) + 1}",
                "claim": claim,
                "source_future_local_ids": list(dict.fromkeys(source_ids)),
            }
        )
    return output


def _unique_nonempty_strings(raw: Any, *, label: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"Task 3 rubric field must be a list: {label}")
    output = []
    seen: set[str] = set()
    for value in raw:
        text = clean_text(value)
        normalized = text.casefold()
        if text and normalized not in seen:
            seen.add(normalized)
            output.append(text)
    return output
