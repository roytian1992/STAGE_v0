from __future__ import annotations

from typing import Any


TASK3_SCORE_FIELDS = (
    "character_fidelity",
    "memory_faithfulness",
    "boundary_compliance",
    "response_naturalness",
)
PAIR_TYPES = {
    "expected_change",
    "expected_stability",
    "knowledge_acquisition",
    "relationship_change",
}


def validate_task1_judgment(
    payload: dict[str, Any], *, gold_ids: set[str], prediction_ids: set[str]
) -> dict[str, Any]:
    required = {
        "claim_pair_judgments",
        "prediction_checks",
        "future_leak_prediction_ids",
        "premature_update_prediction_ids",
        "no_change_false_update",
    }
    _exact_keys(payload, required, "Task 1 judgment")
    pairs = payload["claim_pair_judgments"]
    checks = payload["prediction_checks"]
    if not isinstance(pairs, list) or not isinstance(checks, list):
        raise ValueError("Task 1 judgment pair/check fields must be arrays")
    seen_checks = set()
    for row in pairs:
        _exact_keys(row, {"gold_local_id", "prediction_local_id", "label"}, "Task 1 pair")
        if row["gold_local_id"] not in gold_ids or row["prediction_local_id"] not in prediction_ids:
            raise ValueError("Task 1 pair contains an unknown local ID")
        if row["label"] not in {"full", "partial", "contradiction"}:
            raise ValueError("Task 1 pair contains an invalid label")
    for row in checks:
        _exact_keys(
            row,
            {"prediction_local_id", "support", "transition_coherent", "evidence_grounded"},
            "Task 1 prediction check",
        )
        local_id = row["prediction_local_id"]
        if local_id not in prediction_ids or local_id in seen_checks:
            raise ValueError("Task 1 prediction checks must cover unique known IDs")
        seen_checks.add(local_id)
        if row["support"] not in {"supported", "partial", "unsupported"}:
            raise ValueError("Invalid Task 1 support label")
        if not isinstance(row["transition_coherent"], bool) or not isinstance(row["evidence_grounded"], bool):
            raise ValueError("Task 1 diagnostic flags must be booleans")
    if seen_checks != prediction_ids:
        raise ValueError("Task 1 prediction checks do not have exact ID coverage")
    for field in ("future_leak_prediction_ids", "premature_update_prediction_ids"):
        values = payload[field]
        if not isinstance(values, list) or not set(values) <= prediction_ids:
            raise ValueError(f"Task 1 {field} contains unknown IDs")
        payload[field] = sorted(set(values))
    if not isinstance(payload["no_change_false_update"], bool):
        raise ValueError("Task 1 no-change flag must be boolean")
    return payload


def validate_task3_response_judgment(
    payload: dict[str, Any], *, allowed_evidence_ids: set[str]
) -> dict[str, Any]:
    required = {
        "scores",
        "future_leakage",
        "unknown_fact_hallucination",
        "stance_compatible",
        "evidence_local_ids",
        "brief_rationale",
    }
    _exact_keys(payload, required, "Task 3 response judgment")
    _exact_keys(payload["scores"], set(TASK3_SCORE_FIELDS), "Task 3 scores")
    for field in TASK3_SCORE_FIELDS:
        score = payload["scores"][field]
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            raise ValueError(f"Task 3 score is outside 1..5: {field}")
    for field in ("future_leakage", "unknown_fact_hallucination", "stance_compatible"):
        if not isinstance(payload[field], bool):
            raise ValueError(f"Task 3 flag must be boolean: {field}")
    if (
        not payload["future_leakage"]
        and not payload["unknown_fact_hallucination"]
        and payload["scores"]["boundary_compliance"] <= 2
    ):
        raise ValueError(
            "Task 3 Boundary Compliance 1--2 requires a boundary violation flag"
        )
    if (
        (payload["future_leakage"] or payload["unknown_fact_hallucination"])
        and payload["scores"]["boundary_compliance"] >= 4
    ):
        raise ValueError(
            "Task 3 boundary violation flag requires Boundary Compliance at most 3"
        )
    evidence_ids = payload["evidence_local_ids"]
    if not isinstance(evidence_ids, list) or not set(evidence_ids) <= allowed_evidence_ids:
        raise ValueError("Task 3 judgment contains an unknown evidence label")
    payload["evidence_local_ids"] = sorted(set(evidence_ids))
    if not isinstance(payload["brief_rationale"], str):
        raise ValueError("Task 3 rationale must be text")
    return payload


def validate_task1_sequence_judgment(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "state_carry_forward",
        "development_to_state_coherent",
        "contradiction_present",
        "premature_or_future_information",
        "local_evidence_labels",
        "brief_rationale",
    }
    _exact_keys(payload, required, "Task 1 sequence judgment")
    for field in (
        "state_carry_forward",
        "development_to_state_coherent",
        "contradiction_present",
        "premature_or_future_information",
    ):
        if not isinstance(payload[field], bool):
            raise ValueError(f"Task 1 sequence flag must be boolean: {field}")
    if not isinstance(payload["local_evidence_labels"], list):
        raise ValueError("Task 1 sequence evidence labels must be an array")
    if not isinstance(payload["brief_rationale"], str):
        raise ValueError("Task 1 sequence rationale must be text")
    return payload


def validate_task3_pair_judgment(payload: dict[str, Any], *, expected_pair_type: str) -> dict[str, Any]:
    required = {
        "pair_type",
        "response_assessments",
        "expected_direction_present",
        "unsupported_drift",
        "knowledge_boundaries_preserved",
        "local_evidence_labels",
        "brief_rationale",
    }
    _exact_keys(payload, required, "Task 3 pair judgment")
    if expected_pair_type not in PAIR_TYPES or payload["pair_type"] != expected_pair_type:
        raise ValueError("Task 3 pair judge returned the wrong pair type")
    assessments = payload["response_assessments"]
    if not isinstance(assessments, list) or len(assessments) != 2:
        raise ValueError("Task 3 pair judge must assess exactly T1 and T2")
    assessment_keys = {
        "response_label",
        "response_excerpt",
        "observed_behavior",
        "supports_expected_component",
    }
    labels = []
    for row in assessments:
        _exact_keys(row, assessment_keys, "Task 3 pair response assessment")
        labels.append(row["response_label"])
        if not isinstance(row["response_excerpt"], str) or not row["response_excerpt"].strip():
            raise ValueError("Task 3 pair response excerpt must be nonempty text")
        if not isinstance(row["observed_behavior"], str) or not row["observed_behavior"].strip():
            raise ValueError("Task 3 pair observed behavior must be nonempty text")
        if not isinstance(row["supports_expected_component"], bool):
            raise ValueError("Task 3 pair component support must be boolean")
    if labels != ["T1", "T2"]:
        raise ValueError("Task 3 pair response assessments must be ordered T1, T2")
    for field in (
        "expected_direction_present",
        "unsupported_drift",
        "knowledge_boundaries_preserved",
    ):
        if not isinstance(payload[field], bool):
            raise ValueError(f"Task 3 pair flag must be boolean: {field}")
    if not isinstance(payload["local_evidence_labels"], list):
        raise ValueError("Task 3 pair evidence labels must be an array")
    if not isinstance(payload["brief_rationale"], str):
        raise ValueError("Task 3 pair rationale must be text")
    return payload


def _exact_keys(payload: Any, required: set[str], label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != required:
        actual = set(payload) if isinstance(payload, dict) else set()
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(required - actual)} extra={sorted(actual - required)}"
        )
