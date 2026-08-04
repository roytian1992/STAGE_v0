from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any


TASK3_SCORE_DIMENSIONS = {
    "character_fidelity",
    "response_naturalness",
    "memory_faithfulness",
    "boundary_compliance",
}
TASK3_PAIR_TYPES = {
    "evolving_pair",
    "invariant_pair",
    "inaccessible_pair",
    "post_disclosure_pair",
}


def score_task1(
    task: dict[str, Any], assessments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate instance-level adjudication and compute deterministic Task 1 metrics."""
    instances = _index(task.get("instances"), "instance_id", "Task 1 instance")
    judged = _index(assessments, "instance_id", "Task 1 assessment")
    _require_exact_coverage(instances, judged, "Task 1 assessment")

    current_gold = current_matched = current_predicted = 0
    development_gold = development_matched = development_predicted = 0
    coherent = dimension_correct = progression_correct = 0
    grounded_claims = predicted_claims = 0
    no_change_total = no_change_false_updates = 0
    inaccessible_total = inaccessible_premature = 0
    delayed_values: list[bool] = []
    longitudinal_values: list[bool] = []

    for instance_id, instance in instances.items():
        assessment = judged[instance_id]
        _exact_keys(
            assessment,
            {
                "instance_id",
                "matched_current_state_ids",
                "predicted_current_state_count",
                "matched_development_ids",
                "predicted_development_count",
                "coherent_development_count",
                "dimension_correct_development_count",
                "progression_correct_development_count",
                "grounded_claim_count",
                "predicted_claim_count",
                "premature_update",
                "delayed_timing_correct",
                "longitudinal_consistent",
            },
            "Task 1 assessment",
        )
        reference = instance["evaluator_reference"]
        matched_states = _unique_strings(
            assessment["matched_current_state_ids"], "matched current states"
        )
        matched_developments = _unique_strings(
            assessment["matched_development_ids"], "matched developments"
        )
        if not set(matched_states) <= set(reference["gold_current_state_ids"]):
            raise ValueError(f"Task 1 assessment matches a non-gold state: {instance_id}")
        if not set(matched_developments) <= set(reference["gold_development_ids"]):
            raise ValueError(f"Task 1 assessment matches a non-gold development: {instance_id}")

        predicted_states = _count(
            assessment["predicted_current_state_count"], "predicted current states"
        )
        predicted_developments = _count(
            assessment["predicted_development_count"], "predicted developments"
        )
        if len(matched_states) > predicted_states:
            raise ValueError(f"Task 1 matched states exceed predictions: {instance_id}")
        if len(matched_developments) > predicted_developments:
            raise ValueError(f"Task 1 matched developments exceed predictions: {instance_id}")
        per_development = [
            _count(assessment[key], key)
            for key in (
                "coherent_development_count",
                "dimension_correct_development_count",
                "progression_correct_development_count",
            )
        ]
        if any(value > predicted_developments for value in per_development):
            raise ValueError(f"Task 1 diagnostic count exceeds developments: {instance_id}")
        local_grounded = _count(assessment["grounded_claim_count"], "grounded claims")
        local_claims = _count(assessment["predicted_claim_count"], "predicted claims")
        if local_grounded > local_claims:
            raise ValueError(f"Task 1 grounded claims exceed predicted claims: {instance_id}")
        premature = _boolean(assessment["premature_update"], "premature_update")
        delayed = _optional_boolean(
            assessment["delayed_timing_correct"], "delayed_timing_correct"
        )
        longitudinal = _optional_boolean(
            assessment["longitudinal_consistent"], "longitudinal_consistent"
        )

        current_gold += len(reference["gold_current_state_ids"])
        current_matched += len(matched_states)
        current_predicted += predicted_states
        development_gold += len(reference["gold_development_ids"])
        development_matched += len(matched_developments)
        development_predicted += predicted_developments
        coherent += per_development[0]
        dimension_correct += per_development[1]
        progression_correct += per_development[2]
        grounded_claims += local_grounded
        predicted_claims += local_claims
        controls = set(reference.get("checkpoint_control_types", []))
        legacy_control = reference.get("checkpoint_control_type")
        if legacy_control in {"no_change", "inaccessible", "delayed_consequence"}:
            controls.add(legacy_control)
        if "no_change" in controls:
            no_change_total += 1
            no_change_false_updates += int(predicted_developments > 0)
        if "inaccessible" in controls:
            inaccessible_total += 1
            inaccessible_premature += int(premature)
        if "delayed_consequence" in controls:
            if delayed is None:
                raise ValueError(f"Delayed checkpoint lacks timing assessment: {instance_id}")
            delayed_values.append(delayed)
        if longitudinal is not None:
            longitudinal_values.append(longitudinal)

    current_precision, current_recall, current_f1 = _prf(
        current_matched, current_predicted, current_gold
    )
    development_precision, development_recall, development_f1 = _prf(
        development_matched, development_predicted, development_gold
    )
    return {
        "schema_version": "stage_task1_metrics_v1",
        "task": task.get("task"),
        "instance_count": len(instances),
        "headline": {
            "current_state_fact_precision": current_precision,
            "current_state_fact_recall": current_recall,
            "current_state_fact_f1": current_f1,
            "development_fact_precision": development_precision,
            "development_fact_recall": development_recall,
            "development_fact_f1": development_f1,
            "catalyst_resulting_state_coherence": _ratio(
                coherent, development_predicted
            ),
            "development_dimension_correctness": _ratio(
                dimension_correct, development_predicted
            ),
            "progression_correctness": _ratio(
                progression_correct, development_predicted
            ),
            "evidence_grounding": _ratio(grounded_claims, predicted_claims),
        },
        "diagnostics": {
            "no_change_false_update_rate": _ratio(
                no_change_false_updates, no_change_total
            ),
            "inaccessible_event_premature_update_rate": _ratio(
                inaccessible_premature, inaccessible_total
            ),
            "delayed_update_timing_accuracy": _mean_booleans(delayed_values),
            "longitudinal_consistency": _mean_booleans(longitudinal_values),
        },
        "denominators": {
            "current_state_gold": current_gold,
            "current_state_predicted": current_predicted,
            "development_gold": development_gold,
            "development_predicted": development_predicted,
            "no_change_instances": no_change_total,
            "inaccessible_instances": inaccessible_total,
            "delayed_instances": len(delayed_values),
        },
    }


def score_task3(
    task: dict[str, Any],
    assessments: list[dict[str, Any]],
    pair_assessments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate response and pair adjudication, then aggregate Task 3 metrics."""
    instances = _index(task.get("instances"), "instance_id", "Task 3 instance")
    judged = _index(assessments, "instance_id", "Task 3 assessment")
    _require_exact_coverage(instances, judged, "Task 3 assessment")
    scores: dict[str, list[float]] = defaultdict(list)
    future_leaks = unknown_hallucinations = stance_correct = 0
    invariant_values: list[bool] = []

    for instance_id, instance in instances.items():
        assessment = judged[instance_id]
        _exact_keys(
            assessment,
            {
                "instance_id",
                "scores",
                "leaked_future_fact_ids",
                "hallucinated_unknown_fact_ids",
                "stance_correct",
                "invariant_preserved",
            },
            "Task 3 assessment",
        )
        raw_scores = assessment["scores"]
        if not isinstance(raw_scores, dict) or set(raw_scores) != TASK3_SCORE_DIMENSIONS:
            raise ValueError(f"Task 3 assessment has invalid score dimensions: {instance_id}")
        for key, value in raw_scores.items():
            number = float(value)
            if not 1.0 <= number <= 5.0:
                raise ValueError(f"Task 3 score must be in [1,5]: {instance_id} / {key}")
            scores[key].append(number)
        reference = instance["evaluator_reference"]
        leaked = _unique_strings(
            assessment["leaked_future_fact_ids"], "leaked future facts"
        )
        hallucinated = _unique_strings(
            assessment["hallucinated_unknown_fact_ids"], "hallucinated unknown facts"
        )
        if not set(leaked) <= set(reference["future_forbidden_fact_ids"]):
            raise ValueError(f"Task 3 assessment cites a non-future leak: {instance_id}")
        if not set(hallucinated) <= set(reference["unknown_fact_ids"]):
            raise ValueError(f"Task 3 assessment cites a non-unknown fact: {instance_id}")
        future_leaks += int(bool(leaked))
        unknown_hallucinations += int(bool(hallucinated))
        stance_correct += int(_boolean(assessment["stance_correct"], "stance_correct"))
        invariant = _optional_boolean(
            assessment["invariant_preserved"], "invariant_preserved"
        )
        if invariant is not None:
            invariant_values.append(invariant)

    pair_reports: dict[str, list[bool]] = defaultdict(list)
    seen_pair_ids: set[str] = set()
    for pair in pair_assessments or []:
        _exact_keys(
            pair,
            {"pair_group_id", "pair_type", "instance_ids", "correct"},
            "Task 3 pair assessment",
        )
        pair_id = str(pair["pair_group_id"] or "")
        pair_type = str(pair["pair_type"] or "")
        member_ids = _unique_strings(pair["instance_ids"], "pair instance ids")
        if not pair_id or pair_id in seen_pair_ids or pair_type not in TASK3_PAIR_TYPES:
            raise ValueError(f"Task 3 pair has invalid ID or type: {pair_id}")
        if len(member_ids) < 2 or not set(member_ids) <= set(instances):
            raise ValueError(f"Task 3 pair has invalid members: {pair_id}")
        if any(
            instances[instance_id]["evaluator_reference"]["paired_prompt_group_id"]
            != pair_id
            for instance_id in member_ids
        ):
            raise ValueError(f"Task 3 pair members disagree with frozen references: {pair_id}")
        seen_pair_ids.add(pair_id)
        pair_reports[pair_type].append(_boolean(pair["correct"], "pair correct"))

    return {
        "schema_version": "stage_task3_single_turn_metrics_v1",
        "task": task.get("task"),
        "instance_count": len(instances),
        "headline": {
            key: _round(fmean(values)) for key, values in sorted(scores.items())
        },
        "diagnostics": {
            "future_information_leakage_rate": _ratio(future_leaks, len(instances)),
            "unknown_fact_hallucination_rate": _ratio(
                unknown_hallucinations, len(instances)
            ),
            "stance_accuracy": _ratio(stance_correct, len(instances)),
            "invariant_preservation": _mean_booleans(invariant_values),
            "paired_accuracy": {
                pair_type: _mean_booleans(pair_reports.get(pair_type, []))
                for pair_type in sorted(TASK3_PAIR_TYPES)
            },
        },
        "denominators": {
            "responses": len(instances),
            "invariant_responses": len(invariant_values),
            "pairs": sum(len(values) for values in pair_reports.values()),
        },
    }


def _index(items: Any, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError(f"{label} collection must be an array")
    output: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        value = str(item.get(key) or "")
        if not value or value in output:
            raise ValueError(f"{label} has missing or duplicate {key}: {value}")
        output[value] = item
    return output


def _require_exact_coverage(
    expected: dict[str, Any], actual: dict[str, Any], label: str
) -> None:
    if set(expected) != set(actual):
        raise ValueError(f"{label} coverage must exactly match task instances")


def _exact_keys(payload: dict[str, Any], keys: set[str], label: str) -> None:
    if set(payload) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")


def _unique_strings(values: Any, label: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{label} must be an array of strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicates")
    return values


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    number = int(value)
    if number < 0 or number != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _optional_boolean(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, label)


def _prf(matched: int, predicted: int, gold: int) -> tuple[float | None, float | None, float | None]:
    precision = _ratio(matched, predicted)
    recall = _ratio(matched, gold)
    if precision is None or recall is None:
        return precision, recall, None
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, _round(2 * precision * recall / (precision + recall))


def _ratio(numerator: int, denominator: int) -> float | None:
    return _round(numerator / denominator) if denominator else None


def _mean_booleans(values: list[bool]) -> float | None:
    return _round(fmean(int(value) for value in values)) if values else None


def _round(value: float) -> float:
    return round(float(value), 6)
