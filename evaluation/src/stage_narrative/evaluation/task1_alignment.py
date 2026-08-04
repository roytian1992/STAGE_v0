from __future__ import annotations

from collections.abc import Iterable
from typing import Any


LABEL_WEIGHTS = {
    "full": 1.0,
    "partial": 0.5,
    "none": 0.0,
    "contradiction": 0.0,
}


def maximum_weight_one_to_one(
    gold_ids: Iterable[str],
    prediction_ids: Iterable[str],
    pair_judgments: Iterable[dict[str, Any]],
    *,
    partial_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Find a deterministic maximum-weight bipartite claim alignment.

    The Hungarian implementation adds one zero-weight dummy prediction per gold
    claim, so a gold claim is never forced into a zero/contradictory match.
    """
    gold = sorted(set(gold_ids))
    predictions = sorted(set(prediction_ids))
    if not 0.0 <= partial_weight <= 1.0:
        raise ValueError("partial_weight must be in [0, 1]")
    known_gold = set(gold)
    known_predictions = set(predictions)
    by_pair: dict[tuple[str, str], str] = {}
    rank = {"none": 0, "contradiction": 1, "partial": 2, "full": 3}
    for row in pair_judgments:
        gold_id = str(row.get("gold_local_id", ""))
        prediction_id = str(row.get("prediction_local_id", ""))
        label = str(row.get("label", ""))
        if gold_id not in known_gold or prediction_id not in known_predictions:
            raise ValueError(f"Unknown Task 1 claim pair: {gold_id}/{prediction_id}")
        if label not in LABEL_WEIGHTS:
            raise ValueError(f"Invalid Task 1 pair label: {label}")
        key = (gold_id, prediction_id)
        previous = by_pair.get(key)
        if previous is None or rank[label] < rank[previous]:
            by_pair[key] = label
    if not gold or not predictions:
        return []

    # Integer weights preserve exact ordering; a tiny deterministic tie cost
    # selects stable IDs without changing the full/partial optimum.
    column_count = len(predictions) + len(gold)
    costs: list[list[int]] = []
    for row_index, gold_id in enumerate(gold):
        row = []
        for column_index, prediction_id in enumerate(predictions):
            label = by_pair.get((gold_id, prediction_id), "none")
            weight = 1.0 if label == "full" else partial_weight if label == "partial" else 0.0
            row.append(-int(round(weight * 10000)) + column_index)
        row.extend(range(len(predictions), column_count))
        costs.append(row)
    assignment = _hungarian_minimize(costs)
    output = []
    for row_index, column_index in enumerate(assignment):
        if column_index >= len(predictions):
            continue
        gold_id = gold[row_index]
        prediction_id = predictions[column_index]
        label = by_pair.get((gold_id, prediction_id), "none")
        if label not in {"full", "partial"}:
            continue
        output.append(
            {
                "gold_local_id": gold_id,
                "prediction_local_id": prediction_id,
                "label": label,
                "strict_weight": 1.0 if label == "full" else 0.0,
                "soft_weight": 1.0 if label == "full" else partial_weight,
            }
        )
    return output


def _hungarian_minimize(cost: list[list[int]]) -> list[int]:
    """Rectangular Hungarian algorithm for row_count <= column_count."""
    row_count = len(cost)
    if row_count == 0:
        return []
    column_count = len(cost[0])
    if any(len(row) != column_count for row in cost):
        raise ValueError("Hungarian cost matrix must be rectangular")
    if row_count > column_count:
        raise ValueError("Hungarian solver requires rows <= columns")
    u = [0] * (row_count + 1)
    v = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        column0 = 0
        minimum = [10**18] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column0] = True
            current_row = matched_row[column0]
            delta = 10**18
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = cost[current_row - 1][column - 1] - u[current_row] - v[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return assignment
