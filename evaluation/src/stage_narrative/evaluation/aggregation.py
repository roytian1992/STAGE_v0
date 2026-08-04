from __future__ import annotations

from collections.abc import Iterable


def safe_ratio(numerator: float, denominator: int | float) -> float | None:
    """Return an explicit undefined value for an empty denominator."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None and recall is None:
        return None
    if precision is None or recall is None:
        return 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mean_defined(values: Iterable[float | int | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return safe_ratio(sum(defined), len(defined))
