from __future__ import annotations

from collections import defaultdict
from typing import Any

from .aggregation import f1, mean_defined, safe_ratio
from .task1_alignment import maximum_weight_one_to_one


def score_task1_instance(
    *,
    prediction: list[dict[str, Any]],
    rubric: dict[str, Any],
    judgment: dict[str, Any],
    partial_weight: float = 0.5,
) -> dict[str, Any]:
    pools = {
        "development": {
            "gold": [row["local_id"] for row in rubric["development_claims"]],
            "prediction": [
                row["local_id"]
                for row in prediction
                if row["prediction_type"] == "development"
            ],
        },
        "current_state": {
            "gold": [
                row["local_id"]
                for field in ("current_state_claims", "invariant_claims")
                for row in rubric[field]
            ],
            "prediction": [
                row["local_id"]
                for row in prediction
                if row["prediction_type"] == "current_state"
            ],
        },
    }
    metrics = {}
    alignments = {}
    for name, pool in pools.items():
        relevant = [
            row
            for row in judgment["claim_pair_judgments"]
            if row["gold_local_id"] in set(pool["gold"])
            and row["prediction_local_id"] in set(pool["prediction"])
        ]
        alignment = maximum_weight_one_to_one(
            pool["gold"], pool["prediction"], relevant, partial_weight=partial_weight
        )
        alignments[name] = alignment
        for variant, true_positive in (
            ("strict", sum(row["strict_weight"] for row in alignment)),
            ("soft", sum(row["soft_weight"] for row in alignment)),
        ):
            precision = safe_ratio(true_positive, len(pool["prediction"]))
            recall = safe_ratio(true_positive, len(pool["gold"]))
            metrics[f"{name}_{variant}"] = {
                "true_positive": true_positive,
                "prediction_denominator": len(pool["prediction"]),
                "gold_denominator": len(pool["gold"]),
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        metrics[f"{name}_empty_case"] = _empty_case(
            len(pool["gold"]), len(pool["prediction"])
        )
    scorable = set(pools["development"]["prediction"]) | set(
        pools["current_state"]["prediction"]
    )
    checks = {row["prediction_local_id"]: row for row in judgment["prediction_checks"]}
    developments = set(pools["development"]["prediction"])
    diagnostics = {
        "evidence_grounding": _rate(
            sum(checks[value]["evidence_grounded"] for value in scorable), len(scorable)
        ),
        "unsupported_prediction": _rate(
            sum(checks[value]["support"] == "unsupported" for value in scorable),
            len(scorable),
        ),
        "transition_coherence": _rate(
            sum(checks[value]["transition_coherent"] for value in developments),
            len(developments),
        ),
        "future_leakage": _rate(
            len(set(judgment["future_leak_prediction_ids"]) & scorable), len(scorable)
        ),
        "premature_update": _rate(
            len(set(judgment["premature_update_prediction_ids"]) & scorable), len(scorable)
        ),
        "no_change_false_update": {
            "eligible": not pools["development"]["gold"],
            "value": judgment["no_change_false_update"]
            if not pools["development"]["gold"]
            else None,
        },
        "contradiction_pair_count": sum(
            row["label"] == "contradiction" for row in judgment["claim_pair_judgments"]
        ),
        "unresolved_thread_prediction_count": sum(
            row["prediction_type"] == "unresolved_thread" for row in prediction
        ),
    }
    return {
        "metrics": metrics,
        "alignments": alignments,
        "diagnostics": diagnostics,
        "counts": {
            "gold_claims": sum(len(pool["gold"]) for pool in pools.values()),
            "predicted_claims": len(prediction),
            "scorable_predicted_claims": len(scorable),
        },
    }


def aggregate_checkpoint_task1(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_character[row["character_id"]].append(row)
    characters = []
    for character_id, rows in sorted(by_character.items()):
        metrics = {}
        for pool in ("development", "current_state"):
            for variant in ("strict", "soft"):
                parts = [row["scoring"]["metrics"][f"{pool}_{variant}"] for row in rows]
                true_positive = sum(part["true_positive"] for part in parts)
                prediction_denominator = sum(part["prediction_denominator"] for part in parts)
                gold_denominator = sum(part["gold_denominator"] for part in parts)
                precision = safe_ratio(true_positive, prediction_denominator)
                recall = safe_ratio(true_positive, gold_denominator)
                metrics[f"{pool}_{variant}"] = {
                    "true_positive": true_positive,
                    "prediction_denominator": prediction_denominator,
                    "gold_denominator": gold_denominator,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1(precision, recall),
                }
        diagnostics = {}
        for name in (
            "evidence_grounding",
            "unsupported_prediction",
            "transition_coherence",
            "future_leakage",
            "premature_update",
        ):
            parts = [row["scoring"]["diagnostics"][name] for row in rows]
            numerator = sum(part["numerator"] for part in parts)
            denominator = sum(part["denominator"] for part in parts)
            diagnostics[name] = _rate(numerator, denominator)
        no_change = [
            row["scoring"]["diagnostics"]["no_change_false_update"]
            for row in rows
            if row["scoring"]["diagnostics"]["no_change_false_update"]["eligible"]
        ]
        diagnostics["no_change_false_update"] = _rate(
            sum(bool(row["value"]) for row in no_change), len(no_change)
        )
        characters.append(
            {
                "character_id": character_id,
                "character": rows[0]["character"],
                "checkpoint_count": len(rows),
                "metrics": metrics,
                "diagnostics": diagnostics,
            }
        )
    movie_metrics = {}
    for name in (
        "development_strict",
        "development_soft",
        "current_state_strict",
        "current_state_soft",
    ):
        movie_metrics[name] = {
            field: mean_defined(row["metrics"][name][field] for row in characters)
            for field in ("precision", "recall", "f1")
        }
        movie_metrics[name]["valid_character_count"] = {
            field: sum(row["metrics"][name][field] is not None for row in characters)
            for field in ("precision", "recall", "f1")
        }
    movie_diagnostics = {
        name: {
            "rate": mean_defined(row["diagnostics"][name]["rate"] for row in characters),
            "valid_character_count": sum(
                row["diagnostics"][name]["rate"] is not None for row in characters
            ),
        }
        for name in characters[0]["diagnostics"] if characters
    }
    return {
        "aggregation_order": (
            "checkpoint numerators/denominators within character, then movie macro over characters"
        ),
        "character_count": len(characters),
        "characters": characters,
        "movie_macro": {"metrics": movie_metrics, "diagnostics": movie_diagnostics},
    }


def aggregate_task1_sequences(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_character[row["character_id"]].append(row)
    characters = []
    for character_id, rows in sorted(by_character.items()):
        consistent = sum(bool(row["consistent"]) for row in rows)
        characters.append(
            {
                "character_id": character_id,
                "character": rows[0]["character"],
                "consistent": consistent,
                "denominator": len(rows),
                "rate": safe_ratio(consistent, len(rows)),
            }
        )
    return {
        "characters": characters,
        "movie_macro_rate": mean_defined(row["rate"] for row in characters),
        "valid_character_count": sum(row["rate"] is not None for row in characters),
    }


def _rate(numerator: float, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": safe_ratio(numerator, denominator),
    }


def _empty_case(gold: int, prediction: int) -> str:
    if gold and prediction:
        return "both_nonempty"
    if gold:
        return "gold_nonempty_prediction_empty"
    if prediction:
        return "gold_empty_prediction_nonempty"
    return "both_empty"
