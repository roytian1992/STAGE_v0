from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any, Iterable

from .aggregation import f1, mean_defined, safe_ratio
from .task1_alignment import maximum_weight_one_to_one
from .task1_schemas import (
    validate_task1_checkpoint_judgment,
    validate_task1_prediction,
    validate_task1_trajectory_judgment,
)


PRIMARY_PATHS = {
    "current_state_quality_f1": ("checkpoint", "current_state_quality", "f1"),
    "development_quality_f1": ("checkpoint", "development_quality", "f1"),
    "state_retention_strict_rate": ("state_retention", "strict", "rate"),
}


def localize_task1_prediction(payload: dict[str, Any]) -> list[dict[str, Any]]:
    validate_task1_prediction(payload)
    output = []
    for field, prediction_type in (
        ("current_state", "current_state"),
        ("developments_since_previous_checkpoint", "development"),
    ):
        for row in payload[field]:
            output.append(
                {
                    "local_id": f"P{len(output) + 1}",
                    "prediction_type": prediction_type,
                    "claim": row["claim"].strip(),
                    "evidence_scene_orders": list(row["evidence_scene_orders"]),
                }
            )
    return output


def score_task1_checkpoint(
    *,
    prediction: list[dict[str, Any]],
    rubric: dict[str, Any],
    judgment: dict[str, Any],
    partial_weight: float = 0.5,
) -> dict[str, Any]:
    prediction_ids = {row["local_id"] for row in prediction}
    gold_ids = {
        row["local_id"]
        for field in ("current_state_claims", "development_claims")
        for row in rubric[field]
    }
    inactive_ids = {row["local_id"] for row in rubric["inactive_state_claims"]}
    validate_task1_checkpoint_judgment(
        judgment,
        gold_ids=gold_ids,
        prediction_ids=prediction_ids,
        inactive_state_ids=inactive_ids,
    )
    checks = {row["prediction_local_id"]: row for row in judgment["prediction_checks"]}
    pools = {
        "development": {
            "gold": [row["local_id"] for row in rubric["development_claims"]],
            "prediction": [
                row["local_id"] for row in prediction if row["prediction_type"] == "development"
            ],
        },
        "current_state": {
            "gold": [row["local_id"] for row in rubric["current_state_claims"]],
            "prediction": [
                row["local_id"] for row in prediction if row["prediction_type"] == "current_state"
            ],
        },
    }
    metrics = {}
    alignments = {}
    soft_alignments = {}
    invalid_timing = set(judgment["future_leak_prediction_ids"]) | set(
        judgment["premature_update_prediction_ids"]
    )
    false_persistence_predictions = {
        row["prediction_local_id"] for row in judgment["false_persistence_pairs"]
    }
    for pool_name, pool in pools.items():
        relevant = [
            row
            for row in judgment["claim_pair_judgments"]
            if row["gold_local_id"] in set(pool["gold"])
            and row["prediction_local_id"] in set(pool["prediction"])
        ]
        strict_alignment = maximum_weight_one_to_one(
            pool["gold"], pool["prediction"], relevant, partial_weight=0.0
        )
        soft_alignment = maximum_weight_one_to_one(
            pool["gold"], pool["prediction"], relevant, partial_weight=partial_weight
        )
        alignments[pool_name] = strict_alignment
        soft_alignments[pool_name] = soft_alignment
        for variant, true_positive in (
            ("strict", sum(row["strict_weight"] for row in strict_alignment)),
            ("soft", sum(row["soft_weight"] for row in soft_alignment)),
        ):
            precision = safe_ratio(true_positive, len(pool["prediction"]))
            recall = safe_ratio(true_positive, len(pool["gold"]))
            metrics[f"{pool_name}_{variant}"] = {
                "true_positive": true_positive,
                "prediction_denominator": len(pool["prediction"]),
                "gold_denominator": len(pool["gold"]),
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        metrics[f"{pool_name}_empty_case"] = _empty_case(
            len(pool["gold"]), len(pool["prediction"])
        )
        valid_predictions = sum(
            checks[prediction_id]["support"] == "supported"
            and checks[prediction_id]["evidence_grounded"]
            and checks[prediction_id]["checkpoint_valid"]
            and checks[prediction_id]["salient"]
            and prediction_id not in invalid_timing
            and (
                pool_name != "development"
                or checks[prediction_id]["transition_coherent"]
            )
            and (
                pool_name != "current_state"
                or prediction_id not in false_persistence_predictions
            )
            for prediction_id in pool["prediction"]
        )
        strict_covered_gold = sum(
            row["strict_weight"] for row in strict_alignment
        )
        quality_precision = safe_ratio(valid_predictions, len(pool["prediction"]))
        quality_recall = safe_ratio(strict_covered_gold, len(pool["gold"]))
        metrics[f"{pool_name}_quality"] = {
            "valid_prediction_count": valid_predictions,
            "prediction_denominator": len(pool["prediction"]),
            "covered_gold_count": strict_covered_gold,
            "gold_denominator": len(pool["gold"]),
            "precision": quality_precision,
            "recall": quality_recall,
            "f1": f1(quality_precision, quality_recall),
        }
    scorable = set(pools["development"]["prediction"]) | set(
        pools["current_state"]["prediction"]
    )
    development_predictions = set(pools["development"]["prediction"])
    false_predictions = {
        row["prediction_local_id"] for row in judgment["false_persistence_pairs"]
    }
    diagnostics = {
        "evidence_grounding": _ratio(
            sum(checks[value]["evidence_grounded"] for value in scorable), len(scorable)
        ),
        "unsupported_prediction": _ratio(
            sum(checks[value]["support"] == "unsupported" for value in scorable),
            len(scorable),
        ),
        "transition_coherence": _ratio(
            sum(checks[value]["transition_coherent"] for value in development_predictions),
            len(development_predictions),
        ),
        "future_leakage": _ratio(
            len(set(judgment["future_leak_prediction_ids"]) & scorable), len(scorable)
        ),
        "premature_update": _ratio(
            len(set(judgment["premature_update_prediction_ids"]) & scorable), len(scorable)
        ),
        "false_persistence": _ratio(
            len(false_predictions & set(pools["current_state"]["prediction"])),
            len(pools["current_state"]["prediction"]),
        ),
        "no_change_false_update": {
            "eligible": not pools["development"]["gold"],
            "value": judgment["no_change_false_update"]
            if not pools["development"]["gold"]
            else None,
        },
    }
    return {
        "metrics": metrics,
        "alignments": alignments,
        "soft_alignments": soft_alignments,
        "diagnostics": diagnostics,
        "counts": {
            "gold_current_states": len(pools["current_state"]["gold"]),
            "gold_developments": len(pools["development"]["gold"]),
            "predicted_current_states": len(pools["current_state"]["prediction"]),
            "predicted_developments": len(pools["development"]["prediction"]),
        },
    }


def score_task1_trajectory(
    *,
    trajectory: dict[str, Any],
    checkpoint_results: list[dict[str, Any]],
    trajectory_judgment: dict[str, Any],
    partial_weight: float = 0.5,
) -> dict[str, Any]:
    checkpoint_ids = trajectory["checkpoint_ids"]
    checkpoint_index = {value: index for index, value in enumerate(checkpoint_ids)}
    rubric_by_checkpoint = {
        row["checkpoint_id"]: row for row in trajectory["checkpoint_rubrics"]
    }
    result_by_checkpoint = {row["checkpoint_id"]: row for row in checkpoint_results}
    if set(result_by_checkpoint) != set(checkpoint_ids):
        raise ValueError("Task 1 trajectory result coverage differs from private lineage")
    ordered = [result_by_checkpoint[value] for value in checkpoint_ids]
    development_refs = {
        f"{row['instance_id']}|{prediction['local_id']}"
        for row in ordered
        for prediction in row["localized_prediction"]
        if prediction["prediction_type"] == "development"
    }
    validate_task1_trajectory_judgment(
        trajectory_judgment,
        development_prediction_refs=development_refs,
        adjacent_instance_pairs=[
            (left["instance_id"], right["instance_id"])
            for left, right in zip(ordered, ordered[1:])
        ],
    )

    checkpoint_aggregate = _aggregate_checkpoint_rows(ordered)
    occurrence_to_cluster = {}
    for cluster in trajectory_judgment["development_clusters"]:
        for member in cluster["members"]:
            occurrence_to_cluster[member] = cluster["cluster_id"]
    cluster_ids = sorted(
        {row["cluster_id"] for row in trajectory_judgment["development_clusters"]},
        key=lambda value: int(value[2:]),
    )
    developments = {
        row["stable_development_id"]: row for row in trajectory["developments"]
    }
    states = {row["stable_state_id"]: row for row in trajectory["states"]}

    development_matches: list[dict[str, Any]] = []
    state_coverage: dict[tuple[str, str], float] = defaultdict(float)
    soft_state_coverage: dict[tuple[str, str], float] = defaultdict(float)
    false_persistence: set[tuple[str, str]] = set()
    occurrence_checkpoint: dict[str, int] = {}
    premature_occurrences: set[str] = set()
    for result in ordered:
        checkpoint_id = result["checkpoint_id"]
        rubric = rubric_by_checkpoint[checkpoint_id]
        gold_development = {
            row["local_id"]: row["stable_development_id"]
            for row in rubric["development_claims"]
        }
        gold_state = {
            row["local_id"]: row["stable_state_id"] for row in rubric["current_state_claims"]
        }
        inactive_state = {
            row["local_id"]: row["stable_state_id"] for row in rubric["inactive_state_claims"]
        }
        for prediction in result["localized_prediction"]:
            if prediction["prediction_type"] != "development":
                continue
            ref = f"{result['instance_id']}|{prediction['local_id']}"
            occurrence_checkpoint[ref] = checkpoint_index[checkpoint_id]
            if prediction["local_id"] in result["judgment"]["premature_update_prediction_ids"]:
                premature_occurrences.add(ref)
        for row in result["scoring"]["alignments"]["development"]:
            ref = f"{result['instance_id']}|{row['prediction_local_id']}"
            if ref not in occurrence_to_cluster:
                raise ValueError(f"Task 1 trajectory judgment omits development ref: {ref}")
            development_matches.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_index": checkpoint_index[checkpoint_id],
                    "cluster_id": occurrence_to_cluster[ref],
                    "stable_development_id": gold_development[row["gold_local_id"]],
                    "label": row["label"],
                    "strict_weight": row["strict_weight"],
                    "soft_weight": row["soft_weight"],
                }
            )
        for row in result["scoring"]["alignments"]["current_state"]:
            state_id = gold_state[row["gold_local_id"]]
            key = (checkpoint_id, state_id)
            state_coverage[key] = max(state_coverage[key], row["soft_weight"])
        for row in result["scoring"]["soft_alignments"]["current_state"]:
            state_id = gold_state[row["gold_local_id"]]
            key = (checkpoint_id, state_id)
            soft_state_coverage[key] = max(
                soft_state_coverage[key], row["soft_weight"]
            )
        for pair in result["judgment"]["false_persistence_pairs"]:
            false_persistence.add(
                (checkpoint_id, inactive_state[pair["inactive_state_local_id"]])
            )

    strict_gold_by_cluster: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[tuple[str, str], str] = {}
    rank = {"partial": 1, "full": 2}
    for match in development_matches:
        if match["label"] == "full":
            strict_gold_by_cluster[match["cluster_id"]].add(match["stable_development_id"])
        key = (match["stable_development_id"], match["cluster_id"])
        previous = pair_labels.get(key)
        if previous is None or rank[match["label"]] > rank[previous]:
            pair_labels[key] = match["label"]
    ambiguous_clusters = {
        cluster_id for cluster_id, gold_ids in strict_gold_by_cluster.items() if len(gold_ids) > 1
    }
    unique_pairs = [
        {"gold_local_id": gold_id, "prediction_local_id": cluster_id, "label": label}
        for (gold_id, cluster_id), label in sorted(pair_labels.items())
        if cluster_id not in ambiguous_clusters
    ]
    unique_alignment = maximum_weight_one_to_one(
        developments,
        cluster_ids,
        unique_pairs,
        partial_weight=0.0,
    )
    soft_unique_alignment = maximum_weight_one_to_one(
        developments,
        cluster_ids,
        unique_pairs,
        partial_weight=partial_weight,
    )
    unique_metrics = {}
    for variant, true_positive in (
        ("strict", sum(row["strict_weight"] for row in unique_alignment)),
        ("soft", sum(row["soft_weight"] for row in soft_unique_alignment)),
    ):
        precision = safe_ratio(true_positive, len(cluster_ids))
        recall = safe_ratio(true_positive, len(developments))
        unique_metrics[variant] = {
            "true_positive": true_positive,
            "prediction_denominator": len(cluster_ids),
            "gold_denominator": len(developments),
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        }

    strict_credit = {
        row["gold_local_id"]: row["prediction_local_id"]
        for row in unique_alignment
        if row["label"] == "full"
    }
    placement_rows = []
    for development_id, development in sorted(developments.items()):
        effective_index = checkpoint_index[development["effective_checkpoint_id"]]
        cluster_id = strict_credit.get(development_id)
        if cluster_id is not None:
            occurrences = [
                row
                for row in development_matches
                if row["cluster_id"] == cluster_id
                and row["stable_development_id"] == development_id
                and row["label"] == "full"
            ]
            premature_members = [
                occurrence_checkpoint[ref]
                for ref, observed_cluster in occurrence_to_cluster.items()
                if observed_cluster == cluster_id and ref in premature_occurrences
            ]
            earliest = (
                min(premature_members)
                if premature_members
                else min(row["checkpoint_index"] for row in occurrences)
            )
            lag = earliest - effective_index
            status = "on_time" if lag == 0 else "premature" if lag < 0 else "delayed"
        else:
            ambiguous = any(
                development_id in gold_ids
                for cluster_id, gold_ids in strict_gold_by_cluster.items()
                if cluster_id in ambiguous_clusters
            )
            status = "ambiguous" if ambiguous else "missed"
            lag = None
            earliest = None
        placement_rows.append(
            {
                "stable_development_id": development_id,
                "effective_checkpoint_id": development["effective_checkpoint_id"],
                "credited_cluster_id": cluster_id,
                "earliest_matched_checkpoint_id": checkpoint_ids[earliest]
                if earliest is not None
                else None,
                "status": status,
                "checkpoint_lag": lag,
            }
        )
    placement = {
        name: _ratio(sum(row["status"] == name for row in placement_rows), len(placement_rows))
        for name in ("on_time", "premature", "delayed", "missed", "ambiguous")
    }
    lags = [row["checkpoint_lag"] for row in placement_rows if row["status"] == "delayed"]
    placement.update(
        {
            "mean_delayed_checkpoint_lag": statistics.mean(lags) if lags else None,
            "median_delayed_checkpoint_lag": statistics.median(lags) if lags else None,
            "items": placement_rows,
        }
    )

    explicit_states_by_checkpoint = {
        checkpoint_id: {
            row["stable_state_id"]
            for row in rubric_by_checkpoint[checkpoint_id]["current_state_claims"]
        }
        for checkpoint_id in checkpoint_ids
    }
    obligations = [
        (checkpoint_id, state_id)
        for checkpoint_id in checkpoint_ids
        for state_id in sorted(explicit_states_by_checkpoint[checkpoint_id])
    ]
    strict_covered = sum(state_coverage[key] == 1.0 for key in obligations)
    soft_covered = sum(soft_state_coverage[key] for key in obligations)
    persistence = {
        "strict": {**_ratio(strict_covered, len(obligations)), "covered": strict_covered},
        "soft": {**_ratio(soft_covered, len(obligations)), "covered": soft_covered},
        "obligation_count": len(obligations),
        "false_persistence": _ratio(
            len(false_persistence),
            sum(row["scoring"]["counts"]["predicted_current_states"] for row in ordered),
        ),
    }
    retention_items = []
    possible_pair_count = 0
    for earlier_id, later_id in zip(checkpoint_ids, checkpoint_ids[1:]):
        shared_states = sorted(
            explicit_states_by_checkpoint[earlier_id]
            & explicit_states_by_checkpoint[later_id]
        )
        possible_pair_count += len(shared_states)
        for state_id in shared_states:
            captured_earlier = state_coverage[(earlier_id, state_id)] == 1.0
            retained_later = state_coverage[(later_id, state_id)] == 1.0
            retention_items.append(
                {
                    "stable_state_id": state_id,
                    "earlier_checkpoint_id": earlier_id,
                    "later_checkpoint_id": later_id,
                    "eligible": captured_earlier,
                    "retained": retained_later if captured_earlier else None,
                }
            )
    eligible_retention = [row for row in retention_items if row["eligible"]]
    retained_count = sum(row["retained"] for row in eligible_retention)
    state_retention = {
        "strict": {
            **_ratio(retained_count, len(eligible_retention)),
            "retained": retained_count,
        },
        "possible_pair_count": possible_pair_count,
        "eligible_pair_count": len(eligible_retention),
        "items": retention_items,
    }

    placement_by_id = {row["stable_development_id"]: row for row in placement_rows}
    transition_rows = []
    for development_id, development in sorted(developments.items()):
        lineage_reviewed = bool(development["affected_state_ids"]) and bool(
            development["resulting_state_ids"] or development["superseded_state_ids"]
        )
        if not lineage_reviewed:
            transition_rows.append(
                {
                    "stable_development_id": development_id,
                    "eligible": False,
                    "correct": None,
                    "reason": "missing_reviewed_affected_or_resulting_lineage",
                }
            )
            continue
        effective = development["effective_checkpoint_id"]
        resulting_covered = all(
            state_coverage[(effective, state_id)] == 1.0
            for state_id in development["resulting_state_ids"]
        )
        no_superseded_persistence = all(
            not any(
                state_id == superseded
                and checkpoint_index[checkpoint_id] >= checkpoint_index[effective]
                for checkpoint_id, state_id in false_persistence
            )
            for superseded in development["superseded_state_ids"]
        )
        correct = bool(
            placement_by_id[development_id]["status"] == "on_time"
            and resulting_covered
            and no_superseded_persistence
        )
        transition_rows.append(
            {
                "stable_development_id": development_id,
                "eligible": True,
                "correct": correct,
                "on_time": placement_by_id[development_id]["status"] == "on_time",
                "resulting_states_covered": resulting_covered,
                "superseded_states_not_persisted": no_superseded_persistence,
            }
        )
    eligible_transitions = [row for row in transition_rows if row["eligible"]]
    transition = {
        "correct": sum(row["correct"] for row in eligible_transitions),
        "eligible": len(eligible_transitions),
        "accuracy": safe_ratio(
            sum(row["correct"] for row in eligible_transitions), len(eligible_transitions)
        ),
        "ineligible": len(transition_rows) - len(eligible_transitions),
        "items": transition_rows,
    }

    final_checkpoint = checkpoint_ids[-1]
    final_states = sorted(explicit_states_by_checkpoint[final_checkpoint])
    final_strict = sum(state_coverage[(final_checkpoint, state_id)] == 1.0 for state_id in final_states)
    final_soft = sum(
        soft_state_coverage[(final_checkpoint, state_id)] for state_id in final_states
    )
    final_coverage = {
        "strict": {
            "covered": final_strict,
            "gold_denominator": len(final_states),
            "recall": safe_ratio(final_strict, len(final_states)),
        },
        "soft": {
            "covered": final_soft,
            "gold_denominator": len(final_states),
            "recall": safe_ratio(final_soft, len(final_states)),
        },
    }

    adjacent = trajectory_judgment["adjacent_checks"]
    consistent = sum(
        row["state_carry_forward"]
        and row["development_to_state_coherent"]
        and not row["contradiction_present"]
        and not row["premature_or_future_information"]
        for row in adjacent
    )
    diagnostics = {
        **_aggregate_diagnostics(ordered),
        "ambiguous_development_cluster_count": len(ambiguous_clusters),
        "longitudinal_consistency": _ratio(consistent, len(adjacent)),
    }
    return {
        "movie_id": checkpoint_results[0]["movie_id"],
        "character_id": trajectory["character_id"],
        "character": trajectory["character"],
        "checkpoint_count": len(checkpoint_ids),
        "checkpoint": checkpoint_aggregate,
        "unique_development": unique_metrics,
        "temporal_placement": placement,
        "state_persistence": persistence,
        "state_retention": state_retention,
        "gold_anchored_transition": transition,
        "final_state_coverage": final_coverage,
        "diagnostics": diagnostics,
        "unique_development_alignment": unique_alignment,
        "soft_unique_development_alignment": soft_unique_alignment,
    }


def aggregate_task1(
    trajectory_results: list[dict[str, Any]],
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260727,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise ValueError("Task 1 bootstrap replicate count must be positive")
    by_movie: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_results:
        by_movie[row["movie_id"]].append(row)
    character_macro = {
        key: mean_defined(_path(row, path) for row in trajectory_results)
        for key, path in PRIMARY_PATHS.items()
    }
    character_macro_valid = {
        key: sum(_path(row, path) is not None for row in trajectory_results)
        for key, path in PRIMARY_PATHS.items()
    }
    movie_values = {
        movie_id: {
            key: mean_defined(_path(row, path) for row in rows)
            for key, path in PRIMARY_PATHS.items()
        }
        for movie_id, rows in sorted(by_movie.items())
    }
    movie_macro = {
        key: mean_defined(values[key] for values in movie_values.values())
        for key in PRIMARY_PATHS
    }
    bootstrap = _movie_cluster_bootstrap(
        by_movie,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "aggregation_order": "checkpoint within character; movie macro primary; character macro secondary",
        "trajectory_count": len(trajectory_results),
        "movie_count": len(by_movie),
        "character_macro": character_macro,
        "character_macro_valid_count": character_macro_valid,
        "movie_values": movie_values,
        "movie_macro": movie_macro,
        "movie_cluster_bootstrap": bootstrap,
        "trajectories": trajectory_results,
    }


def _aggregate_checkpoint_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for pool in ("development", "current_state"):
        for variant in ("strict", "soft"):
            parts = [row["scoring"]["metrics"][f"{pool}_{variant}"] for row in rows]
            true_positive = sum(part["true_positive"] for part in parts)
            prediction_denominator = sum(part["prediction_denominator"] for part in parts)
            gold_denominator = sum(part["gold_denominator"] for part in parts)
            precision = safe_ratio(true_positive, prediction_denominator)
            recall = safe_ratio(true_positive, gold_denominator)
            output[f"{pool}_{variant}"] = {
                "true_positive": true_positive,
                "prediction_denominator": prediction_denominator,
                "gold_denominator": gold_denominator,
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        quality_parts = [
            row["scoring"]["metrics"][f"{pool}_quality"] for row in rows
        ]
        valid_predictions = sum(part["valid_prediction_count"] for part in quality_parts)
        prediction_denominator = sum(
            part["prediction_denominator"] for part in quality_parts
        )
        covered_gold = sum(part["covered_gold_count"] for part in quality_parts)
        gold_denominator = sum(part["gold_denominator"] for part in quality_parts)
        precision = safe_ratio(valid_predictions, prediction_denominator)
        recall = safe_ratio(covered_gold, gold_denominator)
        output[f"{pool}_quality"] = {
            "valid_prediction_count": valid_predictions,
            "prediction_denominator": prediction_denominator,
            "covered_gold_count": covered_gold,
            "gold_denominator": gold_denominator,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        }
    return output


def _aggregate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for name in (
        "evidence_grounding",
        "unsupported_prediction",
        "transition_coherence",
        "future_leakage",
        "premature_update",
        "false_persistence",
    ):
        parts = [row["scoring"]["diagnostics"][name] for row in rows]
        output[name] = _ratio(
            sum(part["numerator"] for part in parts),
            sum(part["denominator"] for part in parts),
        )
    no_change = [
        row["scoring"]["diagnostics"]["no_change_false_update"]
        for row in rows
        if row["scoring"]["diagnostics"]["no_change_false_update"]["eligible"]
    ]
    output["no_change_false_update"] = _ratio(
        sum(row["value"] for row in no_change), len(no_change)
    )
    return output


def _movie_cluster_bootstrap(
    by_movie: dict[str, list[dict[str, Any]]], *, replicates: int, seed: int
) -> dict[str, Any]:
    movie_ids = sorted(by_movie)
    if len(movie_ids) < 2:
        return {
            "status": "unavailable",
            "reason": "requires_at_least_two_movie_clusters",
            "movie_count": len(movie_ids),
            "replicates": replicates,
            "seed": seed,
            "intervals": {key: None for key in PRIMARY_PATHS},
        }
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        selected = [rng.choice(movie_ids) for _ in movie_ids]
        for key, path in PRIMARY_PATHS.items():
            value = mean_defined(
                mean_defined(_path(row, path) for row in by_movie[movie_id])
                for movie_id in selected
            )
            if value is not None:
                samples[key].append(value)
    return {
        "status": "available",
        "movie_count": len(movie_ids),
        "replicates": replicates,
        "seed": seed,
        "intervals": {
            key: {
                "lower_2_5": _percentile(values, 0.025),
                "upper_97_5": _percentile(values, 0.975),
                "valid_replicates": len(values),
            }
            for key, values in samples.items()
        },
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _path(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def _ratio(numerator: int | float, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": safe_ratio(numerator, denominator),
    }


def _empty_case(gold_count: int, prediction_count: int) -> str:
    if gold_count == 0 and prediction_count == 0:
        return "both_empty"
    if gold_count == 0:
        return "gold_empty_prediction_nonempty"
    if prediction_count == 0:
        return "gold_nonempty_prediction_empty"
    return "both_nonempty"
