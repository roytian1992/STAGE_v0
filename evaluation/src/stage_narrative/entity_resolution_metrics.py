from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Any, Iterable


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _partition(row: dict[str, Any]) -> tuple[list[set[str]], dict[str, int]]:
    clusters = [set(cluster) for cluster in row.get("clusters") or []]
    if not clusters or any(not cluster for cluster in clusters):
        raise ValueError(f"{row.get('unit_id')}: clusters must be non-empty")
    owner: dict[str, int] = {}
    for index, cluster in enumerate(clusters):
        for mention in cluster:
            if mention in owner:
                raise ValueError(f"{row.get('unit_id')}: duplicate mention {mention}")
            owner[mention] = index
    return clusters, owner


def _pair_set(clusters: Iterable[set[str]]) -> set[tuple[str, str]]:
    output: set[tuple[str, str]] = set()
    for cluster in clusters:
        ordered = sorted(cluster)
        output.update((ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered)))
    return output


def _max_similarity(matrix: list[list[float]]) -> float:
    if not matrix or not matrix[0]:
        return 0.0
    if len(matrix) > len(matrix[0]):
        matrix = [list(row) for row in zip(*matrix)]
    rows = len(matrix)
    columns = len(matrix[0])
    states: dict[int, float] = {0: 0.0}
    for row_index in range(rows):
        updated: dict[int, float] = {}
        for mask, value in states.items():
            for column in range(columns):
                if mask & (1 << column):
                    continue
                next_mask = mask | (1 << column)
                score = value + matrix[row_index][column]
                if score > updated.get(next_mask, float("-inf")):
                    updated[next_mask] = score
        states = updated
    return max(states.values(), default=0.0)


def _aligned_rows(
    gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    gold = {row["unit_id"]: row for row in gold_rows}
    predicted = {row["unit_id"]: row for row in prediction_rows}
    if set(gold) != set(predicted):
        raise ValueError(
            f"Unit coverage mismatch: missing={sorted(set(gold)-set(predicted))}, "
            f"extra={sorted(set(predicted)-set(gold))}"
        )
    output = []
    for unit_id in sorted(gold):
        gold_clusters, gold_owner = _partition(gold[unit_id])
        predicted_clusters, predicted_owner = _partition(predicted[unit_id])
        if set(gold_owner) != set(predicted_owner):
            raise ValueError(f"{unit_id}: gold/predicted mention coverage differs")
        output.append((gold[unit_id], predicted[unit_id]))
    return output


def score_entity_resolution(
    gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    aligned = _aligned_rows(gold_rows, prediction_rows)
    pair_tp = pair_predicted = pair_gold = 0
    muc_p_num = muc_p_den = muc_r_num = muc_r_den = 0.0
    b3_precision_sum = b3_recall_sum = 0.0
    mention_count = 0
    ceaf_similarity = ceaf_predicted = ceaf_gold = 0.0
    movie_ids: set[str] = set()

    for gold_row, predicted_row in aligned:
        movie_ids.add(str(gold_row["movie_id"]))
        gold_clusters, gold_owner = _partition(gold_row)
        predicted_clusters, predicted_owner = _partition(predicted_row)
        gold_pairs = _pair_set(gold_clusters)
        predicted_pairs = _pair_set(predicted_clusters)
        pair_tp += len(gold_pairs & predicted_pairs)
        pair_predicted += len(predicted_pairs)
        pair_gold += len(gold_pairs)

        for cluster in predicted_clusters:
            partitions = {gold_owner[mention] for mention in cluster}
            muc_p_num += len(cluster) - len(partitions)
            muc_p_den += len(cluster) - 1
        for cluster in gold_clusters:
            partitions = {predicted_owner[mention] for mention in cluster}
            muc_r_num += len(cluster) - len(partitions)
            muc_r_den += len(cluster) - 1

        for mention, gold_index in gold_owner.items():
            predicted_index = predicted_owner[mention]
            intersection = len(gold_clusters[gold_index] & predicted_clusters[predicted_index])
            b3_precision_sum += intersection / len(predicted_clusters[predicted_index])
            b3_recall_sum += intersection / len(gold_clusters[gold_index])
            mention_count += 1

        similarity = [
            [
                2 * len(gold_cluster & predicted_cluster)
                / (len(gold_cluster) + len(predicted_cluster))
                for predicted_cluster in predicted_clusters
            ]
            for gold_cluster in gold_clusters
        ]
        ceaf_similarity += _max_similarity(similarity)
        ceaf_gold += len(gold_clusters)
        ceaf_predicted += len(predicted_clusters)

    pair_precision = _ratio(pair_tp, pair_predicted)
    pair_recall = _ratio(pair_tp, pair_gold)
    muc_precision = _ratio(muc_p_num, muc_p_den)
    muc_recall = _ratio(muc_r_num, muc_r_den)
    b3_precision = _ratio(b3_precision_sum, mention_count)
    b3_recall = _ratio(b3_recall_sum, mention_count)
    ceaf_precision = _ratio(ceaf_similarity, ceaf_predicted)
    ceaf_recall = _ratio(ceaf_similarity, ceaf_gold)
    muc_f1 = _f1(muc_precision, muc_recall)
    b3_f1 = _f1(b3_precision, b3_recall)
    ceaf_f1 = _f1(ceaf_precision, ceaf_recall)
    return {
        "unit_count": len(aligned),
        "movie_count": len(movie_ids),
        "mention_count": mention_count,
        "pairwise": {
            "precision": pair_precision,
            "recall": pair_recall,
            "f1": _f1(pair_precision, pair_recall),
            "true_positive_pairs": pair_tp,
            "predicted_positive_pairs": pair_predicted,
            "gold_positive_pairs": pair_gold,
        },
        "muc": {"precision": muc_precision, "recall": muc_recall, "f1": muc_f1},
        "b3": {"precision": b3_precision, "recall": b3_recall, "f1": b3_f1},
        "ceaf_e": {"precision": ceaf_precision, "recall": ceaf_recall, "f1": ceaf_f1},
        "conll_f1": (muc_f1 + b3_f1 + ceaf_f1) / 3,
        "merge_error_count": pair_predicted - pair_tp,
        "split_error_count": pair_gold - pair_tp,
    }


def movie_cluster_bootstrap(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    aligned = _aligned_rows(gold_rows, prediction_rows)
    by_movie: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for gold, predicted in aligned:
        by_movie[str(gold["movie_id"])].append((gold, predicted))
    movies = sorted(by_movie)
    if len(movies) < 2:
        return {"status": "unavailable", "reason": "fewer_than_two_movies"}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled_gold: list[dict[str, Any]] = []
        sampled_prediction: list[dict[str, Any]] = []
        for draw_index in range(len(movies)):
            movie = rng.choice(movies)
            for gold, predicted in by_movie[movie]:
                gold_copy = copy.deepcopy(gold)
                prediction_copy = copy.deepcopy(predicted)
                prefix = f"D{draw_index}:"
                gold_copy["unit_id"] = prefix + gold_copy["unit_id"]
                prediction_copy["unit_id"] = prefix + prediction_copy["unit_id"]
                gold_copy["clusters"] = [[prefix + mention for mention in cluster] for cluster in gold_copy["clusters"]]
                prediction_copy["clusters"] = [[prefix + mention for mention in cluster] for cluster in prediction_copy["clusters"]]
                sampled_gold.append(gold_copy)
                sampled_prediction.append(prediction_copy)
        metrics = score_entity_resolution(sampled_gold, sampled_prediction)
        values["pairwise_f1"].append(metrics["pairwise"]["f1"])
        values["b3_f1"].append(metrics["b3"]["f1"])
        values["ceaf_e_f1"].append(metrics["ceaf_e"]["f1"])
        values["conll_f1"].append(metrics["conll_f1"])

    def interval(samples: list[float]) -> dict[str, float]:
        ordered = sorted(samples)
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        return {"low": low, "high": high}

    return {
        "status": "available",
        "cluster": "movie_id",
        "movie_count": len(movies),
        "replicates": replicates,
        "seed": seed,
        "intervals": {key: interval(samples) for key, samples in values.items()},
    }


def filter_rows(
    rows: list[dict[str, Any]], *, split: str, scope: str
) -> list[dict[str, Any]]:
    if scope not in {"english_character", "full_schema"}:
        raise ValueError(f"Unsupported scope: {scope}")
    return [
        row
        for row in rows
        if row.get("split") == split
        and (scope == "full_schema" or row.get("scope") == "english_character")
    ]
