from __future__ import annotations

from collections import defaultdict
from typing import Any

from .aggregation import mean_defined, safe_ratio
from .schemas import PAIR_TYPES, TASK3_SCORE_FIELDS


def aggregate_task3(
    response_results: list[dict[str, Any]], pair_results: list[dict[str, Any]]
) -> dict[str, Any]:
    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in response_results:
        by_character[str(result["character_id"])].append(result)
    characters = []
    for character_id in sorted(by_character):
        rows = by_character[character_id]
        judgments = [row["judgment"] for row in rows]
        characters.append(
            {
                "character_id": character_id,
                "character": rows[0].get("character"),
                "response_count": len(rows),
                "scores": {
                    field: mean_defined(item["scores"][field] for item in judgments)
                    for field in TASK3_SCORE_FIELDS
                },
                "violations": {
                    "future_leakage_rate": safe_ratio(
                        sum(item["future_leakage"] for item in judgments), len(judgments)
                    ),
                    "unknown_fact_hallucination_rate": safe_ratio(
                        sum(item["unknown_fact_hallucination"] for item in judgments),
                        len(judgments),
                    ),
                    "stance_incompatibility_rate": safe_ratio(
                        sum(not item["stance_compatible"] for item in judgments),
                        len(judgments),
                    ),
                },
            }
        )
    movie_scores = {
        field: mean_defined(item["scores"][field] for item in characters)
        for field in TASK3_SCORE_FIELDS
    }
    movie_violations = {
        field: mean_defined(item["violations"][field] for item in characters)
        for field in (
            "future_leakage_rate",
            "unknown_fact_hallucination_rate",
            "stance_incompatibility_rate",
        )
    }
    pair_types: dict[str, Any] = {}
    for pair_type in sorted(PAIR_TYPES):
        rows = [row for row in pair_results if row["judgment"]["pair_type"] == pair_type]
        correct = [pair_is_correct(row) for row in rows]
        pair_types[pair_type] = {
            "correct": sum(correct),
            "denominator": len(correct),
            "accuracy": safe_ratio(sum(correct), len(correct)),
        }
    overall_correct = [pair_is_correct(row) for row in pair_results]
    return {
        "aggregation_order": "responses within character, then movie macro over characters",
        "character_count": len(characters),
        "characters": characters,
        "movie_macro": {
            "scores": movie_scores,
            "violations": movie_violations,
        },
        "longitudinal_pairs": {
            "by_type": pair_types,
            "overall": {
                "correct": sum(overall_correct),
                "denominator": len(overall_correct),
                "accuracy": safe_ratio(sum(overall_correct), len(overall_correct)),
            },
        },
    }


def pair_is_correct(pair_result: dict[str, Any]) -> bool:
    judgment = pair_result["judgment"]
    return bool(
        all(row["passed"] for row in pair_result["response_prerequisites"])
        and all(row["supports_expected_component"] for row in judgment["response_assessments"])
        and judgment["expected_direction_present"]
        and not judgment["unsupported_drift"]
        and judgment["knowledge_boundaries_preserved"]
    )
