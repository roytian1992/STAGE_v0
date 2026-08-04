from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..models import clean_text


def build_single_turn_release(
    *,
    released_single: dict[str, Any],
    temporal_single: dict[str, Any],
    gold_rubrics: dict[str, Any],
    role_assets: dict[str, Any],
) -> dict[str, Any]:
    """Attach current temporal references and gold rubrics to released prompts."""
    movie_id = _one_movie_id(
        released_single, temporal_single, gold_rubrics, role_assets
    )
    temporal_by_id = _unique_by(temporal_single.get("instances", []), "instance_id")
    rubric_by_id = _unique_by(gold_rubrics.get("rubrics", []), "instance_id")
    role_by_id = _unique_by(role_assets.get("roles", []), "character_id")
    released_ids = {
        clean_text(item.get("instance_id"))
        for item in released_single.get("instances", [])
    }
    if not released_ids.issubset(temporal_by_id) or released_ids != set(rubric_by_id):
        raise ValueError(
            "Every released single-turn instance requires a temporal evaluator ref and "
            "exactly one gold rubric; extra unpromoted temporal candidates are allowed"
        )

    instances = []
    for released in released_single.get("instances", []):
        instance_id = released["instance_id"]
        temporal = temporal_by_id[instance_id]
        rubric = rubric_by_id[instance_id]
        character_id = clean_text(temporal.get("character_id"))
        if character_id not in role_by_id:
            raise ValueError(
                f"Single-turn character lacks a current role asset: {instance_id}"
            )
        evaluator = deepcopy(temporal.get("evaluator_reference", {}))
        evaluator.update(
            {
                "gold_rubric": deepcopy(rubric["rubric"]),
                "gold_review_status": rubric["review_status"],
                "gold_review_note": rubric.get("review_note"),
            }
        )
        item = {
            key: deepcopy(value)
            for key, value in released.items()
            if key != "actor_context_ref"
        }
        item["character_id"] = character_id
        item["role_asset_ref"] = {
            "asset_file": "task_3_role_assets.json",
            "character_id": character_id,
        }
        item["evaluator_reference"] = evaluator
        instances.append(item)

    return {
        "schema_version": "stage_task3_single_turn",
        "task": "task_3_character_role_play",
        "movie_id": movie_id,
        "language": released_single.get("instances", [{}])[0].get("language"),
        "instance_count": len(instances),
        "instances": instances,
    }


def build_multi_turn_release(
    *, legacy_multi: dict[str, Any], role_assets: dict[str, Any]
) -> dict[str, Any]:
    """Keep reviewed conversations while switching runtime to all new memories."""
    movie_id = _one_movie_id(legacy_multi, role_assets)
    roles = role_assets.get("roles", [])
    episodes = []
    for source_episode in legacy_multi.get("episodes", []):
        role = _role_for_name(source_episode.get("character"), roles)
        if role is None:
            raise ValueError(
                "Multi-turn character lacks a current role asset: "
                f"{source_episode.get('character')}"
            )
        turns = []
        for expected_index, source_turn in enumerate(
            source_episode.get("turns", []), start=1
        ):
            if int(source_turn.get("turn_index", -1)) != expected_index:
                raise ValueError(
                    "Legacy multi-turn indices are not contiguous: "
                    f"{source_episode['episode_id']}"
                )
            evaluator = deepcopy(source_turn.get("reference", {}))
            legacy_memory_ids = evaluator.pop("source_memory_ids", [])
            evaluator["migration_provenance"] = {
                "legacy_source_memory_ids": legacy_memory_ids,
                "runtime_dependency": "none",
            }
            turns.append(
                {
                    "turn_index": expected_index,
                    "question_id": source_turn["question_id"],
                    "model_input": deepcopy(source_turn["input"]),
                    "evaluator_reference": evaluator,
                }
            )
        episodes.append(
            {
                "episode_id": source_episode["episode_id"],
                "episode_theme": source_episode.get("episode_theme"),
                "character": source_episode["character"],
                "character_id": role["character_id"],
                "role_asset_ref": {
                    "asset_file": "task_3_role_assets.json",
                    "character_id": role["character_id"],
                },
                "memory_context_policy": {
                    "mode": "all_role_memories",
                    "retrieval": "none",
                },
                "turns": turns,
            }
        )
    return {
        "schema_version": "stage_task3_multi_turn",
        "task": "task_3_character_role_play",
        "movie_id": movie_id,
        "language": legacy_multi.get("language"),
        "interaction_format": "multi_turn",
        "construction_policy": "legacy_questions_with_current_role_memory",
        "episode_count": len(episodes),
        "turn_count": sum(len(item["turns"]) for item in episodes),
        "episodes": episodes,
    }


def _unique_by(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    output = {}
    for item in items:
        value = clean_text(item.get(key))
        if not value or value in output:
            raise ValueError(f"Missing or duplicate {key}: {value}")
        output[value] = item
    return output


def _one_movie_id(*payloads: dict[str, Any]) -> str:
    movie_ids = {clean_text(item.get("movie_id")) for item in payloads}
    movie_ids.discard("")
    if len(movie_ids) != 1:
        raise ValueError(f"Task 3 inputs disagree on movie_id: {sorted(movie_ids)}")
    return next(iter(movie_ids))


def _role_for_name(
    name: Any, roles: list[dict[str, Any]]
) -> dict[str, Any] | None:
    normalized = _normalize_name(name)
    matches = []
    for role in roles:
        names = {
            _normalize_name(role.get("character_name")),
            *(_normalize_name(item) for item in role.get("aliases", [])),
            *(
                _normalize_name(item.get("name"))
                for item in role.get("identity_phases", [])
                if isinstance(item, dict)
            ),
        }
        if normalized and normalized in names:
            matches.append(role)
    if len(matches) > 1:
        raise ValueError(f"Character matches multiple role assets: {name}")
    return matches[0] if matches else None


def _normalize_name(value: Any) -> str:
    return "".join(
        character for character in clean_text(value).casefold() if character.isalnum()
    )
