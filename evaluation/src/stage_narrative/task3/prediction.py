from __future__ import annotations

from copy import deepcopy
from typing import Any

from .memory_visibility import materialize_full_role, materialize_role_at_boundary


def role_index(role_assets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    roles = role_assets.get("roles", [])
    output = {str(item.get("character_id") or ""): item for item in roles}
    if "" in output or len(output) != len(roles):
        raise ValueError("Role assets contain missing or duplicate character IDs")
    return output


def materialize_single_actor_input(
    instance: dict[str, Any], role_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    character_id = str(instance.get("character_id") or "")
    role = role_by_id.get(character_id)
    if role is None:
        raise ValueError(f"Single-turn instance has no role asset: {character_id}")
    model_input = instance["model_input"]
    return {
        "character": instance["character"],
        "role_context": materialize_role_at_boundary(
            role, instance["checkpoint_boundary"]
        ),
        "interaction_context": {
            "setting": model_input.get("interaction_context", ""),
            "dialogue_history": deepcopy(model_input.get("dialogue_history", [])),
        },
        "current_user_turn": model_input["current_user_turn"],
    }


def materialize_multi_actor_input(
    episode: dict[str, Any],
    turn: dict[str, Any],
    role_by_id: dict[str, dict[str, Any]],
    prior_responses: dict[int, str],
) -> dict[str, Any]:
    character_id = str(episode.get("character_id") or "")
    role = role_by_id.get(character_id)
    if role is None:
        raise ValueError(f"Multi-turn episode has no role asset: {character_id}")
    if episode.get("memory_context_policy") != {
        "mode": "all_role_memories",
        "retrieval": "none",
    }:
        raise ValueError(f"Unsupported multi-turn memory policy: {episode.get('episode_id')}")
    model_input = turn["model_input"]
    history = resolve_dialogue_history(
        model_input.get("dialogue_history_template", []), prior_responses
    )
    return {
        "character": episode["character"],
        "role_context": materialize_full_role(role),
        "interaction_context": {
            "episode_theme": episode.get("episode_theme"),
            "dialogue_history": history,
        },
        "current_user_turn": model_input["current_user_turn"],
    }


def resolve_dialogue_history(
    template: list[dict[str, Any]], prior_responses: dict[int, str]
) -> list[dict[str, str]]:
    history = []
    for item in template:
        speaker = str(item.get("speaker") or "").strip()
        if speaker not in {"user", "assistant"}:
            raise ValueError(f"Invalid dialogue-history speaker: {speaker}")
        if item.get("fill_with") == "previous_model_response":
            source_turn = int(item.get("source_turn_index", 0))
            if source_turn not in prior_responses:
                raise ValueError(
                    f"Dialogue history requires missing response from turn {source_turn}"
                )
            text = prior_responses[source_turn]
        else:
            text = str(item.get("text") or "").strip()
        if not text:
            raise ValueError("Dialogue-history entries must contain non-empty text")
        history.append({"speaker": speaker, "text": text})
    return history
