from __future__ import annotations

from copy import deepcopy
from typing import Any


def boundary_key(boundary: dict[str, Any]) -> tuple[int, int]:
    return int(boundary["scene_order"]), int(boundary.get("char_end", 0))


def is_visible(available_from: dict[str, Any], boundary: dict[str, Any]) -> bool:
    return boundary_key(available_from) <= boundary_key(boundary)


def materialize_role_at_boundary(
    role: dict[str, Any], boundary: dict[str, Any]
) -> dict[str, Any]:
    memories = [
        {
            "memory_text": item["memory_text"],
            "importance": item["importance"],
        }
        for item in role.get("memories", [])
        if is_visible(item["available_from"], boundary)
    ]
    scene_order = int(boundary["scene_order"])
    active_phases = [
        item
        for item in role.get("identity_phases", [])
        if int(item.get("valid_from_scene", 0)) <= scene_order
        and (
            item.get("valid_until_scene") is None
            or scene_order <= int(item["valid_until_scene"])
        )
    ]
    identity = max(
        active_phases,
        key=lambda item: int(item.get("valid_from_scene", 0)),
        default=None,
    )
    actor_name = identity.get("name") if identity else role["character_name"]
    actor_aliases = (
        [identity.get("name"), *identity.get("aliases", [])]
        if identity
        else role.get("aliases", [])
    )
    return {
        "character_id": role["character_id"],
        "character_name": actor_name,
        "aliases": list(dict.fromkeys(item for item in actor_aliases if item)),
        "identity": (
            {"name": identity.get("name"), "aliases": identity.get("aliases", [])}
            if identity
            else None
        ),
        "persona_card": _actor_persona(role),
        "memories": memories,
    }


def materialize_full_role(role: dict[str, Any]) -> dict[str, Any]:
    """Materialize safe persona and every memory for one multi-turn episode."""
    return {
        "character_id": role["character_id"],
        "character_name": role["character_name"],
        "aliases": deepcopy(role.get("aliases", [])),
        "identity_phases": [
            {
                "name": item.get("name"),
                "aliases": deepcopy(item.get("aliases", [])),
            }
            for item in role.get("identity_phases", [])
        ],
        "persona_card": _actor_persona(role),
        "memories": [
            {
                "memory_text": item["memory_text"],
                "importance": item["importance"],
            }
            for item in role.get("memories", [])
        ],
    }


def _actor_persona(role: dict[str, Any]) -> dict[str, Any]:
    persona = role.get("persona_card", {})
    return {
        key: deepcopy(persona.get(key, []))
        for key in ("traits", "speaking_style", "constraints")
    }
