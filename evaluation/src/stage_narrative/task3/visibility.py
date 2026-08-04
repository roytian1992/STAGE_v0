from __future__ import annotations

from typing import Any


def boundary_key(boundary: dict[str, Any]) -> tuple[int, int]:
    return int(boundary["scene_order"]), int(boundary.get("char_end", 0))


def is_visible(available_from: dict[str, Any], boundary: dict[str, Any]) -> bool:
    return boundary_key(available_from) <= boundary_key(boundary)


def phase_is_active(phase: dict[str, Any], boundary: dict[str, Any]) -> bool:
    current = boundary_key(boundary)
    if boundary_key(phase["valid_from"]) > current:
        return False
    valid_until = phase.get("valid_until")
    return valid_until is None or current < boundary_key(valid_until)


def materialize_role_at_boundary(
    role: dict[str, Any], boundary: dict[str, Any]
) -> dict[str, Any]:
    memories = [
        {
            "memory_text": item["memory_text"],
            "importance": item["importance"],
            "tags": item["tags"],
        }
        for item in role.get("memories", [])
        if is_visible(item["available_from"], boundary)
    ]
    identity = _latest_active(role.get("identity_phases", []), boundary)
    persona = _latest_active(role.get("persona_phases", []), boundary)
    relationships = [
        {
            "target": item["target"],
            "value": item["value"],
            "polarity": item["polarity"],
        }
        for item in role.get("relationship_phases", [])
        if phase_is_active(item, boundary)
    ]
    return {
        "canonical_name": role["canonical_name"],
        "aliases": role.get("aliases", []),
        "identity": (
            {"name": identity["name"], "aliases": identity.get("aliases", [])}
            if identity
            else None
        ),
        "persona": (
            {
                "traits": persona.get("traits", []),
                "speaking_style": persona.get("speaking_style", []),
                "behavioral_constraints": persona.get(
                    "behavioral_constraints", []
                ),
            }
            if persona
            else {
                "traits": [],
                "speaking_style": [],
                "behavioral_constraints": [],
            }
        ),
        "relationships": relationships,
        "memories": memories,
    }


def _latest_active(
    phases: list[dict[str, Any]], boundary: dict[str, Any]
) -> dict[str, Any] | None:
    active = [item for item in phases if phase_is_active(item, boundary)]
    return (
        max(active, key=lambda item: boundary_key(item["valid_from"]))
        if active
        else None
    )
