from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .models import require_entity_type


COMPATIBLE_TYPE_PAIRS = {frozenset(("Organization", "Location"))}
_CANONICAL_TYPE_PRIORITY = {"Organization": 0, "Location": 1}


def initial_type_profile(raw_type: Any) -> dict[str, Any]:
    entity_type = require_entity_type(raw_type)
    return {
        "raw_type": entity_type,
        "candidate_primary_kinds": [entity_type],
        "facets": [],
        "type_status": "resolved",
    }


def type_compatibility(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_types = _profile_types(left)
    right_types = _profile_types(right)
    if left_types == right_types:
        return "same_type"
    if any(
        frozenset((left_type, right_type)) in COMPATIBLE_TYPE_PAIRS
        for left_type in left_types
        for right_type in right_types
    ):
        return "compatible_type_review"
    return "incompatible_type"


def resolve_cluster_type(mentions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    votes: Counter[str] = Counter()
    for mention in mentions:
        profile = mention.get("type_profile") or initial_type_profile(
            mention.get("entity_type")
        )
        entity_type = _profile_type(profile)
        votes[entity_type] += max(1, int(mention.get("frequency", 1)))

    if not votes:
        raise ValueError("Cannot resolve an entity cluster without legal type observations")
    observed_types = sorted(votes)
    if len(votes) > 1 and any(
        frozenset((left, right)) not in COMPATIBLE_TYPE_PAIRS
        for index, left in enumerate(observed_types)
        for right in observed_types[index + 1 :]
    ):
        raise ValueError(f"Canonical entity cluster contains conflicting types: {dict(votes)}")
    entity_type = sorted(
        votes,
        key=lambda value: (
            -votes[value],
            _CANONICAL_TYPE_PRIORITY.get(value, 100),
            value,
        ),
    )[0]
    return {
        "entity_type": entity_type,
        # Kept temporarily as an exact alias for downstream readers; it is not a second taxonomy.
        "primary_kind": entity_type,
        "facets": [],
        "entity_types": observed_types,
        "raw_types": observed_types,
        "type_status": "resolved" if len(observed_types) == 1 else "resolved_compatible",
        "type_votes": dict(sorted(votes.items())),
    }


def legacy_entity_type(primary_kind: str, facets: Iterable[str] = ()) -> str:
    del facets
    return require_entity_type(primary_kind)


def _profile_type(profile: dict[str, Any]) -> str:
    values = _profile_types(profile)
    if len(values) != 1:
        raise ValueError(f"Type profile is not singular: {profile}")
    return next(iter(values))


def _profile_types(profile: dict[str, Any]) -> set[str]:
    explicit = profile.get("entity_types") or profile.get("raw_types")
    if explicit:
        return {require_entity_type(value) for value in explicit}
    values = profile.get("candidate_primary_kinds") or [
        profile.get("primary_kind"),
        profile.get("raw_type"),
    ]
    legal = [require_entity_type(value) for value in values if value]
    if not legal:
        raise ValueError(f"Type profile has no legal type: {profile}")
    return set(legal)
