from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


BASE_UNIT_KINDS = ("event", "occasion", "interaction")
ENTITY_TYPES = (
    "Character",
    "Location",
    "Occasion",
    "TimePoint",
    "Object",
    "Concept",
    "Organization",
)
ENTITY_SCOPES = ("global", "local")
EPISODE_RELATION_TYPES = {
    "causes",
    "enables",
    "continues",
    "resolves",
    "reverses",
    "reveals",
    "none",
}
EPISODE_RELATION_WEIGHTS = {
    "causes": 1.0,
    "resolves": 0.95,
    "reverses": 0.95,
    "reveals": 0.9,
    "enables": 0.85,
    "continues": 0.8,
}
STORYLINE_FOCUS_TYPES = {
    "character_development",
    "relationship_development",
    "goal_conflict_thread",
    "knowledge_belief_thread",
    "social_institutional_thread",
}
STORYLINE_STATUSES = {"ongoing", "resolved", "reversed", "transformed"}


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    source_scene_id: str
    order: int
    title: str
    subtitle: str
    content: str

    def prompt_text(self) -> str:
        parts = [
            f"Scene {self.order} (source id {self.source_scene_id}): {self.title}"
        ]
        if self.subtitle:
            parts.append(self.subtitle)
        if self.content:
            parts.append(self.content)
        return "\n".join(parts)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique_text(values: Iterable[Any], limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if limit is not None and len(output) >= limit:
            break
    return output


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    text = re.sub(r"^(mr|mrs|ms|miss|dr|doctor|prof|professor|sir)\.?\s+", "", text)
    text = "".join(ch for ch in text if ch.isalnum())
    return text


def normalize_entity_type(value: Any) -> str:
    raw = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    key = re.sub(r"[\s_-]+", "", raw)
    by_key = {
        re.sub(r"[\s_-]+", "", item.casefold()): item for item in ENTITY_TYPES
    }
    aliases: dict[str, str] = {}
    for singular, canonical in by_key.items():
        aliases[singular] = canonical
        aliases[singular + "s"] = canonical
    # British spelling is an orthographic variant, not a semantic type mapping.
    aliases["organisation"] = "Organization"
    aliases["organisations"] = "Organization"
    return aliases.get(key, "")


def require_entity_type(value: Any) -> str:
    normalized = normalize_entity_type(value)
    if not normalized:
        raise ValueError(
            f"Illegal entity_type {clean_text(value)!r}; allowed values are "
            f"{', '.join(ENTITY_TYPES)}"
        )
    return normalized


def require_entity_scope(value: Any) -> str:
    normalized = clean_text(value).casefold()
    if normalized not in ENTITY_SCOPES:
        raise ValueError(
            f"Illegal entity scope {clean_text(value)!r}; allowed values are "
            f"{', '.join(ENTITY_SCOPES)}"
        )
    return normalized


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(clean_text(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def compatible_entity_types(left: str, right: str) -> bool:
    normalized = {normalize_entity_type(left), normalize_entity_type(right)}
    normalized.discard("")
    return len(normalized) == 1 or normalized == {"Organization", "Location"}
