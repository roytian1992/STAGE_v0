from __future__ import annotations

import copy
import json
from typing import Any


def remap_temporal_entity_references(
    payload: Any,
    *,
    entity_id_remap: dict[str, str],
    canonical_names: dict[str, str],
) -> tuple[Any, dict[str, int]]:
    """Remap exact structured entity IDs without rewriting narrative text."""

    audit = {
        "entity_id_value_replacements": 0,
        "entity_id_key_replacements": 0,
        "canonical_name_replacements": 0,
        "deduplicated_id_values": 0,
        "coalesced_character_records": 0,
    }

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            if isinstance(value, str) and value in entity_id_remap:
                audit["entity_id_value_replacements"] += 1
                return entity_id_remap[value]
            return value

        remapped: dict[str, Any] = {}
        for key, item in value.items():
            new_key = entity_id_remap.get(key, key)
            if new_key != key:
                audit["entity_id_key_replacements"] += 1
            if new_key in remapped:
                raise ValueError(
                    f"Entity remap creates a duplicate object key: {key} -> {new_key}"
                )
            remapped[new_key] = visit(item)

        for key, item in list(remapped.items()):
            if key.endswith("_ids") and isinstance(item, list):
                deduplicated = _unique_values(item)
                audit["deduplicated_id_values"] += len(item) - len(deduplicated)
                remapped[key] = deduplicated
        if isinstance(remapped.get("characters"), list):
            characters, merged_count = _coalesce_characters(
                remapped["characters"], canonical_names
            )
            remapped["characters"] = characters
            audit["coalesced_character_records"] += merged_count

        _rewrite_paired_name(remapped, "entity_id", ("canonical_name",), canonical_names, audit)
        _rewrite_paired_name(
            remapped,
            "character_id",
            ("canonical_name", "character_name"),
            canonical_names,
            audit,
        )
        for key, entity_id in list(remapped.items()):
            if not isinstance(entity_id, str) or not key.endswith("_entity_id"):
                continue
            prefix = key[: -len("_entity_id")]
            _rewrite_paired_name(
                remapped,
                key,
                (f"{prefix}_name", f"{prefix}_canonical_name"),
                canonical_names,
                audit,
            )
        return remapped

    return visit(copy.deepcopy(payload)), audit


def _coalesce_characters(
    characters: list[Any], canonical_names: dict[str, str]
) -> tuple[list[Any], int]:
    grouped: dict[str, dict[str, Any]] = {}
    passthrough: list[Any] = []
    merged_count = 0
    for item in characters:
        if not isinstance(item, dict) or not isinstance(item.get("character_id"), str):
            passthrough.append(item)
            continue
        character_id = item["character_id"]
        if character_id not in grouped:
            grouped[character_id] = copy.deepcopy(item)
            continue
        merged_count += 1
        target = grouped[character_id]
        for key, value in item.items():
            if isinstance(value, list):
                target[key] = _unique_values([*(target.get(key) or []), *value])
            elif isinstance(value, bool):
                target[key] = bool(target.get(key)) or value
            elif key == "first_scene_order" and isinstance(value, int):
                target[key] = min(int(target.get(key, value)), value)
            elif key == "last_scene_order" and isinstance(value, int):
                target[key] = max(int(target.get(key, value)), value)
            elif key not in target or target[key] in (None, ""):
                target[key] = copy.deepcopy(value)
        if character_id in canonical_names:
            target["canonical_name"] = canonical_names[character_id]
    return [*grouped.values(), *passthrough], merged_count


def _unique_values(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _rewrite_paired_name(
    payload: dict[str, Any],
    id_key: str,
    name_keys: tuple[str, ...],
    canonical_names: dict[str, str],
    audit: dict[str, int],
) -> None:
    entity_id = payload.get(id_key)
    if not isinstance(entity_id, str):
        return
    canonical_name = canonical_names.get(entity_id)
    if not canonical_name:
        return
    for name_key in name_keys:
        if name_key not in payload or payload[name_key] == canonical_name:
            continue
        payload[name_key] = canonical_name
        audit["canonical_name_replacements"] += 1
