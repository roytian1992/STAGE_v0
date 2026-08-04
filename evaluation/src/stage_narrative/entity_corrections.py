from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

from .models import clean_text, normalize_name, unique_text


def apply_entity_correction_patch(
    registry: dict[str, Any], patch: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if patch.get("schema_version") != "stage_entity_correction_patch_v1":
        raise ValueError("Unsupported entity correction patch schema")
    entities = {
        item["entity_id"]: copy.deepcopy(item)
        for item in registry.get("entities", [])
    }
    original_count = len(entities)
    applied: list[dict[str, Any]] = []
    consumed: set[str] = set()
    mention_overrides: dict[str, dict[str, str]] = {}
    for decision in patch.get("decisions", []):
        action = decision.get("action")
        if action == "split_entity":
            source_ids = list(dict.fromkeys(decision.get("source_entity_ids") or []))
            if len(source_ids) != 1 or source_ids[0] not in entities:
                raise ValueError("split_entity requires exactly one existing source entity")
            source_id = source_ids[0]
            if source_id in consumed:
                raise ValueError(f"Entity appears in multiple corrections: {source_id}")
            consumed.add(source_id)
            created, overrides, audit_item = _split_entity(
                entities[source_id],
                registry.get("mention_map", {}),
                decision=decision,
            )
            entities[source_id] = created.pop(source_id)
            for entity_id, entity in created.items():
                if entity_id in entities:
                    raise ValueError(f"Split target entity already exists: {entity_id}")
                entities[entity_id] = entity
            mention_overrides.update(overrides)
            applied.append(audit_item)
            continue
        if action == "update_entity":
            source_ids = list(dict.fromkeys(decision.get("source_entity_ids") or []))
            if len(source_ids) != 1 or source_ids[0] not in entities:
                raise ValueError("update_entity requires exactly one existing source entity")
            entity_id = source_ids[0]
            if entity_id in consumed:
                raise ValueError(f"Entity appears in multiple corrections: {entity_id}")
            consumed.add(entity_id)
            updated = copy.deepcopy(entities[entity_id])
            remove_keys = {
                clean_text(value).casefold()
                for value in (decision.get("remove_aliases") or [])
                if clean_text(value)
            }
            aliases = [
                value
                for value in (updated.get("aliases") or [])
                if clean_text(value).casefold() not in remove_keys
            ]
            aliases.extend(decision.get("add_aliases") or [])
            updated["aliases"] = unique_text(
                value
                for value in aliases
                if clean_text(value)
                and clean_text(value).casefold()
                != clean_text(updated.get("canonical_name")).casefold()
            )
            if clean_text(decision.get("canonical_name")):
                updated["canonical_name"] = clean_text(decision["canonical_name"])
            entity_type_override = clean_text(decision.get("entity_type_override"))
            if entity_type_override:
                updated.update(
                    {
                        "entity_type": entity_type_override,
                        "primary_kind": entity_type_override,
                        "entity_types": [entity_type_override],
                        "facets": [],
                        "type_status": "agent_corrected",
                        "type_votes": {
                            entity_type_override: updated.get("mention_count", 0)
                        },
                    }
                )
            updated["correction_provenance"] = {
                "decision_id": decision.get("decision_id"),
                "source_entity_ids": [entity_id],
                "reason": decision.get("reason"),
                "evidence_scene_ids": decision.get("evidence_scene_ids") or [],
            }
            entities[entity_id] = updated
            applied.append(
                {
                    "decision_id": decision.get("decision_id"),
                    "action": "update_entity",
                    "source_entity_ids": [entity_id],
                    "target_entity_id": entity_id,
                    "canonical_name": updated["canonical_name"],
                    "entity_type": updated.get("entity_type"),
                    "reason": decision.get("reason"),
                    "evidence_scene_ids": decision.get("evidence_scene_ids") or [],
                    "removed_aliases": sorted(remove_keys),
                }
            )
            continue
        if action == "drop_entity":
            source_ids = list(dict.fromkeys(decision.get("source_entity_ids") or []))
            if len(source_ids) != 1 or source_ids[0] not in entities:
                raise ValueError("drop_entity requires exactly one existing source entity")
            entity_id = source_ids[0]
            if entity_id in consumed:
                raise ValueError(f"Entity appears in multiple corrections: {entity_id}")
            consumed.add(entity_id)
            dropped = entities.pop(entity_id)
            applied.append(
                {
                    "decision_id": decision.get("decision_id"),
                    "action": "drop_entity",
                    "source_entity_ids": source_ids,
                    "target_entity_id": "",
                    "canonical_name": dropped.get("canonical_name"),
                    "entity_type": dropped.get("entity_type"),
                    "reason": decision.get("reason"),
                    "evidence_scene_ids": decision.get("evidence_scene_ids") or [],
                }
            )
            continue
        if action != "merge_entities":
            raise ValueError(f"Unsupported correction action: {decision.get('action')}")
        source_ids = list(dict.fromkeys(decision.get("source_entity_ids") or []))
        target_id = clean_text(decision.get("target_entity_id"))
        if len(source_ids) < 2 or target_id not in source_ids:
            raise ValueError("A merge requires at least two sources including its target")
        missing = [entity_id for entity_id in source_ids if entity_id not in entities]
        if missing:
            raise ValueError(f"Correction references missing entities: {missing}")
        overlap = consumed.intersection(source_ids)
        if overlap:
            raise ValueError(f"Entity appears in multiple correction groups: {sorted(overlap)}")
        consumed.update(source_ids)
        rows = [entities[entity_id] for entity_id in source_ids]
        merged = _merge_entities(rows, decision=decision, target_id=target_id)
        for entity_id in source_ids:
            entities.pop(entity_id)
        entities[target_id] = merged
        applied.append(
            {
                "decision_id": decision.get("decision_id"),
                "action": "merge_entities",
                "source_entity_ids": source_ids,
                "target_entity_id": target_id,
                "canonical_name": merged["canonical_name"],
                "entity_type": merged["entity_type"],
                "reason": decision.get("reason"),
                "evidence_scene_ids": decision.get("evidence_scene_ids") or [],
            }
        )

    result = copy.deepcopy(registry)
    result["entities"] = sorted(
        entities.values(),
        key=lambda item: (
            (item.get("source_scene_ids") or [""])[0],
            item.get("canonical_name", ""),
            item["entity_id"],
        ),
    )
    remap = {
        source_id: item["target_entity_id"]
        for item in applied
        if item["action"] == "merge_entities"
        for source_id in item["source_entity_ids"]
    }
    canonical_by_id = {
        item["entity_id"]: item["canonical_name"] for item in result["entities"]
    }
    result["mention_map"] = {
        key: {
            "entity_id": mention_overrides.get(key, {}).get(
                "entity_id", remap.get(value["entity_id"], value["entity_id"])
            ),
            "canonical_name": canonical_by_id[
                mention_overrides.get(key, {}).get(
                    "entity_id", remap.get(value["entity_id"], value["entity_id"])
                )
            ],
        }
        for key, value in registry.get("mention_map", {}).items()
        if remap.get(value["entity_id"], value["entity_id"]) in canonical_by_id
    }
    result["alias_map"] = _build_alias_map(result["entities"])
    result.setdefault("audit", {})["agent_corrections"] = applied
    result["audit"]["agent_correction_summary"] = {
        "release_tier": patch.get("release_tier", "agent_reviewed_silver"),
        "reviewer": patch.get("reviewer"),
        "decision_count": len(applied),
        "source_entity_count": sum(len(item["source_entity_ids"]) for item in applied),
        "merge_count": sum(item["action"] == "merge_entities" for item in applied),
        "split_count": sum(item["action"] == "split_entity" for item in applied),
        "split_created_entity_count": sum(
            len(item.get("created_entity_ids") or []) for item in applied
        ),
        "update_count": sum(item["action"] == "update_entity" for item in applied),
        "drop_count": sum(item["action"] == "drop_entity" for item in applied),
        "before_entity_count": original_count,
        "after_entity_count": len(result["entities"]),
        "model_calls": 0,
    }
    return result, result["audit"]["agent_correction_summary"]


def _split_entity(
    source: dict[str, Any],
    mention_map: dict[str, dict[str, str]],
    *,
    decision: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    source_id = source["entity_id"]
    retained = decision.get("retained_entity") or {}
    retained_name = clean_text(retained.get("canonical_name")) or source[
        "canonical_name"
    ]
    remove_aliases = {
        clean_text(value).casefold()
        for value in (retained.get("remove_aliases") or [])
        if clean_text(value)
    }
    retained_entity = copy.deepcopy(source)
    retained_entity["canonical_name"] = retained_name
    retained_entity["aliases"] = unique_text(
        value
        for value in [
            *(source.get("aliases") or []),
            *(retained.get("add_aliases") or []),
        ]
        if clean_text(value)
        and clean_text(value).casefold() not in remove_aliases
        and clean_text(value).casefold() != retained_name.casefold()
    )
    retained_entity["correction_provenance"] = {
        "decision_id": decision.get("decision_id"),
        "source_entity_ids": [source_id],
        "reason": decision.get("reason"),
        "evidence_scene_ids": decision.get("evidence_scene_ids") or [],
    }

    created = {source_id: retained_entity}
    overrides: dict[str, dict[str, str]] = {}
    claimed_keys: set[str] = set()
    created_ids: list[str] = []
    for spec in decision.get("new_entities") or []:
        new_id = clean_text(spec.get("entity_id"))
        new_name = clean_text(spec.get("canonical_name"))
        mention_keys = list(dict.fromkeys(spec.get("mention_keys") or []))
        if not new_id or not new_name or not mention_keys:
            raise ValueError(
                "Each split new_entity requires entity_id, canonical_name, and mention_keys"
            )
        if new_id in created:
            raise ValueError(f"Duplicate split target entity: {new_id}")
        overlap = claimed_keys.intersection(mention_keys)
        if overlap:
            raise ValueError(f"Split mention keys overlap: {sorted(overlap)}")
        invalid = [
            key
            for key in mention_keys
            if mention_map.get(key, {}).get("entity_id") != source_id
        ]
        if invalid:
            raise ValueError(
                f"Split mention keys do not belong to {source_id}: {invalid}"
            )
        claimed_keys.update(mention_keys)
        created_ids.append(new_id)
        entity_type = clean_text(spec.get("entity_type_override")) or source.get(
            "entity_type", ""
        )
        scene_ids = sorted({key.split("::", 1)[0] for key in mention_keys})
        created[new_id] = {
            "entity_id": new_id,
            "canonical_name": new_name,
            "aliases": unique_text(spec.get("aliases") or []),
            "entity_type": entity_type,
            "primary_kind": entity_type,
            "entity_types": [entity_type],
            "facets": list(spec.get("facets") or []),
            "raw_types": list(source.get("raw_types") or [entity_type]),
            "type_status": "agent_corrected",
            "type_votes": {entity_type: len(mention_keys)},
            "scope": spec.get("scope", source.get("scope", "global")),
            "mention_count": len(mention_keys),
            "anchor_mention_key": mention_keys[0],
            "source_scene_ids": scene_ids,
            "source_mention_ids": [],
            "descriptions": unique_text(spec.get("descriptions") or []),
            "source_evidence": unique_text(spec.get("source_evidence") or []),
            "correction_provenance": {
                "decision_id": decision.get("decision_id"),
                "split_from_entity_id": source_id,
                "mention_keys": mention_keys,
                "reason": decision.get("reason"),
                "evidence_scene_ids": decision.get("evidence_scene_ids") or [],
            },
        }
        for key in mention_keys:
            overrides[key] = {
                "entity_id": new_id,
                "canonical_name": new_name,
            }
    if not created_ids:
        raise ValueError("split_entity requires at least one new entity")
    audit = {
        "decision_id": decision.get("decision_id"),
        "action": "split_entity",
        "source_entity_ids": [source_id],
        "target_entity_id": source_id,
        "canonical_name": retained_name,
        "entity_type": retained_entity.get("entity_type"),
        "created_entity_ids": created_ids,
        "mention_key_count": len(claimed_keys),
        "reason": decision.get("reason"),
        "evidence_scene_ids": decision.get("evidence_scene_ids") or [],
    }
    return created, overrides, audit


def _merge_entities(
    rows: list[dict[str, Any]], *, decision: dict[str, Any], target_id: str
) -> dict[str, Any]:
    target = next(item for item in rows if item["entity_id"] == target_id)
    canonical_name = clean_text(decision.get("canonical_name")) or target[
        "canonical_name"
    ]
    entity_type_override = clean_text(decision.get("entity_type_override"))
    merged = copy.deepcopy(target)
    merged["canonical_name"] = canonical_name
    merged["aliases"] = unique_text(
        value
        for row in rows
        for value in [row.get("canonical_name"), *(row.get("aliases") or [])]
        if clean_text(value) and clean_text(value) != canonical_name
    )
    for field, limit in (
        ("descriptions", 512),
        ("source_evidence", 512),
        ("source_mention_ids", 2048),
    ):
        merged[field] = unique_text(
            (value for row in rows for value in (row.get(field) or [])), limit=limit
        )
    merged["source_scene_ids"] = sorted(
        {value for row in rows for value in (row.get("source_scene_ids") or [])}
    )
    merged["mention_count"] = sum(int(row.get("mention_count", 0)) for row in rows)
    merged["scope"] = (
        "global" if any(row.get("scope") == "global" for row in rows) else "local"
    )
    merged["anchor_mention_key"] = target.get("anchor_mention_key")
    if entity_type_override:
        merged.update(
            {
                "entity_type": entity_type_override,
                "primary_kind": entity_type_override,
                "entity_types": [entity_type_override],
                "facets": [],
                "type_status": "agent_corrected",
            }
        )
        merged["type_votes"] = {entity_type_override: merged["mention_count"]}
    else:
        merged["entity_types"] = unique_text(
            value for row in rows for value in (row.get("entity_types") or [])
        )
        merged["raw_types"] = unique_text(
            value for row in rows for value in (row.get("raw_types") or [])
        )
        votes: Counter[str] = Counter()
        for row in rows:
            votes.update(row.get("type_votes") or {})
        merged["type_votes"] = dict(sorted(votes.items()))
    merged["correction_provenance"] = {
        "decision_id": decision.get("decision_id"),
        "source_entity_ids": [row["entity_id"] for row in rows],
        "reason": decision.get("reason"),
        "evidence_scene_ids": decision.get("evidence_scene_ids") or [],
    }
    return merged


def _build_alias_map(
    entities: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    candidates: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for entity in entities:
        resolved = {
            "entity_id": entity["entity_id"],
            "canonical_name": entity["canonical_name"],
        }
        for value in [entity["canonical_name"], *(entity.get("aliases") or [])]:
            normalized = normalize_name(value)
            if normalized:
                candidates[normalized][entity["entity_id"]] = resolved
    return {
        normalized: next(iter(values.values()))
        for normalized, values in candidates.items()
        if len(values) == 1
    }
