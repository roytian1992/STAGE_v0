from __future__ import annotations

import copy
import json
from typing import Any

from .relation_resolution import _aggregate_relations


def apply_relation_correction_patch(
    registry: dict[str, Any], patch: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if patch.get("schema_version") != "stage_relation_correction_patch_v1":
        raise ValueError("Unsupported relation correction patch schema")
    output = copy.deepcopy(registry)
    remap = dict(patch.get("entity_id_remap") or {})
    name_by_id = dict(patch.get("canonical_names") or {})
    dropped_entity_ids = set(patch.get("dropped_entity_ids") or [])
    for observation in output.get("observations", []):
        _remap_endpoint(observation, "subject", remap, name_by_id)
        _remap_endpoint(observation, "object", remap, name_by_id)
    for relation in output.get("canonical_relations", []):
        relation["subject_entity_id"] = remap.get(
            relation.get("subject_entity_id"), relation.get("subject_entity_id")
        )
        relation["object_entity_id"] = remap.get(
            relation.get("object_entity_id"), relation.get("object_entity_id")
        )

    endpoint_overrides = {
        item["observation_id"]: item
        for item in patch.get("observation_endpoint_overrides", [])
    }
    known_observation_ids = {
        item.get("observation_id") for item in output.get("observations", [])
    }
    unknown_overrides = sorted(set(endpoint_overrides) - known_observation_ids)
    if unknown_overrides:
        raise ValueError(f"Unknown relation observation overrides: {unknown_overrides}")
    for observation in output.get("observations", []):
        override = endpoint_overrides.get(observation.get("observation_id"))
        if not override:
            continue
        for role in ("subject", "object"):
            entity_id = override.get(f"{role}_entity_id")
            if not entity_id:
                continue
            observation[f"{role}_entity_id"] = entity_id
            reverse = bool(
                observation.get("normalization", {}).get("reverse_direction", False)
            )
            canonical_role = (
                "object" if role == "subject" else "subject"
            ) if reverse else role
            observation[f"canonical_{canonical_role}_entity_id"] = entity_id
            if override.get(f"{role}_name"):
                observation[f"{role}_name"] = override[f"{role}_name"]

    drop_observations = {
        item["observation_id"]: item
        for item in patch.get("drop_observations", [])
    }
    auto_dropped_endpoint_observations: list[str] = []
    for observation in output.get("observations", []):
        decision = drop_observations.get(observation.get("observation_id"))
        if not decision and _uses_dropped_endpoint(observation, dropped_entity_ids):
            decision = {
                "reason_code": "endpoint_entity_removed",
                "reason": (
                    "The relation endpoint was removed by the reviewed entity patch, "
                    "so this observation cannot remain in the canonical KG."
                ),
            }
            auto_dropped_endpoint_observations.append(observation["observation_id"])
        if not decision:
            continue
        normalization = observation.setdefault("normalization", {})
        normalization.update(
            {
                "action": "drop",
                "predicate_id": "",
                "reason_code": decision["reason_code"],
                "normalization_reason": decision["reason_code"],
                "generated_rationale_hint": decision["reason"],
            }
        )
        for key in (
            "canonical_subject_entity_id",
            "canonical_object_entity_id",
            "canonical_predicate_id",
        ):
            observation.pop(key, None)

    drop_relation_ids = set(patch.get("drop_relation_ids") or [])
    kept_observation_ids = {
        item["observation_id"]
        for item in output.get("observations", [])
        if item.get("normalization", {}).get("action") == "keep"
    }
    relations: list[dict[str, Any]] = []
    auto_dropped_self: list[str] = []
    auto_dropped_endpoint_relations: list[str] = []
    for relation in output.get("canonical_relations", []):
        relation_id = relation.get("relation_id")
        if relation_id in drop_relation_ids:
            continue
        if _uses_dropped_endpoint(relation, dropped_entity_ids):
            auto_dropped_endpoint_relations.append(relation_id)
            continue
        if relation.get("subject_entity_id") == relation.get("object_entity_id"):
            auto_dropped_self.append(relation_id)
            continue
        relation["source_observation_ids"] = [
            value
            for value in relation.get("source_observation_ids", [])
            if value in kept_observation_ids
        ]
        if not relation["source_observation_ids"]:
            continue
        relations.append(relation)
    relations, duplicate_merge_audit = _merge_duplicate_relations(relations)
    if endpoint_overrides:
        movie_id = str(patch.get("movie_id") or "agent-reviewed-correction")
        relations = [
            relation
            for relation in _aggregate_relations(movie_id, output.get("observations", []))
            if relation.get("relation_id") not in drop_relation_ids
            and relation.get("subject_entity_id")
            not in dropped_entity_ids
            and relation.get("object_entity_id") not in dropped_entity_ids
            and relation.get("subject_entity_id") != relation.get("object_entity_id")
        ]
        duplicate_merge_audit = []
    output["canonical_relations"] = relations

    decisions_by_id = {
        item.get("observation_id"): item
        for item in output.get("audit", {}).get("decisions", [])
    }
    for observation_id, patch_decision in drop_observations.items():
        if observation_id in decisions_by_id:
            decisions_by_id[observation_id].update(
                {
                    "action": "drop",
                    "predicate_id": "",
                    "reason_code": patch_decision["reason_code"],
                    "normalization_reason": patch_decision["reason_code"],
                    "generated_rationale_hint": patch_decision["reason"],
                }
            )
    for observation_id in auto_dropped_endpoint_observations:
        if observation_id in decisions_by_id:
            decisions_by_id[observation_id].update(
                {
                    "action": "drop",
                    "predicate_id": "",
                    "reason_code": "endpoint_entity_removed",
                    "normalization_reason": "endpoint_entity_removed",
                    "generated_rationale_hint": (
                        "The relation endpoint was removed by the reviewed entity patch."
                    ),
                }
            )
    audit = output.setdefault("audit", {})
    kept_count = sum(
        item.get("normalization", {}).get("action") == "keep"
        for item in output.get("observations", [])
    )
    dropped_count = sum(
        item.get("normalization", {}).get("action") == "drop"
        for item in output.get("observations", [])
    )
    audit.update(
        {
            "kept_count": kept_count,
            "dropped_count": dropped_count,
            "canonical_relation_count": len(relations),
            "canonical_predicate_count": len(
                {item.get("predicate_id") for item in relations}
            ),
            "duplicate_observation_reduction_count": max(
                0, kept_count - len(relations)
            ),
            "agent_correction": {
                "model_calls": 0,
                "entity_id_remap_count": len(remap),
                "observation_endpoint_override_count": len(endpoint_overrides),
                "dropped_entity_ids": sorted(dropped_entity_ids),
                "dropped_observation_ids": sorted(drop_observations),
                "dropped_relation_ids": sorted(drop_relation_ids),
                "auto_dropped_endpoint_observation_ids": sorted(
                    auto_dropped_endpoint_observations
                ),
                "auto_dropped_endpoint_relation_ids": sorted(
                    auto_dropped_endpoint_relations
                ),
                "auto_dropped_self_relation_ids": sorted(auto_dropped_self),
                "duplicate_relation_merges": duplicate_merge_audit,
            },
        }
    )
    return output, audit["agent_correction"]


def _merge_duplicate_relations(
    relations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    merged_relation_ids: dict[tuple[str, str, str, str, str], list[str]] = {}
    for relation in sorted(relations, key=lambda item: str(item.get("relation_id", ""))):
        key = (
            str(relation.get("subject_entity_id", "")),
            str(relation.get("predicate_id", "")),
            str(relation.get("object_entity_id", "")),
            str(relation.get("relation_class", "")),
            json.dumps(
                relation.get("qualifiers", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        if key not in grouped:
            grouped[key] = copy.deepcopy(relation)
            merged_relation_ids[key] = [str(relation.get("relation_id", ""))]
            continue
        target = grouped[key]
        merged_relation_ids[key].append(str(relation.get("relation_id", "")))
        for field in (
            "source_observation_ids",
            "source_scene_ids",
            "source_scene_orders",
            "surface_predicates",
        ):
            target[field] = _unique_json_values(
                [*target.get(field, []), *relation.get(field, [])]
            )
        target["evidence_items"] = _unique_json_values(
            [*target.get("evidence_items", []), *relation.get("evidence_items", [])]
        )
        if relation.get("status") == "asserted":
            target["status"] = "asserted"
    audit = [
        {
            "retained_relation_id": grouped[key].get("relation_id"),
            "merged_relation_ids": relation_ids,
        }
        for key, relation_ids in merged_relation_ids.items()
        if len(relation_ids) > 1
    ]
    return list(grouped.values()), audit


def _unique_json_values(values: list[Any]) -> list[Any]:
    output = []
    seen = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def _uses_dropped_endpoint(item: dict[str, Any], dropped_entity_ids: set[str]) -> bool:
    return any(
        item.get(key) in dropped_entity_ids
        for key in (
            "subject_entity_id",
            "object_entity_id",
            "canonical_subject_entity_id",
            "canonical_object_entity_id",
        )
    )


def _remap_endpoint(
    item: dict[str, Any],
    role: str,
    remap: dict[str, str],
    name_by_id: dict[str, str],
) -> None:
    for prefix in ("", "canonical_"):
        key = f"{prefix}{role}_entity_id"
        if key in item:
            item[key] = remap.get(item[key], item[key])
    entity_id = item.get(f"{role}_entity_id")
    if entity_id in name_by_id:
        item[f"{role}_name"] = name_by_id[entity_id]
