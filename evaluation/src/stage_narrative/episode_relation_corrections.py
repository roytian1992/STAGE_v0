from __future__ import annotations

import copy
from typing import Any

from .models import EPISODE_RELATION_TYPES, EPISODE_RELATION_WEIGHTS, clean_text, stable_id


def apply_episode_relation_correction_patch(
    relation_stage: dict[str, Any],
    episodes: list[dict[str, Any]],
    patch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if patch.get("schema_version") != "stage_episode_relation_correction_patch_v1":
        raise ValueError("Unsupported Episode relation correction patch schema")
    output = copy.deepcopy(relation_stage)
    episode_by_id = {item["episode_id"]: item for item in episodes}
    relation_by_id = {
        item["relation_id"]: item for item in output.get("relations", [])
    }
    drop_by_id = {
        item["relation_id"]: item for item in patch.get("drop_relations", [])
    }
    relabel_by_id = {
        item["relation_id"]: item for item in patch.get("relabel_relations", [])
    }
    unknown = (set(drop_by_id) | set(relabel_by_id)) - set(relation_by_id)
    if unknown:
        raise ValueError(f"Relation correction references unknown relations: {sorted(unknown)}")
    overlap = set(drop_by_id) & set(relabel_by_id)
    if overlap:
        raise ValueError(f"Relations cannot be dropped and relabeled: {sorted(overlap)}")

    relations: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for relation in output.get("relations", []):
        relation_id = relation["relation_id"]
        if relation_id in drop_by_id:
            decisions.append(
                {
                    "relation_id": relation_id,
                    "action": "drop",
                    "source_id": relation["source_id"],
                    "target_id": relation["target_id"],
                    "reason": drop_by_id[relation_id]["reason"],
                }
            )
            continue
        relabel = relabel_by_id.get(relation_id)
        if relabel:
            before = relation["relation_type"]
            relation["relation_type"] = _relation_type(relabel["relation_type"])
            relation["description"] = clean_text(
                relabel.get("description") or relation.get("description")
            )
            relation["evidence"] = list(
                relabel.get("evidence") or relation.get("evidence") or []
            )
            relation["confidence"] = float(
                relabel.get("confidence", relation.get("confidence", 0.8))
            )
            relation["weight"] = round(
                EPISODE_RELATION_WEIGHTS[relation["relation_type"]]
                * relation["confidence"],
                6,
            )
            decisions.append(
                {
                    "relation_id": relation_id,
                    "action": "relabel",
                    "source_id": relation["source_id"],
                    "target_id": relation["target_id"],
                    "before_relation_type": before,
                    "after_relation_type": relation["relation_type"],
                    "reason": relabel["reason"],
                }
            )
        relations.append(relation)

    existing_pairs = {(item["source_id"], item["target_id"]) for item in relations}
    added_ids: list[str] = []
    for addition in patch.get("add_relations", []):
        source_id = clean_text(addition.get("source_id"))
        target_id = clean_text(addition.get("target_id"))
        if source_id not in episode_by_id or target_id not in episode_by_id:
            raise ValueError(f"Added relation has unknown endpoint: {source_id}, {target_id}")
        if int(episode_by_id[source_id]["order"]) >= int(episode_by_id[target_id]["order"]):
            raise ValueError(f"Added relation is not chronological: {source_id}, {target_id}")
        if (source_id, target_id) in existing_pairs:
            raise ValueError(f"Added relation duplicates an existing pair: {source_id}, {target_id}")
        relation_type = _relation_type(addition.get("relation_type"))
        confidence = float(addition.get("confidence", 0.9))
        relation_id = stable_id(
            "episode-relation-agent", source_id, target_id, relation_type
        )
        relation = {
            "relation_id": relation_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "description": clean_text(addition.get("description")),
            "evidence": list(addition.get("evidence") or []),
            "confidence": confidence,
            "weight": round(
                EPISODE_RELATION_WEIGHTS[relation_type] * confidence, 6
            ),
            "agent_added": True,
            "correction_decision_id": addition.get("decision_id"),
        }
        relations.append(relation)
        existing_pairs.add((source_id, target_id))
        added_ids.append(relation_id)
        decisions.append(
            {
                "relation_id": relation_id,
                "action": "add",
                "source_id": source_id,
                "target_id": target_id,
                "after_relation_type": relation_type,
                "reason": addition["reason"],
            }
        )

    relations.sort(
        key=lambda item: (
            int(episode_by_id[item["source_id"]]["order"]),
            int(episode_by_id[item["target_id"]]["order"]),
            item["relation_id"],
        )
    )
    _validate_relations(relations, episode_by_id)
    output["relations"] = relations
    correction_audit = {
        "schema_version": "stage_episode_relation_correction_audit_v1",
        "reviewer": patch.get("reviewer"),
        "release_tier": patch.get("release_tier", "agent_reviewed_silver"),
        "model_calls": 0,
        "input_relation_count": len(relation_by_id),
        "output_relation_count": len(relations),
        "drop_count": len(drop_by_id),
        "relabel_count": len(relabel_by_id),
        "add_count": len(added_ids),
        "decisions": decisions,
    }
    output.setdefault("audit", {})["agent_correction"] = correction_audit
    return output, correction_audit


def _relation_type(value: Any) -> str:
    relation_type = clean_text(value).casefold()
    if relation_type not in EPISODE_RELATION_TYPES:
        raise ValueError(f"Unsupported Episode relation type: {relation_type}")
    return relation_type


def _validate_relations(
    relations: list[dict[str, Any]], episode_by_id: dict[str, dict[str, Any]]
) -> None:
    ids = [item["relation_id"] for item in relations]
    pairs = [(item["source_id"], item["target_id"]) for item in relations]
    if len(ids) != len(set(ids)):
        raise ValueError("Corrected Episode relations contain duplicate IDs")
    if len(pairs) != len(set(pairs)):
        raise ValueError("Corrected Episode relations contain duplicate endpoint pairs")
    for relation in relations:
        source = episode_by_id.get(relation["source_id"])
        target = episode_by_id.get(relation["target_id"])
        if source is None or target is None:
            raise ValueError("Corrected Episode relation has an unknown endpoint")
        if int(source["order"]) >= int(target["order"]):
            raise ValueError("Corrected Episode relation is not chronological")
        _relation_type(relation["relation_type"])
