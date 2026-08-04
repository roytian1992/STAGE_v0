from __future__ import annotations

from collections import Counter
from typing import Any

from .models import stable_id


def build_narrative_graph(
    *,
    movie_id: str,
    scene_records: list[dict[str, Any]],
    entity_registry: dict[str, Any],
    episodes: list[dict[str, Any]],
    episode_relations: list[dict[str, Any]],
    storylines: list[dict[str, Any]],
    relation_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    scene_node_ids: dict[str, str] = {}
    for record in sorted(scene_records, key=lambda item: int(item["scene"]["order"])):
        scene = record["scene"]
        scene_node_id = stable_id("scene", movie_id, scene["scene_id"])
        scene_node_ids[scene["scene_id"]] = scene_node_id
        nodes.append(
            {
                "id": scene_node_id,
                "node_type": "scene",
                **scene,
            }
        )
    for entity in entity_registry["entities"]:
        nodes.append(
            {
                "id": entity["entity_id"],
                "node_type": "entity",
                **entity,
            }
        )
        for scene_id in entity["source_scene_ids"]:
            if scene_id in scene_node_ids:
                edges.append(
                    _edge(
                        movie_id,
                        entity["entity_id"],
                        "appears_in_scene",
                        scene_node_ids[scene_id],
                        source_scene_ids=[scene_id],
                    )
                )

    for record in scene_records:
        for unit in record.get("narrative_units", []):
            nodes.append({"id": unit["unit_id"], "node_type": unit["kind"], **unit})
            edges.append(
                _edge(
                    movie_id,
                    unit["unit_id"],
                    "grounded_in_scene",
                    scene_node_ids[unit["source_scene_id"]],
                    source_scene_ids=[unit["source_scene_id"]],
                )
            )
            for participant in unit.get("participant_entities", []):
                edges.append(
                    _edge(
                        movie_id,
                        participant["entity_id"],
                        "participates_in",
                        unit["unit_id"],
                        source_scene_ids=[unit["source_scene_id"]],
                    )
                )
        if relation_registry is None:
            for relation in record.get("entity_relations", []):
                if not relation.get("subject_entity_id") or not relation.get("object_entity_id"):
                    continue
                if relation["subject_entity_id"] == relation["object_entity_id"]:
                    continue
                edges.append(
                    {
                        "id": relation["relation_id"],
                        "source": relation["subject_entity_id"],
                        "relation_type": relation["predicate"],
                        "target": relation["object_entity_id"],
                        "description": relation.get("description", ""),
                        "evidence": relation.get("evidence", ""),
                        "source_scene_ids": [relation["source_scene_id"]],
                        "edge_layer": "semantic",
                    }
                )

    if relation_registry is not None:
        for relation in relation_registry.get("canonical_relations", []):
            edges.append(
                {
                    "id": relation["relation_id"],
                    "source": relation["subject_entity_id"],
                    "relation_type": relation["predicate_id"],
                    "target": relation["object_entity_id"],
                    "relation_class": relation["relation_class"],
                    "status": relation["status"],
                    "surface_predicates": relation["surface_predicates"],
                    "source_observation_ids": relation["source_observation_ids"],
                    "source_scene_ids": relation["source_scene_ids"],
                    "evidence_items": relation["evidence_items"],
                    "edge_layer": "semantic",
                }
            )

    for episode in episodes:
        nodes.append({"id": episode["episode_id"], "node_type": "episode", **episode})
        for scene_id in episode["source_scene_ids"]:
            edges.append(
                _edge(
                    movie_id,
                    episode["episode_id"],
                    "spans_scene",
                    scene_node_ids[scene_id],
                    source_scene_ids=[scene_id],
                )
            )
        for child_id in episode["child_unit_ids"]:
            edges.append(
                _edge(
                    movie_id,
                    child_id,
                    "part_of_episode",
                    episode["episode_id"],
                    source_scene_ids=episode["source_scene_ids"],
                )
            )
        for occasion_id in episode.get("context_occasion_unit_ids", []):
            edges.append(
                _edge(
                    movie_id,
                    occasion_id,
                    "context_for_episode",
                    episode["episode_id"],
                    source_scene_ids=episode["source_scene_ids"],
                )
            )

    for relation in episode_relations:
        edges.append(
            {
                "id": relation["relation_id"],
                "source": relation["source_id"],
                "relation_type": relation["relation_type"],
                "target": relation["target_id"],
                "description": relation["description"],
                "evidence": relation.get("evidence", []),
                "confidence": relation["confidence"],
                "weight": relation["weight"],
                "edge_layer": "episode_dag",
            }
        )

    for storyline in storylines:
        nodes.append(
            {"id": storyline["storyline_id"], "node_type": "storyline", **storyline}
        )
        for scene_id in storyline["source_scene_ids"]:
            edges.append(
                _edge(
                    movie_id,
                    storyline["storyline_id"],
                    "spans_scene",
                    scene_node_ids[scene_id],
                    source_scene_ids=[scene_id],
                )
            )
        for episode_id in storyline["child_episode_ids"]:
            edges.append(
                _edge(
                    movie_id,
                    episode_id,
                    "part_of_storyline",
                    storyline["storyline_id"],
                    source_scene_ids=storyline["source_scene_ids"],
                )
            )

    node_counts = Counter(node["node_type"] for node in nodes)
    edge_counts = Counter(edge["relation_type"] for edge in edges)
    return {
        "schema_version": "stage_narrative_graph_v1",
        "movie_id": movie_id,
        "nodes": nodes,
        "edges": edges,
        "counts": {
            "nodes_total": len(nodes),
            "edges_total": len(edges),
            "nodes_by_type": dict(sorted(node_counts.items())),
            "edges_by_type": dict(sorted(edge_counts.items())),
        },
    }


def _edge(
    movie_id: str,
    source: str,
    relation_type: str,
    target: str,
    *,
    source_scene_ids: list[str],
) -> dict[str, Any]:
    return {
        "id": stable_id("edge", movie_id, source, relation_type, target),
        "source": source,
        "relation_type": relation_type,
        "target": target,
        "source_scene_ids": source_scene_ids,
        "edge_layer": "narrative_hierarchy",
    }
