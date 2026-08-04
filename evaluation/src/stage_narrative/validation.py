from __future__ import annotations

from collections import Counter
from typing import Any


NARRATIVE_MODALITIES = {
    "asserted",
    "remembered",
    "dreamed",
    "hallucinated",
    "hypothetical",
    "reported",
    "uncertain",
}


def validate_graph(
    *,
    graph: dict[str, Any],
    scene_records: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    episode_relations: list[dict[str, Any]],
    storylines: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    node_ids = [node["id"] for node in graph["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        duplicates = sorted(node_id for node_id, count in Counter(node_ids).items() if count > 1)
        errors.append(f"Duplicate node ids: {duplicates[:20]}")
    known_nodes = set(node_ids)
    edge_ids = [edge["id"] for edge in graph["edges"]]
    if len(edge_ids) != len(set(edge_ids)):
        duplicates = sorted(edge_id for edge_id, count in Counter(edge_ids).items() if count > 1)
        errors.append(f"Duplicate edge ids: {duplicates[:20]}")
    dangling = [
        edge["id"]
        for edge in graph["edges"]
        if edge["source"] not in known_nodes or edge["target"] not in known_nodes
    ]
    if dangling:
        errors.append(f"Dangling graph edges: {dangling[:20]}")

    unit_ids = [
        unit["unit_id"]
        for record in scene_records
        for unit in record.get("narrative_units", [])
    ]
    primary_unit_ids = [
        unit["unit_id"]
        for record in scene_records
        for unit in record.get("narrative_units", [])
        if unit.get("kind") in {"event", "interaction"}
    ]
    invalid_unit_modalities = [
        unit["unit_id"]
        for record in scene_records
        for unit in record.get("narrative_units", [])
        if unit.get("modality") not in NARRATIVE_MODALITIES
    ]
    if invalid_unit_modalities:
        errors.append(
            f"Narrative units contain invalid modality: {invalid_unit_modalities[:20]}"
        )
    unresolved_participant_units = [
        unit["unit_id"]
        for record in scene_records
        for unit in record.get("narrative_units", [])
        if len(unit.get("participants", [])) != len(unit.get("participant_entities", []))
    ]
    if unresolved_participant_units:
        errors.append(
            f"Narrative units contain unresolved participants: {unresolved_participant_units[:20]}"
        )
    unresolved_semantic_relations = [
        relation["relation_id"]
        for record in scene_records
        for relation in record.get("entity_relations", [])
        if not relation.get("subject_entity_id") or not relation.get("object_entity_id")
    ]
    if unresolved_semantic_relations:
        errors.append(
            f"Semantic relations contain unresolved endpoints: {unresolved_semantic_relations[:20]}"
        )
    episode_children = [child for episode in episodes for child in episode["child_unit_ids"]]
    if set(primary_unit_ids) != set(episode_children) or len(primary_unit_ids) != len(episode_children):
        errors.append("Event/Interaction units are not covered by episodes exactly once")
    if any(len(episode.get("source_scene_ids", [])) != 1 for episode in episodes):
        errors.append("Episodes must remain within one source scene")
    invalid_episode_modalities = [
        episode["episode_id"]
        for episode in episodes
        if episode.get("modality") not in NARRATIVE_MODALITIES
    ]
    if invalid_episode_modalities:
        errors.append(
            f"Episodes contain invalid modality: {invalid_episode_modalities[:20]}"
        )

    episode_ids = [episode["episode_id"] for episode in episodes]
    backbones = [
        storyline
        for storyline in storylines
        if storyline.get("focus_type") == "chronological_backbone"
    ]
    if len(backbones) != 1 or backbones[0].get("child_episode_ids") != episode_ids:
        errors.append("Exactly one chronological backbone must cover all episodes in order")
    unknown_storyline_children = sorted(
        {
            child
            for storyline in storylines
            for child in storyline.get("child_episode_ids", [])
            if child not in set(episode_ids)
        }
    )
    if unknown_storyline_children:
        errors.append(
            f"Storylines contain unknown episodes: {unknown_storyline_children[:20]}"
        )

    if _has_cycle(episode_ids, episode_relations):
        errors.append("Episode relation graph contains a directed cycle")
    if episodes and not episode_relations:
        warnings.append(
            "No semantic episode relations were accepted; dependency storylines are unavailable"
        )

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "scenes": len(scene_records),
            "entities": graph["counts"]["nodes_by_type"].get("entity", 0),
            "narrative_units": len(unit_ids),
            "episode_primary_units": len(primary_unit_ids),
            "episodes": len(episodes),
            "episode_relations": len(episode_relations),
            "storylines": len(storylines),
            "graph_nodes": len(graph["nodes"]),
            "graph_edges": len(graph["edges"]),
        },
    }
    return report


def require_valid(report: dict[str, Any]) -> None:
    if report.get("status") != "passed":
        raise ValueError("Narrative graph validation failed: " + "; ".join(report["errors"]))


def _has_cycle(episode_ids: list[str], relations: list[dict[str, Any]]) -> bool:
    outgoing: dict[str, list[str]] = {episode_id: [] for episode_id in episode_ids}
    indegree = {episode_id: 0 for episode_id in episode_ids}
    for relation in relations:
        source = relation["source_id"]
        target = relation["target_id"]
        if source in outgoing and target in indegree:
            outgoing[source].append(target)
            indegree[target] += 1
    queue = sorted(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited != len(episode_ids)
