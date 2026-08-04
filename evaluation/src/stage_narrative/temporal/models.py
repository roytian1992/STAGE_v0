from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ..models import clean_text


STATE_DIMENSIONS = {
    "goal_plan",
    "belief_knowledge",
    "relationship",
    "status_identity",
    "constraint_resource",
}


def normalize_state_dimension(value: Any) -> str:
    """Normalize an unambiguous spelling variant of the fixed state taxonomy."""
    normalized = "_".join(
        token
        for token in re.split(r"[\s_-]+", clean_text(value).casefold())
        if token
    )
    if normalized in STATE_DIMENSIONS:
        return normalized
    if normalized in {"emotion", "emotional_state", "emotional_states"}:
        return "emotion"
    tokens = sorted(normalized.split("_"))
    matches = [
        dimension
        for dimension in STATE_DIMENSIONS
        if sorted(dimension.split("_")) == tokens
    ]
    return matches[0] if len(matches) == 1 else normalized
STATE_DURABILITY = {"transient", "durable"}


def normalize_state_durability(value: Any) -> str:
    """Normalize bounded durability aliases before closed-set validation.

    Models occasionally use epistemic labels (``uncertain``/``unknown``) for
    duration.  Treating those conservatively as transient avoids promoting an
    unsupported observation into a durable state without another model call.
    """
    normalized = "_".join(
        token
        for token in re.split(r"[\s-]+", clean_text(value).casefold())
        if token
    )
    return {
        "temporary": "transient",
        "momentary": "transient",
        "short_lived": "transient",
        "short_term": "transient",
        "uncertain": "transient",
        "unknown": "transient",
        "lasting": "durable",
        "persistent": "durable",
        "long_term": "durable",
        "long_lived": "durable",
    }.get(normalized, normalized)


def normalize_state_polarity(value: Any) -> str:
    """Normalize a small closed-set alias surface without semantic inference."""
    normalized = re.sub(r"[\s-]+", "_", clean_text(value).casefold())
    transition_parts = normalized.split("_to_")
    transition_values = {"positive", "negative", "neutral", "uncertain", "mixed"}
    if len(transition_parts) == 2 and all(
        part in transition_values for part in transition_parts
    ):
        return (
            transition_parts[0]
            if transition_parts[0] == transition_parts[1]
            else "mixed"
        )
    return {
        "certain": "uncertain",
        # These stance descriptors do not encode positive/negative valence.
        "defensive": "neutral",
        "constraint": "neutral",
        "constrained": "neutral",
        "supportive": "positive",
        "threatening": "negative",
    }.get(normalized, normalized)


STATE_OPERATIONS = {"establish", "update", "revoke", "reveal"}
ACCESS_TYPES = {"witnessed", "involved", "told", "inferred", "unknown"}
PERSONA_EVIDENCE_KINDS = {"trait", "behavioral_constraint", "speaking_style"}
PERSONA_STABILITY = {"stable", "phase_specific"}
CHECKPOINT_TYPES = {
    "baseline",
    "change",
    "no_change",
    "inaccessible",
    "delayed_consequence",
    "final",
}
TASK3_PROMPT_FAMILIES = {
    "memory_grounded_reflection",
    "relationship_stance",
    "goal_decision_pressure",
    "knowledge_boundary_probe",
    "persona_invariant",
}


def normalize_task3_prompt_family(value: Any) -> str:
    """Normalize unambiguous model labels into the fixed Task 3 taxonomy."""
    normalized = "_".join(
        token
        for token in re.split(r"[\s_-]+", clean_text(value).casefold())
        if token
    )
    aliases = {
        "belief_knowledge": "memory_grounded_reflection",
        "moral_grounded_reflection": "memory_grounded_reflection",
        "emotionally_grounded_reflection": "memory_grounded_reflection",
        "relationship": "relationship_stance",
        "authority_assertion": "relationship_stance",
        "authority_enforcement": "relationship_stance",
        "command_authority": "relationship_stance",
        "authority_confrontation": "relationship_stance",
        # Leadership/role affirmation is an interpersonal stance about the
        # character's authority, not a separate public prompt family.
        "role_affirmation": "relationship_stance",
        "goal_plan": "goal_decision_pressure",
        "functional_inquiry": "goal_decision_pressure",
        "decision_pressure": "goal_decision_pressure",
        "status_identity": "persona_invariant",
        "identity_reflection": "persona_invariant",
        "identity_conflict": "persona_invariant",
        "identity_assertion": "persona_invariant",
        "identity_declaration": "persona_invariant",
        "identity_affirmation": "persona_invariant",
        "identity_reclamation": "persona_invariant",
        "identity_request": "persona_invariant",
        "identity_rejection": "persona_invariant",
        "identity_grounded_reflection": "persona_invariant",
        "identity_self_affirmation": "persona_invariant",
        "identity_self_reflection": "persona_invariant",
        "identity_self_definition": "persona_invariant",
        "identity_defense": "persona_invariant",
        "identity_negotiation": "persona_invariant",
        "identity_insecurity_probe": "persona_invariant",
        "identity_threat_reflection": "persona_invariant",
        "identity_verification": "persona_invariant",
        "identity_defiance": "persona_invariant",
        "moral_distinction": "memory_grounded_reflection",
        "moral_boundary_probe": "memory_grounded_reflection",
        "moral_accountability": "memory_grounded_reflection",
        "emotional_resolution": "memory_grounded_reflection",
        "constraint_resource": "goal_decision_pressure",
        "constraint_pressure": "goal_decision_pressure",
        # Resource/weapon-allocation prompts are decision pressure about a
        # constrained goal, not a separate public Task 3 family.
        "resource_stance": "goal_decision_pressure",
        "decision_pressure": "goal_decision_pressure",
        "authority_challenge": "relationship_stance",
    }
    return aliases.get(normalized, normalized)


BOUNDARY_RISK_TYPES = {
    "none",
    "future_information",
    "unknown_information",
    "stance_timing",
    "persona_inconsistency",
}


def normalize_boundary_risk_type(value: Any) -> str:
    """Normalize unambiguous surface variants of the fixed Task 3 risk set.

    The field is evaluator metadata, so this only resolves lexical variants
    whose meaning is explicit in the label.  Unknown labels remain unknown and
    are still rejected by the caller rather than being guessed.
    """
    normalized = "_".join(
        token
        for token in re.split(r"[\s-]+", clean_text(value).casefold())
        if token
    )
    aliases = {
        "no_risk": "none",
        "no_boundary_risk": "none",
        "future": "future_information",
        "future_fact": "future_information",
        "future_facts": "future_information",
        "unknown": "unknown_information",
        "unknown_fact": "unknown_information",
        "unknown_facts": "unknown_information",
        "stance": "stance_timing",
        "relationship_timing": "stance_timing",
        "identity_inconsistency": "persona_inconsistency",
        "persona_conflict": "persona_inconsistency",
    }
    return aliases.get(normalized, normalized)
FACT_NODE_TYPES = {"event", "occasion", "interaction"}


@dataclass(frozen=True, slots=True)
class TemporalBuildConfig:
    language: str
    max_concurrency: int = 8
    semantic_attempts: int = 2
    evidence_source_chunk_tokens: int = 700
    min_source_scenes: int = 3
    max_characters: int = 0
    include_character_names: tuple[str, ...] = ()
    task1_min_durable_states: int = 3
    task1_min_developments: int = 1
    task1_min_dimensions: int = 2
    task3_min_dialogue_evidence: int = 3
    task3_min_persona_evidence: int = 2
    task3_min_accessible_facts: int = 2
    task3_prompts_per_checkpoint: int = 2

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, language: str) -> "TemporalBuildConfig":
        selection = _dict(payload, "selection")
        eligibility = _dict(payload, "eligibility")
        task3 = _dict(payload, "task3")
        names = selection.get("include_character_names", [])
        if not isinstance(names, list):
            raise ValueError("temporal.selection.include_character_names must be an array")
        config = cls(
            language=language,
            max_concurrency=max(1, int(payload.get("max_concurrency", 8))),
            semantic_attempts=max(1, int(payload.get("semantic_attempts", 2))),
            evidence_source_chunk_tokens=max(
                1, int(payload.get("evidence_source_chunk_tokens", 700))
            ),
            min_source_scenes=max(1, int(selection.get("min_source_scenes", 3))),
            max_characters=max(0, int(selection.get("max_characters", 0))),
            include_character_names=tuple(
                value for value in (clean_text(name) for name in names) if value
            ),
            task1_min_durable_states=max(
                1, int(eligibility.get("task1_min_durable_states", 3))
            ),
            task1_min_developments=max(
                1, int(eligibility.get("task1_min_developments", 1))
            ),
            task1_min_dimensions=max(
                1, int(eligibility.get("task1_min_dimensions", 2))
            ),
            task3_min_dialogue_evidence=max(
                1, int(eligibility.get("task3_min_dialogue_evidence", 3))
            ),
            task3_min_persona_evidence=max(
                1, int(eligibility.get("task3_min_persona_evidence", 2))
            ),
            task3_min_accessible_facts=max(
                1, int(eligibility.get("task3_min_accessible_facts", 2))
            ),
            task3_prompts_per_checkpoint=max(
                1, int(task3.get("prompts_per_checkpoint", 2))
            ),
        )
        return config


@dataclass(frozen=True, slots=True)
class GraphIndex:
    graph: dict[str, Any]
    nodes_by_id: dict[str, dict[str, Any]]
    scene_by_id: dict[str, dict[str, Any]]
    scene_order_by_id: dict[str, int]
    person_entities: list[dict[str, Any]]
    episodes: list[dict[str, Any]]
    facts: list[dict[str, Any]]

    def node_scene_ids(self, node: dict[str, Any]) -> list[str]:
        values = node.get("source_scene_ids")
        if isinstance(values, list):
            return [clean_text(value) for value in values if clean_text(value)]
        value = clean_text(node.get("source_scene_id"))
        return [value] if value else []

    def node_scene_orders(self, node: dict[str, Any]) -> list[int]:
        values = node.get("source_scene_orders")
        if isinstance(values, list):
            return sorted({int(value) for value in values})
        if node.get("source_scene_order") is not None:
            return [int(node["source_scene_order"])]
        return sorted(
            {
                self.scene_order_by_id[scene_id]
                for scene_id in self.node_scene_ids(node)
                if scene_id in self.scene_order_by_id
            }
        )

    def fact_text(self, node: dict[str, Any]) -> str:
        return clean_text(
            node.get("fact")
            or node.get("description")
            or node.get("name")
        )


def build_graph_index(graph: dict[str, Any]) -> GraphIndex:
    if graph.get("schema_version") != "stage_narrative_graph_v1":
        raise ValueError("Unsupported narrative graph schema")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Narrative graph nodes must be an array")
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Narrative graph node must be an object")
        node_id = clean_text(node.get("id"))
        if not node_id or node_id in nodes_by_id:
            raise ValueError(f"Missing or duplicate narrative graph node id: {node_id}")
        nodes_by_id[node_id] = node
    scene_nodes = [node for node in nodes if node.get("node_type") == "scene"]
    scene_by_id = {clean_text(node.get("scene_id")): node for node in scene_nodes}
    if any(not scene_id for scene_id in scene_by_id):
        raise ValueError("Every scene graph node requires scene_id")
    scene_order_by_id = {
        scene_id: int(node["order"]) for scene_id, node in scene_by_id.items()
    }
    person_entities = sorted(
        [
            node
            for node in nodes
            if node.get("node_type") == "entity" and node.get("entity_type") == "Character"
        ],
        key=lambda node: (
            min(
                (
                    scene_order_by_id[scene_id]
                    for scene_id in node.get("source_scene_ids", [])
                    if scene_id in scene_order_by_id
                ),
                default=10**9,
            ),
            clean_text(node.get("canonical_name")),
            node["id"],
        ),
    )
    episodes = sorted(
        [node for node in nodes if node.get("node_type") == "episode"],
        key=lambda node: (int(node.get("order", 0)), node["id"]),
    )
    facts = sorted(
        [node for node in nodes if node.get("node_type") in FACT_NODE_TYPES],
        key=lambda node: (
            min(
                (
                    int(node.get("source_scene_order"))
                    if node.get("source_scene_order") is not None
                    else scene_order_by_id.get(
                        clean_text(node.get("source_scene_id")), 10**9
                    ),
                ),
                default=10**9,
            ),
            node["id"],
        ),
    )
    if not scene_nodes or not person_entities or not episodes or not facts:
        raise ValueError(
            "Narrative graph requires scene, person entity, episode, and fact/unit nodes"
        )
    return GraphIndex(
        graph=graph,
        nodes_by_id=nodes_by_id,
        scene_by_id=scene_by_id,
        scene_order_by_id=scene_order_by_id,
        person_entities=person_entities,
        episodes=episodes,
        facts=facts,
    )


def _dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"temporal.{key} must be an object")
    return value
