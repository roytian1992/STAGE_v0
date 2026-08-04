from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..io import sha256_json
from ..models import Scene, normalize_name, stable_id, unique_text
from .identity import character_name_at_scene


TASK1_STATE_UPDATE_CONTEXT_PROTOCOL = {
    "screenplay_scope": "previous_checkpoint_exclusive_to_current_checkpoint_inclusive",
    "memory_method": "runtime_entity_centric_chunk_extraction",
    "memory_extractor_inputs": ["character", "aliases", "screenplay_interval"],
    "previous_state_visible_to_memory_extractor": False,
    "memory_shared_between_settings": True,
    "full_screenplay_prefix_used": False,
    "previous_state_model_projection": "claim_text_only",
}

TASK1_STATE_UPDATE_OUTPUT_SCHEMA = {
    "current_state": {
        "type": "array",
        "item_fields": ["claim", "evidence_scene_orders"],
    },
    "developments_since_previous_checkpoint": {
        "type": "array",
        "item_fields": ["claim", "evidence_scene_orders"],
    },
}


def materialize_task1_assets(
    *,
    movie_id: str,
    language: str,
    script_path: str,
    script_sha256: str,
    scenes: list[Scene],
    registry: dict[str, Any],
    checkpoints: dict[str, Any],
    state_ledger: dict[str, Any],
    development_graph: dict[str, Any],
) -> dict[str, Any]:
    characters = {
        item["character_id"]: item
        for item in registry["characters"]
        if item["task1_eligible"]
    }
    states_by_id = {item["state_id"]: item for item in state_ledger["states"]}
    developments_by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tracked_state_keys_by_character: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for item in development_graph["developments"]:
        developments_by_character[item["character_id"]].append(item)
        for state_id in [
            *item["before_state_ids"],
            *item["resulting_state_ids"],
            *item["invariant_state_ids"],
        ]:
            state = states_by_id.get(state_id)
            if state:
                tracked_state_keys_by_character[item["character_id"]].add(
                    _state_track_key(state)
                )
    checkpoints_by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for checkpoint in checkpoints["checkpoints"]:
        if checkpoint["character_id"] in characters:
            checkpoints_by_character[checkpoint["character_id"]].append(checkpoint)

    instances: list[dict[str, Any]] = []
    for character_id, character in characters.items():
        sequence = sorted(
            checkpoints_by_character[character_id],
            key=lambda item: (int(item["scene_order"]), item["checkpoint_id"]),
        )
        previous_order = 0
        tracked_state_keys = tracked_state_keys_by_character[character_id]
        for checkpoint in sequence:
            current_order = int(checkpoint["scene_order"])
            current_states = [
                states_by_id[state_id]
                for state_id in checkpoint["active_state_ids"]
                if state_id in states_by_id
                and _state_track_key(states_by_id[state_id]) in tracked_state_keys
            ]
            interval_developments = [
                item
                for item in developments_by_character[character_id]
                if previous_order < int(item["effective_from_scene"]) <= current_order
            ]
            interval_development_ids = [
                item["development_id"] for item in interval_developments
            ]
            previous_active_ids = {
                state["state_id"]
                for state in state_ledger["states"]
                if state["character_id"] == character_id
                and _state_track_key(state) in tracked_state_keys
                and _state_active(state, previous_order)
            }
            current_active_ids = {state["state_id"] for state in current_states}
            explicit_invariants = {
                state_id
                for development in interval_developments
                for state_id in development["invariant_state_ids"]
            }
            invariant_ids = sorted(
                (previous_active_ids & current_active_ids) | explicit_invariants
            )
            supporting_evidence_ids = unique_text(
                evidence_id
                for state in current_states
                for evidence_id in state["supporting_evidence_ids"]
            )
            supporting_evidence_ids = unique_text(
                [
                    *supporting_evidence_ids,
                    *(
                        evidence_id
                        for development in interval_developments
                        for key in (
                            "evidence_before_ids",
                            "evidence_catalyst_ids",
                            "evidence_after_ids",
                        )
                        for evidence_id in development[key]
                    ),
                ]
            )
            if not current_states and not interval_developments:
                previous_order = current_order
                continue
            instance_id = stable_id(
                "task1-instance",
                movie_id,
                character_id,
                checkpoint["checkpoint_id"],
            )
            instances.append(
                {
                    "instance_id": instance_id,
                    "movie_id": movie_id,
                    "language": language,
                    "character_id": character_id,
                    "model_input": {
                        "focal_character": character_name_at_scene(
                            character, current_order
                        ),
                        "identity_lineage_name": character["canonical_name"],
                        "aliases": character["aliases"],
                        "previous_checkpoint_scene_order": previous_order,
                        "current_checkpoint_scene_order": current_order,
                        "screenplay_prefix_ref": {
                            "script_path": script_path,
                            "script_sha256": script_sha256,
                            "start_scene_order": 1,
                            "end_scene_order": current_order,
                            "scene_count": current_order,
                        },
                        "output_schema": {
                            "current_character_states": [
                                "dimension",
                                "target",
                                "current_state",
                                "supporting_scene_ids",
                            ],
                            "new_developments_since_previous_checkpoint": [
                                "dimension",
                                "target",
                                "before_state",
                                "catalyst",
                                "resulting_state",
                                "downstream_consequence",
                                "supporting_scene_ids",
                            ],
                        },
                    },
                    "evaluator_reference": {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "checkpoint_control_type": checkpoint["checkpoint_type"],
                        "checkpoint_control_types": checkpoint.get("control_types", []),
                        "gold_current_state_ids": [
                            state["state_id"] for state in current_states
                        ],
                        "gold_development_ids": interval_development_ids,
                        "gold_invariant_state_ids": invariant_ids,
                        "unknown_fact_ids": checkpoint["unknown_fact_ids"],
                        "future_forbidden_fact_ids": checkpoint[
                            "future_forbidden_fact_ids"
                        ],
                        "acceptable_fact_paraphrases": [],
                        "explicit_negative_claims": [],
                        "supporting_evidence_ids": supporting_evidence_ids,
                    },
                    "validation_status": "silver_candidate",
                }
            )
            previous_order = current_order
    instances.sort(
        key=lambda item: (
            item["character_id"],
            item["model_input"]["current_checkpoint_scene_order"],
            item["instance_id"],
        )
    )
    return {
        "schema_version": "stage_task1_character_development_tracking_v1",
        "task": "character_development_tracking",
        "movie_id": movie_id,
        "language": language,
        "screenplay": {
            "path": script_path,
            "sha256": script_sha256,
            "scene_count": len(scenes),
        },
        "character_count": len({item["character_id"] for item in instances}),
        "instance_count": len(instances),
        "instances": instances,
    }


def materialize_task1_state_update_assets(
    *,
    legacy_task1: dict[str, Any],
    script_path: str,
    script_sha256: str,
    scene_count: int,
    state_ledger: dict[str, Any],
    development_graph: dict[str, Any],
    evidence_bank: dict[str, Any],
) -> dict[str, Any]:
    """Project the temporal silver asset into bounded RSU/ASU assets.

    This is intentionally deterministic. The temporal pipeline may generate many
    active states, while the benchmark interface exposes at most eight current
    states and four interval developments. Selection is recorded as a silver
    projection and remains subject to the ordinary manual review gate.
    """
    movie_id = str(legacy_task1["movie_id"])
    evidence_by_id = {
        row["evidence_id"]: row for row in evidence_bank.get("evidence_units", [])
    }
    states_by_id = {
        row["state_id"]: row for row in state_ledger.get("states", [])
    }
    developments_by_id = {
        row["development_id"]: row
        for row in development_graph.get("developments", [])
    }
    instances = list(legacy_task1.get("instances", []))
    if not instances:
        shared = {
            "movie_id": movie_id,
            "language": legacy_task1.get("language", ""),
            "script": {"file": "script.json", "sha256": script_sha256, "scene_count": scene_count},
            "task": "Track how a focal character's state changes as the story evolves between consecutive checkpoints.",
            "context_protocol": TASK1_STATE_UPDATE_CONTEXT_PROTOCOL,
            "output_schema": TASK1_STATE_UPDATE_OUTPUT_SCHEMA,
            "counts": {
                "trajectories": 0,
                "checkpoints": 0,
                "current_state_reference_claims": 0,
                "development_reference_claims": 0,
            },
            "construction": {
                "method": "deterministic_compact_projection_from_temporal_silver_asset",
                "llm_calls": 0,
                "review_status": "not_applicable_no_task1_eligible_characters",
                "selection_limits": {"current_state": 8, "developments_since_previous_checkpoint": 4},
            },
        }
        return {
            "reference": {
                "schema": "stage_task1_reference_state_update",
                "setting": "reference_state_update",
                **shared,
                "metrics": [],
                "trajectories": [],
            },
            "autoregressive": {
                "schema": "stage_task1_autoregressive_state_update",
                "setting": "autoregressive_state_update",
                **shared,
                "metrics": [],
                "trajectories": [],
            },
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        grouped[str(instance["character_id"])].append(instance)

    reference_trajectories: list[dict[str, Any]] = []
    prediction_trajectories: list[dict[str, Any]] = []
    state_count = 0
    development_count = 0
    for character_id, rows in sorted(grouped.items()):
        rows.sort(
            key=lambda row: (
                int(row["model_input"]["current_checkpoint_scene_order"]),
                str(row["evaluator_reference"]["checkpoint_id"]),
            )
        )
        aliases: list[str] = []
        for row in rows:
            for value in (
                row["model_input"].get("focal_character"),
                row["model_input"].get("identity_lineage_name"),
                *row["model_input"].get("aliases", []),
            ):
                text = str(value or "").strip()
                if text and text not in aliases:
                    aliases.append(text)
        reference_checkpoints: list[dict[str, Any]] = []
        prediction_checkpoints: list[dict[str, Any]] = []
        previous_checkpoint_id: str | None = None
        previous_scene_order = 0
        previous_reference_state: list[dict[str, Any]] = []
        for sequence_index, instance in enumerate(rows, start=1):
            model_input = instance["model_input"]
            evaluator = instance["evaluator_reference"]
            checkpoint_id = str(evaluator["checkpoint_id"])
            current_scene_order = int(model_input["current_checkpoint_scene_order"])
            if current_scene_order <= previous_scene_order or current_scene_order > scene_count:
                raise ValueError(f"Invalid Task 1 boundary: {movie_id}/{checkpoint_id}")
            selected_state_ids = _select_compact_state_ids(
                evaluator.get("gold_current_state_ids", []),
                states_by_id=states_by_id,
                developments_by_id=developments_by_id,
            )
            selected_development_ids = _select_compact_development_ids(
                evaluator.get("gold_development_ids", []),
                developments_by_id=developments_by_id,
            )
            current_state = [
                _state_claim(states_by_id[state_id], evidence_by_id, current_scene_order)
                for state_id in selected_state_ids
                if state_id in states_by_id
            ]
            developments = [
                _development_claim(
                    developments_by_id[development_id],
                    states_by_id=states_by_id,
                    evidence_by_id=evidence_by_id,
                    previous_scene_order=previous_scene_order,
                    current_scene_order=current_scene_order,
                )
                for development_id in selected_development_ids
                if development_id in developments_by_id
            ]
            current_state = [row for row in current_state if row["evidence_scene_orders"]]
            developments = [
                row
                for row in developments
                if row["evidence_scene_orders"]
                and any(scene > previous_scene_order for scene in row["evidence_scene_orders"])
            ]
            interval = {
                "script_file": "script.json",
                "start_scene_order_exclusive": previous_scene_order,
                "end_scene_order_inclusive": current_scene_order,
            }
            common = {
                "sequence_index": sequence_index,
                "instance_id": instance["instance_id"],
                "checkpoint_id": checkpoint_id,
                "previous_checkpoint_id": previous_checkpoint_id,
                "screenplay_interval": interval,
                "entity_memory": {
                    "memory_slot": f"M{sequence_index}",
                    "source": "shared_runtime_interval_extraction",
                    "extractor_inputs": ["character", "aliases", "screenplay_interval"],
                    "previous_state_visible_to_extractor": False,
                },
            }
            target = {
                "current_state": current_state,
                "developments_since_previous_checkpoint": developments,
            }
            reference_checkpoints.append(
                {
                    **common,
                    "model_input": {
                        "previous_state": previous_reference_state,
                        "previous_state_source": "reference_state",
                    },
                    "evaluation_target": target,
                }
            )
            prediction_checkpoints.append(
                {
                    **common,
                    "model_input": {
                        "previous_state_source": (
                            "empty" if previous_checkpoint_id is None else "autoregressive_prediction"
                        )
                    },
                    "evaluation_target_ref": {
                        "asset_file": "task_1_reference_state_update.json",
                        "checkpoint_id": checkpoint_id,
                    },
                }
            )
            previous_reference_state = [
                {"claim": row["claim"], "evidence_scene_orders": list(row["evidence_scene_orders"])}
                for row in current_state
            ]
            state_count += len(current_state)
            development_count += len(developments)
            previous_checkpoint_id = checkpoint_id
            previous_scene_order = current_scene_order
        trajectory_common = {
            "trajectory_id": f"{movie_id}::{character_id}",
            "character_id": character_id,
            "character": aliases[0],
            "aliases": aliases,
        }
        reference_trajectories.append({**trajectory_common, "checkpoints": reference_checkpoints})
        prediction_trajectories.append(
            {**trajectory_common, "initial_state": [], "checkpoints": prediction_checkpoints}
        )

    shared = {
        "movie_id": movie_id,
        "language": legacy_task1["language"],
        "script": {"file": "script.json", "sha256": script_sha256, "scene_count": scene_count},
        "task": "Track how a focal character's state changes as the story evolves between consecutive checkpoints.",
        "context_protocol": TASK1_STATE_UPDATE_CONTEXT_PROTOCOL,
        "output_schema": TASK1_STATE_UPDATE_OUTPUT_SCHEMA,
        "counts": {
            "trajectories": len(reference_trajectories),
            "checkpoints": len(instances),
            "current_state_reference_claims": state_count,
            "development_reference_claims": development_count,
        },
        "construction": {
            "method": "deterministic_compact_projection_from_temporal_silver_asset",
            "llm_calls": 0,
            "review_status": "silver_candidate_requires_manual_review",
            "selection_limits": {"current_state": 8, "developments_since_previous_checkpoint": 4},
        },
    }
    return {
        "reference": {
            "schema": "stage_task1_reference_state_update",
            "setting": "reference_state_update",
            **shared,
            "metrics": [
                "current_state_reference_coverage",
                "development_reference_coverage",
                "current_state_unsupported_rate",
                "development_unsupported_rate",
                "current_state_contradiction_rate",
                "development_contradiction_rate",
            ],
            "trajectories": reference_trajectories,
        },
        "autoregressive": {
            "schema": "stage_task1_autoregressive_state_update",
            "setting": "autoregressive_state_update",
            **shared,
            "metrics": [
                "current_state_reference_coverage",
                "development_reference_coverage",
                "current_state_unsupported_rate",
                "development_unsupported_rate",
                "current_state_contradiction_rate",
                "development_contradiction_rate",
                "current_state_reference_coverage_accumulation_gap",
                "development_reference_coverage_accumulation_gap",
            ],
            "counts": {"trajectories": len(reference_trajectories), "checkpoints": len(instances)},
            "trajectories": prediction_trajectories,
        },
    }


def _select_compact_state_ids(
    state_ids: list[str],
    *,
    states_by_id: dict[str, dict[str, Any]],
    developments_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    development_state_ids = {
        state_id
        for development in developments_by_id.values()
        for state_id in (
            *development.get("before_state_ids", []),
            *development.get("resulting_state_ids", []),
            *development.get("invariant_state_ids", []),
        )
    }
    ranked = sorted(
        (state_id for state_id in state_ids if state_id in states_by_id),
        key=lambda state_id: (
            state_id not in development_state_ids,
            -len(states_by_id[state_id].get("supporting_evidence_ids", [])),
            -int(states_by_id[state_id].get("valid_from_scene", 0)),
            state_id,
        ),
    )
    return ranked[:8]


def _select_compact_development_ids(
    development_ids: list[str], *, developments_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    ranked = sorted(
        (development_id for development_id in development_ids if development_id in developments_by_id),
        key=lambda development_id: (
            -len(
                developments_by_id[development_id].get("evidence_catalyst_ids", [])
                + developments_by_id[development_id].get("evidence_after_ids", [])
            ),
            int(developments_by_id[development_id].get("effective_from_scene", 0)),
            development_id,
        ),
    )
    return ranked[:4]


def _state_claim(
    state: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]], current_scene_order: int
) -> dict[str, Any]:
    evidence = sorted(
        {
            int(evidence_by_id[evidence_id]["scene_order"])
            for evidence_id in state.get("supporting_evidence_ids", [])
            if evidence_id in evidence_by_id
            and 0 < int(evidence_by_id[evidence_id]["scene_order"]) <= current_scene_order
        }
    )
    if not evidence and int(state.get("valid_from_scene", 0)) <= current_scene_order:
        evidence = [int(state["valid_from_scene"])]
    return {"claim": str(state.get("state_value") or "").strip(), "evidence_scene_orders": evidence, "state_id": state["state_id"]}


def _development_claim(
    development: dict[str, Any],
    *,
    states_by_id: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    previous_scene_order: int,
    current_scene_order: int,
) -> dict[str, Any]:
    before = [states_by_id[state_id].get("state_value", "") for state_id in development.get("before_state_ids", []) if state_id in states_by_id]
    after = [states_by_id[state_id].get("state_value", "") for state_id in development.get("resulting_state_ids", []) if state_id in states_by_id]
    target = str(development.get("target_id_or_text") or "the focal character")
    dimension = str(development.get("dimension") or "state")
    before_text = "; ".join(str(value) for value in before if value) or "the prior state"
    after_text = "; ".join(str(value) for value in after if value) or "a changed state"
    evidence_ids = [
        *development.get("evidence_before_ids", []),
        *development.get("evidence_catalyst_ids", []),
        *development.get("evidence_after_ids", []),
    ]
    evidence = sorted(
        {
            int(evidence_by_id[evidence_id]["scene_order"])
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
            and 0 < int(evidence_by_id[evidence_id]["scene_order"]) <= current_scene_order
        }
    )
    return {
        "claim": f"{target}'s {dimension} develops from {before_text} to {after_text}.",
        "evidence_scene_orders": evidence,
        "development_id": development["development_id"],
    }


def validate_task1_state_update_assets(pair: dict[str, Any]) -> dict[str, Any]:
    """Validate the deterministic RSU/ASU contract used by temporal output."""
    errors: list[str] = []
    reference = pair.get("reference", {})
    autoregressive = pair.get("autoregressive", {})
    reference_ids: list[str] = []
    prediction_ids: list[str] = []
    for trajectory in reference.get("trajectories", []):
        previous_scene = 0
        previous_target: list[dict[str, Any]] = []
        for checkpoint in trajectory.get("checkpoints", []):
            checkpoint_id = checkpoint.get("checkpoint_id", "")
            interval = checkpoint.get("screenplay_interval", {})
            start = interval.get("start_scene_order_exclusive")
            end = interval.get("end_scene_order_inclusive")
            if start != previous_scene or not isinstance(end, int) or end <= start:
                errors.append(f"{checkpoint_id}: non-adjacent screenplay interval")
            target = checkpoint.get("evaluation_target", {})
            for field, limit in (("current_state", 8), ("developments_since_previous_checkpoint", 4)):
                rows = target.get(field, [])
                if len(rows) > limit:
                    errors.append(f"{checkpoint_id}: {field} exceeds {limit}")
                for row in rows:
                    evidence = row.get("evidence_scene_orders", [])
                    if not evidence or any(not isinstance(value, int) or value <= 0 or value > end for value in evidence):
                        errors.append(f"{checkpoint_id}: invalid {field} evidence")
                    if field == "developments_since_previous_checkpoint" and not any(value > start for value in evidence):
                        errors.append(f"{checkpoint_id}: development lacks interval evidence")
            memory = checkpoint.get("entity_memory", {})
            if memory.get("previous_state_visible_to_extractor") is not False:
                errors.append(f"{checkpoint_id}: previous state visible to extractor")
            if "previous_state" in memory.get("extractor_inputs", []):
                errors.append(f"{checkpoint_id}: previous state in extractor inputs")
            expected_previous = [] if not previous_target else [
                {"claim": row["claim"], "evidence_scene_orders": list(row["evidence_scene_orders"])}
                for row in previous_target
            ]
            if checkpoint.get("model_input", {}).get("previous_state") != expected_previous:
                errors.append(f"{checkpoint_id}: RSU previous state drift")
            reference_ids.append(checkpoint_id)
            previous_scene = end
            previous_target = list(target.get("current_state", []))
    for trajectory in autoregressive.get("trajectories", []):
        for checkpoint in trajectory.get("checkpoints", []):
            prediction_ids.append(checkpoint.get("checkpoint_id", ""))
    if reference_ids != prediction_ids:
        errors.append("RSU/ASU checkpoint coverage differs")
    protocol = reference.get("context_protocol", {})
    if protocol != autoregressive.get("context_protocol"):
        errors.append("RSU/ASU context protocol differs")
    return {"status": "passed" if not errors else "failed", "error_count": len(errors), "errors": errors}


def materialize_role_snapshots(
    *,
    movie_id: str,
    registry: dict[str, Any],
    checkpoints: dict[str, Any],
    state_ledger: dict[str, Any],
    evidence_bank: dict[str, Any],
    persona_bank: dict[str, Any],
    prompt_candidates: dict[str, Any],
) -> dict[str, Any]:
    characters = {
        item["character_id"]: item
        for item in registry["characters"]
        if item["task3_single_turn_eligible"]
    }
    states_by_id = {item["state_id"]: item for item in state_ledger["states"]}
    evidence_by_id = {
        item["evidence_id"]: item for item in evidence_bank["evidence_units"]
    }
    persona_by_id = {
        item["persona_evidence_id"]: item
        for item in persona_bank["persona_evidence"]
    }
    prompts_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompt_candidates["prompts"]:
        prompts_by_checkpoint[prompt["checkpoint_id"]].append(prompt)
    snapshots = []
    for checkpoint in checkpoints["checkpoints"]:
        character_id = checkpoint["character_id"]
        if character_id not in characters:
            continue
        character = characters[character_id]
        checkpoint_prompts = prompts_by_checkpoint[checkpoint["checkpoint_id"]]
        prompt_evidence_ids = set(
            unique_text(
                evidence_id
                for prompt in checkpoint_prompts
                for evidence_id in prompt["supporting_evidence_ids"]
            )
        )
        visible_persona_ids = unique_text(
            persona_id
            for prompt in checkpoint_prompts
            for persona_id in prompt["style_evidence_ids"]
            if persona_id in persona_by_id
        )
        if not visible_persona_ids:
            visible_persona_ids = unique_text(
                persona_id
                for persona_id in checkpoint["persona_evidence_ids"]
                if persona_id in persona_by_id
                and bool(
                    set(persona_by_id[persona_id]["supporting_evidence_ids"])
                    & prompt_evidence_ids
                )
            )
        visible_dialogue_ids = unique_text(
            evidence_id
            for evidence_id in prompt_evidence_ids
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id]["evidence_type"] == "dialogue"
            and evidence_by_id[evidence_id]["speaker_character_id"] == character_id
            and int(evidence_by_id[evidence_id]["scene_order"])
            <= int(checkpoint["scene_order"])
        )
        relationship_state_evidence = set(
            evidence_id
            for state_id in checkpoint["active_state_ids"]
            if state_id in states_by_id
            for state in [states_by_id[state_id]]
            if state["dimension"] == "relationship"
            for evidence_id in state["supporting_evidence_ids"]
        )
        relation_evidence = unique_text(
            evidence_id for evidence_id in prompt_evidence_ids
            if evidence_id in relationship_state_evidence
            if evidence_id in evidence_by_id
            and int(evidence_by_id[evidence_id]["scene_order"])
            <= int(checkpoint["scene_order"])
        )
        body = {
            "character_id": character_id,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "scene_order": int(checkpoint["scene_order"]),
            "identity_context": {
                "canonical_name": character_name_at_scene(
                    character, int(checkpoint["scene_order"])
                ),
                "identity_lineage_name": character["canonical_name"],
                "aliases": character["aliases"],
                "identity_phases": character["identity_phases"],
            },
            "visible_persona_evidence_ids": visible_persona_ids,
            "visible_dialogue_exemplar_ids": visible_dialogue_ids,
            "visible_memory_fact_ids": checkpoint["accessible_fact_ids"],
            "visible_relation_evidence_ids": relation_evidence,
        }
        snapshot_hash = sha256_json(body)
        snapshot_id = stable_id(
            "role-snapshot",
            movie_id,
            character_id,
            checkpoint["checkpoint_id"],
            snapshot_hash,
        )
        snapshots.append(
            {
                "role_snapshot_id": snapshot_id,
                **body,
                "snapshot_hash": snapshot_hash,
                "validation_status": "silver_candidate",
            }
        )
    snapshots.sort(
        key=lambda item: (
            item["character_id"], item["checkpoint_id"], item["role_snapshot_id"]
        )
    )
    return {
        "schema_version": "stage_role_snapshot_index_v1",
        "movie_id": movie_id,
        "snapshot_count": len(snapshots),
        "role_snapshots": snapshots,
    }


def materialize_task3_single_turn(
    *,
    movie_id: str,
    language: str,
    registry: dict[str, Any],
    role_snapshots: dict[str, Any],
    prompt_candidates: dict[str, Any],
    epistemic_ledger: dict[str, Any],
) -> dict[str, Any]:
    characters = {item["character_id"]: item for item in registry["characters"]}
    snapshots_by_checkpoint = {
        item["checkpoint_id"]: item for item in role_snapshots["role_snapshots"]
    }
    access_by_id = {
        item["access_id"]: item for item in epistemic_ledger["access_records"]
    }
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for prompt in prompt_candidates["prompts"]:
        pair_group = prompt.get("pair_group", "")
        if pair_group:
            pair_counts[(prompt["character_id"], pair_group)] += 1
    instances = []
    for prompt in prompt_candidates["prompts"]:
        checkpoint_id = prompt["checkpoint_id"]
        if checkpoint_id not in snapshots_by_checkpoint:
            raise ValueError(
                f"Task 3 prompt has no checkpoint role snapshot: {prompt['prompt_id']}"
            )
        snapshot = snapshots_by_checkpoint[checkpoint_id]
        character = characters[prompt["character_id"]]
        required_memory_ids = unique_text(
            access_by_id[access_id]["fact_or_event_id"]
            for access_id in prompt["required_access_ids"]
            if access_id in access_by_id
        )
        pair_group = prompt["pair_group"]
        pair_group_id = (
            stable_id("task3-pair", movie_id, character["character_id"], pair_group)
            if pair_group
            and pair_counts[(character["character_id"], pair_group)] >= 2
            else ""
        )
        instances.append(
            {
                "instance_id": stable_id(
                    "task3-instance", movie_id, prompt["prompt_id"]
                ),
                "movie_id": movie_id,
                "language": language,
                "character_id": character["character_id"],
                "character": character_name_at_scene(
                    character, int(snapshot["scene_order"])
                ),
                "interaction_format": "single_turn",
                "model_input": {
                    "role_snapshot_ref": snapshot["role_snapshot_id"],
                    "interaction_context": prompt["interaction_context"],
                    "dialogue_history": [],
                    "current_user_turn": prompt["current_user_turn"],
                },
                "evaluator_reference": {
                    "checkpoint_id": checkpoint_id,
                    "prompt_id": prompt["prompt_id"],
                    "prompt_family": prompt["prompt_family"],
                    "expected_stances": prompt["expected_stances"],
                    "acceptable_state_fact_ids": prompt["state_ids"],
                    "required_memory_fact_ids": required_memory_ids,
                    "supporting_evidence_ids": prompt["supporting_evidence_ids"],
                    "contradicting_fact_ids": prompt["contradicting_fact_ids"],
                    "unknown_fact_ids": prompt["unknown_fact_ids"],
                    "future_forbidden_fact_ids": prompt[
                        "future_forbidden_fact_ids"
                    ],
                    "style_evidence_ids": prompt["style_evidence_ids"],
                    "boundary_risk_type": prompt["boundary_risk_type"],
                    "paired_prompt_group_id": pair_group_id,
                },
                "validation_status": "silver_candidate",
            }
        )
    instances.sort(
        key=lambda item: (
            item["character_id"],
            item["evaluator_reference"]["checkpoint_id"],
            item["instance_id"],
        )
    )
    return {
        "schema_version": "stage_task3_checkpoint_single_turn_v1",
        "task": "checkpoint_conditioned_in_script_character_role_play",
        "interaction_format": "single_turn",
        "movie_id": movie_id,
        "language": language,
        "character_count": len({item["character_id"] for item in instances}),
        "instance_count": len(instances),
        "pair_group_count": len(
            {
                item["evaluator_reference"]["paired_prompt_group_id"]
                for item in instances
                if item["evaluator_reference"]["paired_prompt_group_id"]
            }
        ),
        "instances": instances,
    }


def _state_active(state: dict[str, Any], scene_order: int) -> bool:
    if scene_order <= 0:
        return False
    return int(state["valid_from_scene"]) <= scene_order and (
        state["valid_until_scene"] is None
        or scene_order <= int(state["valid_until_scene"])
    )


def _state_track_key(state: dict[str, Any]) -> tuple[str, str]:
    return (
        str(state.get("dimension", "")),
        normalize_name(state.get("target_id_or_text", "")),
    )
