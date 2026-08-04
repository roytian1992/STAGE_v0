from __future__ import annotations

"""Cross-asset invariants for a frozen character-temporal release."""

from collections import Counter, defaultdict
from typing import Any

from ..models import normalize_name


def validate_temporal_release(
    *,
    scene_count: int,
    graph: dict[str, Any],
    registry: dict[str, Any],
    evidence_bank: dict[str, Any],
    state_ledger: dict[str, Any],
    development_graph: dict[str, Any],
    epistemic_ledger: dict[str, Any],
    persona_bank: dict[str, Any],
    checkpoints: dict[str, Any],
    task1: dict[str, Any],
    role_snapshots: dict[str, Any],
    task3: dict[str, Any],
    task1_state_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    graph_node_index = {item["id"]: item for item in graph.get("nodes", [])}
    graph_nodes = set(graph_node_index)
    characters = _unique_index(
        registry.get("characters", []), "character_id", "character", errors
    )
    evidence = _unique_index(
        evidence_bank.get("evidence_units", []), "evidence_id", "evidence", errors
    )
    states = _unique_index(state_ledger.get("states", []), "state_id", "state", errors)
    developments = _unique_index(
        development_graph.get("developments", []),
        "development_id",
        "development",
        errors,
    )
    accesses = _unique_index(
        epistemic_ledger.get("access_records", []), "access_id", "access", errors
    )
    persona = _unique_index(
        persona_bank.get("persona_evidence", []),
        "persona_evidence_id",
        "persona evidence",
        errors,
    )
    checkpoint_index = _unique_index(
        checkpoints.get("checkpoints", []), "checkpoint_id", "checkpoint", errors
    )
    snapshots = _unique_index(
        role_snapshots.get("role_snapshots", []),
        "role_snapshot_id",
        "role snapshot",
        errors,
    )

    for character in characters.values():
        phases = sorted(
            character.get("identity_phases", []),
            key=lambda item: (int(item["valid_from_scene"]), item.get("phase_id", "")),
        )
        segments = character.get("identity_segments", [])
        source_ids = set(character.get("source_entity_ids", []))
        if segments and not phases:
            errors.append(
                f"Identity-lineage character has no phases: {character['character_id']}"
            )
        if any(
            int(item.get("valid_from_scene", 0)) < int(character["first_scene_order"])
            or int(item.get("valid_until_scene", 0)) > int(character["last_scene_order"])
            for item in [*phases, *segments]
        ):
            errors.append(
                f"Character identity phase/segment exceeds character span: {character['character_id']}"
            )
        if any(
            int(left["valid_until_scene"]) >= int(right["valid_from_scene"])
            for left, right in zip(phases, phases[1:])
        ):
            errors.append(f"Character identity phases overlap: {character['character_id']}")
        if any(item.get("source_character_id") not in source_ids for item in segments):
            errors.append(
                f"Character identity segment has undeclared source: {character['character_id']}"
            )

    for item in evidence.values():
        if not 1 <= int(item.get("scene_order", 0)) <= scene_count:
            errors.append(f"Evidence scene order outside screenplay: {item['evidence_id']}")
        if int(item.get("char_end", 0)) <= int(item.get("char_start", -1)):
            errors.append(f"Evidence span is empty or reversed: {item['evidence_id']}")
        source_sha256 = str(item.get("source_sha256") or "")
        if len(source_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in source_sha256.casefold()
        ):
            errors.append(f"Evidence source hash is invalid: {item['evidence_id']}")
        for character_id in [
            item.get("speaker_character_id"),
            *item.get("participant_character_ids", []),
            *item.get("direct_observer_character_ids", []),
            *item.get("addressee_character_ids", []),
        ]:
            if character_id and character_id not in characters:
                errors.append(
                    f"Evidence references unknown character {character_id}: {item['evidence_id']}"
                )

    states_by_character_target: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in states.values():
        if item.get("character_id") not in characters:
            errors.append(f"State references unknown character: {item['state_id']}")
        if not _known(item.get("supporting_evidence_ids", []), evidence):
            errors.append(f"State has unknown or empty evidence: {item['state_id']}")
        valid_from = int(item.get("valid_from_scene", 0))
        valid_until = item.get("valid_until_scene")
        if not 1 <= valid_from <= scene_count:
            errors.append(f"State valid_from outside screenplay: {item['state_id']}")
        if valid_until is not None and not valid_from <= int(valid_until) <= scene_count:
            errors.append(f"State valid_until is invalid: {item['state_id']}")
        key = (
            item.get("character_id", ""),
            item.get("dimension", ""),
            normalize_name(item.get("target_id_or_text", "")),
        )
        states_by_character_target[key].append(item)
    for key, values in states_by_character_target.items():
        ordered = sorted(values, key=lambda item: int(item["valid_from_scene"]))
        for left, right in zip(ordered, ordered[1:]):
            left_until = left["valid_until_scene"]
            if left_until is not None and int(left_until) >= int(right["valid_from_scene"]):
                warnings.append(
                    f"Overlapping state intervals for {key}: {left['state_id']} / {right['state_id']}"
                )

    for item in developments.values():
        if item.get("character_id") not in characters:
            errors.append(f"Development references unknown character: {item['development_id']}")
        before = [states.get(value) for value in item.get("before_state_ids", [])]
        resulting = [states.get(value) for value in item.get("resulting_state_ids", [])]
        invariants = [states.get(value) for value in item.get("invariant_state_ids", [])]
        if any(value is None for value in [*before, *resulting, *invariants]):
            errors.append(f"Development references unknown states: {item['development_id']}")
            continue
        if not resulting:
            errors.append(f"Development has no resulting state: {item['development_id']}")
        for state in [*before, *resulting]:
            if state and (
                state["character_id"] != item["character_id"]
                or state["dimension"] != item["dimension"]
                or normalize_name(state["target_id_or_text"])
                != normalize_name(item["target_id_or_text"])
            ):
                errors.append(
                    f"Development before/after state mismatch: {item['development_id']}"
                )
        if not set(item.get("catalyst_event_ids", [])) <= graph_nodes:
            errors.append(f"Development has unknown catalyst: {item['development_id']}")
        if not set(item.get("downstream_consequence_ids", [])) <= graph_nodes:
            errors.append(f"Development has unknown consequence: {item['development_id']}")
        evidence_ids = [
            evidence_id
            for key in (
                "evidence_before_ids",
                "evidence_catalyst_ids",
                "evidence_after_ids",
            )
            for evidence_id in item.get(key, [])
        ]
        if not evidence_ids or not set(evidence_ids) <= set(evidence):
            errors.append(f"Development has invalid evidence: {item['development_id']}")
        if int(item.get("effective_from_scene", 0)) <= 0 or int(
            item.get("consequence_visible_from_scene", 0)
        ) < int(item.get("effective_from_scene", 0)):
            errors.append(f"Development temporal boundary is invalid: {item['development_id']}")
        before_scenes = [int(value["valid_from_scene"]) for value in before if value]
        result_scenes = [int(value["valid_from_scene"]) for value in resulting if value]
        if before_scenes and result_scenes and max(before_scenes) >= min(result_scenes):
            errors.append(
                f"Development state transition is not forward: {item['development_id']}"
            )
        if result_scenes and not (
            max(before_scenes, default=0)
            <= int(item.get("effective_from_scene", 0))
            <= min(result_scenes)
            <= int(item.get("consequence_visible_from_scene", 0))
        ):
            errors.append(
                f"Development boundaries disagree with state scenes: {item['development_id']}"
            )

    access_by_character_fact: set[tuple[str, str]] = set()
    accesses_by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in accesses.values():
        pair = (item.get("character_id", ""), item.get("fact_or_event_id", ""))
        if pair in access_by_character_fact:
            errors.append(f"Duplicate character/fact epistemic record: {pair}")
        access_by_character_fact.add(pair)
        accesses_by_character[item.get("character_id", "")].append(item)
        if item.get("character_id") not in characters:
            errors.append(f"Access references unknown character: {item['access_id']}")
        if item.get("fact_or_event_id") not in graph_nodes:
            errors.append(f"Access references unknown graph fact: {item['access_id']}")
        if item.get("source_character_id") and item["source_character_id"] not in characters:
            errors.append(f"Access has unknown source character: {item['access_id']}")
        if item.get("access_type") == "unknown":
            if item.get("acquired_at_scene") is not None:
                errors.append(f"Unknown access has acquisition scene: {item['access_id']}")
        elif item.get("acquired_at_scene") is None:
            errors.append(f"Known access lacks acquisition scene: {item['access_id']}")
        if not set(item.get("supporting_evidence_ids", [])) <= set(evidence):
            errors.append(f"Access references unknown evidence: {item['access_id']}")

    for item in persona.values():
        if item.get("character_id") not in characters:
            errors.append(f"Persona evidence references unknown character: {item['persona_evidence_id']}")
        if not _known(item.get("supporting_evidence_ids", []), evidence):
            errors.append(f"Persona evidence has unknown or empty evidence: {item['persona_evidence_id']}")
        established = int(item.get("established_from_scene", 0))
        superseded = item.get("superseded_at_scene")
        if not 1 <= established <= scene_count:
            errors.append(f"Persona establishment outside screenplay: {item['persona_evidence_id']}")
        if superseded is not None and int(superseded) < established:
            errors.append(f"Persona interval is invalid: {item['persona_evidence_id']}")

    checkpoints_by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in checkpoint_index.values():
        checkpoints_by_character[item.get("character_id", "")].append(item)
        if item.get("character_id") not in characters:
            errors.append(f"Checkpoint references unknown character: {item['checkpoint_id']}")
        if not 1 <= int(item.get("scene_order", 0)) <= scene_count:
            errors.append(f"Checkpoint outside screenplay: {item['checkpoint_id']}")
        for key, known in (
            ("active_state_ids", states),
            ("new_development_ids", developments),
            ("invariant_state_ids", states),
            ("persona_evidence_ids", persona),
            ("dialogue_exemplar_ids", evidence),
        ):
            if not set(item.get(key, [])) <= set(known):
                errors.append(f"Checkpoint {key} has unknown references: {item['checkpoint_id']}")
        if not set(item.get("accessible_fact_ids", [])) <= graph_nodes:
            errors.append(f"Checkpoint accessible facts are unknown: {item['checkpoint_id']}")
        if not set(item.get("unknown_fact_ids", [])) <= graph_nodes:
            errors.append(f"Checkpoint unknown facts are invalid: {item['checkpoint_id']}")
        if not set(item.get("future_forbidden_fact_ids", [])) <= graph_nodes:
            errors.append(f"Checkpoint future facts are invalid: {item['checkpoint_id']}")
        partitions = [
            set(item.get("accessible_fact_ids", [])),
            set(item.get("unknown_fact_ids", [])),
            set(item.get("future_forbidden_fact_ids", [])),
        ]
        if any(left & right for offset, left in enumerate(partitions) for right in partitions[offset + 1 :]):
            errors.append(f"Checkpoint fact partitions overlap: {item['checkpoint_id']}")
        expected_accessible: set[str] = set()
        expected_unknown: set[str] = set()
        expected_future: set[str] = set()
        scene_order = int(item.get("scene_order", 0))
        for access in accesses_by_character[item.get("character_id", "")]:
            fact_id = access["fact_or_event_id"]
            source_order = min(access.get("fact_source_scene_orders", []), default=10**9)
            acquired = access.get("acquired_at_scene")
            if source_order > scene_order:
                expected_future.add(fact_id)
            elif (
                access.get("access_type") != "unknown"
                and acquired is not None
                and int(acquired) <= scene_order
            ):
                expected_accessible.add(fact_id)
            else:
                expected_unknown.add(fact_id)
        actual_partitions = (
            set(item.get("accessible_fact_ids", [])),
            set(item.get("unknown_fact_ids", [])),
            set(item.get("future_forbidden_fact_ids", [])),
        )
        expected_partitions = (expected_accessible, expected_unknown, expected_future)
        if actual_partitions != expected_partitions:
            errors.append(f"Checkpoint fact partitions disagree with epistemic time: {item['checkpoint_id']}")
        expected_controls = set()
        if not item.get("new_development_ids"):
            expected_controls.add("no_change")
        if expected_unknown:
            expected_controls.add("inaccessible")
        if any(
            development
            and int(development["consequence_visible_from_scene"]) > scene_order
            for development_id in item.get("new_development_ids", [])
            for development in [developments.get(development_id)]
        ):
            expected_controls.add("delayed_consequence")
        if set(item.get("control_types", [])) != expected_controls:
            errors.append(
                f"Checkpoint controls disagree with temporal evidence: {item['checkpoint_id']}"
            )
        for state_id in item.get("active_state_ids", []):
            state = states.get(state_id)
            if state and not _state_active(state, int(item["scene_order"])):
                errors.append(f"Checkpoint contains inactive state: {item['checkpoint_id']} / {state_id}")
            if state and state["character_id"] != item.get("character_id"):
                errors.append(f"Checkpoint contains another character's state: {item['checkpoint_id']}")
        for development_id in item.get("new_development_ids", []):
            development = developments.get(development_id)
            if development and (
                development["character_id"] != item.get("character_id")
                or int(development["effective_from_scene"]) != scene_order
            ):
                errors.append(f"Checkpoint development has wrong owner or time: {item['checkpoint_id']}")
        if item.get("checkpoint_type") == "change" and not item.get("new_development_ids"):
            errors.append(f"Change checkpoint has no development: {item['checkpoint_id']}")
        for persona_id in item.get("persona_evidence_ids", []):
            record = persona.get(persona_id)
            if record and int(record["established_from_scene"]) > int(item["scene_order"]):
                errors.append(f"Checkpoint contains future persona evidence: {item['checkpoint_id']}")
            if record and record["character_id"] != item.get("character_id"):
                errors.append(f"Checkpoint contains another character's persona: {item['checkpoint_id']}")
        for evidence_id in item.get("dialogue_exemplar_ids", []):
            record = evidence.get(evidence_id)
            if record and int(record["scene_order"]) > int(item["scene_order"]):
                errors.append(f"Checkpoint contains future dialogue exemplar: {item['checkpoint_id']}")
    for character_id, values in checkpoints_by_character.items():
        ordered = sorted(values, key=lambda item: int(item["scene_order"]))
        character = characters.get(character_id, {})
        expected_first = int(character.get("first_scene_order") or 0)
        expected_last = int(character.get("last_scene_order") or 0)
        types = Counter(item["checkpoint_type"] for item in ordered)
        if types["baseline"] != 1 or types["final"] != 1:
            errors.append(f"Character checkpoint sequence lacks baseline/final: {character_id}")
        if any(
            item["previous_checkpoint_id"]
            != (ordered[position - 1]["checkpoint_id"] if position else "")
            for position, item in enumerate(ordered)
        ):
            errors.append(f"Character checkpoint previous links are invalid: {character_id}")
        if ordered and (
            ordered[0]["checkpoint_type"] != "baseline"
            or ordered[-1]["checkpoint_type"] != "final"
            or int(ordered[0]["scene_order"]) != expected_first
            or int(ordered[-1]["scene_order"]) != expected_last
        ):
            errors.append(f"Character checkpoint endpoints are invalid: {character_id}")
        covered_developments = [
            development_id
            for checkpoint in ordered
            for development_id in checkpoint.get("new_development_ids", [])
        ]
        expected_developments = {
            development_id
            for development_id, development in developments.items()
            if development.get("character_id") == character_id
        }
        if (
            set(covered_developments) != expected_developments
            or len(covered_developments) != len(expected_developments)
        ):
            errors.append(f"Character checkpoints do not cover developments exactly once: {character_id}")
    task1_instances = _unique_index(
        task1.get("instances", []), "instance_id", "Task 1 instance", errors
    )
    task1_eligible_count = sum(
        bool(item.get("task1_eligible")) for item in characters.values()
    )
    if not task1_instances and task1_eligible_count:
        errors.append("Task 1 release has no instances")
    elif not task1_instances:
        warnings.append("Task 1 has no eligible characters under the frozen thresholds")
    for item in task1_instances.values():
        model_input = item.get("model_input", {})
        reference = item.get("evaluator_reference", {})
        checkpoint = checkpoint_index.get(reference.get("checkpoint_id"))
        if not checkpoint:
            errors.append(f"Task 1 references unknown checkpoint: {item['instance_id']}")
            continue
        if model_input.get("current_checkpoint_scene_order") != checkpoint["scene_order"]:
            errors.append(f"Task 1 prefix/checkpoint mismatch: {item['instance_id']}")
        prefix = model_input.get("screenplay_prefix_ref", {})
        if prefix.get("end_scene_order") != checkpoint["scene_order"]:
            errors.append(f"Task 1 prefix boundary mismatch: {item['instance_id']}")
        if (
            prefix.get("start_scene_order") != 1
            or prefix.get("scene_count") != checkpoint["scene_order"]
        ):
            errors.append(f"Task 1 prefix coverage mismatch: {item['instance_id']}")
        if not set(reference.get("gold_current_state_ids", [])) <= set(states):
            errors.append(f"Task 1 has unknown gold states: {item['instance_id']}")
        tracked_state_keys = {
            _state_track_key(states[state_id])
            for development in developments.values()
            if development.get("character_id") == checkpoint.get("character_id")
            for state_id in [
                *development.get("before_state_ids", []),
                *development.get("resulting_state_ids", []),
                *development.get("invariant_state_ids", []),
            ]
            if state_id in states
        }
        expected_gold_states = {
            state_id
            for state_id in checkpoint.get("active_state_ids", [])
            if state_id in states
            and _state_track_key(states[state_id]) in tracked_state_keys
        }
        if set(reference.get("gold_current_state_ids", [])) != expected_gold_states:
            errors.append(f"Task 1 gold states disagree with checkpoint: {item['instance_id']}")
        if not reference.get("gold_current_state_ids") and not reference.get(
            "gold_development_ids"
        ):
            errors.append(f"Task 1 instance has empty tracking gold: {item['instance_id']}")
        if set(reference.get("checkpoint_control_types", [])) != set(
            checkpoint.get("control_types", [])
        ):
            errors.append(f"Task 1 controls disagree with checkpoint: {item['instance_id']}")
        if not set(reference.get("gold_development_ids", [])) <= set(developments):
            errors.append(f"Task 1 has unknown gold developments: {item['instance_id']}")
        if not set(reference.get("supporting_evidence_ids", [])) <= set(evidence):
            errors.append(f"Task 1 has unknown evidence: {item['instance_id']}")
        if set(model_input) & {
            "gold_current_state_ids",
            "gold_development_ids",
            "unknown_fact_ids",
            "future_forbidden_fact_ids",
            "checkpoint_control_type",
            "checkpoint_control_types",
        }:
            errors.append(f"Task 1 model input leaks evaluator fields: {item['instance_id']}")

    snapshots_by_checkpoint: dict[str, dict[str, Any]] = {}
    for item in snapshots.values():
        checkpoint = checkpoint_index.get(item.get("checkpoint_id"))
        if not checkpoint:
            errors.append(f"Role snapshot has unknown checkpoint: {item['role_snapshot_id']}")
            continue
        if checkpoint["checkpoint_id"] in snapshots_by_checkpoint:
            errors.append(f"Multiple role snapshots for checkpoint: {checkpoint['checkpoint_id']}")
        snapshots_by_checkpoint[checkpoint["checkpoint_id"]] = item
        if item.get("character_id") != checkpoint.get("character_id"):
            errors.append(f"Role snapshot character mismatch: {item['role_snapshot_id']}")
        if not set(item.get("visible_persona_evidence_ids", [])) <= set(
            checkpoint["persona_evidence_ids"]
        ):
            errors.append(f"Role snapshot persona leakage: {item['role_snapshot_id']}")
        if not set(item.get("visible_dialogue_exemplar_ids", [])) <= set(
            checkpoint["dialogue_exemplar_ids"]
        ):
            errors.append(f"Role snapshot dialogue leakage: {item['role_snapshot_id']}")
        if not set(item.get("visible_memory_fact_ids", [])) <= set(
            checkpoint["accessible_fact_ids"]
        ):
            errors.append(f"Role snapshot memory leakage: {item['role_snapshot_id']}")
        if set(item.get("visible_memory_fact_ids", [])) & (
            set(checkpoint["unknown_fact_ids"])
            | set(checkpoint["future_forbidden_fact_ids"])
        ):
            errors.append(f"Role snapshot contains forbidden memory: {item['role_snapshot_id']}")
        relation_evidence = set(item.get("visible_relation_evidence_ids", []))
        expected_relation_evidence = {
            evidence_id
            for state_id in checkpoint.get("active_state_ids", [])
            for state in [states.get(state_id)]
            if state and state.get("dimension") == "relationship"
            for evidence_id in state.get("supporting_evidence_ids", [])
        }
        if not relation_evidence <= expected_relation_evidence:
            errors.append(f"Role snapshot relation evidence leakage: {item['role_snapshot_id']}")
        if any(
            int(evidence[evidence_id]["scene_order"]) > int(checkpoint["scene_order"])
            for evidence_id in relation_evidence
            if evidence_id in evidence
        ):
            errors.append(f"Role snapshot has future relation evidence: {item['role_snapshot_id']}")

    task3_instances = _unique_index(
        task3.get("instances", []), "instance_id", "Task 3 instance", errors
    )
    task3_eligible_count = sum(
        bool(item.get("task3_single_turn_eligible")) for item in characters.values()
    )
    if not task3_instances and task3_eligible_count:
        errors.append("Task 3 single-turn release has no instances")
    elif not task3_instances:
        warnings.append(
            "Task 3 single-turn has no eligible characters under the frozen thresholds"
        )
    pair_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in task3_instances.values():
        model_input = item.get("model_input", {})
        reference = item.get("evaluator_reference", {})
        if set(model_input) & {
            "expected_stances",
            "future_forbidden_fact_ids",
            "unknown_fact_ids",
            "checkpoint_type",
        }:
            errors.append(f"Task 3 actor input leaks evaluator fields: {item['instance_id']}")
        snapshot = snapshots.get(model_input.get("role_snapshot_ref"))
        checkpoint = checkpoint_index.get(reference.get("checkpoint_id"))
        if not snapshot or not checkpoint:
            errors.append(f"Task 3 references unknown snapshot/checkpoint: {item['instance_id']}")
            continue
        if snapshot["checkpoint_id"] != checkpoint["checkpoint_id"]:
            errors.append(f"Task 3 snapshot/checkpoint mismatch: {item['instance_id']}")
        if item.get("character_id") != checkpoint.get("character_id"):
            errors.append(f"Task 3 character/checkpoint mismatch: {item['instance_id']}")
        if not set(reference.get("acceptable_state_fact_ids", [])) <= set(
            checkpoint.get("active_state_ids", [])
        ):
            errors.append(f"Task 3 state reference is not active: {item['instance_id']}")
        if not set(reference.get("required_memory_fact_ids", [])) <= set(
            checkpoint.get("accessible_fact_ids", [])
        ):
            errors.append(f"Task 3 memory reference is not accessible: {item['instance_id']}")
        if not set(reference.get("unknown_fact_ids", [])) <= set(
            checkpoint["unknown_fact_ids"]
        ):
            errors.append(f"Task 3 unknown-fact reference mismatch: {item['instance_id']}")
        if not set(reference.get("future_forbidden_fact_ids", [])) <= set(
            checkpoint["future_forbidden_fact_ids"]
        ):
            errors.append(f"Task 3 future-fact reference mismatch: {item['instance_id']}")
        if not set(reference.get("supporting_evidence_ids", [])) <= set(evidence):
            errors.append(f"Task 3 has unknown evidence: {item['instance_id']}")
        if not set(reference.get("style_evidence_ids", [])) <= set(persona):
            errors.append(f"Task 3 has unknown style evidence: {item['instance_id']}")
        pair_id = str(reference.get("paired_prompt_group_id") or "")
        if pair_id:
            pair_members[pair_id].append(item)

    for pair_id, members in pair_members.items():
        if len(members) != 2:
            errors.append(
                f"Task 3 paired prompt group must contain exactly two instances: {pair_id}"
            )
        if len({item.get("character_id") for item in members}) != 1:
            errors.append(f"Task 3 paired prompt group crosses characters: {pair_id}")
        references = [item.get("evaluator_reference", {}) for item in members]
        if len({item.get("checkpoint_id") for item in references}) != 2:
            errors.append(f"Task 3 paired prompt group does not evolve over checkpoints: {pair_id}")
        if len({item.get("prompt_family") for item in references}) != 1:
            errors.append(f"Task 3 paired prompt group crosses prompt families: {pair_id}")
        state_anchor_sets = [
            {
                (
                    state["dimension"],
                    normalize_name(state["target_id_or_text"]),
                )
                for state_id in reference.get("acceptable_state_fact_ids", [])
                for state in [states.get(state_id)]
                if state
            }
            for reference in references
        ]
        fact_anchor_sets = [
            set(reference.get("required_memory_fact_ids", []))
            | set(reference.get("contradicting_fact_ids", []))
            | set(reference.get("unknown_fact_ids", []))
            | set(reference.get("future_forbidden_fact_ids", []))
            for reference in references
        ]
        if len(members) == 2 and not (
            state_anchor_sets[0] & state_anchor_sets[1]
            or fact_anchor_sets[0] & fact_anchor_sets[1]
        ):
            errors.append(
                f"Task 3 paired prompt group lacks a shared semantic anchor: {pair_id}"
            )
    if "pair_group_count" in task3 and int(task3["pair_group_count"]) != len(pair_members):
        errors.append("Declared pair_group_count disagrees with Task 3 references")

    _check_declared_count(registry, "characters", "character_count", errors)
    _check_declared_count(evidence_bank, "evidence_units", "evidence_count", errors)
    _check_declared_count(state_ledger, "states", "state_count", errors)
    _check_declared_count(development_graph, "developments", "development_count", errors)
    _check_declared_count(epistemic_ledger, "access_records", "record_count", errors)
    _check_declared_count(persona_bank, "persona_evidence", "evidence_count", errors)
    _check_declared_count(role_snapshots, "role_snapshots", "snapshot_count", errors)
    _check_declared_count(task1, "instances", "instance_count", errors)
    _check_declared_count(task3, "instances", "instance_count", errors)
    state_update_audit = None
    if task1_state_updates is not None:
        from .materialization import validate_task1_state_update_assets

        state_update_audit = validate_task1_state_update_assets(task1_state_updates)
        if state_update_audit["status"] != "passed":
            errors.extend(
                f"Task1 state-update asset: {error}"
                for error in state_update_audit["errors"]
            )

    report = {
        "schema_version": "stage_temporal_validation_v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "characters_raw": len(characters),
            "characters_selected": sum(
                bool(item.get("construction_selected")) for item in characters.values()
            ),
            "characters_task1_eligible": sum(
                bool(item.get("task1_eligible")) for item in characters.values()
            ),
            "characters_task3_eligible": sum(
                bool(item.get("task3_single_turn_eligible"))
                for item in characters.values()
            ),
            "evidence_units": len(evidence),
            "states": len(states),
            "developments": len(developments),
            "epistemic_records": len(accesses),
            "persona_evidence": len(persona),
            "checkpoints": len(checkpoint_index),
            "task1_instances": len(task1_instances),
            "role_snapshots": len(snapshots),
            "task3_instances": len(task3_instances),
        },
        "task1_state_update": state_update_audit,
    }
    return report


def require_temporal_valid(report: dict[str, Any]) -> None:
    if report.get("status") != "passed":
        raise ValueError(
            "Character temporal validation failed: " + "; ".join(report.get("errors", []))
        )


def _unique_index(
    items: Any,
    id_key: str,
    label: str,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        errors.append(f"{label} collection must be an array")
        return {}
    output: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            errors.append(f"{label} must be an object")
            continue
        item_id = str(item.get(id_key) or "")
        if not item_id or item_id in output:
            errors.append(f"{label} has missing or duplicate ID: {item_id}")
            continue
        output[item_id] = item
    return output


def _known(values: Any, known: dict[str, Any]) -> bool:
    return isinstance(values, list) and bool(values) and set(values) <= set(known)


def _check_declared_count(
    payload: dict[str, Any], collection_key: str, count_key: str, errors: list[str]
) -> None:
    if count_key not in payload:
        return
    collection = payload.get(collection_key)
    if not isinstance(collection, list) or int(payload[count_key]) != len(collection):
        errors.append(f"Declared {count_key} disagrees with {collection_key}")


def _state_active(state: dict[str, Any], scene_order: int) -> bool:
    return int(state["valid_from_scene"]) <= scene_order and (
        state["valid_until_scene"] is None
        or scene_order <= int(state["valid_until_scene"])
    )


def _state_track_key(state: dict[str, Any]) -> tuple[str, str]:
    return (
        str(state.get("dimension", "")),
        normalize_name(state.get("target_id_or_text", "")),
    )
