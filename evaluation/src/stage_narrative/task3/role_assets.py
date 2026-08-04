from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from ..io import load_json, sha256_file, sha256_json
from ..models import clean_text, stable_id
from ..temporal.models import GraphIndex, build_graph_index


@dataclass(frozen=True, slots=True)
class RoleAssetInputs:
    temporal_run_dir: Path
    graph_path: Path
    movie_id: str
    language: str
    registry: dict[str, Any]
    epistemic: dict[str, Any]
    evidence: dict[str, Any]
    persona: dict[str, Any]
    states: dict[str, Any]
    developments: dict[str, Any]
    task3_single: dict[str, Any]
    graph: dict[str, Any]
    legacy_roles: dict[str, Any]
    input_manifest: dict[str, Any]


def load_role_asset_inputs(
    *,
    temporal_run_dir: Path,
    legacy_role_assets_path: Path,
    language: str,
) -> RoleAssetInputs:
    temporal_run_dir = temporal_run_dir.resolve()
    legacy_role_assets_path = legacy_role_assets_path.resolve()
    paths = {
        "temporal_release_manifest": temporal_run_dir / "release_manifest.json",
        "character_registry": temporal_run_dir / "assets/characters/character_registry.json",
        "epistemic_ledger": temporal_run_dir / "assets/characters/epistemic_ledger.json",
        "evidence_units": temporal_run_dir / "assets/source/evidence_units.json",
        "persona_evidence_bank": temporal_run_dir
        / "assets/characters/persona_evidence_bank.json",
        "state_ledger": temporal_run_dir / "assets/characters/state_ledger.json",
        "development_graph": temporal_run_dir
        / "assets/characters/development_graph.json",
        "task3_single_turn": temporal_run_dir / "assets/tasks/task3_checkpoint_single_turn.json",
        "narrative_graph_ref": temporal_run_dir / "assets/narrative/narrative_graph_ref.json",
        "legacy_role_assets": legacy_role_assets_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Task 3 reconstruction inputs: {missing}")
    release_manifest = load_json(paths["temporal_release_manifest"])
    _verify_temporal_release_artifacts(temporal_run_dir, release_manifest)
    graph_ref = load_json(paths["narrative_graph_ref"])
    graph_path = Path(str(graph_ref.get("graph_path") or "")).resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(f"Narrative graph does not exist: {graph_path}")
    expected_graph_hash = clean_text(graph_ref.get("graph_sha256"))
    actual_graph_hash = sha256_file(graph_path)
    if not expected_graph_hash or expected_graph_hash != actual_graph_hash:
        raise ValueError(
            "Narrative graph hash mismatch: "
            f"expected={expected_graph_hash}, actual={actual_graph_hash}"
        )
    paths["narrative_graph"] = graph_path
    payloads = {name: load_json(path) for name, path in paths.items()}
    movie_ids = {
        clean_text(payload.get("movie_id"))
        for payload in payloads.values()
        if isinstance(payload, dict) and payload.get("movie_id")
    }
    if len(movie_ids) != 1:
        raise ValueError(f"Task 3 inputs disagree on movie_id: {sorted(movie_ids)}")
    movie_id = next(iter(movie_ids))
    manifest = {
        "schema_version": "stage_task3_role_asset_input_snapshot",
        "movie_id": movie_id,
        "language": language,
        "temporal_run_dir": str(temporal_run_dir),
        "inputs": [
            {"name": name, "path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        ],
    }
    return RoleAssetInputs(
        temporal_run_dir=temporal_run_dir,
        graph_path=graph_path,
        movie_id=movie_id,
        language=language,
        registry=payloads["character_registry"],
        epistemic=payloads["epistemic_ledger"],
        evidence=payloads["evidence_units"],
        persona=payloads["persona_evidence_bank"],
        states=payloads["state_ledger"],
        developments=payloads["development_graph"],
        task3_single=payloads["task3_single_turn"],
        graph=payloads["narrative_graph"],
        legacy_roles=payloads["legacy_role_assets"],
        input_manifest=manifest,
    )


def select_release_characters(inputs: RoleAssetInputs) -> list[dict[str, Any]]:
    registry_by_id = {
        item["character_id"]: item for item in inputs.registry.get("characters", [])
    }
    character_ids = list(
        dict.fromkeys(
            clean_text(item.get("character_id"))
            for item in inputs.task3_single.get("instances", [])
        )
    )
    character_ids = [item for item in character_ids if item]
    missing = [item for item in character_ids if item not in registry_by_id]
    if missing:
        raise ValueError(f"Current Task 3 instances reference unknown characters: {missing}")
    return [registry_by_id[item] for item in character_ids]


def build_memory_jobs(inputs: RoleAssetInputs) -> list[dict[str, Any]]:
    index = build_graph_index(inputs.graph)
    selected = select_release_characters(inputs)
    selected_ids = {item["character_id"] for item in selected}
    characters = {item["character_id"]: item for item in selected}
    evidence_by_id = {
        item["evidence_id"]: item for item in inputs.evidence.get("evidence_units", [])
    }
    used_legacy_roles: set[int] = set()
    legacy_by_character = {
        character["character_id"]: _matching_legacy_role(
            character,
            inputs.legacy_roles.get("roles", []),
            used_legacy_roles,
        )
        for character in selected
    }
    facts_by_episode = _assign_facts_to_episodes(index)
    episode_by_fact = {
        fact["id"]: episode_id
        for episode_id, facts in facts_by_episode.items()
        for fact in facts
    }
    episode_by_id = {item["id"]: item for item in index.episodes}
    fact_by_id = {item["id"]: item for item in index.facts}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unassigned = []
    for access in inputs.epistemic.get("access_records", []):
        character_id = clean_text(access.get("character_id"))
        if character_id not in selected_ids or access.get("access_type") == "unknown":
            continue
        fact_id = clean_text(access.get("fact_or_event_id"))
        episode_id = episode_by_fact.get(fact_id)
        if fact_id not in fact_by_id or not episode_id:
            unassigned.append({"character_id": character_id, "fact_id": fact_id})
            continue
        support_ids = access.get("supporting_evidence_ids")
        if not isinstance(support_ids, list) or not support_ids:
            raise ValueError(
                f"Known fact lacks supporting evidence: character={character_id}, fact={fact_id}"
            )
        missing_evidence = [item for item in support_ids if item not in evidence_by_id]
        if missing_evidence:
            raise ValueError(
                f"Known fact references missing evidence: character={character_id}, "
                f"fact={fact_id}, evidence={missing_evidence}"
            )
        grouped[(character_id, episode_id)].append(access)
    if unassigned:
        raise ValueError(
            f"Known facts must resolve to graph episodes; unresolved={unassigned[:10]} "
            f"count={len(unassigned)}"
        )

    jobs = []
    for (character_id, episode_id), accesses in sorted(
        grouped.items(),
        key=lambda item: (
            characters[item[0][0]]["canonical_name"],
            int(episode_by_id[item[0][1]].get("order", 0)),
            item[0][1],
        ),
    ):
        accesses.sort(
            key=lambda item: (
                int(item.get("acquired_at_scene") or 0),
                clean_text(item.get("fact_or_event_id")),
            )
        )
        local_fact_ids = {
            access["fact_or_event_id"]: f"F{index + 1}"
            for index, access in enumerate(accesses)
        }
        evidence_ids = sorted(
            {
                evidence_id
                for access in accesses
                for evidence_id in _safe_supporting_evidence_ids(
                    access, evidence_by_id
                )
                if evidence_id in evidence_by_id
            },
            key=lambda evidence_id: (
                int(evidence_by_id[evidence_id].get("scene_order", 0)),
                int(evidence_by_id[evidence_id].get("char_start", 0)),
                evidence_id,
            ),
        )
        evidence_to_local = {
            evidence_id: f"E{index + 1}" for index, evidence_id in enumerate(evidence_ids)
        }
        facts = []
        for access in accesses:
            fact = fact_by_id[access["fact_or_event_id"]]
            fact_origin = _fact_origin(index, fact)
            available_from = _access_boundary(access, evidence_by_id, index)
            facts.append(
                {
                    "fact_id": local_fact_ids[fact["id"]],
                    "fact": index.fact_text(fact),
                    "access_type": access["access_type"],
                    "acquired_at_scene": int(access["acquired_at_scene"]),
                    "fact_origin": fact_origin,
                    "available_from": available_from,
                    "supporting_evidence_ids": [
                        evidence_to_local[item]
                        for item in _safe_supporting_evidence_ids(
                            access, evidence_by_id
                        )
                        if item in evidence_to_local
                    ],
                }
            )
        evidence = [
            {
                "evidence_id": evidence_to_local[evidence_id],
                "scene_order": int(evidence_by_id[evidence_id]["scene_order"]),
                "char_start": int(evidence_by_id[evidence_id].get("char_start", 0)),
                "char_end": int(evidence_by_id[evidence_id].get("char_end", 0)),
                "evidence_type": evidence_by_id[evidence_id]["evidence_type"],
                "speaker_character_id": clean_text(
                    evidence_by_id[evidence_id].get("speaker_character_id")
                ),
                "participant_character_ids": evidence_by_id[evidence_id].get(
                    "participant_character_ids", []
                ),
                "addressee_character_ids": evidence_by_id[evidence_id].get(
                    "addressee_character_ids", []
                ),
                "direct_observer_character_ids": evidence_by_id[evidence_id].get(
                    "direct_observer_character_ids", []
                ),
                "text": evidence_by_id[evidence_id]["evidence_text"],
            }
            for evidence_id in evidence_ids
        ]
        episode_scene_orders = set(index.node_scene_orders(episode_by_id[episode_id]))
        legacy_role = legacy_by_character[character_id]
        legacy_expression_hints = [
            {
                "hint_id": f"L{hint_index}",
                "memory_text": clean_text(item.get("memory_text")),
            }
            for hint_index, item in enumerate(
                [
                    item
                    for item in (legacy_role or {}).get("memories", [])
                    if int(item.get("scene_order") or 0) in episode_scene_orders
                    and clean_text(item.get("memory_text"))
                ],
                start=1,
            )
        ]
        structured_non_evidence_hints = _structured_hints_for_episode(
            inputs,
            character_id=character_id,
            episode_id=episode_id,
            episode_scene_orders=episode_scene_orders,
        )
        semantic_input = {
            "character": {
                "character_id": character_id,
                "canonical_name": characters[character_id]["canonical_name"],
                "aliases": characters[character_id].get("aliases", []),
                "identity_phases": _build_identity_phases(inputs, characters[character_id]),
            },
            "episode": {
                "episode_id": episode_id,
                "order": int(episode_by_id[episode_id].get("order", 0)),
                "name": clean_text(episode_by_id[episode_id].get("name")),
                "description": clean_text(episode_by_id[episode_id].get("description")),
            },
            "facts": facts,
            "evidence": evidence,
            "structured_non_evidence_hints": structured_non_evidence_hints,
            "legacy_expression_hints": legacy_expression_hints,
            "language": inputs.language,
        }
        jobs.append(
            {
                "job_id": stable_id("task3-memory-job", inputs.movie_id, character_id, episode_id),
                "character_id": character_id,
                "episode_id": episode_id,
                "semantic_input": semantic_input,
                "semantic_input_sha256": sha256_json(semantic_input),
                "fact_local_to_global": {value: key for key, value in local_fact_ids.items()},
                "evidence_local_to_global": {value: key for key, value in evidence_to_local.items()},
            }
        )
    return jobs


def validate_memory_response(job: dict[str, Any], response: dict[str, Any]) -> None:
    if set(response) != {"memories"} or not isinstance(response["memories"], list):
        raise ValueError("Memory response must contain only a memories array")
    expected = set(job["fact_local_to_global"])
    expected_evidence = set(job["evidence_local_to_global"])
    facts_by_id = {
        item["fact_id"]: item for item in job["semantic_input"].get("facts", [])
    }
    seen: set[str] = set()
    for memory in response["memories"]:
        required = {"fact_ids", "evidence_ids", "memory_text", "importance", "tags"}
        if not isinstance(memory, dict) or set(memory) != required:
            raise ValueError(f"Memory keys must be exactly {sorted(required)}")
        fact_ids = memory["fact_ids"]
        if not isinstance(fact_ids, list) or not fact_ids:
            raise ValueError("Each memory must contain at least one fact_id")
        if any(item not in expected for item in fact_ids):
            raise ValueError(f"Memory uses an unknown local fact ID: {fact_ids}")
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError(f"A memory may not repeat a fact ID: {fact_ids}")
        if seen & set(fact_ids):
            raise ValueError(f"A fact may appear in only one memory: {fact_ids}")
        seen.update(fact_ids)
        access_types = {facts_by_id[item]["access_type"] for item in fact_ids}
        if len(access_types) != 1:
            raise ValueError(
                "One canonical memory cannot merge different access types: "
                f"facts={fact_ids}, access_types={sorted(access_types)}"
            )
        origins = {
            (
                facts_by_id[item]["fact_origin"]["scene_id"],
                int(facts_by_id[item]["fact_origin"]["scene_order"]),
            )
            for item in fact_ids
        }
        if len(origins) != 1:
            raise ValueError(
                f"One canonical memory cannot merge different fact origins: {fact_ids}"
            )
        evidence_ids = memory["evidence_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("Each memory must contain at least one evidence_id")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"A memory may not repeat an evidence ID: {evidence_ids}")
        if any(item not in expected_evidence for item in evidence_ids):
            raise ValueError(f"Memory uses an unknown local evidence ID: {evidence_ids}")
        required_evidence = {
            evidence_id
            for fact_id in fact_ids
            for evidence_id in facts_by_id[fact_id]["supporting_evidence_ids"]
        }
        if set(evidence_ids) != required_evidence:
            raise ValueError(
                "Memory evidence_ids must exactly cover the selected facts' evidence: "
                f"expected={sorted(required_evidence)}, actual={sorted(evidence_ids)}"
            )
        if not clean_text(memory["memory_text"]):
            raise ValueError("memory_text must be non-empty")
        if not _has_first_person_marker(
            memory["memory_text"], job["semantic_input"]["language"]
        ):
            raise ValueError("memory_text must use an explicit first-person marker")
        if memory["importance"] not in {"core", "supporting"}:
            raise ValueError("importance must be core or supporting")
        if not isinstance(memory["tags"], list) or not memory["tags"] or not all(
            clean_text(item) for item in memory["tags"]
        ):
            raise ValueError("tags must be a non-empty-string array")
    if seen != expected:
        raise ValueError(f"Memory response must cover every fact exactly once: missing={sorted(expected-seen)}")


def finalize_role_assets(
    inputs: RoleAssetInputs,
    jobs: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    *,
    construction_metadata: dict[str, Any],
) -> dict[str, Any]:
    selected = select_release_characters(inputs)
    selected_by_id = {item["character_id"]: item for item in selected}
    identity_phases_by_character = {
        item["character_id"]: _build_identity_phases(inputs, item) for item in selected
    }
    evidence_by_id = {
        item["evidence_id"]: item for item in inputs.evidence.get("evidence_units", [])
    }
    access_by_character_fact = {
        (item["character_id"], item["fact_or_event_id"]): item
        for item in inputs.epistemic.get("access_records", [])
        if item.get("access_type") != "unknown"
    }
    memories_by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        response = responses[job["job_id"]]
        validate_memory_response(job, response)
        local_fact_by_id = {
            item["fact_id"]: item for item in job["semantic_input"]["facts"]
        }
        for memory in response["memories"]:
            fact_ids = [job["fact_local_to_global"][item] for item in memory["fact_ids"]]
            fact_records = [local_fact_by_id[item] for item in memory["fact_ids"]]
            accesses = [access_by_character_fact[(job["character_id"], item)] for item in fact_ids]
            source_evidence_ids = sorted(
                {
                    job["evidence_local_to_global"][item]
                    for item in memory["evidence_ids"]
                },
                key=lambda item: (
                    int(evidence_by_id[item].get("scene_order", 0)),
                    int(evidence_by_id[item].get("char_start", 0)),
                    item,
                ),
            )
            available_scene, available_char_end = max(
                (
                    int(item["available_from"]["scene_order"]),
                    int(item["available_from"]["char_end"]),
                )
                for item in fact_records
            )
            access_type = fact_records[0]["access_type"]
            fact_origin = deepcopy(fact_records[0]["fact_origin"])
            identity_phase = _phase_id_at_boundary(
                identity_phases_by_character[job["character_id"]],
                {"scene_order": available_scene, "char_end": available_char_end},
            )
            memory_id = stable_id(
                "task3-memory",
                inputs.movie_id,
                job["character_id"],
                access_type,
                fact_origin["scene_id"],
                *sorted(fact_ids),
                *source_evidence_ids,
                available_scene,
                available_char_end,
            )
            memories_by_character[job["character_id"]].append(
                {
                    "memory_id": memory_id,
                    "memory_text": clean_text(memory["memory_text"]),
                    "importance": memory["importance"],
                    "tags": list(dict.fromkeys(clean_text(item) for item in memory["tags"])),
                    "fact_origin": fact_origin,
                    "available_from": {
                        "scene_order": available_scene,
                        "char_end": available_char_end,
                    },
                    "access_type": access_type,
                    "source_episode_ids": [job["episode_id"]],
                    "source_fact_ids": fact_ids,
                    "source_evidence_ids": source_evidence_ids,
                    "grounded_facts": [item["fact"] for item in fact_records],
                    "identity_phase": identity_phase,
                    "knowledge_access": [
                        {
                            "access_id": access.get("access_id"),
                            "fact_id": access["fact_or_event_id"],
                            "access_type": access["access_type"],
                            "acquired_at_scene": int(access["acquired_at_scene"]),
                        }
                        for access in accesses
                    ],
                }
            )

    roles = []
    for character in selected:
        character_id = character["character_id"]
        memories = sorted(
            memories_by_character[character_id],
            key=lambda item: (
                int(item["available_from"]["scene_order"]),
                int(item["available_from"]["char_end"]),
                item["memory_id"],
            ),
        )
        identity_phases = identity_phases_by_character[character_id]
        roles.append(
            {
                "role_id": stable_id("task3-role", inputs.movie_id, character_id),
                "character_id": character_id,
                "canonical_name": character["canonical_name"],
                "aliases": character.get("aliases", []),
                "identity_phases": identity_phases,
                "persona_phases": _build_persona_phases(
                    inputs,
                    character=selected_by_id[character_id],
                    identity_phases=identity_phases,
                ),
                "relationship_phases": _build_relationship_phases(
                    inputs,
                    character=selected_by_id[character_id],
                    identity_phases=identity_phases,
                ),
                "memories": memories,
            }
        )
    return {
        "schema_version": "stage_task3_role_assets",
        "task": "task_3_character_role_play",
        "movie_id": inputs.movie_id,
        "language": inputs.language,
        "role_count": len(roles),
        "roles": roles,
        "construction": construction_metadata,
    }


def _fact_origin(index: GraphIndex, fact: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for scene_id in index.node_scene_ids(fact):
        if scene_id in index.scene_order_by_id:
            pairs.append((index.scene_order_by_id[scene_id], scene_id))
    if not pairs and fact.get("source_scene_order") is not None:
        order = int(fact["source_scene_order"])
        pairs = [
            (order, scene_id)
            for scene_id, scene_order in index.scene_order_by_id.items()
            if scene_order == order
        ]
    if not pairs:
        raise ValueError(f"Fact lacks a resolvable origin scene: {fact.get('id')}")
    scene_order, scene_id = min(pairs)
    return {"scene_id": scene_id, "scene_order": scene_order}


def _safe_supporting_evidence_ids(
    access: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    acquired = int(access["acquired_at_scene"])
    safe = [
        item
        for item in access.get("supporting_evidence_ids", [])
        if item in evidence_by_id
        and int(evidence_by_id[item].get("scene_order", 0)) <= acquired
    ]
    if not safe:
        raise ValueError(
            "Known fact has no supporting evidence visible by its acquisition scene: "
            f"character={access.get('character_id')}, fact={access.get('fact_or_event_id')}, "
            f"acquired_at_scene={acquired}"
        )
    return safe


def _access_boundary(
    access: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    index: GraphIndex,
) -> dict[str, int]:
    access_type = clean_text(access.get("access_type"))
    if access_type not in {"witnessed", "involved", "told", "inferred"}:
        raise ValueError(f"Cannot build a memory boundary for access_type={access_type!r}")
    acquired = int(access["acquired_at_scene"])
    evidence_ids = _safe_supporting_evidence_ids(access, evidence_by_id)
    same_scene_char_ends = [
        int(evidence_by_id[item].get("char_end", 0))
        for item in evidence_ids
        if int(evidence_by_id[item].get("scene_order", 0)) == acquired
        and int(evidence_by_id[item].get("char_end", 0)) > 0
    ]
    if same_scene_char_ends:
        return {"scene_order": acquired, "char_end": max(same_scene_char_ends)}
    later_orders = sorted(
        order for order in set(index.scene_order_by_id.values()) if order > acquired
    )
    return {
        "scene_order": later_orders[0] if later_orders else acquired + 1,
        "char_end": 0,
    }


def _build_identity_phases(
    inputs: RoleAssetInputs, character: dict[str, Any]
) -> list[dict[str, Any]]:
    source_phases = character.get("identity_phases", [])
    if not source_phases:
        source_phases = [
            {
                "phase_id": "registry-default",
                "name": character["canonical_name"],
                "aliases": character.get("aliases", []),
                "valid_from_scene": int(character.get("first_scene_order") or 1),
                "valid_until_scene": None,
            }
        ]
    # A character can be referenced by a grounded state or persona observation
    # before the registry's dialogue-derived first scene.  Align the first
    # identity phase to the earliest source-backed observation so relationship
    # phases do not point outside the character's visible identity interval.
    observed_scene_candidates = [
        int(item["acquired_at_scene"])
        for item in inputs.epistemic.get("access_records", [])
        if item.get("character_id") == character["character_id"]
        and item.get("access_type") in {"witnessed", "involved", "told", "inferred"}
        and item.get("acquired_at_scene") is not None
    ]
    observed_scene_candidates.extend(
        int(item["valid_from_scene"])
        for item in inputs.states.get("states", [])
        if item.get("character_id") == character["character_id"]
        and item.get("valid_from_scene") is not None
    )
    observed_scene_candidates.extend(
        int(item["established_from_scene"])
        for item in inputs.persona.get("persona_evidence", [])
        if item.get("character_id") == character["character_id"]
        and item.get("established_from_scene") is not None
    )
    earliest_access_scene = min(observed_scene_candidates, default=None)
    phases = []
    for source in sorted(
        source_phases,
        key=lambda item: (int(item.get("valid_from_scene") or 0), clean_text(item.get("phase_id"))),
    ):
        valid_from_scene = int(source.get("valid_from_scene") or 0)
        if valid_from_scene <= 0:
            raise ValueError(
                f"Identity phase lacks valid_from_scene: {character['character_id']}"
            )
        if not phases and earliest_access_scene is not None:
            # The ledger is evidence-backed for this character; align only the
            # first phase so a same/preceding scene boundary remains visible.
            valid_from_scene = min(valid_from_scene, earliest_access_scene)
        valid_until_scene = source.get("valid_until_scene")
        phases.append(
            {
                "identity_phase_id": stable_id(
                    "task3-identity-phase",
                    inputs.movie_id,
                    character["character_id"],
                    source.get("phase_id"),
                    valid_from_scene,
                ),
                "name": clean_text(source.get("name") or character["canonical_name"]),
                "aliases": list(
                    dict.fromkeys(
                        clean_text(item)
                        for item in source.get("aliases", [])
                        if clean_text(item)
                    )
                ),
                "valid_from": {"scene_order": valid_from_scene, "char_end": 0},
                "valid_until": (
                    {
                        "scene_order": int(valid_until_scene) + 1,
                        "char_end": 0,
                    }
                    if valid_until_scene is not None
                    else None
                ),
                "source": "character_registry",
            }
        )
    for previous, current in zip(phases, phases[1:]):
        if _boundary_tuple(previous["valid_from"]) >= _boundary_tuple(current["valid_from"]):
            raise ValueError(f"Identity phases overlap or are unsorted: {character['character_id']}")
        previous["valid_until"] = deepcopy(current["valid_from"])
    return phases


def _phase_id_at_boundary(
    phases: list[dict[str, Any]], boundary: dict[str, Any]
) -> str:
    active = [
        item
        for item in phases
        if _phase_active(item, boundary)
    ]
    if not active:
        # Registry identity phases start at scene boundaries.  A memory may be
        # acquired later in that same first scene, so tolerate the equivalent
        # intra-scene boundary without extending visibility to an earlier scene.
        first_scene = min(
            int(item["valid_from"]["scene_order"]) for item in phases
        )
        if first_scene - 1 <= int(boundary["scene_order"]) <= first_scene:
            first_phase = min(phases, key=lambda item: _boundary_tuple(item["valid_from"]))
            return first_phase["identity_phase_id"]
        raise ValueError(f"No identity phase is active at boundary {boundary}")
    return max(active, key=lambda item: _boundary_tuple(item["valid_from"]))[
        "identity_phase_id"
    ]


def _build_persona_phases(
    inputs: RoleAssetInputs,
    *,
    character: dict[str, Any],
    identity_phases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_id = {
        item["evidence_id"]: item for item in inputs.evidence.get("evidence_units", [])
    }
    by_scene: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in inputs.persona.get("persona_evidence", []):
        if item.get("character_id") != character["character_id"]:
            continue
        if not clean_text(item.get("value")):
            continue
        support_ids = [
            evidence_id
            for evidence_id in item.get("supporting_evidence_ids", [])
            if evidence_id in evidence_by_id
        ]
        if not support_ids:
            raise ValueError(f"Persona evidence lacks source evidence: {item.get('persona_evidence_id')}")
        by_scene[int(item["established_from_scene"])].append(item)
    phases = []
    for scene_order, items in sorted(by_scene.items()):
        source_evidence_ids = _sorted_evidence_ids(
            {
                evidence_id
                for item in items
                for evidence_id in item.get("supporting_evidence_ids", [])
                if evidence_id in evidence_by_id
            },
            evidence_by_id,
        )
        valid_from = _evidence_boundary_for_scene(
            scene_order, source_evidence_ids, evidence_by_id
        )
        values_by_kind: dict[str, list[str]] = defaultdict(list)
        for item in items:
            value = clean_text(item["value"])
            if value not in values_by_kind[item["evidence_kind"]]:
                values_by_kind[item["evidence_kind"]].append(value)
        phases.append(
            {
                "persona_phase_id": stable_id(
                    "task3-persona-phase",
                    inputs.movie_id,
                    character["character_id"],
                    scene_order,
                    *source_evidence_ids,
                ),
                "valid_from": valid_from,
                "valid_until": None,
                "traits": values_by_kind.get("trait", []),
                "speaking_style": values_by_kind.get("speaking_style", []),
                "behavioral_constraints": values_by_kind.get(
                    "behavioral_constraint", []
                ),
                "source_persona_evidence_ids": sorted(
                    item["persona_evidence_id"] for item in items
                ),
                "source_evidence_ids": source_evidence_ids,
                "identity_phase": _phase_id_at_boundary(identity_phases, valid_from),
            }
        )
    for current, following in zip(phases, phases[1:]):
        current["valid_until"] = deepcopy(following["valid_from"])
    return phases


def _build_relationship_phases(
    inputs: RoleAssetInputs,
    *,
    character: dict[str, Any],
    identity_phases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_id = {
        item["evidence_id"]: item for item in inputs.evidence.get("evidence_units", [])
    }
    developments = inputs.developments.get("developments", [])
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in inputs.states.get("states", []):
        if (
            state.get("character_id") == character["character_id"]
            and state.get("dimension") == "relationship"
            and clean_text(state.get("target_id_or_text"))
            and clean_text(state.get("state_value"))
        ):
            by_target[clean_text(state["target_id_or_text"])].append(state)
    output = []
    for target, states in sorted(by_target.items()):
        point_scenes = {
            int(item["valid_from_scene"]) for item in states
        } | {
            int(item["valid_until_scene"]) + 1
            for item in states
            if item.get("valid_until_scene") is not None
        }
        points: list[tuple[dict[str, int], list[dict[str, Any]]]] = []
        for scene_order in sorted(point_scenes):
            active = [
                item
                for item in states
                if int(item["valid_from_scene"]) <= scene_order
                and (
                    item.get("valid_until_scene") is None
                    or scene_order <= int(item["valid_until_scene"])
                )
            ]
            if not active:
                continue
            newly_active = [
                item for item in active if int(item["valid_from_scene"]) == scene_order
            ]
            new_evidence_ids = _sorted_evidence_ids(
                {
                    evidence_id
                    for item in newly_active
                    for evidence_id in item.get("supporting_evidence_ids", [])
                    if evidence_id in evidence_by_id
                },
                evidence_by_id,
            )
            boundary = (
                _evidence_boundary_for_scene(
                    scene_order, new_evidence_ids, evidence_by_id
                )
                if new_evidence_ids
                else {"scene_order": scene_order, "char_end": 0}
            )
            points.append((boundary, active))
        points = _coalesce_phase_points(points)
        for point_index, (valid_from, active) in enumerate(points):
            source_state_ids = sorted(item["state_id"] for item in active)
            source_evidence_ids = _sorted_evidence_ids(
                {
                    evidence_id
                    for item in active
                    for evidence_id in item.get("supporting_evidence_ids", [])
                    if evidence_id in evidence_by_id
                },
                evidence_by_id,
            )
            active_state_ids = set(source_state_ids)
            source_development_ids = sorted(
                item["development_id"]
                for item in developments
                if item.get("character_id") == character["character_id"]
                and item.get("dimension") == "relationship"
                and clean_text(item.get("target_id_or_text")) == target
                and active_state_ids
                & set(
                    item.get("before_state_ids", [])
                    + item.get("resulting_state_ids", [])
                    + item.get("invariant_state_ids", [])
                )
            )
            values = list(
                dict.fromkeys(clean_text(item["state_value"]) for item in active)
            )
            polarities = {clean_text(item.get("polarity")) for item in active}
            polarities.discard("")
            output.append(
                {
                    "relationship_phase_id": stable_id(
                        "task3-relationship-phase",
                        inputs.movie_id,
                        character["character_id"],
                        target,
                        valid_from["scene_order"],
                        valid_from["char_end"],
                        *source_state_ids,
                    ),
                    "target": target,
                    "valid_from": valid_from,
                    "valid_until": (
                        deepcopy(points[point_index + 1][0])
                        if point_index + 1 < len(points)
                        else None
                    ),
                    "value": "; ".join(values),
                    "polarity": (
                        next(iter(polarities)) if len(polarities) == 1 else "mixed"
                    ),
                    "source_state_ids": source_state_ids,
                    "source_development_ids": source_development_ids,
                    "source_evidence_ids": source_evidence_ids,
                    "identity_phase": _phase_id_at_boundary(
                        identity_phases, valid_from
                    ),
                }
            )
    return sorted(
        output,
        key=lambda item: (
            _boundary_tuple(item["valid_from"]),
            item["target"],
            item["relationship_phase_id"],
        ),
    )


def _coalesce_phase_points(
    points: list[tuple[dict[str, int], list[dict[str, Any]]]],
) -> list[tuple[dict[str, int], list[dict[str, Any]]]]:
    """Collapse source changes that resolve to the same visible boundary."""
    output: list[tuple[dict[str, int], list[dict[str, Any]]]] = []
    for boundary, active in points:
        if output and _boundary_tuple(boundary) < _boundary_tuple(output[-1][0]):
            raise ValueError("Relationship phase boundaries are not chronological")
        if output and _boundary_tuple(boundary) == _boundary_tuple(output[-1][0]):
            output[-1] = (boundary, active)
        else:
            output.append((boundary, active))
    return output


def _structured_hints_for_episode(
    inputs: RoleAssetInputs,
    *,
    character_id: str,
    episode_id: str,
    episode_scene_orders: set[int],
) -> dict[str, Any]:
    states = [
        {
            "hint_id": f"S{index}",
            "dimension": item.get("dimension"),
            "value": clean_text(item.get("state_value")),
            "valid_from_scene": item.get("valid_from_scene"),
            "valid_until_scene": item.get("valid_until_scene"),
        }
        for index, item in enumerate(
            [
                item
                for item in inputs.states.get("states", [])
                if item.get("character_id") == character_id
                and (
                    episode_id in item.get("source_episode_ids", [])
                    or int(item.get("valid_from_scene") or 0) in episode_scene_orders
                )
            ],
            start=1,
        )
    ]
    developments = [
        {
            "hint_id": f"D{index}",
            "dimension": item.get("dimension"),
            "operation": item.get("operation"),
            "target": clean_text(item.get("target_id_or_text")),
            "effective_from_scene": item.get("effective_from_scene"),
        }
        for index, item in enumerate(
            [
                item
                for item in inputs.developments.get("developments", [])
                if item.get("character_id") == character_id
                and int(item.get("effective_from_scene") or 0)
                in episode_scene_orders
            ],
            start=1,
        )
    ]
    return {
        "status": "non_evidence_hints_only",
        "state_candidates": states,
        "development_candidates": developments,
    }


def _sorted_evidence_ids(
    evidence_ids: set[str], evidence_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    return sorted(
        evidence_ids,
        key=lambda item: (
            int(evidence_by_id[item].get("scene_order", 0)),
            int(evidence_by_id[item].get("char_start", 0)),
            item,
        ),
    )


def _evidence_boundary_for_scene(
    scene_order: int,
    evidence_ids: list[str],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    char_ends = [
        int(evidence_by_id[item].get("char_end", 0))
        for item in evidence_ids
        if int(evidence_by_id[item].get("scene_order", 0)) == scene_order
        and int(evidence_by_id[item].get("char_end", 0)) > 0
    ]
    if char_ends:
        return {"scene_order": scene_order, "char_end": max(char_ends)}
    return {"scene_order": scene_order + 1, "char_end": 0}


def _boundary_tuple(boundary: dict[str, Any]) -> tuple[int, int]:
    return int(boundary["scene_order"]), int(boundary.get("char_end", 0))


def _phase_active(phase: dict[str, Any], boundary: dict[str, Any]) -> bool:
    current = _boundary_tuple(boundary)
    if _boundary_tuple(phase["valid_from"]) > current:
        return False
    valid_until = phase.get("valid_until")
    return valid_until is None or current < _boundary_tuple(valid_until)


def _verify_temporal_release_artifacts(
    temporal_run_dir: Path, release_manifest: dict[str, Any]
) -> None:
    if release_manifest.get("schema_version") != "stage_temporal_release_manifest_v1":
        raise ValueError("Unsupported temporal release manifest schema")
    artifacts = release_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Temporal release manifest requires a non-empty artifacts array")
    root = temporal_run_dir.resolve()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Temporal release artifact entries must be objects")
        relative = Path(clean_text(artifact.get("path")))
        if not relative.as_posix() or relative.is_absolute():
            raise ValueError(f"Temporal release artifact path must be relative: {relative}")
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Temporal release artifact escapes run directory: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Temporal release artifact is missing: {path}")
        expected = clean_text(artifact.get("sha256"))
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(
                f"Temporal release artifact hash mismatch: {relative}; "
                f"expected={expected}, actual={actual}"
            )


def _assign_facts_to_episodes(index: GraphIndex) -> dict[str, list[dict[str, Any]]]:
    episode_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    child_to_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in index.episodes:
        for scene_id in index.node_scene_ids(episode):
            episode_by_scene[scene_id].append(episode)
        for child_id in episode.get("child_unit_ids", []):
            child_to_episode[child_id].append(episode)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in index.facts:
        candidates = child_to_episode.get(fact["id"], []) or [
            episode
            for scene_id in index.node_scene_ids(fact)
            for episode in episode_by_scene.get(scene_id, [])
        ]
        if not candidates:
            continue
        assigned = min(
            {item["id"]: item for item in candidates}.values(),
            key=lambda item: (int(item.get("order", 0)), item["id"]),
        )
        output[assigned["id"]].append(fact)
    return output


def _normalize_name(value: Any) -> str:
    return "".join(character for character in clean_text(value).casefold() if character.isalnum())


def _matching_legacy_role(
    character: dict[str, Any],
    legacy_roles: list[dict[str, Any]],
    used_indices: set[int],
) -> dict[str, Any] | None:
    current_names = {
        _normalize_name(character.get("canonical_name")),
        *(_normalize_name(item) for item in character.get("aliases", [])),
        *(
            _normalize_name(item.get("name"))
            for item in character.get("identity_phases", [])
            if isinstance(item, dict)
        ),
    }
    current_names.discard("")
    matches = []
    for index, role in enumerate(legacy_roles):
        if index in used_indices:
            continue
        legacy_names = {
            _normalize_name(role.get("character_name")),
            *(_normalize_name(item) for item in role.get("aliases", [])),
        }
        legacy_names.discard("")
        if current_names & legacy_names:
            matches.append((index, role))
    if len(matches) > 1:
        raise ValueError(
            f"Current character matches multiple legacy role cards: "
            f"{character['character_id']} -> {[item[1].get('character_name') for item in matches]}"
        )
    if not matches:
        return None
    used_indices.add(matches[0][0])
    return matches[0][1]


def _has_first_person_marker(text: Any, language: str) -> bool:
    value = clean_text(text)
    if language == "Chinese":
        return any(marker in value for marker in ("我", "咱", "俺"))
    return bool(re.search(r"\b(?:I|me|my|mine|we|us|our|ours)\b", value, re.IGNORECASE))
