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
    task3_single: dict[str, Any]
    graph: dict[str, Any]
    legacy_roles: dict[str, Any]
    input_manifest: dict[str, Any]


def load_role_asset_inputs(
    *, temporal_run_dir: Path, legacy_role_assets_path: Path, language: str
) -> RoleAssetInputs:
    temporal_run_dir = temporal_run_dir.resolve()
    legacy_role_assets_path = legacy_role_assets_path.resolve()
    paths = {
        "temporal_release_manifest": temporal_run_dir / "release_manifest.json",
        "character_registry": temporal_run_dir / "assets/characters/character_registry.json",
        "epistemic_ledger": temporal_run_dir / "assets/characters/epistemic_ledger.json",
        "evidence_units": temporal_run_dir / "assets/source/evidence_units.json",
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
            f"Narrative graph hash mismatch: expected={expected_graph_hash}, "
            f"actual={actual_graph_hash}"
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
    snapshot = {
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
        task3_single=payloads["task3_single_turn"],
        graph=payloads["narrative_graph"],
        legacy_roles=payloads["legacy_role_assets"],
        input_manifest=snapshot,
    )


def select_release_characters(inputs: RoleAssetInputs) -> list[dict[str, Any]]:
    registry_by_id = {
        item["character_id"]: item for item in inputs.registry.get("characters", [])
    }
    character_ids = list(
        dict.fromkeys(
            clean_text(item.get("character_id"))
            for item in inputs.task3_single.get("instances", [])
            if clean_text(item.get("character_id"))
        )
    )
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
    present_by_scene = _characters_present_by_scene(
        facts=index.facts,
        evidence_units=inputs.evidence.get("evidence_units", []),
    )
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
        accepted, _, _ = _access_is_perspective_grounded(
            access=access,
            fact=fact_by_id[fact_id],
            evidence_by_id=evidence_by_id,
            present_by_scene=present_by_scene,
        )
        if not accepted:
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
            access["fact_or_event_id"]: f"F{position + 1}"
            for position, access in enumerate(accesses)
        }
        evidence_ids = sorted(
            {
                evidence_id
                for access in accesses
                for evidence_id in access.get("supporting_evidence_ids", [])
            },
            key=lambda evidence_id: (
                int(evidence_by_id[evidence_id].get("scene_order", 0)),
                int(evidence_by_id[evidence_id].get("char_start", 0)),
                evidence_id,
            ),
        )
        evidence_to_local = {
            evidence_id: f"E{position + 1}"
            for position, evidence_id in enumerate(evidence_ids)
        }
        facts = []
        for access in accesses:
            fact = fact_by_id[access["fact_or_event_id"]]
            facts.append(
                {
                    "fact_id": local_fact_ids[fact["id"]],
                    "fact": index.fact_text(fact),
                    "access_type": access["access_type"],
                    "acquired_at_scene": int(access["acquired_at_scene"]),
                    "supporting_evidence_ids": [
                        evidence_to_local[item]
                        for item in access.get("supporting_evidence_ids", [])
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
                "direct_observer_character_ids": evidence_by_id[evidence_id].get(
                    "direct_observer_character_ids", []
                ),
                "text": evidence_by_id[evidence_id]["evidence_text"],
            }
            for evidence_id in evidence_ids
        ]
        semantic_input = {
            "character": {
                "character_id": character_id,
                "canonical_name": characters[character_id]["canonical_name"],
                "aliases": characters[character_id].get("aliases", []),
            },
            "episode": {
                "episode_id": episode_id,
                "order": int(episode_by_id[episode_id].get("order", 0)),
                "name": clean_text(episode_by_id[episode_id].get("name")),
                "description": clean_text(episode_by_id[episode_id].get("description")),
            },
            "facts": facts,
            "evidence": evidence,
            "language": inputs.language,
        }
        jobs.append(
            {
                "job_id": stable_id(
                    "task3-memory-job", inputs.movie_id, character_id, episode_id
                ),
                "character_id": character_id,
                "episode_id": episode_id,
                "semantic_input": semantic_input,
                "semantic_input_sha256": sha256_json(semantic_input),
                "fact_local_to_global": {
                    value: key for key, value in local_fact_ids.items()
                },
                "evidence_local_to_global": {
                    value: key for key, value in evidence_to_local.items()
                },
            }
        )
    return jobs


def audit_memory_access_perspective(inputs: RoleAssetInputs) -> dict[str, Any]:
    """Audit whether each selected known fact was actually available to the role."""
    index = build_graph_index(inputs.graph)
    selected_ids = {
        item["character_id"] for item in select_release_characters(inputs)
    }
    fact_by_id = {item["id"]: item for item in index.facts}
    evidence_by_id = {
        item["evidence_id"]: item for item in inputs.evidence.get("evidence_units", [])
    }
    present_by_scene = _characters_present_by_scene(
        facts=index.facts,
        evidence_units=inputs.evidence.get("evidence_units", []),
    )
    records = []
    for access in inputs.epistemic.get("access_records", []):
        character_id = clean_text(access.get("character_id"))
        if character_id not in selected_ids or access.get("access_type") == "unknown":
            continue
        fact_id = clean_text(access.get("fact_or_event_id"))
        fact = fact_by_id.get(fact_id)
        if fact is None:
            continue
        accepted, reason, requires_review = _access_is_perspective_grounded(
            access=access,
            fact=fact,
            evidence_by_id=evidence_by_id,
            present_by_scene=present_by_scene,
        )
        records.append(
            {
                "character_id": character_id,
                "fact_id": fact_id,
                "access_type": access.get("access_type"),
                "accepted": accepted,
                "requires_review": requires_review,
                "reason": reason,
                "fact_participant_entity_ids": sorted(
                    _fact_participant_ids(fact)
                ),
                "supporting_evidence_ids": list(
                    access.get("supporting_evidence_ids", [])
                ),
            }
        )
    rejected = [item for item in records if not item["accepted"]]
    review = [item for item in records if item["requires_review"]]
    return {
        "schema_version": "stage_task3_memory_perspective_access_audit",
        "movie_id": inputs.movie_id,
        "selected_known_access_count": len(records),
        "accepted_count": len(records) - len(rejected),
        "rejected_count": len(rejected),
        "requires_review_count": len(review),
        "rejected_by_access_type": {
            access_type: sum(
                item["access_type"] == access_type for item in rejected
            )
            for access_type in ("involved", "witnessed", "told", "inferred")
        },
        "records": records,
    }


def _access_is_perspective_grounded(
    *,
    access: dict[str, Any],
    fact: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    present_by_scene: dict[int, set[str]] | None = None,
) -> tuple[bool, str, bool]:
    character_id = clean_text(access.get("character_id"))
    access_type = clean_text(access.get("access_type"))
    participants = _fact_participant_ids(fact)
    evidence = [
        evidence_by_id[item]
        for item in access.get("supporting_evidence_ids", [])
        if item in evidence_by_id
    ]
    speakers = {
        clean_text(item.get("speaker_character_id")) for item in evidence
    }
    addressees = {
        clean_text(character_id)
        for item in evidence
        for character_id in item.get("addressee_character_ids", [])
    }
    observers = {
        clean_text(character_id)
        for item in evidence
        for character_id in item.get("direct_observer_character_ids", [])
    }
    exposed = participants | speakers | addressees | observers
    acquired_scene = int(access.get("acquired_at_scene") or 0)
    present_in_scene = character_id in (present_by_scene or {}).get(
        acquired_scene, set()
    )
    if access_type == "involved":
        if character_id in participants:
            return True, "character_is_explicit_kg_participant", False
        return False, "mentioned_but_not_explicit_kg_participant", False
    if access_type == "told":
        if character_id in addressees | observers | speakers or present_in_scene:
            return (
                True,
                "character_received_or_was_explicitly_present_in_scene",
                False,
            )
        return (
            True,
            "epistemic_access_retained_but_receipt_metadata_is_incomplete",
            True,
        )
    if access_type in {"witnessed", "inferred"}:
        if character_id in exposed or present_in_scene:
            return (
                True,
                "character_is_participant_or_explicitly_present_in_scene",
                False,
            )
        return (
            True,
            "epistemic_access_retained_but_exposure_metadata_is_incomplete",
            True,
        )
    return False, f"unsupported_access_type:{access_type}", False


def _fact_participant_ids(fact: dict[str, Any]) -> set[str]:
    values = {
        clean_text(item) for item in fact.get("participant_entity_ids", [])
    }
    for key in ("subject_entity_id", "object_entity_id"):
        value = clean_text(fact.get(key))
        if value:
            values.add(value)
    values.discard("")
    return values


def _characters_present_by_scene(
    *, facts: list[dict[str, Any]], evidence_units: list[dict[str, Any]]
) -> dict[int, set[str]]:
    present: dict[int, set[str]] = defaultdict(set)
    for fact in facts:
        scene_order = int(fact.get("source_scene_order") or 0)
        if scene_order:
            present[scene_order].update(_fact_participant_ids(fact))
    for evidence in evidence_units:
        scene_order = int(evidence.get("scene_order") or 0)
        if not scene_order:
            continue
        speaker = clean_text(evidence.get("speaker_character_id"))
        if speaker:
            present[scene_order].add(speaker)
        for key in ("addressee_character_ids", "direct_observer_character_ids"):
            present[scene_order].update(
                clean_text(item) for item in evidence.get(key, []) if clean_text(item)
            )
    return present


def validate_memory_response(job: dict[str, Any], response: dict[str, Any]) -> None:
    if set(response) != {"memories"} or not isinstance(response["memories"], list):
        raise ValueError("Memory response must contain only a memories array")
    expected = set(job["fact_local_to_global"])
    seen: set[str] = set()
    for memory in response["memories"]:
        required = {"fact_ids", "memory_text", "importance", "tags"}
        if not isinstance(memory, dict) or set(memory) != required:
            raise ValueError(f"Memory keys must be exactly {sorted(required)}")
        fact_ids = memory["fact_ids"]
        if not isinstance(fact_ids, list) or not fact_ids:
            raise ValueError("Each memory must contain at least one fact_id")
        if any(item not in expected for item in fact_ids):
            raise ValueError(f"Memory uses an unknown local fact ID: {fact_ids}")
        if seen & set(fact_ids):
            raise ValueError(f"A fact may appear in only one memory: {fact_ids}")
        seen.update(fact_ids)
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
        raise ValueError(
            f"Memory response must cover every fact exactly once: missing={sorted(expected-seen)}"
        )


def finalize_role_assets(
    inputs: RoleAssetInputs,
    jobs: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    *,
    construction_metadata: dict[str, Any],
) -> dict[str, Any]:
    selected = select_release_characters(inputs)
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
        for memory in response["memories"]:
            fact_ids = [job["fact_local_to_global"][item] for item in memory["fact_ids"]]
            accesses = [
                access_by_character_fact[(job["character_id"], item)] for item in fact_ids
            ]
            source_evidence_ids = sorted(
                {
                    item
                    for access in accesses
                    for item in access.get("supporting_evidence_ids", [])
                },
                key=lambda item: (
                    int(evidence_by_id[item].get("scene_order", 0)),
                    int(evidence_by_id[item].get("char_start", 0)),
                    item,
                ),
            )
            boundaries = []
            for access in accesses:
                acquired = int(access["acquired_at_scene"])
                char_ends = [
                    int(evidence_by_id[item].get("char_end", 0))
                    for item in access.get("supporting_evidence_ids", [])
                    if int(evidence_by_id[item].get("scene_order", 0)) == acquired
                ]
                boundaries.append((acquired, max(char_ends, default=0)))
            available_scene, available_char_end = max(boundaries)
            memory_id = stable_id(
                "task3-memory",
                inputs.movie_id,
                job["character_id"],
                *sorted(fact_ids),
            )
            memories_by_character[job["character_id"]].append(
                {
                    "memory_id": memory_id,
                    "memory_text": clean_text(memory["memory_text"]),
                    "importance": memory["importance"],
                    "tags": list(
                        dict.fromkeys(clean_text(item) for item in memory["tags"])
                    ),
                    "available_from": {
                        "scene_order": available_scene,
                        "char_end": available_char_end,
                    },
                    "source_episode_ids": [job["episode_id"]],
                    "source_fact_ids": fact_ids,
                    "source_evidence_ids": source_evidence_ids,
                    "knowledge_access": [
                        {
                            "fact_id": access["fact_or_event_id"],
                            "access_type": access["access_type"],
                            "acquired_at_scene": int(access["acquired_at_scene"]),
                        }
                        for access in accesses
                    ],
                }
            )

    roles = []
    used_legacy_roles: set[int] = set()
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
        legacy_role = _matching_legacy_role(
            character, inputs.legacy_roles.get("roles", []), used_legacy_roles
        )
        persona_card = (
            deepcopy(legacy_role.get("persona_card", {}))
            if legacy_role
            else {
                "traits": [],
                "speaking_style": [],
                "constraints": [],
                "dialogue_exemplars": [],
            }
        )
        relations = deepcopy(legacy_role.get("relations", [])) if legacy_role else []
        roles.append(
            {
                "character_id": character_id,
                "character_name": character["canonical_name"],
                "aliases": character.get("aliases", []),
                "identity_phases": character.get("identity_phases", []),
                "persona_card": persona_card,
                "relations": relations,
                "relation_count": len(relations),
                "profile_source": (
                    {
                        "type": "legacy_role_asset",
                        "legacy_character_name": legacy_role["character_name"],
                    }
                    if legacy_role
                    else {"type": "none"}
                ),
                "memories": memories,
                "memory_count": len(memories),
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
        relative = Path(clean_text(artifact.get("path")))
        path = (root / relative).resolve()
        if relative.is_absolute() or (path != root and root not in path.parents):
            raise ValueError(f"Temporal release artifact path escapes run: {relative}")
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
        names = {
            _normalize_name(role.get("character_name")),
            *(_normalize_name(item) for item in role.get("aliases", [])),
        }
        names.discard("")
        if current_names & names:
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


def _normalize_name(value: Any) -> str:
    return "".join(
        character for character in clean_text(value).casefold() if character.isalnum()
    )


def _has_first_person_marker(text: Any, language: str) -> bool:
    value = clean_text(text)
    if language == "Chinese":
        return any(marker in value for marker in ("我", "咱", "俺"))
    return bool(
        re.search(r"\b(?:I|me|my|mine|we|us|our|ours)\b", value, re.IGNORECASE)
    )
