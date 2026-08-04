from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .clients import ModelResponseParseError, ModelTransportError

from .chunking import TokenCounter
from .models import (
    EPISODE_RELATION_TYPES,
    EPISODE_RELATION_WEIGHTS,
    STORYLINE_FOCUS_TYPES,
    STORYLINE_STATUSES,
    clean_text,
    normalize_name,
    stable_id,
    unique_text,
)
from .io import atomic_write_json, load_json, sha256_json
from .prompt_loader import PROMPTS


_EPISODE_PROMPT = PROMPTS.get("episode")
EPISODE_SYSTEM, EPISODE_USER = _EPISODE_PROMPT.system, _EPISODE_PROMPT.user
_EPISODE_REPAIR_PROMPT = PROMPTS.get("episode_repair")
EPISODE_REPAIR_SYSTEM = _EPISODE_REPAIR_PROMPT.system
EPISODE_REPAIR_USER = _EPISODE_REPAIR_PROMPT.user
_EPISODE_RELATION_PROMPT = PROMPTS.get("episode_relation")
EPISODE_RELATION_SYSTEM = _EPISODE_RELATION_PROMPT.system
EPISODE_RELATION_USER = _EPISODE_RELATION_PROMPT.user
_STORYLINE_PROMPT = PROMPTS.get("storyline")
STORYLINE_SYSTEM, STORYLINE_USER = _STORYLINE_PROMPT.system, _STORYLINE_PROMPT.user


class JsonClient(Protocol):
    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class HierarchyConfig:
    language: str
    relation_candidate_window: int = 8
    relation_batch_size: int = 12
    relation_min_confidence: float = 0.55
    semantic_attempts: int = 2
    max_concurrency: int = 4


class NarrativeHierarchyBuilder:
    def __init__(
        self,
        *,
        movie_id: str,
        llm_client: JsonClient,
        config: HierarchyConfig,
        token_counter: TokenCounter,
        max_input_tokens: int,
        checkpoint_root: Path | None = None,
        entity_neighbors: dict[str, set[str]] | None = None,
    ):
        self.movie_id = movie_id
        self.llm_client = llm_client
        self.config = config
        self.token_counter = token_counter
        self.max_input_tokens = max_input_tokens
        self.checkpoint_root = checkpoint_root
        self.entity_neighbors = entity_neighbors or {}
        self._semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

    async def build_episodes(
        self,
        scene_records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prepared_packs = prepare_episode_pack_inputs(
            scene_records=scene_records,
            config=self.config,
            token_counter=self.token_counter,
            max_input_tokens=self.max_input_tokens,
            checkpoint_root=self.checkpoint_root,
        )
        unit_by_id = {
            clean_text(unit.get("unit_id")): unit
            for record in scene_records
            for unit in record.get("narrative_units", [])
            if clean_text(unit.get("unit_id"))
        }

        async def process_scene_pack(
            prepared: dict[str, Any],
        ) -> tuple[int, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
            pack_index = prepared["pack_index"]
            pack = prepared["pack"]
            local_to_unit = prepared["local_to_unit"]
            prompt_units = prepared["prompt_units"]
            checkpoint_path = prepared["checkpoint_path"]
            if checkpoint_path is not None and checkpoint_path.exists():
                cached = _load_hierarchy_checkpoint(
                    checkpoint_path,
                    kind="episode_pack",
                    index=pack_index,
                    input_sha256=prepared["input_sha256"],
                )
                normalized = _validate_episode_payload(
                    {"episodes": cached["result"]},
                    local_to_unit=local_to_unit,
                )
                call = {
                    **cached["call_metadata"],
                    "checkpoint_reused": True,
                    "checkpoint_path": str(checkpoint_path),
                }
            else:
                recovered = self._recover_episode_failure(
                    pack_index=pack_index,
                    local_to_unit=local_to_unit,
                )
                if recovered is None:
                    call, normalized = await self._validated_episode_call(
                        pack_index=pack_index,
                        user_prompt=prepared["user_prompt"],
                        prompt_tokens=prepared["prompt_tokens"],
                        prompt_units=prompt_units,
                        local_to_unit=local_to_unit,
                    )
                else:
                    call, normalized = recovered
                if checkpoint_path is not None:
                    _write_hierarchy_checkpoint(
                        checkpoint_path,
                        kind="episode_pack",
                        index=pack_index,
                        input_sha256=prepared["input_sha256"],
                        result=normalized,
                        call_metadata=call,
                    )
                    call = {**call, "checkpoint_path": str(checkpoint_path)}
            episodes, materialization_audit = _materialize_scene_episodes(
                movie_id=self.movie_id,
                pack_id=pack["pack_id"],
                normalized=normalized,
                local_to_unit=local_to_unit,
                unit_by_id=unit_by_id,
            )
            return pack_index, episodes, call, {
                "pack_id": pack["pack_id"],
                "source_scene_ids": pack["scene_ids"],
                "source_chunk_ids": pack["source_chunk_ids"],
                "source_tokens": pack["source_tokens"],
                "unit_count": len(pack["units"]),
                "primary_unit_count": sum(
                    unit.get("kind") in {"event", "interaction"}
                    for unit in pack["units"]
                ),
                "occasion_unit_count": sum(
                    unit.get("kind") == "occasion" for unit in pack["units"]
                ),
                **materialization_audit,
            }

        results = await asyncio.gather(
            *(process_scene_pack(prepared) for prepared in prepared_packs)
        )
        results.sort(key=lambda item: item[0])
        episodes = [episode for _, pack_episodes, _, _ in results for episode in pack_episodes]
        episodes.sort(
            key=lambda episode: (
                min(episode["source_scene_orders"]),
                episode["created_pack_id"],
                episode["scene_episode_order"],
            )
        )
        for order, episode in enumerate(episodes, start=1):
            episode["order"] = order
        _assert_exact_episode_coverage(scene_records, episodes)
        return episodes, {
            "aggregation_scope": "scene_local",
            "max_concurrency": max(1, self.config.max_concurrency),
            "scene_pack_count": len(prepared_packs),
            "packs": [item[3] for item in results],
            "llm_calls": [item[2] for item in results],
        }

    async def _validated_episode_call(
        self,
        *,
        pack_index: int,
        user_prompt: str,
        prompt_tokens: int,
        prompt_units: list[dict[str, Any]],
        local_to_unit: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        generation_calls: list[dict[str, Any]] = []
        repair_calls: list[dict[str, Any]] = []
        candidate_payload: Any = {}
        generation_metadata: dict[str, Any] = {
            "stage": f"episode_extraction:{pack_index:04d}",
            "call_kind": "formal_review",
            "prompt_tokens": prompt_tokens,
        }
        try:
            call = await self._generate_json(
                system_prompt=EPISODE_SYSTEM,
                user_prompt=user_prompt,
                stage=f"episode_extraction:{pack_index:04d}",
            )
            candidate_payload, deterministic_corrections = (
                _normalize_episode_candidate_payload(
                    call.data, local_to_unit=local_to_unit
                )
            )
            generation_metadata = {
                **call.metadata,
                "semantic_attempt": 1,
                "call_kind": "formal_review",
                "prompt_tokens": prompt_tokens,
                "deterministic_payload_corrections": deterministic_corrections,
            }
            generation_calls.append(generation_metadata)
            normalized = _validate_episode_payload(
                candidate_payload,
                local_to_unit=local_to_unit,
            )
            return {
                **generation_metadata,
                "generation_calls": generation_calls,
                "repair_calls": repair_calls,
            }, normalized
        except ModelTransportError:
            raise
        except ModelResponseParseError as parse_error:
            # A malformed JSON response gets one bounded schema repair.  The raw
            # response is preserved for the repair model; no second semantic
            # sampling or voting is introduced.
            generation_metadata = {
                **parse_error.metadata,
                "semantic_attempt": 1,
                "call_kind": "formal_review",
                "prompt_tokens": prompt_tokens,
                "response_parse_failed": True,
                "raw_response": parse_error.raw_text,
            }
            generation_calls.append(generation_metadata)
            candidate_payload = {"episodes": []}
            self._write_episode_failure(
                pack_index,
                phase="formal_parse",
                candidate_payload=candidate_payload,
                validation_error=parse_error,
                raw_response=parse_error.raw_text,
            )
            normalized, repair_metadata = await self._repair_episode_payload(
                pack_index=pack_index,
                candidate_payload=candidate_payload,
                validation_error=parse_error,
                prompt_units=prompt_units,
                local_to_unit=local_to_unit,
                raw_response=parse_error.raw_text,
            )
            repair_calls.extend(repair_metadata)
            return {
                **generation_metadata,
                "generation_calls": generation_calls,
                "repair_calls": repair_calls,
                "repaired": True,
            }, normalized
        except Exception as validation_error:
            self._write_episode_failure(
                pack_index,
                phase="formal",
                candidate_payload=candidate_payload,
                validation_error=validation_error,
            )
            normalized, repair_metadata = await self._repair_episode_payload(
                pack_index=pack_index,
                candidate_payload=candidate_payload,
                validation_error=validation_error,
                prompt_units=prompt_units,
                local_to_unit=local_to_unit,
            )
            repair_calls.extend(repair_metadata)
            return {
                **generation_metadata,
                "generation_calls": generation_calls,
                "repair_calls": repair_calls,
                "repaired": True,
            }, normalized

    async def _repair_episode_payload(
        self,
        *,
        pack_index: int,
        candidate_payload: Any,
        validation_error: Exception,
        prompt_units: list[dict[str, Any]],
        local_to_unit: dict[str, dict[str, Any]],
        raw_response: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        repair_prompt = EPISODE_REPAIR_USER.format(
            language=self.config.language,
            validation_error=clean_text(validation_error),
            narrative_units=json.dumps(prompt_units, ensure_ascii=False, indent=2),
            candidate_payload=json.dumps(candidate_payload, ensure_ascii=False, indent=2),
            raw_response=raw_response,
        )
        prompt_tokens = self.token_counter.count(
            EPISODE_REPAIR_SYSTEM + repair_prompt
        )
        if prompt_tokens > self.max_input_tokens:
            raise ValueError(
                "Episode repair prompt exceeds input budget: "
                f"pack={pack_index} tokens={prompt_tokens} "
                f"budget={self.max_input_tokens}"
            )
        call = await self._generate_json(
            system_prompt=EPISODE_REPAIR_SYSTEM,
            user_prompt=repair_prompt,
            stage=f"episode_repair:{pack_index:04d}",
        )
        metadata = {
            **call.metadata,
            "repair_attempt": 1,
            "call_kind": (
                "targeted_schema_repair" if raw_response else "targeted_repair"
            ),
            "prompt_tokens": prompt_tokens,
            "validation_error": clean_text(validation_error),
        }
        if raw_response:
            metadata["original_parse_error"] = clean_text(validation_error)
            metadata["original_raw_response"] = raw_response
        try:
            repaired_payload, deterministic_corrections = (
                _normalize_episode_candidate_payload(
                    call.data, local_to_unit=local_to_unit
                )
            )
            normalized = _validate_episode_payload(
                repaired_payload,
                local_to_unit=local_to_unit,
            )
            metadata["deterministic_payload_corrections"] = deterministic_corrections
        except Exception as repair_error:
            self._write_episode_failure(
                pack_index,
                phase="repair",
                candidate_payload=call.data,
                validation_error=repair_error,
            )
            raise
        return normalized, [metadata]

    def _recover_episode_failure(
        self,
        *,
        pack_index: int,
        local_to_unit: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if self.checkpoint_root is None:
            return None
        path = self.checkpoint_root / "04_episode_failures" / f"{pack_index:04d}.json"
        if not path.is_file():
            return None
        attempts = load_json(path).get("attempts", [])
        for attempt in reversed(attempts):
            payload, corrections = _normalize_episode_candidate_payload(
                attempt.get("candidate_payload"), local_to_unit=local_to_unit
            )
            try:
                normalized = _validate_episode_payload(
                    payload,
                    local_to_unit=local_to_unit,
                )
            except Exception:
                continue
            return {
                "stage": f"episode_recovery:{pack_index:04d}",
                "call_kind": "deterministic_failure_recovery",
                "recovered_from_failure_audit": str(path),
                "recovered_phase": attempt.get("phase"),
                "deterministic_payload_corrections": corrections,
                "generation_calls": [],
                "repair_calls": [],
                "model_calls_added": 0,
            }, normalized
        return None

    def _write_episode_failure(
        self,
        pack_index: int,
        *,
        phase: str,
        candidate_payload: Any,
        validation_error: Exception,
        raw_response: str = "",
    ) -> None:
        if self.checkpoint_root is None:
            return
        path = (
            self.checkpoint_root
            / "04_episode_failures"
            / f"{pack_index:04d}.json"
        )
        payload = load_json(path) if path.is_file() else {
            "schema_version": "stage_episode_failure_audit_v1",
            "pack_index": pack_index,
            "attempts": [],
        }
        payload["attempts"].append(
            {
                "phase": phase,
                "validation_error": clean_text(validation_error),
                "candidate_payload": candidate_payload,
                **({"raw_response": raw_response} if raw_response else {}),
            }
        )
        atomic_write_json(path, payload)

    def _write_storyline_failure(
        self,
        component_index: int,
        *,
        semantic_attempt: int,
        candidate_payload: Any,
        validation_error: Exception,
    ) -> None:
        if self.checkpoint_root is None:
            return
        path = (
            self.checkpoint_root
            / "06_storyline_failures"
            / f"{component_index:04d}.json"
        )
        payload = load_json(path) if path.is_file() else {
            "schema_version": "stage_storyline_failure_audit_v1",
            "component_index": component_index,
            "attempts": [],
        }
        payload["attempts"].append(
            {
                "semantic_attempt": semantic_attempt,
                "validation_error": clean_text(validation_error),
                "candidate_payload": candidate_payload,
            }
        )
        atomic_write_json(path, payload)

    def _recover_storyline_failure(
        self,
        component_index: int,
        *,
        episode_by_local_id: dict[str, dict[str, Any]],
        relation_by_local_id: dict[str, dict[str, Any]],
        entity_by_local_id: dict[str, dict[str, str]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if self.checkpoint_root is None:
            return None
        path = (
            self.checkpoint_root
            / "06_storyline_failures"
            / f"{component_index:04d}.json"
        )
        if not path.is_file():
            return None
        failure = load_json(path)
        empty_recovery: tuple[dict[str, Any], list[dict[str, Any]]] | None = None
        for attempt in reversed(failure.get("attempts", [])):
            normalized, corrections = _normalize_storyline_component_payload(
                attempt.get("candidate_payload"),
                episode_by_local_id=episode_by_local_id,
                relation_by_local_id=relation_by_local_id,
                entity_by_local_id=entity_by_local_id,
            )
            try:
                storylines = _validate_storyline_component_payload(
                    normalized,
                    episode_by_local_id=episode_by_local_id,
                    relation_by_local_id=relation_by_local_id,
                    entity_by_local_id=entity_by_local_id,
                )
            except Exception:
                if not isinstance(normalized, dict) or not isinstance(
                    normalized.get("storylines"), list
                ):
                    continue
                valid_raw_storylines: list[dict[str, Any]] = []
                for storyline_index, raw_storyline in enumerate(
                    normalized["storylines"]
                ):
                    try:
                        _validate_storyline_component_payload(
                            {"storylines": [raw_storyline]},
                            episode_by_local_id=episode_by_local_id,
                            relation_by_local_id=relation_by_local_id,
                            entity_by_local_id=entity_by_local_id,
                        )
                    except Exception as storyline_error:
                        corrections.append(
                            {
                                "action": "drop_invalid_storyline_after_failed_repair",
                                "storyline_index": storyline_index,
                                "name": clean_text(
                                    raw_storyline.get("name")
                                    if isinstance(raw_storyline, dict)
                                    else ""
                                ),
                                "validation_error": clean_text(storyline_error),
                            }
                        )
                        continue
                    valid_raw_storylines.append(raw_storyline)
                try:
                    storylines = _validate_storyline_component_payload(
                        {"storylines": valid_raw_storylines},
                        episode_by_local_id=episode_by_local_id,
                        relation_by_local_id=relation_by_local_id,
                        entity_by_local_id=entity_by_local_id,
                    )
                except Exception:
                    continue
            recovery = ({
                "stage": f"storyline_recovery:{component_index:04d}",
                "call_kind": "deterministic_failure_recovery",
                "recovered_from_failure_audit": str(path),
                "recovered_semantic_attempt": attempt.get("semantic_attempt"),
                "deterministic_payload_corrections": corrections,
                "model_calls_added": 0,
            }, storylines)
            if storylines:
                return recovery
            if empty_recovery is None:
                empty_recovery = recovery
        return empty_recovery

    async def build_episode_relations(
        self, episodes: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidates = make_episode_relation_candidates(
            episodes,
            window=max(1, self.config.relation_candidate_window),
            entity_neighbors=self.entity_neighbors,
        )
        candidate_batches = _pack_episode_relation_candidates(
            candidates,
            max_items=max(1, self.config.relation_batch_size),
            token_counter=self.token_counter,
            max_input_tokens=self.max_input_tokens,
        )
        accepted: list[dict[str, Any]] = []
        all_decisions: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []

        async def process_batch(
            batch_index: int,
            batch: list[dict[str, Any]],
        ) -> tuple[
            int,
            dict[str, dict[str, Any]],
            list[dict[str, Any]],
            dict[str, Any],
        ]:
            pair_by_id = {candidate["pair_id"]: candidate for candidate in batch}
            prompt_batch = [
                {
                    "pair_id": candidate["pair_id"],
                    "candidate_signals": candidate["prompt_candidate_signals"],
                    "earlier_episode": candidate["earlier_episode"],
                    "later_episode": candidate["later_episode"],
                }
                for candidate in batch
            ]
            checkpoint_path = self._checkpoint_path(
                "05_episode_relation_batches", f"{batch_index:04d}.json"
            )
            input_sha256 = sha256_json(
                {
                    "schema_version": "stage_episode_relation_batch_input_v1",
                    "system_prompt": EPISODE_RELATION_SYSTEM,
                    "user_prompt": EPISODE_RELATION_USER,
                    "candidate_pairs": prompt_batch,
                }
            )
            if checkpoint_path is not None and checkpoint_path.exists():
                cached = _load_hierarchy_checkpoint(
                    checkpoint_path,
                    kind="episode_relation_batch",
                    index=batch_index,
                    input_sha256=input_sha256,
                )
                decisions = _validate_relation_decisions(
                    {"decisions": cached["result"]}, pair_by_id
                )
                call_metadata = {
                    **cached["call_metadata"],
                    "checkpoint_reused": True,
                    "checkpoint_path": str(checkpoint_path),
                }
            else:
                last_error: Exception | None = None
                for semantic_attempt in range(
                    1, min(2, max(1, self.config.semantic_attempts)) + 1
                ):
                    try:
                        call = await self._generate_json(
                            system_prompt=EPISODE_RELATION_SYSTEM,
                            user_prompt=EPISODE_RELATION_USER.format(
                                candidate_pairs=json.dumps(
                                    prompt_batch, ensure_ascii=False, indent=2
                                )
                            )
                            + _validation_feedback(
                                last_error, "Decide every supplied pair_id."
                            ),
                            stage=f"episode_relations:{batch_index:04d}",
                        )
                        decisions = _validate_relation_decisions(call.data, pair_by_id)
                        call_metadata = {
                            **call.metadata,
                            "semantic_attempt": semantic_attempt,
                            "call_kind": (
                                "formal_review"
                                if semantic_attempt == 1
                                else "targeted_repair"
                            ),
                        }
                        last_error = None
                        break
                    except ModelTransportError:
                        raise
                    except Exception as exc:
                        last_error = exc
                if last_error is not None:
                    # A malformed response can omit one pair even when the rest of the
                    # batch is valid. Use one explicit schema-repair call for this batch;
                    # do not silently infer a relation or resample the whole run.
                    repair_prompt = (
                        _validation_feedback(
                            last_error,
                            "Return exactly one decision for every supplied pair_id. "
                            "Do not omit any pair, and do not add any pair_id.",
                        )
                        + "\nExact required pair_ids:\n"
                        + json.dumps(list(pair_by_id), ensure_ascii=False)
                    )
                    try:
                        repair_call = await self._generate_json(
                            system_prompt=EPISODE_RELATION_SYSTEM,
                            user_prompt=EPISODE_RELATION_USER.format(
                                candidate_pairs=json.dumps(
                                    prompt_batch, ensure_ascii=False, indent=2
                                )
                            )
                            + repair_prompt,
                            stage=f"episode_relations_targeted_repair:{batch_index:04d}",
                        )
                        decisions = _validate_relation_decisions(
                            repair_call.data, pair_by_id
                        )
                        call_metadata = {
                            **repair_call.metadata,
                            "semantic_attempt": (
                                min(2, max(1, self.config.semantic_attempts)) + 1
                            ),
                            "call_kind": "targeted_schema_repair",
                            "repair_reason": str(last_error),
                            "repair_pair_ids": list(pair_by_id),
                        }
                        last_error = None
                    except ModelTransportError:
                        raise
                    except Exception as repair_error:
                        last_error = repair_error
                if last_error is not None:
                    raise RuntimeError(
                        f"Episode relation batch {batch_index} failed semantic validation: "
                        f"{last_error}"
                    ) from last_error
                if checkpoint_path is not None:
                    _write_hierarchy_checkpoint(
                        checkpoint_path,
                        kind="episode_relation_batch",
                        index=batch_index,
                        input_sha256=input_sha256,
                        result=decisions,
                        call_metadata=call_metadata,
                    )
                    call_metadata = {
                        **call_metadata,
                        "checkpoint_path": str(checkpoint_path),
                    }
            return batch_index, pair_by_id, decisions, call_metadata

        batch_results = await asyncio.gather(
            *(
                process_batch(batch_index, batch)
                for batch_index, batch in enumerate(candidate_batches)
            )
        )
        batch_results.sort(key=lambda result: result[0])

        for _, pair_by_id, decisions, call_metadata in batch_results:
            calls.append(call_metadata)
            for decision in decisions:
                candidate = pair_by_id[decision["pair_id"]]
                audit_decision = {**decision, "candidate_signals": candidate["candidate_signals"]}
                all_decisions.append(audit_decision)
                if decision["relation_type"] == "none":
                    continue
                if decision["confidence"] < self.config.relation_min_confidence:
                    continue
                relation_type = decision["relation_type"]
                accepted.append(
                    {
                        "relation_id": stable_id(
                            "episode-relation",
                            self.movie_id,
                            candidate["source_id"],
                            relation_type,
                            candidate["target_id"],
                        ),
                        "source_id": candidate["source_id"],
                        "target_id": candidate["target_id"],
                        "relation_type": relation_type,
                        "description": decision["description"],
                        "evidence": decision["evidence"],
                        "confidence": decision["confidence"],
                        "weight": round(
                            EPISODE_RELATION_WEIGHTS[relation_type]
                            * decision["confidence"],
                            6,
                        ),
                    }
                )

        dag_relations, removed = break_cycles(accepted)
        return dag_relations, {
            "window": max(1, self.config.relation_candidate_window),
            "window_pair_count": _window_pair_count(
                len(episodes), max(1, self.config.relation_candidate_window)
            ),
            "candidate_count": len(candidates),
            "rejected_no_bridge_count": max(
                0,
                _window_pair_count(
                    len(episodes), max(1, self.config.relation_candidate_window)
                )
                - len(candidates),
            ),
            "candidate_signal_counts": dict(
                sorted(
                    Counter(
                        signal
                        for candidate in candidates
                        for signal in candidate["candidate_signals"]
                        if signal != "episode_distance"
                    ).items()
                )
            ),
            "candidate_degree": _candidate_degree(candidates),
            "batch_count": len(candidate_batches),
            "batch_sizes": [len(batch) for batch in candidate_batches],
            "candidates": candidates,
            "decisions": all_decisions,
            "llm_calls": calls,
            "cycle_edges_removed": removed,
        }

    async def build_storylines(
        self,
        episodes: list[dict[str, Any]],
        dag_relations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidates = build_storyline_candidates(episodes, dag_relations)
        episode_by_id = {episode["episode_id"]: episode for episode in episodes}
        relation_by_id = {
            relation["relation_id"]: relation for relation in dag_relations
        }
        candidates = _split_storyline_candidates_for_budget(
            candidates,
            episode_by_id=episode_by_id,
            relation_by_id=relation_by_id,
            language=self.config.language,
            token_counter=self.token_counter,
            max_input_tokens=self.max_input_tokens,
        )
        storylines = (
            [_build_chronological_backbone(self.movie_id, episodes)]
            if episodes
            else []
        )
        calls: list[dict[str, Any]] = []

        async def process_component(
            component_index: int, candidate: dict[str, Any]
        ) -> tuple[int, dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
            assets = _storyline_component_prompt_assets(
                candidate,
                episode_by_id=episode_by_id,
                relation_by_id=relation_by_id,
            )
            checkpoint_path = self._checkpoint_path(
                "06_storyline_components", f"{component_index:04d}.json"
            )
            input_sha256 = _storyline_component_input_sha256(
                language=self.config.language,
                candidate=candidate,
                assets=assets,
            )
            if checkpoint_path is not None and checkpoint_path.exists():
                cached = _load_hierarchy_checkpoint(
                    checkpoint_path,
                    kind="storyline_component",
                    index=component_index,
                    input_sha256=input_sha256,
                )
                component_storylines = _validate_storyline_component_payload(
                    {"storylines": cached["result"]},
                    episode_by_local_id=assets["episode_by_local_id"],
                    relation_by_local_id=assets["relation_by_local_id"],
                    entity_by_local_id=assets["entity_by_local_id"],
                )
                call_metadata = {
                    **cached["call_metadata"],
                    "checkpoint_reused": True,
                    "checkpoint_path": str(checkpoint_path),
                }
            else:
                recovered = self._recover_storyline_failure(
                    component_index,
                    episode_by_local_id=assets["episode_by_local_id"],
                    relation_by_local_id=assets["relation_by_local_id"],
                    entity_by_local_id=assets["entity_by_local_id"],
                )
                last_error: Exception | None = None
                if recovered is not None:
                    call_metadata, component_storylines = recovered
                semantic_attempts = (
                    range(1, min(2, max(1, self.config.semantic_attempts)) + 1)
                    if recovered is None
                    else ()
                )
                for semantic_attempt in semantic_attempts:
                    call = None
                    try:
                        user_prompt = STORYLINE_USER.format(
                            language=self.config.language,
                            component_id=candidate["component_id"],
                            available_entities=_storyline_json(
                                assets["prompt_entities"]
                            ),
                            episodes=_storyline_json(assets["prompt_episodes"]),
                            supporting_relations=_storyline_json(
                                assets["prompt_relations"]
                            ),
                        ) + _validation_feedback(
                            last_error,
                            "Use only supplied local IDs and omit unsupported storylines.",
                        )
                        prompt_tokens = self.token_counter.count(
                            STORYLINE_SYSTEM + user_prompt
                        )
                        if prompt_tokens > self.max_input_tokens:
                            raise ValueError(
                                f"Storyline component {component_index} exceeds input budget: "
                                f"{prompt_tokens}>{self.max_input_tokens}"
                            )
                        call = await self._generate_json(
                            system_prompt=STORYLINE_SYSTEM,
                            user_prompt=user_prompt,
                            stage=f"storyline_extraction:{component_index:04d}",
                        )
                        normalized_payload, corrections = (
                            _normalize_storyline_component_payload(
                                call.data,
                                episode_by_local_id=assets["episode_by_local_id"],
                                relation_by_local_id=assets["relation_by_local_id"],
                                entity_by_local_id=assets["entity_by_local_id"],
                            )
                        )
                        component_storylines = _validate_storyline_component_payload(
                            normalized_payload,
                            episode_by_local_id=assets["episode_by_local_id"],
                            relation_by_local_id=assets["relation_by_local_id"],
                            entity_by_local_id=assets["entity_by_local_id"],
                        )
                        call_metadata = {
                            **call.metadata,
                            "semantic_attempt": semantic_attempt,
                            "call_kind": (
                                "formal_review"
                                if semantic_attempt == 1
                                else "targeted_repair"
                            ),
                            "deterministic_payload_corrections": corrections,
                        }
                        last_error = None
                        break
                    except ModelTransportError:
                        raise
                    except Exception as exc:
                        self._write_storyline_failure(
                            component_index,
                            semantic_attempt=semantic_attempt,
                            candidate_payload=(call.data if call is not None else None),
                            validation_error=exc,
                        )
                        last_error = exc
                if last_error is not None:
                    raise RuntimeError(
                        f"Storyline component {component_index} failed semantic validation: "
                        f"{last_error}"
                    ) from last_error
                if checkpoint_path is not None:
                    _write_hierarchy_checkpoint(
                        checkpoint_path,
                        kind="storyline_component",
                        index=component_index,
                        input_sha256=input_sha256,
                        result=component_storylines,
                        call_metadata=call_metadata,
                    )
                    call_metadata = {
                        **call_metadata,
                        "checkpoint_path": str(checkpoint_path),
                    }
            return (
                component_index,
                candidate,
                assets,
                component_storylines,
                call_metadata,
            )

        gathered_results = await asyncio.gather(
            *(
                process_component(component_index, candidate)
                for component_index, candidate in enumerate(candidates, start=1)
            ),
            return_exceptions=True,
        )
        failures = [
            (index, result)
            for index, result in enumerate(gathered_results, start=1)
            if isinstance(result, BaseException)
        ]
        if failures:
            details = "; ".join(
                f"component {index}: {clean_text(error)}"
                for index, error in failures
            )
            raise RuntimeError(f"Storyline components failed after all calls settled: {details}")
        component_results = [
            result for result in gathered_results if not isinstance(result, BaseException)
        ]
        component_results.sort(key=lambda result: result[0])

        for _, candidate, assets, component_storylines, call_metadata in component_results:
            calls.append(call_metadata)
            for payload in component_storylines:
                child_episode_ids = [
                    assets["episode_id_by_local"][local_id]
                    for local_id in payload["episode_ids"]
                ]
                supporting_relation_ids = [
                    assets["relation_id_by_local"][local_id]
                    for local_id in payload["supporting_relation_ids"]
                ]
                focus_entities = [
                    assets["entity_by_local_id"][local_id]
                    for local_id in payload["focus_entity_ids"]
                ]
                selected_episodes = [
                    episode_by_id[episode_id] for episode_id in child_episode_ids
                ]
                storylines.append(
                    {
                        "storyline_id": stable_id(
                            "storyline",
                            self.movie_id,
                            payload["focus_type"],
                            *(entity["entity_id"] for entity in focus_entities),
                            *child_episode_ids,
                        ),
                        "name": payload["name"],
                        "description": payload["description"],
                        "focus_type": payload["focus_type"],
                        "focus_entity_ids": [
                            entity["entity_id"] for entity in focus_entities
                        ],
                        "focus_entities": focus_entities,
                        "component_id": candidate["component_id"],
                        "initial_state": payload["initial_state"],
                        "ordered_transitions": [
                            {
                                **transition,
                                "catalyst_episode_ids": [
                                    assets["episode_id_by_local"][local_id]
                                    for local_id in transition["catalyst_episode_ids"]
                                ],
                                "evidence_episode_ids": [
                                    assets["episode_id_by_local"][local_id]
                                    for local_id in transition["evidence_episode_ids"]
                                ],
                                "supporting_relation_ids": [
                                    assets["relation_id_by_local"][local_id]
                                    for local_id in transition[
                                        "supporting_relation_ids"
                                    ]
                                ],
                            }
                            for transition in payload["ordered_transitions"]
                        ],
                        "turning_point_episode_ids": [
                            assets["episode_id_by_local"][local_id]
                            for local_id in payload["turning_point_episode_ids"]
                        ],
                        "resolution_or_current_state": payload[
                            "resolution_or_current_state"
                        ],
                        "status": payload["status"],
                        "supporting_relation_ids": supporting_relation_ids,
                        "participant_entities": _unique_entity_refs(
                            participant
                            for episode in selected_episodes
                            for participant in episode.get("participant_entities", [])
                        ),
                        "child_episode_ids": child_episode_ids,
                        "source_scene_ids": unique_text(
                            scene_id
                            for episode in selected_episodes
                            for scene_id in episode["source_scene_ids"]
                        ),
                    }
                )
        for order, storyline in enumerate(storylines, start=1):
            storyline["order"] = order
        _assert_evolving_storyline_coverage(episodes, storylines)
        membership_counts = Counter(
            episode_id
            for storyline in storylines
            for episode_id in storyline["child_episode_ids"]
        )
        return storylines, {
            "candidate_count": len(candidates),
            "candidate_episode_counts": [
                len(candidate["episode_ids"]) for candidate in candidates
            ],
            "candidate_relation_counts": [
                len(candidate["supporting_relation_ids"])
                for candidate in candidates
            ],
            "accepted_evolution_storyline_count": max(0, len(storylines) - 1),
            "empty_component_count": sum(
                not result[3] for result in component_results
            ),
            "episode_membership_counts": dict(sorted(membership_counts.items())),
            "overlapping_episode_count": sum(
                count > 1 for count in membership_counts.values()
            ),
            "candidates": candidates,
            "llm_calls": calls,
        }

    def _checkpoint_path(self, directory: str, filename: str) -> Path | None:
        if self.checkpoint_root is None:
            return None
        return self.checkpoint_root / directory / filename

    async def _generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any:
        async with self._semaphore:
            return await self.llm_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage=stage,
            )


def _load_hierarchy_checkpoint(
    path: Path,
    *,
    kind: str,
    index: int,
    input_sha256: str,
) -> dict[str, Any]:
    payload = load_json(path)
    expected = {
        "schema_version",
        "kind",
        "index",
        "input_sha256",
        "result",
        "call_metadata",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"Invalid hierarchy checkpoint schema: {path}")
    if payload["schema_version"] != "stage_hierarchy_call_checkpoint_v1":
        raise ValueError(f"Unsupported hierarchy checkpoint version: {path}")
    if payload["kind"] != kind or payload["index"] != index:
        raise ValueError(f"Hierarchy checkpoint identity mismatch: {path}")
    if payload["input_sha256"] != input_sha256:
        raise ValueError(f"Hierarchy checkpoint input checksum mismatch: {path}")
    if not isinstance(payload["call_metadata"], dict):
        raise ValueError(f"Hierarchy checkpoint call metadata must be an object: {path}")
    return payload


def _write_hierarchy_checkpoint(
    path: Path,
    *,
    kind: str,
    index: int,
    input_sha256: str,
    result: Any,
    call_metadata: dict[str, Any],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": "stage_hierarchy_call_checkpoint_v1",
            "kind": kind,
            "index": index,
            "input_sha256": input_sha256,
            "result": result,
            "call_metadata": call_metadata,
        },
    )


def make_episode_packs(
    scene_records: list[dict[str, Any]],
    config: HierarchyConfig,
    *,
    token_counter: TokenCounter,
) -> list[dict[str, Any]]:
    chunks = _chunk_lookup(scene_records)
    packs: list[dict[str, Any]] = []

    for record in sorted(scene_records, key=lambda item: int(item["scene"]["order"])):
        units = list(record.get("narrative_units", []))
        primary_units = [
            unit for unit in units if unit.get("kind") in {"event", "interaction"}
        ]
        if not primary_units:
            continue
        pack = _episode_pack_from_units(
            units,
            chunk_lookup=chunks,
            token_counter=token_counter,
        )
        if len(pack["scene_ids"]) != 1:
            raise ValueError("Episode aggregation input must contain exactly one scene")
        packs.append(pack)
    for index, pack in enumerate(packs, start=1):
        pack["pack_id"] = f"PACK-{index:04d}"
    return packs


def prepare_episode_pack_inputs(
    *,
    scene_records: list[dict[str, Any]],
    config: HierarchyConfig,
    token_counter: TokenCounter,
    max_input_tokens: int,
    checkpoint_root: Path | None,
) -> list[dict[str, Any]]:
    chunk_lookup = _chunk_lookup(scene_records)
    packs = make_episode_packs(
        scene_records,
        config,
        token_counter=token_counter,
    )
    packs = _fit_episode_packs_to_prompt_budget(
        packs,
        chunk_lookup=chunk_lookup,
        language=config.language,
        token_counter=token_counter,
        max_input_tokens=max(1, max_input_tokens),
    )
    prepared: list[dict[str, Any]] = []
    for pack_index, pack in enumerate(packs, start=1):
        prepared.append(
            _prepare_episode_pack_input(
                pack=pack,
                pack_index=pack_index,
                chunk_lookup=chunk_lookup,
                config=config,
                token_counter=token_counter,
                checkpoint_root=checkpoint_root,
            )
        )
    return prepared


def _prepare_episode_pack_input(
    *,
    pack: dict[str, Any],
    pack_index: int,
    chunk_lookup: dict[str, dict[str, Any]],
    config: HierarchyConfig,
    token_counter: TokenCounter,
    checkpoint_root: Path | None,
) -> dict[str, Any]:
    pack = dict(pack)
    pack["pack_id"] = f"PACK-{pack_index:04d}"
    local_to_unit = {
        f"U{index:04d}": unit
        for index, unit in enumerate(pack["units"], start=1)
    }
    unit_id_to_local = {
        clean_text(unit.get("unit_id")): local_id
        for local_id, unit in local_to_unit.items()
    }
    source_scenes, local_chunk_ids = _episode_source_chunk_records(pack, chunk_lookup)
    prompt_units = [
        _episode_unit_prompt_record(
            local_id, unit, local_chunk_ids, unit_id_to_local
        )
        for local_id, unit in local_to_unit.items()
    ]
    user_prompt = _format_episode_prompt(
        language=config.language,
        source_scenes=source_scenes,
        narrative_units=prompt_units,
    )
    input_sha256 = sha256_json(
        {
            "schema_version": "stage_scene_local_episode_input_v1",
            "episode_system": EPISODE_SYSTEM,
            "episode_user": EPISODE_USER,
            "episode_repair_system": EPISODE_REPAIR_SYSTEM,
            "episode_repair_user": EPISODE_REPAIR_USER,
            "language": config.language,
            "aggregation_scope": "single_scene",
            "source_scenes": source_scenes,
            "narrative_units": prompt_units,
        }
    )
    return {
        "pack_index": pack_index,
        "pack": pack,
        "local_to_unit": local_to_unit,
        "source_scenes": source_scenes,
        "prompt_units": prompt_units,
        "user_prompt": user_prompt,
        "prompt_tokens": token_counter.count(EPISODE_SYSTEM + user_prompt),
        "input_sha256": input_sha256,
        "checkpoint_path": (
            checkpoint_root / "04_episode_packs" / f"{pack_index:04d}.json"
            if checkpoint_root is not None
            else None
        ),
    }


def build_episode_call_budget(
    *,
    scene_records: list[dict[str, Any]],
    config: HierarchyConfig,
    token_counter: TokenCounter,
    max_input_tokens: int,
    checkpoint_root: Path,
) -> dict[str, Any]:
    prepared = prepare_episode_pack_inputs(
        scene_records=scene_records,
        config=config,
        token_counter=token_counter,
        max_input_tokens=max_input_tokens,
        checkpoint_root=checkpoint_root,
    )
    packs: list[dict[str, Any]] = []
    for item in prepared:
        path = item["checkpoint_path"]
        cache_hit = False
        if path is not None and path.is_file():
            _load_hierarchy_checkpoint(
                path,
                kind="episode_pack",
                index=item["pack_index"],
                input_sha256=item["input_sha256"],
            )
            cache_hit = True
        packs.append(
            {
                "pack_index": item["pack_index"],
                "pack_id": item["pack"]["pack_id"],
                "unit_count": len(item["pack"]["units"]),
                "source_scene_ids": item["pack"]["scene_ids"],
                "source_tokens": item["pack"]["source_tokens"],
                "prompt_tokens_measured": item["prompt_tokens"],
                "input_sha256": item["input_sha256"],
                "checkpoint_path": str(path) if path else None,
                "cache_hit": cache_hit,
            }
        )
    misses = sum(not item["cache_hit"] for item in packs)
    return {
        "schema_version": "stage_episode_call_budget_v1",
        "scene_count": len(scene_records),
        "narrative_unit_count": sum(
            len(record.get("narrative_units", [])) for record in scene_records
        ),
        "pack_count": len(packs),
        "cache_hit_count": len(packs) - misses,
        "required_formal_calls": misses,
        "maximum_targeted_repair_calls": misses,
        "maximum_semantic_calls": misses * 2,
        "scene_local": True,
        "parallelizable": True,
        "max_concurrency": max(1, config.max_concurrency),
        "model_input_token_budget": max_input_tokens,
        "note": (
            "Each call is scene-local. Long scenes may be losslessly sharded by "
            "related Event/Interaction components and source chunks; this does not "
            "impose an Episode-count limit."
        ),
        "formal_prompt_tokens_measured": sum(
            item["prompt_tokens_measured"] for item in packs if not item["cache_hit"]
        ),
        "packs": packs,
    }


def _chunk_lookup(scene_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in scene_records:
        scene = record["scene"]
        for chunk in record.get("chunks", []):
            chunk_id = clean_text(chunk.get("chunk_id"))
            if not chunk_id:
                continue
            output[chunk_id] = {
                **chunk,
                "scene_id": scene["scene_id"],
                "source_scene_id": scene.get("source_scene_id", ""),
                "scene_order": scene["order"],
                "title": scene.get("title", ""),
                "subtitle": scene.get("subtitle", ""),
            }
    return output


def _episode_pack_from_units(
    units: list[dict[str, Any]],
    *,
    chunk_lookup: dict[str, dict[str, Any]],
    token_counter: TokenCounter,
) -> dict[str, Any]:
    scene_ids = unique_text(unit.get("source_scene_id") for unit in units)
    source_chunk_ids = unique_text(
        chunk_id
        for unit in units
        for chunk_id in unit.get("source_chunk_ids", [unit.get("source_chunk_id")])
        if chunk_id in chunk_lookup
    )
    source_tokens = sum(
        int(chunk_lookup[chunk_id].get("token_count") or token_counter.count(
            clean_text(chunk_lookup[chunk_id].get("content"))
        ))
        for chunk_id in source_chunk_ids
    )
    return {
        "pack_id": "",
        "scene_ids": scene_ids,
        "source_chunk_ids": source_chunk_ids,
        "source_tokens": source_tokens,
        "units": list(units),
    }


def _episode_source_chunk_records(
    pack: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    local_ids = {
        chunk_id: f"C{index:04d}"
        for index, chunk_id in enumerate(pack["source_chunk_ids"], start=1)
    }
    records = []
    for chunk_id in pack["source_chunk_ids"]:
        chunk = chunk_lookup[chunk_id]
        records.append(
            {
                "local_chunk_id": local_ids[chunk_id],
                "scene_order": chunk["scene_order"],
                "chunk_order": chunk["order"],
                "title": chunk["title"],
                "subtitle": chunk["subtitle"],
                "content": chunk["content"],
            }
        )
    return records, local_ids


def _fit_episode_packs_to_prompt_budget(
    packs: list[dict[str, Any]],
    *,
    chunk_lookup: dict[str, dict[str, Any]],
    language: str,
    token_counter: TokenCounter,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    queue = list(packs)
    fitted: list[dict[str, Any]] = []
    while queue:
        pack = queue.pop(0)
        if _episode_pack_prompt_tokens(
            pack,
            chunk_lookup=chunk_lookup,
            language=language,
            token_counter=token_counter,
        ) <= max_input_tokens:
            fitted.append(pack)
            continue
        shards = _split_episode_pack_to_prompt_budget(
            pack,
            chunk_lookup=chunk_lookup,
            language=language,
            token_counter=token_counter,
            max_input_tokens=max_input_tokens,
        )
        if len(shards) <= 1:
            raise ValueError(
                f"Scene-local Episode prompt cannot be losslessly sharded: "
                f"scene={pack['scene_ids'][0]}"
            )
        queue = [*shards, *queue]
    for index, pack in enumerate(fitted, start=1):
        pack["pack_id"] = f"PACK-{index:04d}"
    return fitted


def _episode_pack_prompt_tokens(
    pack: dict[str, Any],
    *,
    chunk_lookup: dict[str, dict[str, Any]],
    language: str,
    token_counter: TokenCounter,
) -> int:
    source_records, local_chunk_ids = _episode_source_chunk_records(pack, chunk_lookup)
    local_to_unit = {
        f"U{index:04d}": unit
        for index, unit in enumerate(pack["units"], start=1)
    }
    unit_id_to_local = {
        clean_text(unit.get("unit_id")): local_id
        for local_id, unit in local_to_unit.items()
    }
    prompt_units = [
        _episode_unit_prompt_record(
            local_id, unit, local_chunk_ids, unit_id_to_local
        )
        for local_id, unit in local_to_unit.items()
    ]
    prompt = _format_episode_prompt(
        language=language,
        source_scenes=source_records,
        narrative_units=prompt_units,
    )
    return token_counter.count(EPISODE_SYSTEM + prompt)


def _split_episode_pack_to_prompt_budget(
    pack: dict[str, Any],
    *,
    chunk_lookup: dict[str, dict[str, Any]],
    language: str,
    token_counter: TokenCounter,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    units = list(pack["units"])
    primary = [
        unit for unit in units if unit.get("kind") in {"event", "interaction"}
    ]
    occasions = [unit for unit in units if unit.get("kind") == "occasion"]
    if len(primary) <= 1:
        return [pack]

    primary_by_id = {
        clean_text(unit.get("unit_id")): unit
        for unit in primary
        if clean_text(unit.get("unit_id"))
    }
    parent = {item: item for item in primary_by_id}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for unit in primary:
        unit_id = clean_text(unit.get("unit_id"))
        related = clean_text(unit.get("related_event_unit_id"))
        if unit_id in parent and related in parent:
            union(unit_id, related)

    component_by_root: dict[str, list[dict[str, Any]]] = {}
    unkeyed: list[list[dict[str, Any]]] = []
    for unit in primary:
        unit_id = clean_text(unit.get("unit_id"))
        if unit_id in parent:
            component_by_root.setdefault(find(unit_id), []).append(unit)
        else:
            unkeyed.append([unit])
    components = [*component_by_root.values(), *unkeyed]
    primary_order = {id(unit): index for index, unit in enumerate(primary)}
    components.sort(key=lambda values: min(primary_order[id(unit)] for unit in values))

    bundles = [
        _episode_component_bundle(component, occasions)
        for component in components
    ]
    shards: list[dict[str, Any]] = []
    pending_units: list[dict[str, Any]] = []
    for bundle in bundles:
        candidate_units = _unique_units([*pending_units, *bundle])
        candidate = _episode_pack_from_units(
            candidate_units,
            chunk_lookup=chunk_lookup,
            token_counter=token_counter,
        )
        if _episode_pack_prompt_tokens(
            candidate,
            chunk_lookup=chunk_lookup,
            language=language,
            token_counter=token_counter,
        ) <= max_input_tokens:
            pending_units = candidate_units
            continue
        if pending_units:
            shards.append(
                _episode_pack_from_units(
                    pending_units,
                    chunk_lookup=chunk_lookup,
                    token_counter=token_counter,
                )
            )
            pending_units = list(bundle)
        else:
            component_primary = [
                unit
                for unit in bundle
                if unit.get("kind") in {"event", "interaction"}
            ]
            if len(component_primary) <= 1:
                return [pack]
            bundles.extend(
                _episode_component_bundle([unit], occasions)
                for unit in component_primary
            )
    if pending_units:
        shards.append(
            _episode_pack_from_units(
                pending_units,
                chunk_lookup=chunk_lookup,
                token_counter=token_counter,
            )
        )
    return shards


def _episode_component_bundle(
    primary_units: list[dict[str, Any]],
    occasions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_ids = {
        clean_text(unit.get("unit_id")) for unit in primary_units
    }
    related_occasion_ids = {
        clean_text(unit.get("related_occasion_unit_id")) for unit in primary_units
    }
    primary_chunk_ids = {
        chunk_id
        for unit in primary_units
        for chunk_id in unit.get(
            "source_chunk_ids", [unit.get("source_chunk_id")]
        )
        if clean_text(chunk_id)
    }
    context = [
        unit
        for unit in occasions
        if clean_text(unit.get("unit_id")) in related_occasion_ids
        or clean_text(unit.get("related_event_unit_id")) in primary_ids
        or bool(
            primary_chunk_ids
            & {
                clean_text(chunk_id)
                for chunk_id in unit.get(
                    "source_chunk_ids", [unit.get("source_chunk_id")]
                )
                if clean_text(chunk_id)
            }
        )
    ]
    return _unique_units([*primary_units, *context])


def _unique_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for unit in units:
        key = clean_text(unit.get("unit_id")) or f"object:{id(unit)}"
        if key not in seen:
            seen.add(key)
            output.append(unit)
    return output


def _episode_unit_prompt_record(
    local_id: str,
    unit: dict[str, Any],
    local_chunk_ids: dict[str, str],
    unit_id_to_local: dict[str, str],
) -> dict[str, Any]:
    record = {
        "local_unit_id": local_id,
        "kind": unit["kind"],
        "modality": clean_text(unit.get("modality")) or "asserted",
        "subtype": clean_text(
            unit.get("event_subtype")
            or unit.get("occasion_type")
            or unit.get("interaction_type")
        ),
        "name": unit["name"],
        "description": unit["description"],
        "participants": _prompt_entity_names(unit.get("participant_entities", [])),
        "locations": _prompt_entity_names(unit.get("location_entities", [])),
        "times": _prompt_entity_names(unit.get("time_entities", [])),
        "subject": clean_text(unit.get("subject")),
        "object": clean_text(unit.get("object")),
        "occasion": unit["name"] if unit.get("kind") == "occasion" else "",
        "related_event_local_unit_id": unit_id_to_local.get(
            clean_text(unit.get("related_event_unit_id")), ""
        ),
        "related_occasion_local_unit_id": unit_id_to_local.get(
            clean_text(unit.get("related_occasion_unit_id")), ""
        ),
        "related_occasion": clean_text(unit.get("related_occasion")),
        "state_before": clean_text(unit.get("state_before")),
        "state_after": clean_text(unit.get("state_after")),
        "intent": clean_text(unit.get("intent")),
        "cause_hints": list(unit.get("cause_hints") or []),
        "effect_hints": list(unit.get("effect_hints") or []),
        "outcome": clean_text(unit.get("outcome")),
        "polarity": clean_text(unit.get("polarity")),
        "institutional_context": clean_text(unit.get("institutional_context")),
        "setting": clean_text(unit.get("setting")),
        "source_chunk_ids": [
            local_chunk_ids[chunk_id]
            for chunk_id in unit.get(
                "source_chunk_ids", [unit.get("source_chunk_id")]
            )
            if chunk_id in local_chunk_ids
        ],
        "evidence": clean_text(unit.get("evidence")),
    }
    return record


def _prompt_entity_names(values: Any) -> list[str]:
    return unique_text(
        value.get("canonical_name") if isinstance(value, dict) else value
        for value in (values or [])
    )


def _format_episode_prompt(
    *,
    language: str,
    source_scenes: list[dict[str, Any]],
    narrative_units: list[dict[str, Any]],
) -> str:
    return EPISODE_USER.format(
        language=language,
        source_scenes=json.dumps(source_scenes, ensure_ascii=False, indent=2),
        narrative_units=json.dumps(narrative_units, ensure_ascii=False, indent=2),
    )


def _episode_anchor_assets(
    child_units: list[dict[str, Any]],
    *,
    participant_units: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    participant_units = participant_units if participant_units is not None else child_units
    participant_entities = _unique_entity_refs(
        entity
        for unit in participant_units
        for entity in unit.get("participant_entities", [])
    )
    location_entities = _unique_entity_refs(
        entity
        for unit in child_units
        for entity in unit.get("location_entities", [])
    )
    time_entities = _unique_entity_refs(
        entity
        for unit in child_units
        for entity in unit.get("time_entities", [])
    )
    occasion_entities = _unique_entity_refs(
        [
            *(
                {
                    "entity_id": unit.get("entity_id"),
                    "canonical_name": unit.get("name"),
                }
                for unit in child_units
                if unit.get("kind") == "occasion" and unit.get("entity_id")
            ),
            *(
                {
                    "entity_id": unit.get("related_occasion_entity_id"),
                    "canonical_name": unit.get("related_occasion"),
                }
                for unit in child_units
                if unit.get("related_occasion_entity_id")
            ),
        ]
    )
    endpoint_entities = _unique_entity_refs(
        {
            "entity_id": unit.get(field + "_entity_id"),
            "canonical_name": unit.get(field),
        }
        for unit in participant_units
        for field in ("subject", "object")
        if unit.get(field + "_entity_id")
    )
    roles_by_id: dict[str, set[str]] = defaultdict(set)
    names_by_id: dict[str, str] = {}
    for role, entities in (
        ("participant", participant_entities),
        ("location", location_entities),
        ("time", time_entities),
        ("occasion", occasion_entities),
        ("interaction_endpoint", endpoint_entities),
    ):
        for entity in entities:
            entity_id = entity["entity_id"]
            roles_by_id[entity_id].add(role)
            names_by_id.setdefault(entity_id, entity["canonical_name"])
    anchor_entities = [
        {
            "entity_id": entity_id,
            "canonical_name": names_by_id[entity_id],
            "roles": sorted(roles),
        }
        for entity_id, roles in sorted(roles_by_id.items())
    ]
    related_unit_ids = unique_text(
        unit.get(field)
        for unit in child_units
        for field in ("related_event_unit_id", "related_occasion_unit_id")
        if unit.get(field)
    )
    return {
        "participant_entities": participant_entities,
        "participant_entity_ids": [item["entity_id"] for item in participant_entities],
        "location_entities": location_entities,
        "location_entity_ids": [item["entity_id"] for item in location_entities],
        "time_entities": time_entities,
        "time_entity_ids": [item["entity_id"] for item in time_entities],
        "occasion_entities": occasion_entities,
        "occasion_entity_ids": [item["entity_id"] for item in occasion_entities],
        "interaction_endpoint_entities": endpoint_entities,
        "interaction_endpoint_entity_ids": [item["entity_id"] for item in endpoint_entities],
        "anchor_entities": anchor_entities,
        "anchor_entity_ids": [item["entity_id"] for item in anchor_entities],
        "related_unit_ids": related_unit_ids,
    }


def _normalize_episode_candidate_payload(
    payload: Any,
    *,
    local_to_unit: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        return payload, []
    episodes: list[Any] = []
    corrections: list[dict[str, Any]] = []
    seen_primary_ids: set[str] = set()
    primary_ids = {
        local_id
        for local_id, unit in (local_to_unit or {}).items()
        if unit.get("kind") in {"event", "interaction"}
    }
    occasion_ids = {
        local_id
        for local_id, unit in (local_to_unit or {}).items()
        if unit.get("kind") == "occasion"
    }
    for index, item in enumerate(payload["episodes"], start=1):
        if (
            local_to_unit is None
            and isinstance(item, dict)
            and isinstance(item.get("child_unit_ids"), list)
            and not any(clean_text(value) for value in item["child_unit_ids"])
        ):
            corrections.append(
                {
                    "episode_index": index,
                    "action": "drop_empty_episode",
                    "name": clean_text(item.get("name")),
                }
            )
            continue
        normalized_item = dict(item) if isinstance(item, dict) else item
        if (
            local_to_unit is not None
            and isinstance(normalized_item, dict)
            and isinstance(normalized_item.get("child_unit_ids"), list)
        ):
            retained_children = []
            moved_occasions = []
            dropped_children = []
            duplicate_children = []
            for value in normalized_item["child_unit_ids"]:
                local_id = clean_text(value)
                if local_id in occasion_ids:
                    moved_occasions.append(local_id)
                elif local_id not in primary_ids:
                    dropped_children.append(local_id)
                elif local_id in seen_primary_ids:
                    duplicate_children.append(local_id)
                else:
                    seen_primary_ids.add(local_id)
                    retained_children.append(local_id)
            normalized_item["child_unit_ids"] = unique_text(retained_children)
            normalized_item["context_occasion_unit_ids"] = unique_text(
                [
                    *normalized_item.get("context_occasion_unit_ids", []),
                    *moved_occasions,
                ]
            )
            if moved_occasions:
                corrections.append(
                    {
                        "episode_index": index,
                        "action": "move_occasion_children_to_context",
                        "local_unit_ids": unique_text(moved_occasions),
                    }
                )
            if dropped_children:
                corrections.append(
                    {
                        "episode_index": index,
                        "action": "drop_unknown_child_ids",
                        "local_unit_ids": unique_text(dropped_children),
                    }
                )
            if duplicate_children:
                corrections.append(
                    {
                        "episode_index": index,
                        "action": "drop_duplicate_primary_children_after_first",
                        "local_unit_ids": unique_text(duplicate_children),
                    }
                )
            trigger_ids = [
                clean_text(value)
                for value in normalized_item.get("trigger_unit_ids", [])
                if clean_text(value) in normalized_item["child_unit_ids"]
            ]
            if not trigger_ids and normalized_item["child_unit_ids"]:
                trigger_ids = [normalized_item["child_unit_ids"][0]]
                corrections.append(
                    {
                        "episode_index": index,
                        "action": "derive_trigger_from_first_retained_child",
                    }
                )
            normalized_item["trigger_unit_ids"] = unique_text(trigger_ids)
            if not normalized_item["child_unit_ids"]:
                corrections.append(
                    {
                        "episode_index": index,
                        "action": "drop_empty_episode_after_child_normalization",
                        "name": clean_text(normalized_item.get("name")),
                    }
                )
                continue
        if (
            local_to_unit is not None
            and isinstance(normalized_item, dict)
            and isinstance(normalized_item.get("context_occasion_unit_ids"), list)
        ):
            valid_contexts: list[str] = []
            invalid_contexts: list[str] = []
            for value in normalized_item["context_occasion_unit_ids"]:
                local_id = clean_text(value)
                unit = local_to_unit.get(local_id)
                if unit is not None and unit.get("kind") == "occasion":
                    valid_contexts.append(local_id)
                else:
                    invalid_contexts.append(local_id)
            if invalid_contexts:
                normalized_item["context_occasion_unit_ids"] = unique_text(
                    valid_contexts
                )
                corrections.append(
                    {
                        "episode_index": index,
                        "action": "drop_invalid_occasion_context",
                        "local_unit_ids": unique_text(invalid_contexts),
                    }
                )
        if (
            isinstance(normalized_item, dict)
            and not clean_text(normalized_item.get("initial_situation"))
            and isinstance(normalized_item.get("progression_steps"), list)
            and normalized_item["progression_steps"]
        ):
            normalized_item["initial_situation"] = clean_text(
                normalized_item["progression_steps"][0]
            )
            corrections.append(
                {
                    "episode_index": index,
                    "action": "derive_missing_initial_situation_from_first_progression_step",
                }
            )
        episodes.append(normalized_item)
    if local_to_unit is not None:
        for local_id in sorted(primary_ids - seen_primary_ids):
            episodes.append(
                _deterministic_singleton_episode(
                    local_id,
                    local_to_unit[local_id],
                )
            )
            corrections.append(
                {
                    "action": "add_missing_primary_as_singleton_episode",
                    "local_unit_id": local_id,
                }
            )
    normalized = dict(payload)
    normalized["episodes"] = episodes
    return normalized, corrections


def _deterministic_singleton_episode(
    local_id: str, unit: dict[str, Any]
) -> dict[str, Any]:
    description = clean_text(unit.get("description")) or clean_text(unit.get("name"))
    state_before = clean_text(unit.get("state_before"))
    state_after = clean_text(unit.get("state_after"))
    outcome = (
        state_after
        or clean_text(unit.get("outcome"))
        or description
    )
    state_changes = []
    if state_before or state_after:
        state_changes.append(
            f"{state_before or 'prior state unspecified'} -> {state_after or outcome}"
        )
    return {
        "name": clean_text(unit.get("name")) or f"Episode for {local_id}",
        "description": description,
        "child_unit_ids": [local_id],
        "context_occasion_unit_ids": [],
        "trigger_unit_ids": [local_id],
        "participants": _prompt_entity_names(
            unit.get("participant_entities") or unit.get("participants", [])
        ),
        "setting": clean_text(unit.get("setting")),
        "initial_situation": state_before or description,
        "progression_steps": [description],
        "outcome": outcome,
        "state_changes": state_changes,
        "causal_links": unique_text(unit.get("cause_hints") or []),
        "open_threads": [],
        "closed_threads": [],
        "key_evidence": unique_text([unit.get("evidence") or description]),
    }


def _normalize_episode_text_list(
    value: Any,
) -> tuple[list[str], str | None]:
    if isinstance(value, str):
        text = clean_text(value)
        return ([text] if text else []), "wrapped_scalar_string"
    if not isinstance(value, (list, tuple, set)):
        return [], None
    values = unique_text(value)
    if len(values) >= 4 and all(len(item) <= 1 for item in values):
        return ["".join(values)], "joined_character_sequence"
    return values, None


def _normalize_local_reference_list(
    value: Any,
    allowed: dict[str, Any],
    label: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Normalize harmless zero-padding/case drift in component-local IDs.

    Storyline prompts expose IDs such as ``R033``. Models occasionally return
    ``R33`` or ``r33`` for the same supplied item. We only repair a reference
    when its alphabetic prefix and integer suffix identify exactly one allowed
    local ID; unknown IDs remain untouched for normal validation to reject.
    """
    values, _ = _normalize_episode_text_list(value)
    allowed_keys = set(allowed)
    normalized: list[str] = []
    corrections: list[dict[str, str]] = []
    for item in values:
        if item in allowed_keys:
            normalized.append(item)
            continue
        match = re.fullmatch(r"([A-Za-z]+)[ _-]?(\d+)", item)
        if not match:
            normalized.append(item)
            continue
        prefix = match.group(1).casefold()
        number = int(match.group(2))
        matches = [
            candidate
            for candidate in allowed_keys
            if (candidate_match := re.fullmatch(
                r"([A-Za-z]+)[ _-]?(\d+)", candidate
            ))
            and candidate_match.group(1).casefold() == prefix
            and int(candidate_match.group(2)) == number
        ]
        if len(matches) != 1:
            normalized.append(item)
            continue
        canonical = matches[0]
        normalized.append(canonical)
        corrections.append(
            {
                "action": "normalize_component_local_id",
                "label": label,
                "from": item,
                "to": canonical,
            }
        )
    return normalized, corrections


def _validate_episode_payload(
    payload: dict[str, Any],
    *,
    local_to_unit: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(payload) != {"episodes"} or not isinstance(payload["episodes"], list):
        raise ValueError("Episode extraction must return exactly an episodes array")
    if not payload["episodes"]:
        raise ValueError("A scene with Event/Interaction units requires Episodes")
    primary_ids = {
        local_id
        for local_id, unit in local_to_unit.items()
        if unit.get("kind") in {"event", "interaction"}
    }
    occasion_ids = {
        local_id
        for local_id, unit in local_to_unit.items()
        if unit.get("kind") == "occasion"
    }
    seen: list[str] = []
    output: list[dict[str, Any]] = []
    allowed_participants = {
        normalize_name(participant): participant
        for unit in local_to_unit.values()
        for participant in [
            *list(unit.get("participants", [])),
            *[
                item.get("canonical_name", "")
                for item in unit.get("participant_entities", [])
                if isinstance(item, dict)
            ],
        ]
        if normalize_name(participant)
    }
    for raw in payload["episodes"]:
        if not isinstance(raw, dict):
            raise ValueError("Every episode must be an object")
        name = clean_text(raw.get("name"))
        description = clean_text(raw.get("description"))
        children = raw.get("child_unit_ids")
        if not name or not description or not isinstance(children, list) or not children:
            raise ValueError("Every episode requires name, description, and child_unit_ids")
        child_ids = [clean_text(child) for child in children]
        if any(child not in primary_ids for child in child_ids):
            raise ValueError(
                "Episode child_unit_ids may contain only supplied Event/Interaction ids"
            )
        context_ids = [
            clean_text(value) for value in raw.get("context_occasion_unit_ids", [])
        ]
        if any(value not in occasion_ids for value in context_ids):
            raise ValueError(
                "context_occasion_unit_ids may contain only supplied Occasion ids"
            )
        trigger_ids = [clean_text(value) for value in raw.get("trigger_unit_ids", [])]
        if not set(trigger_ids).issubset(child_ids):
            raise ValueError("trigger_unit_ids must be a subset of child_unit_ids")
        text_field_normalizations: list[dict[str, str]] = []
        generated_participants, participant_normalization = _normalize_episode_text_list(
            raw.get("participants")
        )
        if participant_normalization:
            text_field_normalizations.append(
                {"field": "participants", "action": participant_normalization}
            )
        participants: list[str] = []
        unmatched_participants: list[str] = []
        for participant in generated_participants:
            normalized = normalize_name(participant)
            if normalized not in allowed_participants:
                unmatched_participants.append(participant)
                continue
            participants.append(allowed_participants[normalized])
        seen.extend(child_ids)
        normalized_text_fields: dict[str, list[str]] = {}
        for field in (
            "progression_steps",
            "state_changes",
            "causal_links",
            "open_threads",
            "closed_threads",
            "key_evidence",
        ):
            values, action = _normalize_episode_text_list(raw.get(field))
            normalized_text_fields[field] = values
            if action:
                text_field_normalizations.append({"field": field, "action": action})
        initial_situation = clean_text(raw.get("initial_situation"))
        outcome = clean_text(raw.get("outcome"))
        missing_process_fields = [
            field
            for field, present in (
                ("trigger_unit_ids", bool(trigger_ids)),
                ("initial_situation", bool(initial_situation)),
                ("progression_steps", bool(normalized_text_fields["progression_steps"])),
                ("outcome", bool(outcome)),
                ("key_evidence", bool(normalized_text_fields["key_evidence"])),
            )
            if not present
        ]
        if missing_process_fields:
            raise ValueError(
                "Episode lacks cause-progression-result structure: "
                + ", ".join(missing_process_fields)
            )
        output.append(
            {
                "name": name,
                "description": description,
                "child_unit_ids": child_ids,
                "context_occasion_unit_ids": unique_text(context_ids),
                "trigger_unit_ids": unique_text(trigger_ids),
                "participants": participants,
                "unmatched_generated_participants": unmatched_participants,
                "text_field_normalizations": text_field_normalizations,
                "setting": clean_text(raw.get("setting")),
                "initial_situation": initial_situation,
                "progression_steps": normalized_text_fields["progression_steps"],
                "outcome": outcome,
                "state_changes": normalized_text_fields["state_changes"],
                "causal_links": normalized_text_fields["causal_links"],
                "open_threads": normalized_text_fields["open_threads"],
                "closed_threads": normalized_text_fields["closed_threads"],
                "key_evidence": normalized_text_fields["key_evidence"],
            }
        )
    expected = primary_ids
    if set(seen) != expected or len(seen) != len(expected):
        missing = sorted(expected - set(seen))
        duplicates = sorted({item for item in seen if seen.count(item) > 1})
        raise ValueError(
            f"Episode child coverage is not exact; missing={missing}, duplicates={duplicates}"
        )
    return output


def _materialize_scene_episodes(
    *,
    movie_id: str,
    pack_id: str,
    normalized: list[dict[str, Any]],
    local_to_unit: dict[str, dict[str, Any]],
    unit_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    local_id_by_unit_id = {
        clean_text(unit.get("unit_id")): local_id
        for local_id, unit in local_to_unit.items()
    }
    episodes: list[dict[str, Any]] = []
    participant_corrections: list[dict[str, Any]] = []
    text_field_normalizations: list[dict[str, Any]] = []
    for index, item in enumerate(normalized, start=1):
        child_ids = [
            clean_text(local_to_unit[local_id].get("unit_id"))
            for local_id in item["child_unit_ids"]
        ]
        context_local_ids = list(item["context_occasion_unit_ids"])
        for child_id in child_ids:
            related_id = clean_text(unit_by_id[child_id].get("related_occasion_unit_id"))
            local_id = local_id_by_unit_id.get(related_id)
            if local_id and local_to_unit[local_id].get("kind") == "occasion":
                context_local_ids.append(local_id)
        context_local_ids = unique_text(context_local_ids)
        context_ids = [
            clean_text(local_to_unit[local_id].get("unit_id"))
            for local_id in context_local_ids
        ]
        primary_units = [unit_by_id[unit_id] for unit_id in child_ids]
        episode_units = [*primary_units, *(unit_by_id[unit_id] for unit_id in context_ids)]
        source_modalities = unique_text(
            clean_text(unit_by_id[unit_id].get("modality")) or "asserted"
            for unit_id in child_ids
        )
        episode_modality = (
            source_modalities[0] if len(source_modalities) == 1 else "uncertain"
        )
        anchors = _episode_anchor_assets(
            episode_units,
            participant_units=primary_units,
        )
        episode_id = stable_id("episode", movie_id, pack_id, index, *child_ids)
        unmatched = item.get("unmatched_generated_participants", [])
        if unmatched:
            participant_corrections.append(
                {
                    "episode_id": episode_id,
                    "unmatched_generated_participants": unmatched,
                    "canonical_participants": [
                        ref["canonical_name"] for ref in anchors["participant_entities"]
                    ],
                    "mechanism": "derive_from_scene_local_child_units",
                }
            )
        trigger_ids = [
            clean_text(local_to_unit[local_id].get("unit_id"))
            for local_id in item["trigger_unit_ids"]
        ]
        for correction in item.get("text_field_normalizations", []):
            text_field_normalizations.append(
                {"episode_id": episode_id, **correction}
            )
        episodes.append(
            {
                "episode_id": episode_id,
                **{key: value for key, value in item.items() if key not in {
                    "child_unit_ids", "context_occasion_unit_ids", "trigger_unit_ids",
                    "participants", "unmatched_generated_participants",
                    "text_field_normalizations",
                }},
                "participants": [
                    ref["canonical_name"] for ref in anchors["participant_entities"]
                ],
                "main_participants": item["participants"],
                "child_unit_ids": child_ids,
                "primary_unit_ids": child_ids,
                "context_occasion_unit_ids": context_ids,
                "trigger_unit_ids": trigger_ids,
                "modality": episode_modality,
                "source_modalities": source_modalities,
                **anchors,
                "source_scene_ids": unique_text(
                    unit["source_scene_id"] for unit in episode_units
                ),
                "source_scene_orders": sorted(
                    {int(unit["source_scene_order"]) for unit in episode_units}
                ),
                "created_pack_id": pack_id,
                "scene_episode_order": index,
            }
        )
    if any(len(episode["source_scene_ids"]) != 1 for episode in episodes):
        raise ValueError("A scene-local Episode cannot span multiple scenes")
    return episodes, {
        "episode_count": len(episodes),
        "participant_grounding_corrections": participant_corrections,
        "text_field_normalizations": text_field_normalizations,
    }


def _assert_exact_episode_coverage(
    scene_records: list[dict[str, Any]], episodes: list[dict[str, Any]]
) -> None:
    expected = [
        unit["unit_id"]
        for record in scene_records
        for unit in record.get("narrative_units", [])
        if unit.get("kind") in {"event", "interaction"}
    ]
    actual = [child for episode in episodes for child in episode["child_unit_ids"]]
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError(
            "Global Event/Interaction to Episode coverage is not exactly one-to-one"
        )
    if any(len(episode.get("source_scene_ids", [])) != 1 for episode in episodes):
        raise ValueError("Episode aggregation produced a cross-scene Episode")


def make_episode_relation_candidates(
    episodes: list[dict[str, Any]],
    *,
    window: int,
    entity_neighbors: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(episodes, key=lambda episode: int(episode["order"]))
    entity_neighbors = entity_neighbors or {}
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(ordered):
        for right_index in range(
            left_index + 1, min(len(ordered), left_index + max(1, window) + 1)
        ):
            right = ordered[right_index]
            distance = right_index - left_index
            signals = _episode_bridge_signals(left, right, entity_neighbors)
            if not _has_direct_episode_bridge(signals):
                continue
            pair_id = f"R{len(candidates) + 1:06d}"
            candidates.append(
                {
                    "pair_id": pair_id,
                    "source_id": left["episode_id"],
                    "target_id": right["episode_id"],
                    "candidate_signals": {
                        "episode_distance": distance,
                        **signals,
                    },
                    "prompt_candidate_signals": _episode_bridge_prompt_signals(
                        {"episode_distance": distance, **signals}, left, right
                    ),
                    "earlier_episode": _episode_prompt_record(left),
                    "later_episode": _episode_prompt_record(right),
                }
            )
    return candidates


def _has_direct_episode_bridge(signals: dict[str, Any]) -> bool:
    return any(
        signals.get(key)
        for key in (
            "shared_participant_entity_ids",
            "shared_location_entity_ids",
            "shared_occasion_entity_ids",
            "shared_interaction_endpoint_entity_ids",
            "participant_endpoint_bridge_entity_ids",
            "explicit_related_unit_ids",
            "shared_state_transition_keys",
        )
    )


def _episode_bridge_signals(
    left: dict[str, Any],
    right: dict[str, Any],
    entity_neighbors: dict[str, set[str]],
) -> dict[str, Any]:
    signals: dict[str, Any] = {}
    for field, signal in (
        ("participant_entity_ids", "shared_participant_entity_ids"),
        ("location_entity_ids", "shared_location_entity_ids"),
        ("occasion_entity_ids", "shared_occasion_entity_ids"),
        (
            "interaction_endpoint_entity_ids",
            "shared_interaction_endpoint_entity_ids",
        ),
    ):
        shared = sorted(set(left.get(field, [])) & set(right.get(field, [])))
        if shared:
            signals[signal] = shared

    left_participants = set(left.get("participant_entity_ids", []))
    right_participants = set(right.get("participant_entity_ids", []))
    left_endpoints = set(left.get("interaction_endpoint_entity_ids", []))
    right_endpoints = set(right.get("interaction_endpoint_entity_ids", []))
    cross_roles = sorted(
        (left_participants & right_endpoints)
        | (right_participants & left_endpoints)
    )
    if cross_roles:
        signals["participant_endpoint_bridge_entity_ids"] = cross_roles

    left_children = set(left.get("child_unit_ids", []))
    right_children = set(right.get("child_unit_ids", []))
    explicit_links = sorted(
        (set(left.get("related_unit_ids", [])) & right_children)
        | (set(right.get("related_unit_ids", [])) & left_children)
    )
    if explicit_links:
        signals["explicit_related_unit_ids"] = explicit_links

    left_anchors = set(left.get("anchor_entity_ids", []))
    right_anchors = set(right.get("anchor_entity_ids", []))
    left_neighbors = set().union(
        *(entity_neighbors.get(entity_id, set()) for entity_id in left_anchors)
    ) if left_anchors else set()
    right_neighbors = set().union(
        *(entity_neighbors.get(entity_id, set()) for entity_id in right_anchors)
    ) if right_anchors else set()
    shared_neighbors = sorted((left_neighbors & right_neighbors) - left_anchors - right_anchors)
    if shared_neighbors:
        signals["shared_kg_neighbor_entity_ids"] = shared_neighbors

    left_state_keys = _episode_state_keys(left)
    right_state_keys = _episode_state_keys(right)
    shared_state_keys = sorted(left_state_keys & right_state_keys)
    if shared_state_keys:
        signals["shared_state_transition_keys"] = shared_state_keys
    return signals


def _episode_bridge_prompt_signals(
    signals: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    names_by_id = {
        clean_text(item.get("entity_id")): clean_text(item.get("canonical_name"))
        for episode in (left, right)
        for item in episode.get("anchor_entities", [])
        if isinstance(item, dict) and clean_text(item.get("entity_id"))
    }
    output: dict[str, Any] = {"episode_distance": signals["episode_distance"]}
    named_signals = {
        "shared_participant_entity_ids": "shared_participants",
        "shared_location_entity_ids": "shared_locations",
        "shared_occasion_entity_ids": "shared_occasions",
        "shared_interaction_endpoint_entity_ids": "shared_interaction_endpoints",
        "participant_endpoint_bridge_entity_ids": "participant_endpoint_bridges",
    }
    for source_key, target_key in named_signals.items():
        if source_key not in signals:
            continue
        names = unique_text(names_by_id.get(clean_text(value)) for value in signals[source_key])
        output[target_key if names else target_key + "_count"] = (
            names if names else len(signals[source_key])
        )
    if signals.get("explicit_related_unit_ids"):
        output["explicit_related_unit_link_count"] = len(
            signals["explicit_related_unit_ids"]
        )
    if signals.get("shared_kg_neighbor_entity_ids"):
        output["shared_kg_neighbor_count"] = len(
            signals["shared_kg_neighbor_entity_ids"]
        )
    if signals.get("shared_state_transition_keys"):
        output["shared_state_transition_keys"] = signals[
            "shared_state_transition_keys"
        ]
    return output


def _episode_state_keys(episode: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for value in episode.get("state_changes", []):
        if isinstance(value, dict):
            subject = clean_text(value.get("subject_entity_id") or value.get("subject"))
            dimension = clean_text(value.get("dimension"))
            key = normalize_name(f"{subject} {dimension}")
        else:
            key = normalize_name(value)
        if key:
            keys.add(key)
    return keys


def _window_pair_count(episode_count: int, window: int) -> int:
    return sum(min(max(0, window), episode_count - index - 1) for index in range(episode_count))


def _candidate_degree(candidates: list[dict[str, Any]]) -> dict[str, int]:
    degree: Counter[str] = Counter()
    for candidate in candidates:
        degree[candidate["source_id"]] += 1
        degree[candidate["target_id"]] += 1
    return dict(sorted(degree.items()))


def _pack_episode_relation_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_items: int,
    token_counter: TokenCounter,
    max_input_tokens: int,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []

    def prompt_tokens(values: list[dict[str, Any]]) -> int:
        prompt_values = [
            {
                "pair_id": item["pair_id"],
                "candidate_signals": item["prompt_candidate_signals"],
                "earlier_episode": item["earlier_episode"],
                "later_episode": item["later_episode"],
            }
            for item in values
        ]
        prompt = EPISODE_RELATION_USER.format(
            candidate_pairs=json.dumps(prompt_values, ensure_ascii=False, indent=2)
        )
        return token_counter.count(EPISODE_RELATION_SYSTEM + prompt)

    for candidate in candidates:
        projected = [*pending, candidate]
        if pending and (
            len(projected) > max_items
            or prompt_tokens(projected) > max_input_tokens
        ):
            batches.append(pending)
            pending = [candidate]
        else:
            pending = projected
        if prompt_tokens(pending) > max_input_tokens:
            raise ValueError(
                f"One episode relation candidate exceeds input budget: {candidate['pair_id']}"
            )
    if pending:
        batches.append(pending)
    return batches


def build_entity_neighbor_index(
    relation_registry: dict[str, Any],
) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for relation in relation_registry.get("canonical_relations", []):
        source = clean_text(relation.get("subject_entity_id"))
        target = clean_text(relation.get("object_entity_id"))
        if not source or not target or source == target:
            continue
        neighbors[source].add(target)
        neighbors[target].add(source)
    return dict(neighbors)


def _episode_prompt_record(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "order": episode["order"],
        "modality": clean_text(episode.get("modality")) or "asserted",
        "name": episode["name"],
        "description": episode["description"],
        "participants": episode.get("participants", []),
        "locations": _prompt_entity_names(episode.get("location_entities", [])),
        "occasions": _prompt_entity_names(episode.get("occasion_entities", [])),
        "anchor_entities": [
            {
                "canonical_name": clean_text(item.get("canonical_name")),
                "roles": list(item.get("roles") or []),
            }
            for item in episode.get("anchor_entities", [])
            if isinstance(item, dict) and clean_text(item.get("canonical_name"))
        ],
        "initial_situation": episode.get("initial_situation", ""),
        "progression_steps": episode["progression_steps"],
        "outcome": episode.get("outcome", ""),
        "state_changes": episode["state_changes"],
        "causal_links": episode["causal_links"],
        "open_threads": episode.get("open_threads", []),
        "closed_threads": episode.get("closed_threads", []),
        "key_evidence": episode.get("key_evidence", []),
        "source_scene_orders": episode.get("source_scene_orders", []),
    }


def _validate_relation_decisions(
    payload: dict[str, Any], pair_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
        raise ValueError("Episode relation output must contain exactly a decisions array")
    decisions: dict[str, dict[str, Any]] = {}
    for raw in payload["decisions"]:
        if not isinstance(raw, dict):
            raise ValueError("Episode relation decision must be an object")
        pair_id = clean_text(raw.get("pair_id"))
        if pair_id not in pair_by_id or pair_id in decisions:
            raise ValueError(f"Unknown or duplicate episode pair id: {pair_id}")
        relation_type = clean_text(raw.get("relation_type")).casefold()
        relation_type = {
            "repeats": "continues",
            "repeated": "continues",
        }.get(relation_type, relation_type)
        if relation_type not in EPISODE_RELATION_TYPES:
            raise ValueError(f"Invalid episode relation type for {pair_id}: {relation_type}")
        confidence = float(raw.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence outside [0,1] for {pair_id}")
        if relation_type != "none" and confidence == 0.0:
            raise ValueError(
                f"Non-none episode relation requires positive confidence: {pair_id}"
            )
        evidence, _ = _normalize_episode_text_list(raw.get("evidence"))
        decisions[pair_id] = {
            "pair_id": pair_id,
            "relation_type": relation_type,
            "description": clean_text(raw.get("description")),
            "evidence": evidence,
            "confidence": confidence,
        }
        if relation_type != "none" and (
            not decisions[pair_id]["description"]
            or not decisions[pair_id]["evidence"]
        ):
            raise ValueError(
                f"Non-none episode relation requires description and evidence: {pair_id}"
            )
    if set(decisions) != set(pair_by_id):
        raise ValueError(
            f"Missing episode relation decisions: {sorted(set(pair_by_id) - set(decisions))}"
        )
    return [decisions[pair_id] for pair_id in pair_by_id]


def break_cycles(
    relations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove the weakest SCC-internal edge until the directed graph is acyclic.

    Adapted from the corrected SABER cycle cleanup in the audited LightRAG/NKW
    implementation. The removed edge has minimum effective weight, not maximum.
    """
    edge_by_pair = {
        (relation["source_id"], relation["target_id"]): dict(relation)
        for relation in relations
    }
    removed: list[dict[str, Any]] = []
    while True:
        components = [component for component in _tarjan_scc(edge_by_pair) if len(component) > 1]
        if not components:
            break
        candidates: list[tuple[tuple[str, str], dict[str, Any]]] = []
        for component in components:
            nodes = set(component)
            candidates.extend(
                (pair, relation)
                for pair, relation in edge_by_pair.items()
                if pair[0] in nodes and pair[1] in nodes
            )
        if not candidates:
            raise RuntimeError("Cycle detected but no SCC-internal edge is selectable")
        pair, relation = min(
            candidates,
            key=lambda item: (
                float(item[1].get("weight", 0.0)),
                float(item[1].get("confidence", 0.0)),
                item[1].get("relation_type", ""),
                item[0],
            ),
        )
        removed.append(edge_by_pair.pop(pair))
    kept = sorted(
        edge_by_pair.values(),
        key=lambda relation: (relation["source_id"], relation["target_id"]),
    )
    return kept, removed


def _tarjan_scc(
    edge_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> list[list[str]]:
    nodes = sorted({node for pair in edge_by_pair for node in pair})
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edge_by_pair:
        outgoing[source].append(target)
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in outgoing.get(node, []):
            if target not in indices:
                visit(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], indices[target])
        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(sorted(component))

    for node in nodes:
        if node not in indices:
            visit(node)
    return components


def weighted_primary_path_cover(
    episodes: list[dict[str, Any]], relations: list[dict[str, Any]]
) -> list[list[str]]:
    """Build deterministic, non-overlapping primary paths that cover every episode."""
    episode_by_id = {episode["episode_id"]: episode for episode in episodes}
    order = {episode["episode_id"]: int(episode["order"]) for episode in episodes}
    outgoing: dict[str, list[tuple[str, float]]] = defaultdict(list)
    incoming: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        source = relation["source_id"]
        target = relation["target_id"]
        if source in episode_by_id and target in episode_by_id and source != target:
            outgoing[source].append((target, float(relation.get("weight", 0.0))))
            incoming[target].add(source)
    for source in outgoing:
        outgoing[source].sort(key=lambda item: (-item[1], order[item[0]], item[0]))

    uncovered = set(episode_by_id)
    paths: list[list[str]] = []
    while uncovered:
        starts = [
            episode_id
            for episode_id in uncovered
            if not (incoming.get(episode_id, set()) & uncovered)
        ]
        if not starts:
            raise ValueError("Cannot create path cover from a cyclic episode graph")
        current = min(starts, key=lambda episode_id: (order[episode_id], episode_id))
        path = [current]
        uncovered.remove(current)
        while True:
            children = [item for item in outgoing.get(current, []) if item[0] in uncovered]
            if not children:
                break
            current = children[0][0]
            path.append(current)
            uncovered.remove(current)
        paths.append(path)
    return paths


def build_storyline_candidates(
    episodes: list[dict[str, Any]], relations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ordered = sorted(episodes, key=lambda item: int(item["order"]))
    if not ordered:
        return []
    episode_by_id = {episode["episode_id"]: episode for episode in ordered}
    order = {episode_id: int(episode["order"]) for episode_id, episode in episode_by_id.items()}
    undirected: dict[str, set[str]] = defaultdict(set)
    valid_relations: list[dict[str, Any]] = []
    for relation in relations:
        source = clean_text(relation.get("source_id"))
        target = clean_text(relation.get("target_id"))
        if source not in episode_by_id or target not in episode_by_id or source == target:
            continue
        if order[source] >= order[target]:
            raise ValueError("Storyline candidates require chronological DAG relations")
        valid_relations.append(relation)
        undirected[source].add(target)
        undirected[target].add(source)

    seen: set[str] = set()
    components: list[list[str]] = []
    for episode in ordered:
        episode_id = episode["episode_id"]
        if episode_id in seen:
            continue
        pending = [episode_id]
        seen.add(episode_id)
        component: list[str] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in sorted(undirected.get(current, set()), key=lambda item: order[item]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                pending.append(neighbor)
        components.append(sorted(component, key=lambda item: order[item]))

    candidates: list[dict[str, Any]] = []
    for component in components:
        if len(component) < 2:
            continue
        source_scene_ids = unique_text(
            scene_id
            for episode_id in component
            for scene_id in episode_by_id[episode_id].get("source_scene_ids", [])
        )
        if len(source_scene_ids) < 2:
            continue
        members = set(component)
        component_relations = [
            relation
            for relation in valid_relations
            if relation["source_id"] in members and relation["target_id"] in members
        ]
        candidates.append(
            {
                "component_id": f"COMP-{len(candidates) + 1:03d}",
                "episode_ids": component,
                "supporting_relation_ids": [
                    relation["relation_id"] for relation in component_relations
                ],
                "source_scene_ids": source_scene_ids,
            }
        )
    return candidates


def _build_chronological_backbone(
    movie_id: str, episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(episodes, key=lambda item: int(item["order"]))
    episode_ids = [episode["episode_id"] for episode in ordered]
    return {
        "storyline_id": stable_id("storyline-backbone", movie_id, *episode_ids),
        "name": "Chronological Backbone",
        "description": "Deterministic chronological index of all screenplay episodes.",
        "focus_type": "chronological_backbone",
        "focus_entity_ids": [],
        "focus_entities": [],
        "component_id": "",
        "initial_state": "",
        "ordered_transitions": [],
        "turning_point_episode_ids": [],
        "resolution_or_current_state": "",
        "status": "index",
        "supporting_relation_ids": [],
        "participant_entities": _unique_entity_refs(
            entity
            for episode in ordered
            for entity in episode.get("participant_entities", [])
        ),
        "child_episode_ids": episode_ids,
        "source_scene_ids": unique_text(
            scene_id
            for episode in ordered
            for scene_id in episode.get("source_scene_ids", [])
        ),
    }


def _storyline_component_prompt_assets(
    candidate: dict[str, Any],
    *,
    episode_by_id: dict[str, dict[str, Any]],
    relation_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    component_episodes = [episode_by_id[item] for item in candidate["episode_ids"]]
    component_relations = [
        relation_by_id[item] for item in candidate["supporting_relation_ids"]
    ]
    component_entities = _unique_entity_refs(
        entity
        for episode in component_episodes
        for entity in episode.get("participant_entities", [])
    )
    episode_local = {
        episode["episode_id"]: f"E{index:03d}"
        for index, episode in enumerate(component_episodes, start=1)
    }
    relation_local = {
        relation["relation_id"]: f"R{index:03d}"
        for index, relation in enumerate(component_relations, start=1)
    }
    entity_local = {
        entity["entity_id"]: f"P{index:03d}"
        for index, entity in enumerate(component_entities, start=1)
    }
    episode_by_local_id = {
        episode_local[episode["episode_id"]]: episode for episode in component_episodes
    }
    relation_by_local_id = {
        relation_local[relation["relation_id"]]: relation
        for relation in component_relations
    }
    entity_by_local_id = {
        entity_local[entity["entity_id"]]: entity for entity in component_entities
    }
    return {
        "prompt_entities": [
            {
                "local_entity_id": entity_local[entity["entity_id"]],
                "canonical_name": entity["canonical_name"],
            }
            for entity in component_entities
        ],
        "prompt_episodes": [
            {
                "local_episode_id": episode_local[episode["episode_id"]],
                "order": episode["order"],
                "source_scene_orders": episode.get("source_scene_orders")
                or [episode["order"]],
                "name": episode["name"],
                "participant_entity_ids": [
                    entity_local[entity_id]
                    for entity_id in episode.get("participant_entity_ids", [])
                    if entity_id in entity_local
                ],
                "initial_situation": episode.get("initial_situation", ""),
                "progression_steps": episode.get("progression_steps", []),
                "outcome": episode.get("outcome", ""),
                "state_changes": episode.get("state_changes", []),
                "causal_links": episode.get("causal_links", []),
                "open_threads": episode.get("open_threads", []),
                "closed_threads": episode.get("closed_threads", []),
                "key_evidence": episode.get("key_evidence", []),
            }
            for episode in component_episodes
        ],
        "prompt_relations": [
            {
                "local_relation_id": relation_local[relation["relation_id"]],
                "earlier_episode_id": episode_local[relation["source_id"]],
                "later_episode_id": episode_local[relation["target_id"]],
                "relation_type": relation["relation_type"],
                "confidence": relation["confidence"],
            }
            for relation in component_relations
        ],
        "episode_by_local_id": episode_by_local_id,
        "relation_by_local_id": relation_by_local_id,
        "entity_by_local_id": entity_by_local_id,
        "episode_id_by_local": {
            local_id: episode["episode_id"]
            for local_id, episode in episode_by_local_id.items()
        },
        "relation_id_by_local": {
            local_id: relation["relation_id"]
            for local_id, relation in relation_by_local_id.items()
        },
    }


def _storyline_component_input_sha256(
    *,
    language: str,
    candidate: dict[str, Any],
    assets: dict[str, Any],
) -> str:
    return sha256_json(
        {
            "schema_version": "stage_storyline_component_input_v2",
            "system_prompt": STORYLINE_SYSTEM,
            "user_prompt": STORYLINE_USER,
            "json_serialization": "compact_v1",
            "language": language,
            "component_id": candidate["component_id"],
            "available_entities": assets["prompt_entities"],
            "episodes": assets["prompt_episodes"],
            "supporting_relations": assets["prompt_relations"],
        }
    )


def _storyline_prompt_token_count(
    *,
    language: str,
    candidate: dict[str, Any],
    assets: dict[str, Any],
    token_counter: TokenCounter,
) -> int:
    user_prompt = STORYLINE_USER.format(
        language=language,
        component_id=candidate["component_id"],
        available_entities=_storyline_json(assets["prompt_entities"]),
        episodes=_storyline_json(assets["prompt_episodes"]),
        supporting_relations=_storyline_json(assets["prompt_relations"]),
    )
    return token_counter.count(STORYLINE_SYSTEM + user_prompt)


def _storyline_json(value: Any) -> str:
    """Serialize large storyline prompt blocks without indentation overhead.

    This preserves every field and value while leaving more of the fixed 24k
    context window for the model. The serialization policy is included in the
    component input hash so checkpoints cannot be reused across prompt formats.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ": "))


def _split_storyline_candidates_for_budget(
    candidates: list[dict[str, Any]],
    *,
    episode_by_id: dict[str, dict[str, Any]],
    relation_by_id: dict[str, dict[str, Any]],
    language: str,
    token_counter: TokenCounter,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    """Partition oversized storyline components without truncating evidence.

    Components are packed in screenplay order. A one-episode overlap is retained at
    a boundary when the preceding chunk has more than one episode, so a relation that
    crosses the boundary remains visible in the following chunk. The split is driven
    by the configured token budget, not by a fixed episode count.
    """
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        episode_ids = list(candidate["episode_ids"])
        if not episode_ids:
            continue

        def make_part(part_ids: list[str], part_index: int | None = None) -> dict[str, Any]:
            members = set(part_ids)
            relation_ids = [
                relation_id
                for relation_id in candidate["supporting_relation_ids"]
                if relation_id in relation_by_id
                and relation_by_id[relation_id]["source_id"] in members
                and relation_by_id[relation_id]["target_id"] in members
            ]
            payload = {
                **candidate,
                "episode_ids": part_ids,
                "supporting_relation_ids": relation_ids,
                "source_scene_ids": unique_text(
                    scene_id
                    for episode_id in part_ids
                    for scene_id in episode_by_id[episode_id].get("source_scene_ids", [])
                ),
            }
            if part_index is not None:
                payload["component_id"] = (
                    f"{candidate['component_id']}.part-{part_index:03d}"
                )
                payload["split_from_component_id"] = candidate["component_id"]
            return payload

        whole = make_part(episode_ids)
        whole_assets = _storyline_component_prompt_assets(
            whole, episode_by_id=episode_by_id, relation_by_id=relation_by_id
        )
        whole_tokens = _storyline_prompt_token_count(
            language=language,
            candidate=whole,
            assets=whole_assets,
            token_counter=token_counter,
        )
        if whole_tokens <= max_input_tokens:
            output.append(whole)
            continue

        parts: list[dict[str, Any]] = []
        start = 0
        part_index = 1
        while start < len(episode_ids):
            best_end: int | None = None
            for end in range(start + 1, len(episode_ids) + 1):
                trial = make_part(episode_ids[start:end], part_index)
                assets = _storyline_component_prompt_assets(
                    trial, episode_by_id=episode_by_id, relation_by_id=relation_by_id
                )
                tokens = _storyline_prompt_token_count(
                    language=language,
                    candidate=trial,
                    assets=assets,
                    token_counter=token_counter,
                )
                if tokens > max_input_tokens:
                    break
                best_end = end
            if best_end is None:
                episode_id = episode_ids[start]
                raise ValueError(
                    "A single storyline episode exceeds the configured input budget: "
                    f"{candidate['component_id']} episode={episode_id}"
                )
            parts.append(make_part(episode_ids[start:best_end], part_index))
            if best_end >= len(episode_ids):
                break
            # Retain the last episode as context for the next component when it fits.
            start = best_end - 1 if best_end - start > 1 else best_end
            part_index += 1
        output.extend(parts)
    return output


def _normalize_storyline_component_payload(
    payload: Any,
    *,
    episode_by_local_id: dict[str, dict[str, Any]],
    relation_by_local_id: dict[str, dict[str, Any]],
    entity_by_local_id: dict[str, dict[str, str]] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("storylines"), list):
        return payload, []
    kept: list[Any] = []
    corrections: list[dict[str, Any]] = []
    global_to_local_episode = {
        episode["episode_id"]: local_id
        for local_id, episode in episode_by_local_id.items()
    }
    episode_order = {
        local_id: int(episode["order"])
        for local_id, episode in episode_by_local_id.items()
    }

    def normalize_ids(
        value: Any,
        allowed: dict[str, Any],
        label: str,
    ) -> list[str]:
        values, id_corrections = _normalize_local_reference_list(value, allowed, label)
        corrections.extend(id_corrections)
        return values

    for index, raw in enumerate(payload["storylines"]):
        if not isinstance(raw, dict):
            kept.append(raw)
            continue
        episode_ids = normalize_ids(
            raw.get("episode_ids"), episode_by_local_id, "episode"
        )
        if not episode_ids:
            kept.append(raw)
            continue
        unknown_episode_ids = [
            episode_id
            for episode_id in episode_ids
            if episode_id not in episode_by_local_id
        ]
        if unknown_episode_ids:
            corrections.append(
                {
                    "action": "drop_storyline_with_unknown_episode",
                    "storyline_index": index,
                    "name": clean_text(raw.get("name")),
                    "episode_ids": unknown_episode_ids,
                }
            )
            continue
        # A model can omit an episode from the storyline-level membership while
        # explicitly using it in a transition and naming an internal relation
        # that connects it to a selected member. This is a local schema mismatch,
        # not a new semantic inference, so recover only that exact bounded case.
        transition_episode_ids: set[str] = set()
        declared_relation_ids = normalize_ids(
            raw.get("supporting_relation_ids"),
            relation_by_local_id,
            "supporting relation",
        )
        for transition in raw.get("ordered_transitions", []):
            if not isinstance(transition, dict):
                continue
            for field in ("catalyst_episode_ids", "evidence_episode_ids"):
                values = normalize_ids(
                    transition.get(field), episode_by_local_id, field
                )
                transition_episode_ids.update(values)
            values = normalize_ids(
                transition.get("supporting_relation_ids"),
                relation_by_local_id,
                "transition supporting relation",
            )
            declared_relation_ids.extend(values)
        selected_episode_ids = set(episode_ids)
        added_transition_members: list[str] = []
        initial_source_scene_ids = {
            scene_id
            for episode_id in episode_ids
            for scene_id in episode_by_local_id[episode_id].get("source_scene_ids", [])
        }
        changed = len(episode_ids) >= 2 and len(initial_source_scene_ids) >= 2
        while changed:
            changed = False
            for relation_id in unique_text(declared_relation_ids):
                relation = relation_by_local_id.get(relation_id)
                if relation is None:
                    continue
                endpoints = {
                    global_to_local_episode[relation["source_id"]],
                    global_to_local_episode[relation["target_id"]],
                }
                missing = endpoints - selected_episode_ids
                if (
                    len(missing) == 1
                    and endpoints & selected_episode_ids
                    and missing <= transition_episode_ids
                ):
                    episode_id = next(iter(missing))
                    selected_episode_ids.add(episode_id)
                    added_transition_members.append(episode_id)
                    changed = True
        if added_transition_members:
            episode_ids = sorted(selected_episode_ids, key=lambda item: episode_order[item])
            corrections.append(
                {
                    "action": "add_relation_bound_transition_episode_to_storyline",
                    "storyline_index": index,
                    "name": clean_text(raw.get("name")),
                    "episode_ids": unique_text(added_transition_members),
                }
            )
        source_scene_ids = {
            scene_id
            for episode_id in episode_ids
            for scene_id in episode_by_local_id[episode_id].get("source_scene_ids", [])
        }
        if len(source_scene_ids) < 2:
            corrections.append(
                {
                    "action": "drop_single_scene_storyline",
                    "storyline_index": index,
                    "name": clean_text(raw.get("name")),
                    "episode_ids": episode_ids,
                }
            )
            continue
        if entity_by_local_id is not None:
            focus_entity_ids = normalize_ids(
                raw.get("focus_entity_ids"), entity_by_local_id, "focus entity"
            )
            invalid_focus_ids = [
                focus_id
                for focus_id in focus_entity_ids
                if focus_id not in entity_by_local_id
                or sum(
                    entity_by_local_id[focus_id]["entity_id"]
                    in episode_by_local_id[episode_id].get(
                        "participant_entity_ids", []
                    )
                    for episode_id in episode_ids
                )
                < 2
            ]
            if invalid_focus_ids:
                corrections.append(
                    {
                        "action": "drop_storyline_with_unsupported_focus",
                        "storyline_index": index,
                        "name": clean_text(raw.get("name")),
                        "focus_entity_ids": invalid_focus_ids,
                    }
                )
                continue
        chronological_episode_ids = sorted(
            episode_ids,
            key=lambda item: episode_order[item],
        )
        if chronological_episode_ids != episode_ids:
            corrections.append(
                {
                    "action": "sort_storyline_episode_ids_chronologically",
                    "storyline_index": index,
                    "name": clean_text(raw.get("name")),
                    "from_episode_ids": list(episode_ids),
                    "to_episode_ids": list(chronological_episode_ids),
                }
            )
            episode_ids = chronological_episode_ids
        normalized_raw = dict(raw)
        normalized_raw["episode_ids"] = episode_ids
        relation_ids = normalize_ids(
            raw.get("supporting_relation_ids"),
            relation_by_local_id,
            "supporting relation",
        )
        kept_relation_ids: list[str] = []
        dropped_relation_ids: list[str] = []
        for relation_id in relation_ids:
            relation = relation_by_local_id.get(relation_id)
            if relation is None:
                kept_relation_ids.append(relation_id)
                continue
            endpoints = {
                global_to_local_episode[relation["source_id"]],
                global_to_local_episode[relation["target_id"]],
            }
            if endpoints <= set(episode_ids):
                kept_relation_ids.append(relation_id)
            else:
                dropped_relation_ids.append(relation_id)
        available_relation_ids = sorted(
            relation_id
            for relation_id, relation in relation_by_local_id.items()
            if {
                global_to_local_episode[relation["source_id"]],
                global_to_local_episode[relation["target_id"]],
            }
            <= set(episode_ids)
        )
        if not kept_relation_ids and available_relation_ids:
            kept_relation_ids.append(available_relation_ids[0])
            corrections.append(
                {
                    "action": "add_storyline_internal_relation_support",
                    "storyline_index": index,
                    "relation_id": available_relation_ids[0],
                }
            )
        selected_episode_ids = set(episode_ids)
        if kept_relation_ids and not _local_storyline_is_connected(
            episode_ids,
            kept_relation_ids,
            relation_by_local_id=relation_by_local_id,
            global_to_local_episode=global_to_local_episode,
        ):
            added_connecting_relations: list[str] = []
            for relation_id in available_relation_ids:
                if relation_id in kept_relation_ids:
                    continue
                kept_relation_ids.append(relation_id)
                added_connecting_relations.append(relation_id)
                if _local_storyline_is_connected(
                    episode_ids,
                    kept_relation_ids,
                    relation_by_local_id=relation_by_local_id,
                    global_to_local_episode=global_to_local_episode,
                ):
                    break
            if added_connecting_relations:
                corrections.append(
                    {
                        "action": "add_storyline_connecting_relation_support",
                        "storyline_index": index,
                        "relation_ids": added_connecting_relations,
                    }
                )
        if not kept_relation_ids or not _local_storyline_is_connected(
            episode_ids,
            kept_relation_ids,
            relation_by_local_id=relation_by_local_id,
            global_to_local_episode=global_to_local_episode,
        ):
            corrections.append(
                {
                    "action": "drop_disconnected_storyline",
                    "storyline_index": index,
                    "name": clean_text(raw.get("name")),
                    "episode_ids": sorted(
                        selected_episode_ids,
                        key=lambda item: episode_order[item],
                    ),
                }
            )
            continue
        normalized_raw["supporting_relation_ids"] = kept_relation_ids
        normalized_raw["turning_point_episode_ids"] = normalize_ids(
            raw.get("turning_point_episode_ids"),
            episode_by_local_id,
            "turning-point episode",
        )
        if dropped_relation_ids:
            corrections.append(
                {
                    "action": "drop_relation_outside_storyline",
                    "storyline_index": index,
                    "relation_ids": dropped_relation_ids,
                }
            )
        transitions = raw.get("ordered_transitions")
        if isinstance(transitions, list):
            normalized_transitions: list[Any] = []
            for transition_index, transition in enumerate(transitions):
                if not isinstance(transition, dict):
                    normalized_transitions.append(transition)
                    continue
                normalized_transition = dict(transition)
                transition_relation_ids = normalize_ids(
                    transition.get("supporting_relation_ids"),
                    relation_by_local_id,
                    "transition supporting relation",
                )
                kept_transition_relation_ids: list[str] = []
                dropped_transition_relation_ids: list[str] = []
                required_evidence_ids: list[str] = []
                for relation_id in transition_relation_ids:
                    relation = relation_by_local_id.get(relation_id)
                    if relation is None:
                        kept_transition_relation_ids.append(relation_id)
                        continue
                    if relation_id not in kept_relation_ids:
                        dropped_transition_relation_ids.append(relation_id)
                        continue
                    kept_transition_relation_ids.append(relation_id)
                    required_evidence_ids.extend(
                        [
                            global_to_local_episode[relation["source_id"]],
                            global_to_local_episode[relation["target_id"]],
                        ]
                    )
                normalized_transition["supporting_relation_ids"] = (
                    kept_transition_relation_ids
                )
                evidence_ids = normalize_ids(
                    transition.get("evidence_episode_ids"),
                    episode_by_local_id,
                    "transition evidence episode",
                )
                catalyst_ids = normalize_ids(
                    transition.get("catalyst_episode_ids"),
                    episode_by_local_id,
                    "transition catalyst episode",
                )
                out_of_scope_evidence_ids = [
                    item for item in evidence_ids if item not in selected_episode_ids
                ]
                if out_of_scope_evidence_ids:
                    evidence_ids = [
                        item for item in evidence_ids if item in selected_episode_ids
                    ]
                    corrections.append(
                        {
                            "action": "drop_transition_evidence_outside_storyline",
                            "storyline_index": index,
                            "transition_index": transition_index,
                            "episode_ids": out_of_scope_evidence_ids,
                        }
                    )
                if not kept_transition_relation_ids:
                    transition_scope = set([*evidence_ids, *catalyst_ids])
                    scoped_relation_ids = [
                        relation_id
                        for relation_id in kept_relation_ids
                        if {
                            global_to_local_episode[
                                relation_by_local_id[relation_id]["source_id"]
                            ],
                            global_to_local_episode[
                                relation_by_local_id[relation_id]["target_id"]
                            ],
                        }
                        <= transition_scope
                    ]
                    fallback_relation_ids = scoped_relation_ids or kept_relation_ids
                    if fallback_relation_ids:
                        relation_id = fallback_relation_ids[0]
                        kept_transition_relation_ids.append(relation_id)
                        relation = relation_by_local_id[relation_id]
                        required_evidence_ids.extend(
                            [
                                global_to_local_episode[relation["source_id"]],
                                global_to_local_episode[relation["target_id"]],
                            ]
                        )
                        corrections.append(
                            {
                                "action": "add_transition_internal_relation_support",
                                "storyline_index": index,
                                "transition_index": transition_index,
                                "relation_id": relation_id,
                            }
                        )
                normalized_transition["supporting_relation_ids"] = (
                    kept_transition_relation_ids
                )
                normalized_transition["catalyst_episode_ids"] = catalyst_ids
                completed_evidence_ids = unique_text(
                    [*evidence_ids, *required_evidence_ids]
                )
                known_evidence_ids = [
                    item for item in completed_evidence_ids if item in episode_order
                ]
                unknown_evidence_ids = [
                    item for item in completed_evidence_ids if item not in episode_order
                ]
                normalized_transition["evidence_episode_ids"] = [
                    *sorted(known_evidence_ids, key=lambda item: episode_order[item]),
                    *unknown_evidence_ids,
                ]
                if dropped_transition_relation_ids:
                    corrections.append(
                        {
                            "action": "drop_transition_relation_outside_storyline",
                            "storyline_index": index,
                            "transition_index": transition_index,
                            "relation_ids": dropped_transition_relation_ids,
                        }
                    )
                added_evidence_ids = [
                    item for item in required_evidence_ids if item not in evidence_ids
                ]
                if added_evidence_ids:
                    corrections.append(
                        {
                            "action": "add_transition_relation_endpoints_to_evidence",
                            "storyline_index": index,
                            "transition_index": transition_index,
                            "episode_ids": unique_text(added_evidence_ids),
                        }
                    )
                normalized_transitions.append(normalized_transition)
            _ensure_cross_scene_transition_support(
                normalized_transitions,
                storyline_index=index,
                storyline_relation_ids=kept_relation_ids,
                episode_by_local_id=episode_by_local_id,
                relation_by_local_id=relation_by_local_id,
                global_to_local_episode=global_to_local_episode,
                episode_order=episode_order,
                corrections=corrections,
            )
            normalized_raw["ordered_transitions"] = normalized_transitions
        kept.append(normalized_raw)
    return {**payload, "storylines": kept}, corrections


def _ensure_cross_scene_transition_support(
    transitions: list[Any],
    *,
    storyline_index: int,
    storyline_relation_ids: list[str],
    episode_by_local_id: dict[str, dict[str, Any]],
    relation_by_local_id: dict[str, dict[str, Any]],
    global_to_local_episode: dict[str, str],
    episode_order: dict[str, int],
    corrections: list[dict[str, Any]],
) -> None:
    valid_transitions = [item for item in transitions if isinstance(item, dict)]
    if not valid_transitions:
        return

    def relation_endpoints(relation_id: str) -> tuple[str, str]:
        relation = relation_by_local_id[relation_id]
        return (
            global_to_local_episode[relation["source_id"]],
            global_to_local_episode[relation["target_id"]],
        )

    def spans_scenes(episode_ids: list[str]) -> bool:
        scenes = {
            scene_id
            for episode_id in episode_ids
            for scene_id in episode_by_local_id[episode_id].get(
                "source_scene_ids", []
            )
        }
        return len(scenes) >= 2

    if any(
        spans_scenes(
            [
                episode_id
                for episode_id in transition.get("evidence_episode_ids", [])
                if episode_id in episode_by_local_id
            ]
        )
        for transition in valid_transitions
    ):
        return
    cross_scene_relations = [
        relation_id
        for relation_id in storyline_relation_ids
        if relation_id in relation_by_local_id
        and spans_scenes(list(relation_endpoints(relation_id)))
    ]
    if not cross_scene_relations:
        return
    best: tuple[int, int, int, str] | None = None
    for transition_index, transition in enumerate(valid_transitions):
        transition_scope = set(
            [
                *transition.get("catalyst_episode_ids", []),
                *transition.get("evidence_episode_ids", []),
            ]
        )
        for relation_id in cross_scene_relations:
            endpoints = relation_endpoints(relation_id)
            score = len(transition_scope.intersection(endpoints))
            candidate = (
                score,
                -min(episode_order[item] for item in endpoints),
                -transition_index,
                relation_id,
            )
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return
    relation_id = best[3]
    endpoints = relation_endpoints(relation_id)
    transition_index = -best[2]
    transition = valid_transitions[transition_index]
    transition["supporting_relation_ids"] = unique_text(
        [*transition.get("supporting_relation_ids", []), relation_id]
    )
    transition["evidence_episode_ids"] = sorted(
        unique_text([*transition.get("evidence_episode_ids", []), *endpoints]),
        key=lambda item: episode_order[item],
    )
    corrections.append(
        {
            "action": "add_cross_scene_transition_relation_support",
            "storyline_index": storyline_index,
            "transition_index": transition_index,
            "relation_id": relation_id,
            "episode_ids": list(endpoints),
            "selection_policy": "maximum_transition_endpoint_overlap_then_chronology",
        }
    )


def _validate_storyline_component_payload(
    payload: dict[str, Any],
    *,
    episode_by_local_id: dict[str, dict[str, Any]],
    relation_by_local_id: dict[str, dict[str, Any]],
    entity_by_local_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if set(payload) != {"storylines"} or not isinstance(payload["storylines"], list):
        raise ValueError("Storyline extraction must return exactly a storylines array")
    expected_fields = {
        "name",
        "description",
        "focus_type",
        "focus_entity_ids",
        "episode_ids",
        "supporting_relation_ids",
        "initial_state",
        "ordered_transitions",
        "turning_point_episode_ids",
        "resolution_or_current_state",
        "status",
    }
    transition_fields = {
        "dimension",
        "before_state",
        "catalyst_episode_ids",
        "after_state",
        "evidence_episode_ids",
        "supporting_relation_ids",
    }
    episode_order = {
        local_id: int(episode["order"])
        for local_id, episode in episode_by_local_id.items()
    }
    global_to_local_episode = {
        episode["episode_id"]: local_id
        for local_id, episode in episode_by_local_id.items()
    }
    output: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...]] = set()
    for raw in payload["storylines"]:
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError("Each storyline must match the exact component schema")
        name = clean_text(raw["name"])
        description = clean_text(raw["description"])
        focus_type = clean_text(raw["focus_type"]).casefold()
        status = clean_text(raw["status"]).casefold()
        initial_state = clean_text(raw["initial_state"])
        final_state = clean_text(raw["resolution_or_current_state"])
        if not name or not description or not initial_state or not final_state:
            raise ValueError("Storyline summary and boundary states must be non-empty")
        if focus_type not in STORYLINE_FOCUS_TYPES:
            raise ValueError(f"Invalid storyline focus type: {focus_type}")
        if status not in STORYLINE_STATUSES:
            raise ValueError(f"Invalid storyline status: {status}")
        focus_entity_ids = _validated_local_ids(
            raw["focus_entity_ids"], entity_by_local_id, "focus entity"
        )
        if focus_type == "relationship_development":
            if len(focus_entity_ids) != 2:
                raise ValueError("Relationship storyline requires exactly two focus entities")
        elif not focus_entity_ids:
            raise ValueError("Evolution storyline requires at least one focus entity")
        episode_ids = _validated_local_ids(
            raw["episode_ids"], episode_by_local_id, "episode"
        )
        if len(episode_ids) < 2:
            raise ValueError("Evolution storyline requires at least two episodes")
        if episode_ids != sorted(episode_ids, key=lambda item: episode_order[item]):
            raise ValueError("Storyline episodes must be in chronological order")
        source_scene_ids = {
            scene_id
            for local_id in episode_ids
            for scene_id in episode_by_local_id[local_id].get("source_scene_ids", [])
        }
        if len(source_scene_ids) < 2:
            raise ValueError("Evolution storyline must span at least two scenes")
        supporting_relation_ids = _validated_local_ids(
            raw["supporting_relation_ids"],
            relation_by_local_id,
            "supporting relation",
        )
        if not supporting_relation_ids:
            raise ValueError("Evolution storyline requires supporting relations")
        selected_episode_set = set(episode_ids)
        for relation_id in supporting_relation_ids:
            relation = relation_by_local_id[relation_id]
            endpoints = {
                global_to_local_episode[relation["source_id"]],
                global_to_local_episode[relation["target_id"]],
            }
            if not endpoints <= selected_episode_set:
                raise ValueError("Storyline relation endpoints must be selected episodes")
        if not _local_storyline_is_connected(
            episode_ids,
            supporting_relation_ids,
            relation_by_local_id=relation_by_local_id,
            global_to_local_episode=global_to_local_episode,
        ):
            raise ValueError("Selected storyline episodes must be relation-connected")
        for focus_entity_id in focus_entity_ids:
            global_entity_id = entity_by_local_id[focus_entity_id]["entity_id"]
            occurrence_count = sum(
                global_entity_id
                in episode_by_local_id[local_id].get("participant_entity_ids", [])
                for local_id in episode_ids
            )
            if occurrence_count < 2:
                raise ValueError("Each focus entity must occur in at least two selected episodes")
        if not isinstance(raw["ordered_transitions"], list) or not raw[
            "ordered_transitions"
        ]:
            raise ValueError("Evolution storyline requires ordered transitions")
        transitions: list[dict[str, Any]] = []
        has_cross_scene_transition = False
        for transition in raw["ordered_transitions"]:
            if not isinstance(transition, dict) or set(transition) != transition_fields:
                raise ValueError("Each transition must match the exact transition schema")
            dimension = clean_text(transition["dimension"])
            before_state = clean_text(transition["before_state"])
            after_state = clean_text(transition["after_state"])
            if not dimension or not before_state or not after_state:
                raise ValueError("Transition dimension and before/after states are required")
            if before_state.casefold() == after_state.casefold():
                raise ValueError("Transition before and after states must differ")
            catalyst_ids = _validated_local_ids(
                transition["catalyst_episode_ids"],
                episode_by_local_id,
                "transition catalyst episode",
            )
            evidence_ids = _validated_local_ids(
                transition["evidence_episode_ids"],
                episode_by_local_id,
                "transition evidence episode",
            )
            transition_relation_ids = _validated_local_ids(
                transition["supporting_relation_ids"],
                relation_by_local_id,
                "transition supporting relation",
            )
            if not catalyst_ids or len(evidence_ids) < 2 or not transition_relation_ids:
                raise ValueError(
                    "Each transition requires catalyst, two evidence episodes, and relation support"
                )
            if not set(catalyst_ids + evidence_ids) <= selected_episode_set:
                raise ValueError("Transition episodes must belong to the storyline")
            if not set(transition_relation_ids) <= set(supporting_relation_ids):
                raise ValueError("Transition relations must belong to the storyline")
            evidence_set = set(evidence_ids)
            for relation_id in transition_relation_ids:
                relation = relation_by_local_id[relation_id]
                endpoints = {
                    global_to_local_episode[relation["source_id"]],
                    global_to_local_episode[relation["target_id"]],
                }
                if not endpoints <= evidence_set:
                    raise ValueError(
                        "Transition evidence must include its relation endpoints"
                    )
            evidence_scenes = {
                scene_id
                for local_id in evidence_ids
                for scene_id in episode_by_local_id[local_id].get("source_scene_ids", [])
            }
            has_cross_scene_transition |= len(evidence_scenes) >= 2
            transitions.append(
                {
                    "dimension": dimension,
                    "before_state": before_state,
                    "catalyst_episode_ids": catalyst_ids,
                    "after_state": after_state,
                    "evidence_episode_ids": evidence_ids,
                    "supporting_relation_ids": transition_relation_ids,
                }
            )
        if not has_cross_scene_transition:
            raise ValueError("At least one transition must have cross-scene evidence")
        turning_point_ids = _validated_local_ids(
            raw["turning_point_episode_ids"],
            episode_by_local_id,
            "turning-point episode",
            allow_empty=True,
        )
        if not set(turning_point_ids) <= selected_episode_set:
            raise ValueError("Turning points must belong to the storyline")
        signature = (
            focus_type,
            tuple(sorted(focus_entity_ids)),
            tuple(episode_ids),
        )
        if signature in signatures:
            raise ValueError("Duplicate storyline in one component response")
        signatures.add(signature)
        output.append(
            {
                "name": name,
                "description": description,
                "focus_type": focus_type,
                "focus_entity_ids": focus_entity_ids,
                "episode_ids": episode_ids,
                "supporting_relation_ids": supporting_relation_ids,
                "initial_state": initial_state,
                "ordered_transitions": transitions,
                "turning_point_episode_ids": turning_point_ids,
                "resolution_or_current_state": final_state,
                "status": status,
            }
        )
    return output


def _validated_local_ids(
    value: Any,
    allowed: dict[str, Any],
    label: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    raw_values = list(value) if isinstance(value, (list, tuple, set)) else [value]
    cleaned_values = [clean_text(item) for item in raw_values if clean_text(item)]
    if len(cleaned_values) != len(set(cleaned_values)):
        raise ValueError(f"Duplicate {label} id")
    values, _ = _normalize_episode_text_list(value)
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown {label} id: {unknown[0]}")
    if not values and not allow_empty:
        raise ValueError(f"At least one {label} id is required")
    return values


def _local_storyline_is_connected(
    episode_ids: list[str],
    relation_ids: list[str],
    *,
    relation_by_local_id: dict[str, dict[str, Any]],
    global_to_local_episode: dict[str, str],
) -> bool:
    selected = set(episode_ids)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for relation_id in relation_ids:
        relation = relation_by_local_id[relation_id]
        source = global_to_local_episode[relation["source_id"]]
        target = global_to_local_episode[relation["target_id"]]
        if source in selected and target in selected:
            adjacency[source].add(target)
            adjacency[target].add(source)
    visited = {episode_ids[0]}
    pending = [episode_ids[0]]
    while pending:
        current = pending.pop()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return visited == selected


def _assert_evolving_storyline_coverage(
    episodes: list[dict[str, Any]], storylines: list[dict[str, Any]]
) -> None:
    expected = [
        episode["episode_id"]
        for episode in sorted(episodes, key=lambda item: int(item["order"]))
    ]
    expected_set = set(expected)
    backbones = [
        storyline
        for storyline in storylines
        if storyline.get("focus_type") == "chronological_backbone"
    ]
    if len(backbones) != 1 or backbones[0].get("child_episode_ids") != expected:
        raise ValueError("Exactly one chronological backbone must cover all episodes")
    for storyline in storylines:
        children = storyline.get("child_episode_ids", [])
        if not children or len(children) != len(set(children)):
            raise ValueError("Storyline episode membership must be non-empty and unique")
        if any(child not in expected_set for child in children):
            raise ValueError("Storyline references an unknown episode")
        if storyline.get("focus_type") == "chronological_backbone":
            continue
        if len(children) < 2 or len(storyline.get("source_scene_ids", [])) < 2:
            raise ValueError("Evolution storylines require two episodes across two scenes")
        if storyline.get("focus_type") not in STORYLINE_FOCUS_TYPES:
            raise ValueError("Evolution storyline has an unknown focus type")
        if not storyline.get("ordered_transitions"):
            raise ValueError("Evolution storyline requires structured transitions")


def _unique_entity_refs(values: Any) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        entity_id = clean_text(value.get("entity_id"))
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        output.append(
            {
                "entity_id": entity_id,
                "canonical_name": clean_text(value.get("canonical_name")),
            }
        )
    return output


def _validation_feedback(error: Exception | None, instruction: str) -> str:
    if error is None:
        return ""
    return (
        "\n\nThe previous output failed schema validation with this error: "
        f"{clean_text(error)}. {instruction} Return corrected JSON only."
    )
