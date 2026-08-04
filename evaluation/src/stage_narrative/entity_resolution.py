from __future__ import annotations

import asyncio
import copy
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

from .io import atomic_write_json, load_json, sha256_json
from .models import (
    clean_text,
    normalize_name,
    require_entity_scope,
    require_entity_type,
    stable_id,
    unique_text,
)
from .prompt_loader import PROMPTS
from .type_resolution import initial_type_profile, legacy_entity_type, resolve_cluster_type
from .type_resolution import type_compatibility as compare_types


_ENTITY_PAIR_PROMPT = PROMPTS.get("entity_pair_review")
ENTITY_PAIR_SYSTEM = _ENTITY_PAIR_PROMPT.system
ENTITY_PAIR_USER = _ENTITY_PAIR_PROMPT.user
_ENTITY_CLUSTER_REVIEW_PROMPT = PROMPTS.get("entity_cluster_review")
ENTITY_CLUSTER_REVIEW_SYSTEM = _ENTITY_CLUSTER_REVIEW_PROMPT.system
ENTITY_CLUSTER_REVIEW_USER = _ENTITY_CLUSTER_REVIEW_PROMPT.user


class EmbeddingClient(Protocol):
    async def embed(
        self, texts: list[str], *, stage: str
    ) -> tuple[list[list[float]], dict[str, Any]]: ...


class JsonClient(Protocol):
    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ResolutionConfig:
    lexical_threshold: float = 0.88
    embedding_threshold: float = 0.86
    embedding_top_k: int = 8
    embedding_batch_size: int = 128
    decision_batch_size: int = 16
    cluster_decision_batch_size: int = 4
    decision_min_confidence: float = 0.75
    same_identity_probability_threshold: float = 0.70
    different_identity_probability_threshold: float = 0.25
    cluster_review_rounds: int = 2
    semantic_attempts: int = 2
    max_concurrency: int = 4


class _UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}
        self.members = {value: {value} for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if min(self.members[right_root]) < min(self.members[left_root]):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.members[left_root].update(self.members.pop(right_root))

    def would_violate(
        self, left: str, right: str, blocked_pairs: set[frozenset[str]]
    ) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        return any(
            frozenset((left_member, right_member)) in blocked_pairs
            for left_member in self.members[left_root]
            for right_member in self.members[right_root]
        )


class EntityResolver:
    def __init__(
        self,
        *,
        movie_id: str,
        llm_client: JsonClient,
        embedding_client: EmbeddingClient,
        config: ResolutionConfig,
        token_counter: TokenCounter | None = None,
        max_input_tokens: int | None = None,
        checkpoint_dir: Path | None = None,
    ):
        self.movie_id = movie_id
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.token_counter = token_counter
        self.max_input_tokens = max_input_tokens
        self.checkpoint_dir = checkpoint_dir.resolve() if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def resolve(self, scene_records: list[dict[str, Any]]) -> dict[str, Any]:
        mentions, explicit_aliases = _collect_mentions(scene_records)
        if not mentions:
            return _empty_registry()

        rejected_alias_pairs, ambiguous_alias_pairs = _alias_pair_constraints(
            mentions, explicit_aliases
        )
        same_scene_character_conflicts = _same_scene_character_conflicts(
            mentions, explicit_aliases
        )
        deterministic_blocked_pairs = {
            *rejected_alias_pairs,
            *same_scene_character_conflicts,
        }

        mention_keys = sorted(
            mentions,
            key=lambda key: (
                mentions[key]["source_scene_order"],
                mentions[key]["normalized_name"],
                key,
            ),
        )
        vectors, embedding_calls = await self._embed_mentions(mention_keys, mentions)
        retrieved_candidates = _build_candidates(
            mentions=mentions,
            mention_keys=mention_keys,
            vectors=vectors,
            explicit_aliases=explicit_aliases,
            rejected_alias_pairs=rejected_alias_pairs,
            ambiguous_alias_pairs=ambiguous_alias_pairs,
            hard_conflict_pairs=same_scene_character_conflicts,
            config=self.config,
        )
        candidates, deterministic_exclusions = _partition_reviewable_candidates(
            retrieved_candidates
        )
        decisions, decision_calls = await self._judge_candidates(candidates, mentions)

        union_find = _UnionFind(mention_keys)
        model_strong_negative_pairs = _strong_negative_pairs(decisions, self.config)
        effective_negative_pairs, suppressed_negative_pairs = (
            _reconcile_strong_negative_pairs(
                mention_keys=mention_keys,
                decisions=decisions,
                deterministic_blocked_pairs=deterministic_blocked_pairs,
                strong_negative_pairs=model_strong_negative_pairs,
                config=self.config,
            )
        )
        blocked_pairs = {
            *deterministic_blocked_pairs,
            *effective_negative_pairs,
        }
        _apply_identity_decisions(
            union_find=union_find,
            decisions=decisions,
            blocked_pairs=blocked_pairs,
            config=self.config,
        )

        cluster_rounds: list[dict[str, Any]] = []
        cluster_calls: list[dict[str, Any]] = []
        cluster_decisions_all: list[dict[str, Any]] = []
        cluster_exclusions: list[dict[str, Any]] = []
        for round_index in range(1, max(0, self.config.cluster_review_rounds) + 1):
            groups = _groups(union_find, mention_keys)
            retrieved_cluster_candidates = _build_cluster_candidates(
                groups,
                mentions,
                blocked_pairs=blocked_pairs,
            )
            cluster_candidates, round_exclusions = _partition_reviewable_candidates(
                retrieved_cluster_candidates
            )
            cluster_exclusions.extend(
                {**item, "review_round": round_index} for item in round_exclusions
            )
            if not cluster_candidates:
                if retrieved_cluster_candidates:
                    cluster_rounds.append(
                        {
                            "round": round_index,
                            "input_cluster_count": len(groups),
                            "retrieved_candidate_count": len(
                                retrieved_cluster_candidates
                            ),
                            "deterministic_exclusion_count": len(round_exclusions),
                            "candidate_count": 0,
                            "merge_count": 0,
                        }
                    )
                break
            cluster_decisions, calls = await self._judge_clusters(
                cluster_candidates,
                groups=groups,
                mentions=mentions,
                round_index=round_index,
            )
            cluster_calls.extend(calls)
            cluster_decisions_all.extend(cluster_decisions)
            merge_count = _apply_cluster_identity_decisions(
                union_find=union_find,
                decisions=cluster_decisions,
                blocked_pairs=blocked_pairs,
                config=self.config,
            )
            cluster_rounds.append(
                {
                    "round": round_index,
                    "input_cluster_count": len(groups),
                    "retrieved_candidate_count": len(retrieved_cluster_candidates),
                    "deterministic_exclusion_count": len(round_exclusions),
                    "candidate_count": len(cluster_candidates),
                    "merge_count": merge_count,
                }
            )
            if merge_count == 0:
                break

        groups = _groups(union_find, mention_keys)
        entities, mention_map, alias_map = _materialize_registry(
            movie_id=self.movie_id,
            groups=groups,
            mentions=mentions,
            pair_decisions=decisions,
            cluster_decisions=cluster_decisions_all,
        )
        return {
            "schema_version": "stage_entity_registry_v2",
            "entities": entities,
            "alias_map": alias_map,
            "mention_map": mention_map,
            "audit": {
                "embedding_calls": embedding_calls,
                "retrieved_candidate_count": len(retrieved_candidates),
                "deterministic_exclusion_count": len(deterministic_exclusions),
                "deterministic_exclusions": deterministic_exclusions,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "decision_calls": decision_calls,
                "decisions": decisions,
                "cluster_review_rounds": cluster_rounds,
                "cluster_review_calls": cluster_calls,
                "cluster_decisions": cluster_decisions_all,
                "cluster_deterministic_exclusions": cluster_exclusions,
                "deterministic_blocked_pair_count": len(
                    deterministic_blocked_pairs
                ),
                "model_strong_negative_pair_count": len(
                    model_strong_negative_pairs
                ),
                "effective_strong_negative_pair_count": len(
                    effective_negative_pairs
                ),
                "suppressed_strong_negative_pairs": suppressed_negative_pairs,
                "review_rejected_alias_pair_count": len(rejected_alias_pairs),
                "ambiguous_alias_pair_count": len(ambiguous_alias_pairs),
                "same_scene_character_conflict_count": len(
                    same_scene_character_conflicts
                ),
                "note": (
                    "generated_rationale_hint is model-generated audit context, "
                    "not screenplay evidence"
                ),
            },
        }

    async def _embed_mentions(
        self, mention_keys: list[str], mentions: dict[str, dict[str, Any]]
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        representations = [_embedding_text(mentions[key]) for key in mention_keys]
        vectors: list[list[float]] = []
        calls: list[dict[str, Any]] = []
        batch_size = max(1, self.config.embedding_batch_size)
        for start in range(0, len(representations), batch_size):
            batch_vectors, metadata = await self.embedding_client.embed(
                representations[start : start + batch_size],
                stage=f"entity_embedding:{start // batch_size:04d}",
            )
            vectors.extend(batch_vectors)
            calls.append(metadata)
        if len(vectors) != len(mention_keys):
            raise ValueError("Entity embedding count does not match entity mention count")
        return vectors, calls
    async def _judge_candidates(
        self,
        candidates: list[dict[str, Any]],
        mentions: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        batch_size = max(1, self.config.decision_batch_size)
        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def judge_batch(
            start: int,
            *,
            width: int | None = None,
            split_path: tuple[int, ...] = (),
        ):
            actual_width = max(1, int(width or batch_size))
            batch = candidates[start : start + actual_width]
            pair_by_id: dict[str, dict[str, Any]] = {}
            prompt_rows: list[dict[str, Any]] = []
            for offset, candidate in enumerate(batch, start=start + 1):
                pair_id = f"P{offset:06d}"
                pair_by_id[pair_id] = candidate
                prompt_rows.append(
                    {
                        "pair_id": pair_id,
                        "left": _mention_prompt_record(
                            mentions[candidate["left_mention_key"]]
                        ),
                        "right": _mention_prompt_record(
                            mentions[candidate["right_mention_key"]]
                        ),
                        "candidate_signals": candidate["signals"],
                    }
                )
            batch_number = start // batch_size
            if split_path:
                split_label = ".".join(str(item) for item in split_path)
                stage = f"entity_resolution:{batch_number:04d}:split:{split_label}"
            else:
                stage = f"entity_resolution:{batch_number:04d}"
            user_prompt = ENTITY_PAIR_USER.format(
                candidate_pairs=json.dumps(prompt_rows, ensure_ascii=False, indent=2)
            )
            allowed_names = {
                pair_id: {
                    mentions[item["left_mention_key"]]["normalized_name"],
                    mentions[item["right_mention_key"]]["normalized_name"],
                }
                for pair_id, item in pair_by_id.items()
            }
            if split_path:
                split_label = ".".join(str(item) for item in split_path)
                checkpoint_path = self._checkpoint_path_named(
                    "pairs_split",
                    f"{batch_number:04d}_{start:06d}_{actual_width:04d}_{split_label}.json",
                )
            else:
                checkpoint_path = self._checkpoint_path("pairs", batch_number)
            cached = self._load_identity_checkpoint(
                checkpoint_path=checkpoint_path,
                stage=stage,
                system_prompt=ENTITY_PAIR_SYSTEM,
                user_prompt=user_prompt,
                pair_by_id=pair_by_id,
                allowed_names=allowed_names,
            )
            if cached is not None:
                normalized, metadata = cached
            else:
                async with semaphore:
                    normalized, metadata, raw_response = await self._semantic_identity_call(
                        system_prompt=ENTITY_PAIR_SYSTEM,
                        user_prompt=user_prompt,
                        stage=stage,
                        pair_by_id=pair_by_id,
                        allowed_names=allowed_names,
                    )
                self._write_identity_checkpoint(
                    checkpoint_path=checkpoint_path,
                    stage=stage,
                    system_prompt=ENTITY_PAIR_SYSTEM,
                    user_prompt=user_prompt,
                    expected_ids=set(pair_by_id),
                    raw_response=raw_response,
                    normalized_decisions=normalized,
                    generator_metadata=metadata,
                )
            for item in normalized:
                candidate = pair_by_id[item["pair_id"]]
                item.update(
                    {
                        "left_mention_key": candidate["left_mention_key"],
                        "right_mention_key": candidate["right_mention_key"],
                        "signals": candidate["signals"],
                        "merge_applied": False,
                        "merge_block_reason": "",
                    }
                )
            if split_path:
                metadata = {
                    **metadata,
                    "checkpoint_split_fallback": True,
                    "checkpoint_split_path": list(split_path),
                    "checkpoint_split_width": actual_width,
                }
            return start, normalized, [metadata]

        async def judge_resilient(
            start: int,
            *,
            width: int | None = None,
            split_path: tuple[int, ...] = (),
        ):
            actual_width = min(
                max(1, int(width or batch_size)), len(candidates) - start
            )
            try:
                return await judge_batch(
                    start, width=actual_width, split_path=split_path
                )
            except Exception:
                # A malformed/partial semantic response is isolated to this
                # batch. Split only the failed batch; successful neighboring
                # checkpoints remain reusable and the split children have
                # deterministic paths for future resume.
                if actual_width <= 1:
                    raise
                left_width = max(1, actual_width // 2)
                right_start = start + left_width
                right_width = actual_width - left_width
                left = await judge_resilient(
                    start,
                    width=left_width,
                    split_path=(*split_path, 0),
                )
                right = await judge_resilient(
                    right_start,
                    width=right_width,
                    split_path=(*split_path, 1),
                )
                return (
                    start,
                    [*left[1], *right[1]],
                    [*left[2], *right[2]],
                )

        results = await asyncio.gather(
            *(
                judge_resilient(start)
                for start in range(0, len(candidates), batch_size)
            )
        )
        results.sort(key=lambda item: item[0])
        decisions = [decision for _, rows, _ in results for decision in rows]
        calls = [metadata for _, _, metadata_rows in results for metadata in metadata_rows]
        return decisions, calls

    async def _judge_clusters(
        self,
        candidates: list[dict[str, Any]],
        *,
        groups: list[list[str]],
        mentions: dict[str, dict[str, Any]],
        round_index: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        group_by_anchor = {min(group): group for group in groups}
        batch_size = max(1, self.config.cluster_decision_batch_size)
        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def judge_batch(
            start: int,
            *,
            width: int | None = None,
            split_path: tuple[int, ...] = (),
        ):
            actual_width = min(
                max(1, int(width or batch_size)), len(candidates) - start
            )
            batch = candidates[start : start + actual_width]
            pair_by_id: dict[str, dict[str, Any]] = {}
            prompt_rows: list[dict[str, Any]] = []
            allowed_names: dict[str, set[str]] = {}
            for offset, candidate in enumerate(batch, start=start + 1):
                pair_id = f"C{round_index:02d}-{offset:06d}"
                pair_by_id[pair_id] = candidate
                left_group = group_by_anchor[candidate["left_anchor"]]
                right_group = group_by_anchor[candidate["right_anchor"]]
                left_summary = _cluster_prompt_record(left_group, mentions)
                right_summary = _cluster_prompt_record(right_group, mentions)
                allowed_names[pair_id] = {
                    normalize_name(left_summary["canonical_name_candidate"]),
                    normalize_name(right_summary["canonical_name_candidate"]),
                }
                prompt_rows.append(
                    {
                        "pair_id": pair_id,
                        "left_cluster": left_summary,
                        "right_cluster": right_summary,
                        "candidate_signals": candidate["signals"],
                    }
                )
            batch_label = f"{start // batch_size:04d}"
            if split_path:
                batch_label += ":split:" + "".join(str(item) for item in split_path)
            stage = f"entity_cluster_review:{round_index:02d}:{batch_label}"
            user_prompt = ENTITY_CLUSTER_REVIEW_USER.format(
                cluster_pairs=json.dumps(prompt_rows, ensure_ascii=False, indent=2)
            )
            if split_path:
                checkpoint_path = self._checkpoint_path_named(
                    f"clusters_{round_index:02d}",
                    f"{start // batch_size:04d}-split-{''.join(str(item) for item in split_path)}.json",
                )
            else:
                checkpoint_path = self._checkpoint_path(
                    f"clusters_{round_index:02d}", start // batch_size
                )
            cached = self._load_identity_checkpoint(
                checkpoint_path=checkpoint_path,
                stage=stage,
                system_prompt=ENTITY_CLUSTER_REVIEW_SYSTEM,
                user_prompt=user_prompt,
                pair_by_id=pair_by_id,
                allowed_names=allowed_names,
            )
            if cached is not None:
                normalized, metadata = cached
            else:
                async with semaphore:
                    normalized, metadata, raw_response = await self._semantic_identity_call(
                        system_prompt=ENTITY_CLUSTER_REVIEW_SYSTEM,
                        user_prompt=user_prompt,
                        stage=stage,
                        pair_by_id=pair_by_id,
                        allowed_names=allowed_names,
                    )
                self._write_identity_checkpoint(
                    checkpoint_path=checkpoint_path,
                    stage=stage,
                    system_prompt=ENTITY_CLUSTER_REVIEW_SYSTEM,
                    user_prompt=user_prompt,
                    expected_ids=set(pair_by_id),
                    raw_response=raw_response,
                    normalized_decisions=normalized,
                    generator_metadata=metadata,
                )
            for item in normalized:
                candidate = pair_by_id[item["pair_id"]]
                item.update(
                    {
                        "left_anchor": candidate["left_anchor"],
                        "right_anchor": candidate["right_anchor"],
                        "signals": candidate["signals"],
                        "review_round": round_index,
                        "merge_applied": False,
                    }
                )
            return start, normalized, [metadata]

        async def judge_resilient(
            start: int,
            *,
            width: int | None = None,
            split_path: tuple[int, ...] = (),
        ):
            actual_width = min(
                max(1, int(width or batch_size)), len(candidates) - start
            )
            try:
                return await judge_batch(
                    start, width=actual_width, split_path=split_path
                )
            except Exception:
                # Cluster summaries can be much larger than pair summaries.
                # If a batch exceeds the model budget or returns a malformed
                # response, isolate only that batch and retry deterministic
                # halves; successful neighboring checkpoints remain reusable.
                if actual_width <= 1:
                    raise
                left_width = max(1, actual_width // 2)
                right_start = start + left_width
                right_width = actual_width - left_width
                left = await judge_resilient(
                    start,
                    width=left_width,
                    split_path=(*split_path, 0),
                )
                right = await judge_resilient(
                    right_start,
                    width=right_width,
                    split_path=(*split_path, 1),
                )
                return (
                    start,
                    [*left[1], *right[1]],
                    [*left[2], *right[2]],
                )

        results = await asyncio.gather(
            *(
                judge_resilient(start)
                for start in range(0, len(candidates), batch_size)
            )
        )
        results.sort(key=lambda item: item[0])
        decisions = [decision for _, rows, _ in results for decision in rows]
        calls = [metadata for _, _, metadata_rows in results for metadata in metadata_rows]
        return decisions, calls

    async def _semantic_identity_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
        pair_by_id: dict[str, dict[str, Any]],
        allowed_names: dict[str, set[str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        last_error: Exception | None = None
        partial_payload: dict[str, Any] | None = None
        partial_missing_ids: set[str] = set()
        for attempt in range(1, max(1, self.config.semantic_attempts) + 1):
            targeted_prompt = partial_payload is not None and bool(partial_missing_ids)
            try:
                if targeted_prompt:
                    complete_prompt = user_prompt + _partial_identity_repair_feedback(
                        partial_missing_ids
                    )
                else:
                    complete_prompt = user_prompt + _validation_feedback(last_error)
                measured_tokens = self._require_prompt_budget(
                    system_prompt, complete_prompt, stage
                )
                call = await self.llm_client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=complete_prompt,
                    stage=stage,
                )
                raw_response = call.data
                targeted_repair = False
                try:
                    normalized = _validate_identity_decisions(
                        raw_response,
                        expected_ids=set(pair_by_id),
                        allowed_names=allowed_names,
                    )
                except Exception:
                    # A truncated/partial response is repaired on the next
                    # bounded semantic attempt by requesting only the missing
                    # pair IDs. Existing valid rows are retained and merged;
                    # malformed or ambiguous rows still fail strict validation.
                    if partial_payload is None:
                        partial = _partial_identity_payload(
                            raw_response, expected_ids=set(pair_by_id)
                        )
                        if partial is not None:
                            partial_payload, partial_missing_ids = partial
                    if partial_payload is None or not partial_missing_ids:
                        raise
                    merged = _merge_partial_identity_payload(
                        partial_payload,
                        raw_response,
                        expected_ids=set(pair_by_id),
                        missing_ids=partial_missing_ids,
                    )
                    if merged is None:
                        raise
                    normalized = _validate_identity_decisions(
                        merged,
                        expected_ids=set(pair_by_id),
                        allowed_names=allowed_names,
                    )
                    raw_response = merged
                    targeted_repair = True
                call_metadata = {
                    **call.metadata,
                    "semantic_attempt": attempt,
                    "prompt_tokens_measured": measured_tokens,
                }
                if targeted_prompt:
                    call_metadata.update(
                        {
                            "targeted_partial_identity_repair": True,
                            "targeted_repair_pair_ids": sorted(partial_missing_ids),
                            "targeted_repair_response_mode": (
                                "merged_missing_rows" if targeted_repair else "full_batch"
                            ),
                        }
                    )
                return (
                    normalized,
                    call_metadata,
                    raw_response,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"{stage} failed semantic validation: {last_error}") from last_error

    def _checkpoint_path(self, family: str, batch_index: int) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        path = self.checkpoint_dir / family / f"{batch_index:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _checkpoint_path_named(self, family: str, name: str) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        path = self.checkpoint_dir / family / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_identity_checkpoint(
        self,
        *,
        checkpoint_path: Path | None,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        pair_by_id: dict[str, dict[str, Any]],
        allowed_names: dict[str, set[str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        if checkpoint_path is None or not checkpoint_path.is_file():
            return None
        checkpoint = load_json(checkpoint_path)
        prompt_sha256 = sha256_json(
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        expected_ids = set(pair_by_id)
        if (
            checkpoint.get("stage") != stage
            or checkpoint.get("prompt_sha256") != prompt_sha256
            or set(checkpoint.get("expected_pair_ids", [])) != expected_ids
        ):
            # A checkpoint produced by an earlier prompt/code revision is a
            # stale cache, not a valid semantic result. Invalidate only this
            # batch and regenerate it; never reuse drifted decisions and never
            # fail the entire lineage because one cache is stale.
            return None
        normalized = _validate_identity_decisions(
            checkpoint["raw_response"],
            expected_ids=expected_ids,
            allowed_names=allowed_names,
        )
        return normalized, {
            **checkpoint["generator_metadata"],
            "checkpoint_reused": True,
        }

    def _write_identity_checkpoint(
        self,
        *,
        checkpoint_path: Path | None,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        expected_ids: set[str],
        raw_response: dict[str, Any],
        normalized_decisions: list[dict[str, Any]],
        generator_metadata: dict[str, Any],
    ) -> None:
        if checkpoint_path is None:
            return
        atomic_write_json(
            checkpoint_path,
            {
                "schema_version": "stage_entity_resolution_batch_checkpoint_v1",
                "stage": stage,
                "prompt_sha256": sha256_json(
                    {"system_prompt": system_prompt, "user_prompt": user_prompt}
                ),
                "expected_pair_ids": sorted(expected_ids),
                "raw_response": raw_response,
                "normalized_decisions": normalized_decisions,
                "generator_metadata": generator_metadata,
            },
        )

    def _require_prompt_budget(
        self, system_prompt: str, user_prompt: str, stage: str
    ) -> int | None:
        if self.token_counter is None or self.max_input_tokens is None:
            return None
        measured = self.token_counter.count(system_prompt + "\n" + user_prompt)
        if measured > self.max_input_tokens:
            raise ValueError(
                f"{stage} prompt exceeds input budget: "
                f"{measured}>{self.max_input_tokens}"
            )
        return measured


def replay_entity_resolution_decisions(
    *,
    movie_id: str,
    scene_records: list[dict[str, Any]],
    prior_registry: dict[str, Any],
    config: ResolutionConfig,
) -> dict[str, Any]:
    """Re-materialize a registry from frozen pair decisions without model calls."""

    mentions, explicit_aliases = _collect_mentions(scene_records)
    if not mentions:
        return _empty_registry()
    mention_keys = sorted(
        mentions,
        key=lambda key: (
            mentions[key]["source_scene_order"],
            mentions[key]["normalized_name"],
            key,
        ),
    )
    prior_audit = copy.deepcopy(prior_registry.get("audit") or {})
    decisions = copy.deepcopy(prior_audit.get("decisions") or [])
    known_keys = set(mention_keys)
    decision_keys = {
        value
        for item in decisions
        for value in (
            item.get("left_mention_key"),
            item.get("right_mention_key"),
        )
    }
    unknown_keys = sorted(value for value in decision_keys if value not in known_keys)
    if unknown_keys:
        raise ValueError(
            "Prior identity decisions reference unknown mentions: "
            + ", ".join(unknown_keys[:8])
        )

    rejected_alias_pairs, ambiguous_alias_pairs = _alias_pair_constraints(
        mentions, explicit_aliases
    )
    same_scene_character_conflicts = _same_scene_character_conflicts(
        mentions, explicit_aliases
    )
    deterministic_blocked_pairs = {
        *rejected_alias_pairs,
        *same_scene_character_conflicts,
    }
    for decision in decisions:
        decision["merge_applied"] = False
        decision["merge_block_reason"] = ""

    model_strong_negative_pairs = _strong_negative_pairs(decisions, config)
    effective_negative_pairs, suppressed_negative_pairs = (
        _reconcile_strong_negative_pairs(
            mention_keys=mention_keys,
            decisions=decisions,
            deterministic_blocked_pairs=deterministic_blocked_pairs,
            strong_negative_pairs=model_strong_negative_pairs,
            config=config,
        )
    )
    blocked_pairs = {
        *deterministic_blocked_pairs,
        *effective_negative_pairs,
    }
    union_find = _UnionFind(mention_keys)
    _apply_identity_decisions(
        union_find=union_find,
        decisions=decisions,
        blocked_pairs=blocked_pairs,
        config=config,
    )

    cluster_decisions = copy.deepcopy(prior_audit.get("cluster_decisions") or [])
    for decision in cluster_decisions:
        decision["merge_applied"] = False
        decision["merge_block_reason"] = ""
    _apply_cluster_identity_decisions(
        union_find=union_find,
        decisions=cluster_decisions,
        blocked_pairs=blocked_pairs,
        config=config,
    )
    groups = _groups(union_find, mention_keys)
    entities, mention_map, alias_map = _materialize_registry(
        movie_id=movie_id,
        groups=groups,
        mentions=mentions,
        pair_decisions=decisions,
        cluster_decisions=cluster_decisions,
    )
    prior_audit.update(
        {
            "decisions": decisions,
            "cluster_decisions": cluster_decisions,
            "deterministic_blocked_pair_count": len(
                deterministic_blocked_pairs
            ),
            "model_strong_negative_pair_count": len(
                model_strong_negative_pairs
            ),
            "effective_strong_negative_pair_count": len(
                effective_negative_pairs
            ),
            "suppressed_strong_negative_pairs": suppressed_negative_pairs,
            "review_rejected_alias_pair_count": len(rejected_alias_pairs),
            "ambiguous_alias_pair_count": len(ambiguous_alias_pairs),
            "same_scene_character_conflict_count": len(
                same_scene_character_conflicts
            ),
            "decision_replay": {
                "model_calls": 0,
                "source_registry_schema_version": prior_registry.get(
                    "schema_version"
                ),
                "policy": "reconcile frozen decisions and re-materialize clusters",
            },
        }
    )
    return {
        "schema_version": "stage_entity_registry_v2",
        "entities": entities,
        "alias_map": alias_map,
        "mention_map": mention_map,
        "audit": prior_audit,
    }


async def build_entity_resolution_call_budget(
    scene_records: list[dict[str, Any]],
    *,
    embedding_client: EmbeddingClient,
    config: ResolutionConfig,
) -> dict[str, Any]:
    """Build a no-LLM candidate and call budget from frozen reviewed scenes."""
    mentions, explicit_aliases = _collect_mentions(scene_records)
    if not mentions:
        return {
            "schema_version": "stage_entity_resolution_call_budget_v1",
            "mention_count": 0,
            "embedding_call_count": 0,
            "pair_review": {
                "retrieved_candidate_count": 0,
                "deterministic_exclusion_count": 0,
                "formal_call_count": 0,
                "maximum_targeted_repair_call_count": 0,
            },
            "cluster_review": {
                "rounds": config.cluster_review_rounds,
                "pre_pair_candidate_estimate": 0,
                "pre_pair_formal_call_estimate": 0,
                "structural_maximum_formal_calls": 0,
            },
        }

    rejected_alias_pairs, ambiguous_alias_pairs = _alias_pair_constraints(
        mentions, explicit_aliases
    )
    character_conflicts = _same_scene_character_conflicts(mentions, explicit_aliases)
    blocked_pairs = {*rejected_alias_pairs, *character_conflicts}
    mention_keys = sorted(
        mentions,
        key=lambda key: (
            mentions[key]["source_scene_order"],
            mentions[key]["normalized_name"],
            key,
        ),
    )
    representations = [_embedding_text(mentions[key]) for key in mention_keys]
    vectors: list[list[float]] = []
    embedding_calls: list[dict[str, Any]] = []
    embedding_batch_size = max(1, config.embedding_batch_size)
    for start in range(0, len(representations), embedding_batch_size):
        batch_vectors, metadata = await embedding_client.embed(
            representations[start : start + embedding_batch_size],
            stage=f"entity_budget_embedding:{start // embedding_batch_size:04d}",
        )
        vectors.extend(batch_vectors)
        embedding_calls.append(metadata)
    if len(vectors) != len(mention_keys):
        raise ValueError("Entity budget embedding count does not match mention count")

    retrieved = _build_candidates(
        mentions=mentions,
        mention_keys=mention_keys,
        vectors=vectors,
        explicit_aliases=explicit_aliases,
        rejected_alias_pairs=rejected_alias_pairs,
        ambiguous_alias_pairs=ambiguous_alias_pairs,
        hard_conflict_pairs=character_conflicts,
        config=config,
    )
    reviewable, excluded = _partition_reviewable_candidates(retrieved)
    pair_batch_size = max(1, config.decision_batch_size)
    pair_formal_calls = _ceil_div(len(reviewable), pair_batch_size)

    real_keys = [
        key for key in mention_keys if int(mentions[key].get("frequency", 0)) > 0
    ]
    singleton_groups = [[key] for key in real_keys]
    pre_pair_cluster_retrieved = _build_cluster_candidates(
        singleton_groups,
        mentions,
        blocked_pairs=blocked_pairs,
    )
    pre_pair_cluster_reviewable, pre_pair_cluster_excluded = (
        _partition_reviewable_candidates(pre_pair_cluster_retrieved)
    )
    cluster_batch_size = max(1, config.cluster_decision_batch_size)
    pre_pair_cluster_calls = _ceil_div(
        len(pre_pair_cluster_reviewable), cluster_batch_size
    )
    structural_cluster_pairs = len(real_keys) * max(0, len(real_keys) - 1) // 2
    structural_cluster_calls = (
        _ceil_div(structural_cluster_pairs, cluster_batch_size)
        * max(0, config.cluster_review_rounds)
    )
    repair_multiplier = max(0, config.semantic_attempts - 1)
    return {
        "schema_version": "stage_entity_resolution_call_budget_v1",
        "policy": {
            "formal_review": "one decision per retrieved reviewable pair",
            "targeted_repair": (
                "only after a formal response fails deterministic semantic validation"
            ),
            "transport_retry": "configured separately by the model client",
            "no_llm_calls_during_budget": True,
        },
        "mention_count": len(mentions),
        "real_mention_count": len(real_keys),
        "synthetic_alias_mention_count": len(mentions) - len(real_keys),
        "explicit_alias_claim_count": len(explicit_aliases),
        "review_rejected_alias_pair_count": len(rejected_alias_pairs),
        "ambiguous_alias_pair_count": len(ambiguous_alias_pairs),
        "same_scene_character_conflict_count": len(character_conflicts),
        "embedding_call_count": len(embedding_calls),
        "embedding_calls": embedding_calls,
        "pair_review": {
            "retrieved_candidate_count": len(retrieved),
            "deterministic_exclusion_count": len(excluded),
            "reviewable_candidate_count": len(reviewable),
            "batch_size": pair_batch_size,
            "formal_call_count": pair_formal_calls,
            "maximum_targeted_repair_call_count": (
                pair_formal_calls * repair_multiplier
            ),
        },
        "cluster_review": {
            "rounds": config.cluster_review_rounds,
            "batch_size": cluster_batch_size,
            "pre_pair_retrieved_candidate_estimate": len(
                pre_pair_cluster_retrieved
            ),
            "pre_pair_deterministic_exclusion_estimate": len(
                pre_pair_cluster_excluded
            ),
            "pre_pair_candidate_estimate": len(pre_pair_cluster_reviewable),
            "pre_pair_formal_call_estimate": pre_pair_cluster_calls,
            "structural_maximum_pair_count": structural_cluster_pairs,
            "structural_maximum_formal_calls": structural_cluster_calls,
            "note": (
                "Exact cluster calls are known only after pair decisions form clusters; "
                "the pre-pair estimate is operational, while the structural maximum is "
                "a conservative non-planning bound."
            ),
        },
    }


def _empty_registry() -> dict[str, Any]:
    return {
        "schema_version": "stage_entity_registry_v2",
        "entities": [],
        "alias_map": {},
        "mention_map": {},
        "audit": {
            "retrieved_candidate_count": 0,
            "deterministic_exclusion_count": 0,
            "deterministic_exclusions": [],
            "candidate_count": 0,
            "candidates": [],
            "decisions": [],
            "cluster_decisions": [],
            "cluster_deterministic_exclusions": [],
            "deterministic_blocked_pair_count": 0,
            "review_rejected_alias_pair_count": 0,
            "ambiguous_alias_pair_count": 0,
            "same_scene_character_conflict_count": 0,
        },
    }


def _ceil_div(value: int, divisor: int) -> int:
    return 0 if value <= 0 else (value + divisor - 1) // divisor


def _collect_mentions(
    scene_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    raw: dict[str, dict[str, Any]] = {}
    explicit_aliases: list[tuple[str, str]] = []
    for record in scene_records:
        scene_id = clean_text(record["scene"]["scene_id"])
        scene_order = int(record["scene"]["order"])
        modality_context = _scene_modality_context(record)
        for entity in record.get("entities", []):
            name = clean_text(entity.get("name"))
            normalized = normalize_name(name)
            if not name or not normalized:
                continue
            key = _mention_key(scene_id, normalized)
            item = raw.setdefault(
                key,
                {
                    "name_variants": Counter(),
                    "types": Counter(),
                    "scopes": Counter(),
                    "descriptions": [],
                    "evidence": [],
                    "source_mention_ids": [],
                    "source_scene_ids": {scene_id},
                    "declared_aliases": [],
                    "frequency": 0,
                    "source_scene_order": scene_order,
                    "normalized_name": normalized,
                    "scene_modality_context": [],
                    "review_reason_codes": [],
                },
            )
            item["name_variants"][name] += 1
            item["types"][require_entity_type(entity.get("entity_type"))] += 1
            item["scopes"][require_entity_scope(entity.get("scope"))] += 1
            item["descriptions"].extend(unique_text([entity.get("description")]))
            item["evidence"].extend(unique_text([entity.get("evidence")]))
            item["source_mention_ids"].extend(
                unique_text(
                    [entity.get("mention_id"), *(entity.get("source_entity_ids") or [])]
                )
            )
            item["frequency"] += 1
            item["scene_modality_context"].extend(modality_context)
            item["review_reason_codes"].extend(
                unique_text([entity.get("review", {}).get("reason_code")])
            )
            for alias in unique_text(entity.get("aliases") or [], limit=64):
                alias_normalized = normalize_name(alias)
                if not alias_normalized or alias_normalized == normalized:
                    continue
                alias_key = _mention_key(scene_id, alias_normalized)
                explicit_aliases.append((key, alias_key))
                item["declared_aliases"].append(alias)
                alias_item = raw.setdefault(
                    alias_key,
                    {
                        "name_variants": Counter(),
                        "types": Counter(),
                        "scopes": Counter(),
                        "descriptions": [],
                        "evidence": [],
                        "source_mention_ids": [],
                        "source_scene_ids": {scene_id},
                        "declared_aliases": [],
                        "frequency": 0,
                        "source_scene_order": scene_order,
                        "normalized_name": alias_normalized,
                        "scene_modality_context": [],
                        "review_reason_codes": [],
                    },
                )
                alias_item["name_variants"][alias] += 0
                alias_item["types"][require_entity_type(entity.get("entity_type"))] += 0
                alias_item["scopes"][require_entity_scope(entity.get("scope"))] += 0
                alias_item["source_mention_ids"].extend(item["source_mention_ids"])
                alias_item["scene_modality_context"].extend(modality_context)

    mentions: dict[str, dict[str, Any]] = {}
    for key, item in raw.items():
        variants: Counter[str] = item["name_variants"]
        display = sorted(variants, key=lambda value: (-variants[value], -len(value), value))[0]
        known_types = Counter(item["types"])
        if not known_types:
            raise ValueError(f"Mention {display!r} has no legal entity type")
        entity_type = known_types.most_common(1)[0][0]
        known_scopes = Counter(item["scopes"])
        scope = known_scopes.most_common(1)[0][0] if known_scopes else "local"
        mentions[key] = {
            "name": display,
            "name_variants": sorted(
                variants, key=lambda value: (-variants[value], -len(value), value)
            ),
            "normalized_name": item["normalized_name"],
            "entity_type": entity_type,
            "scope": scope,
            "type_profile": initial_type_profile(entity_type),
            "raw_types": sorted(item["types"]),
            "descriptions": unique_text(item["descriptions"]),
            "evidence": unique_text(item["evidence"]),
            "source_mention_ids": unique_text(item["source_mention_ids"], limit=256),
            "source_scene_ids": sorted(item["source_scene_ids"]),
            "declared_aliases": unique_text(item["declared_aliases"]),
            "frequency": int(item["frequency"]),
            "source_scene_order": int(item["source_scene_order"]),
            "scene_modality_context": unique_text(
                item["scene_modality_context"], limit=16
            ),
            "review_reason_codes": unique_text(item["review_reason_codes"]),
        }
    return mentions, explicit_aliases


def _alias_pair_constraints(
    mentions: dict[str, dict[str, Any]],
    explicit_aliases: list[tuple[str, str]],
) -> tuple[set[frozenset[str]], set[frozenset[str]]]:
    """Separate explicitly rejected aliases from globally ambiguous alias claims."""
    claims_by_alias: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for owner_key, alias_key in explicit_aliases:
        if owner_key not in mentions or alias_key not in mentions:
            continue
        alias_name = mentions[alias_key]["normalized_name"]
        claims_by_alias[alias_name].append((owner_key, alias_key))

    rejected: set[frozenset[str]] = set()
    ambiguous: set[frozenset[str]] = set()
    for claims in claims_by_alias.values():
        for owner_key, alias_key in claims:
            if "alias_conflict" in set(
                mentions[owner_key].get("review_reason_codes") or []
            ):
                rejected.add(frozenset((owner_key, alias_key)))

        owner_keys = sorted({owner_key for owner_key, _ in claims})
        has_incompatible_owners = any(
            not _identity_name_compatible(
                mentions[left]["normalized_name"],
                mentions[right]["normalized_name"],
            )
            for left, right in combinations(owner_keys, 2)
        )
        if has_incompatible_owners:
            ambiguous.update(
                frozenset((owner_key, alias_key))
                for owner_key, alias_key in claims
            )
    return rejected, ambiguous


def _same_scene_character_conflicts(
    mentions: dict[str, dict[str, Any]],
    explicit_aliases: list[tuple[str, str]] | None = None,
) -> set[frozenset[str]]:
    """Treat separately extracted, co-present named Characters as distinct."""
    declared_alias_names = {
        frozenset(
            (
                mentions[left]["normalized_name"],
                mentions[right]["normalized_name"],
            )
        )
        for left, right in (explicit_aliases or [])
        if left in mentions and right in mentions
    }
    by_scene: dict[str, list[str]] = defaultdict(list)
    for key, mention in mentions.items():
        if int(mention.get("frequency", 0)) <= 0:
            continue
        if "Character" not in set(
            mention.get("type_profile", {}).get("candidate_primary_kinds") or []
        ):
            continue
        for scene_id in mention.get("source_scene_ids", []):
            by_scene[scene_id].append(key)

    blocked: set[frozenset[str]] = set()
    for keys in by_scene.values():
        for left, right in combinations(sorted(set(keys)), 2):
            if _identity_name_compatible(
                mentions[left]["normalized_name"],
                mentions[right]["normalized_name"],
            ):
                continue
            if _name_token_containment_candidate(
                mentions[left]["name"], mentions[right]["name"]
            ):
                continue
            if (
                frozenset(
                    (
                        mentions[left]["normalized_name"],
                        mentions[right]["normalized_name"],
                    )
                )
                in declared_alias_names
            ):
                continue
            blocked.add(frozenset((left, right)))
    return blocked


def _identity_name_compatible(left: str, right: str) -> bool:
    return left == right or _name_containment_candidate(left, right)


def _build_candidates(
    *,
    mentions: dict[str, dict[str, Any]],
    mention_keys: list[str],
    vectors: list[list[float]],
    explicit_aliases: list[tuple[str, str]],
    config: ResolutionConfig,
    rejected_alias_pairs: set[frozenset[str]] | None = None,
    ambiguous_alias_pairs: set[frozenset[str]] | None = None,
    hard_conflict_pairs: set[frozenset[str]] | None = None,
) -> list[dict[str, Any]]:
    rejected_alias_pairs = rejected_alias_pairs or set()
    ambiguous_alias_pairs = ambiguous_alias_pairs or set()
    hard_conflict_pairs = hard_conflict_pairs or set()
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    def add(left: str, right: str, signal: str, score: float | None = None) -> None:
        if left == right or left not in mentions or right not in mentions:
            return
        pair = tuple(sorted((left, right)))
        item = candidates.setdefault(
            pair,
            {
                "left_mention_key": pair[0],
                "right_mention_key": pair[1],
                "signals": [],
            },
        )
        detail: dict[str, Any] = {"kind": signal}
        if score is not None:
            detail["score"] = round(score, 6)
        if detail not in item["signals"]:
            item["signals"].append(detail)

    for left, right in explicit_aliases:
        add(left, right, "explicit_alias")

    by_normalized_name: dict[str, list[str]] = defaultdict(list)
    for key in mention_keys:
        by_normalized_name[mentions[key]["normalized_name"]].append(key)
    for same_name_keys in by_normalized_name.values():
        ordered = sorted(
            same_name_keys,
            key=lambda key: (mentions[key]["source_scene_order"], key),
        )
        anchor = ordered[0]
        for key in ordered[1:]:
            add(anchor, key, "exact_name_anchor")
        for left, right in zip(ordered, ordered[1:]):
            add(left, right, "exact_name_adjacent")

    for left, right in combinations(mention_keys, 2):
        left_name = mentions[left]["normalized_name"]
        right_name = mentions[right]["normalized_name"]
        if left_name == right_name:
            continue
        if _name_containment_candidate(left_name, right_name):
            add(left, right, "name_containment")
        if (
            _mentions_are_character_compatible(mentions[left], mentions[right])
            and _name_token_containment_candidate(
                mentions[left]["name"], mentions[right]["name"]
            )
        ):
            add(left, right, "name_token_containment")
        lexical = SequenceMatcher(None, left_name, right_name).ratio()
        if lexical >= config.lexical_threshold:
            add(left, right, "lexical", lexical)

    for left, right, similarity in _embedding_topk_candidates(
        mention_keys=mention_keys,
        vectors=vectors,
        threshold=config.embedding_threshold,
        top_k=max(1, config.embedding_top_k),
        matrix_batch_size=max(1, config.embedding_batch_size),
    ):
        add(left, right, "embedding", similarity)

    for item in candidates.values():
        pair = frozenset(
            (item["left_mention_key"], item["right_mention_key"])
        )
        if pair in rejected_alias_pairs:
            item["signals"].append({"kind": "review_rejected_alias"})
        if pair in ambiguous_alias_pairs:
            item["signals"].append({"kind": "ambiguous_alias_ownership"})
        if pair in hard_conflict_pairs:
            item["signals"].append(
                {
                    "kind": "hard_identity_conflict",
                    "value": "same_scene_distinct_characters",
                }
            )
        left = mentions[item["left_mention_key"]]
        right = mentions[item["right_mention_key"]]
        item["signals"].append(
            {"kind": "type_compatibility", "value": compare_types(
                left["type_profile"], right["type_profile"]
            )}
        )
        if (
            left["normalized_name"] != right["normalized_name"]
            and
            (left.get("scope") == "local" or right.get("scope") == "local")
            and set(left.get("source_scene_ids", [])).isdisjoint(
                right.get("source_scene_ids", [])
            )
        ):
            item["signals"].append(
                {"kind": "scope_compatibility", "value": "local_scope_conflict"}
            )
    return [candidates[pair] for pair in sorted(candidates)]


def _partition_reviewable_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reviewable: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        reason = _deterministic_exclusion_reason(candidate.get("signals", []))
        if reason:
            excluded.append({**candidate, "exclusion_reason": reason})
        else:
            reviewable.append(candidate)
    return reviewable, excluded


def _deterministic_exclusion_reason(signals: list[dict[str, Any]]) -> str:
    if any(
        item.get("kind") == "type_compatibility"
        and item.get("value") == "incompatible_type"
        for item in signals
    ):
        return "incompatible_type"
    if any(
        item.get("kind") == "scope_compatibility"
        and item.get("value") == "local_scope_conflict"
        for item in signals
    ):
        return "local_scope_conflict"
    if any(item.get("kind") == "hard_identity_conflict" for item in signals):
        return "hard_identity_conflict"
    if any(item.get("kind") == "review_rejected_alias" for item in signals):
        return "review_rejected_alias"
    return ""


def _embedding_topk_candidates(
    *,
    mention_keys: list[str],
    vectors: list[list[float]],
    threshold: float,
    top_k: int,
    matrix_batch_size: int,
) -> list[tuple[str, str, float]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for entity candidate retrieval") from exc
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(mention_keys) or matrix.shape[1] == 0:
        raise ValueError("Entity embeddings must form a non-empty rectangular matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    output: list[tuple[str, str, float]] = []
    for start in range(0, len(mention_keys), matrix_batch_size):
        similarities = normalized[start : start + matrix_batch_size] @ normalized.T
        for local_index, row in enumerate(similarities):
            left_index = start + local_index
            scored = [
                (float(value), right_index)
                for right_index, value in enumerate(row)
                if right_index != left_index and float(value) >= threshold
            ]
            for similarity, right_index in sorted(scored, reverse=True)[:top_k]:
                output.append(
                    (mention_keys[left_index], mention_keys[right_index], similarity)
                )
    return output


def _validate_identity_decisions(
    payload: dict[str, Any],
    *,
    expected_ids: set[str],
    allowed_names: dict[str, set[str]],
) -> list[dict[str, Any]]:
    if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
        raise ValueError("Identity resolution must return exactly a decisions array")
    raw_decisions = payload["decisions"]
    # A model can preserve the requested row count while mistyping one pair ID or
    # repeating a neighboring ID. Repair only the balanced case; otherwise keep
    # the strict failure so an ambiguous response is never silently relabeled.
    if len(raw_decisions) == len(expected_ids):
        seen_ids: set[str] = set()
        anomalous_indices: list[int] = []
        for index, raw in enumerate(raw_decisions):
            pair_id = clean_text(raw.get("pair_id")) if isinstance(raw, dict) else ""
            if pair_id not in expected_ids or pair_id in seen_ids:
                anomalous_indices.append(index)
            else:
                seen_ids.add(pair_id)
        missing_ids = sorted(expected_ids - seen_ids)
        if anomalous_indices and len(anomalous_indices) == len(missing_ids):
            repaired_decisions = [dict(item) for item in raw_decisions]
            for index, replacement in zip(anomalous_indices, missing_ids):
                repaired_decisions[index]["_pair_id_repaired_from"] = clean_text(
                    repaired_decisions[index].get("pair_id")
                )
                repaired_decisions[index]["pair_id"] = replacement
            raw_decisions = repaired_decisions
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ValueError("Identity decision must be an object")
        pair_id = clean_text(raw.get("pair_id"))
        if pair_id not in expected_ids or pair_id in by_id:
            raise ValueError(f"Unknown or duplicate identity pair id: {pair_id}")
        decision = clean_text(raw.get("decision"))
        if decision not in {"same_identity", "different_identity", "uncertain"}:
            raise ValueError(f"Unsupported identity decision for {pair_id}: {decision}")
        canonical = clean_text(raw.get("canonical_name"))
        normalization_reason = ""
        if decision == "same_identity":
            if normalize_name(canonical) not in allowed_names[pair_id]:
                decision = "uncertain"
                canonical = ""
                normalization_reason = "invalid_canonical_name"
        else:
            canonical = ""
        repaired_from = clean_text(raw.get("_pair_id_repaired_from"))
        if repaired_from:
            normalization_reason = ";".join(
                value
                for value in (normalization_reason, "deterministic_pair_id_repair")
                if value
            )
        probability = float(raw.get("same_identity_probability", -1))
        confidence = float(raw.get("decision_confidence", -1))
        if not 0.0 <= probability <= 1.0 or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Identity probabilities outside [0,1] for {pair_id}")
        by_id[pair_id] = {
            "pair_id": pair_id,
            "decision": decision,
            "canonical_name": canonical,
            "same_identity_probability": probability,
            "decision_confidence": confidence,
            "generated_rationale_hint": clean_text(raw.get("generated_rationale_hint")),
            "normalization_reason": normalization_reason,
        }
    if set(by_id) != expected_ids:
        raise ValueError(f"Missing identity decisions: {sorted(expected_ids - set(by_id))}")
    return [by_id[pair_id] for pair_id in sorted(by_id)]


def _apply_identity_decisions(
    *,
    union_find: _UnionFind,
    decisions: list[dict[str, Any]],
    blocked_pairs: set[frozenset[str]],
    config: ResolutionConfig,
) -> None:
    for decision in sorted(
        decisions,
        key=lambda item: (
            float(item["same_identity_probability"]),
            float(item["decision_confidence"]),
        ),
        reverse=True,
    ):
        if not _accepted_same_identity(decision, config):
            decision["merge_block_reason"] = decision["decision"]
            continue
        left = decision["left_mention_key"]
        right = decision["right_mention_key"]
        if union_find.would_violate(left, right, blocked_pairs):
            decision["merge_block_reason"] = "conflicts_with_strong_negative"
            continue
        union_find.union(left, right)
        decision["merge_applied"] = True


def _apply_cluster_identity_decisions(
    *,
    union_find: _UnionFind,
    decisions: list[dict[str, Any]],
    blocked_pairs: set[frozenset[str]],
    config: ResolutionConfig,
) -> int:
    merge_count = 0
    for decision in sorted(
        decisions,
        key=lambda item: (
            float(item["same_identity_probability"]),
            float(item["decision_confidence"]),
        ),
        reverse=True,
    ):
        decision["merge_applied"] = False
        decision.setdefault("merge_block_reason", "")
        if not _accepted_same_identity(decision, config):
            decision["merge_block_reason"] = decision["decision"]
            continue
        left = decision["left_anchor"]
        right = decision["right_anchor"]
        if union_find.find(left) == union_find.find(right):
            decision["merge_block_reason"] = "already_same_cluster"
            continue
        if union_find.would_violate(left, right, blocked_pairs):
            decision["merge_block_reason"] = "conflicts_with_blocked_identity_pair"
            continue
        union_find.union(left, right)
        decision["merge_applied"] = True
        merge_count += 1
    return merge_count


def _accepted_same_identity(
    decision: dict[str, Any], config: ResolutionConfig
) -> bool:
    signals = decision.get("signals", [])
    if any(
        signal.get("kind") == "type_compatibility"
        and signal.get("value") == "incompatible_type"
        for signal in signals
    ):
        return False
    if any(signal.get("kind") == "hard_identity_conflict" for signal in signals):
        return False
    if any(signal.get("kind") == "review_rejected_alias" for signal in signals):
        return False
    if any(
        signal.get("kind") == "scope_compatibility"
        and signal.get("value") == "local_scope_conflict"
        for signal in signals
    ):
        return False
    signal_kinds = {signal.get("kind") for signal in signals}
    if (
        any(
            signal.get("kind") == "type_compatibility"
            and signal.get("value") == "role_identity_review"
            for signal in signals
        )
        and "name_containment" not in signal_kinds
    ):
        return False
    if "cluster_review" in signal_kinds and "duplicate_name_or_alias" not in signal_kinds:
        return False
    return (
        decision["decision"] == "same_identity"
        and float(decision["same_identity_probability"])
        >= config.same_identity_probability_threshold
        and float(decision["decision_confidence"])
        >= config.decision_min_confidence
    )


def _strong_negative_pairs(
    decisions: list[dict[str, Any]], config: ResolutionConfig
) -> set[frozenset[str]]:
    return {
        frozenset((item["left_mention_key"], item["right_mention_key"]))
        for item in decisions
        if item["decision"] == "different_identity"
        and float(item["same_identity_probability"])
        <= config.different_identity_probability_threshold
        and float(item["decision_confidence"]) >= config.decision_min_confidence
    }


def _reconcile_strong_negative_pairs(
    *,
    mention_keys: list[str],
    decisions: list[dict[str, Any]],
    deterministic_blocked_pairs: set[frozenset[str]],
    strong_negative_pairs: set[frozenset[str]],
    config: ResolutionConfig,
) -> tuple[set[frozenset[str]], list[dict[str, Any]]]:
    """Keep model negatives only when accepted positive evidence does not refute them.

    Identity is transitive. A pair-level negative contradicted by an accepted
    positive path would otherwise split a well-supported cluster based on edge
    application order. Deterministic conflicts remain absolute.
    """

    positive_union = _UnionFind(mention_keys)
    positive_adjacency: dict[str, set[str]] = defaultdict(set)
    for decision in sorted(
        decisions,
        key=lambda item: (
            float(item["same_identity_probability"]),
            float(item["decision_confidence"]),
        ),
        reverse=True,
    ):
        if not _accepted_same_identity(decision, config):
            continue
        left = decision["left_mention_key"]
        right = decision["right_mention_key"]
        if positive_union.would_violate(
            left, right, deterministic_blocked_pairs
        ):
            continue
        positive_union.union(left, right)
        positive_adjacency[left].add(right)
        positive_adjacency[right].add(left)

    effective: set[frozenset[str]] = set()
    suppressed: list[dict[str, Any]] = []
    for pair in sorted(strong_negative_pairs, key=lambda value: sorted(value)):
        left, right = sorted(pair)
        if pair in deterministic_blocked_pairs:
            effective.add(pair)
            continue
        if positive_union.find(left) != positive_union.find(right):
            effective.add(pair)
            continue
        path = _shortest_identity_path(
            left=left,
            right=right,
            adjacency=positive_adjacency,
        )
        if not path:
            effective.add(pair)
            continue
        suppressed.append(
            {
                "left_mention_key": left,
                "right_mention_key": right,
                "reason": "contradicted_by_accepted_positive_path",
                "positive_path": path,
            }
        )
    return effective, suppressed


def _shortest_identity_path(
    *, left: str, right: str, adjacency: dict[str, set[str]]
) -> list[str]:
    frontier = [left]
    predecessor: dict[str, str | None] = {left: None}
    for node in frontier:
        for neighbor in sorted(adjacency.get(node, set())):
            if neighbor in predecessor:
                continue
            predecessor[neighbor] = node
            if neighbor == right:
                path = [right]
                cursor: str | None = right
                while predecessor[cursor] is not None:
                    cursor = predecessor[cursor]
                    path.append(cursor)
                return list(reversed(path))
            frontier.append(neighbor)
    return []


def _groups(union_find: _UnionFind, mention_keys: list[str]) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in mention_keys:
        grouped[union_find.find(key)].append(key)
    return sorted((sorted(group) for group in grouped.values()), key=lambda group: min(group))


def _build_cluster_candidates(
    groups: list[list[str]],
    mentions: dict[str, dict[str, Any]],
    *,
    blocked_pairs: set[frozenset[str]] | None = None,
) -> list[dict[str, Any]]:
    blocked_pairs = blocked_pairs or set()
    output: list[dict[str, Any]] = []
    for left_group, right_group in combinations(groups, 2):
        if not _group_frequency(left_group, mentions) or not _group_frequency(
            right_group, mentions
        ):
            continue
        left_names = _cluster_names(left_group, mentions)
        right_names = _cluster_names(right_group, mentions)
        signals: list[dict[str, Any]] = []
        normalized_overlap = {
            normalize_name(value) for value in left_names
        } & {normalize_name(value) for value in right_names}
        if normalized_overlap:
            signals.append(
                {"kind": "duplicate_name_or_alias", "values": sorted(normalized_overlap)}
            )
        containment = sorted(
            {
                f"{left}|{right}"
                for left in left_names
                for right in right_names
                if _name_containment_candidate(normalize_name(left), normalize_name(right))
            }
        )
        if containment:
            signals.append({"kind": "name_containment", "values": containment[:8]})
        if not signals:
            continue
        if any(
            frozenset((left, right)) in blocked_pairs
            for left in left_group
            for right in right_group
        ):
            signals.append(
                {
                    "kind": "hard_identity_conflict",
                    "value": "blocked_member_pair",
                }
            )
        left_type = resolve_cluster_type(mentions[key] for key in left_group)
        right_type = resolve_cluster_type(mentions[key] for key in right_group)
        signals.append(
            {
                "kind": "type_compatibility",
                "value": compare_types(left_type, right_type),
            }
        )
        left_scopes = {mentions[key].get("scope", "local") for key in left_group}
        right_scopes = {mentions[key].get("scope", "local") for key in right_group}
        left_scenes = {
            scene_id for key in left_group for scene_id in mentions[key]["source_scene_ids"]
        }
        right_scenes = {
            scene_id for key in right_group for scene_id in mentions[key]["source_scene_ids"]
        }
        if (
            not normalized_overlap
            and
            ("local" in left_scopes or "local" in right_scopes)
            and left_scenes.isdisjoint(right_scenes)
        ):
            signals.append(
                {"kind": "scope_compatibility", "value": "local_scope_conflict"}
            )
        signals.append({"kind": "cluster_review"})
        output.append(
            {
                "left_anchor": min(left_group),
                "right_anchor": min(right_group),
                "signals": signals,
            }
        )
    return output


def _cluster_names(group: list[str], mentions: dict[str, dict[str, Any]]) -> list[str]:
    return unique_text(
        value
        for key in group
        for value in [
            mentions[key]["name"],
            *mentions[key]["name_variants"],
        ]
    )


def _group_frequency(group: list[str], mentions: dict[str, dict[str, Any]]) -> int:
    return sum(int(mentions[key].get("frequency", 0)) for key in group)


def _group_primary_kinds(
    group: list[str], mentions: dict[str, dict[str, Any]]
) -> set[str]:
    kinds: set[str] = set()
    for key in group:
        profile = mentions[key].get("type_profile", {})
        kinds.update(profile.get("candidate_primary_kinds") or [])
    return kinds


def _cluster_prompt_record(
    group: list[str], mentions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = [mentions[key] for key in group]
    return {
        "canonical_name_candidate": _choose_canonical_name(group, mentions, []),
        "names_and_aliases": _cluster_names(group, mentions),
        "type_observations": unique_text(
            value for row in rows for value in row.get("raw_types", [])
        ),
        "scope_observations": unique_text(row.get("scope", "local") for row in rows),
        "scene_span": sorted(
            {scene_id for row in rows for scene_id in row["source_scene_ids"]}
        ),
        "descriptions": unique_text(
            (value for row in rows for value in row["descriptions"]), limit=64
        ),
        "source_evidence": unique_text(
            (value for row in rows for value in row["evidence"]), limit=64
        ),
        "scene_modality_context": unique_text(
            (
                value
                for row in rows
                for value in row.get("scene_modality_context", [])
            ),
            limit=24,
        ),
    }


def _materialize_registry(
    *,
    movie_id: str,
    groups: list[list[str]],
    mentions: dict[str, dict[str, Any]],
    pair_decisions: list[dict[str, Any]],
    cluster_decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    entities: list[dict[str, Any]] = []
    mention_map: dict[str, dict[str, str]] = {}
    aliases_to_entities: dict[str, list[dict[str, str]]] = defaultdict(list)
    accepted_decisions = [
        item
        for item in [*pair_decisions, *cluster_decisions]
        if item.get("decision") == "same_identity"
    ]
    for group in groups:
        if not _group_frequency(group, mentions):
            continue
        rows = [mentions[key] for key in group]
        canonical_name = _choose_canonical_name(group, mentions, accepted_decisions)
        type_resolution = resolve_cluster_type(rows)
        anchor = min(
            group,
            key=lambda key: (
                mentions[key]["source_scene_order"],
                mentions[key]["normalized_name"],
                key,
            ),
        )
        entity_id = stable_id("entity", movie_id, anchor)
        names = _cluster_names(group, mentions)
        aliases = [value for value in names if value != canonical_name]
        entity = {
            "entity_id": entity_id,
            "canonical_name": canonical_name,
            "entity_type": legacy_entity_type(
                type_resolution["primary_kind"], type_resolution["facets"]
            ),
            **type_resolution,
            "scope": (
                "global" if any(row.get("scope") == "global" for row in rows) else "local"
            ),
            "aliases": unique_text(aliases, limit=256),
            "descriptions": unique_text(
                (value for row in rows for value in row["descriptions"]), limit=256
            ),
            "source_scene_ids": sorted(
                {scene_id for row in rows for scene_id in row["source_scene_ids"]}
            ),
            "source_evidence": unique_text(
                (value for row in rows for value in row["evidence"]), limit=256
            ),
            "source_mention_ids": unique_text(
                (value for row in rows for value in row["source_mention_ids"]),
                limit=1024,
            ),
            "anchor_mention_key": anchor,
            "mention_count": sum(row["frequency"] for row in rows),
        }
        entities.append(entity)
        resolved = {"entity_id": entity_id, "canonical_name": canonical_name}
        for key in group:
            mention_map[key] = resolved
        for value in [canonical_name, *entity["aliases"]]:
            normalized = normalize_name(value)
            if normalized:
                aliases_to_entities[normalized].append(resolved)

    alias_map: dict[str, dict[str, str]] = {}
    for normalized, values in aliases_to_entities.items():
        unique = {value["entity_id"]: value for value in values}
        if len(unique) == 1:
            alias_map[normalized] = next(iter(unique.values()))
    entities.sort(key=lambda item: (item["source_scene_ids"][0], item["canonical_name"], item["entity_id"]))
    return entities, mention_map, alias_map


def _choose_canonical_name(
    group: list[str],
    mentions: dict[str, dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> str:
    votes: Counter[str] = Counter()
    members = set(group)
    for decision in decisions:
        endpoints = {
            clean_text(decision.get("left_mention_key") or decision.get("left_anchor")),
            clean_text(decision.get("right_mention_key") or decision.get("right_anchor")),
        }
        endpoints.discard("")
        if endpoints and not endpoints.issubset(members):
            continue
        canonical = normalize_name(decision.get("canonical_name"))
        if canonical:
            votes[canonical] += max(
                1, round(float(decision.get("same_identity_probability", 0)) * 10)
            )
    names = unique_text(
        value for key in group for value in mentions[key]["name_variants"]
    )
    return sorted(
        names,
        key=lambda name: (
            -votes[normalize_name(name)],
            -len(normalize_name(name)),
            -sum(
                mentions[key]["frequency"]
                for key in group
                if normalize_name(name) == mentions[key]["normalized_name"]
            ),
            name,
        ),
    )[0]


def canonicalize_scene_records(
    scene_records: list[dict[str, Any]], entity_registry: dict[str, Any]
) -> list[dict[str, Any]]:
    output = copy.deepcopy(scene_records)
    alias_map = entity_registry.get("alias_map", {})
    mention_map = entity_registry.get("mention_map", {})
    entity_by_id = {
        entity["entity_id"]: entity for entity in entity_registry.get("entities", [])
    }

    def resolve(value: Any, scene_id: str) -> tuple[str, str]:
        original = clean_text(value)
        normalized = normalize_name(original)
        resolved = mention_map.get(_mention_key(scene_id, normalized))
        if not resolved:
            resolved = alias_map.get(normalized)
        if not resolved:
            return original, ""
        return resolved["canonical_name"], resolved["entity_id"]

    for record in output:
        scene_id = clean_text(record["scene"]["scene_id"])
        for entity in record.get("entities", []):
            name, entity_id = resolve(entity.get("name"), scene_id)
            entity["canonical_name"] = name
            entity["canonical_entity_id"] = entity_id
            canonical = entity_by_id.get(entity_id, {})
            entity["primary_kind"] = canonical.get("entity_type", "")
            entity["entity_types"] = list(canonical.get("entity_types", []))
            entity["scope"] = canonical.get("scope", entity.get("scope", "local"))
            entity["facets"] = list(canonical.get("facets", []))
        for relation in record.get("entity_relations", []):
            relation["subject"], relation["subject_entity_id"] = resolve(
                relation.get("subject"), scene_id
            )
            relation["object"], relation["object_entity_id"] = resolve(
                relation.get("object"), scene_id
            )
        for unit in record.get("narrative_units", []):
            unit.setdefault("modality", "asserted")
            if unit.get("kind") == "occasion":
                unit["name"], unit["entity_id"] = resolve(unit.get("name"), scene_id)
            participants: list[dict[str, str]] = []
            names: list[str] = []
            seen: set[str] = set()
            for value in unit.get("participants", []):
                name, entity_id = resolve(value, scene_id)
                names.append(name)
                if entity_id and entity_id not in seen:
                    seen.add(entity_id)
                    participants.append(
                        {"entity_id": entity_id, "canonical_name": name}
                    )
            unit["participants"] = unique_text(names)
            unit["participant_entities"] = participants
            unit["participant_entity_ids"] = [
                item["entity_id"] for item in participants
            ]
            for field, output_field in (
                ("locations", "location_entities"),
                ("times", "time_entities"),
            ):
                resolved_values: list[str] = []
                resolved_entities: list[dict[str, str]] = []
                seen_ids: set[str] = set()
                for value in unit.get(field, []):
                    name, entity_id = resolve(value, scene_id)
                    resolved_values.append(name)
                    if entity_id and entity_id not in seen_ids:
                        seen_ids.add(entity_id)
                        resolved_entities.append(
                            {"entity_id": entity_id, "canonical_name": name}
                        )
                unit[field] = unique_text(resolved_values)
                unit[output_field] = resolved_entities
                unit[f"{field[:-1]}_entity_ids"] = [
                    item["entity_id"] for item in resolved_entities
                ]
            if unit.get("related_occasion"):
                (
                    unit["related_occasion"],
                    unit["related_occasion_entity_id"],
                ) = resolve(unit.get("related_occasion"), scene_id)
            for field in ("subject", "object"):
                if unit.get(field):
                    unit[field], unit[f"{field}_entity_id"] = resolve(
                        unit.get(field), scene_id
                    )
        units_by_kind_and_name: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for unit in record.get("narrative_units", []):
            units_by_kind_and_name[
                (clean_text(unit.get("kind")), normalize_name(unit.get("name")))
            ].append(unit)
        for unit in record.get("narrative_units", []):
            for field, kind, output_field in (
                ("related_event", "event", "related_event_unit_id"),
                ("related_occasion", "occasion", "related_occasion_unit_id"),
            ):
                value = clean_text(unit.get(field))
                if not value:
                    continue
                candidates = units_by_kind_and_name.get((kind, normalize_name(value)), [])
                if len(candidates) == 1:
                    unit[output_field] = candidates[0]["unit_id"]
    finalized, _ = finalize_narrative_references(output, entity_registry)
    return finalized


def finalize_narrative_references(
    scene_records: list[dict[str, Any]], entity_registry: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve deterministic narrative links and remove unsupported optional links."""
    output = copy.deepcopy(scene_records)
    occasion_entities_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    character_surfaces_by_scene: dict[
        str, dict[str, list[dict[str, str]]]
    ] = defaultdict(lambda: defaultdict(list))
    global_character_surfaces: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entity in entity_registry.get("entities", []):
        entity_type = clean_text(
            entity.get("primary_kind") or entity.get("entity_type")
        )
        if entity_type == "Occasion":
            for scene_id in entity.get("source_scene_ids", []):
                occasion_entities_by_scene[clean_text(scene_id)].append(entity)
        if entity_type == "Character":
            character = {
                "entity_id": clean_text(entity.get("entity_id")),
                "canonical_name": clean_text(entity.get("canonical_name")),
            }
            for scene_id in entity.get("source_scene_ids", []):
                scene_surfaces = character_surfaces_by_scene[clean_text(scene_id)]
                for surface in unique_text(
                    [entity.get("canonical_name"), *(entity.get("aliases") or [])]
                ):
                    normalized = normalize_name(surface)
                    if normalized and character not in scene_surfaces[normalized]:
                        scene_surfaces[normalized].append({**character, "surface": surface})
            if clean_text(entity.get("scope")) == "global":
                for surface in unique_text(
                    [entity.get("canonical_name"), *(entity.get("aliases") or [])]
                ):
                    normalized = normalize_name(surface)
                    candidate = {**character, "surface": surface}
                    if normalized and candidate not in global_character_surfaces[normalized]:
                        global_character_surfaces[normalized].append(candidate)

    corrections: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    dropped_empty_occasion_unit_ids: set[str] = set()
    for record in output:
        scene_id = clean_text(record.get("scene", {}).get("scene_id"))
        units = record.get("narrative_units", [])
        units_by_kind_and_name: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for unit in units:
            units_by_kind_and_name[
                (clean_text(unit.get("kind")), normalize_name(unit.get("name")))
            ].append(unit)

        scene_occasions = occasion_entities_by_scene.get(scene_id, [])
        scene_character_surfaces = _merge_character_surface_maps(
            global_character_surfaces,
            character_surfaces_by_scene.get(scene_id, {}),
        )
        for unit in units:
            unit_id = clean_text(unit.get("unit_id"))
            _complete_unit_character_participants(
                unit=unit,
                scene_character_surfaces=scene_character_surfaces,
                scene_id=scene_id,
                corrections=corrections,
            )
            if unit.get("kind") == "occasion" and not unit.get("entity_id"):
                unit_name = normalize_name(unit.get("name"))
                name_matches = [
                    entity
                    for entity in scene_occasions
                    if unit_name
                    and unit_name
                    in {
                        normalize_name(entity.get("canonical_name")),
                        *(normalize_name(alias) for alias in entity.get("aliases", [])),
                    }
                ]
                candidates = name_matches or (
                    scene_occasions if len(scene_occasions) == 1 else []
                )
                if len(candidates) == 1:
                    entity = candidates[0]
                    previous_name = clean_text(unit.get("name"))
                    unit["name"] = clean_text(entity.get("canonical_name"))
                    unit["entity_id"] = clean_text(entity.get("entity_id"))
                    corrections.append(
                        {
                            "scene_id": scene_id,
                            "unit_id": unit_id,
                            "field": "occasion_entity",
                            "action": "map",
                            "strategy": (
                                "same_scene_name_match"
                                if name_matches
                                else "unique_same_scene_occasion"
                            ),
                            "from": previous_name,
                            "to": unit["name"],
                            "entity_id": unit["entity_id"],
                        }
                    )
                else:
                    if not unit_name:
                        dropped_empty_occasion_unit_ids.add(unit_id)
                        corrections.append(
                            {
                                "scene_id": scene_id,
                                "unit_id": unit_id,
                                "field": "occasion_entity",
                                "action": "drop",
                                "strategy": "empty_unresolved_occasion_unit",
                                "candidate_count": len(candidates),
                            }
                        )
                    else:
                        unresolved.append(
                            {
                                "scene_id": scene_id,
                                "unit_id": unit_id,
                                "field": "occasion_entity",
                                "candidate_count": len(candidates),
                            }
                        )

            for field, kind, output_field in (
                ("related_event", "event", "related_event_unit_id"),
                ("related_occasion", "occasion", "related_occasion_unit_id"),
            ):
                value = clean_text(unit.get(field))
                if not value or unit.get(output_field):
                    continue
                candidates = units_by_kind_and_name.get(
                    (kind, normalize_name(value)), []
                )
                if len(candidates) == 1:
                    unit[output_field] = clean_text(candidates[0].get("unit_id"))
                    corrections.append(
                        {
                            "scene_id": scene_id,
                            "unit_id": unit_id,
                            "field": field,
                            "action": "map",
                            "strategy": "exact_same_scene_unit_name",
                            "from": value,
                            "to": value,
                            "target_unit_id": unit[output_field],
                        }
                    )
                    continue
                unit[field] = ""
                unit[output_field] = ""
                if field == "related_occasion":
                    unit["related_occasion_entity_id"] = ""
                corrections.append(
                    {
                        "scene_id": scene_id,
                        "unit_id": unit_id,
                        "field": field,
                        "action": "clear",
                        "strategy": "unsupported_optional_same_scene_link",
                        "from": value,
                        "to": "",
                        "candidate_count": len(candidates),
                    }
                )

        if dropped_empty_occasion_unit_ids:
            record["narrative_units"] = [
                unit
                for unit in record.get("narrative_units", [])
                if clean_text(unit.get("unit_id"))
                not in dropped_empty_occasion_unit_ids
            ]
            for unit in record["narrative_units"]:
                if unit.get("related_occasion_unit_id") in dropped_empty_occasion_unit_ids:
                    unit["related_occasion_unit_id"] = ""
                    unit["related_occasion"] = ""

    audit = {
        "schema_version": "stage_narrative_reference_finalization_audit_v1",
        "deterministic": True,
        "correction_count": len(corrections),
        "unresolved_count": len(unresolved),
        "dropped_empty_occasion_unit_count": len(dropped_empty_occasion_unit_ids),
        "corrections": corrections,
        "unresolved": unresolved,
    }
    return output, audit


def _merge_character_surface_maps(
    *maps: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for mapping in maps:
        for normalized, candidates in mapping.items():
            for candidate in candidates:
                if candidate not in output[normalized]:
                    output[normalized].append(candidate)
    return dict(output)


def _complete_unit_character_participants(
    *,
    unit: dict[str, Any],
    scene_character_surfaces: dict[str, list[dict[str, str]]],
    scene_id: str,
    corrections: list[dict[str, Any]],
) -> None:
    """Add only unambiguous Character references grounded in unit evidence/endpoints."""
    participant_entities: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    existing_name_ids: dict[str, set[str]] = defaultdict(set)
    for raw_item in unit.get("participant_entities", []):
        entity_id = clean_text(raw_item.get("entity_id"))
        canonical_name = clean_text(raw_item.get("canonical_name"))
        if not entity_id or entity_id in seen_ids:
            continue
        normalized_name = normalize_name(canonical_name)
        previous_ids = existing_name_ids.get(normalized_name, set())
        if normalized_name and previous_ids:
            corrections.append(
                {
                    "scene_id": scene_id,
                    "unit_id": clean_text(unit.get("unit_id")),
                    "field": "participants",
                    "action": "deduplicate",
                    "strategy": "same_surface_first_entity_wins",
                    "canonical_name": canonical_name,
                    "dropped_entity_id": entity_id,
                    "kept_entity_ids": sorted(previous_ids),
                }
            )
            continue
        seen_ids.add(entity_id)
        if normalized_name:
            existing_name_ids.setdefault(normalized_name, set()).add(entity_id)
        participant_entities.append(
            {"entity_id": entity_id, "canonical_name": canonical_name}
        )
    additions: list[dict[str, str]] = []

    def add(entity_id: str, canonical_name: str, mechanism: str, surface: str) -> None:
        entity_id = clean_text(entity_id)
        canonical_name = clean_text(canonical_name)
        if not entity_id or entity_id in seen_ids:
            return
        normalized_name = normalize_name(canonical_name)
        existing_ids = existing_name_ids.get(normalized_name, set())
        if existing_ids:
            # Keep one reference per canonical surface when an evidence-based
            # character completion collides with an already resolved entity
            # (for example, a group Organization and Character sharing a name).
            # The entity-level ambiguity remains auditable in the KG review;
            # it must not make participant arrays internally inconsistent.
            corrections.append(
                {
                    "scene_id": scene_id,
                    "unit_id": clean_text(unit.get("unit_id")),
                    "field": "participants",
                    "action": "skip",
                    "strategy": "same_surface_already_resolved",
                    "matched_surface": clean_text(surface),
                    "candidate_entity_id": entity_id,
                    "existing_entity_ids": sorted(existing_ids),
                }
            )
            return
        seen_ids.add(entity_id)
        if normalized_name:
            existing_name_ids.setdefault(normalized_name, set()).add(entity_id)
        item = {"entity_id": entity_id, "canonical_name": canonical_name}
        participant_entities.append(item)
        additions.append(
            {
                **item,
                "mechanism": mechanism,
                "matched_surface": clean_text(surface),
            }
        )

    for role in ("subject", "object"):
        add(
            unit.get(f"{role}_entity_id", ""),
            unit.get(role, ""),
            "resolved_interaction_endpoint",
            unit.get(role, ""),
        )

    evidence = clean_text(unit.get("evidence"))
    searchable_sources = [
        ("unique_scene_character_surface_in_evidence", evidence)
    ]
    if not seen_ids:
        searchable_sources.append(
            ("unique_scene_character_surface_in_unit_name", clean_text(unit.get("name")))
        )
    for mechanism, source_text in searchable_sources:
        if not source_text:
            continue
        for owners in scene_character_surfaces.values():
            unique_owners = {
                item["entity_id"]: item for item in owners if item.get("entity_id")
            }
            if len(unique_owners) != 1:
                continue
            owner = next(iter(unique_owners.values()))
            matching_surfaces = [
                item["surface"]
                for item in owners
                if _name_surface_occurs_in_text(item["surface"], source_text)
            ]
            if matching_surfaces:
                add(
                    owner["entity_id"],
                    owner["canonical_name"],
                    mechanism,
                    max(matching_surfaces, key=len),
                )

    existing_names = unique_text(unit.get("participants", []))
    unit["participant_entities"] = participant_entities
    unit["participant_entity_ids"] = [
        item["entity_id"] for item in participant_entities
    ]
    unit["participants"] = unique_text(
        [*existing_names, *(item["canonical_name"] for item in additions)]
    )
    if not additions:
        return
    corrections.append(
        {
            "scene_id": scene_id,
            "unit_id": clean_text(unit.get("unit_id")),
            "field": "participants",
            "action": "add_grounded_character_participants",
            "strategy": "resolved_endpoint_or_unique_scene_evidence_surface",
            "additions": additions,
        }
    )


def _name_surface_occurs_in_text(surface: Any, text: Any) -> bool:
    surface_text = clean_text(surface).casefold()
    source_text = clean_text(text).casefold()
    if not surface_text or not source_text:
        return False
    if any(character.isascii() and character.isalpha() for character in surface_text):
        # Keep contractions intact so a character such as Don is not inferred
        # from "don't". Possessive forms remain valid name evidence.
        token_pattern = r"[^\W_]+(?:['’][^\W_]+)*"
        surface_tokens = re.findall(token_pattern, surface_text)
        source_tokens = re.findall(token_pattern, source_text)
        if not surface_tokens or len(surface_tokens) > len(source_tokens):
            return False
        width = len(surface_tokens)
        return any(
            all(
                source_token == surface_token
                or source_token in {f"{surface_token}'s", f"{surface_token}’s"}
                for surface_token, source_token in zip(
                    surface_tokens, source_tokens[start : start + width]
                )
            )
            for start in range(len(source_tokens) - width + 1)
        )
    return normalize_name(surface_text) in normalize_name(source_text)


def _mention_prompt_record(mention: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": mention["name"],
        "entity_type_observation": mention["entity_type"],
        "scope": mention.get("scope", "local"),
        "type_profile": mention["type_profile"],
        "descriptions": mention["descriptions"],
        "declared_aliases": mention["declared_aliases"],
        "source_scene_ids": mention["source_scene_ids"],
        "source_scene_order": mention["source_scene_order"],
        "source_evidence": mention["evidence"],
        "scene_modality_context": mention.get("scene_modality_context", []),
    }


def _scene_modality_context(record: dict[str, Any]) -> list[str]:
    output: list[str] = []
    title = clean_text(record.get("scene", {}).get("title"))
    if title:
        output.append(f"scene: {title}")
    for unit in record.get("narrative_units", []):
        summary = " | ".join(
            value
            for value in (
                clean_text(unit.get("name")),
                clean_text(unit.get("description")),
            )
            if value
        )
        if summary:
            output.append(summary)
    return unique_text(output, limit=12)


def _embedding_text(mention: dict[str, Any]) -> str:
    parts = [
        f"name: {mention['name']}",
        f"type observation: {mention['entity_type']}",
        *[f"alias: {value}" for value in mention["declared_aliases"]],
        *[f"description: {value}" for value in mention["descriptions"]],
    ]
    return "\n".join(parts)


def _name_containment_candidate(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) < 2 or len(shorter) / len(longer) < 0.5:
        return False
    return longer.startswith(shorter) or longer.endswith(shorter)


def _name_token_containment_candidate(left: str, right: str) -> bool:
    left_tokens = tuple(
        token for token in re.findall(r"[^\W_]+", clean_text(left).casefold()) if token
    )
    right_tokens = tuple(
        token for token in re.findall(r"[^\W_]+", clean_text(right).casefold()) if token
    )
    if not left_tokens or not right_tokens or left_tokens == right_tokens:
        return False
    shorter, longer = sorted((left_tokens, right_tokens), key=len)
    if len(shorter) >= len(longer) or any(len(token) < 2 for token in shorter):
        return False
    return set(shorter).issubset(longer)


def _mentions_are_character_compatible(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    return all(
        "Character"
        in set(mention.get("type_profile", {}).get("candidate_primary_kinds") or [])
        for mention in (left, right)
    )


def _mention_key(scene_id: str, normalized_name: str) -> str:
    return f"{scene_id}::{normalized_name}"


def _validation_feedback(error: Exception | None) -> str:
    if error is None:
        return ""
    return (
        "\n\nThe previous output failed strict validation with this error: "
        f"{clean_text(error)}. Return one corrected decision for every supplied pair id."
    )


def _partial_identity_payload(
    payload: Any, *, expected_ids: set[str]
) -> tuple[dict[str, Any], set[str]] | None:
    """Recognize a usable partial response for one bounded repair attempt."""
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        return None
    rows = payload["decisions"]
    if not rows or len(rows) >= len(expected_ids):
        return None
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return None
        pair_id = clean_text(row.get("pair_id"))
        if pair_id not in expected_ids or pair_id in seen:
            return None
        seen.add(pair_id)
    missing = expected_ids - seen
    if not missing:
        return None
    return {"decisions": [dict(row) for row in rows]}, missing


def _partial_identity_repair_feedback(missing_ids: set[str]) -> str:
    ids = ", ".join(sorted(missing_ids))
    return (
        "\n\nTargeted schema repair: the previous response contained valid rows but "
        f"omitted these pair IDs: {ids}. Return a decisions array containing "
        "exactly one decision for each of those missing IDs only. Do not repeat "
        "any other pair ID and do not add a new ID."
    )


def _merge_partial_identity_payload(
    previous: dict[str, Any],
    repair: Any,
    *,
    expected_ids: set[str],
    missing_ids: set[str],
) -> dict[str, Any] | None:
    if not isinstance(repair, dict) or not isinstance(repair.get("decisions"), list):
        return None
    repair_rows = repair["decisions"]
    seen: set[str] = set()
    for row in repair_rows:
        if not isinstance(row, dict):
            return None
        pair_id = clean_text(row.get("pair_id"))
        if pair_id not in missing_ids or pair_id in seen:
            return None
        seen.add(pair_id)
    if seen != missing_ids:
        return None
    previous_rows = previous.get("decisions")
    if not isinstance(previous_rows, list):
        return None
    merged = [*previous_rows, *repair_rows]
    if len(merged) != len(expected_ids):
        return None
    return {"decisions": merged}
