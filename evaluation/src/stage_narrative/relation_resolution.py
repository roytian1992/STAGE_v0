from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .clients import ModelResponseParseError
from .io import atomic_write_json, load_json, sha256_json
from .models import clean_text, stable_id, unique_text
from .prompt_loader import PROMPTS


_PREDICATE_REVIEW_PROMPT = PROMPTS.get("predicate_review")
PREDICATE_REVIEW_SYSTEM = _PREDICATE_REVIEW_PROMPT.system
PREDICATE_REVIEW_USER = _PREDICATE_REVIEW_PROMPT.user
_PREDICATE_SCHEMA_REPAIR_PROMPT = PROMPTS.get("predicate_schema_repair")
PREDICATE_SCHEMA_REPAIR_SYSTEM = _PREDICATE_SCHEMA_REPAIR_PROMPT.system
PREDICATE_SCHEMA_REPAIR_USER = _PREDICATE_SCHEMA_REPAIR_PROMPT.user


class JsonClient(Protocol):
    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


_ACTORS = ["Character", "Organization"]
_PLACES = ["Location", "Organization"]
_REFERENTS = [
    "Character",
    "Location",
    "Occasion",
    "Organization",
    "Object",
    "Concept",
]


PREDICATE_REGISTRY: dict[str, dict[str, Any]] = {
    "about": {
        "description": "has content or a theme about a referent",
        "domain": ["Object", "Concept"],
        "range": _REFERENTS,
    },
    "affiliated_with": {
        "description": "is institutionally affiliated with",
        "domain": _ACTORS,
        "range": ["Organization"],
    },
    "child_of": {
        "description": "is a child of",
        "domain": ["Character"],
        "range": ["Character"],
        "inverse_of": "parent_of",
    },
    "collaborates_with": {
        "description": "has an ongoing collaboration with",
        "domain": _ACTORS,
        "range": _ACTORS,
        "symmetric": True,
    },
    "controls": {
        "description": "exercises durable control over",
        "domain": _ACTORS,
        "range": ["Concept", "Organization", "Location", "Object"],
    },
    "created_by": {
        "description": "was created by",
        "domain": ["Object", "Concept", "Organization"],
        "range": _ACTORS,
    },
    "employs": {
        "description": "employs or formally manages",
        "domain": ["Organization"],
        "range": ["Character"],
        "inverse_of": "works_at",
    },
    "friend_of": {
        "description": "has a friendship with",
        "domain": ["Character"],
        "range": ["Character"],
        "symmetric": True,
    },
    "has_nationality": {
        "description": "has a nationality or national identity",
        "domain": ["Character"],
        "range": ["Location", "Concept"],
    },
    "has_role": {
        "description": "has a durable occupation, title, or social role",
        "domain": ["Character"],
        "range": ["Concept"],
    },
    "knows": {
        "description": "is personally acquainted with",
        "domain": ["Character"],
        "range": ["Character"],
        "symmetric": True,
    },
    "leads": {
        "description": "leads a group or organization",
        "domain": ["Character"],
        "range": ["Organization"],
    },
    "lives_in": {
        "description": "resides in a place",
        "domain": ["Character"],
        "range": _PLACES,
    },
    "located_in": {
        "description": "is geographically or physically located in",
        "domain": ["Character", "Location", "Organization", "Object"],
        "range": _PLACES,
    },
    "loves": {
        "description": "has a romantic or enduring affectionate stance toward",
        "domain": ["Character"],
        "range": ["Character"],
    },
    "member_of": {
        "description": "is a member of a group or organization",
        "domain": _ACTORS,
        "range": ["Organization"],
    },
    "opposes": {
        "description": "has a durable oppositional stance toward",
        "domain": _ACTORS,
        "range": [*_ACTORS, "Concept"],
    },
    "originates_from": {
        "description": "comes from a place, group, or source",
        "domain": ["Character", "Organization", "Object"],
        "range": ["Location", "Organization"],
    },
    "owns": {
        "description": "owns an object, venue, or resource",
        "domain": _ACTORS,
        "range": ["Object", "Location", "Organization"],
    },
    "parent_of": {
        "description": "is a parent of",
        "domain": ["Character"],
        "range": ["Character"],
        "inverse_of": "child_of",
    },
    "part_of": {
        "description": "is a constituent part of",
        "domain": ["Location", "Organization", "Object", "Concept"],
        "range": ["Location", "Organization", "Object", "Concept", "Occasion"],
    },
    "partner_of": {
        "description": "has a durable partnership with",
        "domain": _ACTORS,
        "range": _ACTORS,
        "symmetric": True,
    },
    "possesses": {
        "description": "possesses or holds an object or resource",
        "domain": [*_ACTORS, "Object"],
        "range": ["Object"],
    },
    "pursues": {
        "description": "maintains an ongoing pursuit of",
        "domain": _ACTORS,
        "range": ["Character", "Organization", "Object", "Concept"],
    },
    "resembles": {
        "description": "has a durable physical or visual resemblance to",
        "domain": ["Character"],
        "range": ["Character"],
        "symmetric": True,
    },
    "spouse_of": {
        "description": "is married to",
        "domain": ["Character"],
        "range": ["Character"],
        "symmetric": True,
    },
    "supports": {
        "description": "has a durable supportive stance toward",
        "domain": _ACTORS,
        "range": [*_ACTORS, "Concept"],
    },
    "uses": {
        "description": "habitually or characteristically uses",
        "domain": _ACTORS,
        "range": ["Object"],
    },
    "works_at": {
        "description": "works at or serves an organization or venue",
        "domain": ["Character"],
        "range": _PLACES,
        "inverse_of": "employs",
    },
    "participates_in": {
        "description": "participates in an organized occasion",
        "domain": ["Character", "Organization"],
        "range": ["Occasion"],
    },
    "organized_by": {
        "description": "is organized or hosted by",
        "domain": ["Occasion"],
        "range": ["Character", "Organization"],
    },
    "occurs_at": {
        "description": "takes place at a physical location",
        "domain": ["Occasion"],
        "range": ["Location"],
    },
    "occurs_on": {
        "description": "takes place at a time point",
        "domain": ["Occasion"],
        "range": ["TimePoint"],
    },
}


@dataclass(frozen=True, slots=True)
class RelationResolutionConfig:
    batch_size: int = 16
    semantic_attempts: int = 2
    max_concurrency: int = 4


class RelationResolver:
    def __init__(
        self,
        *,
        movie_id: str,
        llm_client: JsonClient,
        config: RelationResolutionConfig,
        token_counter: TokenCounter | None = None,
        max_input_tokens: int | None = None,
        checkpoint_dir: Path | None = None,
    ):
        self.movie_id = movie_id
        self.llm_client = llm_client
        self.config = config
        self.token_counter = token_counter
        self.max_input_tokens = max_input_tokens
        self.checkpoint_dir = checkpoint_dir.resolve() if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def resolve(self, scene_records: list[dict[str, Any]]) -> dict[str, Any]:
        observations = _collect_observations(scene_records)
        if not observations:
            return {
                "schema_version": "stage_relation_registry_v2",
                "predicate_registry": PREDICATE_REGISTRY,
                "observations": [],
                "canonical_relations": [],
                "audit": {
                    "input_count": 0,
                    "kept_count": 0,
                    "dropped_count": 0,
                    "input_surface_predicate_count": 0,
                    "canonical_predicate_count": 0,
                    "canonical_relation_count": 0,
                    "duplicate_observation_reduction_count": 0,
                    "predicate_domain_range_drop_count": 0,
                    "temporal_observation_count": 0,
                    "modal_observation_count": 0,
                    "contradiction_group_count": 0,
                    "contradiction_groups": [],
                    "decision_calls": [],
                    "decisions": [],
                },
            }

        batches = [
            observations[start : start + max(1, self.config.batch_size)]
            for start in range(0, len(observations), max(1, self.config.batch_size))
        ]
        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def run_batch(batch_index: int, batch: list[dict[str, Any]]):
            async with semaphore:
                return await self._review_batch(batch_index, batch)

        results = await asyncio.gather(
            *(run_batch(index, batch) for index, batch in enumerate(batches))
        )
        results.sort(key=lambda item: item[0])
        decisions = [decision for _, batch_decisions, _ in results for decision in batch_decisions]
        calls = [metadata for _, _, metadata in results]
        decision_by_id = {item["observation_id"]: item for item in decisions}

        normalized_observations: list[dict[str, Any]] = []
        for observation in observations:
            decision = decision_by_id[observation["observation_id"]]
            normalized = {**observation, "normalization": decision}
            if decision["action"] == "keep":
                source = observation["subject_entity_id"]
                target = observation["object_entity_id"]
                if source == target:
                    raise ValueError(
                        f"Predicate review retained a self relation: "
                        f"{observation['observation_id']}"
                    )
                if decision["reverse_direction"]:
                    source, target = target, source
                predicate_id = decision["predicate_id"]
                if PREDICATE_REGISTRY[predicate_id].get("symmetric") and source > target:
                    source, target = target, source
                qualifiers = dict(observation.get("qualifiers", {}))
                if decision["relation_class"] == "temporal":
                    qualifiers["valid_at_scene_id"] = observation["source_scene_id"]
                else:
                    qualifiers.pop("valid_at_scene_id", None)
                normalized.update(
                    {
                        "canonical_subject_entity_id": source,
                        "canonical_predicate_id": predicate_id,
                        "canonical_object_entity_id": target,
                        "relation_class": decision["relation_class"],
                        "qualifiers": qualifiers,
                    }
                )
            normalized_observations.append(normalized)

        canonical_relations = _aggregate_relations(
            self.movie_id, normalized_observations
        )
        contradiction_groups = find_contradiction_groups(canonical_relations)
        return {
            "schema_version": "stage_relation_registry_v2",
            "predicate_registry": PREDICATE_REGISTRY,
            "observations": normalized_observations,
            "canonical_relations": canonical_relations,
            "audit": {
                "input_count": len(observations),
                "kept_count": sum(item["action"] == "keep" for item in decisions),
                "dropped_count": sum(item["action"] == "drop" for item in decisions),
                "input_surface_predicate_count": len(
                    {item["surface_predicate"] for item in observations}
                ),
                "canonical_predicate_count": len(
                    {item["predicate_id"] for item in decisions if item["action"] == "keep"}
                ),
                "canonical_relation_count": len(canonical_relations),
                "duplicate_observation_reduction_count": max(
                    0,
                    sum(item["action"] == "keep" for item in decisions)
                    - len(canonical_relations),
                ),
                "predicate_domain_range_drop_count": sum(
                    item.get("normalization_reason")
                    == "predicate_domain_range_mismatch"
                    for item in decisions
                ),
                "temporal_observation_count": sum(
                    item.get("normalization", {}).get("action") == "keep"
                    and item.get("relation_class") == "temporal"
                    for item in normalized_observations
                ),
                "modal_observation_count": sum(
                    item.get("normalization", {}).get("action") == "keep"
                    and item.get("qualifiers", {}).get("modality") != "asserted"
                    for item in normalized_observations
                ),
                "contradiction_group_count": len(contradiction_groups),
                "contradiction_groups": contradiction_groups,
                "decision_calls": calls,
                "decisions": decisions,
                "note": (
                    "generated_rationale_hint is model-generated audit context, "
                    "not screenplay evidence"
                ),
            },
        }

    async def _review_batch(
        self, batch_index: int, batch: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        prompt_rows = [
            {
                "observation_id": item["observation_id"],
                "subject": item["subject_name"],
                "subject_primary_kind": item["subject_primary_kind"],
                "subject_entity_types": item.get("subject_entity_types", []),
                "subject_facets": item.get("subject_facets", []),
                "surface_predicate": item["surface_predicate"],
                "object": item["object_name"],
                "object_primary_kind": item["object_primary_kind"],
                "object_entity_types": item.get("object_entity_types", []),
                "object_facets": item.get("object_facets", []),
                "relation_class_observation": item["relation_class"],
                "qualifiers": item.get("qualifiers", {}),
                "description": item["description"],
                "source_scene_id": item["source_scene_id"],
                "source_evidence": item["evidence"],
            }
            for item in batch
        ]
        user_prompt = PREDICATE_REVIEW_USER.format(
            predicate_registry=json.dumps(PREDICATE_REGISTRY, ensure_ascii=False, indent=2),
            relation_observations=json.dumps(prompt_rows, ensure_ascii=False, indent=2),
        )
        expected_ids = {item["observation_id"] for item in batch}
        prompt_sha256 = sha256_json(
            {
                "system_prompt": PREDICATE_REVIEW_SYSTEM,
                "user_prompt": user_prompt,
            }
        )
        checkpoint_path = (
            self.checkpoint_dir / f"{batch_index:04d}.json"
            if self.checkpoint_dir is not None
            else None
        )
        if checkpoint_path is not None and checkpoint_path.is_file():
            checkpoint = load_json(checkpoint_path)
            if (
                checkpoint.get("batch_index") != batch_index
                or checkpoint.get("prompt_sha256") != prompt_sha256
                or set(checkpoint.get("expected_observation_ids", [])) != expected_ids
            ):
                # Treat drift as a stale cache and regenerate only this batch;
                # the strict response validator still guards the new result.
                checkpoint = None
            if checkpoint is not None:
                decisions = _validate_predicate_decisions(
                    checkpoint["raw_response"],
                    expected_ids,
                    self_relation_ids={
                        item["observation_id"]
                        for item in batch
                        if item["subject_entity_id"] == item["object_entity_id"]
                    },
                    observations_by_id={
                        item["observation_id"]: item for item in batch
                    },
                )
                return batch_index, decisions, {
                    **checkpoint["generator_metadata"],
                    "checkpoint_reused": True,
                }

        measured_tokens = self._prompt_tokens(
            PREDICATE_REVIEW_SYSTEM,
            user_prompt,
            f"predicate_review:{batch_index:04d}",
        )
        expected_validation_kwargs = {
            "expected_ids": expected_ids,
            "self_relation_ids": {
                item["observation_id"]
                for item in batch
                if item["subject_entity_id"] == item["object_entity_id"]
            },
            "observations_by_id": {
                item["observation_id"]: item for item in batch
            },
        }
        try:
            call = await self.llm_client.generate_json(
                system_prompt=PREDICATE_REVIEW_SYSTEM,
                user_prompt=user_prompt,
                stage=f"predicate_review:{batch_index:04d}",
            )
            decisions = _validate_predicate_decisions(
                call.data, **expected_validation_kwargs
            )
            schema_audit = _predicate_decision_schema_audit(call.data, expected_ids)
            metadata = {
                **call.metadata,
                "semantic_attempt": 1,
                "call_kind": "formal_review",
                "prompt_tokens_measured": measured_tokens,
                "schema_normalization": schema_audit,
                "deterministic_schema_drops": sum(
                    bool(item.get("normalization_reason")) for item in decisions
                ),
            }
            raw_response = call.data
        except ModelResponseParseError as parse_error:
            repair_items = json.dumps(
                {
                    "parse_error": clean_text(parse_error),
                    "raw_response": parse_error.raw_text,
                    "required_observation_ids": sorted(expected_ids),
                },
                ensure_ascii=False,
                indent=2,
            )
            repair_prompt = PREDICATE_SCHEMA_REPAIR_USER.format(
                predicate_registry=json.dumps(
                    PREDICATE_REGISTRY, ensure_ascii=False, indent=2
                ),
                repair_items=repair_items,
            )
            repair_tokens = self._prompt_tokens(
                PREDICATE_SCHEMA_REPAIR_SYSTEM,
                repair_prompt,
                f"predicate_schema_repair:{batch_index:04d}",
            )
            repair_call = await self.llm_client.generate_json(
                system_prompt=PREDICATE_SCHEMA_REPAIR_SYSTEM,
                user_prompt=repair_prompt,
                stage=f"predicate_schema_repair:{batch_index:04d}",
            )
            decisions = _validate_predicate_decisions(
                repair_call.data, **expected_validation_kwargs
            )
            schema_audit = _predicate_decision_schema_audit(
                repair_call.data, expected_ids
            )
            metadata = {
                **repair_call.metadata,
                "semantic_attempt": 1,
                "call_kind": "targeted_schema_repair",
                "prompt_tokens_measured": repair_tokens,
                "formal_prompt_tokens_measured": measured_tokens,
                "formal_parse_error": clean_text(parse_error),
                "formal_call_metadata": parse_error.metadata,
                "repair_attempt": 1,
                "schema_normalization": schema_audit,
                "deterministic_schema_drops": sum(
                    bool(item.get("normalization_reason")) for item in decisions
                ),
            }
            raw_response = repair_call.data
        if checkpoint_path is not None:
            atomic_write_json(
                checkpoint_path,
                {
                    "schema_version": "stage_predicate_review_batch_checkpoint_v1",
                    "batch_index": batch_index,
                    "prompt_sha256": prompt_sha256,
                    "expected_observation_ids": sorted(expected_ids),
                    "raw_response": raw_response,
                    "normalized_decisions": decisions,
                    "generator_metadata": metadata,
                },
            )
        return batch_index, decisions, metadata

    def _prompt_tokens(
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


def _collect_observations(scene_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entity_info: dict[str, dict[str, Any]] = {}
    for record in scene_records:
        for entity in record.get("entities", []):
            entity_id = clean_text(entity.get("canonical_entity_id"))
            if entity_id:
                entity_info.setdefault(
                    entity_id,
                    {
                        "name": clean_text(entity.get("canonical_name")),
                        "primary_kind": clean_text(
                            entity.get("primary_kind") or entity.get("entity_type")
                        ),
                        "entity_types": unique_text(
                            entity.get("entity_types")
                            or [entity.get("entity_type")]
                        ),
                        "facets": sorted(
                            clean_text(value)
                            for value in (entity.get("facets") or [])
                            if clean_text(value)
                        ),
                    },
                )
    observations: list[dict[str, Any]] = []
    for record in scene_records:
        for relation in record.get("entity_relations", []):
            relation_id = clean_text(relation.get("relation_id"))
            subject_id = clean_text(relation.get("subject_entity_id"))
            object_id = clean_text(relation.get("object_entity_id"))
            if not relation_id or not subject_id or not object_id:
                continue
            subject_info = entity_info.get(
                subject_id,
                {
                    "name": clean_text(relation.get("subject")),
                    "primary_kind": "",
                    "entity_types": [],
                    "facets": [],
                },
            )
            object_info = entity_info.get(
                object_id,
                {
                    "name": clean_text(relation.get("object")),
                    "primary_kind": "",
                    "entity_types": [],
                    "facets": [],
                },
            )
            relation_class = clean_text(relation.get("relation_class")) or "stable"
            polarity = clean_text(relation.get("polarity"))
            if polarity not in {"positive", "negative"}:
                polarity = "positive"
            modality = clean_text(relation.get("modality"))
            if modality not in {
                "asserted",
                "remembered",
                "dreamed",
                "hallucinated",
                "hypothetical",
                "reported",
                "uncertain",
            }:
                modality = "asserted"
            source_scene_id = clean_text(relation.get("source_scene_id"))
            qualifiers = {
                "polarity": polarity,
                "modality": modality,
            }
            if relation_class == "temporal":
                qualifiers["valid_at_scene_id"] = source_scene_id
            observations.append(
                {
                    "observation_id": stable_id("relation-observation", relation_id),
                    "source_relation_id": relation_id,
                    "subject_entity_id": subject_id,
                    "subject_name": subject_info["name"],
                    "subject_primary_kind": subject_info["primary_kind"],
                    "subject_entity_types": subject_info["entity_types"],
                    "subject_facets": subject_info["facets"],
                    "surface_predicate": clean_text(relation.get("predicate")),
                    "object_entity_id": object_id,
                    "object_name": object_info["name"],
                    "object_primary_kind": object_info["primary_kind"],
                    "object_entity_types": object_info["entity_types"],
                    "object_facets": object_info["facets"],
                    "relation_class": relation_class,
                    "description": clean_text(relation.get("description")),
                    "evidence": clean_text(relation.get("evidence")),
                    "source_scene_id": source_scene_id,
                    "source_scene_order": int(record.get("scene", {}).get("order", 0)),
                    "qualifiers": qualifiers,
                }
            )
    return observations


def _validate_predicate_decisions(
    payload: dict[str, Any],
    expected_ids: set[str],
    self_relation_ids: set[str] | None = None,
    observations_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
        raise ValueError("Predicate review must return exactly a decisions array")
    normalized_rows = _normalize_predicate_decision_rows(payload, expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    allowed_reasons = {
        "supported",
        "event_like",
        "unsupported",
        "redundant",
        "malformed",
        "type_mismatch",
    }
    for raw in normalized_rows:
        if not isinstance(raw, dict):
            raise ValueError("Predicate decision must be an object")
        observation_id = clean_text(raw.get("observation_id"))
        if observation_id not in expected_ids or observation_id in by_id:
            raise ValueError(f"Invalid normalized observation id: {observation_id}")
        action = clean_text(raw.get("action"))
        predicate_id = clean_text(raw.get("predicate_id"))
        raw_predicate_id = predicate_id
        relation_class = clean_text(raw.get("relation_class"))
        reason_code = clean_text(raw.get("reason_code"))
        raw_reason_code = reason_code
        reverse_direction = raw.get("reverse_direction")
        normalization_reason = clean_text(raw.get("_schema_correction"))
        if action not in {"keep", "drop"}:
            raise ValueError(f"Unsupported predicate action for {observation_id}: {action}")
        if not isinstance(reverse_direction, bool):
            raise ValueError(f"reverse_direction must be boolean for {observation_id}")
        if reason_code not in allowed_reasons:
            reason_code = (
                "supported"
                if action == "keep" and relation_class in {"stable", "temporal"}
                else "event_like"
                if relation_class == "event_like"
                else "unsupported"
            )
        if action == "keep":
            if observation_id in (self_relation_ids or set()):
                action = "drop"
                predicate_id = ""
                relation_class = "event_like"
                reason_code = "malformed"
            elif predicate_id not in PREDICATE_REGISTRY:
                action = "drop"
                predicate_id = ""
                relation_class = "event_like"
                reason_code = "unsupported"
            elif not predicate_allows_observation(
                predicate_id,
                (observations_by_id or {}).get(observation_id, {}),
                reverse_direction=reverse_direction,
            ):
                action = "drop"
                predicate_id = ""
                relation_class = "event_like"
                reason_code = "type_mismatch"
                normalization_reason = "predicate_domain_range_mismatch"
            elif relation_class not in {"stable", "temporal"}:
                action = "drop"
                predicate_id = ""
                relation_class = "event_like"
                reason_code = "event_like"
        elif predicate_id:
            predicate_id = ""
        by_id[observation_id] = {
            "observation_id": observation_id,
            "action": action,
            "predicate_id": predicate_id,
            "raw_predicate_id": raw_predicate_id,
            "reverse_direction": reverse_direction,
            "relation_class": relation_class or "event_like",
            "reason_code": reason_code,
            "raw_reason_code": raw_reason_code,
            "generated_rationale_hint": clean_text(raw.get("generated_rationale_hint")),
            "normalization_reason": normalization_reason,
        }
    if set(by_id) != expected_ids:
        raise ValueError(f"Missing predicate decisions: {sorted(expected_ids - set(by_id))}")
    return [by_id[item] for item in sorted(by_id)]


def _normalize_predicate_decision_rows(
    payload: dict[str, Any], expected_ids: set[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        item: [] for item in expected_ids
    }
    for raw in payload["decisions"]:
        if not isinstance(raw, dict):
            raise ValueError("Predicate decision must be an object")
        observation_id = clean_text(raw.get("observation_id"))
        if observation_id in grouped:
            grouped[observation_id].append(raw)
    output = []
    for observation_id in sorted(expected_ids):
        rows = grouped[observation_id]
        if not rows:
            output.append(
                _fail_closed_predicate_decision(
                    observation_id,
                    correction="missing_model_decision_fail_closed",
                )
            )
            continue
        if len(rows) == 1:
            output.append(dict(rows[0]))
            continue
        comparable = [
            {key: value for key, value in row.items() if key != "generated_rationale_hint"}
            for row in rows
        ]
        if all(row == comparable[0] for row in comparable[1:]):
            collapsed = dict(rows[0])
            collapsed["_schema_correction"] = "exact_duplicate_decision_collapsed"
            output.append(collapsed)
        else:
            output.append(
                _fail_closed_predicate_decision(
                    observation_id,
                    correction="conflicting_duplicate_decisions_fail_closed",
                )
            )
    return output


def _fail_closed_predicate_decision(
    observation_id: str, *, correction: str
) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "action": "drop",
        "predicate_id": "",
        "reverse_direction": False,
        "relation_class": "event_like",
        "reason_code": "unsupported",
        "generated_rationale_hint": correction,
        "_schema_correction": correction,
    }


def _predicate_decision_schema_audit(
    payload: dict[str, Any], expected_ids: set[str]
) -> dict[str, Any]:
    raw_ids = [
        clean_text(item.get("observation_id"))
        for item in payload.get("decisions", [])
        if isinstance(item, dict)
    ]
    known_ids = [item for item in raw_ids if item in expected_ids]
    duplicate_ids = sorted(
        {item for item in known_ids if known_ids.count(item) > 1}
    )
    return {
        "expected_count": len(expected_ids),
        "raw_decision_count": len(raw_ids),
        "unknown_decision_ids": sorted(
            {item for item in raw_ids if item not in expected_ids}
        ),
        "duplicate_decision_ids": duplicate_ids,
        "missing_decision_ids": sorted(expected_ids - set(known_ids)),
        "policy": (
            "drop unknown; collapse exact duplicates; fail-close conflicting duplicates "
            "and missing decisions"
        ),
    }


def _aggregate_relations(
    movie_id: str, observations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for item in observations:
        if item["normalization"]["action"] != "keep":
            continue
        key = (
            item["canonical_subject_entity_id"],
            item["canonical_predicate_id"],
            item["canonical_object_entity_id"],
            item["relation_class"],
            json.dumps(item.get("qualifiers", {}), ensure_ascii=False, sort_keys=True),
        )
        grouped.setdefault(key, []).append(item)
    output: list[dict[str, Any]] = []
    for (
        subject,
        predicate,
        object_id,
        relation_class,
        qualifier_signature,
    ), rows in sorted(grouped.items()):
        qualifiers = json.loads(qualifier_signature)
        output.append(
            {
                "relation_id": stable_id(
                    "canonical-relation",
                    movie_id,
                    subject,
                    predicate,
                    object_id,
                    relation_class,
                    qualifier_signature,
                ),
                "subject_entity_id": subject,
                "predicate_id": predicate,
                "object_entity_id": object_id,
                "relation_class": relation_class,
                "status": (
                    "negated"
                    if qualifiers.get("polarity") == "negative"
                    else "uncertain"
                    if qualifiers.get("modality") == "uncertain"
                    else "asserted"
                ),
                "qualifiers": qualifiers,
                "surface_predicates": unique_text(
                    row["surface_predicate"] for row in rows
                ),
                "source_observation_ids": [row["observation_id"] for row in rows],
                "source_scene_ids": sorted({row["source_scene_id"] for row in rows}),
                "source_scene_orders": sorted(
                    {int(row.get("source_scene_order", 0)) for row in rows}
                ),
                "evidence_items": [
                    {
                        "observation_id": row["observation_id"],
                        "source_scene_id": row["source_scene_id"],
                        "evidence": row["evidence"],
                        "description": row["description"],
                    }
                    for row in rows
                ],
            }
        )
    return output


def predicate_allows_observation(
    predicate_id: str,
    observation: dict[str, Any],
    *,
    reverse_direction: bool = False,
) -> bool:
    spec = PREDICATE_REGISTRY.get(predicate_id)
    if not spec or not observation:
        return False
    subject = {
        "primary_kind": clean_text(observation.get("subject_primary_kind")),
        "entity_types": observation.get("subject_entity_types") or [],
        "facets": observation.get("subject_facets") or [],
    }
    object_profile = {
        "primary_kind": clean_text(observation.get("object_primary_kind")),
        "entity_types": observation.get("object_entity_types") or [],
        "facets": observation.get("object_facets") or [],
    }
    if reverse_direction:
        subject, object_profile = object_profile, subject
    return _profile_matches(subject, spec.get("domain", [])) and _profile_matches(
        object_profile, spec.get("range", [])
    )


def _profile_matches(profile: dict[str, Any], allowed: list[str]) -> bool:
    observed = {
        clean_text(profile.get("primary_kind")),
        *(clean_text(value) for value in (profile.get("entity_types") or [])),
        *(clean_text(value) for value in (profile.get("facets") or [])),
    }
    observed.discard("")
    return bool(observed & set(allowed))


def find_contradiction_groups(
    canonical_relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for relation in canonical_relations:
        qualifiers = relation.get("qualifiers", {})
        key = (
            clean_text(relation.get("subject_entity_id")),
            clean_text(relation.get("predicate_id")),
            clean_text(relation.get("object_entity_id")),
            clean_text(qualifiers.get("modality")) or "asserted",
            clean_text(qualifiers.get("valid_at_scene_id")) or "stable",
        )
        grouped.setdefault(key, []).append(relation)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        polarities = {
            clean_text(row.get("qualifiers", {}).get("polarity")) or "positive"
            for row in rows
        }
        if {"positive", "negative"}.issubset(polarities):
            output.append(
                {
                    "subject_entity_id": key[0],
                    "predicate_id": key[1],
                    "object_entity_id": key[2],
                    "modality": key[3],
                    "valid_at_scene_id": None if key[4] == "stable" else key[4],
                    "relation_ids": sorted(row["relation_id"] for row in rows),
                    "polarities": sorted(polarities),
                }
            )
    return output


def _validation_feedback(error: Exception | None) -> str:
    if error is None:
        return ""
    return (
        "\n\nThe previous output failed strict validation with this error: "
        f"{clean_text(error)}. Return one corrected decision for every observation id."
    )
