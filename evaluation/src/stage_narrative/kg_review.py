from __future__ import annotations

import copy
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol

from .models import (
    clean_text,
    normalize_name,
    require_entity_scope,
    require_entity_type,
    stable_id,
    unique_text,
)
from .prompt_loader import PROMPTS
from .prompt_assets import entity_type_definitions
from .type_resolution import initial_type_profile


_SCENE_GRAPH_REVIEW_PROMPT = PROMPTS.get("scene_graph_review")
SCENE_GRAPH_REVIEW_SYSTEM = _SCENE_GRAPH_REVIEW_PROMPT.system
SCENE_GRAPH_REVIEW_USER = _SCENE_GRAPH_REVIEW_PROMPT.user


class JsonClient(Protocol):
    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class SceneReviewConfig:
    language: str
    semantic_attempts: int = 2
    max_entities_per_pack: int = 8
    max_relations_per_pack: int = 8


class SceneGraphReviewer:
    def __init__(
        self,
        *,
        movie_id: str,
        llm_client: JsonClient,
        token_counter: TokenCounter,
        max_input_tokens: int,
        config: SceneReviewConfig,
    ):
        self.movie_id = movie_id
        self.llm_client = llm_client
        self.token_counter = token_counter
        self.max_input_tokens = max_input_tokens
        self.review_input_budget = max(1, max_input_tokens - 512)
        self.config = config

    async def review_scene(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_payload = _build_review_payload(record)
        calls: list[dict[str, Any]] = []
        prompt_token_counts: list[int] = []
        entity_packs = _pack_entity_rows(
            rows=prompt_payload["entities"],
            build_prompt=lambda rows: _format_review_prompt(
                language=self.config.language,
                scene_text=prompt_payload["review_context"],
                entity_context=[],
                entities=rows,
                relations=[],
                references=[],
            ),
            fits=lambda prompt: self._prompt_tokens(prompt) <= self.review_input_budget,
            max_rows=max(1, self.config.max_entities_per_pack),
        )
        entity_decisions: list[dict[str, Any]] = []
        for pack_index, entity_pack in enumerate(entity_packs):
            user_prompt = _format_review_prompt(
                language=self.config.language,
                scene_text=prompt_payload["review_context"],
                entity_context=[],
                entities=entity_pack,
                relations=[],
                references=[],
            )
            part, metadata, prompt_tokens = await self._call_pack(
                record=record,
                user_prompt=user_prompt,
                expected_entity_ids={item["entity_id"] for item in entity_pack},
                expected_relation_ids=set(),
                expected_reference_ids=set(),
                entity_specs={item["entity_id"]: item for item in entity_pack},
                relation_specs={},
                reference_specs={},
                allowed_entity_names=set(),
                stage_suffix=f"entities:{pack_index:04d}",
            )
            entity_decisions.extend(part["entity_decisions"])
            calls.append(metadata)
            prompt_token_counts.append(prompt_tokens)

        preview_payload = {
            "entity_decisions": entity_decisions,
            "relation_decisions": [
                _drop_relation_decision(local_id)
                for local_id in prompt_payload["local_relations"]
            ],
            "reference_decisions": [
                _drop_reference_decision(local_id)
                for local_id in prompt_payload["local_references"]
            ],
        }
        preview, _ = _apply_review_payload(
            record=record,
            payload=preview_payload,
            movie_id=self.movie_id,
            local_entities=prompt_payload["local_entities"],
            local_relations=prompt_payload["local_relations"],
            local_references=prompt_payload["local_references"],
            source_text=prompt_payload["scene_text"],
        )
        entity_context = [
            {
                "name": item["name"],
                "entity_type": item["entity_type"],
                "aliases": item.get("aliases", []),
            }
            for item in preview["entities"]
        ]
        canonical_by_name = _canonical_names(preview["entities"])
        reference_decisions = [
            _resolve_reference_deterministically(reference, canonical_by_name)
            for reference in prompt_payload["references"]
        ]

        mixed_rows = [("relation", item) for item in prompt_payload["relations"]]
        mixed_packs = _pack_mixed_rows(
            rows=mixed_rows,
            build_prompt=lambda relations, references: _format_review_prompt(
                language=self.config.language,
                scene_text=prompt_payload["review_context"],
                entity_context=entity_context,
                entities=[],
                relations=relations,
                references=references,
            ),
            fits=lambda prompt: self._prompt_tokens(prompt) <= self.review_input_budget,
            max_rows=max(1, self.config.max_relations_per_pack),
        )
        relation_decisions: list[dict[str, Any]] = []
        for pack_index, (relation_pack, reference_pack) in enumerate(mixed_packs):
            user_prompt = _format_review_prompt(
                language=self.config.language,
                scene_text=prompt_payload["review_context"],
                entity_context=entity_context,
                entities=[],
                relations=relation_pack,
                references=reference_pack,
            )
            part, metadata, prompt_tokens = await self._call_pack(
                record=record,
                user_prompt=user_prompt,
                expected_entity_ids=set(),
                expected_relation_ids={item["relation_id"] for item in relation_pack},
                expected_reference_ids={item["reference_id"] for item in reference_pack},
                entity_specs={},
                relation_specs={item["relation_id"]: item for item in relation_pack},
                reference_specs={
                    item["reference_id"]: prompt_payload["local_references"][
                        item["reference_id"]
                    ]
                    for item in reference_pack
                },
                allowed_entity_names=set(canonical_by_name),
                stage_suffix=f"relations_references:{pack_index:04d}",
            )
            relation_decisions.extend(part["relation_decisions"])
            reference_decisions.extend(part["reference_decisions"])
            calls.append(metadata)
            prompt_token_counts.append(prompt_tokens)
        payload = {
            "entity_decisions": entity_decisions,
            "relation_decisions": relation_decisions,
            "reference_decisions": reference_decisions,
        }

        reviewed, summary = _apply_review_payload(
            record=record,
            payload=payload,
            movie_id=self.movie_id,
            local_entities=prompt_payload["local_entities"],
            local_relations=prompt_payload["local_relations"],
            local_references=prompt_payload["local_references"],
            source_text=prompt_payload["scene_text"],
        )
        audit = {
            "schema_version": "stage_scene_graph_review_audit_v1",
            "scene_id": record["scene"]["scene_id"],
            "minimal_review_policy": True,
            "packing_enabled": len(prompt_token_counts) > 2,
            "prompt_pack_count": len(prompt_token_counts),
            "max_prompt_tokens": max(prompt_token_counts, default=0),
            "prompt_token_counts": prompt_token_counts,
            "input_entity_count": len(prompt_payload["entities"]),
            "input_relation_count": len(prompt_payload["relations"]),
            "input_reference_count": len(prompt_payload["references"]),
            "reference_map_count": sum(
                item["action"] == "map" for item in reference_decisions
            ),
            "reference_drop_count": sum(
                item["action"] == "drop" for item in reference_decisions
            ),
            **summary,
            "llm_calls": calls,
        }
        reviewed["scene_graph_review_audit"] = {
            key: value for key, value in audit.items() if key != "llm_calls"
        }
        return reviewed, audit

    def _prompt_tokens(self, user_prompt: str) -> int:
        return self.token_counter.count(SCENE_GRAPH_REVIEW_SYSTEM + "\n" + user_prompt)

    async def _call_pack(
        self,
        *,
        record: dict[str, Any],
        user_prompt: str,
        expected_entity_ids: set[str],
        expected_relation_ids: set[str],
        expected_reference_ids: set[str],
        entity_specs: dict[str, dict[str, Any]],
        relation_specs: dict[str, dict[str, Any]],
        reference_specs: dict[str, dict[str, Any]],
        allowed_entity_names: set[str],
        stage_suffix: str,
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        prompt_tokens = self._prompt_tokens(user_prompt)
        if prompt_tokens > self.review_input_budget:
            raise ValueError(
                f"Scene review pack exceeds input budget for {record['scene']['scene_id']}: "
                f"{prompt_tokens}>{self.review_input_budget}"
            )
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.config.semantic_attempts) + 1):
            try:
                complete_prompt = user_prompt + _validation_feedback(last_error)
                prompt_tokens = self._prompt_tokens(complete_prompt)
                if prompt_tokens > self.review_input_budget:
                    raise ValueError(
                        f"Scene review retry pack exceeds input budget for "
                        f"{record['scene']['scene_id']}: "
                        f"{prompt_tokens}>{self.review_input_budget}"
                    )
                call = await self.llm_client.generate_json(
                    system_prompt=SCENE_GRAPH_REVIEW_SYSTEM,
                    user_prompt=complete_prompt,
                    stage=(
                        f"scene_graph_review:{record['scene']['scene_id']}:"
                        f"{stage_suffix}"
                    ),
                )
                (
                    filtered_payload,
                    ignored_extraneous_decisions,
                    duplicate_decision_drops,
                ) = (
                    _filter_extraneous_pack_decisions(
                        call.data,
                        expected_entity_ids=expected_entity_ids,
                        expected_relation_ids=expected_relation_ids,
                        expected_reference_ids=expected_reference_ids,
                    )
                )
                completed_payload, missing_decision_defaults = (
                    _complete_missing_pack_decisions(
                        filtered_payload,
                        expected_entity_ids=expected_entity_ids,
                        expected_relation_ids=expected_relation_ids,
                        expected_reference_ids=expected_reference_ids,
                        entity_specs=entity_specs,
                    )
                )
                normalized_payload, deterministic_relation_drops = (
                    _drop_invalid_retained_relations(
                        completed_payload,
                        source_text=_scene_source_text(record),
                        allowed_entity_names=allowed_entity_names,
                    )
                )
                normalized_payload, evidence_corrections = (
                    _align_review_entity_evidence(
                        normalized_payload,
                        record=record,
                        source_text=_scene_source_text(record),
                    )
                )
                normalized_payload, replacement_cardinality_corrections = (
                    _normalize_entity_replacement_cardinality(normalized_payload)
                )
                _validate_partial_review_payload(
                    normalized_payload,
                    expected_entity_ids=expected_entity_ids,
                    expected_relation_ids=expected_relation_ids,
                    expected_reference_ids=expected_reference_ids,
                )
                _validate_partial_review_semantics(
                    normalized_payload,
                    source_text=_scene_source_text(record),
                    reference_specs=reference_specs,
                )
                return normalized_payload, {
                    **call.metadata,
                    "semantic_attempt": attempt,
                    "prompt_tokens_measured": prompt_tokens,
                    "deterministic_invalid_relation_drops": deterministic_relation_drops,
                    "deterministic_entity_evidence_corrections": evidence_corrections,
                    "deterministic_entity_evidence_correction_count": len(
                        evidence_corrections
                    ),
                    "deterministic_entity_replacement_cardinality_corrections": (
                        replacement_cardinality_corrections
                    ),
                    "deterministic_entity_replacement_cardinality_correction_count": len(
                        replacement_cardinality_corrections
                    ),
                    "ignored_extraneous_pack_decisions": ignored_extraneous_decisions,
                    "deterministic_duplicate_decision_drops": duplicate_decision_drops,
                    "deterministic_missing_decision_defaults": missing_decision_defaults,
                    "deterministic_missing_decision_default_count": sum(
                        len(values) for values in missing_decision_defaults.values()
                    ),
                }, prompt_tokens
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Scene graph review pack failed semantic validation for "
            f"{record['scene']['scene_id']}:{stage_suffix}: {last_error}"
        ) from last_error


def _format_review_prompt(
    *,
    language: str,
    scene_text: str,
    entity_context: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> str:
    _, user_prompt = PROMPTS.render(
        "scene_graph_review",
        language=language,
        type_definitions=entity_type_definitions(),
        scene_text=scene_text,
        entity_context=json.dumps(entity_context, ensure_ascii=False, indent=2),
        entities=json.dumps(entities, ensure_ascii=False, indent=2),
        relations=json.dumps(relations, ensure_ascii=False, indent=2),
        references=json.dumps(references, ensure_ascii=False, indent=2),
    )
    return user_prompt


def _pack_entity_rows(
    *,
    rows: list[dict[str, Any]],
    build_prompt: Any,
    fits: Any,
    max_rows: int,
) -> list[list[dict[str, Any]]]:
    packs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if current and len(current) >= max_rows:
            packs.append(current)
            current = []
        trial = [*current, row]
        if fits(build_prompt(trial)):
            current = trial
            continue
        if not current:
            raise ValueError(f"One entity review row exceeds input budget: {row['entity_id']}")
        packs.append(current)
        current = [row]
        if not fits(build_prompt(current)):
            raise ValueError(f"One entity review row exceeds input budget: {row['entity_id']}")
    if current:
        packs.append(current)
    return packs


def _pack_mixed_rows(
    *,
    rows: list[tuple[str, dict[str, Any]]],
    build_prompt: Any,
    fits: Any,
    max_rows: int,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    packs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    relations: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for kind, row in rows:
        if relations or references:
            if len(relations) + len(references) >= max_rows:
                packs.append((relations, references))
                relations, references = [], []
        trial_relations = [*relations, row] if kind == "relation" else list(relations)
        trial_references = [*references, row] if kind == "reference" else list(references)
        if fits(build_prompt(trial_relations, trial_references)):
            relations, references = trial_relations, trial_references
            continue
        if not relations and not references:
            local_id = row.get("relation_id") or row.get("reference_id")
            raise ValueError(f"One relation/reference review row exceeds input budget: {local_id}")
        packs.append((relations, references))
        relations = [row] if kind == "relation" else []
        references = [row] if kind == "reference" else []
        if not fits(build_prompt(relations, references)):
            local_id = row.get("relation_id") or row.get("reference_id")
            raise ValueError(f"One relation/reference review row exceeds input budget: {local_id}")
    if relations or references:
        packs.append((relations, references))
    return packs


def _complete_missing_pack_decisions(
    payload: dict[str, Any],
    *,
    expected_entity_ids: set[str],
    expected_relation_ids: set[str],
    expected_reference_ids: set[str],
    entity_specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    normalized = copy.deepcopy(payload)
    if set(normalized) != {
        "entity_decisions", "relation_decisions", "reference_decisions"
    }:
        return normalized, {"entity": [], "relation": [], "reference": []}
    if any(not isinstance(normalized[key], list) for key in normalized):
        return normalized, {"entity": [], "relation": [], "reference": []}

    actual_entities = {
        clean_text(item.get("source_entity_id"))
        for item in normalized["entity_decisions"]
        if isinstance(item, dict)
    }
    actual_relations = {
        clean_text(item.get("source_relation_id"))
        for item in normalized["relation_decisions"]
        if isinstance(item, dict)
    }
    actual_references = {
        clean_text(item.get("reference_id"))
        for item in normalized["reference_decisions"]
        if isinstance(item, dict)
    }
    missing_entities = sorted(expected_entity_ids - actual_entities)
    missing_relations = sorted(expected_relation_ids - actual_relations)
    missing_references = sorted(expected_reference_ids - actual_references)
    for local_id in missing_entities:
        spec = entity_specs[local_id]
        normalized["entity_decisions"].append(
            {
                "source_entity_id": local_id,
                "action": "keep",
                "entities": [
                    {
                        key: copy.deepcopy(value)
                        for key, value in spec.items()
                        if key != "entity_id"
                    }
                ],
                "reason_code": "review_omitted_default_keep",
                "generated_rationale_hint": "",
            }
        )
    for local_id in missing_relations:
        decision = _drop_relation_decision(local_id)
        decision["reason_code"] = "review_omitted_default_drop"
        normalized["relation_decisions"].append(decision)
    for local_id in missing_references:
        decision = _drop_reference_decision(local_id)
        decision["reason_code"] = "review_omitted_default_drop"
        normalized["reference_decisions"].append(decision)
    return normalized, {
        "entity": missing_entities,
        "relation": missing_relations,
        "reference": missing_references,
    }


def _validate_partial_review_payload(
    payload: dict[str, Any],
    *,
    expected_entity_ids: set[str],
    expected_relation_ids: set[str],
    expected_reference_ids: set[str],
) -> None:
    expected_keys = {"entity_decisions", "relation_decisions", "reference_decisions"}
    if set(payload) != expected_keys:
        raise ValueError(f"Scene graph review must return exactly {sorted(expected_keys)}")
    if any(not isinstance(payload[key], list) for key in expected_keys):
        raise ValueError("Every scene graph review field must be an array")
    _by_source_id(payload["entity_decisions"], "source_entity_id", expected_entity_ids)
    _by_source_id(
        payload["relation_decisions"], "source_relation_id", expected_relation_ids
    )


def _align_review_entity_evidence(
    payload: dict[str, Any],
    *,
    record: dict[str, Any],
    source_text: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = copy.deepcopy(payload)
    raw_by_local_id = {
        f"E{index:04d}": entity
        for index, entity in enumerate(record.get("entities", []), start=1)
    }
    corrections: list[dict[str, Any]] = []
    for decision in normalized.get("entity_decisions", []):
        local_id = clean_text(decision.get("source_entity_id"))
        raw_entity = raw_by_local_id.get(local_id, {})
        aligned_entities = []
        for replacement_index, entity in enumerate(decision.get("entities") or []):
            if not isinstance(entity, dict):
                continue
            evidence = clean_text(entity.get("evidence"))
            try:
                _require_source_evidence(evidence, source_text, f"entity {local_id}")
                aligned_entities.append(entity)
                continue
            except ValueError:
                replacement = _minimal_source_evidence(
                    evidence,
                    source_text,
                    fallbacks=[entity.get("name"), raw_entity.get("evidence")],
                )
            if not replacement:
                corrections.append(
                    {
                        "source_entity_id": local_id,
                        "replacement_index": replacement_index,
                        "generated_evidence": evidence,
                        "replacement_evidence": "",
                        "mechanism": "drop_ungrounded_replacement",
                    }
                )
                continue
            _require_source_evidence(replacement, source_text, f"entity {local_id}")
            entity["evidence"] = replacement
            aligned_entities.append(entity)
            corrections.append(
                {
                    "source_entity_id": local_id,
                    "replacement_index": replacement_index,
                    "generated_evidence": evidence,
                    "replacement_evidence": replacement,
                    "mechanism": "align_to_minimal_source_substring",
                }
            )
        decision["entities"] = aligned_entities
        if not aligned_entities:
            decision.update(
                {
                    "action": "drop",
                    "reason_code": "deterministic_ungrounded_entity_drop",
                    "generated_rationale_hint": (
                        "No reviewed replacement could be aligned to a source substring."
                    ),
                }
            )
        elif clean_text(decision.get("action")) == "split" and len(aligned_entities) == 1:
            decision.update(
                {
                    "action": "repair",
                    "reason_code": "deterministic_single_grounded_split_repair",
                }
            )
    return normalized, corrections
    _by_source_id(payload["reference_decisions"], "reference_id", expected_reference_ids)


def _normalize_entity_replacement_cardinality(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Make malformed entity action/cardinality pairs internally consistent.

    Review replacements are already source-evidence aligned at this point.
    Multiple aligned replacements represent a split even when the model labels
    the action as keep/repair; an empty aligned replacement list is a drop.
    This is a schema-preserving correction, not a semantic replacement.
    """
    normalized = copy.deepcopy(payload)
    corrections: list[dict[str, Any]] = []
    for decision in normalized.get("entity_decisions", []):
        if not isinstance(decision, dict):
            continue
        local_id = clean_text(decision.get("source_entity_id"))
        action = clean_text(decision.get("action"))
        replacements = decision.get("entities")
        if not isinstance(replacements, list):
            continue
        if action == "drop" and replacements:
            decision["entities"] = []
            corrections.append(
                {
                    "source_entity_id": local_id,
                    "from_action": action,
                    "to_action": "drop",
                    "replacement_count": len(replacements),
                    "mechanism": "clear_replacements_for_drop",
                }
            )
            continue
        if not replacements and action != "drop":
            decision["action"] = "drop"
            decision["reason_code"] = "deterministic_empty_replacement_drop"
            corrections.append(
                {
                    "source_entity_id": local_id,
                    "from_action": action,
                    "to_action": "drop",
                    "replacement_count": 0,
                    "mechanism": "empty_aligned_replacement_drop",
                }
            )
            continue
        if len(replacements) > 1 and action in {"keep", "repair"}:
            decision["action"] = "split"
            decision["reason_code"] = "deterministic_multi_replacement_split"
            corrections.append(
                {
                    "source_entity_id": local_id,
                    "from_action": action,
                    "to_action": "split",
                    "replacement_count": len(replacements),
                    "mechanism": "multi_replacement_split_normalization",
                }
            )
    return normalized, corrections


def _filter_extraneous_pack_decisions(
    payload: dict[str, Any],
    *,
    expected_entity_ids: set[str],
    expected_relation_ids: set[str],
    expected_reference_ids: set[str],
) -> tuple[dict[str, Any], int, int]:
    expected_keys = {"entity_decisions", "relation_decisions", "reference_decisions"}
    if not isinstance(payload, dict) or any(
        key in payload and not isinstance(payload.get(key), list)
        for key in expected_keys
    ):
        return payload, 0, 0
    specs = (
        ("entity_decisions", "source_entity_id", expected_entity_ids),
        ("relation_decisions", "source_relation_id", expected_relation_ids),
        ("reference_decisions", "reference_id", expected_reference_ids),
    )
    normalized = {
        key: copy.deepcopy(payload.get(key, []))
        for key in expected_keys
    }
    ignored = len(set(payload) - expected_keys)
    duplicate_drops = 0
    for array_name, id_field, expected_ids in specs:
        rows = normalized[array_name]
        kept = []
        seen: set[str] = set()
        for row in rows:
            local_id = clean_text(row.get(id_field)) if isinstance(row, dict) else ""
            if local_id not in expected_ids:
                ignored += 1
                continue
            if local_id in seen:
                duplicate_drops += 1
                continue
            seen.add(local_id)
            kept.append(row)
        normalized[array_name] = kept
    return normalized, ignored, duplicate_drops


def _validate_partial_review_semantics(
    payload: dict[str, Any],
    *,
    source_text: str,
    reference_specs: dict[str, dict[str, Any]],
) -> None:
    for decision in payload["entity_decisions"]:
        local_id = clean_text(decision.get("source_entity_id"))
        action = clean_text(decision.get("action"))
        if action not in {"keep", "repair", "split", "drop"}:
            raise ValueError(f"Unsupported entity review action for {local_id}: {action}")
        replacements = decision.get("entities")
        if not isinstance(replacements, list):
            raise ValueError(f"Entity review replacements must be an array for {local_id}")
        expected_count = 0 if action == "drop" else (2 if action == "split" else 1)
        if (action == "split" and len(replacements) < expected_count) or (
            action != "split" and len(replacements) != expected_count
        ):
            raise ValueError(f"Invalid replacement count for {local_id} action={action}")
        for replacement in replacements:
            _require_source_evidence(
                clean_text(replacement.get("evidence")),
                source_text,
                f"entity {local_id}",
            )

    for decision in payload["relation_decisions"]:
        local_id = clean_text(decision.get("source_relation_id"))
        action = clean_text(decision.get("action"))
        if action not in {"keep", "repair", "drop"}:
            raise ValueError(f"Unsupported relation review action for {local_id}: {action}")
        if action != "drop":
            _require_source_evidence(
                clean_text(decision.get("evidence")),
                source_text,
                f"relation {local_id}",
            )

    for decision in payload["reference_decisions"]:
        local_id = clean_text(decision.get("reference_id"))
        action = clean_text(decision.get("action"))
        if action not in {"map", "create", "drop"}:
            raise ValueError(f"Unsupported reference action for {local_id}: {action}")
        values = decision.get("resolved_entity_names")
        if not isinstance(values, list):
            raise ValueError(f"resolved_entity_names must be an array for {local_id}")
        if action == "drop" and values:
            raise ValueError(f"Dropped reference cannot resolve names for {local_id}")
        if action != "drop" and not values:
            raise ValueError(f"Mapped reference requires reviewed entity names for {local_id}")
        if not reference_specs[local_id]["allows_multiple"] and len(values) > 1:
            raise ValueError(f"Scalar reference cannot map to multiple entities for {local_id}")
        if action == "create":
            created = decision.get("created_entity")
            if not isinstance(created, dict):
                raise ValueError(f"Reference create requires created_entity for {local_id}")
            _require_source_evidence(
                clean_text(created.get("evidence")),
                source_text,
                f"created reference entity {local_id}",
            )


def _drop_invalid_retained_relations(
    payload: dict[str, Any],
    *,
    source_text: str,
    allowed_entity_names: set[str],
) -> tuple[dict[str, Any], int]:
    normalized = copy.deepcopy(payload)
    drop_count = 0
    for decision in normalized.get("relation_decisions", []):
        if clean_text(decision.get("action")) == "drop":
            continue
        subject = clean_text(decision.get("subject")).casefold()
        object_name = clean_text(decision.get("object")).casefold()
        valid = (
            subject in allowed_entity_names
            and object_name in allowed_entity_names
            and subject != object_name
            and bool(clean_text(decision.get("predicate")))
            and bool(clean_text(decision.get("description")))
            and clean_text(decision.get("relation_class"))
            in {"stable", "temporal", "event_like"}
        )
        try:
            _require_source_evidence(
                clean_text(decision.get("evidence")),
                source_text,
                f"relation {clean_text(decision.get('source_relation_id'))}",
            )
        except ValueError:
            valid = False
        if valid:
            continue
        decision.update(
            {
                "action": "drop",
                "subject": "",
                "predicate": "",
                "object": "",
                "relation_class": "event_like",
                "description": "",
                "evidence": "",
                "reason_code": "invalid_reviewed_relation",
                "generated_rationale_hint": "",
            }
        )
        drop_count += 1
    return normalized, drop_count


def _drop_relation_decision(local_id: str) -> dict[str, Any]:
    return {
        "source_relation_id": local_id,
        "action": "drop",
        "subject": "",
        "predicate": "",
        "object": "",
        "relation_class": "event_like",
        "description": "",
        "evidence": "",
        "reason_code": "preview_only",
        "generated_rationale_hint": "",
    }


def _drop_reference_decision(local_id: str) -> dict[str, Any]:
    return {
        "reference_id": local_id,
        "action": "drop",
        "resolved_entity_names": [],
        "created_entity": None,
        "reason_code": "preview_only",
        "generated_rationale_hint": "",
    }


def _resolve_reference_deterministically(
    reference: dict[str, Any], canonical_by_name: dict[str, str]
) -> dict[str, Any]:
    value = clean_text(reference.get("value"))
    exact = canonical_by_name.get(value.casefold())
    resolved = [exact] if exact else []
    reason_code = "exact_reviewed_name_or_alias"
    if not resolved and reference.get("allows_multiple"):
        normalized_value = normalize_name(value)
        resolved = unique_text(
            canonical
            for alias, canonical in canonical_by_name.items()
            if len(normalize_name(alias)) >= 2
            and normalize_name(alias) in normalized_value
        )
        reason_code = "composite_reviewed_names"
    resolved = [
        canonical
        for canonical in resolved
        if canonical_by_name.get(clean_text(canonical).casefold()) == canonical
    ]
    if not resolved:
        return {
            "reference_id": reference["reference_id"],
            "action": "drop",
            "resolved_entity_names": [],
            "created_entity": None,
            "reason_code": "not_a_reviewed_entity_reference",
            "generated_rationale_hint": "",
        }
    return {
        "reference_id": reference["reference_id"],
        "action": "map",
        "resolved_entity_names": resolved,
        "created_entity": None,
        "reason_code": reason_code,
        "generated_rationale_hint": "",
    }


def _build_review_payload(record: dict[str, Any]) -> dict[str, Any]:
    source_text = _scene_source_text(record)
    local_entities: dict[str, dict[str, Any]] = {}
    entities: list[dict[str, Any]] = []
    for index, entity in enumerate(record.get("entities", []), start=1):
        local_id = f"E{index:04d}"
        local_entities[local_id] = entity
        entities.append(
            {
                "entity_id": local_id,
                "name": clean_text(entity.get("name")),
                "entity_type": require_entity_type(entity.get("entity_type")),
                "scope": require_entity_scope(entity.get("scope")),
                "description": clean_text(entity.get("description")),
                "aliases": unique_text(entity.get("aliases") or []),
                "evidence": _minimal_source_evidence(
                    entity.get("evidence"),
                    source_text,
                    fallbacks=[entity.get("name")],
                ),
            }
        )

    local_relations: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []
    for index, relation in enumerate(record.get("entity_relations", []), start=1):
        local_id = f"R{index:04d}"
        local_relations[local_id] = relation
        relations.append(
            {
                "relation_id": local_id,
                "subject": clean_text(relation.get("subject")),
                "predicate": clean_text(relation.get("predicate")),
                "object": clean_text(relation.get("object")),
                "description": clean_text(relation.get("description")),
                "evidence": _minimal_source_evidence(
                    relation.get("evidence"),
                    source_text,
                    fallbacks=[relation.get("subject"), relation.get("object")],
                ),
            }
        )

    references, local_references = _collect_reference_records(record, source_text)
    scene = record["scene"]
    return {
        "scene_text": source_text,
        "review_context": json.dumps(
            {
                "scene_id": scene.get("scene_id"),
                "scene_order": scene.get("order"),
                "title": clean_text(scene.get("title")),
                "subtitle": clean_text(scene.get("subtitle")),
                "policy": "Use only evidence snippets embedded in the supplied records.",
            },
            ensure_ascii=False,
        ),
        "entities": entities,
        "relations": relations,
        "references": references,
        "local_entities": local_entities,
        "local_relations": local_relations,
        "local_references": local_references,
    }


def _collect_reference_records(
    record: dict[str, Any], source_text: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    references: list[dict[str, Any]] = []
    local: dict[str, dict[str, Any]] = {}

    def add(
        *,
        value: Any,
        kind: str,
        owner_id: str,
        evidence: Any,
        path: tuple[Any, ...],
        allows_multiple: bool = False,
    ) -> None:
        name = clean_text(value)
        if not name:
            return
        local_id = f"X{len(references) + 1:04d}"
        item = {
            "reference_id": local_id,
            "value": name,
            "kind": kind,
            "owner_id": owner_id,
            "evidence": _minimal_source_evidence(
                evidence, source_text, fallbacks=[name]
            ),
            "allows_multiple": allows_multiple,
        }
        references.append(item)
        local[local_id] = {**item, "path": path}

    for unit_index, unit in enumerate(record.get("narrative_units", [])):
        owner_id = clean_text(unit.get("unit_id"))
        if unit.get("kind") == "occasion":
            add(
                value=unit.get("name"),
                kind="occasion_entity",
                owner_id=owner_id,
                evidence=unit.get("evidence"),
                path=("narrative_units", unit_index, "name"),
            )
        for participant_index, participant in enumerate(unit.get("participants", [])):
            add(
                value=participant,
                kind="narrative_participant",
                owner_id=owner_id,
                evidence=unit.get("evidence"),
                path=("narrative_units", unit_index, "participants", participant_index),
                allows_multiple=True,
            )
        for field in ("subject", "object"):
            if clean_text(unit.get(field)):
                add(
                    value=unit.get(field),
                    kind=f"narrative_{field}",
                    owner_id=owner_id,
                    evidence=unit.get("evidence"),
                    path=("narrative_units", unit_index, field),
                )
        for field in ("locations", "times"):
            for value_index, value in enumerate(unit.get(field, [])):
                add(
                    value=value,
                    kind=f"narrative_{field}",
                    owner_id=owner_id,
                    evidence=unit.get("evidence"),
                    path=("narrative_units", unit_index, field, value_index),
                )
        if clean_text(unit.get("related_occasion")):
            add(
                value=unit.get("related_occasion"),
                kind="narrative_related_occasion",
                owner_id=owner_id,
                evidence=unit.get("evidence"),
                path=("narrative_units", unit_index, "related_occasion"),
            )

    return references, local


def _apply_review_payload(
    *,
    record: dict[str, Any],
    payload: dict[str, Any],
    movie_id: str,
    local_entities: dict[str, dict[str, Any]],
    local_relations: dict[str, dict[str, Any]],
    local_references: dict[str, dict[str, Any]],
    source_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_keys = {"entity_decisions", "relation_decisions", "reference_decisions"}
    if set(payload) != expected_keys:
        raise ValueError(f"Scene graph review must return exactly {sorted(expected_keys)}")
    if any(not isinstance(payload[key], list) for key in expected_keys):
        raise ValueError("Every scene graph review field must be an array")

    entity_decisions = _by_source_id(
        payload["entity_decisions"], "source_entity_id", set(local_entities)
    )
    reviewed_entities: list[dict[str, Any]] = []
    entity_action_counts: dict[str, int] = {}
    entity_reason_counts: dict[str, int] = {}
    deterministic_event_entity_drops = 0
    deterministic_invalid_type_entity_drops: list[dict[str, Any]] = []
    deterministic_occasion_type_corrections = 0
    deterministic_occasion_action_corrections = 0
    deterministic_occasion_name_corrections = 0
    deterministic_occasion_ungrounded_drops = 0
    dropped_occasion_unit_names: set[str] = set()
    for local_id, raw_entity in local_entities.items():
        decision = entity_decisions[local_id]
        action = clean_text(decision.get("action"))
        if action not in {"keep", "repair", "split", "drop"}:
            raise ValueError(f"Unsupported entity review action for {local_id}: {action}")
        raw_type = require_entity_type(raw_entity.get("entity_type"))
        if raw_type == "Occasion" and action in {"drop", "split"}:
            # Preserve the historical Occasion lock only when the original
            # candidate has verifiable source evidence. A paraphrased raw
            # evidence field must not be reintroduced after alignment drops it.
            raw_evidence = _minimal_source_evidence(
                raw_entity.get("evidence"),
                source_text,
                fallbacks=[raw_entity.get("name")],
            )
            if raw_evidence:
                original_action = action
                action = "keep"
                decision = {
                    **decision,
                    "action": "keep",
                    "entities": [
                        {
                            "name": clean_text(raw_entity.get("name")),
                            "entity_type": "Occasion",
                            "scope": require_entity_scope(raw_entity.get("scope")),
                            "description": clean_text(raw_entity.get("description")),
                            "aliases": unique_text(raw_entity.get("aliases") or []),
                            "evidence": raw_evidence,
                        }
                    ],
                    "reason_code": "pre_extracted_occasion_action_lock",
                    "generated_rationale_hint": clean_text(
                        decision.get("generated_rationale_hint")
                    ),
                }
                deterministic_occasion_action_corrections += 1
            else:
                original_action = ""
                deterministic_occasion_ungrounded_drops += 1
                raw_occasion_name = normalize_name(raw_entity.get("name"))
                if raw_occasion_name:
                    dropped_occasion_unit_names.add(raw_occasion_name)
                decision = {
                    **decision,
                    "action": "drop",
                    "entities": [],
                    "reason_code": "deterministic_ungrounded_occasion_drop",
                    "generated_rationale_hint": (
                        "The pre-extracted Occasion evidence is not a source substring."
                    ),
                }
                action = "drop"
        else:
            original_action = ""
        entity_action_counts[action] = entity_action_counts.get(action, 0) + 1
        reason_code = clean_text(decision.get("reason_code")) or "unspecified"
        entity_reason_counts[reason_code] = entity_reason_counts.get(reason_code, 0) + 1
        replacements = decision.get("entities")
        if not isinstance(replacements, list):
            raise ValueError(f"Entity review replacements must be an array for {local_id}")
        expected_count = 0 if action == "drop" else (2 if action == "split" else 1)
        if (action == "split" and len(replacements) < expected_count) or (
            action != "split" and len(replacements) != expected_count
        ):
            raise ValueError(f"Invalid replacement count for {local_id} action={action}")
        for replacement_index, replacement in enumerate(replacements, start=1):
            generated_type = clean_text(
                replacement.get("entity_type") if isinstance(replacement, dict) else None
            )
            forced_occasion_type = ""
            try:
                require_entity_type(generated_type)
            except ValueError:
                if raw_type == "Occasion" and isinstance(replacement, dict):
                    forced_occasion_type = generated_type
                    replacement = {**replacement, "entity_type": "Occasion"}
                else:
                    deterministic_invalid_type_entity_drops.append(
                        {
                            "source_entity_id": local_id,
                            "replacement_index": replacement_index,
                            "generated_name": clean_text(
                                replacement.get("name")
                                if isinstance(replacement, dict)
                                else None
                            ),
                            "generated_entity_type": generated_type,
                            "mechanism": "drop_review_entity_with_illegal_type",
                        }
                    )
                    if generated_type.casefold() == "event":
                        deterministic_event_entity_drops += 1
                    continue
            normalized_entity = _normalize_reviewed_entity(
                    replacement,
                    movie_id=movie_id,
                    scene_id=record["scene"]["scene_id"],
                    source_text=source_text,
                    source_entity_ids=[clean_text(raw_entity.get("mention_id"))],
                    fallback_evidence=raw_entity.get("evidence"),
                    local_id=local_id,
                    replacement_index=replacement_index,
                    reason_code=decision.get("reason_code"),
                    generated_rationale_hint=decision.get("generated_rationale_hint"),
            )
            if raw_type == "Occasion" and normalized_entity["entity_type"] != "Occasion":
                reviewed_type = normalized_entity["entity_type"]
                normalized_entity["entity_type"] = "Occasion"
                normalized_entity["type_profile"] = initial_type_profile("Occasion")
                normalized_entity["review"]["deterministic_type_correction"] = {
                    "from": reviewed_type,
                    "to": "Occasion",
                    "reason": "pre_extracted_occasion_type_lock",
                }
                deterministic_occasion_type_corrections += 1
            elif raw_type == "Occasion" and forced_occasion_type:
                normalized_entity["review"]["deterministic_type_correction"] = {
                    "from": forced_occasion_type,
                    "to": "Occasion",
                    "reason": "pre_extracted_occasion_type_lock",
                }
                deterministic_occasion_type_corrections += 1
            if raw_type == "Occasion":
                raw_name = clean_text(raw_entity.get("name"))
                reviewed_name = normalized_entity["name"]
                if reviewed_name != raw_name:
                    normalized_entity["name"] = raw_name
                    normalized_entity["mention_id"] = stable_id(
                        "reviewed-mention",
                        movie_id,
                        record["scene"]["scene_id"],
                        local_id,
                        replacement_index,
                        raw_name,
                    )
                    normalized_entity["review"]["deterministic_name_correction"] = {
                        "from": reviewed_name,
                        "to": raw_name,
                        "reason": "pre_extracted_occasion_name_lock",
                    }
                    deterministic_occasion_name_corrections += 1
            if original_action:
                normalized_entity["review"]["deterministic_action_correction"] = {
                    "from": original_action,
                    "to": "keep",
                    "reason": "pre_extracted_occasion_action_lock",
                }
            reviewed_entities.append(normalized_entity)

    reference_decisions = _by_source_id(
        payload["reference_decisions"], "reference_id", set(local_references)
    )
    created_entities: list[dict[str, Any]] = []
    for local_id, spec in local_references.items():
        decision = reference_decisions[local_id]
        if clean_text(decision.get("action")) != "create":
            continue
        created = decision.get("created_entity")
        if not isinstance(created, dict):
            raise ValueError(f"Reference create requires created_entity for {local_id}")
        generated_type = clean_text(created.get("entity_type"))
        try:
            require_entity_type(generated_type)
        except ValueError:
            deterministic_invalid_type_entity_drops.append(
                {
                    "source_entity_id": local_id,
                    "replacement_index": 1,
                    "generated_name": clean_text(created.get("name")),
                    "generated_entity_type": generated_type,
                    "mechanism": "drop_review_entity_with_illegal_type",
                }
            )
            if generated_type.casefold() == "event":
                deterministic_event_entity_drops += 1
            continue
        created_entities.append(
            _normalize_reviewed_entity(
                created,
                movie_id=movie_id,
                scene_id=record["scene"]["scene_id"],
                source_text=source_text,
                source_entity_ids=[],
                fallback_evidence=None,
                local_id=local_id,
                replacement_index=1,
                reason_code=decision.get("reason_code"),
                generated_rationale_hint=decision.get("generated_rationale_hint"),
            )
        )
    reviewed_entities.extend(created_entities)
    canonical_by_name = _canonical_names(reviewed_entities)

    reviewed = copy.deepcopy(record)
    reviewed["entities"] = reviewed_entities
    dropped_occasion_unit_count = 0
    dropped_occasion_unit_ids: set[str] = set()
    if dropped_occasion_unit_names:
        dropped_occasion_unit_ids = {
            clean_text(unit.get("unit_id"))
            for unit in reviewed.get("narrative_units", [])
            if (
                unit.get("kind") == "occasion"
                and normalize_name(unit.get("name")) in dropped_occasion_unit_names
            )
        }
        dropped_occasion_unit_count = sum(
            1
            for unit in reviewed.get("narrative_units", [])
            if (
                unit.get("kind") == "occasion"
                and normalize_name(unit.get("name")) in dropped_occasion_unit_names
            )
        )
    remaining_references = {
        reference_id: spec
        for reference_id, spec in local_references.items()
        if clean_text(spec.get("owner_id")) not in dropped_occasion_unit_ids
    }
    _apply_reference_decisions(
        reviewed,
        reference_decisions=reference_decisions,
        local_references=remaining_references,
        canonical_by_name=canonical_by_name,
    )
    # Reference paths are indexed against the original narrative-unit list.
    # Apply them before removing ungrounded Occasion units so later unit indexes
    # cannot shift underneath a valid decision.
    if dropped_occasion_unit_names:
        reviewed["narrative_units"] = [
            unit
            for unit in reviewed.get("narrative_units", [])
            if not (
                unit.get("kind") == "occasion"
                and normalize_name(unit.get("name")) in dropped_occasion_unit_names
            )
        ]

    relation_decisions = _by_source_id(
        payload["relation_decisions"], "source_relation_id", set(local_relations)
    )
    reviewed_relations: list[dict[str, Any]] = []
    relation_action_counts: dict[str, int] = {}
    relation_reason_counts: dict[str, int] = {}
    deterministic_occasion_relation_drops = 0
    for local_id, raw_relation in local_relations.items():
        decision = relation_decisions[local_id]
        action = clean_text(decision.get("action"))
        if action not in {"keep", "repair", "drop"}:
            raise ValueError(f"Unsupported relation review action for {local_id}: {action}")
        relation_action_counts[action] = relation_action_counts.get(action, 0) + 1
        reason_code = clean_text(decision.get("reason_code")) or "unspecified"
        relation_reason_counts[reason_code] = relation_reason_counts.get(reason_code, 0) + 1
        if action == "drop":
            continue
        if dropped_occasion_unit_names and any(
            normalize_name(raw_relation.get(field)) in dropped_occasion_unit_names
            for field in ("subject", "object")
        ):
            deterministic_occasion_relation_drops += 1
            relation_action_counts[action] = max(
                0, relation_action_counts.get(action, 0) - 1
            )
            relation_action_counts["drop"] = relation_action_counts.get("drop", 0) + 1
            continue
        subject = _canonical_endpoint(decision.get("subject"), canonical_by_name)
        object_name = _canonical_endpoint(decision.get("object"), canonical_by_name)
        predicate = clean_text(decision.get("predicate"))
        relation_class = clean_text(decision.get("relation_class"))
        description = clean_text(decision.get("description"))
        evidence = clean_text(decision.get("evidence"))
        if not subject or not object_name or not predicate or not description:
            raise ValueError(f"Incomplete retained relation review for {local_id}")
        if relation_class not in {"stable", "temporal", "event_like"}:
            raise ValueError(f"Unsupported relation_class for {local_id}: {relation_class}")
        _require_source_evidence(evidence, source_text, f"relation {local_id}")
        reviewed_relations.append(
            {
                **raw_relation,
                "relation_id": stable_id(
                    "reviewed-relation",
                    movie_id,
                    record["scene"]["scene_id"],
                    raw_relation.get("relation_id"),
                    subject,
                    predicate,
                    object_name,
                ),
                "subject": subject,
                "predicate": predicate,
                "object": object_name,
                "description": description,
                "evidence": evidence,
                "relation_class": relation_class,
                "source_relation_ids": [clean_text(raw_relation.get("relation_id"))],
                "review": {
                    "action": action,
                    "reason_code": clean_text(decision.get("reason_code")),
                    "generated_rationale_hint": clean_text(
                        decision.get("generated_rationale_hint")
                    ),
                },
            }
        )
    reviewed["entity_relations"] = reviewed_relations
    reviewed["scene_graph_reviewed"] = True
    return reviewed, {
        "output_entity_count": len(reviewed_entities),
        "output_relation_count": len(reviewed_relations),
        "created_entity_count": len(created_entities),
        "deterministic_event_entity_drop_count": deterministic_event_entity_drops,
        "deterministic_invalid_type_entity_drop_count": len(
            deterministic_invalid_type_entity_drops
        ),
        "deterministic_invalid_type_entity_drops": (
            deterministic_invalid_type_entity_drops
        ),
        "deterministic_occasion_type_correction_count": (
            deterministic_occasion_type_corrections
        ),
        "deterministic_occasion_action_correction_count": (
            deterministic_occasion_action_corrections
        ),
        "deterministic_occasion_name_correction_count": (
            deterministic_occasion_name_corrections
        ),
        "deterministic_occasion_ungrounded_drop_count": (
            deterministic_occasion_ungrounded_drops
        ),
        "deterministic_occasion_unit_drop_count": dropped_occasion_unit_count,
        "deterministic_occasion_relation_drop_count": (
            deterministic_occasion_relation_drops
        ),
        "entity_keep_count": entity_action_counts.get("keep", 0),
        "entity_repair_count": entity_action_counts.get("repair", 0),
        "entity_split_count": entity_action_counts.get("split", 0),
        "entity_drop_count": entity_action_counts.get("drop", 0),
        "relation_keep_count": relation_action_counts.get("keep", 0),
        "relation_repair_count": relation_action_counts.get("repair", 0),
        "relation_drop_count": relation_action_counts.get("drop", 0),
        "entity_reason_counts": dict(sorted(entity_reason_counts.items())),
        "relation_reason_counts": dict(sorted(relation_reason_counts.items())),
    }


def _normalize_reviewed_entity(
    raw: dict[str, Any],
    *,
    movie_id: str,
    scene_id: str,
    source_text: str,
    source_entity_ids: list[str],
    fallback_evidence: Any,
    local_id: str,
    replacement_index: int,
    reason_code: Any,
    generated_rationale_hint: Any,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Reviewed entity must be an object for {local_id}")
    name = clean_text(raw.get("name"))
    entity_type = require_entity_type(raw.get("entity_type"))
    scope = require_entity_scope(raw.get("scope"))
    description = clean_text(raw.get("description"))
    evidence = clean_text(raw.get("evidence"))
    aliases = unique_text(raw.get("aliases") or [], limit=32)
    if not name or not description:
        raise ValueError(f"Reviewed entity requires name and description for {local_id}")
    evidence_correction: dict[str, str] | None = None
    try:
        _require_source_evidence(evidence, source_text, f"entity {local_id}")
    except ValueError:
        replacement = _minimal_source_evidence(
            evidence,
            source_text,
            fallbacks=[name, fallback_evidence],
        )
        _require_source_evidence(replacement, source_text, f"entity {local_id}")
        evidence_correction = {
            "generated_evidence": evidence,
            "replacement_evidence": replacement,
            "reason": "align_to_minimal_source_substring",
        }
        evidence = replacement
    normalized = {
        "mention_id": stable_id(
            "reviewed-mention", movie_id, scene_id, local_id, replacement_index, name
        ),
        "name": name,
        "entity_type": entity_type,
        "scope": scope,
        "type_profile": initial_type_profile(entity_type),
        "description": description,
        "aliases": aliases,
        "evidence": evidence,
        "source_scene_id": scene_id,
        "source_entity_ids": [value for value in source_entity_ids if value],
        "review": {
            "reason_code": clean_text(reason_code),
            "generated_rationale_hint": clean_text(generated_rationale_hint),
        },
    }
    if evidence_correction is not None:
        normalized["review"]["deterministic_evidence_correction"] = (
            evidence_correction
        )
    return normalized


def _canonical_names(entities: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    ambiguous: set[str] = set()
    for entity in entities:
        for value in [entity["name"], *entity.get("aliases", [])]:
            key = clean_text(value).casefold()
            if key in mapping and mapping[key] != entity["name"]:
                ambiguous.add(key)
            else:
                mapping[key] = entity["name"]
    for key in ambiguous:
        mapping.pop(key, None)
    return mapping


def _apply_reference_decisions(
    reviewed: dict[str, Any],
    *,
    reference_decisions: dict[str, dict[str, Any]],
    local_references: dict[str, dict[str, Any]],
    canonical_by_name: dict[str, str],
) -> None:
    participant_replacements: dict[tuple[int, int], list[str]] = {}
    scalar_replacements: list[tuple[tuple[Any, ...], str]] = []
    for local_id, spec in local_references.items():
        decision = reference_decisions[local_id]
        action = clean_text(decision.get("action"))
        if action not in {"map", "create", "drop"}:
            raise ValueError(f"Unsupported reference action for {local_id}: {action}")
        values = decision.get("resolved_entity_names")
        if not isinstance(values, list):
            raise ValueError(f"resolved_entity_names must be an array for {local_id}")
        resolved = [_canonical_endpoint(value, canonical_by_name) for value in values]
        if spec["kind"] == "occasion_entity" and (action == "drop" or not all(resolved)):
            locked_name = _canonical_endpoint(spec["value"], canonical_by_name)
            if locked_name:
                action = "map"
                resolved = [locked_name]
        if action == "drop":
            if resolved:
                raise ValueError(f"Dropped reference cannot resolve names for {local_id}")
        elif not resolved or any(not value for value in resolved):
            raise ValueError(f"Mapped reference requires reviewed entity names for {local_id}")
        resolved = unique_text(resolved)
        if not spec["allows_multiple"] and len(resolved) > 1:
            raise ValueError(f"Scalar reference cannot map to multiple entities for {local_id}")
        path = spec["path"]
        if path[0] == "narrative_units" and path[2] == "participants":
            participant_replacements[(int(path[1]), int(path[3]))] = resolved
        else:
            scalar_replacements.append((path, resolved[0] if resolved else ""))

    for unit_index, unit in enumerate(reviewed.get("narrative_units", [])):
        original = list(unit.get("participants", []))
        replaced: list[str] = []
        for participant_index in range(len(original)):
            replaced.extend(participant_replacements.get((unit_index, participant_index), []))
        unit["participants"] = unique_text(replaced, limit=48)
    for path, value in scalar_replacements:
        _set_path(reviewed, path, value)


def _canonical_endpoint(value: Any, mapping: dict[str, str]) -> str:
    return mapping.get(clean_text(value).casefold(), "")


def _by_source_id(
    rows: list[dict[str, Any]], id_field: str, expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Review decision in {id_field} array must be an object")
        local_id = clean_text(row.get(id_field))
        if local_id not in expected_ids or local_id in by_id:
            raise ValueError(f"Unknown or duplicate {id_field}: {local_id}")
        by_id[local_id] = row
    if set(by_id) != expected_ids:
        raise ValueError(f"Missing {id_field} decisions: {sorted(expected_ids - set(by_id))}")
    return by_id


def _scene_source_text(record: dict[str, Any]) -> str:
    scene = record["scene"]
    content = "".join(chunk.get("content", "") for chunk in record.get("chunks", []))
    return "\n".join(
        value
        for value in (
            clean_text(scene.get("title")),
            clean_text(scene.get("subtitle")),
            content,
        )
        if value
    )


def _require_source_evidence(evidence: str, source_text: str, label: str) -> None:
    if not evidence:
        raise ValueError(f"Reviewed {label} requires source evidence")
    normalized_evidence = re.sub(r"\s+", "", evidence)
    normalized_source = re.sub(r"\s+", "", source_text)
    if (
        evidence not in source_text
        and clean_text(evidence) not in clean_text(source_text)
        and normalized_evidence not in normalized_source
    ):
        raise ValueError(f"Reviewed {label} evidence is not a source substring: {evidence}")


def _minimal_source_evidence(
    evidence: Any,
    source_text: str,
    *,
    fallbacks: list[Any] | None = None,
) -> str:
    """Return one short source span without exposing the surrounding scene to review."""
    value = clean_text(evidence)
    direct = _source_substring(value, source_text)
    if direct:
        return direct

    title_match = re.search(r"\bTitle\s*[:：]\s*(.+)$", value, flags=re.IGNORECASE)
    candidates = [title_match.group(1)] if title_match else []
    candidates.extend(
        part
        for part in re.split(r"(?:\.{3,}|…{1,}|。{3,})", value)
        if clean_text(part)
    )
    candidates.extend(fallbacks or [])
    for candidate in candidates:
        direct = _source_substring(clean_text(candidate), source_text)
        if direct:
            return direct

    evidence_chars, _ = _alignment_chars(value)
    source_chars, source_offsets = _alignment_chars(source_text)
    if evidence_chars and source_chars:
        match = SequenceMatcher(
            None, evidence_chars, source_chars, autojunk=False
        ).find_longest_match()
        minimum = min(12, max(4, len(evidence_chars) // 3))
        if match.size >= minimum:
            start = source_offsets[match.b]
            end = source_offsets[match.b + match.size - 1] + 1
            return clean_text(source_text[start:end])
    return ""


def _source_substring(value: str, source_text: str) -> str:
    if not value:
        return ""
    if value in source_text:
        return value
    compact_value = re.sub(r"\s+", "", value)
    compact_source, offsets = _compact_with_offsets(source_text)
    start = compact_source.find(compact_value)
    if start < 0:
        return ""
    source_start = offsets[start]
    source_end = offsets[start + len(compact_value) - 1] + 1
    return clean_text(source_text[source_start:source_end])


def _compact_with_offsets(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(value):
        if char.isspace():
            continue
        chars.append(char)
        offsets.append(index)
    return "".join(chars), offsets


def _alignment_chars(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for index, raw_char in enumerate(value):
        for char in unicodedata.normalize("NFKC", raw_char).casefold():
            if char.isalnum():
                chars.append(char)
                offsets.append(index)
    return "".join(chars), offsets


def _set_path(payload: dict[str, Any], path: tuple[Any, ...], value: Any) -> None:
    current: Any = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _validation_feedback(error: Exception | None) -> str:
    if error is None:
        return ""
    return (
        "\n\nThe previous review failed strict validation with this error: "
        f"{clean_text(error)}. Return a complete corrected decision for every supplied id."
    )
