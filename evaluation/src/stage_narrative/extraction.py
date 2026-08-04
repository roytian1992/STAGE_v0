from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .chunking import ChunkingConfig, ScreenplayChunk, TokenCounter, chunk_scene
from .clients import ModelResponseParseError
from .models import (
    BASE_UNIT_KINDS,
    Scene,
    clean_text,
    normalize_name,
    require_entity_scope,
    require_entity_type,
    stable_id,
    unique_text,
)
from .prompt_loader import PROMPTS
from .prompt_assets import extraction_prompt_values


_SCENE_RECONCILIATION_PROMPT = PROMPTS.get("scene_reconciliation")
SCENE_RECONCILIATION_SYSTEM = _SCENE_RECONCILIATION_PROMPT.system
SCENE_RECONCILIATION_USER = _SCENE_RECONCILIATION_PROMPT.user

INTERACTION_POLARITIES = frozenset({"positive", "negative", "neutral"})
INTERACTION_POLARITY_ALIASES = {
    "tense": "neutral",
    "mixed": "neutral",
    "ambivalent": "neutral",
    "ambiguous": "neutral",
    "uncertain": "neutral",
    "hostile": "negative",
    "threatening": "negative",
    "aggressive": "negative",
    "adversarial": "negative",
    "confrontational": "negative",
    "confronting": "negative",
    # Vulnerability is an adverse interaction stance in the closed schema.
    "vulnerable": "negative",
    "friendly": "positive",
    "supportive": "positive",
    "warm": "positive",
}
NARRATIVE_MODALITY_ALIASES = {
    # A screenplay vision is a non-asserted perceptual event in the fixed
    # narrative modality taxonomy.
    "vision": "hallucinated",
    # Some screenplay descriptions use fantasy as a surface label for a
    # non-asserted imagined event; keep the public modality taxonomy closed.
    "fantasy": "hypothetical",
}
NARRATIVE_MODALITIES = frozenset(
    {
        "asserted",
        "remembered",
        "dreamed",
        "hallucinated",
        "hypothetical",
        "reported",
        "uncertain",
    }
)

NARRATIVE_ITEM_DEFAULTS: dict[str, dict[str, Any]] = {
    "events": {
        "name": "", "description": "", "event_subtype": "other",
        "modality": "asserted",
        "state_before": "", "state_after": "", "intent": "",
        "cause_hints": [], "effect_hints": [], "participants": [],
        "locations": [], "times": [], "related_occasion": "",
        "setting": "", "evidence": "",
    },
    "occasions": {
        "name": "", "description": "", "occasion_type": "other",
        "modality": "asserted",
        "institutional_context": "", "participants": [], "locations": [],
        "times": [], "setting": "", "evidence": "",
    },
    "interactions": {
        "name": "", "subject": "", "object": "", "interaction_type": "other",
        "modality": "asserted",
        "polarity": "neutral", "description": "", "participants": [], "tags": [],
        "outcome": "", "related_event": "", "related_occasion": "",
        "locations": [], "times": [], "evidence": "",
    },
}


class JsonClient(Protocol):
    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    language: str
    semantic_attempts: int = 2
    max_concurrency: int = 4


class ScreenplayExtractor:
    def __init__(
        self,
        client: JsonClient,
        config: ExtractionConfig,
        movie_id: str,
        *,
        token_counter: TokenCounter,
        chunking_config: ChunkingConfig,
    ):
        self.client = client
        self.config = config
        self.movie_id = movie_id
        self.token_counter = token_counter
        self.chunking_config = chunking_config
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrency))

    async def extract_scene(self, scene: Scene) -> dict[str, Any]:
        content_budget = self._scene_content_budget(scene)
        chunks = chunk_scene(
            movie_id=self.movie_id,
            scene=scene,
            token_counter=self.token_counter,
            max_content_tokens=content_budget,
        )
        chunk_records: list[dict[str, Any]] = []
        all_calls: list[dict[str, Any]] = []
        context_audit: list[dict[str, Any]] = []
        for chunk in chunks:
            prior_context, context_meta = _build_prior_context(
                chunk_records,
                token_counter=self.token_counter,
                max_tokens=self.chunking_config.carry_context_max_tokens,
            )
            record = await self._extract_chunk(
                scene=scene,
                chunk=chunk,
                prior_context=prior_context,
            )
            chunk_records.append(record)
            all_calls.extend(record.pop("llm_calls"))
            context_audit.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_order": chunk.order,
                    **context_meta,
                }
            )

        narrative_units = [
            unit for record in chunk_records for unit in record["narrative_units"]
        ]
        reconciliation_audit: dict[str, Any] = {
            "enabled": len(chunks) > 1,
            "input_unit_count": len(narrative_units),
            "output_unit_count": len(narrative_units),
            "rounds": [],
            "llm_calls": [],
        }
        if len(chunks) > 1 and narrative_units:
            narrative_units, reconciliation_audit = await self._reconcile_scene_units(
                scene, narrative_units
            )
            all_calls.extend(reconciliation_audit["llm_calls"])

        return {
            "scene": {
                "scene_id": scene.scene_id,
                "source_scene_id": scene.source_scene_id,
                "order": scene.order,
                "title": scene.title,
                "subtitle": scene.subtitle,
            },
            "chunks": [chunk.as_record() for chunk in chunks],
            "entities": [item for record in chunk_records for item in record["entities"]],
            "entity_relations": [
                item for record in chunk_records for item in record["entity_relations"]
            ],
            "narrative_units": narrative_units,
            "llm_calls": all_calls,
            "chunk_context_audit": context_audit,
            "reconciliation_audit": reconciliation_audit,
            "chunking": {
                "content_token_budget": content_budget,
                "chunk_count": len(chunks),
                "source_content_tokens": self.token_counter.count(scene.content),
            },
        }

    async def _extract_chunk(
        self,
        *,
        scene: Scene,
        chunk: ScreenplayChunk,
        prior_context: str,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for semantic_attempt in (1,):
            try:
                validation_feedback = _validation_feedback(last_error)
                narrative_system, narrative_user = PROMPTS.render(
                    "narrative_extraction",
                    **extraction_prompt_values(
                        "narrative_extraction",
                        language=self.config.language,
                        scene_text=chunk.prompt_text(scene),
                        prior_context=prior_context,
                    ),
                )
                narrative_call_audit: list[dict[str, Any]] = []
                narrative_repair_calls: list[dict[str, Any]] = []
                try:
                    narrative_call = await self._generate_json(
                        system_prompt=narrative_system,
                        user_prompt=narrative_user + validation_feedback,
                        stage=f"narrative_extraction:{scene.scene_id}:{chunk.order:04d}",
                    )
                    narrative_payload, formal_structure_repairs = (
                        _normalize_narrative_structure(narrative_call.data)
                    )
                    narrative_payload, formal_empty_value_drops = (
                        _drop_empty_narrative_list_values(narrative_payload)
                    )
                    narrative_payload, formal_duplicate_drops = (
                        _drop_cross_family_narrative_duplicates(narrative_payload)
                    )
                    narrative_payload, polarity_corrections = (
                        _normalize_narrative_surface_variants(narrative_payload)
                    )
                    narrative_call_audit.append(
                        {
                            **narrative_call.metadata,
                            "chunk_id": chunk.chunk_id,
                            "deterministic_structure_repairs": formal_structure_repairs,
                            "deterministic_structure_repair_count": len(
                                formal_structure_repairs
                            ),
                            "deterministic_empty_value_drops": formal_empty_value_drops,
                            "deterministic_empty_value_drop_count": len(
                                formal_empty_value_drops
                            ),
                            "deterministic_cross_family_duplicate_drops": formal_duplicate_drops,
                            "deterministic_cross_family_duplicate_drop_count": len(
                                formal_duplicate_drops
                            ),
                            "deterministic_polarity_corrections": polarity_corrections,
                            "deterministic_polarity_correction_count": len(
                                polarity_corrections
                            ),
                        }
                    )
                    _validate_narrative_payload(narrative_payload)
                except ModelResponseParseError as narrative_error:
                    narrative_call_audit.append(
                        {**narrative_error.metadata, "chunk_id": chunk.chunk_id}
                    )
                    narrative_payload, narrative_repair_calls = (
                        await self._repair_narrative_payload(
                            scene=scene,
                            chunk=chunk,
                            candidate=narrative_error.raw_text,
                            validation_error=narrative_error,
                        )
                    )
                except ValueError as narrative_error:
                    narrative_payload, narrative_repair_calls = (
                        await self._repair_narrative_payload(
                            scene=scene,
                            chunk=chunk,
                            candidate=narrative_payload,
                            validation_error=narrative_error,
                        )
                    )
                occasion_candidates = _occasion_candidates(narrative_payload)
                occasion_json = json.dumps(
                    occasion_candidates, ensure_ascii=False, indent=2
                )
                entity_system, entity_user = PROMPTS.render(
                    "entity_extraction",
                    **extraction_prompt_values(
                        "entity_extraction",
                        language=self.config.language,
                        scene_text=chunk.prompt_text(scene),
                        prior_context=prior_context,
                        occasion_candidates=occasion_json,
                    ),
                )
                entity_call_audit: list[dict[str, Any]] = []
                repair_calls: list[dict[str, Any]] = []
                try:
                    entity_call = await self._generate_json(
                        system_prompt=entity_system,
                        user_prompt=entity_user + validation_feedback,
                        stage=f"entity_extraction:{scene.scene_id}:{chunk.order:04d}",
                    )
                    entity_payload = entity_call.data
                    entity_payload, deterministic_entity_corrections = (
                        _drop_unlocked_entity_candidates(
                            entity_payload, occasion_candidates
                        )
                    )
                    entity_call_audit.append(
                        {
                            **entity_call.metadata,
                            "chunk_id": chunk.chunk_id,
                            "deterministic_payload_corrections": (
                                deterministic_entity_corrections
                            ),
                        }
                    )
                    _validate_entity_payload(entity_payload, occasion_candidates)
                except ModelResponseParseError as entity_error:
                    entity_call_audit.append(
                        {**entity_error.metadata, "chunk_id": chunk.chunk_id}
                    )
                    entity_payload, repair_calls = await self._repair_entity_payload(
                        scene=scene,
                        chunk=chunk,
                        candidate=entity_error.raw_text,
                        validation_error=entity_error,
                        semantic_attempt=semantic_attempt,
                        occasion_candidates=occasion_candidates,
                    )
                except ValueError as entity_error:
                    entity_payload, repair_calls = await self._repair_entity_payload(
                        scene=scene,
                        chunk=chunk,
                        candidate=entity_payload,
                        validation_error=entity_error,
                        semantic_attempt=semantic_attempt,
                        occasion_candidates=occasion_candidates,
                    )
                locked_entities = _locked_entities_for_relation_prompt(entity_payload)
                relation_system, relation_user = PROMPTS.render(
                    "relation_extraction",
                    **extraction_prompt_values(
                        "relation_extraction",
                        language=self.config.language,
                        scene_text=chunk.prompt_text(scene),
                        prior_context=prior_context,
                        locked_entities=json.dumps(
                            locked_entities, ensure_ascii=False, indent=2
                        ),
                    ),
                )
                relation_call_audit: list[dict[str, Any]] = []
                relation_repair_calls: list[dict[str, Any]] = []
                try:
                    relation_call = await self._generate_json(
                        system_prompt=relation_system,
                        user_prompt=relation_user + validation_feedback,
                        stage=f"relation_extraction:{scene.scene_id}:{chunk.order:04d}",
                    )
                    relation_payload = relation_call.data
                    relation_payload, relation_schema_corrections = (
                        _normalize_relation_payload_structure(relation_payload)
                    )
                    relation_call_audit.append(
                        {
                            **relation_call.metadata,
                            "chunk_id": chunk.chunk_id,
                            "deterministic_schema_corrections": (
                                relation_schema_corrections
                            ),
                            "deterministic_schema_correction_count": len(
                                relation_schema_corrections
                            ),
                        }
                    )
                    _validate_relation_payload(relation_payload, entity_payload)
                except ModelResponseParseError as relation_error:
                    relation_call_audit.append(
                        {**relation_error.metadata, "chunk_id": chunk.chunk_id}
                    )
                    relation_payload, relation_repair_calls = (
                        await self._repair_relation_payload(
                            scene=scene,
                            chunk=chunk,
                            candidate=relation_error.raw_text,
                            entity_payload=entity_payload,
                            validation_error=relation_error,
                            semantic_attempt=semantic_attempt,
                        )
                    )
                except ValueError as relation_error:
                    relation_payload, relation_repair_calls = (
                        await self._repair_relation_payload(
                            scene=scene,
                            chunk=chunk,
                            candidate=relation_payload,
                            entity_payload=entity_payload,
                            validation_error=relation_error,
                            semantic_attempt=semantic_attempt,
                        )
                    )
                record = _normalize_chunk_result(
                    movie_id=self.movie_id,
                    scene=scene,
                    chunk=chunk,
                    entity_payload=entity_payload,
                    relation_payload=relation_payload,
                    narrative_payload=narrative_payload,
                )
                record["llm_calls"] = [
                    *narrative_call_audit,
                    *narrative_repair_calls,
                    *entity_call_audit,
                    *repair_calls,
                    *relation_call_audit,
                    *relation_repair_calls,
                ]
                record["semantic_attempt"] = semantic_attempt
                return record
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Scene {scene.scene_id} chunk {chunk.order} failed semantic validation after "
            f"one formal attempt: {last_error}"
        ) from last_error

    async def _repair_narrative_payload(
        self,
        *,
        scene: Scene,
        chunk: ScreenplayChunk,
        candidate: dict[str, Any] | str,
        validation_error: Exception,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        system_prompt, user_prompt = PROMPTS.render(
            "narrative_repair",
            **extraction_prompt_values(
                "narrative_repair",
                language=self.config.language,
                validation_error=clean_text(validation_error),
                candidate_payload=_candidate_payload_text(candidate),
                scene_text=chunk.prompt_text(scene),
            ),
        )
        call = await self._generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage=f"narrative_repair:{scene.scene_id}:{chunk.order:04d}:01",
        )
        payload, structural_repairs = _normalize_narrative_structure(call.data)
        payload, empty_value_drops = _drop_empty_narrative_list_values(payload)
        payload, deterministic_drops = _drop_cross_family_narrative_duplicates(
            payload
        )
        payload, polarity_corrections = _normalize_narrative_surface_variants(payload)
        _validate_narrative_payload(payload)
        return payload, [
            {
                **call.metadata,
                "chunk_id": chunk.chunk_id,
                "repair_attempt": 1,
                "prompt_tokens_measured": self.token_counter.count(
                    system_prompt + user_prompt
                ),
                "deterministic_cross_family_duplicate_drops": deterministic_drops,
                "deterministic_cross_family_duplicate_drop_count": len(
                    deterministic_drops
                ),
                "deterministic_structure_repairs": structural_repairs,
                "deterministic_structure_repair_count": len(structural_repairs),
                "deterministic_empty_value_drops": empty_value_drops,
                "deterministic_empty_value_drop_count": len(empty_value_drops),
                "deterministic_polarity_corrections": polarity_corrections,
                "deterministic_polarity_correction_count": len(polarity_corrections),
            }
        ]

    async def _repair_entity_payload(
        self,
        *,
        scene: Scene,
        chunk: ScreenplayChunk,
        candidate: dict[str, Any] | str,
        validation_error: Exception,
        semantic_attempt: int,
        occasion_candidates: list[dict[str, str]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        current = candidate
        last_error = validation_error
        calls: list[dict[str, Any]] = []
        for repair_attempt in (1, 2):
            system_prompt, user_prompt = PROMPTS.render(
                "entity_repair",
                **extraction_prompt_values(
                    "entity_repair",
                    language=self.config.language,
                    validation_error=clean_text(last_error),
                    candidate_payload=_candidate_payload_text(current),
                    occasion_candidates=json.dumps(
                        occasion_candidates, ensure_ascii=False, indent=2
                    ),
                    scene_text=chunk.prompt_text(scene),
                ),
            )
            prompt_tokens = self.token_counter.count(system_prompt + user_prompt)
            if prompt_tokens > self.chunking_config.usable_model_input_tokens:
                raise ValueError(
                    "Entity repair prompt exceeds the configured model input budget: "
                    f"tokens={prompt_tokens} "
                    f"budget={self.chunking_config.usable_model_input_tokens}"
                )
            repair_stage = (
                f"entity_repair:{scene.scene_id}:{chunk.order:04d}:"
                f"{semantic_attempt:02d}:{repair_attempt:02d}"
            )
            try:
                call = await self._generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    stage=repair_stage,
                )
            except ModelResponseParseError as parse_error:
                calls.append(
                    {
                        **parse_error.metadata,
                        "chunk_id": chunk.chunk_id,
                        "semantic_attempt": semantic_attempt,
                        "repair_attempt": repair_attempt,
                        "parse_error": clean_text(parse_error),
                        "targeted_schema_retry": repair_attempt < 2,
                    }
                )
                last_error = parse_error
                continue
            current = call.data
            current, deterministic_entity_corrections = (
                _drop_unlocked_entity_candidates(current, occasion_candidates)
            )
            calls.append(
                {
                    **call.metadata,
                    "chunk_id": chunk.chunk_id,
                    "semantic_attempt": semantic_attempt,
                    "repair_attempt": repair_attempt,
                    "prompt_tokens_measured": prompt_tokens,
                    "deterministic_payload_corrections": (
                        deterministic_entity_corrections
                    ),
                }
            )
            try:
                _validate_entity_payload(current, occasion_candidates)
                return current, calls
            except Exception as exc:
                last_error = exc
        raise ValueError(
            "Entity repair failed strict validation after "
            f"two targeted repairs: {last_error}"
        ) from last_error

    async def _repair_relation_payload(
        self,
        *,
        scene: Scene,
        chunk: ScreenplayChunk,
        candidate: dict[str, Any] | str,
        entity_payload: dict[str, Any],
        validation_error: Exception,
        semantic_attempt: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        system_prompt, user_prompt = PROMPTS.render(
            "relation_repair",
            **extraction_prompt_values(
                "relation_repair",
                language=self.config.language,
                validation_error=clean_text(validation_error),
                candidate_payload=_candidate_payload_text(candidate),
                locked_entities=json.dumps(
                    _locked_entities_for_relation_prompt(entity_payload),
                    ensure_ascii=False,
                    indent=2,
                ),
                scene_text=chunk.prompt_text(scene),
            ),
        )
        call = await self._generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage=(
                f"relation_repair:{scene.scene_id}:{chunk.order:04d}:"
                f"{semantic_attempt:02d}:01"
            ),
        )
        repaired_payload, relation_schema_corrections = (
            _normalize_relation_payload_structure(call.data)
        )
        audit = {
            **call.metadata,
            "chunk_id": chunk.chunk_id,
            "semantic_attempt": semantic_attempt,
            "repair_attempt": 1,
            "prompt_tokens_measured": self.token_counter.count(
                system_prompt + user_prompt
            ),
            "deterministic_schema_corrections": relation_schema_corrections,
            "deterministic_schema_correction_count": len(
                relation_schema_corrections
            ),
        }
        try:
            _validate_relation_payload(repaired_payload, entity_payload)
            return repaired_payload, [audit]
        except ValueError as repair_error:
            sanitized, dropped = _drop_relations_with_unlocked_endpoints(
                repaired_payload, entity_payload
            )
            sanitized, filled_descriptions = _fill_missing_relation_descriptions(
                sanitized
            )
            sanitized, dropped_unusable = _drop_unusable_relations(sanitized)
            _validate_relation_payload(sanitized, entity_payload)
            mechanisms: list[str] = []
            if dropped:
                mechanisms.append("deterministic_drop_unlocked_endpoint_relations")
            if filled_descriptions:
                mechanisms.append("deterministic_fill_description_from_evidence")
            if dropped_unusable:
                mechanisms.append("deterministic_drop_ungrounded_relations")
            fallback = {
                "stage": (
                    f"relation_repair_fallback:{scene.scene_id}:"
                    f"{chunk.order:04d}:{semantic_attempt:02d}"
                ),
                "mechanism": "+".join(mechanisms),
                "chunk_id": chunk.chunk_id,
                "semantic_attempt": semantic_attempt,
                "validation_error": clean_text(repair_error),
                "dropped_relation_count": len(dropped),
                "dropped_relations": dropped,
                "filled_description_count": len(filled_descriptions),
                "filled_descriptions": filled_descriptions,
                "dropped_ungrounded_relation_count": len(dropped_unusable),
                "dropped_ungrounded_relations": dropped_unusable,
            }
            return sanitized, [audit, fallback]

    async def _generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any:
        async with self._semaphore:
            return await self.client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage=stage,
            )

    def _scene_content_budget(self, scene: Scene) -> int:
        header = ScreenplayChunk(
            chunk_id="",
            order=1,
            char_start=0,
            char_end=0,
            text="",
            token_count=0,
        ).prompt_text(scene)
        narrative_prompts = PROMPTS.render(
            "narrative_extraction",
            **extraction_prompt_values(
                "narrative_extraction",
                language=self.config.language,
                scene_text=header,
                prior_context="[]",
            ),
        )
        entity_prompts = PROMPTS.render(
            "entity_extraction",
            **extraction_prompt_values(
                "entity_extraction",
                language=self.config.language,
                scene_text=header,
                prior_context="[]",
                occasion_candidates="[]",
            ),
        )
        relation_prompts = PROMPTS.render(
            "relation_extraction",
            **extraction_prompt_values(
                "relation_extraction",
                language=self.config.language,
                scene_text=header,
                prior_context="[]",
                locked_entities="[]",
            ),
        )
        prompt_overheads = [
            self.token_counter.count(system + user)
            for system, user in (narrative_prompts, entity_prompts, relation_prompts)
        ]
        available = (
            self.chunking_config.usable_model_input_tokens
            - max(prompt_overheads)
            - self.chunking_config.carry_context_max_tokens
        )
        budget = min(self.chunking_config.target_chunk_tokens, available)
        if budget <= 0:
            raise ValueError(
                "Extraction prompt and carry-context budgets leave no room for scene content"
            )
        return budget

    async def _reconcile_scene_units(
        self,
        scene: Scene,
        units: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        original_leaf_ids = {
            source_id for unit in units for source_id in unit["source_unit_ids"]
        }
        current = list(units)
        audit: dict[str, Any] = {
            "enabled": True,
            "input_unit_count": len(units),
            "output_unit_count": len(units),
            "rounds": [],
            "llm_calls": [],
        }
        for round_index in range(1, 9):
            packs = _pack_units_for_reconciliation(
                current,
                language=self.config.language,
                token_counter=self.token_counter,
                max_input_tokens=self.chunking_config.reconciliation_max_input_tokens,
            )
            next_units: list[dict[str, Any]] = []
            round_calls: list[dict[str, Any]] = []
            for pack_index, pack in enumerate(packs, start=1):
                if len(pack) == 1:
                    next_units.extend(pack)
                    continue
                reconciled, metadata = await self._reconcile_unit_pack(
                    scene=scene,
                    units=pack,
                    round_index=round_index,
                    pack_index=pack_index,
                )
                next_units.extend(reconciled)
                round_calls.append(metadata)
            audit["rounds"].append(
                {
                    "round": round_index,
                    "pack_count": len(packs),
                    "input_unit_count": len(current),
                    "output_unit_count": len(next_units),
                }
            )
            audit["llm_calls"].extend(round_calls)
            previous_count = len(current)
            current = next_units
            if len(packs) == 1 or len(current) >= previous_count:
                break

        final_leaf_ids = [
            source_id for unit in current for source_id in unit["source_unit_ids"]
        ]
        if set(final_leaf_ids) != original_leaf_ids or len(final_leaf_ids) != len(
            original_leaf_ids
        ):
            raise ValueError("Scene reconciliation did not preserve exact leaf-unit coverage")
        audit["output_unit_count"] = len(current)
        return current, audit

    async def _reconcile_unit_pack(
        self,
        *,
        scene: Scene,
        units: list[dict[str, Any]],
        round_index: int,
        pack_index: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local_to_unit = {
            f"N{index:04d}": unit for index, unit in enumerate(units, start=1)
        }
        prompt_units = [
            {"local_unit_id": local_id, **_unit_prompt_record(unit)}
            for local_id, unit in local_to_unit.items()
        ]
        base_prompt = SCENE_RECONCILIATION_USER.format(
            language=self.config.language,
            narrative_units=json.dumps(prompt_units, ensure_ascii=False, indent=2),
        )
        prompt_tokens = self.token_counter.count(
            SCENE_RECONCILIATION_SYSTEM + base_prompt
        )
        if prompt_tokens > self.chunking_config.reconciliation_max_input_tokens:
            raise ValueError(
                f"Reconciliation pack exceeds input budget: tokens={prompt_tokens} "
                f"budget={self.chunking_config.reconciliation_max_input_tokens}"
            )
        last_error: Exception | None = None
        for semantic_attempt in range(1, max(1, self.config.semantic_attempts) + 1):
            try:
                call = await self.client.generate_json(
                    system_prompt=SCENE_RECONCILIATION_SYSTEM,
                    user_prompt=base_prompt + _validation_feedback(last_error),
                    stage=(
                        f"scene_reconciliation:{scene.scene_id}:"
                        f"{round_index:02d}:{pack_index:04d}"
                    ),
                )
                evidence_corrections: list[dict[str, Any]] = []
                group_corrections: list[dict[str, Any]] = []
                reconciled = _normalize_reconciliation_payload(
                    call.data,
                    local_to_unit=local_to_unit,
                    movie_id=self.movie_id,
                    scene=scene,
                    evidence_corrections=evidence_corrections,
                    group_corrections=group_corrections,
                )
                return reconciled, {
                    **call.metadata,
                    "semantic_attempt": semantic_attempt,
                    "round": round_index,
                    "pack": pack_index,
                    "input_unit_count": len(units),
                    "output_unit_count": len(reconciled),
                    "prompt_tokens": prompt_tokens,
                    "deterministic_evidence_correction_count": len(
                        evidence_corrections
                    ),
                    "deterministic_evidence_corrections": evidence_corrections,
                    "deterministic_group_correction_count": len(group_corrections),
                    "deterministic_group_corrections": group_corrections,
                }
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Scene reconciliation failed for {scene.scene_id}, round={round_index}, "
            f"pack={pack_index}: {last_error}"
        ) from last_error


def _validation_feedback(error: Exception | None) -> str:
    if error is None:
        return ""
    return (
        "\n\nThe previous output failed schema validation with this error: "
        f"{clean_text(error)}. Return a corrected object that satisfies every required field."
    )


def _candidate_payload_text(candidate: dict[str, Any] | str) -> str:
    if isinstance(candidate, str):
        return candidate
    return json.dumps(candidate, ensure_ascii=False, indent=2)


def _build_prior_context(
    chunk_records: list[dict[str, Any]],
    *,
    token_counter: TokenCounter,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in chunk_records:
        for entity in record.get("entities", []):
            candidates.append(
                {
                    "context_type": "entity",
                    "name": entity["name"],
                    "entity_type": entity["entity_type"],
                    "description": entity["description"],
                }
            )
        for unit in record.get("narrative_units", []):
            candidates.append(
                {
                    "context_type": unit["kind"],
                    "name": unit["name"],
                    "description": unit["description"],
                    "participants": unit.get("participants", []),
                }
            )
    if max_tokens <= 0 or not candidates:
        return "[]", {
            "candidate_items": len(candidates),
            "selected_items": 0,
            "skipped_items": len(candidates),
            "context_tokens": 0,
        }
    selected_reversed: list[dict[str, Any]] = []
    for item in reversed(candidates):
        projected = list(reversed([item, *selected_reversed]))
        encoded = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
        if token_counter.count(encoded) <= max_tokens:
            selected_reversed.append(item)
    selected = list(reversed(selected_reversed))
    context = json.dumps(selected, ensure_ascii=False, indent=2)
    return context, {
        "candidate_items": len(candidates),
        "selected_items": len(selected),
        "skipped_items": len(candidates) - len(selected),
        "context_tokens": token_counter.count(context),
    }


def _unit_prompt_record(unit: dict[str, Any]) -> dict[str, Any]:
    record = {
        "kind": unit["kind"],
        "name": unit["name"],
        "description": unit["description"],
        "modality": unit.get("modality", "asserted"),
        "participants": unit.get("participants", []),
        "setting": unit.get("setting", ""),
        "evidence": unit.get("evidence", ""),
        "source_chunk_order": unit.get("source_chunk_order", 0),
    }
    if unit["kind"] == "interaction":
        record.update(
            {
                "subject": unit.get("subject", ""),
                "object": unit.get("object", ""),
                "interaction_type": unit.get("interaction_type", "other"),
                "related_event": unit.get("related_event", ""),
                "related_occasion": unit.get("related_occasion", ""),
                "polarity": unit.get("polarity", "neutral"),
                "tags": unit.get("tags", []),
                "outcome": unit.get("outcome", ""),
                "locations": unit.get("locations", []),
                "times": unit.get("times", []),
            }
        )
    elif unit["kind"] == "event":
        for key in (
            "event_subtype", "state_before", "state_after", "intent",
            "cause_hints", "effect_hints", "locations", "times", "related_occasion",
        ):
            record[key] = unit.get(key, [] if key in {
                "cause_hints", "effect_hints", "locations", "times",
            } else "")
    elif unit["kind"] == "occasion":
        for key in (
            "occasion_type", "institutional_context", "locations", "times",
        ):
            record[key] = unit.get(
                key, [] if key in {"locations", "times"} else ""
            )
    return record


def _pack_units_for_reconciliation(
    units: list[dict[str, Any]],
    *,
    language: str,
    token_counter: TokenCounter,
    max_input_tokens: int,
) -> list[list[dict[str, Any]]]:
    packs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def prompt_tokens(values: list[dict[str, Any]]) -> int:
        records = [
            {"local_unit_id": f"N{index:04d}", **_unit_prompt_record(unit)}
            for index, unit in enumerate(values, start=1)
        ]
        prompt = SCENE_RECONCILIATION_USER.format(
            language=language,
            narrative_units=json.dumps(records, ensure_ascii=False, indent=2),
        )
        return token_counter.count(SCENE_RECONCILIATION_SYSTEM + prompt)

    for unit in units:
        projected = [*current, unit]
        if current and prompt_tokens(projected) > max_input_tokens:
            packs.append(current)
            current = [unit]
        else:
            current = projected
        if prompt_tokens(current) > max_input_tokens:
            raise ValueError(
                f"One narrative unit exceeds reconciliation input budget: {unit['unit_id']}"
            )
    if current:
        packs.append(current)
    return packs


def _normalize_reconciliation_payload(
    payload: dict[str, Any],
    *,
    local_to_unit: dict[str, dict[str, Any]],
    movie_id: str,
    scene: Scene,
    evidence_corrections: list[dict[str, Any]] | None = None,
    group_corrections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if set(payload) != {"groups"} or not isinstance(payload["groups"], list):
        raise ValueError("Scene reconciliation must return exactly a groups array")
    if not payload["groups"]:
        raise ValueError("Scene reconciliation returned no groups")
    seen_local_ids: list[str] = []
    output: list[dict[str, Any]] = []
    for raw in payload["groups"]:
        if not isinstance(raw, dict):
            raise ValueError("Every reconciliation group must be an object")
        local_ids = raw.get("source_unit_ids")
        if not isinstance(local_ids, list) or not local_ids:
            raise ValueError("Every reconciliation group requires source_unit_ids")
        local_ids = [clean_text(value) for value in local_ids]
        if any(local_id not in local_to_unit for local_id in local_ids):
            raise ValueError("Reconciliation group contains an unknown source_unit_id")
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("Reconciliation group contains duplicate source_unit_ids")
        source_units = [local_to_unit[local_id] for local_id in local_ids]
        source_kinds = {unit["kind"] for unit in source_units}
        kind = clean_text(raw.get("kind")).casefold()
        if len(source_kinds) != 1 or kind not in source_kinds:
            if group_corrections is None:
                raise ValueError("Reconciliation may only merge units of the same kind")
            group_corrections.append(
                {
                    "source_unit_ids": local_ids,
                    "source_kinds": sorted(source_kinds),
                    "generated_kind": kind,
                    "mechanism": "preserve_sources_for_invalid_kind_merge",
                }
            )
            output.extend(source_units)
            seen_local_ids.extend(local_ids)
            continue
        name = clean_text(raw.get("name"))
        description = clean_text(raw.get("description"))
        evidence = clean_text(raw.get("evidence"))
        if not name or not description or not evidence:
            raise ValueError("Every reconciled unit requires name, description, and evidence")
        allowed_evidence = {clean_text(unit.get("evidence")) for unit in source_units}
        if evidence not in allowed_evidence:
            replacement = clean_text(source_units[0].get("evidence"))
            if not replacement:
                raise ValueError("Reconciled evidence must be copied from a source unit")
            if evidence_corrections is not None:
                evidence_corrections.append(
                    {
                        "source_unit_ids": local_ids,
                        "generated_evidence": evidence,
                        "replacement_evidence": replacement,
                        "mechanism": "select_first_source_unit_evidence",
                    }
                )
            evidence = replacement
        allowed_participants = {
            normalize_name(participant): participant
            for unit in source_units
            for participant in unit.get("participants", [])
            if normalize_name(participant)
        }
        participants: list[str] = []
        for participant in unique_text(raw.get("participants") or [], limit=64):
            normalized = normalize_name(participant)
            if normalized not in allowed_participants:
                if group_corrections is None:
                    raise ValueError(
                        f"Reconciled participant is not grounded in source units: {participant}"
                    )
                group_corrections.append(
                    {
                        "source_unit_ids": local_ids,
                        "generated_participant": participant,
                        "mechanism": "drop_ungrounded_reconciled_participant",
                    }
                )
                continue
            participants.append(allowed_participants[normalized])
        leaf_ids = unique_text(
            source_id
            for unit in source_units
            for source_id in unit.get("source_unit_ids", [unit["unit_id"]])
        )
        source_chunk_ids = unique_text(
            chunk_id
            for unit in source_units
            for chunk_id in unit.get("source_chunk_ids", [unit.get("source_chunk_id")])
        )
        source_spans: list[dict[str, Any]] = []
        seen_spans: set[tuple[Any, Any, Any]] = set()
        for unit in source_units:
            for span in unit.get("source_spans", []):
                key = (span.get("chunk_id"), span.get("char_start"), span.get("char_end"))
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                source_spans.append(dict(span))
        unit_id = stable_id(
            "unit", movie_id, scene.scene_id, "reconciled", kind, *leaf_ids, name, evidence
        )
        unit = {
            "unit_id": unit_id,
            "source_unit_ids": leaf_ids,
            "reconciled_from_unit_ids": [unit["unit_id"] for unit in source_units],
            "kind": kind,
            "name": name,
            "description": description,
            "participants": participants,
            "setting": clean_text(raw.get("setting")),
            "evidence": evidence,
            "source_scene_id": scene.scene_id,
            "source_scene_raw_id": scene.source_scene_id,
            "source_scene_order": scene.order,
            "source_chunk_id": source_chunk_ids[0] if source_chunk_ids else "",
            "source_chunk_ids": source_chunk_ids,
            "source_chunk_order": min(
                int(unit.get("source_chunk_order", 0)) for unit in source_units
            ),
            "source_spans": source_spans,
            "reconciled": True,
        }
        if kind == "event":
            unit.update(_merge_event_fields(source_units))
        elif kind == "occasion":
            unit.update(_merge_occasion_fields(source_units))
        if kind == "interaction":
            subject = clean_text(raw.get("subject"))
            object_name = clean_text(raw.get("object"))
            if not subject or normalize_name(subject) not in allowed_participants:
                raise ValueError("Reconciled interaction requires a grounded subject")
            if object_name and normalize_name(object_name) not in allowed_participants:
                raise ValueError("Reconciled interaction object is not grounded")
            unit.update(
                {
                    "subject": allowed_participants[normalize_name(subject)],
                    "object": (
                        allowed_participants[normalize_name(object_name)]
                        if object_name
                        else ""
                    ),
                    "interaction_type": _require_open_label(
                        raw.get("interaction_type"), "reconciled interaction_type"
                    ),
                    "related_event": clean_text(raw.get("related_event")),
                    "related_occasion": clean_text(raw.get("related_occasion")),
                    **_merge_interaction_fields(source_units),
                }
            )
            unit["participants"] = unique_text(
                [unit["subject"], unit["object"], *unit["participants"]], limit=64
            )
        output_participants = {
            normalize_name(participant)
            for participant in unit["participants"]
            if normalize_name(participant)
        }
        missing_participants = [
            canonical
            for normalized, canonical in allowed_participants.items()
            if normalized not in output_participants
        ]
        if missing_participants:
            if group_corrections is None:
                raise ValueError(
                    "Reconciliation must preserve the exact participant union of source units"
                )
            group_corrections.append(
                {
                    "source_unit_ids": local_ids,
                    "missing_source_participants": missing_participants,
                    "mechanism": "restore_source_participant_union",
                }
            )
            unit["participants"] = unique_text(
                [*unit["participants"], *missing_participants], limit=64
            )
            output_participants = {
                normalize_name(participant)
                for participant in unit["participants"]
                if normalize_name(participant)
            }
        if output_participants != set(allowed_participants):
            raise ValueError(
                "Reconciliation must preserve the exact participant union of source units"
            )
        output.append(unit)
        seen_local_ids.extend(local_ids)
    expected = set(local_to_unit)
    if set(seen_local_ids) != expected or len(seen_local_ids) != len(expected):
        raise ValueError("Scene reconciliation source-unit coverage is not exactly one-to-one")
    return sorted(
        output,
        key=lambda unit: (
            int(unit.get("source_chunk_order", 0)),
            min(
                (int(span.get("char_start", 0)) for span in unit.get("source_spans", [])),
                default=0,
            ),
            unit["unit_id"],
        ),
    )


def _validate_entity_payload(
    payload: dict[str, Any],
    occasion_candidates: list[dict[str, str]] | None = None,
) -> None:
    if set(payload) != {"entities"}:
        raise ValueError("Entity extraction must return exactly an entities array")
    if not isinstance(payload["entities"], list):
        raise ValueError("Entity extraction entities field must be an array")
    for entity in payload["entities"]:
        if not isinstance(entity, dict) or not clean_text(entity.get("name")):
            raise ValueError("Every entity requires a non-empty name")
        if set(entity) != {
            "name", "entity_type", "scope", "description", "aliases", "evidence"
        }:
            raise ValueError("Every entity must match the exact entity output schema")
        if not clean_text(entity.get("description")) or not clean_text(entity.get("evidence")):
            raise ValueError("Every entity requires a grounded description and evidence")
        require_entity_type(entity.get("entity_type"))
        require_entity_scope(entity.get("scope"))
        aliases = entity.get("aliases")
        if not isinstance(aliases, list) or any(not clean_text(alias) for alias in aliases):
            raise ValueError("Every entity aliases field must be an array of non-empty names")
    extracted = [
        (clean_text(entity.get("name")).casefold(), require_entity_type(entity.get("entity_type")))
        for entity in payload["entities"]
    ]
    extracted_occasions = sorted(
        name for name, entity_type in extracted if entity_type == "Occasion"
    )
    expected_occasions = sorted(
        clean_text(occasion.get("name")).casefold()
        for occasion in occasion_candidates or []
    )
    if extracted_occasions != expected_occasions:
        raise ValueError(
            "Occasion entities must exactly match the pre-extracted narrative candidates: "
            f"expected={expected_occasions}, extracted={extracted_occasions}"
        )


def _drop_unlocked_entity_candidates(
    payload: dict[str, Any],
    occasion_candidates: list[dict[str, str]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        return payload, []
    expected_occasions = {
        clean_text(item.get("name")).casefold()
        for item in occasion_candidates or []
        if clean_text(item.get("name"))
    }
    kept = []
    corrections = []
    for entity in payload["entities"]:
        if not isinstance(entity, dict):
            kept.append(entity)
            continue
        type_key = re.sub(r"[\s_-]+", "", clean_text(entity.get("entity_type"))).casefold()
        name = clean_text(entity.get("name"))
        if type_key in {"event", "events", "interaction", "interactions"}:
            corrections.append(
                {
                    "action": "drop_non_kg_narrative_entity",
                    "name": name,
                    "entity_type": entity.get("entity_type"),
                }
            )
            continue
        if type_key in {"occasion", "occasions"} and name.casefold() not in expected_occasions:
            corrections.append(
                {
                    "action": "drop_unlocked_occasion_entity",
                    "name": name,
                    "expected_occasion_names": sorted(expected_occasions),
                }
            )
            continue
        kept.append(entity)
    return {**payload, "entities": kept}, corrections


def _validate_relation_payload(
    payload: dict[str, Any], entity_payload: dict[str, Any]
) -> None:
    if set(payload) != {"relations"} or not isinstance(payload["relations"], list):
        raise ValueError("Relation extraction must return exactly a relations array")
    known_names = {
        clean_text(name).casefold()
        for entity in entity_payload["entities"]
        for name in [entity.get("name"), *(entity.get("aliases") or [])]
        if clean_text(name)
    }
    for relation in payload["relations"]:
        if not isinstance(relation, dict):
            raise ValueError("Every entity relation must be an object")
        if set(relation) != {"subject", "predicate", "object", "description", "evidence"}:
            raise ValueError("Every entity relation must match the exact relation output schema")
        if not all(clean_text(relation.get(key)) for key in ("subject", "predicate", "object")):
            raise ValueError("Every entity relation requires subject, predicate, and object")
        if not clean_text(relation.get("description")):
            raise ValueError("Every entity relation requires a grounded description")
        if not clean_text(relation.get("evidence")):
            raise ValueError("Every entity relation requires source evidence")
        endpoints = {
            clean_text(relation.get("subject")).casefold(),
            clean_text(relation.get("object")).casefold(),
        }
        unknown_endpoints = sorted(endpoints - known_names)
        if unknown_endpoints:
            raise ValueError(
                "Entity relation endpoints must exactly match an extracted entity name or alias. "
                f"Undeclared endpoints: {unknown_endpoints}. "
                f"Declared endpoints: {sorted(known_names)}. "
                "Add each grounded endpoint to entities or omit its unsupported relation."
            )


def _normalize_relation_payload_structure(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project relation objects to the frozen schema without inventing values."""
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), list):
        return payload, []
    allowed = {"subject", "predicate", "object", "description", "evidence"}
    corrections: list[dict[str, Any]] = []
    relations: list[Any] = []
    for index, raw in enumerate(payload["relations"]):
        if not isinstance(raw, dict):
            relations.append(raw)
            continue
        extra = sorted(set(raw) - allowed)
        if extra:
            corrections.append(
                {
                    "action": "drop_relation_schema_extra_fields",
                    "relation_index": index,
                    "dropped_fields": extra,
                }
            )
        relations.append({key: raw[key] for key in allowed if key in raw})
    return {"relations": relations}, corrections


def _locked_entities_for_relation_prompt(
    entity_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "name": clean_text(entity.get("name")),
            "entity_type": require_entity_type(entity.get("entity_type")),
            "aliases": unique_text(entity.get("aliases") or []),
            "description": clean_text(entity.get("description")),
        }
        for entity in entity_payload["entities"]
    ]


def _drop_relations_with_unlocked_endpoints(
    payload: dict[str, Any], entity_payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(payload) != {"relations"} or not isinstance(payload.get("relations"), list):
        raise ValueError("Cannot sanitize a malformed relation payload")
    known_names = {
        clean_text(name).casefold()
        for entity in entity_payload["entities"]
        for name in [entity.get("name"), *(entity.get("aliases") or [])]
        if clean_text(name)
    }
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for index, relation in enumerate(payload["relations"]):
        if not isinstance(relation, dict):
            raise ValueError("Cannot sanitize a non-object relation")
        endpoints = {
            clean_text(relation.get("subject")).casefold(),
            clean_text(relation.get("object")).casefold(),
        }
        unlocked = sorted(value for value in endpoints - known_names if value)
        if not unlocked:
            kept.append(relation)
            continue
        dropped.append(
            {
                "relation_index": index,
                "subject": clean_text(relation.get("subject")),
                "predicate": clean_text(relation.get("predicate")),
                "object": clean_text(relation.get("object")),
                "unlocked_endpoints": unlocked,
            }
        )
    return {"relations": kept}, dropped


def _fill_missing_relation_descriptions(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(payload) != {"relations"} or not isinstance(payload.get("relations"), list):
        raise ValueError("Cannot sanitize a malformed relation payload")
    relations: list[Any] = []
    filled: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["relations"]):
        if not isinstance(raw, dict):
            relations.append(raw)
            continue
        relation = dict(raw)
        evidence = clean_text(relation.get("evidence"))
        if not clean_text(relation.get("description")) and evidence:
            relation["description"] = evidence
            filled.append(
                {
                    "relation_index": index,
                    "subject": clean_text(relation.get("subject")),
                    "predicate": clean_text(relation.get("predicate")),
                    "object": clean_text(relation.get("object")),
                    "source": "evidence",
                }
            )
        relations.append(relation)
    return {"relations": relations}, filled


def _drop_unusable_relations(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep only relations that can be grounded and audited deterministically."""
    if set(payload) != {"relations"} or not isinstance(payload.get("relations"), list):
        raise ValueError("Cannot sanitize a malformed relation payload")
    required = {"subject", "predicate", "object", "description", "evidence"}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for index, relation in enumerate(payload["relations"]):
        reason = None
        if not isinstance(relation, dict):
            reason = "relation_is_not_an_object"
        elif set(relation) != required:
            reason = "relation_schema_incomplete"
        elif not all(
            clean_text(relation.get(key))
            for key in ("subject", "predicate", "object", "description", "evidence")
        ):
            reason = "relation_lacks_grounded_field"
        if reason is not None:
            dropped.append(
                {
                    "relation_index": index,
                    "reason": reason,
                    "subject": clean_text(relation.get("subject"))
                    if isinstance(relation, dict)
                    else "",
                    "predicate": clean_text(relation.get("predicate"))
                    if isinstance(relation, dict)
                    else "",
                    "object": clean_text(relation.get("object"))
                    if isinstance(relation, dict)
                    else "",
                }
            )
            continue
        kept.append(relation)
    return {"relations": kept}, dropped


def _normalize_narrative_surface_variants(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize unambiguous closed-set surface variants without a retry."""
    if not isinstance(payload, dict) or not isinstance(payload.get("interactions"), list):
        return payload, []
    corrections: list[dict[str, Any]] = []
    for index, interaction in enumerate(payload["interactions"]):
        if not isinstance(interaction, dict):
            continue
        raw = clean_text(interaction.get("polarity")).casefold()
        normalized = re.sub(r"[\s_-]+", "_", raw)
        replacement = INTERACTION_POLARITY_ALIASES.get(normalized)
        if replacement is None or replacement == raw:
            continue
        interaction["polarity"] = replacement
        corrections.append(
            {
                "action": "normalize_interaction_polarity_alias",
                "interaction_index": index,
                "requested_polarity": raw,
                "resolved_polarity": replacement,
            }
        )
    for family in ("events", "occasions", "interactions"):
        for index, item in enumerate(payload.get(family, [])):
            if not isinstance(item, dict):
                continue
            raw = clean_text(item.get("modality")).casefold()
            normalized = re.sub(r"[\s-]+", "_", raw)
            replacement = NARRATIVE_MODALITY_ALIASES.get(normalized)
            if replacement is None or replacement == raw:
                continue
            item["modality"] = replacement
            corrections.append(
                {
                    "action": "normalize_narrative_modality_alias",
                    "family": family,
                    "item_index": index,
                    "requested_modality": raw,
                    "resolved_modality": replacement,
                }
            )
    return payload, corrections


def _validate_narrative_payload(payload: dict[str, Any]) -> None:
    expected = {"events", "occasions", "interactions"}
    if set(payload) != expected:
        raise ValueError(f"Narrative extraction must return exactly {sorted(expected)}")
    if any(not isinstance(payload[key], list) for key in expected):
        raise ValueError("Every narrative extraction field must be an array")
    item_fields = {
        key: set(defaults) for key, defaults in NARRATIVE_ITEM_DEFAULTS.items()
    }
    for key in ("events", "occasions", "interactions"):
        for unit in payload[key]:
            if not isinstance(unit, dict) or not clean_text(unit.get("name")):
                raise ValueError(f"Every {key} item requires a non-empty name")
            if set(unit) != item_fields[key]:
                raise ValueError(
                    f"Every {key} item must match the exact output schema; "
                    f"missing={sorted(item_fields[key] - set(unit))}, "
                    f"extra={sorted(set(unit) - item_fields[key])}"
                )
            if not clean_text(unit.get("description")):
                raise ValueError(f"Every {key} item requires a non-empty description")
            if not clean_text(unit.get("evidence")):
                raise ValueError(f"Every {key} item requires source evidence")
    for interaction in payload["interactions"]:
        if not clean_text(interaction.get("subject")):
            raise ValueError("Every interaction requires a subject")
    for event in payload["events"]:
        subtype = _require_open_label(event.get("event_subtype"), "events event_subtype")
        event["event_subtype"] = subtype
    for occasion in payload["occasions"]:
        occasion["occasion_type"] = _require_open_label(
            occasion.get("occasion_type"), "occasions occasion_type"
        )
    for interaction in payload["interactions"]:
        interaction["interaction_type"] = _require_open_label(
            interaction.get("interaction_type"), "interactions interaction_type"
        )
        interaction["polarity"] = _require_closed_label(
            interaction.get("polarity"),
            INTERACTION_POLARITIES,
            "interactions polarity",
        )
    for key in ("events", "occasions", "interactions"):
        for unit in payload[key]:
            unit["modality"] = _require_closed_label(
                unit.get("modality"),
                NARRATIVE_MODALITIES,
                f"{key} modality",
            )
    list_fields = {
        "events": (
            "participants", "locations", "times", "cause_hints", "effect_hints",
        ),
        "occasions": ("participants", "locations", "times"),
        "interactions": ("participants", "locations", "times", "tags"),
    }
    for key in ("events", "occasions", "interactions"):
        for unit in payload[key]:
            for field in list_fields[key]:
                values = unit.get(field)
                if not isinstance(values, list):
                    raise ValueError(f"Every {key} {field} field must be an array")
                if any(not clean_text(value) for value in values):
                    raise ValueError(f"Every {key} {field} value must be non-empty")


def _occasion_candidates(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "name": clean_text(raw.get("name")),
            "entity_type": "Occasion",
            "description": clean_text(raw.get("description")),
            "evidence": clean_text(raw.get("evidence")),
        }
        for raw in payload.get("occasions", [])
    ]


def _require_closed_label(value: Any, allowed: frozenset[str], field: str) -> str:
    raw = clean_text(value)
    normalized = _normalize_closed_label(raw)
    if normalized not in allowed:
        raise ValueError(
            f"Every {field} must be one of {sorted(allowed)}; received {raw!r}"
        )
    return normalized


def _require_open_label(value: Any, field: str) -> str:
    normalized = _normalize_closed_label(value)
    if not normalized:
        raise ValueError(f"Every {field} requires a non-empty normalized label")
    return normalized


def _drop_cross_family_narrative_duplicates(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(payload) != {"events", "occasions", "interactions"}:
        return payload, []
    if any(not isinstance(payload.get(key), list) for key in payload):
        return payload, []

    interactions = [
        item for item in payload["interactions"] if isinstance(item, dict)
    ]
    kept_events: list[Any] = []
    dropped: list[dict[str, Any]] = []
    for index, event in enumerate(payload["events"]):
        if not isinstance(event, dict):
            kept_events.append(event)
            continue
        duplicate = next(
            (
                interaction
                for interaction in interactions
                if not _event_has_independent_change(event)
                and _same_or_contained_evidence(
                    event.get("evidence"), interaction.get("evidence")
                )
            ),
            None,
        )
        if duplicate is None:
            kept_events.append(event)
            continue
        dropped.append(
            {
                "family": "events",
                "index": index,
                "name": clean_text(event.get("name")),
                "event_subtype": clean_text(event.get("event_subtype")),
                "matched_interaction_name": clean_text(duplicate.get("name")),
                "evidence": clean_text(event.get("evidence")),
                "mechanism": "cross_family_exact_evidence_duplicate",
            }
        )
    return {**payload, "events": kept_events}, dropped


def _normalize_narrative_structure(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(payload) != set(NARRATIVE_ITEM_DEFAULTS):
        return payload, []
    if any(not isinstance(payload.get(key), list) for key in NARRATIVE_ITEM_DEFAULTS):
        return payload, []
    normalized: dict[str, list[Any]] = {}
    repairs: list[dict[str, Any]] = []
    for family, defaults in NARRATIVE_ITEM_DEFAULTS.items():
        rows: list[Any] = []
        for index, raw in enumerate(payload[family]):
            if not isinstance(raw, dict):
                rows.append(raw)
                continue
            removed = sorted(set(raw) - set(defaults))
            filled = sorted(set(defaults) - set(raw))
            row = {
                field: (
                    list(default) if isinstance(default, list) else default
                )
                for field, default in defaults.items()
            }
            row.update({field: raw[field] for field in defaults if field in raw})
            rows.append(row)
            if removed or filled:
                repairs.append(
                    {
                        "family": family,
                        "index": index,
                        "name": clean_text(raw.get("name")),
                        "removed_fields": removed,
                        "filled_fields": filled,
                        "mechanism": "project_to_declared_narrative_schema",
                    }
                )
        normalized[family] = rows
    return normalized, repairs


def _drop_empty_narrative_list_values(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Remove empty list members without inventing narrative content."""
    if not isinstance(payload, dict):
        return payload, []
    corrections: list[dict[str, Any]] = []
    for family, defaults in NARRATIVE_ITEM_DEFAULTS.items():
        rows = payload.get(family)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            for field, default in defaults.items():
                if not isinstance(default, list) or not isinstance(row.get(field), list):
                    continue
                original = row[field]
                cleaned = [clean_text(value) for value in original if clean_text(value)]
                if len(cleaned) == len(original) and all(
                    isinstance(value, str) for value in original
                ):
                    continue
                row[field] = cleaned
                corrections.append(
                    {
                        "action": "drop_empty_narrative_list_values",
                        "family": family,
                        "index": index,
                        "field": field,
                        "dropped_count": len(original) - len(cleaned),
                    }
                )
    return payload, corrections


def _normalize_closed_label(value: Any) -> str:
    return re.sub(r"_+", "_", re.sub(r"[\s-]+", "_", clean_text(value).casefold()))


def _same_or_contained_evidence(left: Any, right: Any) -> bool:
    left_key = normalize_name(left)
    right_key = normalize_name(right)
    if min(len(left_key), len(right_key)) < 8:
        return False
    return left_key == right_key or left_key in right_key or right_key in left_key


def _event_has_independent_change(event: dict[str, Any]) -> bool:
    return bool(
        clean_text(event.get("state_before"))
        or clean_text(event.get("state_after"))
        or unique_text(event.get("effect_hints") or [])
    )


def _merged_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    return unique_text(value for row in rows for value in row.get(field, []))


def _first_value(rows: list[dict[str, Any]], field: str) -> str:
    return next((clean_text(row.get(field)) for row in rows if clean_text(row.get(field))), "")


def _last_value(rows: list[dict[str, Any]], field: str) -> str:
    return next(
        (clean_text(row.get(field)) for row in reversed(rows) if clean_text(row.get(field))),
        "",
    )


def _merge_event_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_subtype": _first_value(rows, "event_subtype") or "other",
        "modality": _first_value(rows, "modality") or "asserted",
        "state_before": _first_value(rows, "state_before"),
        "state_after": _last_value(rows, "state_after"),
        "intent": _first_value(rows, "intent"),
        "cause_hints": _merged_values(rows, "cause_hints"),
        "effect_hints": _merged_values(rows, "effect_hints"),
        "locations": _merged_values(rows, "locations"),
        "times": _merged_values(rows, "times"),
        "related_occasion": _first_value(rows, "related_occasion"),
    }


def _merge_occasion_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "occasion_type": _first_value(rows, "occasion_type") or "other",
        "modality": _first_value(rows, "modality") or "asserted",
        "institutional_context": _first_value(rows, "institutional_context"),
        "locations": _merged_values(rows, "locations"),
        "times": _merged_values(rows, "times"),
    }


def _merge_interaction_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "modality": _first_value(rows, "modality") or "asserted",
        "polarity": _first_value(rows, "polarity") or "neutral",
        "tags": _merged_values(rows, "tags"),
        "outcome": _last_value(rows, "outcome"),
        "locations": _merged_values(rows, "locations"),
        "times": _merged_values(rows, "times"),
    }


def _normalize_chunk_result(
    *,
    movie_id: str,
    scene: Scene,
    chunk: ScreenplayChunk,
    entity_payload: dict[str, Any],
    relation_payload: dict[str, Any],
    narrative_payload: dict[str, Any],
) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    for index, raw in enumerate(entity_payload["entities"], start=1):
        name = clean_text(raw.get("name"))
        entity_type = require_entity_type(raw.get("entity_type"))
        aliases = unique_text(raw.get("aliases") or [], limit=24)
        entities.append(
            {
                "mention_id": stable_id(
                    "mention", movie_id, scene.scene_id, chunk.chunk_id, index, name
                ),
                "name": name,
                "entity_type": entity_type,
                "scope": require_entity_scope(raw.get("scope")),
                "description": clean_text(raw.get("description")),
                "aliases": aliases,
                "evidence": clean_text(raw.get("evidence")),
                "source_scene_id": scene.scene_id,
                "source_scene_raw_id": scene.source_scene_id,
                "source_scene_order": scene.order,
                "source_chunk_id": chunk.chunk_id,
                "source_chunk_order": chunk.order,
                "source_char_start": chunk.char_start,
                "source_char_end": chunk.char_end,
            }
        )

    relations: list[dict[str, Any]] = []
    for index, raw in enumerate(relation_payload["relations"], start=1):
        subject = clean_text(raw.get("subject"))
        predicate = clean_text(raw.get("predicate"))
        object_name = clean_text(raw.get("object"))
        relations.append(
            {
                "relation_id": stable_id(
                    "semantic-relation",
                    movie_id,
                    scene.scene_id,
                    chunk.chunk_id,
                    index,
                    subject,
                    predicate,
                    object_name,
                ),
                "subject": subject,
                "predicate": predicate,
                "object": object_name,
                "description": clean_text(raw.get("description")),
                "evidence": clean_text(raw.get("evidence")),
                "source_scene_id": scene.scene_id,
                "source_scene_raw_id": scene.source_scene_id,
                "source_scene_order": scene.order,
                "source_chunk_id": chunk.chunk_id,
                "source_chunk_order": chunk.order,
                "source_char_start": chunk.char_start,
                "source_char_end": chunk.char_end,
            }
        )

    narrative_units: list[dict[str, Any]] = []
    kind_to_key = {"event": "events", "occasion": "occasions", "interaction": "interactions"}
    for kind in BASE_UNIT_KINDS:
        for index, raw in enumerate(narrative_payload[kind_to_key[kind]], start=1):
            name = clean_text(raw.get("name"))
            evidence = clean_text(raw.get("evidence"))
            participants = unique_text(raw.get("participants") or [], limit=48)
            if kind == "interaction":
                participants = unique_text(
                    [raw.get("subject"), raw.get("object"), *participants], limit=48
                )
            unit = {
                "unit_id": stable_id(
                    "unit",
                    movie_id,
                    scene.scene_id,
                    chunk.chunk_id,
                    kind,
                    index,
                    name,
                    evidence,
                ),
                "kind": kind,
                "name": name,
                "description": clean_text(raw.get("description")),
                "participants": participants,
                "setting": clean_text(raw.get("setting")),
                "evidence": evidence,
                "source_scene_id": scene.scene_id,
                "source_scene_raw_id": scene.source_scene_id,
                "source_scene_order": scene.order,
                "source_chunk_id": chunk.chunk_id,
                "source_chunk_ids": [chunk.chunk_id],
                "source_chunk_order": chunk.order,
                "source_spans": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                    }
                ],
            }
            unit["source_unit_ids"] = [unit["unit_id"]]
            if kind == "interaction":
                unit.update(
                    {
                        "subject": clean_text(raw.get("subject")),
                        "object": clean_text(raw.get("object")),
                        "interaction_type": clean_text(raw.get("interaction_type")) or "other",
                        "related_event": clean_text(raw.get("related_event")),
                        "related_occasion": clean_text(raw.get("related_occasion")),
                        "polarity": clean_text(raw.get("polarity")) or "neutral",
                        "tags": unique_text(raw.get("tags") or [], limit=24),
                        "outcome": clean_text(raw.get("outcome")),
                        "locations": unique_text(raw.get("locations") or [], limit=24),
                        "times": unique_text(raw.get("times") or [], limit=24),
                    }
                )
            elif kind == "event":
                unit.update(_merge_event_fields([raw]))
            elif kind == "occasion":
                unit.update(_merge_occasion_fields([raw]))
            narrative_units.append(unit)

    return {
        "entities": entities,
        "entity_relations": relations,
        "narrative_units": narrative_units,
    }
