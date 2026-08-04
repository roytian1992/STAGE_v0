from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from typing import Any, Protocol

from .models import clean_text
from .prompt_loader import PROMPTS
from .relation_resolution import (
    PREDICATE_REGISTRY,
    _aggregate_relations,
    _collect_observations,
    _validate_predicate_decisions,
    find_contradiction_groups,
    predicate_allows_observation,
)
from .review_protocol import (
    ImmutableReviewCache,
    budget_record,
    review_cache_key,
)


_PREDICATE_SCHEMA_REPAIR_PROMPT = PROMPTS.get("predicate_schema_repair")
PREDICATE_SCHEMA_REPAIR_SYSTEM = _PREDICATE_SCHEMA_REPAIR_PROMPT.system
PREDICATE_SCHEMA_REPAIR_USER = _PREDICATE_SCHEMA_REPAIR_PROMPT.user


REPAIR_OUTPUT_SCHEMA = {
    "decisions": [
        {
            "observation_id": "string",
            "action": "keep|drop",
            "predicate_id": "registered predicate or empty",
            "reverse_direction": "boolean",
            "relation_class": "stable|temporal|event_like",
            "reason_code": "controlled reason",
            "generated_rationale_hint": "string",
        }
    ]
}


class JsonClient(Protocol):
    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class RelationRepairConfig:
    batch_size: int = 8
    max_concurrency: int = 4


def analyze_relation_revision(
    scene_records: list[dict[str, Any]],
    prior_registry: dict[str, Any],
) -> dict[str, Any]:
    observations = _collect_observations(scene_records)
    prior_by_id = {
        item.get("observation_id"): item
        for item in prior_registry.get("observations", [])
        if item.get("observation_id")
    }
    unaffected: list[dict[str, Any]] = []
    affected: list[dict[str, Any]] = []
    for observation in observations:
        prior = prior_by_id.get(observation["observation_id"])
        error = _prior_decision_error(observation, prior)
        if error:
            affected.append(
                {
                    "observation": observation,
                    "prior_decision": (prior or {}).get("normalization"),
                    "validation_error": error,
                }
            )
        else:
            unaffected.append(
                {
                    "observation": observation,
                    "decision": dict(prior["normalization"]),
                }
            )
    current_ids = {item["observation_id"] for item in observations}
    return {
        "observations": observations,
        "unaffected": unaffected,
        "affected": affected,
        "orphan_prior_observation_ids": sorted(set(prior_by_id) - current_ids),
    }


def build_repair_packs(
    analysis: dict[str, Any], batch_size: int
) -> list[list[dict[str, Any]]]:
    affected = analysis["affected"]
    size = max(1, batch_size)
    return [affected[start : start + size] for start in range(0, len(affected), size)]


def build_relation_repair_budget(
    *,
    scene_records: list[dict[str, Any]],
    prior_registry: dict[str, Any],
    client: Any,
    cache: ImmutableReviewCache,
    batch_size: int,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    analysis = analyze_relation_revision(scene_records, prior_registry)
    packs = build_repair_packs(analysis, batch_size)
    records = []
    for index, pack in enumerate(packs):
        input_payload, user_prompt = _repair_prompt(pack)
        cache_key, hashes = review_cache_key(
            namespace="relation_schema_repair",
            input_payload=input_payload,
            system_prompt=PREDICATE_SCHEMA_REPAIR_SYSTEM,
            user_prompt=user_prompt,
            output_schema=REPAIR_OUTPUT_SCHEMA,
            client=client,
        )
        record = budget_record(
            namespace="relation_schema_repair",
            cache_key=cache_key,
            hashes=hashes,
            cache=cache,
            item_count=len(pack),
            call_kind="targeted_repair",
        )
        record.update(
            {
                "pack_index": index,
                "observation_ids": [
                    item["observation"]["observation_id"] for item in pack
                ],
                "prompt_tokens_measured": token_counter.count(
                    PREDICATE_SCHEMA_REPAIR_SYSTEM + "\n" + user_prompt
                )
                if token_counter
                else None,
            }
        )
        records.append(record)
    cache_hits = sum(item["cache_hit"] for item in records)
    return {
        "schema_version": "stage_relation_repair_budget_v1",
        "total_observations": len(analysis["observations"]),
        "preserved_decisions": len(analysis["unaffected"]),
        "affected_observations": len(analysis["affected"]),
        "orphan_prior_observation_ids": analysis["orphan_prior_observation_ids"],
        "repair_pack_count": len(packs),
        "cache_hit_count": cache_hits,
        "required_model_calls": len(packs) - cache_hits,
        "maximum_semantic_calls": len(packs) - cache_hits,
        "repair_of_repair_calls": 0,
        "packs": records,
    }


class RelationSchemaRepairer:
    def __init__(
        self,
        *,
        movie_id: str,
        llm_client: JsonClient,
        cache: ImmutableReviewCache,
        config: RelationRepairConfig,
        token_counter: TokenCounter | None = None,
        max_input_tokens: int | None = None,
    ):
        self.movie_id = movie_id
        self.llm_client = llm_client
        self.cache = cache
        self.config = config
        self.token_counter = token_counter
        self.max_input_tokens = max_input_tokens

    async def repair(
        self,
        scene_records: list[dict[str, Any]],
        prior_registry: dict[str, Any],
    ) -> dict[str, Any]:
        analysis = analyze_relation_revision(scene_records, prior_registry)
        packs = build_repair_packs(analysis, self.config.batch_size)
        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def run_pack(index: int, pack: list[dict[str, Any]]):
            async with semaphore:
                return await self._repair_pack(index, pack)

        results = await asyncio.gather(
            *(run_pack(index, pack) for index, pack in enumerate(packs))
        )
        results.sort(key=lambda item: item[0])
        repaired_decisions = {
            item["observation_id"]: item
            for _, rows, _, _ in results
            for item in rows
        }
        human_queue = [item for _, _, items, _ in results for item in items]
        decisions = {
            item["observation"]["observation_id"]: item["decision"]
            for item in analysis["unaffected"]
        }
        decisions.update(repaired_decisions)
        for queued in human_queue:
            observation_id = queued["observation_id"]
            decisions[observation_id] = {
                "observation_id": observation_id,
                "action": "human_review_required",
                "predicate_id": "",
                "reverse_direction": False,
                "relation_class": "event_like",
                "reason_code": "invalid_targeted_repair",
                "generated_rationale_hint": "",
                "normalization_reason": queued["validation_error"],
            }

        normalized = [
            _materialize_observation(observation, decisions[observation["observation_id"]])
            for observation in analysis["observations"]
        ]
        canonical_relations = _aggregate_relations(self.movie_id, normalized)
        contradictions = find_contradiction_groups(canonical_relations)
        all_decisions = [decisions[item["observation_id"]] for item in analysis["observations"]]
        calls = [metadata for _, _, _, metadata in results]
        return {
            "schema_version": "stage_relation_registry_v3",
            "predicate_registry": PREDICATE_REGISTRY,
            "observations": normalized,
            "canonical_relations": canonical_relations,
            "human_review_required": human_queue,
            "audit": {
                "input_count": len(normalized),
                "preserved_decision_count": len(analysis["unaffected"]),
                "affected_observation_count": len(analysis["affected"]),
                "repaired_decision_count": len(repaired_decisions),
                "human_review_required_count": len(human_queue),
                "kept_count": sum(item["action"] == "keep" for item in all_decisions),
                "dropped_count": sum(item["action"] == "drop" for item in all_decisions),
                "input_surface_predicate_count": len(
                    {item["surface_predicate"] for item in normalized}
                ),
                "canonical_predicate_count": len(
                    {item["predicate_id"] for item in all_decisions if item["action"] == "keep"}
                ),
                "canonical_relation_count": len(canonical_relations),
                "duplicate_observation_reduction_count": max(
                    0,
                    sum(item["action"] == "keep" for item in all_decisions)
                    - len(canonical_relations),
                ),
                "predicate_domain_range_drop_count": 0,
                "contradiction_group_count": len(contradictions),
                "contradiction_groups": contradictions,
                "decision_calls": calls,
                "decisions": all_decisions,
                "orphan_prior_observation_ids": analysis[
                    "orphan_prior_observation_ids"
                ],
                "policy": (
                    "Unchanged decisions were preserved exactly; each invalid decision pack "
                    "received at most one cached targeted-repair call."
                ),
            },
        }

    async def _repair_pack(
        self, index: int, pack: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        input_payload, user_prompt = _repair_prompt(pack)
        if self.token_counter and self.max_input_tokens is not None:
            measured = self.token_counter.count(
                PREDICATE_SCHEMA_REPAIR_SYSTEM + "\n" + user_prompt
            )
            if measured > self.max_input_tokens:
                raise ValueError(
                    f"relation_schema_repair:{index:04d} prompt exceeds input budget: "
                    f"{measured}>{self.max_input_tokens}"
                )
        else:
            measured = None
        cache_key, hashes = review_cache_key(
            namespace="relation_schema_repair",
            input_payload=input_payload,
            system_prompt=PREDICATE_SCHEMA_REPAIR_SYSTEM,
            user_prompt=user_prompt,
            output_schema=REPAIR_OUTPUT_SCHEMA,
            client=self.llm_client,
        )
        checkpoint = self.cache.load(
            namespace="relation_schema_repair", cache_key=cache_key, hashes=hashes
        )
        if checkpoint:
            payload = checkpoint["result"]
            call_metadata = list(checkpoint.get("call_metadata", []))
            cache_hit = True
        else:
            call = await self.llm_client.generate_json(
                system_prompt=PREDICATE_SCHEMA_REPAIR_SYSTEM,
                user_prompt=user_prompt,
                stage=f"relation_schema_repair:{index:04d}",
            )
            payload = call.data
            call_metadata = [call.metadata]
            cache_hit = False

        decisions, human_queue = _validate_repair_payload(payload, pack)
        status = "accepted" if not human_queue else "human_review_required"
        if not checkpoint:
            self.cache.commit(
                namespace="relation_schema_repair",
                cache_key=cache_key,
                hashes=hashes,
                call_kind="targeted_repair",
                status=status,
                result=payload,
                call_metadata=call_metadata,
                validation_error="; ".join(
                    item["validation_error"] for item in human_queue
                ),
            )
        return index, decisions, human_queue, {
            "stage": f"relation_schema_repair:{index:04d}",
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "item_count": len(pack),
            "accepted_count": len(decisions),
            "human_review_required_count": len(human_queue),
            "prompt_tokens_measured": measured,
            "semantic_call_count": 0 if cache_hit else 1,
            "model_call_metadata": call_metadata,
        }


def apply_relation_human_adjudications(
    *,
    movie_id: str,
    registry: dict[str, Any],
    adjudication_payload: dict[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    if not clean_text(reviewer):
        raise ValueError("Human adjudication requires a non-empty reviewer")
    queued = list(registry.get("human_review_required", []) or [])
    queued_ids = {item["observation_id"] for item in queued}
    raw_rows = adjudication_payload.get("decisions")
    if not isinstance(raw_rows, list):
        raise ValueError("Human adjudication must contain a decisions array")
    supplied_ids = [
        clean_text(item.get("observation_id"))
        for item in raw_rows
        if isinstance(item, dict)
    ]
    if len(supplied_ids) != len(set(supplied_ids)) or set(supplied_ids) != queued_ids:
        raise ValueError(
            "Human adjudication must cover every queued observation exactly once"
        )
    observation_by_id = {
        item["observation_id"]: item for item in registry.get("observations", [])
    }
    pack = [
        {
            "observation": observation_by_id[item["observation_id"]],
            "prior_decision": item.get("prior_decision"),
            "validation_error": item.get("validation_error", "human_review_required"),
        }
        for item in queued
    ]
    decisions, invalid = _validate_repair_payload(adjudication_payload, pack)
    if invalid:
        errors = ", ".join(
            f"{item['observation_id']}:{item['validation_error']}" for item in invalid
        )
        raise ValueError(f"Invalid human adjudication decisions: {errors}")

    output = copy.deepcopy(registry)
    decision_by_id = {
        item["observation_id"]: dict(item.get("normalization", {}))
        for item in output.get("observations", [])
    }
    decision_by_id.update({item["observation_id"]: item for item in decisions})
    normalized = [
        _materialize_observation(
            {key: value for key, value in item.items() if key not in {
                "normalization",
                "canonical_subject_entity_id",
                "canonical_predicate_id",
                "canonical_object_entity_id",
            }},
            decision_by_id[item["observation_id"]],
        )
        for item in output.get("observations", [])
    ]
    canonical_relations = _aggregate_relations(movie_id, normalized)
    contradictions = find_contradiction_groups(canonical_relations)
    all_decisions = [item["normalization"] for item in normalized]
    output.update(
        {
            "observations": normalized,
            "canonical_relations": canonical_relations,
            "human_review_required": [],
            "human_adjudication": {
                "reviewer": clean_text(reviewer),
                "decision_count": len(decisions),
                "decisions": decisions,
                "note": "Human decisions passed the same deterministic predicate validator.",
            },
        }
    )
    audit = output.setdefault("audit", {})
    audit.update(
        {
            "human_review_required_count": 0,
            "human_adjudicated_count": len(decisions),
            "kept_count": sum(item["action"] == "keep" for item in all_decisions),
            "dropped_count": sum(item["action"] == "drop" for item in all_decisions),
            "canonical_predicate_count": len(
                {item["predicate_id"] for item in all_decisions if item["action"] == "keep"}
            ),
            "canonical_relation_count": len(canonical_relations),
            "duplicate_observation_reduction_count": max(
                0,
                sum(item["action"] == "keep" for item in all_decisions)
                - len(canonical_relations),
            ),
            "contradiction_group_count": len(contradictions),
            "contradiction_groups": contradictions,
            "decisions": all_decisions,
        }
    )
    return output


def _prior_decision_error(
    observation: dict[str, Any], prior_observation: dict[str, Any] | None
) -> str:
    if not prior_observation:
        return "missing_prior_observation"
    decision = prior_observation.get("normalization")
    if not isinstance(decision, dict):
        return "missing_prior_normalization"
    action = clean_text(decision.get("action"))
    if action == "drop":
        return ""
    if action != "keep":
        return f"invalid_prior_action:{action or 'empty'}"
    predicate_id = clean_text(decision.get("predicate_id"))
    if predicate_id not in PREDICATE_REGISTRY:
        return f"unregistered_predicate:{predicate_id or 'empty'}"
    reverse = decision.get("reverse_direction")
    if not isinstance(reverse, bool):
        return "reverse_direction_not_boolean"
    if not predicate_allows_observation(
        predicate_id, observation, reverse_direction=reverse
    ):
        return f"predicate_domain_range_mismatch:{predicate_id}"
    if clean_text(decision.get("relation_class")) not in {"stable", "temporal"}:
        return "kept_relation_has_invalid_class"
    return ""


def _repair_prompt(
    pack: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for item in pack:
        observation = item["observation"]
        rows.append(
            {
                "observation_id": observation["observation_id"],
                "subject": observation["subject_name"],
                "subject_primary_kind": observation["subject_primary_kind"],
                "subject_facets": observation.get("subject_facets", []),
                "surface_predicate": observation["surface_predicate"],
                "object": observation["object_name"],
                "object_primary_kind": observation["object_primary_kind"],
                "object_facets": observation.get("object_facets", []),
                "relation_class_observation": observation["relation_class"],
                "qualifiers": observation.get("qualifiers", {}),
                "source_scene_id": observation["source_scene_id"],
                "source_evidence": observation["evidence"],
                "prior_decision": item["prior_decision"],
                "validation_error": item["validation_error"],
            }
        )
    return rows, PREDICATE_SCHEMA_REPAIR_USER.format(
        predicate_registry=json.dumps(PREDICATE_REGISTRY, ensure_ascii=False, indent=2),
        repair_items=json.dumps(rows, ensure_ascii=False, indent=2),
    )


def _validate_repair_payload(
    payload: Any, pack: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {item["observation"]["observation_id"]: item for item in pack}
    raw_rows = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(raw_rows, list):
        raw_rows = []
    supplied: dict[str, list[Any]] = {}
    for raw in raw_rows:
        observation_id = clean_text(raw.get("observation_id")) if isinstance(raw, dict) else ""
        supplied.setdefault(observation_id, []).append(raw)
    decisions: list[dict[str, Any]] = []
    human_queue: list[dict[str, Any]] = []
    for observation_id, item in expected.items():
        rows = supplied.get(observation_id, [])
        error = ""
        decision: dict[str, Any] | None = None
        if len(rows) != 1:
            error = "missing_repair_decision" if not rows else "duplicate_repair_decision"
        else:
            raw = rows[0]
            try:
                decision = _validate_predicate_decisions(
                    {"decisions": [raw]},
                    {observation_id},
                    self_relation_ids={observation_id}
                    if item["observation"]["subject_entity_id"]
                    == item["observation"]["object_entity_id"]
                    else set(),
                    observations_by_id={observation_id: item["observation"]},
                )[0]
                if clean_text(raw.get("action")) == "keep" and decision["action"] != "keep":
                    error = (
                        decision.get("normalization_reason")
                        or "invalid_kept_repair_decision"
                    )
                    decision = None
                elif clean_text(raw.get("action")) == "drop" and clean_text(
                    raw.get("predicate_id")
                ):
                    error = "dropped_repair_has_predicate"
                    decision = None
            except Exception as exc:
                error = clean_text(exc)
                decision = None
        if decision is not None:
            decisions.append(decision)
        else:
            human_queue.append(
                {
                    "observation_id": observation_id,
                    "prior_decision": item["prior_decision"],
                    "prior_validation_error": item["validation_error"],
                    "validation_error": error or "invalid_targeted_repair",
                    "source_evidence": item["observation"]["evidence"],
                    "review_status": "pending_human_review",
                }
            )
    return decisions, human_queue


def _materialize_observation(
    observation: dict[str, Any], decision: dict[str, Any]
) -> dict[str, Any]:
    normalized = {**observation, "normalization": decision}
    if decision["action"] != "keep":
        return normalized
    source = observation["subject_entity_id"]
    target = observation["object_entity_id"]
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
    return normalized
