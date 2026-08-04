from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

from .models import ENTITY_SCOPES, ENTITY_TYPES, clean_text, normalize_name, stable_id
from .io import sha256_json
from .prompt_loader import PROMPTS
from .relation_resolution import (
    find_contradiction_groups,
    predicate_allows_observation,
)


_GLOBAL_KG_REVIEW_PROMPT = PROMPTS.get("global_kg_review")
GLOBAL_KG_REVIEW_SYSTEM = _GLOBAL_KG_REVIEW_PROMPT.system
GLOBAL_KG_REVIEW_USER = _GLOBAL_KG_REVIEW_PROMPT.user


class JsonClient(Protocol):
    async def generate_json(
        self, *, system_prompt: str, user_prompt: str, stage: str
    ) -> Any: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class KGQualityConfig:
    review_batch_size: int = 16
    semantic_attempts: int = 2
    max_concurrency: int = 4


class KGQualityReviewer:
    def __init__(
        self,
        *,
        llm_client: JsonClient,
        config: KGQualityConfig,
        token_counter: TokenCounter | None = None,
        max_input_tokens: int | None = None,
    ):
        self.llm_client = llm_client
        self.config = config
        self.token_counter = token_counter
        self.max_input_tokens = max_input_tokens

    async def review(self, issues: list[dict[str, Any]]) -> dict[str, Any]:
        if not issues:
            return {
                "schema_version": "stage_global_kg_review_v1",
                "decisions": [],
                "llm_calls": [],
            }
        batches = [
            issues[start : start + max(1, self.config.review_batch_size)]
            for start in range(0, len(issues), max(1, self.config.review_batch_size))
        ]
        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def run_batch(index: int, batch: list[dict[str, Any]]):
            async with semaphore:
                return await self._review_batch(index, batch)

        results = await asyncio.gather(
            *(run_batch(index, batch) for index, batch in enumerate(batches))
        )
        results.sort(key=lambda item: item[0])
        return {
            "schema_version": "stage_global_kg_review_v1",
            "decisions": [item for _, rows, _ in results for item in rows],
            "llm_calls": [metadata for _, _, metadata in results],
            "note": (
                "generated_rationale_hint is model-generated audit context, "
                "not screenplay evidence"
            ),
        }

    async def _review_batch(
        self, batch_index: int, issues: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        expected_ids = {item["issue_id"] for item in issues}
        base_prompt = GLOBAL_KG_REVIEW_USER.format(
            issues=json.dumps(issues, ensure_ascii=False, indent=2)
        )
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.config.semantic_attempts) + 1):
            try:
                complete_prompt = base_prompt + _validation_feedback(last_error)
                measured_tokens = self._prompt_tokens(
                    GLOBAL_KG_REVIEW_SYSTEM,
                    complete_prompt,
                    f"global_kg_review:{batch_index:04d}",
                )
                call = await self.llm_client.generate_json(
                    system_prompt=GLOBAL_KG_REVIEW_SYSTEM,
                    user_prompt=complete_prompt,
                    stage=f"global_kg_review:{batch_index:04d}",
                )
                decisions = _validate_global_review(
                    call.data,
                    expected_ids,
                    issue_types={item["issue_id"]: item["issue_type"] for item in issues},
                )
                return batch_index, decisions, {
                    **call.metadata,
                    "semantic_attempt": attempt,
                    "prompt_tokens_measured": measured_tokens,
                }
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Global KG review batch {batch_index} failed semantic validation: {last_error}"
        ) from last_error

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


def reuse_exact_global_review_decisions(
    *,
    prior_issues: list[dict[str, Any]],
    prior_review: dict[str, Any],
    current_issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse a decision only when the complete issue payload is byte-semantically equal."""
    prior_issue_by_id = {item["issue_id"]: item for item in prior_issues}
    prior_decision_by_id = {
        item["issue_id"]: item for item in prior_review.get("decisions", [])
    }
    reused: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    reused_hashes: dict[str, str] = {}
    for issue in current_issues:
        issue_id = issue["issue_id"]
        if (
            prior_issue_by_id.get(issue_id) == issue
            and issue_id in prior_decision_by_id
        ):
            reused.append(dict(prior_decision_by_id[issue_id]))
            reused_hashes[issue_id] = sha256_json(issue)
        else:
            pending.append(issue)
    return (
        {
            "schema_version": "stage_global_kg_review_v2",
            "decisions": reused,
            "llm_calls": [],
            "reuse_audit": {
                "policy": "exact_issue_payload_only",
                "reused_decision_count": len(reused),
                "pending_issue_count": len(pending),
                "reused_issue_sha256": reused_hashes,
                "source_review_schema_version": prior_review.get("schema_version"),
            },
        },
        pending,
    )


def build_quality_issues(
    *,
    entity_registry: dict[str, Any],
    relation_registry: dict[str, Any],
    scene_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    entities = entity_registry.get("entities", [])
    entity_by_id = {item["entity_id"]: item for item in entities}
    source_by_scene = {
        record["scene"]["scene_id"]: _scene_source_text(record)
        for record in (scene_records or [])
    }
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alias_to_entities: dict[str, set[str]] = defaultdict(set)
    related_entity_ids = {
        entity_id
        for relation in relation_registry.get("canonical_relations", [])
        for entity_id in (
            relation.get("subject_entity_id"),
            relation.get("object_entity_id"),
        )
        if entity_id
    }
    for entity in entities:
        by_name[normalize_name(entity.get("canonical_name"))].append(entity)
        for value in [entity.get("canonical_name"), *(entity.get("aliases") or [])]:
            normalized = normalize_name(value)
            if normalized:
                alias_to_entities[normalized].add(entity["entity_id"])

        name = clean_text(entity.get("canonical_name"))
        if entity.get("entity_type") not in ENTITY_TYPES:
            issues.append(
                _issue(
                    "illegal_entity_type",
                    entity["entity_id"],
                    {
                        "canonical_name": name,
                        "entity_type": entity.get("entity_type"),
                        "allowed_entity_types": list(ENTITY_TYPES),
                    },
                )
            )
        if entity.get("scope") not in ENTITY_SCOPES:
            issues.append(
                _issue(
                    "illegal_entity_scope",
                    entity["entity_id"],
                    {
                        "canonical_name": name,
                        "scope": entity.get("scope"),
                        "allowed_entity_scopes": list(ENTITY_SCOPES),
                    },
                )
            )
        observed_types = entity.get("entity_types")
        valid_type_sets = (
            isinstance(observed_types, list)
            and bool(observed_types)
            and all(value in ENTITY_TYPES for value in observed_types)
            and entity.get("entity_type") in observed_types
            and (
                len(set(observed_types)) == 1
                or set(observed_types) == {"Location", "Organization"}
            )
        )
        if not valid_type_sets:
            issues.append(
                _issue(
                    "inconsistent_entity_types",
                    entity["entity_id"],
                    {
                        "canonical_name": name,
                        "entity_type": entity.get("entity_type"),
                        "entity_types": observed_types,
                    },
                )
            )
        if _low_quality_name(
            name,
            clean_text(entity.get("primary_kind")),
            entity.get("aliases") or [],
        ):
            issues.append(
                _issue(
                    "low_quality_entity_name",
                    entity["entity_id"],
                    {
                        "canonical_name": name,
                        "primary_kind": entity.get("primary_kind"),
                        "aliases": entity.get("aliases", []),
                        "descriptions": entity.get("descriptions", [])[:8],
                        "source_evidence": entity.get("source_evidence", [])[:8],
                    },
                )
            )
        if entity.get("type_status") in {"ambiguous", "conflict"}:
            issues.append(
                _issue(
                    "unresolved_entity_type",
                    entity["entity_id"],
                    {
                        "canonical_name": name,
                        "primary_kind": entity.get("primary_kind"),
                        "raw_types": entity.get("raw_types", []),
                        "type_votes": entity.get("type_votes", {}),
                        "descriptions": entity.get("descriptions", [])[:8],
                    },
                )
            )
        if not entity.get("source_evidence"):
            issues.append(
                _issue(
                    "entity_without_source_evidence",
                    entity["entity_id"],
                    {"canonical_name": name},
                )
            )
        elif source_by_scene and any(
            not _evidence_in_any_scene(
                evidence, entity.get("source_scene_ids", []), source_by_scene
            )
            for evidence in entity.get("source_evidence", [])
        ):
            issues.append(
                _issue(
                    "entity_evidence_not_source",
                    entity["entity_id"],
                    {
                        "canonical_name": name,
                        "source_scene_ids": entity.get("source_scene_ids", []),
                        "source_evidence": entity.get("source_evidence", []),
                    },
                )
            )
        if (
            len(entity.get("source_scene_ids", [])) <= 1
            and not entity.get("descriptions")
            and entity.get("entity_id") not in related_entity_ids
        ):
            issues.append(
                _issue(
                    "orphan_sparse_entity",
                    entity["entity_id"],
                    {
                        "canonical_name": name,
                        "primary_kind": entity.get("primary_kind"),
                        "source_scene_ids": entity.get("source_scene_ids", []),
                    },
                )
            )

    for normalized, rows in sorted(by_name.items()):
        if normalized and len(rows) > 1:
            issues.append(
                _issue(
                    "duplicate_canonical_name",
                    normalized,
                    {
                        "clusters": [
                            {
                                "entity_id": row["entity_id"],
                                "canonical_name": row["canonical_name"],
                                "primary_kind": row.get("primary_kind"),
                                "aliases": row.get("aliases", []),
                                "descriptions": row.get("descriptions", [])[:12],
                                "source_scene_ids": row.get("source_scene_ids", []),
                                "source_evidence": row.get("source_evidence", [])[:12],
                            }
                            for row in rows
                        ]
                    },
                )
            )
    for normalized, entity_ids in sorted(alias_to_entities.items()):
        if len(entity_ids) > 1:
            summaries = [
                _entity_issue_summary(entity_by_id[entity_id])
                for entity_id in sorted(entity_ids)
            ]
            issues.append(
                _issue(
                    "alias_collision",
                    normalized,
                    {
                        "normalized_alias": normalized,
                        "entity_ids": sorted(entity_ids),
                        "clusters": summaries,
                    },
                )
            )

    assertions = relation_registry.get("canonical_relations", [])
    for relation in assertions:
        if not relation.get("evidence_items"):
            issues.append(
                _issue(
                    "relation_without_source_evidence",
                    relation["relation_id"],
                    {"relation_id": relation["relation_id"]},
                )
            )
        invalid_evidence = [
            item
            for item in relation.get("evidence_items", [])
            if source_by_scene
            and not _evidence_in_source(
                item.get("evidence", ""),
                source_by_scene.get(item.get("source_scene_id"), ""),
            )
        ]
        if invalid_evidence:
            issues.append(
                _issue(
                    "relation_evidence_not_source",
                    relation["relation_id"],
                    {"invalid_evidence_items": invalid_evidence},
                )
            )
        if relation.get("relation_class") == "event_like":
            issues.append(
                _issue(
                    "event_like_canonical_relation",
                    relation["relation_id"],
                    {"relation_id": relation["relation_id"]},
                )
            )
        if relation.get("predicate_id") == "related_to":
            issues.append(
                _issue(
                    "generic_related_to_relation",
                    relation["relation_id"],
                    {"relation_id": relation["relation_id"]},
                )
            )
        if not _canonical_relation_schema_valid(relation, entity_by_id):
            issues.append(
                _issue(
                    "relation_domain_range_mismatch",
                    relation["relation_id"],
                    {
                        "relation_id": relation["relation_id"],
                        "predicate_id": relation.get("predicate_id"),
                        "subject": _entity_issue_summary(
                            entity_by_id.get(relation.get("subject_entity_id"), {})
                        ),
                        "object": _entity_issue_summary(
                            entity_by_id.get(relation.get("object_entity_id"), {})
                        ),
                    },
                )
            )
        qualifiers = relation.get("qualifiers")
        if not isinstance(qualifiers, dict) or qualifiers.get("polarity") not in {
            "positive",
            "negative",
        } or not qualifiers.get("modality"):
            issues.append(
                _issue(
                    "relation_qualifier_missing",
                    relation["relation_id"],
                    {"relation_id": relation["relation_id"], "qualifiers": qualifiers},
                )
            )
        elif relation.get("relation_class") == "temporal" and not qualifiers.get(
            "valid_at_scene_id"
        ):
            issues.append(
                _issue(
                    "temporal_relation_without_validity",
                    relation["relation_id"],
                    {"relation_id": relation["relation_id"], "qualifiers": qualifiers},
                )
            )
    relation_keys: Counter[tuple[str, str, str, str, str]] = Counter(
        (
            item["subject_entity_id"],
            item["predicate_id"],
            item["object_entity_id"],
            item["relation_class"],
            json.dumps(item.get("qualifiers", {}), ensure_ascii=False, sort_keys=True),
        )
        for item in assertions
    )
    for key, count in relation_keys.items():
        if count > 1:
            issues.append(
                _issue("duplicate_canonical_relation", "|".join(key), {"count": count})
            )
    for contradiction in find_contradiction_groups(assertions):
        issues.append(
            _issue(
                "contradictory_canonical_relations",
                "|".join(contradiction["relation_ids"]),
                contradiction,
            )
        )
    return sorted(issues, key=lambda item: item["issue_id"])


def build_kg_quality_report(
    *,
    raw_scene_records: list[dict[str, Any]],
    reviewed_scene_records: list[dict[str, Any]],
    canonical_scene_records: list[dict[str, Any]],
    entity_registry: dict[str, Any],
    relation_registry: dict[str, Any],
    issues: list[dict[str, Any]],
    global_review: dict[str, Any],
    named_character_review_approved: bool = False,
) -> dict[str, Any]:
    decision_by_id = {
        item["issue_id"]: item for item in global_review.get("decisions", [])
    }
    issue_counts = Counter(item["issue_type"] for item in issues)
    hard_issue_types = {
        "low_quality_entity_name",
        "unresolved_entity_type",
        "illegal_entity_type",
        "illegal_entity_scope",
        "inconsistent_entity_types",
        "entity_without_source_evidence",
        "entity_evidence_not_source",
        "orphan_sparse_entity",
        "relation_without_source_evidence",
        "relation_evidence_not_source",
        "relation_domain_range_mismatch",
        "event_like_canonical_relation",
        "generic_related_to_relation",
        "relation_qualifier_missing",
        "temporal_relation_without_validity",
        "contradictory_canonical_relations",
        "duplicate_canonical_relation",
    }
    entity_hard_issue_types = {
        "low_quality_entity_name",
        "unresolved_entity_type",
        "illegal_entity_type",
        "illegal_entity_scope",
        "inconsistent_entity_types",
        "entity_without_source_evidence",
        "entity_evidence_not_source",
        "orphan_sparse_entity",
    }
    unresolved_hard = [
        item
        for item in issues
        if item["issue_type"] in hard_issue_types
    ]
    unresolved_identity = [
        item
        for item in issues
        if item["issue_type"] in {"duplicate_canonical_name", "alias_collision"}
        and decision_by_id.get(item["issue_id"], {}).get("disposition")
        != "accepted_distinct"
    ]
    unresolved_entity_hard = [
        item for item in issues if item["issue_type"] in entity_hard_issue_types
    ]
    human_review = [
        item
        for item in global_review.get("decisions", [])
        if item["disposition"] == "human_review_required"
    ]
    relation_human_review = list(
        relation_registry.get("human_review_required", []) or []
    )
    unresolved_refs = _unresolved_reference_ids(canonical_scene_records)
    unresolved_relation_endpoints = [
        relation.get("relation_id")
        for record in canonical_scene_records
        for relation in record.get("entity_relations", [])
        if not relation.get("subject_entity_id") or not relation.get("object_entity_id")
    ]
    relation_registry_ids = set(relation_registry.get("predicate_registry", {}))
    invalid_predicates = [
        item["relation_id"]
        for item in relation_registry.get("canonical_relations", [])
        if item.get("predicate_id") not in relation_registry_ids
    ]
    entity_ids = [item["entity_id"] for item in entity_registry.get("entities", [])]
    relation_ids = [
        item["relation_id"] for item in relation_registry.get("canonical_relations", [])
    ]
    known_entity_ids = set(entity_ids)
    dangling_canonical_relations = [
        item["relation_id"]
        for item in relation_registry.get("canonical_relations", [])
        if item.get("subject_entity_id") not in known_entity_ids
        or item.get("object_entity_id") not in known_entity_ids
    ]

    review_audits = [
        record.get("scene_graph_review_audit", {}) for record in reviewed_scene_records
    ]
    entity_reason_counts: Counter[str] = Counter()
    relation_reason_counts: Counter[str] = Counter()
    for audit in review_audits:
        entity_reason_counts.update(audit.get("entity_reason_counts", {}))
        relation_reason_counts.update(audit.get("relation_reason_counts", {}))

    raw_entity_count = sum(len(record.get("entities", [])) for record in raw_scene_records)
    raw_relation_count = sum(
        len(record.get("entity_relations", [])) for record in raw_scene_records
    )
    reviewed_entity_count = sum(
        len(record.get("entities", [])) for record in reviewed_scene_records
    )
    reviewed_relation_count = sum(
        len(record.get("entity_relations", [])) for record in reviewed_scene_records
    )
    entity_review_decision_count = sum(
        int(audit.get(field, 0))
        for audit in review_audits
        for field in (
            "entity_keep_count",
            "entity_repair_count",
            "entity_split_count",
            "entity_drop_count",
        )
    )
    relation_review_decision_count = sum(
        int(audit.get(field, 0))
        for audit in review_audits
        for field in (
            "relation_keep_count",
            "relation_repair_count",
            "relation_drop_count",
        )
    )
    relation_audit = relation_registry.get("audit", {})
    observations = relation_registry.get("observations", [])
    assertions = relation_registry.get("canonical_relations", [])
    normalized_decision_count = (
        int(relation_audit.get("kept_count", 0))
        + int(relation_audit.get("dropped_count", 0))
        + int(relation_audit.get("human_review_required_count", 0))
    )
    kept_observation_ids = {
        item.get("observation_id")
        for item in observations
        if item.get("normalization", {}).get("action") == "keep"
    }
    asserted_observation_ids = {
        value
        for relation in assertions
        for value in relation.get("source_observation_ids", [])
    }
    qualifier_problems = _relation_qualifier_problems(assertions, observations)
    reviewed_issue_ids = set(decision_by_id)
    expected_issue_ids = {item["issue_id"] for item in issues}
    entity_resolution_audit = entity_registry.get("audit", {})
    entity_merge_decisions = sum(
        item.get("decision") == "same_identity"
        for item in [
            *(entity_resolution_audit.get("decisions") or []),
            *(entity_resolution_audit.get("cluster_decisions") or []),
        ]
    )
    entity_count = len(entity_registry.get("entities", []))
    relation_count = len(assertions)
    raw_predicate_count = int(relation_audit.get("input_surface_predicate_count", 0))
    canonical_predicate_count = int(
        relation_audit.get("canonical_predicate_count", 0)
    )
    kept_relation_count = int(relation_audit.get("kept_count", 0))
    entity_evidence_count = sum(
        bool(item.get("source_evidence")) for item in entity_registry.get("entities", [])
    )
    relation_evidence_count = sum(bool(item.get("evidence_items")) for item in assertions)

    silver_gates = {
        "scene_review_coverage": all(
            record.get("scene_graph_reviewed") for record in reviewed_scene_records
        ),
        "raw_entity_review_provenance_complete": (
            entity_review_decision_count == raw_entity_count
        ),
        "raw_relation_review_provenance_complete": (
            relation_review_decision_count == raw_relation_count
        ),
        "relation_normalization_coverage": (
            normalized_decision_count == len(observations)
            and kept_observation_ids == asserted_observation_ids
        ),
        "global_issue_review_coverage": reviewed_issue_ids == expected_issue_ids,
        "no_unresolved_references": not unresolved_refs,
        "no_unresolved_relation_endpoints": not unresolved_relation_endpoints,
        "no_low_quality_or_untyped_entities": not unresolved_entity_hard,
        "no_unresolved_hard_quality_issues": not unresolved_hard,
        "all_canonical_entities_grounded": entity_evidence_count == entity_count,
        "explicit_relation_human_queue_complete": int(
            relation_audit.get("human_review_required_count", 0)
        )
        == len(relation_human_review),
        "canonical_predicates_registered": not invalid_predicates,
        "predicate_domain_range_valid": not issue_counts[
            "relation_domain_range_mismatch"
        ],
        "no_event_like_canonical_relations": not issue_counts[
            "event_like_canonical_relation"
        ],
        "no_generic_related_to_relations": not issue_counts[
            "generic_related_to_relation"
        ],
        "relation_qualifiers_preserved": not qualifier_problems,
        "all_canonical_relations_grounded": relation_evidence_count == relation_count,
        "no_unresolved_contradictions": not issue_counts[
            "contradictory_canonical_relations"
        ],
        "unique_entity_and_relation_ids": (
            len(entity_ids) == len(set(entity_ids))
            and len(relation_ids) == len(set(relation_ids))
        ),
        "no_dangling_canonical_relations": not dangling_canonical_relations,
        "no_duplicate_canonical_relations": not issue_counts[
            "duplicate_canonical_relation"
        ],
    }
    benchmark_release_gates = {
        **silver_gates,
        "duplicate_identity_issues_adjudicated": not unresolved_identity,
        "no_human_review_required": not human_review and not relation_human_review,
        "named_character_review_approved": named_character_review_approved,
    }
    silver_status = "passed" if all(silver_gates.values()) else "failed"
    benchmark_release_status = (
        "passed" if all(benchmark_release_gates.values()) else "failed"
    )
    return {
        "schema_version": "stage_kg_quality_report_v3",
        "status": silver_status,
        "silver_status": silver_status,
        "benchmark_release_status": benchmark_release_status,
        "gates": silver_gates,
        "silver_gates": silver_gates,
        "benchmark_release_gates": benchmark_release_gates,
        "counts": {
            "scenes": len(raw_scene_records),
            "raw_entity_mentions": raw_entity_count,
            "reviewed_entity_mentions": reviewed_entity_count,
            "canonical_entities": entity_count,
            "raw_relations": raw_relation_count,
            "reviewed_relations": reviewed_relation_count,
            "relation_observations": len(relation_registry.get("observations", [])),
            "kept_relation_observations": int(relation_audit.get("kept_count", 0)),
            "dropped_relation_observations": int(relation_audit.get("dropped_count", 0)),
            "canonical_relations": relation_count,
            "raw_surface_predicates": int(
                relation_audit.get("input_surface_predicate_count", 0)
            ),
            "canonical_predicates_used": int(
                relation_audit.get("canonical_predicate_count", 0)
            ),
            "entity_merge_decisions": entity_merge_decisions,
            "relation_duplicate_observations_merged": int(
                relation_audit.get("duplicate_observation_reduction_count", 0)
            ),
            "duplicate_canonical_relations": int(
                issue_counts.get("duplicate_canonical_relation", 0)
            ),
            "contradiction_groups": int(
                issue_counts.get("contradictory_canonical_relations", 0)
            ),
            "predicate_domain_range_drops": int(
                relation_audit.get("predicate_domain_range_drop_count", 0)
            ),
            "orphan_sparse_entities": int(
                issue_counts.get("orphan_sparse_entity", 0)
            ),
            "quality_issues": len(issues),
            "human_review_required": len(human_review) + len(relation_human_review),
            "global_human_review_required": len(human_review),
            "relation_human_review_required": len(relation_human_review),
            "relation_decisions_preserved": int(
                relation_audit.get("preserved_decision_count", 0)
            ),
            "relation_schema_repair_candidates": int(
                relation_audit.get("affected_observation_count", 0)
            ),
            "relation_schema_repairs_accepted": int(
                relation_audit.get("repaired_decision_count", 0)
            ),
            "scene_entity_keep": sum(
                int(item.get("entity_keep_count", 0)) for item in review_audits
            ),
            "scene_entity_repair": sum(
                int(item.get("entity_repair_count", 0)) for item in review_audits
            ),
            "scene_entity_split": sum(
                int(item.get("entity_split_count", 0)) for item in review_audits
            ),
            "scene_entity_drop": sum(
                int(item.get("entity_drop_count", 0)) for item in review_audits
            ),
            "scene_relation_keep": sum(
                int(item.get("relation_keep_count", 0)) for item in review_audits
            ),
            "scene_relation_repair": sum(
                int(item.get("relation_repair_count", 0)) for item in review_audits
            ),
            "scene_relation_drop": sum(
                int(item.get("relation_drop_count", 0)) for item in review_audits
            ),
            "entity_cluster_reduction": max(
                0, reviewed_entity_count - len(entity_registry.get("entities", []))
            ),
            "entity_retype_decisions": int(entity_reason_counts.get("wrong_type", 0)),
        },
        "rates": {
            "predicate_compression_rate": _reduction_rate(
                raw_predicate_count, canonical_predicate_count
            ),
            "relation_observation_merge_rate": _reduction_rate(
                kept_relation_count, relation_count
            ),
            "entity_cluster_reduction_rate": _reduction_rate(
                reviewed_entity_count, entity_count
            ),
            "canonical_entity_evidence_coverage": _coverage_rate(
                entity_evidence_count, entity_count
            ),
            "canonical_relation_evidence_coverage": _coverage_rate(
                relation_evidence_count, relation_count
            ),
        },
        "entity_review_reason_counts": dict(sorted(entity_reason_counts.items())),
        "relation_review_reason_counts": dict(sorted(relation_reason_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues,
        "unresolved_reference_ids": unresolved_refs,
        "unresolved_relation_endpoint_ids": unresolved_relation_endpoints,
        "invalid_predicate_relation_ids": invalid_predicates,
        "dangling_canonical_relation_ids": dangling_canonical_relations,
        "relation_qualifier_problems": qualifier_problems,
        "relation_human_review_required": relation_human_review,
        "manual_release_requirements": {
            "named_character_alias_review": (
                "approved" if named_character_review_approved else "pending"
            ),
            "required_for_benchmark_release": True,
            "blocks_silver_kg": False,
            "note": (
                "The named-character and alias queue was approved by the designated "
                "reviewer."
                if named_character_review_approved
                else "The machine-reviewed KG is a silver asset. A designated reviewer "
                "must approve the named-character and alias queue before benchmark release."
            ),
        },
        "global_review": global_review,
    }


def require_quality(report: dict[str, Any], tier: str = "silver") -> None:
    if tier not in {"silver", "benchmark_release"}:
        raise ValueError(f"Unknown KG quality tier: {tier}")
    status_key = "silver_status" if tier == "silver" else "benchmark_release_status"
    gates_key = "silver_gates" if tier == "silver" else "benchmark_release_gates"
    if report.get(status_key, report.get("status")) != "passed":
        failed = [
            name for name, passed in report.get(gates_key, report.get("gates", {})).items()
            if not passed
        ]
        raise ValueError(f"KG {tier} quality gates failed: " + ", ".join(failed))


def build_named_character_review_queue(
    entity_registry: dict[str, Any],
) -> dict[str, Any]:
    items = [
        {
            "entity_id": entity["entity_id"],
            "canonical_name": entity.get("canonical_name"),
            "aliases": entity.get("aliases", []),
            "primary_kind": entity.get("primary_kind"),
            "facets": entity.get("facets", []),
            "source_scene_ids": entity.get("source_scene_ids", []),
            "descriptions": entity.get("descriptions", []),
            "source_evidence": entity.get("source_evidence", []),
            "review_status": "pending_human_approval",
        }
        for entity in entity_registry.get("entities", [])
        if entity.get("entity_type") == "Character"
    ]
    items.sort(key=lambda item: (normalize_name(item["canonical_name"]), item["entity_id"]))
    return {
        "schema_version": "stage_named_character_review_queue_v1",
        "status": "pending_human_approval" if items else "not_applicable",
        "count": len(items),
        "items": items,
        "note": (
            "This queue is not sent to the LLM. Human approval is required before "
            "benchmark release, but it does not block the silver KG quality gate."
        ),
    }


def build_semantic_kg(
    *,
    movie_id: str,
    scene_records: list[dict[str, Any]],
    entity_registry: dict[str, Any],
    relation_registry: dict[str, Any],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    scene_node_ids: dict[str, str] = {}
    for record in sorted(scene_records, key=lambda item: int(item["scene"]["order"])):
        scene = record["scene"]
        scene_node_id = stable_id("scene", movie_id, scene["scene_id"])
        scene_node_ids[scene["scene_id"]] = scene_node_id
        nodes.append({"id": scene_node_id, "node_type": "scene", **scene})
    for entity in entity_registry.get("entities", []):
        nodes.append({"id": entity["entity_id"], "node_type": "entity", **entity})
        for scene_id in entity.get("source_scene_ids", []):
            if scene_id in scene_node_ids:
                edges.append(
                    {
                        "id": stable_id(
                            "edge", movie_id, entity["entity_id"], "appears_in_scene", scene_id
                        ),
                        "source": entity["entity_id"],
                        "relation_type": "appears_in_scene",
                        "target": scene_node_ids[scene_id],
                        "source_scene_ids": [scene_id],
                        "edge_layer": "grounding",
                    }
                )
    for relation in relation_registry.get("canonical_relations", []):
        edges.append(
            {
                "id": relation["relation_id"],
                "source": relation["subject_entity_id"],
                "relation_type": relation["predicate_id"],
                "target": relation["object_entity_id"],
                "relation_class": relation["relation_class"],
                "status": relation["status"],
                "qualifiers": relation.get("qualifiers", {}),
                "surface_predicates": relation["surface_predicates"],
                "source_observation_ids": relation["source_observation_ids"],
                "source_scene_ids": relation["source_scene_ids"],
                "source_scene_orders": relation.get("source_scene_orders", []),
                "evidence_items": relation["evidence_items"],
                "edge_layer": "semantic",
            }
        )
    node_counts = Counter(node["node_type"] for node in nodes)
    edge_counts = Counter(edge["relation_type"] for edge in edges)
    return {
        "schema_version": "stage_semantic_kg_v2",
        "movie_id": movie_id,
        "nodes": nodes,
        "edges": edges,
        "counts": {
            "nodes_total": len(nodes),
            "edges_total": len(edges),
            "nodes_by_type": dict(sorted(node_counts.items())),
            "edges_by_type": dict(sorted(edge_counts.items())),
        },
    }


def _unresolved_reference_ids(scene_records: list[dict[str, Any]]) -> list[str]:
    unresolved: list[str] = []
    for record in scene_records:
        for unit in record.get("narrative_units", []):
            if unit.get("kind") == "occasion" and not unit.get("entity_id"):
                unresolved.append(f"{unit.get('unit_id')}:occasion_entity")
            if len(unit.get("participants", [])) != len(unit.get("participant_entities", [])):
                unresolved.append(clean_text(unit.get("unit_id")))
            if unit.get("participant_entity_ids") != [
                item.get("entity_id") for item in unit.get("participant_entities", [])
            ]:
                unresolved.append(f"{unit.get('unit_id')}:participant_entity_ids")
            for field, entity_field in (
                ("locations", "location_entities"),
                ("times", "time_entities"),
            ):
                if len(unit.get(field, [])) != len(unit.get(entity_field, [])):
                    unresolved.append(f"{unit.get('unit_id')}:{field}")
            if unit.get("related_occasion") and not unit.get(
                "related_occasion_entity_id"
            ):
                unresolved.append(f"{unit.get('unit_id')}:related_occasion")
            if unit.get("related_event") and not unit.get("related_event_unit_id"):
                unresolved.append(f"{unit.get('unit_id')}:related_event_unit")
            if unit.get("related_occasion") and not unit.get(
                "related_occasion_unit_id"
            ):
                unresolved.append(f"{unit.get('unit_id')}:related_occasion_unit")
            for field in ("subject", "object"):
                if unit.get(field) and not unit.get(f"{field}_entity_id"):
                    unresolved.append(f"{unit.get('unit_id')}:{field}")
    return sorted(set(value for value in unresolved if value))


def _low_quality_name(
    name: str,
    primary_kind: str,
    aliases: list[str] | tuple[str, ...] = (),
) -> bool:
    if not name or len(name) > 80:
        return True
    if primary_kind == "Character":
        if re.search(r"[、;/]", name) or re.search(r"(?:和|与).+", name):
            return True
        if "," in name or "，" in name:
            # Appositive historical names such as ``Mary, Queen of Scots``
            # are valid when the leading name is explicitly retained as an
            # alias and the suffix is a title/descriptor, not a name list.
            parts = [part.strip() for part in re.split(r"[,，]", name)]
            alias_names = {normalize_name(value) for value in aliases if value}
            suffix = " ".join(parts[1:]) if len(parts) > 1 else ""
            appositive = (
                len(parts) == 2
                and normalize_name(parts[0]) in alias_names
                and bool(re.search(r"\bof\b|\bthe\b", suffix, re.IGNORECASE))
            )
            if not appositive:
                return True
    return False


def _scene_source_text(record: dict[str, Any]) -> str:
    scene = record.get("scene", {})
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


def _evidence_in_any_scene(
    evidence: Any, scene_ids: list[str], source_by_scene: dict[str, str]
) -> bool:
    return any(
        _evidence_in_source(evidence, source_by_scene.get(scene_id, ""))
        for scene_id in scene_ids
    )


def _evidence_in_source(evidence: Any, source: str) -> bool:
    value = clean_text(evidence)
    return bool(value) and (
        value in source
        or value in clean_text(source)
        or re.sub(r"\s+", "", value) in re.sub(r"\s+", "", source)
    )


def _canonical_relation_schema_valid(
    relation: dict[str, Any], entity_by_id: dict[str, dict[str, Any]]
) -> bool:
    subject = entity_by_id.get(clean_text(relation.get("subject_entity_id")))
    object_entity = entity_by_id.get(clean_text(relation.get("object_entity_id")))
    if not subject or not object_entity:
        return False
    return predicate_allows_observation(
        clean_text(relation.get("predicate_id")),
        {
            "subject_primary_kind": subject.get("primary_kind"),
            "subject_entity_types": subject.get("entity_types", []),
            "subject_facets": subject.get("facets", []),
            "object_primary_kind": object_entity.get("primary_kind"),
            "object_entity_types": object_entity.get("entity_types", []),
            "object_facets": object_entity.get("facets", []),
        },
    )


def _relation_qualifier_problems(
    canonical_relations: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observation_by_id = {
        item.get("observation_id"): item
        for item in observations
        if item.get("observation_id")
    }
    occurrence_counts: Counter[str] = Counter(
        observation_id
        for relation in canonical_relations
        for observation_id in relation.get("source_observation_ids", [])
    )
    problems: list[dict[str, Any]] = []
    for relation in canonical_relations:
        relation_id = clean_text(relation.get("relation_id"))
        qualifiers = relation.get("qualifiers")
        if not isinstance(qualifiers, dict):
            problems.append({"relation_id": relation_id, "reason": "missing_qualifiers"})
            continue
        if qualifiers.get("polarity") not in {"positive", "negative"}:
            problems.append({"relation_id": relation_id, "reason": "invalid_polarity"})
        if not qualifiers.get("modality"):
            problems.append({"relation_id": relation_id, "reason": "missing_modality"})
        if relation.get("relation_class") == "temporal" and not qualifiers.get(
            "valid_at_scene_id"
        ):
            problems.append(
                {"relation_id": relation_id, "reason": "missing_temporal_validity"}
            )
        for observation_id in relation.get("source_observation_ids", []):
            observation = observation_by_id.get(observation_id)
            if observation is None:
                problems.append(
                    {
                        "relation_id": relation_id,
                        "observation_id": observation_id,
                        "reason": "missing_source_observation",
                    }
                )
                continue
            if observation.get("qualifiers", {}) != qualifiers:
                problems.append(
                    {
                        "relation_id": relation_id,
                        "observation_id": observation_id,
                        "reason": "qualifier_mismatch",
                    }
                )
            if occurrence_counts[observation_id] != 1:
                problems.append(
                    {
                        "relation_id": relation_id,
                        "observation_id": observation_id,
                        "reason": "observation_not_uniquely_aggregated",
                    }
                )
    return problems


def _reduction_rate(input_count: int, output_count: int) -> float:
    if input_count <= 0:
        return 0.0
    return round(max(0.0, 1.0 - output_count / input_count), 6)


def _coverage_rate(covered_count: int, total_count: int) -> float:
    if total_count <= 0:
        return 1.0
    return round(covered_count / total_count, 6)


def _issue(issue_type: str, key: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_id": stable_id("kg-issue", issue_type, key),
        "issue_type": issue_type,
        "details": details,
    }


def _entity_issue_summary(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": entity.get("entity_id"),
        "canonical_name": entity.get("canonical_name"),
        "aliases": entity.get("aliases", []),
        "primary_kind": entity.get("primary_kind"),
        "source_scene_ids": entity.get("source_scene_ids", []),
        "descriptions": entity.get("descriptions", [])[:12],
        "source_evidence": entity.get("source_evidence", [])[:12],
    }


def _validate_global_review(
    payload: dict[str, Any],
    expected_ids: set[str],
    issue_types: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
        raise ValueError("Global KG review must return exactly a decisions array")
    by_id: dict[str, dict[str, Any]] = {}
    allowed = {"resolved", "accepted_distinct", "human_review_required"}
    for raw in payload["decisions"]:
        if not isinstance(raw, dict):
            raise ValueError("Global KG review decision must be an object")
        issue_id = clean_text(raw.get("issue_id"))
        disposition = clean_text(raw.get("disposition"))
        if issue_id not in expected_ids or issue_id in by_id:
            raise ValueError(f"Unknown or duplicate KG issue id: {issue_id}")
        if disposition not in allowed:
            raise ValueError(f"Unsupported KG issue disposition: {disposition}")
        normalization_reason = ""
        if (
            (issue_types or {}).get(issue_id)
            in {"duplicate_canonical_name", "alias_collision"}
            and disposition == "resolved"
        ):
            disposition = "human_review_required"
            normalization_reason = "duplicate_issue_cannot_be_resolved_without_merge"
        by_id[issue_id] = {
            "issue_id": issue_id,
            "disposition": disposition,
            "generated_rationale_hint": clean_text(raw.get("generated_rationale_hint")),
            "normalization_reason": normalization_reason,
        }
    if set(by_id) != expected_ids:
        raise ValueError(f"Missing KG issue decisions: {sorted(expected_ids - set(by_id))}")
    return [by_id[item] for item in sorted(by_id)]


def _validation_feedback(error: Exception | None) -> str:
    if error is None:
        return ""
    return (
        "\n\nThe previous output failed strict validation with this error: "
        f"{clean_text(error)}. Return one corrected decision for every issue id."
    )
