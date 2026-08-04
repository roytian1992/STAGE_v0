from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .io import load_json, load_scenes, sha256_file, sha256_json


ALLOWED_ISSUE_TYPES = {"semantic_duplicate", "low_quality"}
MAX_REPLACEMENTS_PER_MOVIE = 5
TASK2_FIELDS = [
    "id",
    "movie_id",
    "language",
    "related_scenes",
    "question",
    "answer",
    "evidence_or_reason",
    "question_type",
]


def validate_task2_review_response(
    payload: dict[str, Any],
    *,
    qa_by_id: dict[str, dict[str, str]],
    evidence_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
        raise ValueError("Task 2 review response must contain only a decisions list")
    decisions = payload["decisions"]
    if len(decisions) > MAX_REPLACEMENTS_PER_MOVIE:
        raise ValueError("Task 2 review exceeds the five-replacement limit")
    seen_sources: set[str] = set()
    existing_questions = {
        _normalize_text(row.get("question", "")) for row in qa_by_id.values()
    }
    validated = []
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Task 2 review decision must be an object")
        source_id = decision.get("source_qa_id")
        if source_id not in qa_by_id or source_id in seen_sources:
            raise ValueError(f"Invalid or duplicate Task 2 source QA: {source_id}")
        issue_type = decision.get("issue_type")
        if issue_type not in ALLOWED_ISSUE_TYPES:
            raise ValueError(f"Invalid Task 2 issue type: {issue_type}")
        duplicate_with = decision.get("duplicate_with")
        if (
            not isinstance(duplicate_with, list)
            or len(duplicate_with) != len(set(duplicate_with))
            or any(item not in qa_by_id or item == source_id for item in duplicate_with)
            or (issue_type == "semantic_duplicate" and not duplicate_with)
        ):
            raise ValueError(f"Invalid duplicate references for {source_id}")
        reason = decision.get("reason")
        replacement = decision.get("replacement")
        if not isinstance(reason, str) or not reason.strip() or not isinstance(replacement, dict):
            raise ValueError(f"Incomplete Task 2 decision: {source_id}")
        expected_replacement_fields = {
            "question",
            "answer",
            "evidence_or_reason",
            "question_type",
            "related_scene_orders",
            "source_evidence_ids",
        }
        if set(replacement) != expected_replacement_fields:
            raise ValueError(f"Task 2 replacement fields mismatch: {source_id}")
        for field in ("question", "answer", "evidence_or_reason", "question_type"):
            if not isinstance(replacement[field], str) or not replacement[field].strip():
                raise ValueError(f"Empty Task 2 replacement field: {source_id}/{field}")
        if replacement["question_type"] != qa_by_id[source_id]["question_type"]:
            raise ValueError(f"Task 2 replacement changes question type: {source_id}")
        normalized_question = _normalize_text(replacement["question"])
        if not normalized_question or normalized_question in existing_questions:
            raise ValueError(f"Task 2 replacement duplicates an existing question: {source_id}")
        evidence_ids = replacement["source_evidence_ids"]
        scene_orders = replacement["related_scene_orders"]
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) != len(set(evidence_ids))
            or any(item not in evidence_by_id for item in evidence_ids)
        ):
            raise ValueError(f"Invalid Task 2 evidence references: {source_id}")
        allowed_scenes = {
            int(scene)
            for evidence_id in evidence_ids
            for scene in evidence_by_id[evidence_id]["scene_orders"]
        }
        if (
            not isinstance(scene_orders, list)
            or not scene_orders
            or len(scene_orders) != len(set(scene_orders))
            or any(
                not isinstance(scene, int)
                or isinstance(scene, bool)
                or scene not in allowed_scenes
                for scene in scene_orders
            )
        ):
            raise ValueError(f"Invalid Task 2 replacement scene orders: {source_id}")
        seen_sources.add(source_id)
        validated.append(decision)
    return validated


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)


def read_task2_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != TASK2_FIELDS:
        raise ValueError(
            f"Task 2 CSV fields differ from the frozen schema: {reader.fieldnames}"
        )
    if not rows:
        raise ValueError(f"Task 2 CSV is empty: {path}")
    return list(reader.fieldnames), rows


def apply_reviewed_task2_decisions(
    *,
    movie_spec: dict[str, Any],
    draft: dict[str, Any],
    source_task2_path: Path,
    source_script_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if movie_spec.get("review_status") != "approved_full_audit":
        raise ValueError("Task 2 movie review is not approved_full_audit")
    for field in ("slot_id", "movie_id", "title", "language"):
        if movie_spec.get(field) != draft.get(field):
            raise ValueError(f"Task 2 review identity mismatch: {field}")
    if sha256_file(source_task2_path) != draft.get("source_task2_sha256"):
        raise ValueError("Task 2 baseline hash drift")
    if sha256_file(source_script_path) != draft.get("source_script_sha256"):
        raise ValueError("Task 2 screenplay hash drift")

    fieldnames, rows = read_task2_csv(source_task2_path)
    local_ids = [f"Q{index}" for index in range(1, len(rows) + 1)]
    coverage = movie_spec.get("review_coverage") or {}
    if coverage != {
        "mode": "all_baseline_rows",
        "qa_count": len(rows),
        "qa_ids_sha256": sha256_json(local_ids),
    }:
        raise ValueError("Task 2 review does not prove exact full-row coverage")
    if draft.get("reviewed_qa_ids") != local_ids:
        raise ValueError("Task 2 draft review coverage differs from baseline")

    scenes = load_scenes(source_script_path)
    evidence_by_id = {
        item["evidence_id"]: item for item in draft.get("reviewed_evidence", [])
    }
    replacements = movie_spec.get("replacements")
    if not isinstance(replacements, list) or len(replacements) > MAX_REPLACEMENTS_PER_MOVIE:
        raise ValueError("Task 2 replacement list is invalid or exceeds five")
    output_rows = [dict(row) for row in rows]
    seen_sources: set[str] = set()
    audit_rows = []
    for decision in replacements:
        source_id = decision.get("source_qa_id")
        if source_id not in local_ids or source_id in seen_sources:
            raise ValueError(f"Invalid or duplicate Task 2 source QA: {source_id}")
        index = local_ids.index(source_id)
        before = rows[index]
        if str(decision.get("source_row_id")) != before["id"]:
            raise ValueError(f"Task 2 source row ID mismatch: {source_id}")
        issue_type = decision.get("issue_type")
        if issue_type not in ALLOWED_ISSUE_TYPES:
            raise ValueError(f"Invalid Task 2 issue type: {issue_type}")
        duplicates = decision.get("duplicate_with")
        if (
            not isinstance(duplicates, list)
            or len(duplicates) != len(set(duplicates))
            or any(item not in local_ids or item == source_id for item in duplicates)
            or (issue_type == "semantic_duplicate" and not duplicates)
        ):
            raise ValueError(f"Invalid duplicate references for {source_id}")
        if not str(decision.get("reason", "")).strip():
            raise ValueError(f"Task 2 decision has no reason: {source_id}")
        if decision.get("review_status") != "approved":
            raise ValueError(f"Task 2 decision is not approved: {source_id}")
        replacement = decision.get("replacement")
        expected_fields = {
            "question",
            "answer",
            "evidence_or_reason",
            "question_type",
            "related_scene_orders",
            "source_evidence_ids",
        }
        if not isinstance(replacement, dict) or set(replacement) != expected_fields:
            raise ValueError(f"Task 2 replacement fields mismatch: {source_id}")
        for field in ("question", "answer", "evidence_or_reason", "question_type"):
            if not isinstance(replacement[field], str) or not replacement[field].strip():
                raise ValueError(f"Empty Task 2 replacement field: {source_id}/{field}")
        if replacement["question_type"] != before["question_type"]:
            raise ValueError(f"Task 2 replacement changes question type: {source_id}")
        scene_orders = replacement["related_scene_orders"]
        if (
            not isinstance(scene_orders, list)
            or not scene_orders
            or len(scene_orders) != len(set(scene_orders))
            or any(
                not isinstance(order, int)
                or isinstance(order, bool)
                or order < 1
                or order > len(scenes)
                for order in scene_orders
            )
        ):
            raise ValueError(f"Invalid Task 2 screenplay scene orders: {source_id}")
        evidence_ids = replacement["source_evidence_ids"]
        if (
            not isinstance(evidence_ids, list)
            or len(evidence_ids) != len(set(evidence_ids))
            or any(item not in evidence_by_id for item in evidence_ids)
        ):
            raise ValueError(f"Invalid Task 2 reviewed evidence IDs: {source_id}")
        if evidence_ids:
            evidence_scenes = {
                int(order)
                for evidence_id in evidence_ids
                for order in evidence_by_id[evidence_id]["scene_orders"]
            }
            if not evidence_scenes.intersection(scene_orders):
                raise ValueError(
                    f"Task 2 reviewed evidence has no screenplay-scene overlap: {source_id}"
                )
        after = dict(before)
        after.update(
            {
                "related_scenes": " ; ".join(scenes[order - 1].title for order in scene_orders),
                "question": replacement["question"].strip(),
                "answer": replacement["answer"].strip(),
                "evidence_or_reason": replacement["evidence_or_reason"].strip(),
                "question_type": replacement["question_type"],
            }
        )
        output_rows[index] = after
        audit_rows.append(
            {
                "source_qa_id": source_id,
                "source_row_id": before["id"],
                "issue_type": issue_type,
                "duplicate_with": duplicates,
                "reason": decision["reason"],
                "review_status": decision["review_status"],
                "source_evidence_ids": evidence_ids,
                "source_scene_orders": scene_orders,
                "before": before,
                "after": after,
            }
        )
        seen_sources.add(source_id)

    normalized = [_normalize_text(row["question"]) for row in output_rows]
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("Reviewed Task 2 contains empty or exact-normalized duplicate questions")
    if Counter(row["question_type"] for row in rows) != Counter(
        row["question_type"] for row in output_rows
    ):
        raise ValueError("Task 2 question type distribution changed")
    for index, (before, after) in enumerate(zip(rows, output_rows, strict=True), start=1):
        for field in ("id", "movie_id", "language", "question_type"):
            if before[field] != after[field]:
                raise ValueError(f"Task 2 frozen field changed at row {index}: {field}")
        if f"Q{index}" not in seen_sources and before != after:
            raise ValueError(f"Unreviewed Task 2 row changed: Q{index}")
    audit = {
        "schema_version": "stage_task2_movie_quality_review_v1",
        "status": "human_reviewed",
        "slot_id": draft["slot_id"],
        "movie_id": draft["movie_id"],
        "title": draft["title"],
        "language": draft["language"],
        "review_coverage": coverage,
        "counts": {
            "baseline_rows": len(rows),
            "reviewed_rows": len(rows),
            "replacements": len(audit_rows),
            "semantic_duplicates": sum(
                item["issue_type"] == "semantic_duplicate" for item in audit_rows
            ),
            "low_quality": sum(item["issue_type"] == "low_quality" for item in audit_rows),
        },
        "source_task2_path": str(source_task2_path.resolve()),
        "source_task2_sha256": sha256_file(source_task2_path),
        "source_script_path": str(source_script_path.resolve()),
        "source_script_sha256": sha256_file(source_script_path),
        "source_draft_input_sha256": draft.get("input_sha256"),
        "replacements": audit_rows,
        "fieldnames": fieldnames,
        "type_distribution": dict(sorted(Counter(row["question_type"] for row in rows).items())),
    }
    return output_rows, audit


def validate_task2_movie_review(
    audit_path: Path, *, reviewed_task2_path: Path | None = None
) -> dict[str, Any]:
    audit = load_json(audit_path)
    if not isinstance(audit, dict) or audit.get("schema_version") != "stage_task2_movie_quality_review_v1":
        raise ValueError("Unsupported Task 2 movie review schema")
    if audit.get("status") != "human_reviewed":
        raise ValueError("Task 2 movie review is not human_reviewed")
    counts = audit.get("counts") or {}
    if int(counts.get("baseline_rows", -1)) != int(counts.get("reviewed_rows", -2)):
        raise ValueError("Task 2 row count changed")
    replacement_count = int(counts.get("replacements", -1))
    if replacement_count < 0 or replacement_count > MAX_REPLACEMENTS_PER_MOVIE:
        raise ValueError("Task 2 replacement count is outside 0-5")
    replacements = audit.get("replacements")
    if not isinstance(replacements, list) or len(replacements) != replacement_count:
        raise ValueError("Task 2 replacement provenance count mismatch")
    baseline = Path(str(audit.get("source_task2_path", "")))
    if not baseline.is_file() or sha256_file(baseline) != audit.get("source_task2_sha256"):
        raise ValueError("Task 2 baseline artifact drift")
    if reviewed_task2_path is not None:
        _, rows = read_task2_csv(reviewed_task2_path)
        if len(rows) != int(counts["reviewed_rows"]):
            raise ValueError("Reviewed Task 2 CSV row count mismatch")
        expected = audit.get("reviewed_task2_sha256")
        if expected and sha256_file(reviewed_task2_path) != expected:
            raise ValueError("Reviewed Task 2 CSV hash drift")
    return audit


def qa_ids_sha256(count: int) -> str:
    return sha256_json([f"Q{index}" for index in range(1, count + 1)])
