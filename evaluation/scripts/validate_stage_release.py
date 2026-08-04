#!/usr/bin/env python3
"""Fail-closed structural and checksum validation for a STAGE public release."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.io import atomic_write_json, load_json, sha256_file  # noqa: E402
from stage_narrative.screenplay_titles import redundant_public_scene_titles  # noqa: E402
from stage_narrative.task3.release_validation import validate_role_assets  # noqa: E402


PUBLIC_TASK2_TYPES = {
    "Scene Grounding",
    "Character Understanding",
    "Causal-Motivational Reasoning",
    "Temporal Reasoning",
    "Narrative Progression",
    "Role-Relation Continuity",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    manifest = load_json(root / "manifest.json")
    include_screenplay = bool(
        manifest.get("scope", {}).get("screenplay_files_included", False)
    )
    errors: list[str] = []
    counts: Counter[str] = Counter()
    if manifest.get("release_status") != "complete":
        errors.append("top-level manifest is not complete")
    entries = manifest.get("entries", [])
    if len(entries) != 151:
        errors.append(f"expected 151 ordinary entries, found {len(entries)}")
    ordinary_ids = set()
    task2_signatures: dict[str, dict[str, list[tuple[str, str]]]] = {
        "STAGE": {},
        "STAGE_Anon": {},
    }
    observed_task2_types: set[str] = set()
    for condition_dir in ("STAGE", "STAGE_Anon"):
        collection_path = root / condition_dir / "manifest.json"
        collection = load_json(collection_path)
        collection_entries = collection.get("entries", [])
        if len(collection_entries) != 151:
            errors.append(
                f"{condition_dir}: expected 151 entries, found {len(collection_entries)}"
            )
        ids = {item.get("movie_id") for item in collection_entries}
        if len(ids) != len(collection_entries):
            errors.append(f"{condition_dir}: duplicate movie IDs")
        if condition_dir == "STAGE":
            ordinary_ids = ids
        elif ids != ordinary_ids:
            errors.append("ordinary and anonymous movie inventories differ")
        for item in collection_entries:
            movie_dir = root / condition_dir / item["path"]
            try:
                error = _validate_movie(
                    movie_dir,
                    cohort=item.get("cohort"),
                    include_screenplay=include_screenplay,
                )
            except Exception as exc:
                error = f"movie validation failed: {type(exc).__name__}: {exc}"
            if error:
                errors.append(f"{condition_dir}/{item.get('movie_id')}: {error}")
            try:
                signature = _task2_signature(movie_dir)
                task2_signatures[condition_dir][str(item["movie_id"])] = signature
                counts[f"{condition_dir}_task2_questions"] += len(signature)
                if condition_dir == "STAGE":
                    observed_task2_types.update(value for _, value in signature)
            except Exception as exc:
                errors.append(
                    f"{condition_dir}/{item.get('movie_id')}: Task 2 validation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            counts[f"{condition_dir}_movies"] += 1
    for movie_id in ordinary_ids:
        if task2_signatures["STAGE"].get(str(movie_id)) != task2_signatures[
            "STAGE_Anon"
        ].get(str(movie_id)):
            errors.append(f"ordinary and anonymous Task 2 pairing differs: {movie_id}")
    if observed_task2_types != PUBLIC_TASK2_TYPES:
        errors.append(
            "Task 2 public type inventory mismatch: "
            f"observed={sorted(observed_task2_types)}"
        )
    expected_task2 = manifest.get("counts", {}).get("task2_questions")
    if expected_task2 != counts["STAGE_task2_questions"]:
        errors.append(
            "root Task 2 count mismatch: "
            f"manifest={expected_task2} actual={counts['STAGE_task2_questions']}"
        )
    for artifact in manifest.get("artifacts", {}).get("files", []):
        path = root / artifact["path"]
        if not path.is_file():
            errors.append(f"missing checksummed artifact: {artifact['path']}")
            continue
        if path.stat().st_size != int(artifact["bytes"]):
            errors.append(f"size mismatch: {artifact['path']}")
        elif sha256_file(path) != artifact["sha256"]:
            errors.append(f"checksum mismatch: {artifact['path']}")
        counts["checksummed_files"] += 1
    payload: dict[str, Any] = {
        "schema_version": "stage_complete_release_validation_v1",
        "status": "passed" if not errors else "failed",
        "release_root": str(root),
        "counts": dict(sorted(counts.items())),
        "errors": errors,
    }
    output = args.output.resolve() if args.output else root / "validation_report.json"
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


def _validate_movie(
    movie_dir: Path,
    *,
    cohort: str | None,
    include_screenplay: bool,
) -> str | None:
    required = {
        "info.json",
        "task_1_reference_state_update.json",
        "task_1_autoregressive_state_update.json",
        "task_3_role_assets.json",
        "task_3_single_turn.json",
        "task_3_multi_turn.json",
    }
    if include_screenplay:
        required.add("script.json")
    names = {path.name for path in movie_dir.iterdir()} if movie_dir.is_dir() else set()
    missing = required - names
    if missing:
        return f"missing files: {sorted(missing)}"
    task2 = names & {"task_2_question_answering.csv", "task_2_question_answering.json"}
    if task2 != {"task_2_question_answering.csv"}:
        return "expected exactly one Task 2 CSV"
    info = load_json(movie_dir / "info.json")
    signature = _task2_signature(movie_dir)
    if not 30 <= len(signature) <= 37:
        return f"Task 2 question count outside 30--37: {len(signature)}"
    if len(signature) != len({item[0] for item in signature}):
        return "duplicate Task 2 question IDs"
    invalid_types = {value for _, value in signature if value not in PUBLIC_TASK2_TYPES}
    if invalid_types:
        return f"invalid Task 2 question types: {sorted(invalid_types)}"
    if info.get("counts", {}).get("task2_questions") != len(signature):
        return "info.json Task 2 count mismatch"
    task2_path = movie_dir / "task_2_question_answering.csv"
    task2_meta = info.get("files", {}).get(task2_path.name)
    if not isinstance(task2_meta, dict):
        return "info.json is missing Task 2 CSV metadata"
    if task2_meta.get("bytes") != task2_path.stat().st_size:
        return "info.json Task 2 byte count mismatch"
    if task2_meta.get("sha256") != sha256_file(task2_path):
        return "info.json Task 2 checksum mismatch"
    payloads = {}
    for name in required - {"info.json"}:
        try:
            payloads[name] = load_json(movie_dir / name)
        except Exception as exc:
            return f"invalid JSON in {name}: {type(exc).__name__}: {exc}"
    if include_screenplay:
        redundant = redundant_public_scene_titles(payloads["script.json"])
        if redundant:
            return f"redundant scene-title numbers: {len(redundant)}"
        script_sha256 = sha256_file(movie_dir / "script.json")
        for name in (
            "task_1_reference_state_update.json",
            "task_1_autoregressive_state_update.json",
        ):
            script_ref = payloads[name].get("script", {})
            if script_ref.get("file") != "script.json":
                return f"Task 1 script filename drift in {name}: {script_ref.get('file')}"
            if script_ref.get("sha256") != script_sha256:
                return f"Task 1 script checksum drift in {name}"
    movie_id = movie_dir.name
    for name, payload in payloads.items():
        declared = payload.get("movie_id") if isinstance(payload, dict) else None
        if declared is not None and str(declared) != movie_id:
            return f"movie_id mismatch in {name}: {declared}"
    try:
        validate_role_assets(payloads["task_3_role_assets.json"])
        multi = payloads["task_3_multi_turn.json"]
        roles = {
            item["character_id"]
            for item in payloads["task_3_role_assets.json"].get("roles", [])
        }
        episodes = multi.get("episodes")
        if multi.get("schema_version") != "stage_task3_multi_turn" or not isinstance(
            episodes, list
        ):
            return "unsupported Task 3 multi-turn schema"
        if int(multi.get("episode_count", -1)) != len(episodes):
            return "Task 3 multi-turn episode_count mismatch"
        turns = 0
        for episode in episodes:
            character_id = episode.get("character_id")
            if character_id not in roles:
                return f"Task 3 multi-turn role reference missing: {character_id}"
            if episode.get("role_asset_ref") != {
                "asset_file": "task_3_role_assets.json",
                "character_id": character_id,
            }:
                return f"Task 3 multi-turn role reference invalid: {character_id}"
            if episode.get("memory_context_policy") != {
                "mode": "all_role_memories",
                "retrieval": "none",
            }:
                return f"Task 3 multi-turn memory policy invalid: {character_id}"
            turns += len(episode.get("turns", []))
        if int(multi.get("turn_count", -1)) != turns:
            return "Task 3 multi-turn turn_count mismatch"
    except Exception as exc:
        return f"Task 3 multi-turn validation failed: {type(exc).__name__}: {exc}"
    return None


def _task2_signature(movie_dir: Path) -> list[tuple[str, str]]:
    path = movie_dir / "task_2_question_answering.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "id",
        "movie_id",
        "language",
        "related_scenes",
        "question",
        "answer",
        "evidence_or_reason",
        "question_type",
    }
    if not rows:
        raise ValueError("empty Task 2 CSV")
    if set(rows[0]) != required:
        raise ValueError(f"Task 2 CSV fields mismatch: {sorted(rows[0])}")
    nonempty = {"id", "movie_id", "language", "question", "answer", "question_type"}
    for index, row in enumerate(rows, start=1):
        if row["movie_id"] != movie_dir.name:
            raise ValueError(f"movie_id mismatch at row {index}")
        if any(not row[field].strip() for field in nonempty):
            raise ValueError(f"empty required field at row {index}")
    return [(row["id"], row["question_type"]) for row in rows]


if __name__ == "__main__":
    main()
