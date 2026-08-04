#!/usr/bin/env python3
"""Freeze paired question-only Task 2 inputs from a complete STAGE release."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.io import atomic_write_json, load_json  # noqa: E402
from stage_narrative.task2_hybrid import artifact, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-root", type=Path, required=True)
    parser.add_argument("--anonymous-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Task 2 question run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    roots = {
        "ordinary": args.ordinary_root.resolve(),
        "anonymous": args.anonymous_root.resolve(),
    }
    entries = {variant: _entries(root) for variant, root in roots.items()}
    if set(entries["ordinary"]) != set(entries["anonymous"]):
        raise ValueError("Ordinary and anonymous Task 2 movie inventories differ")
    rows: dict[str, list[dict[str, Any]]] = {"ordinary": [], "anonymous": []}
    movie_records = []
    for movie_id in sorted(entries["ordinary"]):
        paired = {}
        for variant in rows:
            entry = entries[variant][movie_id]
            paired[variant] = _read_questions(roots[variant] / entry["path"])
        left_ids = [item["question_id"] for item in paired["ordinary"]]
        right_ids = [item["question_id"] for item in paired["anonymous"]]
        if left_ids != right_ids:
            raise ValueError(f"Paired Task 2 question IDs differ: {movie_id}")
        language = "zh" if entries["ordinary"][movie_id]["language"] == "Chinese" else "en"
        for variant in rows:
            for item in paired[variant]:
                rows[variant].append(
                    {
                        "pairing_id": f"{movie_id}:{item['question_id']}",
                        "question_id": item["question_id"],
                        "movie_id": movie_id,
                        "language": language,
                        "variant": variant,
                        "question": item["question"],
                        "question_type": item["question_type"],
                        "reference_answer": item["answer"],
                        "reference_evidence_or_reason": item["evidence"],
                    }
                )
        movie_records.append(
            {
                "movie_id": movie_id,
                "language": language,
                "cohort": entries["ordinary"][movie_id]["cohort"],
                "question_count_per_variant": len(paired["ordinary"]),
            }
        )
    question_artifacts = {}
    for variant in rows:
        path = output_root / "questions" / f"{variant}.jsonl"
        write_jsonl(path, rows[variant])
        question_artifacts[variant] = artifact(path)
    manifest = {
        "schema_version": "stage_task2_direct_question_manifest",
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_roots": {key: str(value) for key, value in roots.items()},
        "input_contract": {
            "prediction_visible_fields": ["question"],
            "screenplay_provided": False,
            "retrieved_passages_provided": False,
            "memory_provided": False,
            "scene_metadata_provided": False,
            "reference_answer_provided": False,
            "reference_evidence_provided": False,
            "rag_used": False,
        },
        "counts": {
            "movie_count": len(movie_records),
            "questions_per_variant": len(rows["ordinary"]),
            "questions_total": len(rows["ordinary"]) + len(rows["anonymous"]),
        },
        "questions": question_artifacts,
        "movies": movie_records,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(output_root / "manifest.json")
    print(manifest["counts"])


def _entries(root: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(root / "manifest.json")
    supported = {
        "stage_complete_condition_release_v1",
        "stage_final_release_condition_v1",
    }
    if manifest.get("schema_version") not in supported:
        raise ValueError(f"Unsupported STAGE condition manifest: {root}")
    return {item["movie_id"]: item for item in manifest.get("entries", [])}


def _read_questions(movie_dir: Path) -> list[dict[str, str]]:
    json_path = movie_dir / "task_2_question_answering.json"
    if json_path.is_file():
        payload = load_json(json_path)
        return [
            {
                "question_id": str(item["question_id"]),
                "question": str(item["question"]),
                "answer": str(item["answer"]),
                "question_type": str(item.get("question_family") or "multi_scene"),
                "evidence": "\n".join(
                    f"scene {claim['scene_order']}: {claim['claim']}"
                    for claim in item.get("evidence_claims", [])
                ),
            }
            for item in payload.get("questions", [])
        ]
    csv_path = movie_dir / "task_2_question_answering.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    output = []
    for index, item in enumerate(source, start=1):
        output.append(
            {
                "question_id": str(
                    item.get("id") or item.get("question_id") or f"Q{index:04d}"
                ),
                "question": str(item.get("question") or ""),
                "answer": str(item.get("answer") or ""),
                "question_type": str(
                    item.get("question_type") or item.get("type") or "legacy"
                ),
                "evidence": str(
                    item.get("evidence_or_reason")
                    or item.get("reason")
                    or item.get("evidence")
                    or ""
                ),
            }
        )
    return output


if __name__ == "__main__":
    main()
