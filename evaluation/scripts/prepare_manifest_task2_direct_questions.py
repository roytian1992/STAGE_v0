#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.io import atomic_write_json  # noqa: E402
from stage_narrative.task2_hybrid import (  # noqa: E402
    artifact,
    load_manifest_entries,
    read_task2_rows,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze paired Manifest Task 2 questions for closed-book direct QA."
    )
    parser.add_argument("--ordinary-root", type=Path, required=True)
    parser.add_argument("--anonymous-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite direct-question run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    ordinary = load_manifest_entries(args.ordinary_root)
    anonymous = load_manifest_entries(args.anonymous_root)
    if set(ordinary) != set(anonymous):
        raise ValueError("Ordinary and anonymous manifests have different movie coverage")

    output: dict[str, list[dict[str, Any]]] = {"ordinary": [], "anonymous": []}
    movie_records = []
    for movie_id in sorted(ordinary):
        left = ordinary[movie_id]
        right = anonymous[movie_id]
        if left["language"] != right["language"]:
            raise ValueError(f"Paired movie languages differ: {movie_id}")
        paths = {
            "ordinary": left["movie_dir"] / "task_2_question_answering.csv",
            "anonymous": right["movie_dir"] / "task_2_question_answering.csv",
        }
        rows = {variant: read_task2_rows(path) for variant, path in paths.items()}
        if [row["id"] for row in rows["ordinary"]] != [
            row["id"] for row in rows["anonymous"]
        ]:
            raise ValueError(f"Paired Task 2 question ids differ: {movie_id}")
        if [row["question_type"] for row in rows["ordinary"]] != [
            row["question_type"] for row in rows["anonymous"]
        ]:
            raise ValueError(f"Paired Task 2 question types differ: {movie_id}")
        for variant in ("ordinary", "anonymous"):
            for row in rows[variant]:
                output[variant].append(
                    {
                        "pairing_id": f"{movie_id}:{row['id']}",
                        "question_id": row["id"],
                        "movie_id": movie_id,
                        "language": left["language"],
                        "variant": variant,
                        "question": row["question"],
                        "question_type": row["question_type"],
                        "reference_answer": row["answer"],
                        "reference_evidence_or_reason": row["evidence_or_reason"],
                    }
                )
        movie_records.append(
            {
                "movie_id": movie_id,
                "language": left["language"],
                "ordinary_task2": artifact(paths["ordinary"]),
                "anonymous_task2": artifact(paths["anonymous"]),
                "question_count_per_variant": len(rows["ordinary"]),
            }
        )

    question_artifacts = {}
    for variant in ("ordinary", "anonymous"):
        path = output_root / "questions" / f"{variant}.jsonl"
        write_jsonl(path, output[variant])
        question_artifacts[variant] = artifact(path)
    pair_keys = {
        variant: [(row["movie_id"], row["question_id"]) for row in output[variant]]
        for variant in output
    }
    if pair_keys["ordinary"] != pair_keys["anonymous"]:
        raise AssertionError("Direct question pairs are not aligned")

    manifest_path = output_root / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task2_direct_question_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "completed",
            "python_executable": sys.executable,
            "source_roots": {
                "ordinary": str(args.ordinary_root.resolve()),
                "anonymous": str(args.anonymous_root.resolve()),
            },
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
                "questions_per_variant": len(output["ordinary"]),
                "questions_total": sum(len(rows) for rows in output.values()),
            },
            "questions": question_artifacts,
            "movies": movie_records,
        },
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
