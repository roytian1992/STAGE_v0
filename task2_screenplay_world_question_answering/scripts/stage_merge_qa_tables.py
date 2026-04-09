import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from stage_refine_plus_qas import clean_text, iter_movie_paths, load_csv_rows


OUTPUT_COLUMNS = [
    "id",
    "related_scenes",
    "question",
    "answer",
    "related_context",
    "question_type",
    "question_source",
]


def normalize_text(value: object) -> str:
    return clean_text(value)


def classify_broad_coverage_source(scene_text: str) -> str:
    return "broad_coverage_event_chain" if "|" in scene_text else "broad_coverage_scene"


def load_doc2chunks(movie_path: Path) -> Dict[str, object]:
    path = movie_path / "doc2chunks.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scene_title_from_key(doc2chunks: Dict[str, object], scene_key: str) -> str:
    try:
        return normalize_text(doc2chunks[scene_key]["document_metadata"].get("doc_title")) or scene_key
    except Exception:
        return scene_key


def motif_related_scenes(row: Dict[str, str], doc2chunks: Dict[str, object]) -> str:
    support_scene_keys = [normalize_text(x) for x in str(row.get("support_scene_keys", "")).split("|") if normalize_text(x)]
    if not support_scene_keys:
        return normalize_text(row.get("scene"))

    titles: List[str] = []
    seen = set()
    for key in support_scene_keys:
        title = scene_title_from_key(doc2chunks, key)
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return "|".join(titles) if titles else normalize_text(row.get("scene"))


def from_question_pairs(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged_rows: List[Dict[str, str]] = []
    for row in rows:
        scene = normalize_text(row.get("scene"))
        merged_rows.append(
            {
                "related_scenes": scene,
                "question": normalize_text(row.get("question")),
                "answer": normalize_text(row.get("answer")),
                "related_context": normalize_text(row.get("evidence")),
                "question_type": normalize_text(row.get("qa_type")),
                "question_source": classify_broad_coverage_source(scene),
            }
        )
    return merged_rows


def from_question_pairs_motif(
    rows: List[Dict[str, str]],
    doc2chunks: Dict[str, object],
    full_rows_by_id: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    merged_rows: List[Dict[str, str]] = []
    for row in rows:
        full_row = None
        if full_rows_by_id is not None:
            full_row = full_rows_by_id.get(normalize_text(row.get("id")))
        merged_rows.append(
            {
                "related_scenes": motif_related_scenes(full_row or row, doc2chunks),
                "question": normalize_text(row.get("question")),
                "answer": normalize_text(row.get("answer")),
                "related_context": normalize_text(row.get("evidence")),
                "question_type": normalize_text(row.get("qa_type")),
                "question_source": "supplementary_cross_scene",
            }
        )
    return merged_rows


def write_rows(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def merge_movie(movie_path: Path, output_name: str) -> Optional[int]:
    question_pairs_path = movie_path / "question_pairs.csv"
    motif_path = movie_path / "question_pairs_motif.csv"
    motif_full_path = movie_path / "question_pairs_motif_full.csv"
    if not question_pairs_path.exists() and not motif_path.exists() and not motif_full_path.exists():
        return None

    merged_rows: List[Dict[str, str]] = []
    if question_pairs_path.exists():
        merged_rows.extend(from_question_pairs(load_csv_rows(question_pairs_path)))
    if motif_path.exists() or motif_full_path.exists():
        motif_rows = load_csv_rows(motif_path) if motif_path.exists() else load_csv_rows(motif_full_path)
        full_rows_by_id = None
        if motif_full_path.exists():
            full_rows_by_id = {
                normalize_text(row.get("id")): row
                for row in load_csv_rows(motif_full_path)
            }
        merged_rows.extend(from_question_pairs_motif(motif_rows, load_doc2chunks(movie_path), full_rows_by_id))

    for idx, row in enumerate(merged_rows, start=1):
        row["id"] = str(idx)

    write_rows(merged_rows, movie_path / output_name)
    return len(merged_rows)


def write_global_index(root_dir: Path, output_name: str, global_name: str) -> int:
    all_rows: List[Dict[str, str]] = []
    for movie_path in iter_movie_paths(root_dir, None):
        merged_path = movie_path / output_name
        if not merged_path.exists():
            continue
        movie_id = movie_path.name
        for row in load_csv_rows(merged_path):
            current = dict(row)
            current["id"] = f"{movie_id}_{normalize_text(row.get('id'))}"
            all_rows.append(current)

    write_rows(all_rows, root_dir / global_name)
    return len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge broad-coverage and motif QA tables into one normalized table.")
    parser.add_argument("--root-dir", type=Path, default=None, help="Path like STAGEBenchmark/")
    parser.add_argument("--movie-path", type=Path, default=None, help="Single movie directory")
    parser.add_argument("--output-name", default="question_pairs_merged.csv", help="Per-movie merged output name")
    parser.add_argument(
        "--global-name",
        default="question_pairs_merged_all.csv",
        help="Optional root-level aggregate name when --root-dir is used",
    )
    args = parser.parse_args()

    if args.root_dir is None and args.movie_path is None:
        raise SystemExit("Provide either --root-dir or --movie-path.")

    total_movies = 0
    total_rows = 0
    for movie_path in iter_movie_paths(args.root_dir, args.movie_path):
        result = merge_movie(movie_path, args.output_name)
        if result is None:
            continue
        total_movies += 1
        total_rows += result

    if args.root_dir is not None:
        global_rows = write_global_index(args.root_dir, args.output_name, args.global_name)
        print(f"Merged {total_movies} movies, wrote {total_rows} rows, global table rows: {global_rows}.")
        return

    print(f"Merged {total_movies} movies, wrote {total_rows} rows.")


if __name__ == "__main__":
    main()
