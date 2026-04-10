import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional

from stage_bootstrap_story_dynamics import dump_json
from stage_build_gold_character_timelines import build_client, build_outputs_for_movie


def iter_movie_paths(root_dir: Path, language_filter: Optional[str]) -> Iterable[Path]:
    language_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    for language_dir in language_dirs:
        language_name = language_dir.name.lower()
        if language_filter and language_name != language_filter.lower():
            continue
        for movie_path in sorted(path for path in language_dir.iterdir() if path.is_dir()):
            yield movie_path


def is_eligible_movie(movie_path: Path, require_local_v2: bool) -> bool:
    required = [
        movie_path / "story_dynamics_global_refined.json",
        movie_path / "state_facts.json",
        movie_path / "doc2chunks.json",
    ]
    if require_local_v2:
        required.append(movie_path / "story_dynamics_local_v2.json")
    else:
        has_any_local = any(
            path.exists()
            for path in (
                movie_path / "story_dynamics_local_refined.json",
                movie_path / "story_dynamics_local_v2.json",
                movie_path / "story_dynamics_local.json",
            )
        )
        if not has_any_local:
            return False
    return all(path.exists() for path in required)


def has_existing_outputs(movie_path: Path) -> bool:
    return (
        (movie_path / "gold_character_timelines_v1.json").exists()
        and (movie_path / "gold_cross_scene_arcs_v1.json").exists()
        and (movie_path / "gold_character_timeline_builder_audit_v1.json").exists()
    )


def emit_log_line(payload: dict, log_path: Optional[Path]) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch build gold focal-character timelines and cross-scene arcs.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/vepfs-mlp2/c20250513/241404044/users/roytian/benchmarks/STAGEBenchmark"),
    )
    parser.add_argument("--language", type=str, default=None, help="Optional language directory filter: Chinese or English")
    parser.add_argument("--max-movies", type=int, default=None)
    parser.add_argument("--max-characters", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-final-nodes", type=int, default=6)
    parser.add_argument("--max-final-nodes", type=int, default=12)
    parser.add_argument("--max-bundles-per-character", type=int, default=0)
    parser.add_argument("--api-base", default="https://api.xiaomimimo.com/v1/chat/completions")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="mimo-v2-pro")
    parser.add_argument("--fallback-api-base", default="http://localhost:8001/v1")
    parser.add_argument("--fallback-api-key", default="token-abc123")
    parser.add_argument("--fallback-model", default="Qwen3-235B")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-missing-local-v2", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client(args)
    max_bundles_per_character = args.max_bundles_per_character if args.max_bundles_per_character > 0 else None
    require_local_v2 = not args.allow_missing_local_v2

    processed_movies = 0
    ok_movies = 0
    skipped_movies = 0
    error_movies = 0
    summary_rows: List[dict] = []

    for movie_path in iter_movie_paths(args.root_dir, args.language):
        if not is_eligible_movie(movie_path, require_local_v2=require_local_v2):
            continue
        if args.max_movies is not None and processed_movies >= args.max_movies:
            break
        if not args.no_resume and has_existing_outputs(movie_path):
            skipped_movies += 1
            emit_log_line(
                {"movie_id": movie_path.name, "language": movie_path.parent.name, "status": "skipped_existing"},
                args.log_path,
            )
            continue

        emit_log_line(
            {"movie_id": movie_path.name, "language": movie_path.parent.name, "status": "started"},
            args.log_path,
        )
        processed_movies += 1
        last_exc: Optional[Exception] = None
        for attempt in range(args.max_retries + 1):
            try:
                if attempt > 0:
                    emit_log_line(
                        {
                            "movie_id": movie_path.name,
                            "language": movie_path.parent.name,
                            "status": "retrying",
                            "attempt": attempt,
                        },
                        args.log_path,
                    )
                timeline_output, arc_output, audit_output = build_outputs_for_movie(
                    movie_dir=movie_path,
                    client=client,
                    max_characters=args.max_characters,
                    batch_size=args.batch_size,
                    max_workers=args.workers,
                    min_final_nodes=args.min_final_nodes,
                    max_final_nodes=args.max_final_nodes,
                    max_bundles_per_character=max_bundles_per_character,
                )
                dump_json(movie_path / "gold_character_timelines_v1.json", timeline_output)
                dump_json(movie_path / "gold_cross_scene_arcs_v1.json", arc_output)
                dump_json(movie_path / "gold_character_timeline_builder_audit_v1.json", audit_output)
                row = {
                    "movie_id": movie_path.name,
                    "language": movie_path.parent.name,
                    "status": "ok",
                    "selected_focal_character_count": timeline_output["build_summary"]["selected_focal_character_count"],
                    "timeline_node_count": timeline_output["build_summary"]["timeline_node_count"],
                    "arc_count": arc_output["build_summary"]["arc_count"],
                    "attempt": attempt,
                }
                ok_movies += 1
                summary_rows.append(row)
                emit_log_line(row, args.log_path)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= args.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        if last_exc is not None:
            error_movies += 1
            row = {
                "movie_id": movie_path.name,
                "language": movie_path.parent.name,
                "status": "error",
                "error": f"{type(last_exc).__name__}: {last_exc}",
                "attempt": args.max_retries,
            }
            summary_rows.append(row)
            emit_log_line(row, args.log_path)

    final_summary = {
        "processed_movies": processed_movies,
        "ok_movies": ok_movies,
        "skipped_movies": skipped_movies,
        "error_movies": error_movies,
        "require_local_v2": require_local_v2,
    }
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(final_summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
