#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian")
STAGE_ROOT = ROOT / "STAGE_v0"
TOOLS_DIR = STAGE_ROOT / "tools" / "task3"
DEFAULT_PYTHON_BIN = "/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python"
DEFAULT_ROLES_JSON = ROOT / "task3_single_turn_bundle_manifest40_20260420_run1_roles.json"
DEFAULT_OUTPUT_ROOT = ROOT / "task3_manifest40_multi_turn_eval_20260421_qwen3_8001_run1"
DEFAULT_BASE_URL = "http://localhost:8001/v1"
DEFAULT_API_KEY = "token-abc123"
DEFAULT_MODEL = "Qwen3-235B"
SUPPORTED_MODES = ["persona_only", "full_memory_all_in", "bm25_topk", "embedding_topk"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run Task3 multi-turn evaluation over a role list."
    )
    parser.add_argument("--roles-json", type=Path, default=DEFAULT_ROLES_JSON)
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--language", action="append", choices=["Chinese", "English"])
    parser.add_argument("--movie-id", action="append")
    parser.add_argument("--character", action="append")
    parser.add_argument("--modes", nargs="+", default=SUPPORTED_MODES)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-base-url", default="")
    parser.add_argument("--fallback-api-key", default="")
    parser.add_argument("--fallback-model", default="")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--sample-max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=2.0)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--skip-finished", action="store_true")
    return parser.parse_args()


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_roles(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
    elif isinstance(data, dict) and isinstance(data.get("rows"), list):
        rows = [row for row in data["rows"] if isinstance(row, dict)]
    else:
        raise ValueError(f"Unsupported roles json format: {path}")
    return rows


def filter_roles(
    rows: Sequence[Dict[str, Any]],
    *,
    languages: Optional[Sequence[str]],
    movie_ids: Optional[Sequence[str]],
    characters: Optional[Sequence[str]],
) -> List[Dict[str, str]]:
    language_filter = {normalize_ws(item) for item in (languages or []) if normalize_ws(item)}
    movie_filter = {normalize_ws(item) for item in (movie_ids or []) if normalize_ws(item)}
    character_filter = {normalize_ws(item) for item in (characters or []) if normalize_ws(item)}
    out: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        language = normalize_ws(row.get("language"))
        movie_id = normalize_ws(row.get("movie_id"))
        character = normalize_ws(row.get("character"))
        if not language or not movie_id or not character:
            continue
        if language_filter and language not in language_filter:
            continue
        if movie_filter and movie_id not in movie_filter:
            continue
        if character_filter and character not in character_filter:
            continue
        key = (language, movie_id, character)
        if key in seen:
            continue
        seen.add(key)
        out.append({"language": language, "movie_id": movie_id, "character": character})
    out.sort(key=lambda row: (row["language"], row["movie_id"], row["character"]))
    return out


def gather_episode_ids(stage_root: Path, *, language: str, movie_id: str, character: str) -> List[str]:
    payload = load_json(stage_root / language / movie_id / "task_3_in_script_character_role_play_multi_turn.json")
    episode_ids: List[str] = []
    for episode in payload.get("episodes", []) or []:
        if normalize_ws(episode.get("character")) != character:
            continue
        episode_id = normalize_ws(episode.get("episode_id"))
        instance_id = normalize_ws(episode.get("instance_id"))
        legacy_id = normalize_ws(episode.get("episode_instance_id"))
        sample_id = episode_id or instance_id or legacy_id
        if sample_id:
            episode_ids.append(sample_id)
    episode_ids = sorted(dict.fromkeys(episode_ids))
    return episode_ids


def summarize_output(output_path: Path) -> Dict[str, Any]:
    data = load_json(output_path)
    path_score = (
        (data.get("episode_path_compatibility") or {}).get("average_score")
        if isinstance(data.get("episode_path_compatibility"), dict)
        else None
    )
    legacy_followup = (
        (data.get("followup_compatibility") or {}).get("average_score")
        if isinstance(data.get("followup_compatibility"), dict)
        else None
    )
    return {
        "score": data.get("core_multi_turn_score"),
        "episode_path_compatibility": path_score if path_score is not None else legacy_followup,
        "followup_compatibility": legacy_followup if legacy_followup is not None else path_score,
        "retrieval_diagnostics": data.get("retrieval_diagnostics"),
    }


def build_job_output_path(
    output_root: Path,
    *,
    language: str,
    movie_id: str,
    character: str,
    mode: str,
    episode_id: str,
) -> Path:
    return (
        output_root
        / "runs"
        / language
        / movie_id
        / safe_name(character)
        / mode
        / f"{safe_name(episode_id)}.json"
    )


def run_job(
    *,
    python_bin: str,
    stage_root: Path,
    language: str,
    movie_id: str,
    character: str,
    episode_id: str,
    mode: str,
    output_root: Path,
    base_url: str,
    api_key: str,
    model: str,
    fallback_base_url: str,
    fallback_api_key: str,
    fallback_model: str,
    top_k: int,
    sample_max_attempts: int,
    retry_delay_sec: float,
    skip_finished: bool,
) -> Dict[str, Any]:
    output_path = build_job_output_path(
        output_root,
        language=language,
        movie_id=movie_id,
        character=character,
        mode=mode,
        episode_id=episode_id,
    )
    log_path = output_path.with_suffix(".log")
    if skip_finished and output_path.exists():
        result = {
            "language": language,
            "movie_id": movie_id,
            "character": character,
            "episode_id": episode_id,
            "mode": mode,
            "status": "skipped_finished",
            "output_path": str(output_path),
            "log_path": str(log_path),
            "attempt_count": 0,
            "elapsed_sec": 0.0,
        }
        result.update(summarize_output(output_path))
        return result

    cmd = [
        python_bin,
        str(TOOLS_DIR / "run_task3_multi_turn_episode_eval.py"),
        "--stage-root",
        str(stage_root),
        "--language",
        language,
        "--movie-id",
        movie_id,
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--model",
        model,
        "--top-k",
        str(top_k),
        "--memory-mode",
        mode,
        "--episode-instance-id",
        episode_id,
        "--output-path",
        str(output_path),
    ]

    env = os.environ.copy()
    if fallback_base_url:
        env["TASK3_FALLBACK_BASE_URL"] = fallback_base_url
        env["TASK3_FALLBACK_API_KEY"] = fallback_api_key or api_key
        env["TASK3_FALLBACK_MODEL"] = fallback_model or model

    attempts: List[Dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    total_elapsed = 0.0
    last_row: Optional[Dict[str, Any]] = None
    for attempt in range(1, max(1, sample_max_attempts) + 1):
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(TOOLS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        elapsed_sec = round(time.time() - started, 3)
        total_elapsed += elapsed_sec
        row = {
            "attempt": attempt,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed_sec,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        attempts.append(row)
        last_row = row
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now_ts()}] attempt={attempt} returncode={proc.returncode} elapsed_sec={elapsed_sec}\n")
            if proc.stdout:
                handle.write("[stdout]\n")
                handle.write(proc.stdout[-12000:])
                handle.write("\n")
            if proc.stderr:
                handle.write("[stderr]\n")
                handle.write(proc.stderr[-12000:])
                handle.write("\n")
        if proc.returncode == 0 and output_path.exists():
            break
        if attempt < max(1, sample_max_attempts):
            time.sleep(max(0.0, retry_delay_sec))

    assert last_row is not None
    result = {
        "language": language,
        "movie_id": movie_id,
        "character": character,
        "episode_id": episode_id,
        "mode": mode,
        "status": "finished" if last_row["returncode"] == 0 and output_path.exists() else "failed",
        "returncode": last_row["returncode"],
        "attempt_count": len(attempts),
        "elapsed_sec": round(total_elapsed, 3),
        "attempts": attempts,
        "stdout_tail": last_row["stdout_tail"],
        "stderr_tail": last_row["stderr_tail"],
        "output_path": str(output_path),
        "log_path": str(log_path),
    }
    if result["status"] == "finished":
        result.update(summarize_output(output_path))
    return result


def average(values: Sequence[Any]) -> Optional[float]:
    filtered = [float(value) for value in values if isinstance(value, (int, float))]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 4)


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = normalize_ws(row.get("status")) or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in SUPPORTED_MODES:
        mode_rows = [row for row in rows if row.get("mode") == mode and row.get("status") in {"finished", "skipped_finished"}]
        if not mode_rows:
            continue
        by_mode[mode] = {
            "run_count": len(mode_rows),
            "avg_score": average([row.get("score") for row in mode_rows]),
            "avg_episode_path_compatibility": average([row.get("episode_path_compatibility") for row in mode_rows]),
            "avg_followup_compatibility": average([row.get("followup_compatibility") for row in mode_rows]),
            "elapsed_sec": round(sum(float(row.get("elapsed_sec") or 0.0) for row in mode_rows), 3),
        }

    role_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            normalize_ws(row.get("language")),
            normalize_ws(row.get("movie_id")),
            normalize_ws(row.get("character")),
        )
        role_entry = role_map.setdefault(
            key,
            {
                "language": key[0],
                "movie_id": key[1],
                "character": key[2],
                "job_count": 0,
                "status_counts": {},
                "mode_scores": {},
            },
        )
        status = normalize_ws(row.get("status")) or "unknown"
        role_entry["job_count"] += 1
        role_entry["status_counts"][status] = role_entry["status_counts"].get(status, 0) + 1
    for key, role_entry in role_map.items():
        language, movie_id, character = key
        for mode in SUPPORTED_MODES:
            mode_rows = [
                row
                for row in rows
                if normalize_ws(row.get("language")) == language
                and normalize_ws(row.get("movie_id")) == movie_id
                and normalize_ws(row.get("character")) == character
                and normalize_ws(row.get("mode")) == mode
                and normalize_ws(row.get("status")) in {"finished", "skipped_finished"}
            ]
            if not mode_rows:
                continue
            role_entry["mode_scores"][mode] = {
                "episode_count": len(mode_rows),
                "avg_score": average([row.get("score") for row in mode_rows]),
                "avg_episode_path_compatibility": average([row.get("episode_path_compatibility") for row in mode_rows]),
            }

    return {
        "status_counts": status_counts,
        "aggregates_by_mode": by_mode,
        "role_summaries": sorted(role_map.values(), key=lambda row: (row["language"], row["movie_id"], row["character"])),
    }


def write_status(
    output_root: Path,
    *,
    started_at: str,
    total_roles: int,
    roles_with_episodes: int,
    total_jobs: int,
    rows: Sequence[Dict[str, Any]],
) -> None:
    summary = summarize_rows(rows)
    payload = {
        "updated_at": now_ts(),
        "started_at": started_at,
        "total_roles": total_roles,
        "roles_with_episodes": roles_with_episodes,
        "roles_without_episodes": total_roles - roles_with_episodes,
        "total_jobs": total_jobs,
        "completed_jobs": len(rows),
        "pending_jobs": max(total_jobs - len(rows), 0),
        "status_counts": summary["status_counts"],
        "aggregates_by_mode": summary["aggregates_by_mode"],
        "role_summaries": summary["role_summaries"],
        "rows": list(rows),
    }
    dump_json(output_root / "status.json", payload)


def main() -> None:
    args = parse_args()
    roles = filter_roles(
        load_roles(args.roles_json),
        languages=args.language,
        movie_ids=args.movie_id,
        characters=args.character,
    )
    started_at = now_ts()
    args.output_root.mkdir(parents=True, exist_ok=True)

    planned_rows: List[Dict[str, Any]] = []
    jobs: List[Dict[str, str]] = []
    for role in roles:
        episode_ids = gather_episode_ids(
            args.stage_root,
            language=role["language"],
            movie_id=role["movie_id"],
            character=role["character"],
        )
        if not episode_ids:
            for mode in args.modes:
                planned_rows.append(
                    {
                        "language": role["language"],
                        "movie_id": role["movie_id"],
                        "character": role["character"],
                        "episode_id": "",
                        "mode": mode,
                        "status": "no_episodes",
                    }
                )
            continue
        for episode_id in episode_ids:
            for mode in args.modes:
                jobs.append(
                    {
                        "language": role["language"],
                        "movie_id": role["movie_id"],
                        "character": role["character"],
                        "episode_id": episode_id,
                        "mode": mode,
                    }
                )

    rows: List[Dict[str, Any]] = list(planned_rows)
    roles_with_episodes = len({(job["language"], job["movie_id"], job["character"]) for job in jobs})
    write_status(
        args.output_root,
        started_at=started_at,
        total_roles=len(roles),
        roles_with_episodes=roles_with_episodes,
        total_jobs=len(planned_rows) + len(jobs),
        rows=rows,
    )

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {
            pool.submit(
                run_job,
                python_bin=args.python_bin,
                stage_root=args.stage_root,
                language=job["language"],
                movie_id=job["movie_id"],
                character=job["character"],
                episode_id=job["episode_id"],
                mode=job["mode"],
                output_root=args.output_root,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                fallback_base_url=args.fallback_base_url,
                fallback_api_key=args.fallback_api_key,
                fallback_model=args.fallback_model,
                top_k=args.top_k,
                sample_max_attempts=args.sample_max_attempts,
                retry_delay_sec=args.retry_delay_sec,
                skip_finished=args.skip_finished,
            ): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                failures += 1
                row = {
                    **job,
                    "status": "failed",
                    "error": str(exc),
                }
            rows.append(row)
            rows.sort(
                key=lambda item: (
                    normalize_ws(item.get("language")),
                    normalize_ws(item.get("movie_id")),
                    normalize_ws(item.get("character")),
                    normalize_ws(item.get("mode")),
                    normalize_ws(item.get("episode_id")),
                )
            )
            write_status(
                args.output_root,
                started_at=started_at,
                total_roles=len(roles),
                roles_with_episodes=roles_with_episodes,
                total_jobs=len(planned_rows) + len(jobs),
                rows=rows,
            )
            print(
                json.dumps(
                    {
                        "stage": "job_complete",
                        "language": row.get("language"),
                        "movie_id": row.get("movie_id"),
                        "character": row.get("character"),
                        "mode": row.get("mode"),
                        "episode_id": row.get("episode_id"),
                        "status": row.get("status"),
                        "score": row.get("score"),
                        "episode_path_compatibility": row.get("episode_path_compatibility"),
                        "elapsed_sec": row.get("elapsed_sec"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = summarize_rows(rows)
    dump_json(
        args.output_root / "summary.json",
        {
            "started_at": started_at,
            "finished_at": now_ts(),
            "stage_root": str(args.stage_root),
            "roles_json": str(args.roles_json),
            "output_root": str(args.output_root),
            "base_url": args.base_url,
            "fallback_base_url": args.fallback_base_url,
            "model": args.model,
            "fallback_model": args.fallback_model or args.model,
            "workers": args.workers,
            "modes": args.modes,
            "total_roles": len(roles),
            "roles_with_episodes": roles_with_episodes,
            "roles_without_episodes": len(roles) - roles_with_episodes,
            "planned_job_count": len(planned_rows) + len(jobs),
            "summary": summary,
        },
    )
    if failures or any(normalize_ws(row.get("status")) == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
