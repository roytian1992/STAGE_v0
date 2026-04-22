#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


SUPPORTED_MODES = ["persona_only", "full_memory_all_in", "bm25_topk", "embedding_topk"]
DEFAULT_STAGE_ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0")
DEFAULT_ROLES_JSON = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/task3_single_turn_bundle_manifest40_20260420_run1_roles.json")
DEFAULT_PYTHON_BIN = "/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python"
DEFAULT_BASE_URL = "http://localhost:8001/v1"
DEFAULT_API_KEY = "token-abc123"
DEFAULT_MODEL = "Qwen3-235B"
DEFAULT_FALLBACK_BASE_URL = "http://localhost:8002/v1"
DEFAULT_FALLBACK_API_KEY = "token-abc123"
DEFAULT_FALLBACK_MODEL = "Qwen3-235B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manifest40 Task3 single-turn evaluation in batch across focal roles and memory modes."
    )
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--roles-json", type=Path, default=DEFAULT_ROLES_JSON)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--fallback-base-url", default=DEFAULT_FALLBACK_BASE_URL)
    parser.add_argument("--fallback-api-key", default=DEFAULT_FALLBACK_API_KEY)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=SUPPORTED_MODES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=2.0)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--skip-finished", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_character_name(text: Any) -> str:
    return " ".join(str(text or "").replace("_", " ").split()).strip().lower()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def load_roles(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [row for row in data["rows"] if isinstance(row, dict)]
    raise ValueError(f"Unsupported roles json format: {path}")


def collect_tasks(stage_root: Path, roles: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    roles_with_instances = 0
    roles_without_instances = 0
    per_language_role_counts: Dict[str, int] = {}
    per_language_instance_counts: Dict[str, int] = {}
    zero_instance_roles: List[Dict[str, str]] = []

    for role in roles:
        language = str(role["language"])
        movie_id = str(role["movie_id"])
        character = str(role["character"])
        per_language_role_counts[language] = per_language_role_counts.get(language, 0) + 1

        single_turn_path = stage_root / language / movie_id / "task_3_in_script_character_role_play_single_turn.json"
        data = load_json(single_turn_path)
        target_name = normalize_character_name(character)
        instance_ids = [
            str(row["instance_id"])
            for row in data.get("instances", [])
            if isinstance(row, dict)
            and normalize_character_name(row.get("character")) == target_name
        ]
        if instance_ids:
            roles_with_instances += 1
        else:
            roles_without_instances += 1
            zero_instance_roles.append(
                {"language": language, "movie_id": movie_id, "character": character}
            )
        per_language_instance_counts[language] = per_language_instance_counts.get(language, 0) + len(instance_ids)
        for instance_id in instance_ids:
            for mode in modes:
                tasks.append(
                    {
                        "language": language,
                        "movie_id": movie_id,
                        "character": character,
                        "instance_id": instance_id,
                        "mode": mode,
                    }
                )

    return {
        "tasks": tasks,
        "stats": {
            "role_count": len(roles),
            "roles_with_instances": roles_with_instances,
            "roles_without_instances": roles_without_instances,
            "per_language_role_counts": per_language_role_counts,
            "per_language_instance_counts": per_language_instance_counts,
            "instance_count": sum(per_language_instance_counts.values()),
            "mode_count": len(modes),
            "task_count": len(tasks),
            "zero_instance_roles": zero_instance_roles,
        },
    }


def summarize_rows(rows: List[Dict[str, Any]], modes: List[str]) -> Dict[str, Any]:
    aggregates: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        mode_rows = [row for row in rows if row.get("mode") == mode]
        completed = [row for row in mode_rows if int(row.get("returncode", 1)) == 0]
        failed = [row for row in mode_rows if int(row.get("returncode", 1)) != 0]
        scores = [float(row["single_turn_score"]) for row in completed if isinstance(row.get("single_turn_score"), (int, float))]
        support_hits = [
            float((row.get("retrieval_diagnostics") or {}).get("support_hit_at_k"))
            for row in completed
            if isinstance((row.get("retrieval_diagnostics") or {}).get("support_hit_at_k"), (int, float))
        ]
        support_recalls = [
            float((row.get("retrieval_diagnostics") or {}).get("support_recall_at_k"))
            for row in completed
            if isinstance((row.get("retrieval_diagnostics") or {}).get("support_recall_at_k"), (int, float))
        ]
        aggregates[mode] = {
            "completed": len(completed),
            "failed": len(failed),
            "avg_single_turn_score": round(sum(scores) / len(scores), 4) if scores else None,
            "avg_support_hit_at_k": round(sum(support_hits) / len(support_hits), 4) if support_hits else None,
            "avg_support_recall_at_k": round(sum(support_recalls) / len(support_recalls), 4) if support_recalls else None,
        }
    return aggregates


def output_path_for(output_root: Path, task: Dict[str, str]) -> Path:
    return (
        output_root
        / "results"
        / str(task["mode"])
        / str(task["language"])
        / str(task["movie_id"])
        / safe_name(str(task["character"]))
        / f"{task['instance_id']}.json"
    )


def run_one(
    *,
    task: Dict[str, str],
    args: argparse.Namespace,
    tools_dir: Path,
) -> Dict[str, Any]:
    output_path = output_path_for(args.output_root, task)
    if args.skip_finished and output_path.exists():
        try:
            existing = load_json(output_path)
            return {
                **task,
                "status": "skipped_finished",
                "output_path": str(output_path),
                "returncode": 0,
                "attempt_count": 0,
                "elapsed_sec": 0.0,
                "single_turn_score": existing.get("single_turn_score"),
                "retrieval_diagnostics": existing.get("retrieval_diagnostics"),
            }
        except Exception:
            pass

    cmd = [
        args.python_bin,
        str(tools_dir / "run_task3_single_turn_eval.py"),
        "--stage-root",
        str(args.stage_root),
        "--language",
        str(task["language"]),
        "--movie-id",
        str(task["movie_id"]),
        "--instance-id",
        str(task["instance_id"]),
        "--output-path",
        str(output_path),
        "--memory-mode",
        str(task["mode"]),
        "--top-k",
        str(args.top_k),
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--judge-base-url",
        args.judge_base_url or args.base_url,
        "--judge-api-key",
        args.judge_api_key or args.api_key,
        "--judge-model",
        args.judge_model or args.model,
    ]
    env = os.environ.copy()
    if args.fallback_base_url:
        env["TASK3_FALLBACK_BASE_URL"] = args.fallback_base_url
        env["TASK3_FALLBACK_API_KEY"] = args.fallback_api_key or args.api_key
        env["TASK3_FALLBACK_MODEL"] = args.fallback_model or args.model

    attempts: List[Dict[str, Any]] = []
    last_proc: subprocess.CompletedProcess[str] | None = None
    total_elapsed = 0.0
    for attempt in range(1, max(1, args.sample_max_attempts) + 1):
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(tools_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        elapsed = round(time.time() - started, 3)
        total_elapsed += elapsed
        attempts.append(
            {
                "attempt": attempt,
                "returncode": proc.returncode,
                "elapsed_sec": elapsed,
                "stdout_tail": proc.stdout[-3000:],
                "stderr_tail": proc.stderr[-3000:],
            }
        )
        last_proc = proc
        if proc.returncode == 0 and output_path.exists():
            break
        if attempt < max(1, args.sample_max_attempts):
            time.sleep(max(0.0, args.retry_delay_sec))

    result: Dict[str, Any] = {
        **task,
        "status": "finished" if last_proc and last_proc.returncode == 0 and output_path.exists() else "failed",
        "output_path": str(output_path),
        "returncode": None if last_proc is None else last_proc.returncode,
        "attempt_count": len(attempts),
        "elapsed_sec": round(total_elapsed, 3),
        "attempts": attempts,
        "stdout_tail": "" if not attempts else attempts[-1]["stdout_tail"],
        "stderr_tail": "" if not attempts else attempts[-1]["stderr_tail"],
    }
    if result["returncode"] == 0 and output_path.exists():
        data = load_json(output_path)
        result["single_turn_score"] = data.get("single_turn_score")
        result["retrieval_diagnostics"] = data.get("retrieval_diagnostics")
    return result


def build_status_payload(
    *,
    args: argparse.Namespace,
    discovery: Dict[str, Any],
    rows: List[Dict[str, Any]],
    total_tasks: int,
    started_at: str,
    phase: str,
) -> Dict[str, Any]:
    completed = len(rows)
    failed = sum(1 for row in rows if int(row.get("returncode", 1)) != 0)
    skipped_finished = sum(1 for row in rows if str(row.get("status") or "") == "skipped_finished")
    return {
        "updated_at": now_ts(),
        "phase": phase,
        "started_at": started_at,
        "stage_root": str(args.stage_root),
        "roles_json": str(args.roles_json),
        "output_root": str(args.output_root),
        "base_url": args.base_url,
        "model": args.model,
        "judge_base_url": args.judge_base_url or args.base_url,
        "judge_model": args.judge_model or args.model,
        "fallback_base_url": args.fallback_base_url,
        "fallback_model": args.fallback_model,
        "workers": args.workers,
        "top_k": args.top_k,
        "modes": args.modes,
        "sample_max_attempts": args.sample_max_attempts,
        "retry_delay_sec": args.retry_delay_sec,
        "skip_finished": args.skip_finished,
        "discovery": discovery["stats"],
        "progress": {
            "completed_tasks": completed,
            "pending_tasks": max(total_tasks - completed, 0),
            "failed_tasks": failed,
            "skipped_finished_tasks": skipped_finished,
            "success_tasks": completed - failed,
            "total_tasks": total_tasks,
        },
        "aggregates": summarize_rows(rows, args.modes),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    tools_dir = Path(__file__).resolve().parent
    roles = load_roles(args.roles_json)
    discovery = collect_tasks(args.stage_root, roles, args.modes)
    tasks = discovery["tasks"]
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "manifest40_single_turn_eval_status.json"
    started_at = now_ts()
    rows: List[Dict[str, Any]] = []
    dump_json(
        status_path,
        build_status_payload(
            args=args,
            discovery=discovery,
            rows=rows,
            total_tasks=len(tasks),
            started_at=started_at,
            phase="running",
        ),
    )

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(run_one, task=task, args=args, tools_dir=tools_dir) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                rows.sort(key=lambda item: (str(item["mode"]), str(item["language"]), str(item["movie_id"]), str(item["instance_id"])))
                dump_json(
                    status_path,
                    build_status_payload(
                        args=args,
                        discovery=discovery,
                        rows=rows,
                        total_tasks=len(tasks),
                        started_at=started_at,
                        phase="running",
                    ),
                )
                print(
                    json.dumps(
                        {
                            "stage": "task_complete",
                            "mode": row["mode"],
                            "language": row["language"],
                            "movie_id": row["movie_id"],
                            "instance_id": row["instance_id"],
                            "returncode": row["returncode"],
                            "attempt_count": row["attempt_count"],
                            "single_turn_score": row.get("single_turn_score"),
                            "status": row["status"],
                            "elapsed_sec": row["elapsed_sec"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        dump_json(
            status_path,
            build_status_payload(
                args=args,
                discovery=discovery,
                rows=rows,
                total_tasks=len(tasks),
                started_at=started_at,
                phase="completed",
            ),
        )
        print(
            json.dumps(
                {
                    "stage": "all_complete",
                    "status_path": str(status_path),
                    "completed_tasks": len(rows),
                    "total_tasks": len(tasks),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
