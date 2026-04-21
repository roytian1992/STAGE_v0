#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

from task3_llm_fallback import build_clients, build_routes
from run_task3_multi_turn_episode_eval import (
    CORE_METRICS,
    build_core_metric_messages,
    call_text,
    extract_json_object,
    sanitize_metric_result,
)
from run_task3_multi_turn_batch_eval import summarize_output, summarize_rows
from task3_runtime_loader import load_multi_turn_episode, normalize_ws


DEFAULT_JUDGE_BASE_URL = "http://localhost:8001/v1"
DEFAULT_JUDGE_API_KEY = "token-abc123"
DEFAULT_JUDGE_MODEL = "Qwen3-235B"
DEFAULT_STAGE_ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0")
DEFAULT_PYTHON_ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tighten and rejudge Task3 multi-turn cross-turn consistency on existing result files."
    )
    parser.add_argument(
        "--output-roots",
        nargs="+",
        required=True,
        help="Existing multi-turn evaluation output roots containing runs/ and status.json",
    )
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=DEFAULT_JUDGE_API_KEY)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--fallback-base-url", default="")
    parser.add_argument("--fallback-api-key", default="")
    parser.add_argument("--fallback-model", default="")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=520)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--sample-max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=2.0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def list_result_files(output_root: Path) -> List[Path]:
    return sorted((output_root / "runs").glob("**/*.json"))


def rebuild_rollout_turns(report: Dict[str, Any], episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    turn_by_index = {int(turn["turn_index"]): turn for turn in episode.get("turns", [])}
    rebuilt: List[Dict[str, Any]] = []
    for row in report.get("rollout_turns", []) or []:
        try:
            turn_index = int(row.get("turn_index"))
        except Exception:
            continue
        turn = turn_by_index.get(turn_index)
        if turn is None:
            raise KeyError(f"turn_index not found in episode: {turn_index}")
        rebuilt.append(
            {
                "turn": turn,
                "resolved_history": list(row.get("resolved_history", []) or []),
                "current_user_turn": normalize_ws(row.get("current_user_turn")),
                "response": normalize_ws(row.get("response")),
            }
        )
    return rebuilt


def rejudge_one(
    *,
    path: Path,
    stage_root: Path,
    clients: List[Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    report = load_json(path)
    _movie_dir, episode, _role_asset = load_multi_turn_episode(
        stage_root=stage_root,
        language=normalize_ws(report.get("language")),
        movie_id=normalize_ws(report.get("movie_id")),
        episode_instance_id=normalize_ws(report.get("episode_instance_id") or report.get("episode_id")),
    )
    rollout_turns = rebuild_rollout_turns(report, episode)
    messages = build_core_metric_messages(
        "cross_turn_consistency",
        episode=episode,
        rollout_turns=rollout_turns,
        runtime_loader=SimpleNamespace(persona_card={}),
        memory_mode=normalize_ws(report.get("memory_mode")),
    )

    last_error: Exception | None = None
    for attempt in range(1, max(1, args.sample_max_attempts) + 1):
        try:
            text, usage, latency_ms, route_name = call_text(
                clients,
                messages=messages,
                temperature=args.judge_temperature,
                max_tokens=args.judge_max_tokens,
                max_retries=args.max_retries,
            )
            raw = extract_json_object(text)
            metric = sanitize_metric_result("cross_turn_consistency", raw)
            metric["latency_ms"] = latency_ms
            metric["route_name"] = route_name
            metric["usage"] = usage
            old_score = ((report.get("core_metrics") or {}).get("cross_turn_consistency") or {}).get("score")
            old_core_multi_turn_score = report.get("core_multi_turn_score")
            report.setdefault("core_metrics", {})["cross_turn_consistency"] = metric
            report["core_multi_turn_score"] = round(
                mean(report["core_metrics"][name]["score"] for name in CORE_METRICS),
                4,
            )
            report.setdefault("rejudge_metadata", {})
            report["rejudge_metadata"]["old_cross_turn_consistency_score"] = old_score
            report["rejudge_metadata"]["new_cross_turn_consistency_score"] = metric["score"]
            report["rejudge_metadata"]["old_core_multi_turn_score"] = old_core_multi_turn_score
            report["rejudge_metadata"]["new_core_multi_turn_score"] = report["core_multi_turn_score"]
            report["rejudge_metadata"]["cross_turn_consistency_tightened_at"] = now_ts()
            report["rejudge_metadata"]["cross_turn_consistency_judge_model"] = args.judge_model
            report["rejudge_metadata"]["cross_turn_consistency_judge_base_url"] = args.judge_base_url
            dump_json(path, report)
            return {
                "output_path": str(path),
                "old_score": old_score,
                "new_score": metric["score"],
                "core_multi_turn_score": report["core_multi_turn_score"],
                "status": "finished",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max(1, args.sample_max_attempts):
                time.sleep(max(0.0, args.retry_delay_sec))
    return {
        "output_path": str(path),
        "status": "failed",
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def refresh_status(output_root: Path) -> None:
    status_path = output_root / "status.json"
    payload = load_json(status_path)
    rows = list(payload.get("rows", []))
    updated_rows: List[Dict[str, Any]] = []
    for row in rows:
        out_path = normalize_ws(row.get("output_path"))
        if out_path and Path(out_path).exists() and normalize_ws(row.get("status")) in {"finished", "skipped_finished"}:
            summary = summarize_output(Path(out_path))
            row = {
                **row,
                "score": summary.get("score"),
                "episode_path_compatibility": summary.get("episode_path_compatibility"),
                "followup_compatibility": summary.get("followup_compatibility"),
                "retrieval_diagnostics": summary.get("retrieval_diagnostics"),
            }
        updated_rows.append(row)
    payload["rows"] = updated_rows
    payload["updated_at"] = now_ts()
    summary = summarize_rows(updated_rows)
    payload["status_counts"] = summary["status_counts"]
    payload["aggregates_by_mode"] = summary["aggregates_by_mode"]
    payload["role_summaries"] = summary["role_summaries"]
    payload.setdefault("rejudge_metadata", {})
    payload["rejudge_metadata"]["cross_turn_consistency_tightened_at"] = payload["updated_at"]
    dump_json(status_path, payload)


def main() -> None:
    args = parse_args()
    os.environ["TASK3_FALLBACK_BASE_URL"] = args.fallback_base_url
    os.environ["TASK3_FALLBACK_API_KEY"] = args.fallback_api_key
    os.environ["TASK3_FALLBACK_MODEL"] = args.fallback_model
    routes = build_routes(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        model=args.judge_model,
    )
    clients = build_clients(routes, timeout=180)

    all_files: List[Path] = []
    for root_text in args.output_roots:
        root = Path(root_text)
        all_files.extend(list_result_files(root))
    all_files = sorted(dict.fromkeys(all_files))
    print(
        json.dumps(
            {
                "stage": "discovered",
                "file_count": len(all_files),
                "output_roots": args.output_roots,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {
            pool.submit(rejudge_one, path=path, stage_root=args.stage_root, clients=clients, args=args): path
            for path in all_files
        }
        for future in as_completed(future_map):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    for root_text in args.output_roots:
        refresh_status(Path(root_text))

    finished = sum(1 for row in results if row.get("status") == "finished")
    failed = sum(1 for row in results if row.get("status") != "finished")
    print(
        json.dumps(
            {
                "stage": "complete",
                "finished": finished,
                "failed": failed,
                "output_roots": args.output_roots,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
