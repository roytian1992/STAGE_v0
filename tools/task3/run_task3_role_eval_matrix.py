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
DEFAULT_PYTHON_BIN = "/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full Task3 role evaluation matrix with bounded concurrency."
    )
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--language", choices=["Chinese", "English"], required=True)
    parser.add_argument("--movie-id", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fallback-base-url", default="")
    parser.add_argument("--fallback-api-key", default="")
    parser.add_argument("--fallback-model", default="")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=SUPPORTED_MODES)
    parser.add_argument("--mode-concurrency", type=int, default=2)
    parser.add_argument("--single-turn-concurrency", type=int, default=1)
    parser.add_argument("--multi-turn-concurrency", type=int, default=1)
    parser.add_argument("--sample-max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=2.0)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def gather_samples(stage_root: Path, language: str, movie_id: str, character: str) -> tuple[List[str], List[str]]:
    movie_dir = stage_root / language / movie_id
    single = load_json(movie_dir / "task_3_in_script_character_role_play_single_turn.json")
    multi = load_json(movie_dir / "task_3_in_script_character_role_play_multi_turn.json")
    single_ids = [row["instance_id"] for row in single.get("instances", []) if row.get("character") == character]
    multi_ids = [row["episode_id"] for row in multi.get("episodes", []) if row.get("character") == character]
    return single_ids, multi_ids


def summarize_output(kind: str, output_path: Path) -> Dict[str, Any]:
    data = load_json(output_path)
    if kind == "single_turn":
        return {
            "score": data.get("single_turn_score"),
            "retrieval_diagnostics": data.get("retrieval_diagnostics"),
        }
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


def run_one(
    *,
    python_bin: str,
    tools_dir: Path,
    stage_root: Path,
    language: str,
    movie_id: str,
    base_url: str,
    api_key: str,
    model: str,
    fallback_base_url: str,
    fallback_api_key: str,
    fallback_model: str,
    top_k: int,
    kind: str,
    sample_id: str,
    mode: str,
    output_root: Path,
    sample_max_attempts: int,
    retry_delay_sec: float,
) -> Dict[str, Any]:
    if kind == "single_turn":
        script = tools_dir / "run_task3_single_turn_eval.py"
        output_path = output_root / kind / mode / f"{sample_id}.json"
        cmd = [
            python_bin,
            str(script),
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
            "--instance-id",
            sample_id,
            "--output-path",
            str(output_path),
        ]
    else:
        script = tools_dir / "run_task3_multi_turn_episode_eval.py"
        output_path = output_root / kind / mode / f"{sample_id}.json"
        cmd = [
            python_bin,
            str(script),
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
            sample_id,
            "--output-path",
            str(output_path),
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if fallback_base_url:
        env["TASK3_FALLBACK_BASE_URL"] = fallback_base_url
        env["TASK3_FALLBACK_API_KEY"] = fallback_api_key or api_key
        env["TASK3_FALLBACK_MODEL"] = fallback_model or model

    attempts: List[Dict[str, Any]] = []
    last_row: Dict[str, Any] | None = None
    for attempt in range(1, max(1, sample_max_attempts) + 1):
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(tools_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        row = {
            "attempt": attempt,
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 3),
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
        }
        attempts.append(row)
        last_row = row
        if proc.returncode == 0 and output_path.exists():
            break
        if attempt < max(1, sample_max_attempts):
            time.sleep(max(0.0, retry_delay_sec))

    assert last_row is not None
    result = {
        "kind": kind,
        "sample_id": sample_id,
        "mode": mode,
        "output_path": str(output_path),
        "returncode": last_row["returncode"],
        "elapsed_sec": round(sum(float(item["elapsed_sec"]) for item in attempts), 3),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "stdout_tail": last_row["stdout_tail"],
        "stderr_tail": last_row["stderr_tail"],
    }
    if last_row["returncode"] == 0 and output_path.exists():
        result.update(summarize_output(kind, output_path))
    return result


def run_group(
    *,
    python_bin: str,
    tools_dir: Path,
    stage_root: Path,
    language: str,
    movie_id: str,
    base_url: str,
    api_key: str,
    model: str,
    fallback_base_url: str,
    fallback_api_key: str,
    fallback_model: str,
    top_k: int,
    kind: str,
    sample_ids: List[str],
    mode: str,
    output_root: Path,
    max_workers: int,
    sample_max_attempts: int,
    retry_delay_sec: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not sample_ids:
        return rows
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_one,
                python_bin=python_bin,
                tools_dir=tools_dir,
                stage_root=stage_root,
                language=language,
                movie_id=movie_id,
                base_url=base_url,
                api_key=api_key,
                model=model,
                fallback_base_url=fallback_base_url,
                fallback_api_key=fallback_api_key,
                fallback_model=fallback_model,
                top_k=top_k,
                kind=kind,
                sample_id=sample_id,
                mode=mode,
                output_root=output_root,
                sample_max_attempts=sample_max_attempts,
                retry_delay_sec=retry_delay_sec,
            )
            for sample_id in sample_ids
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "stage": "run_complete",
                        "kind": row["kind"],
                        "mode": row["mode"],
                        "sample_id": row["sample_id"],
                        "returncode": row["returncode"],
                        "attempt_count": row["attempt_count"],
                        "score": row.get("score"),
                        "episode_path_compatibility": row.get("episode_path_compatibility"),
                        "followup_compatibility": row.get("followup_compatibility"),
                        "elapsed_sec": row["elapsed_sec"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if row["returncode"] != 0:
                raise RuntimeError(
                    f"task failed: kind={row['kind']} mode={row['mode']} sample_id={row['sample_id']}"
                )
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def average(values: List[float | int | None]) -> float | None:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 4)


def main() -> None:
    args = parse_args()
    tools_dir = Path(__file__).resolve().parent
    single_ids, multi_ids = gather_samples(args.stage_root, args.language, args.movie_id, args.character)
    summary = {
        "stage_root": str(args.stage_root),
        "language": args.language,
        "movie_id": args.movie_id,
        "character": args.character,
        "model": args.model,
        "base_url": args.base_url,
        "fallback_base_url": args.fallback_base_url,
        "fallback_model": args.fallback_model or args.model,
        "top_k": args.top_k,
        "modes": args.modes,
        "single_turn_instance_ids": single_ids,
        "multi_turn_episode_ids": multi_ids,
        "mode_concurrency": args.mode_concurrency,
        "single_turn_concurrency": args.single_turn_concurrency,
        "multi_turn_concurrency": args.multi_turn_concurrency,
        "sample_max_attempts": args.sample_max_attempts,
        "retry_delay_sec": args.retry_delay_sec,
        "concurrency_strategy": {
            "mode_scheduling": f"parallel x{args.mode_concurrency}",
            "single_turn": f"parallel x{args.single_turn_concurrency}",
            "multi_turn": f"parallel x{args.multi_turn_concurrency}",
            "rationale": "Run modes in parallel, retry sample-level failures automatically, and let child evaluators fall back from 8001 to 8002 on the same model when needed.",
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": [],
    }

    def run_mode(mode: str) -> List[Dict[str, Any]]:
        mode_rows: List[Dict[str, Any]] = []
        mode_rows.extend(
            run_group(
                python_bin=args.python_bin,
                tools_dir=tools_dir,
                stage_root=args.stage_root,
                language=args.language,
                movie_id=args.movie_id,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                fallback_base_url=args.fallback_base_url,
                fallback_api_key=args.fallback_api_key,
                fallback_model=args.fallback_model,
                top_k=args.top_k,
                kind="single_turn",
                sample_ids=single_ids,
                mode=mode,
                output_root=args.output_root,
                max_workers=args.single_turn_concurrency,
                sample_max_attempts=args.sample_max_attempts,
                retry_delay_sec=args.retry_delay_sec,
            )
        )
        mode_rows.extend(
            run_group(
                python_bin=args.python_bin,
                tools_dir=tools_dir,
                stage_root=args.stage_root,
                language=args.language,
                movie_id=args.movie_id,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                fallback_base_url=args.fallback_base_url,
                fallback_api_key=args.fallback_api_key,
                fallback_model=args.fallback_model,
                top_k=args.top_k,
                kind="multi_turn",
                sample_ids=multi_ids,
                mode=mode,
                output_root=args.output_root,
                max_workers=args.multi_turn_concurrency,
                sample_max_attempts=args.sample_max_attempts,
                retry_delay_sec=args.retry_delay_sec,
            )
        )
        return mode_rows

    try:
        with ThreadPoolExecutor(max_workers=args.mode_concurrency) as executor:
            future_to_mode = {executor.submit(run_mode, mode): mode for mode in args.modes}
            for future in as_completed(future_to_mode):
                mode = future_to_mode[future]
                mode_rows = future.result()
                summary["runs"].extend(mode_rows)
                print(
                    json.dumps(
                        {
                            "stage": "mode_complete",
                            "mode": mode,
                            "run_count": len(mode_rows),
                            "mode_elapsed_sec": round(sum(float(row["elapsed_sec"]) for row in mode_rows), 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    aggregates: Dict[str, Dict[str, Any]] = {}
    for mode in args.modes:
        mode_rows = [row for row in summary["runs"] if row["mode"] == mode]
        single_rows = [row for row in mode_rows if row["kind"] == "single_turn"]
        multi_rows = [row for row in mode_rows if row["kind"] == "multi_turn"]
        aggregates[mode] = {
            "single_turn_count": len(single_rows),
            "single_turn_avg_score": average([row.get("score") for row in single_rows]),
            "multi_turn_count": len(multi_rows),
            "multi_turn_avg_score": average([row.get("score") for row in multi_rows]),
            "multi_turn_avg_episode_path_compatibility": average(
                [row.get("episode_path_compatibility") for row in multi_rows]
            ),
            "multi_turn_avg_followup_compatibility": average(
                [row.get("followup_compatibility") for row in multi_rows]
            ),
            "mode_elapsed_sec": round(sum(float(row["elapsed_sec"]) for row in mode_rows), 3),
        }
    summary["aggregates"] = aggregates
    dump_json(args.output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "stage": "all_complete",
                "summary_path": str(args.output_root / "summary.json"),
                "aggregates": aggregates,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
