#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, LLMClient, detect_language, load_scenes
from metrics import evaluate_v5, load_target_characters
from recall import (
    HIGH_MILESTONE_SCORE,
    arc_prompt,
    build_role_scene_cards,
    build_segments,
    card_feature_tags,
    clean_text,
    flush_partial_outputs,
    llm_json,
    milestone_priority,
    normalize_arc_payload,
    novelty_tuple,
    pairwise_better_than,
    pairwise_milestone_judge_prompt,
    prompt_messages,
    refine_milestones_with_pairwise,
    render_timeline_nodes,
    scene_by_order_map,
    same_mini_arc,
    scene_role_prompt,
    selection_penalty,
    should_skip_for_redundancy,
    stable_id,
    summarize_segment_prompt,
    temporal_bucket,
)


def dynamic_node_budget(cards: Sequence[Dict[str, Any]]) -> int:
    if not cards:
        return 20
    high = sum(1 for c in cards if int(c.get("milestone_score", 0) or 0) >= 4)
    very_high = sum(1 for c in cards if int(c.get("milestone_score", 0) or 0) >= 5)
    medium = sum(1 for c in cards if int(c.get("milestone_score", 0) or 0) >= 3)
    active = sum(1 for c in cards if c.get("scene_role") in {"foreground", "active"})
    budget = 20
    if len(cards) >= 18:
        budget += 1
    if len(cards) >= 28:
        budget += 1
    if len(cards) >= 40:
        budget += 1
    if medium >= 10:
        budget += 1
    if high >= 8:
        budget += 1
    if very_high >= 5:
        budget += 1
    if active >= 16:
        budget += 1
    return max(20, min(24, budget))


def coverage_window_size(max_scene_order: int) -> int:
    if max_scene_order >= 54:
        return 3
    if max_scene_order >= 24:
        return 2
    return 1


def bucket_targets(cards: Sequence[Dict[str, Any]], max_nodes: int, max_scene_order: int) -> Dict[str, int]:
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for card in cards:
        by_bucket[temporal_bucket(int(card["scene_order"]), max_scene_order)].append(card)

    non_empty = [bucket for bucket in ("early", "middle", "late") if by_bucket.get(bucket)]
    targets = {bucket: 0 for bucket in ("early", "middle", "late")}
    if not non_empty:
        return targets

    min_per_bucket = 3 if max_nodes >= 12 and len(non_empty) == 3 else 2
    used = 0
    for bucket in non_empty:
        quota = min(min_per_bucket, len(by_bucket[bucket]))
        targets[bucket] = quota
        used += quota

    remaining = max(0, max_nodes - used)
    weights = {
        bucket: (
            len(by_bucket[bucket]),
            sum(max(0, int(x.get("milestone_score", 0) or 0) - 1) for x in by_bucket[bucket]),
        )
        for bucket in non_empty
    }
    while remaining > 0:
        best_bucket = None
        best_key = None
        for bucket in non_empty:
            if targets[bucket] >= len(by_bucket[bucket]):
                continue
            key = (weights[bucket][1], weights[bucket][0], -targets[bucket], bucket == "late")
            if best_bucket is None or key > best_key:
                best_bucket = bucket
                best_key = key
        if best_bucket is None:
            break
        targets[best_bucket] += 1
        remaining -= 1
    return targets


def candidate_key_v66(
    row: Dict[str, Any],
    chosen: Sequence[Dict[str, Any]],
    max_scene_order: int,
    targets: Dict[str, int],
    bucket_counts: Counter,
) -> Tuple[int, int, int, int, int, int, int, int, int, int, int]:
    bucket = temporal_bucket(int(row["scene_order"]), max_scene_order)
    deficit = max(0, int(targets.get(bucket, 0)) - int(bucket_counts.get(bucket, 0)))
    score = int(row.get("milestone_score", 0) or 0)
    novelty = novelty_tuple(row, chosen)
    penalty = selection_penalty(row, chosen, max_scene_order)
    priority = milestone_priority(row)
    late_bonus = 1 if bucket == "late" and deficit > 0 else 0
    coverage_bonus = 1 if deficit > 0 and score >= 4 else 0
    return (
        min(3, deficit),
        coverage_bonus,
        late_bonus,
        1 if score >= 5 else 0,
        1 if score >= 4 else 0,
        novelty[0],
        novelty[1],
        novelty[2],
        penalty,
        priority[0],
        -int(row.get("scene_order", 0)),
    )


def best_bucket_candidate(
    rows: Sequence[Dict[str, Any]],
    chosen: Sequence[Dict[str, Any]],
    used_scene_ids: set,
    max_scene_order: int,
    targets: Dict[str, int],
    bucket_counts: Counter,
) -> Optional[Dict[str, Any]]:
    best = None
    best_key = None
    for row in rows:
        if str(row["scene_id"]) in used_scene_ids:
            continue
        if should_skip_for_redundancy(row, chosen) and int(row.get("milestone_score", 0) or 0) < HIGH_MILESTONE_SCORE:
            continue
        key = candidate_key_v66(row, chosen, max_scene_order, targets, bucket_counts)
        if best is None or key > best_key:
            best = row
            best_key = key
    return best


def rebalance_for_story_coverage(
    chosen: Sequence[Dict[str, Any]],
    cards: Sequence[Dict[str, Any]],
    max_nodes: int,
    max_scene_order: int,
) -> List[Dict[str, Any]]:
    chosen_rows = [dict(x) for x in chosen]
    bucket_counts = Counter(temporal_bucket(int(x["scene_order"]), max_scene_order) for x in chosen_rows)
    targets = bucket_targets(cards, max_nodes, max_scene_order)
    chosen_ids = {str(x["scene_id"]) for x in chosen_rows}
    excluded = [x for x in cards if str(x["scene_id"]) not in chosen_ids]
    seen_states = set()

    for _ in range(12):
        state = tuple(sorted(str(x["scene_id"]) for x in chosen_rows))
        if state in seen_states:
            break
        seen_states.add(state)
        deficits = [b for b in ("early", "middle", "late") if bucket_counts[b] < targets.get(b, 0)]
        if not deficits:
            break
        deficit_bucket = max(deficits, key=lambda b: (targets.get(b, 0) - bucket_counts[b], b == "late"))
        candidates = [
            x
            for x in excluded
            if temporal_bucket(int(x["scene_order"]), max_scene_order) == deficit_bucket and int(x.get("milestone_score", 0) or 0) >= 4
        ]
        if not candidates:
            break
        candidate = sorted(candidates, key=lambda x: (milestone_priority(x)[0], -int(x["scene_order"])), reverse=True)[0]

        removable = []
        for row in chosen_rows:
            bucket = temporal_bucket(int(row["scene_order"]), max_scene_order)
            if bucket_counts[bucket] <= max(1, targets.get(bucket, 0)):
                continue
            removable.append(row)
        if not removable:
            removable = sorted(chosen_rows, key=lambda x: (candidate_key_v66(x, [], max_scene_order, targets, Counter()), -milestone_priority(x)[0]))
        weak = sorted(
            removable,
            key=lambda x: (
                int(x.get("milestone_score", 0) or 0),
                milestone_priority(x)[0],
                temporal_bucket(int(x["scene_order"]), max_scene_order) != "early",
                int(x["scene_order"]),
            ),
        )[0]

        if milestone_priority(candidate)[0] + 4 < milestone_priority(weak)[0]:
            break

        chosen_rows = [candidate if str(x["scene_id"]) == str(weak["scene_id"]) else x for x in chosen_rows]
        chosen_ids = {str(x["scene_id"]) for x in chosen_rows}
        excluded = [x for x in cards if str(x["scene_id"]) not in chosen_ids]
        bucket_counts = Counter(temporal_bucket(int(x["scene_order"]), max_scene_order) for x in chosen_rows)

    deduped = []
    seen = set()
    for row in sorted(chosen_rows, key=lambda x: int(x["scene_order"])):
        sid = str(row["scene_id"])
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(row)
    return deduped[:max_nodes]


def select_milestones(cards: Sequence[Dict[str, Any]], max_nodes: Optional[int] = None) -> List[Dict[str, Any]]:
    if not cards:
        return []
    if max_nodes is None:
        max_nodes = dynamic_node_budget(cards)

    max_scene_order = max(int(c["scene_order"]) for c in cards)
    window_size = coverage_window_size(max_scene_order)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for card in cards:
        by_bucket[temporal_bucket(int(card["scene_order"]), max_scene_order)].append(card)

    targets = bucket_targets(cards, max_nodes, max_scene_order)
    chosen: List[Dict[str, Any]] = []
    used_scene_ids = set()
    bucket_counts: Counter = Counter()

    start = 1
    while start <= max_scene_order and len(chosen) < max_nodes:
        end = min(max_scene_order, start + window_size - 1)
        window_rows = [
            row
            for row in cards
            if start <= int(row["scene_order"]) <= end and int(row.get("milestone_score", 0) or 0) >= 2
        ]
        if not window_rows:
            window_rows = [
                row
                for row in cards
                if start <= int(row["scene_order"]) <= end and row.get("scene_role") in {"foreground", "active", "indirect"}
            ]
        if window_rows:
            row = sorted(window_rows, key=lambda x: (milestone_priority(x)[0], -int(x["scene_order"])), reverse=True)[0]
            if str(row["scene_id"]) not in used_scene_ids:
                chosen.append(row)
                used_scene_ids.add(str(row["scene_id"]))
                bucket_counts[temporal_bucket(int(row["scene_order"]), max_scene_order)] += 1
        start = end + 1

    progress = True
    while progress and len(chosen) < max_nodes:
        progress = False
        need_buckets = [b for b in ("early", "middle", "late") if bucket_counts[b] < targets.get(b, 0)]
        need_buckets.sort(key=lambda b: (targets.get(b, 0) - bucket_counts[b], b == "late"), reverse=True)
        for bucket in need_buckets:
            row = best_bucket_candidate(by_bucket.get(bucket, []), chosen, used_scene_ids, max_scene_order, targets, bucket_counts)
            if row is None:
                continue
            chosen.append(row)
            used_scene_ids.add(str(row["scene_id"]))
            bucket_counts[temporal_bucket(int(row["scene_order"]), max_scene_order)] += 1
            progress = True
            if len(chosen) >= max_nodes:
                break

    high_priority_rows = sorted(
        [x for x in cards if int(x.get("milestone_score", 0) or 0) >= HIGH_MILESTONE_SCORE],
        key=lambda x: candidate_key_v66(x, chosen, max_scene_order, targets, bucket_counts),
        reverse=True,
    )
    for row in high_priority_rows:
        if len(chosen) >= max_nodes:
            break
        if str(row["scene_id"]) in used_scene_ids:
            continue
        if should_skip_for_redundancy(row, chosen) and int(row.get("milestone_score", 0) or 0) < 5:
            continue
        chosen.append(row)
        used_scene_ids.add(str(row["scene_id"]))
        bucket_counts[temporal_bucket(int(row["scene_order"]), max_scene_order)] += 1

    all_rows = sorted(
        cards,
        key=lambda x: candidate_key_v66(x, chosen, max_scene_order, targets, bucket_counts),
        reverse=True,
    )
    for row in all_rows:
        if len(chosen) >= max_nodes:
            break
        if str(row["scene_id"]) in used_scene_ids:
            continue
        if should_skip_for_redundancy(row, chosen):
            continue
        chosen.append(row)
        used_scene_ids.add(str(row["scene_id"]))
        bucket_counts[temporal_bucket(int(row["scene_order"]), max_scene_order)] += 1

    chosen = rebalance_for_story_coverage(chosen, cards, max_nodes, max_scene_order)
    return sorted(chosen, key=lambda x: int(x["scene_order"]))[:max_nodes]


def run_workflow_v66(movie_dir: Path, output_dir: Path) -> Dict[str, Any]:
    started = time.time()
    language = detect_language(movie_dir)
    scenes = load_scenes(movie_dir / "script.json", language)
    llm = LLMClient(DEFAULT_LLM_MODEL, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_characters = load_target_characters(movie_dir)
    timeline_payload = {
        "movie_id": movie_dir.name,
        "language": language,
        "task_name": "Story Dynamics Structuring",
        "task_version": "workflow_v66_recall_balanced",
        "focal_character_timelines": [],
        "build_summary": {},
    }
    arc_payload = {
        "movie_id": movie_dir.name,
        "language": language,
        "task_name": "Story Dynamics Structuring",
        "task_version": "workflow_v66_recall_balanced",
        "cross_scene_arcs": [],
        "build_summary": {},
    }
    role_scene_cards: Dict[str, Any] = {}
    character_segments: Dict[str, Any] = {}
    character_milestones: Dict[str, Any] = {}

    for character in selected_characters:
        print(json.dumps({"stage": "character_start", "character": character["character_name"]}, ensure_ascii=False), flush=True)
        cards = build_role_scene_cards(llm, scenes, character, language, movie_dir=movie_dir)
        role_scene_cards[character["character_name"]] = cards
        print(json.dumps({"stage": "role_scene_cards_ready", "character": character["character_name"], "scene_card_count": len(cards)}, ensure_ascii=False), flush=True)
        segments = build_segments(llm, language, character["character_name"], cards)
        character_segments[character["character_name"]] = segments
        max_nodes = dynamic_node_budget(cards)
        milestones = select_milestones(cards, max_nodes=max_nodes)
        milestones = refine_milestones_with_pairwise(
            llm,
            language,
            character["character_name"],
            milestones,
            cards,
            max_nodes=max_nodes,
            max_pairwise_checks=14,
        )
        if not milestones and cards:
            milestones = sorted(
                cards,
                key=lambda row: (milestone_priority(row)[0], -int(row.get("scene_order", 0) or 0)),
                reverse=True,
            )[:max_nodes]
        if milestones:
            max_scene_order = max(int(c["scene_order"]) for c in cards)
            milestones = rebalance_for_story_coverage(milestones, cards, max_nodes, max_scene_order)
        character_milestones[character["character_name"]] = milestones
        print(
            json.dumps(
                {
                    "stage": "milestones_ready",
                    "character": character["character_name"],
                    "max_nodes": max_nodes,
                    "milestone_count": len(milestones),
                    "scene_orders": [m["scene_order"] for m in milestones],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        timeline_summary, nodes = render_timeline_nodes(llm, language, character, milestones, scenes)
        timeline_payload["focal_character_timelines"].append(
            {
                "character_name": character["character_name"],
                "aliases": character.get("aliases", []) or [],
                "selection_reason": "provided externally via benchmark focal-role list" if language == "en" else "由 benchmark 焦点角色列表提供",
                "task3_relevance": "Target character provided by the benchmark focal-role list and rendered through a recall-balanced milestone pipeline." if language == "en" else "由 benchmark 焦点角色列表指定目标角色，并通过 recall-balanced milestone 流程进行渲染。",
                "timeline_nodes": nodes,
                "timeline_summary": timeline_summary,
            }
        )
        if nodes:
            try:
                arc_raw = llm_json(
                    llm,
                    arc_prompt(
                        language,
                        character["character_name"],
                        [
                            {
                                "timeline_node_id": n["timeline_node_id"],
                                "scene_id": n["scene_id"],
                                "scene_order": n["scene_order"],
                                "scene_title": n["scene_title"],
                                "salient_development": n["salient_development"],
                                "goal_state": n.get("goal_state"),
                                "resulting_state": n.get("resulting_state"),
                                "unresolved_issue": n.get("unresolved_issue"),
                            }
                            for n in nodes
                        ],
                    ),
                    max_tokens=1800,
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "stage": "arc_generation_failed",
                            "character": character["character_name"],
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                arc_raw = {"arcs": []}
            node_id_to_scene_id = {n["timeline_node_id"]: n["scene_id"] for n in nodes}
            for item in normalize_arc_payload(arc_raw):
                linked_ids = [clean_text(x) for x in (item.get("linked_timeline_node_ids") or []) if clean_text(x) in node_id_to_scene_id]
                linked_ids = list(dict.fromkeys(linked_ids))
                if len(linked_ids) < 2:
                    continue
                arc_payload["cross_scene_arcs"].append(
                    {
                        "arc_id": stable_id(character["character_name"], clean_text(item.get("title")), prefix="v6arc"),
                        "character_name": character["character_name"],
                        "title": clean_text(item.get("title")),
                        "arc_focus": clean_text(item.get("arc_focus")) or "mixed",
                        "linked_timeline_node_ids": linked_ids,
                        "arc_summary": clean_text(item.get("arc_summary")),
                        "start_state": clean_text(item.get("start_state")) or None,
                        "end_state": clean_text(item.get("end_state")) or None,
                        "unresolved_issue": clean_text(item.get("unresolved_issue")) or None,
                    }
                )
        flush_partial_outputs(output_dir, role_scene_cards, character_segments, character_milestones, timeline_payload, arc_payload)
        print(
            json.dumps(
                {
                    "stage": "character_complete",
                    "character": character["character_name"],
                    "node_count": len(nodes),
                    "arc_count": len([x for x in arc_payload["cross_scene_arcs"] if x["character_name"] == character["character_name"]]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    timeline_payload["build_summary"] = {
        "selected_focal_character_count": len(timeline_payload["focal_character_timelines"]),
        "timeline_node_count": sum(len(x.get("timeline_nodes", []) or []) for x in timeline_payload["focal_character_timelines"]),
        "elapsed_sec": round(time.time() - started, 2),
    }
    arc_payload["build_summary"] = {
        "selected_focal_character_count": len(timeline_payload["focal_character_timelines"]),
        "arc_count": len(arc_payload["cross_scene_arcs"]),
        "elapsed_sec": round(time.time() - started, 2),
    }

    flush_partial_outputs(output_dir, role_scene_cards, character_segments, character_milestones, timeline_payload, arc_payload)
    return {"timeline": timeline_payload, "arcs": arc_payload}


def discover_movie_dirs(benchmark_root: Path, languages: Sequence[str]) -> List[Path]:
    dirs: List[Path] = []
    for lang in languages:
        lang_dir = benchmark_root / ("English" if lang == "en" else "Chinese")
        if not lang_dir.exists():
            continue
        for movie_dir in sorted(x for x in lang_dir.iterdir() if x.is_dir()):
            if (movie_dir / "script.json").exists():
                dirs.append(movie_dir)
    return dirs


def already_done(out_dir: Path, evaluate: bool) -> bool:
    has_build = (
        (out_dir / "pred_task_1_character_timelines.json").exists()
        and (out_dir / "pred_task_1_cross_scene_arcs.json").exists()
    )
    if not has_build:
        return False
    if not evaluate:
        return True
    return (out_dir / "eval_v3.json").exists()


def run_one_movie(
    movie_dir: Path,
    output_root: Path,
    overwrite: bool,
    evaluate: bool,
    timeout_sec: int,
) -> Dict[str, object]:
    out_dir = output_root / movie_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if already_done(out_dir, evaluate=evaluate) and not overwrite:
        eval_path = out_dir / "eval_v3.json"
        eval_data = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
        return {
            "movie_id": movie_dir.name,
            "language": "zh" if movie_dir.parent.name == "Chinese" else "en",
            "status": "skipped",
            "output_dir": str(out_dir),
            "eval": eval_data,
        }
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "one",
        "--movie-dir",
        str(movie_dir),
        "--output-dir",
        str(out_dir),
    ]
    if evaluate:
        cmd.append("--evaluate")
    started = time.time()
    error_log_path = out_dir / "run_error_details.json"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        returncode = proc.returncode
        stdout_tail = "\n".join(proc.stdout.strip().splitlines()[-10:]) if proc.stdout.strip() else ""
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-14:]) if proc.stderr.strip() else ""
        status = "ok" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        proc = exc
        returncode = None
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        stdout_tail = "\n".join(str(stdout_text).strip().splitlines()[-10:]) if str(stdout_text).strip() else ""
        stderr_tail = "\n".join(str(stderr_text).strip().splitlines()[-14:]) if str(stderr_text).strip() else ""
        build_ready = already_done(out_dir, evaluate=evaluate)
        status = "ok_timeout" if build_ready else "timeout"
    elapsed = round(time.time() - started, 2)
    eval_path = out_dir / "eval_v3.json"
    eval_data = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    if status in {"failed", "timeout"}:
        error_payload = {
            "movie_id": movie_dir.name,
            "status": status,
            "elapsed_sec": elapsed,
            "returncode": returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        error_log_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "movie_id": movie_dir.name,
        "language": "zh" if movie_dir.parent.name == "Chinese" else "en",
        "status": status,
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "output_dir": str(out_dir),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "eval": eval_data,
    }


def summarize_batch(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metric_keys = [
        "legacy_scene_grounding_precision",
        "legacy_scene_grounding_recall",
        "legacy_scene_grounding_f1",
        "node_grounding_precision",
        "node_grounding_recall",
        "node_grounding_f1",
        "gold_fact_recall",
        "pred_fact_precision",
        "fact_f1",
        "important_pred_transition_coherence",
        "development_correctness",
        "state_transition_correctness",
        "pred_transition_coherence",
        "arc_narrative_aspect_correctness",
        "arc_progression_correctness",
        "overall",
    ]

    def group_rows(lang: str) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for item in results:
            if item.get("status") not in {"ok", "ok_timeout", "skipped"}:
                continue
            if item.get("language") != lang:
                continue
            eval_data = item.get("eval") or {}
            if isinstance(eval_data, dict) and eval_data:
                rows.append(eval_data)  # type: ignore[arg-type]
        return rows

    grouped = {"en": group_rows("en"), "zh": group_rows("zh")}
    all_rows = grouped["en"] + grouped["zh"]

    def mean_block(rows: List[Dict[str, float]]) -> Dict[str, object]:
        if not rows:
            return {"n": 0}
        out: Dict[str, object] = {"n": len(rows)}
        for key in metric_keys:
            vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
            out[key] = round(sum(vals) / len(vals), 4)
        return out

    return {
        "overall": mean_block(all_rows),
        "english": mean_block(grouped["en"]),
        "chinese": mean_block(grouped["zh"]),
    }


def run_batch(args: argparse.Namespace) -> None:
    benchmark_root = Path(args.benchmark_root)
    output_root = Path(args.output_root)
    report_path = Path(args.report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    languages = [x.strip() for x in args.languages.split(",") if x.strip() in {"en", "zh"}]
    requested_ids = {x.strip() for x in args.movie_ids.split(",") if x.strip()}
    movie_dirs = discover_movie_dirs(benchmark_root, languages)
    if requested_ids:
        movie_dirs = [x for x in movie_dirs if x.name in requested_ids]

    results: List[Dict[str, object]] = []
    print(json.dumps({"stage": "batch_start", "movie_count": len(movie_dirs), "max_workers": args.max_workers}, ensure_ascii=False), flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {
            executor.submit(run_one_movie, movie_dir, output_root, args.overwrite, args.evaluate, args.per_movie_timeout_sec): movie_dir.name
            for movie_dir in movie_dirs
        }
        pending = set(future_map.keys())
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results.append(result)
                brief = {
                    "movie_id": result.get("movie_id"),
                    "language": result.get("language"),
                    "status": result.get("status"),
                    "elapsed_sec": result.get("elapsed_sec"),
                }
                eval_data = result.get("eval")
                if args.evaluate and isinstance(eval_data, dict) and eval_data:
                    brief["overall"] = eval_data.get("overall")
                    brief["node_grounding_f1"] = eval_data.get("node_grounding_f1")
                print(json.dumps(brief, ensure_ascii=False), flush=True)

    report = {
        "movie_count": len(results),
        "ok_count": sum(1 for r in results if r.get("status") == "ok"),
        "ok_timeout_count": sum(1 for r in results if r.get("status") == "ok_timeout"),
        "skipped_count": sum(1 for r in results if r.get("status") == "skipped"),
        "failed_count": sum(1 for r in results if r.get("status") == "failed"),
        "timeout_count": sum(1 for r in results if r.get("status") == "timeout"),
        "benchmark_root": str(benchmark_root),
        "output_root": str(output_root),
        "summary": summarize_batch(results),
        "results": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "batch_complete", "report_path": str(report_path), "summary": report["summary"]}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="STAGE Task 1 rebuild and evaluation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    one_parser = subparsers.add_parser("one", help="Run the Task 1 pipeline for one movie.")
    one_parser.add_argument("--movie-dir", required=True)
    one_parser.add_argument("--output-dir", required=True)
    one_parser.add_argument("--evaluate", action="store_true")

    batch_parser = subparsers.add_parser("batch", help="Run the Task 1 pipeline for a batch of movies.")
    batch_parser.add_argument("--benchmark-root", default=str(Path(__file__).resolve().parents[2]))
    batch_parser.add_argument("--languages", default="en,zh")
    batch_parser.add_argument("--movie-ids", default="")
    batch_parser.add_argument("--output-root", required=True)
    batch_parser.add_argument("--report-path", required=True)
    batch_parser.add_argument("--max-workers", type=int, default=4)
    batch_parser.add_argument("--per-movie-timeout-sec", type=int, default=5400)
    batch_parser.add_argument("--overwrite", action="store_true")
    batch_parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()
    if args.command == "one":
        movie_dir = Path(args.movie_dir)
        output_dir = Path(args.output_dir)
        run_workflow_v66(movie_dir, output_dir)
        if args.evaluate:
            summary = evaluate_v5(movie_dir, output_dir)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    run_batch(args)


if __name__ == "__main__":
    main()
