#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLMClient,
    SceneRecord,
    arc_prompt,
    clean_text,
    detect_language,
    llm_json,
    load_scenes,
    stable_id,
    tokenize,
)
from recall import alias_evidence_quotes, build_aliases, build_timeline_summary, prompt_messages

EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "for", "in", "on", "at", "by", "with", "from",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those", "his", "her",
    "their", "them", "he", "she", "they", "it", "as", "into", "over", "under", "after", "before", "during",
    "while", "through", "about", "because", "when", "then", "than", "very", "more", "most", "just", "still",
}

UPPER_LINE_RE = re.compile(r"^[A-Z][A-Z .\-']{1,40}$")


def is_placeholder_summary(text: str) -> bool:
    text = clean_text(text)
    return text.startswith("Milestone-based timeline for ") or text.startswith("基于 milestone 的")


def summary_needs_repair(language: str, text: str) -> bool:
    text = clean_text(text)
    if not text or is_placeholder_summary(text):
        return True
    if language == "zh":
        return len(text) < 20 or len(text) > 140
    word_count = len(text.split())
    return word_count < 12 or word_count > 110


def candidate_lines(scene: SceneRecord) -> List[str]:
    out: List[str] = []
    for raw in scene.content.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 4:
            continue
        if len(line) > 220:
            continue
        if line not in out:
            out.append(line)
    return out


def keyword_set(texts: Sequence[str], language: str) -> set[str]:
    terms: set[str] = set()
    for text in texts:
        for token in tokenize(clean_text(text), language):
            token = clean_text(token).lower()
            if not token:
                continue
            if language == "en":
                if token in EN_STOPWORDS or len(token) <= 2:
                    continue
            terms.add(token)
    return terms


def deterministic_evidence_quotes(scene: SceneRecord, aliases: Sequence[str], node: Dict[str, Any], max_quotes: int = 3) -> List[str]:
    alias_quotes = alias_evidence_quotes(scene, aliases, max_quotes=max_quotes)
    keywords = keyword_set(
        [
            node.get("role_in_context"),
            node.get("salient_development"),
            node.get("goal_state"),
            node.get("resulting_state"),
            node.get("unresolved_issue"),
        ],
        scene.language,
    )
    scored: List[Tuple[int, int, str]] = []
    for idx, line in enumerate(candidate_lines(scene)):
        norm_line = clean_text(line).lower()
        line_tokens = keyword_set([line], scene.language)
        alias_hit = any(clean_text(alias).lower() in norm_line for alias in aliases if clean_text(alias))
        overlap = len(keywords & line_tokens)
        score = overlap
        if alias_hit:
            score += 4
        if UPPER_LINE_RE.match(line):
            score -= 1
        if score <= 0:
            continue
        scored.append((score, -idx, line))
    quotes: List[str] = []
    for _, _, line in sorted(scored, reverse=True):
        if line not in quotes:
            quotes.append(line)
        if len(quotes) >= max_quotes:
            break
    if not quotes:
        quotes = alias_quotes
    return quotes[:max_quotes]


def batch_node_repair_prompt(
    language: str,
    character_name: str,
    aliases: Sequence[str],
    items: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    rows = []
    for item in items:
        rows.append(
            {
                "timeline_node_id": item["timeline_node_id"],
                "scene_order": int(item["scene_order"]),
                "scene_title": item["scene_title"],
                "current_node": {
                    "role_in_context": item.get("role_in_context"),
                    "salient_development": item.get("salient_development"),
                    "goal_state": item.get("goal_state"),
                    "resulting_state": item.get("resulting_state"),
                    "unresolved_issue": item.get("unresolved_issue"),
                    "evidence_quotes": item.get("evidence_quotes", []),
                },
                "scene_text": item["scene_text"],
            }
        )
    if language == "zh":
        system = "你在批量修复角色时间线节点。只输出 JSON。"
        user = (
            f"角色: {character_name}\n"
            f"别名: {json.dumps(list(aliases), ensure_ascii=False)}\n"
            "请只修复给定节点中明显偏弱的字段，重点补 goal_state，并尽量给出 1-3 条可在场景原文中直接找到的逐字 evidence_quotes。\n"
            "要求：保持 scene-grounded；不要夸大角色变化；如果无法可靠判断 goal_state，可返回 null。\n"
            "输出格式: {\"repairs\": [{\"timeline_node_id\":\"...\", \"role_in_context\":\"...\", \"salient_development\":\"...\", \"goal_state\":\"...或null\", \"resulting_state\":\"...或null\", \"unresolved_issue\":\"...或null\", \"evidence_quotes\":[\"...\"]}]}\n"
            f"待修节点:\n{json.dumps(rows, ensure_ascii=False, indent=2)}"
        )
    else:
        system = "You are repairing focal-character timeline nodes in batch. Output JSON only."
        user = (
            f"Character: {character_name}\n"
            f"Aliases: {json.dumps(list(aliases), ensure_ascii=False)}\n"
            "Repair only clearly weak fields in the supplied nodes. Prioritize filling goal_state, and extract 1-3 exact evidence_quotes that appear verbatim in the scene text when possible.\n"
            "Stay scene-grounded, avoid exaggeration, and return null for goal_state if the scene does not support a reliable inference.\n"
            "Return format: {\"repairs\": [{\"timeline_node_id\":\"...\", \"role_in_context\":\"...\", \"salient_development\":\"...\", \"goal_state\":\"... or null\", \"resulting_state\":\"... or null\", \"unresolved_issue\":\"... or null\", \"evidence_quotes\":[\"...\"]}]}\n"
            f"Nodes to repair:\n{json.dumps(rows, ensure_ascii=False, indent=2)}"
        )
    return prompt_messages(system, user)


def build_fallback_arc(language: str, character_name: str, nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(nodes) < 2:
        return []
    if len(nodes) >= 3:
        linked = [nodes[0], nodes[len(nodes) // 2], nodes[-1]]
    else:
        linked = [nodes[0], nodes[-1]]
    linked_ids = [clean_text(node["timeline_node_id"]) for node in linked]
    start_state = clean_text(nodes[0].get("goal_state") or nodes[0].get("salient_development")) or None
    end_state = clean_text(nodes[-1].get("resulting_state") or nodes[-1].get("salient_development")) or None
    unresolved = clean_text(nodes[-1].get("unresolved_issue")) or None
    if language == "zh":
        title = "角色主线推进"
        summary = (
            f"{character_name}从前段的{clean_text(nodes[0].get('salient_development')) or '初始处境'}，"
            f"逐步推进到后段的{clean_text(nodes[-1].get('salient_development')) or '关键落点'}，"
            "中间节点体现了这条叙事主线的持续演化。"
        )
    else:
        title = "Primary Story Trajectory"
        summary = (
            f"{character_name} moves from {clean_text(nodes[0].get('salient_development')) or 'an initial defining situation'} "
            f"to {clean_text(nodes[-1].get('salient_development')) or 'a later decisive position'}, "
            "with the linked nodes capturing the main cross-scene trajectory."
        )
    return [
        {
            "arc_id": stable_id(character_name, title, prefix="v6arc"),
            "character_name": character_name,
            "title": title,
            "arc_focus": "mixed",
            "linked_timeline_node_ids": linked_ids,
            "arc_summary": summary,
            "start_state": start_state,
            "end_state": end_state,
            "unresolved_issue": unresolved,
        }
    ]


def normalize_repaired_quotes(scene: SceneRecord, aliases: Sequence[str], node: Dict[str, Any], raw_quotes: Sequence[Any]) -> List[str]:
    quotes: List[str] = []
    for quote in raw_quotes or []:
        text = clean_text(quote)
        if text and text in scene.content and text not in quotes:
            quotes.append(text)
    if not quotes:
        quotes = deterministic_evidence_quotes(scene, aliases, node)
    return quotes[:4]


def repair_character_nodes(
    llm: LLMClient,
    language: str,
    character_name: str,
    aliases: Sequence[str],
    nodes: List[Dict[str, Any]],
    scene_map: Dict[str, SceneRecord],
) -> int:
    changed = 0
    pending: List[Dict[str, Any]] = []
    node_by_id = {clean_text(node.get("timeline_node_id")): node for node in nodes}

    for node in nodes:
        scene = scene_map.get(str(node.get("scene_id")))
        if scene is None:
            continue
        if not (node.get("evidence_quotes") or []):
            repaired_quotes = deterministic_evidence_quotes(scene, aliases, node)
            if repaired_quotes:
                node["evidence_quotes"] = repaired_quotes
                changed += 1
        if not clean_text(node.get("goal_state")) or not (node.get("evidence_quotes") or []):
            pending.append(
                {
                    "timeline_node_id": clean_text(node.get("timeline_node_id")),
                    "scene_order": int(node.get("scene_order", 0) or 0),
                    "scene_title": clean_text(node.get("scene_title")),
                    "role_in_context": clean_text(node.get("role_in_context")),
                    "salient_development": clean_text(node.get("salient_development")),
                    "goal_state": clean_text(node.get("goal_state")) or None,
                    "resulting_state": clean_text(node.get("resulting_state")) or None,
                    "unresolved_issue": clean_text(node.get("unresolved_issue")) or None,
                    "evidence_quotes": list(node.get("evidence_quotes") or []),
                    "scene_text": scene.content[:5000],
                }
            )

    for start in range(0, len(pending), 3):
        batch = pending[start : start + 3]
        try:
            raw = llm_json(llm, batch_node_repair_prompt(language, character_name, aliases, batch), max_tokens=2200)
            repairs = raw.get("repairs", []) or []
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "stage": "task1_quality_node_repair_failed",
                        "character": character_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            repairs = []
        for repair in repairs:
            node_id = clean_text(repair.get("timeline_node_id"))
            node = node_by_id.get(node_id)
            if not node:
                continue
            scene = scene_map.get(str(node.get("scene_id")))
            if scene is None:
                continue
            before = json.dumps(node, ensure_ascii=False, sort_keys=True)
            role_in_context = clean_text(repair.get("role_in_context")) or node.get("role_in_context")
            salient = clean_text(repair.get("salient_development")) or node.get("salient_development")
            goal = clean_text(repair.get("goal_state")) or node.get("goal_state") or None
            result = clean_text(repair.get("resulting_state")) or node.get("resulting_state") or None
            issue = clean_text(repair.get("unresolved_issue")) or node.get("unresolved_issue") or None
            quotes = normalize_repaired_quotes(scene, aliases, node, repair.get("evidence_quotes") or [])
            node.update(
                {
                    "role_in_context": role_in_context,
                    "salient_development": salient,
                    "goal_state": goal,
                    "resulting_state": result,
                    "unresolved_issue": issue,
                    "evidence_quotes": quotes,
                }
            )
            after = json.dumps(node, ensure_ascii=False, sort_keys=True)
            if before != after:
                changed += 1

    remaining_goal_missing = [node for node in nodes if not clean_text(node.get("goal_state"))]
    for node in remaining_goal_missing:
        scene = scene_map.get(str(node.get("scene_id")))
        if scene is None:
            continue
        item = {
            "timeline_node_id": clean_text(node.get("timeline_node_id")),
            "scene_order": int(node.get("scene_order", 0) or 0),
            "scene_title": clean_text(node.get("scene_title")),
            "role_in_context": clean_text(node.get("role_in_context")),
            "salient_development": clean_text(node.get("salient_development")),
            "goal_state": None,
            "resulting_state": clean_text(node.get("resulting_state")) or None,
            "unresolved_issue": clean_text(node.get("unresolved_issue")) or None,
            "evidence_quotes": list(node.get("evidence_quotes") or []),
            "scene_text": scene.content[:5000],
        }
        try:
            raw = llm_json(llm, batch_node_repair_prompt(language, character_name, aliases, [item]), max_tokens=1200)
            repairs = raw.get("repairs", []) or []
        except Exception:
            repairs = []
        if not repairs:
            continue
        repair = repairs[0]
        goal = clean_text(repair.get("goal_state"))
        if goal:
            node["goal_state"] = goal
            changed += 1
    return changed


def repair_character_arcs(
    llm: LLMClient,
    language: str,
    character_name: str,
    nodes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if len(nodes) < 2:
        return []
    raw_nodes = [
        {
            "timeline_node_id": node["timeline_node_id"],
            "scene_id": node["scene_id"],
            "scene_order": node["scene_order"],
            "scene_title": node["scene_title"],
            "salient_development": node["salient_development"],
            "goal_state": node.get("goal_state"),
            "resulting_state": node.get("resulting_state"),
            "unresolved_issue": node.get("unresolved_issue"),
        }
        for node in nodes
    ]
    try:
        raw = llm_json(llm, arc_prompt(language, character_name, raw_nodes), max_tokens=1800)
        items = raw.get("arcs", []) or []
    except Exception as exc:
        print(
            json.dumps(
                {
                    "stage": "task1_quality_arc_repair_failed",
                    "character": character_name,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        items = []

    valid: List[Dict[str, Any]] = []
    node_ids = {clean_text(node["timeline_node_id"]) for node in nodes}
    for item in items:
        linked_ids = [clean_text(x) for x in (item.get("linked_timeline_node_ids") or []) if clean_text(x) in node_ids]
        linked_ids = list(dict.fromkeys(linked_ids))
        if len(linked_ids) < 2:
            continue
        valid.append(
            {
                "arc_id": stable_id(character_name, clean_text(item.get("title")), prefix="v6arc"),
                "character_name": character_name,
                "title": clean_text(item.get("title")),
                "arc_focus": clean_text(item.get("arc_focus")) or "mixed",
                "linked_timeline_node_ids": linked_ids,
                "arc_summary": clean_text(item.get("arc_summary")),
                "start_state": clean_text(item.get("start_state")) or None,
                "end_state": clean_text(item.get("end_state")) or None,
                "unresolved_issue": clean_text(item.get("unresolved_issue")) or None,
            }
        )
    if valid:
        return valid
    return build_fallback_arc(language, character_name, nodes)


def repair_movie(movie_dir: Path) -> Dict[str, Any]:
    language = detect_language(movie_dir)
    llm = LLMClient(DEFAULT_LLM_MODEL, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY)
    scenes = load_scenes(movie_dir / "script.json", language)
    scene_map = {scene.scene_id: scene for scene in scenes}

    timeline_path = movie_dir / "task_1_character_timelines.json"
    arc_path = movie_dir / "task_1_cross_scene_arcs.json"
    timeline_payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    arc_payload = json.loads(arc_path.read_text(encoding="utf-8"))

    changed_nodes = 0
    changed_summaries = 0
    added_arcs = 0

    arcs_by_character: Dict[str, List[Dict[str, Any]]] = {}
    for arc in arc_payload.get("cross_scene_arcs", []) or []:
        arcs_by_character.setdefault(clean_text(arc.get("character_name")), []).append(arc)

    for item in timeline_payload.get("focal_character_timelines", []) or []:
        character_name = clean_text(item.get("character_name"))
        aliases = build_aliases({"character_name": character_name, "aliases": item.get("aliases", []) or []})
        nodes = item.get("timeline_nodes", []) or []

        changed_nodes += repair_character_nodes(llm, language, character_name, aliases, nodes, scene_map)

        summary = clean_text(item.get("timeline_summary"))
        if summary_needs_repair(language, summary):
            item["timeline_summary"] = build_timeline_summary(llm, language, character_name, nodes)
            changed_summaries += 1

        if not arcs_by_character.get(character_name):
            new_arcs = repair_character_arcs(llm, language, character_name, nodes)
            if new_arcs:
                arc_payload.setdefault("cross_scene_arcs", []).extend(new_arcs)
                arcs_by_character[character_name] = list(new_arcs)
                added_arcs += len(new_arcs)

    timeline_payload.setdefault("build_summary", {})
    timeline_payload["build_summary"]["timeline_node_count"] = sum(
        len(item.get("timeline_nodes", []) or []) for item in timeline_payload.get("focal_character_timelines", []) or []
    )
    arc_payload.setdefault("build_summary", {})
    arc_payload["build_summary"]["arc_count"] = len(arc_payload.get("cross_scene_arcs", []) or [])

    timeline_path.write_text(json.dumps(timeline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    arc_path.write_text(json.dumps(arc_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "movie_id": movie_dir.name,
        "language": language,
        "changed_nodes": changed_nodes,
        "changed_summaries": changed_summaries,
        "added_arcs": added_arcs,
    }


def discover_movies(benchmark_root: Path, languages: Sequence[str], movie_ids: Optional[set[str]]) -> List[Path]:
    rows: List[Path] = []
    for language in languages:
        lang_dir = benchmark_root / ("English" if language == "en" else "Chinese")
        if not lang_dir.exists():
            continue
        for movie_dir in sorted(path for path in lang_dir.iterdir() if path.is_dir()):
            if movie_ids and movie_dir.name not in movie_ids:
                continue
            if (movie_dir / "script.json").exists() and (movie_dir / "task_1_character_timelines.json").exists():
                rows.append(movie_dir)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=Path("/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0"))
    parser.add_argument("--languages", nargs="*", choices=["en", "zh"], default=["en", "zh"])
    parser.add_argument("--movie-id", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    movie_ids = {clean_text(x) for x in args.movie_id if clean_text(x)} or None
    movies = discover_movies(args.benchmark_root, args.languages, movie_ids)
    print(json.dumps({"stage": "task1_quality_repair_start", "movie_count": len(movies), "max_workers": args.max_workers}, ensure_ascii=False), flush=True)

    started = time.time()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {executor.submit(repair_movie, movie_dir): movie_dir.name for movie_dir in movies}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(json.dumps({"stage": "task1_quality_repair_movie_done", **result}, ensure_ascii=False), flush=True)

    summary = {
        "movie_count": len(results),
        "changed_movies": sum(1 for row in results if row["changed_nodes"] or row["changed_summaries"] or row["added_arcs"]),
        "changed_nodes": sum(row["changed_nodes"] for row in results),
        "changed_summaries": sum(row["changed_summaries"] for row in results),
        "added_arcs": sum(row["added_arcs"] for row in results),
        "elapsed_sec": round(time.time() - started, 2),
    }
    print(json.dumps({"stage": "task1_quality_repair_complete", "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
