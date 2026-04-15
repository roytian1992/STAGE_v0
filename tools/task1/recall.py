#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
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
)
from metrics import evaluate_v5, load_target_characters

UPPER_LINE_RE = re.compile(r"^[A-Z][A-Z .\-']{1,40}$")
SHORTLIST_STRONG_HIT = 6
SHORTLIST_MAX_SCENES = 120
SHORTLIST_PER_BUCKET = 24
HIGH_MILESTONE_SCORE = 4
SCENE_ROLE_BATCH_SIZE = 2

SCENE_ROLE_PROMPT_EN = """You are building a role-conditioned scene card for a screenplay benchmark.

You will receive one scene and one focal character.
Your job is to determine whether this scene is meaningfully part of the focal character's storyline, and if so, what durable narrative function it plays for that character.

Be conservative. Prefer externally verifiable changes over broad interpretation.
If the character is not meaningfully active in this scene, mark the scene_role as absent.

Return JSON only:
{
  "scene_role": "foreground|active|indirect|absent",
  "role_in_scene": "string",
  "external_change": "string or null",
  "relation_shift": "string or null",
  "goal_shift": "string or null",
  "state_pressure": "string or null",
  "mission_relevance": "string or null",
  "milestone_score": 0,
  "milestone_reason": "string",
  "evidence_quotes": ["exact quote", "..."]
}

Rules:
- milestone_score should be 0-5.
- Use 4-5 only when this scene clearly changes the character's status, relationship, goal, worldview, or role in the story.
- Use 0 when the character is absent or only trivially mentioned.
- evidence_quotes should be exact substrings from the scene when possible.
"""

SCENE_ROLE_PROMPT_ZH = """你在为 screenplay benchmark 构建“角色条件化场景卡”。

你会收到一场戏和一个焦点角色。
你的任务是判断这场戏是否真正属于该角色的叙事线；如果属于，要说明它对该角色产生了什么可持续的叙事作用。

请保守判断，优先保留可外显验证的变化，不要泛化脑补。
如果该角色在这场戏里并没有真正参与其叙事发展，就把 scene_role 标成 absent。

只输出 JSON：
{
  "scene_role": "foreground|active|indirect|absent",
  "role_in_scene": "字符串",
  "external_change": "字符串或 null",
  "relation_shift": "字符串或 null",
  "goal_shift": "字符串或 null",
  "state_pressure": "字符串或 null",
  "mission_relevance": "字符串或 null",
  "milestone_score": 0,
  "milestone_reason": "字符串",
  "evidence_quotes": ["逐字引文", "..."]
}

要求：
- milestone_score 取值 0-5。
- 只有当该场戏明显改变角色的状态、关系、目标、认知或其在主叙事中的位置时，才给到 4-5。
- 如果角色缺席或只是被顺带提到，给 0。
- evidence_quotes 尽量使用场景原文中的逐字片段。
"""

BATCH_SCENE_ROLE_PROMPT_EN = """You are building role-conditioned scene cards for a screenplay benchmark.

You will receive one focal character and a batch of scenes.
For each supplied scene, determine whether it is meaningfully part of the focal character's storyline, and if so, what durable narrative function it plays for that character.

Be conservative. Prefer externally verifiable changes over broad interpretation.
If the character is not meaningfully active in a scene, mark the scene_role as absent and milestone_score as 0.

Return JSON only:
{
  "scene_cards": [
    {
      "scene_id": "string",
      "scene_role": "foreground|active|indirect|absent",
      "role_in_scene": "string",
      "external_change": "string or null",
      "relation_shift": "string or null",
      "goal_shift": "string or null",
      "state_pressure": "string or null",
      "mission_relevance": "string or null",
      "milestone_score": 0,
      "milestone_reason": "string",
      "evidence_quotes": ["exact quote", "..."]
    }
  ]
}

Rules:
- Output exactly one item for each supplied scene_id.
- milestone_score should be 0-5.
- Use 4-5 only when this scene clearly changes the character's status, relationship, goal, worldview, or role in the story.
- evidence_quotes should be exact substrings from that scene when possible.
"""

BATCH_SCENE_ROLE_PROMPT_ZH = """你在为 screenplay benchmark 批量构建“角色条件化场景卡”。

你会收到一个焦点角色和一批场景。
对每一场给定的戏，判断它是否真正属于该角色的叙事线；如果属于，要说明它对该角色产生了什么可持续的叙事作用。

请保守判断，优先保留可外显验证的变化，不要泛化脑补。
如果角色在某场戏里并没有真正参与其叙事发展，就把 scene_role 标成 absent，milestone_score 设为 0。

只输出 JSON：
{
  "scene_cards": [
    {
      "scene_id": "字符串",
      "scene_role": "foreground|active|indirect|absent",
      "role_in_scene": "字符串",
      "external_change": "字符串或 null",
      "relation_shift": "字符串或 null",
      "goal_shift": "字符串或 null",
      "state_pressure": "字符串或 null",
      "mission_relevance": "字符串或 null",
      "milestone_score": 0,
      "milestone_reason": "字符串",
      "evidence_quotes": ["逐字引文", "..."]
    }
  ]
}

要求：
- 每个给定 scene_id 都必须且只输出一条结果。
- milestone_score 取值 0-5。
- 只有当该场戏明显改变角色的状态、关系、目标、认知或其在主叙事中的位置时，才给到 4-5。
- evidence_quotes 尽量使用对应场景原文中的逐字片段。
"""

PAIRWISE_MILESTONE_JUDGE_PROMPT_EN = """You are resolving a local milestone-selection conflict for a screenplay benchmark.

You will receive one focal character and two nearby candidate scenes from the same mini-arc.
Choose which scene is the better representative benchmark milestone for the character's durable development.

Prefer the scene that is more benchmark-useful:
- captures a more durable, externally verifiable development
- is less redundant with a nearby continuation/restatement scene
- better represents the local turning point for the character

Use "both" only when the two scenes clearly capture different durable developments that both deserve separate timeline nodes.
Use "a" or "b" in all normal cases.

Return JSON only:
{
  "decision": "a|b|both",
  "reason": "string"
}
"""

PAIRWISE_MILESTONE_JUDGE_PROMPT_ZH = """你在为 screenplay benchmark 解决一个局部 milestone 选择冲突。

你会收到一个焦点角色，以及同一 mini-arc 里相近的两场候选戏。
请判断哪一场更适合作为该角色“持续性发展”的 benchmark milestone 代表。

优先选择更适合 benchmark 的那一场：
- 更能体现持续、可外显验证的发展
- 不只是附近另一场戏的延续或情绪复述
- 更能代表该局部段落里的关键转折

只有当两场戏明显对应两种不同、都值得单独保留的持续性发展时，才选 "both"。
一般情况应输出 "a" 或 "b"。

只输出 JSON：
{
  "decision": "a|b|both",
  "reason": "字符串"
}
"""

SEGMENT_SUMMARY_PROMPT_EN = """You are summarizing a focal character's storyline segment from scene-role cards.
Return JSON only in the format {"segment_summary":"...", "dominant_aspect":"..."}.
Use concise wording and stay close to the evidence cards."""

SEGMENT_SUMMARY_PROMPT_ZH = """你在根据场景卡总结某个角色的一段叙事阶段。
只输出 JSON，格式为 {"segment_summary":"...", "dominant_aspect":"..."}。
表达简洁，并尽量贴近给定证据。"""

NODE_RENDER_PROMPT_EN = """You are rendering one milestone scene into a benchmark-style Task 1 timeline node.
Use only the supplied scene-role card and scene text. Do not import other scenes.
Return JSON only:
{
  "role_in_context": "string",
  "salient_development": "string",
  "goal_state": "string or null",
  "resulting_state": "string or null",
  "unresolved_issue": "string or null",
  "evidence_quotes": ["exact quote", "..."]
}
"""

NODE_RENDER_PROMPT_ZH = """你在把一个 milestone 场景渲染成 benchmark 风格的 Task 1 时间线节点。
只能使用给定的场景卡和场景原文，不要引入别的场景。
只输出 JSON：
{
  "role_in_context": "字符串",
  "salient_development": "字符串",
  "goal_state": "字符串或 null",
  "resulting_state": "字符串或 null",
  "unresolved_issue": "字符串或 null",
  "evidence_quotes": ["逐字引文", "..."]
}
"""


def prompt_messages(system: str, user: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text).lower())


def scene_speaker_names(scene: SceneRecord) -> List[str]:
    names: List[str] = []
    for raw in scene.content.splitlines():
        line = raw.strip().replace("(O.S.)", "").replace("(V.O.)", "")
        line = line.strip(" .:-")
        if not line or not UPPER_LINE_RE.match(line):
            continue
        if len(line.split()) > 4:
            continue
        names.append(clean_text(line.title().replace(" '", "'")))
    return list(dict.fromkeys(names))


def build_aliases(character: Dict[str, Any]) -> List[str]:
    aliases = [character["character_name"]] + list(character.get("aliases", []) or []) + list(character.get("_provenance_aliases", []) or [])
    out: List[str] = []
    for alias in aliases:
        alias = clean_text(alias)
        if alias and alias not in out:
            out.append(alias)
    return out


def alias_evidence_quotes(scene: SceneRecord, aliases: Sequence[str], max_quotes: int = 3) -> List[str]:
    quotes: List[str] = []
    haystacks = [scene.scene_title, scene.content]
    for alias in aliases:
        alias = clean_text(alias)
        if not alias:
            continue
        for text in haystacks:
            if alias in text and alias not in quotes:
                quotes.append(alias)
                break
        if len(quotes) >= max_quotes:
            break
    return quotes[:max_quotes]


def deterministic_scene_role_card(
    character_name: str,
    aliases: Sequence[str],
    scene: SceneRecord,
    language: str,
    score: int,
) -> Dict[str, Any]:
    speaker_norms = {normalize_text_for_match(x) for x in scene_speaker_names(scene)}
    alias_norms = [normalize_text_for_match(x) for x in aliases if clean_text(x)]
    has_speaker_hit = any(alias in speaker_norms for alias in alias_norms if alias)
    if has_speaker_hit and score >= SHORTLIST_STRONG_HIT + 2:
        scene_role = "foreground"
        milestone_score = 3
    elif has_speaker_hit or score >= SHORTLIST_STRONG_HIT + 1:
        scene_role = "active"
        milestone_score = 2
    else:
        scene_role = "indirect"
        milestone_score = 1 if score > 0 else 0
    reason = (
        "deterministic recall fallback from alias/speaker evidence"
        if language == "en"
        else "基于别名/说话人证据的确定性 recall 回退"
    )
    return {
        "character_name": character_name,
        "scene_id": scene.scene_id,
        "scene_order": scene.scene_order,
        "scene_title": scene.scene_title,
        "scene_role": scene_role,
        "role_in_scene": "",
        "external_change": None,
        "relation_shift": None,
        "goal_shift": None,
        "state_pressure": None,
        "mission_relevance": None,
        "milestone_score": milestone_score,
        "milestone_reason": reason,
        "evidence_quotes": alias_evidence_quotes(scene, aliases),
    }


def batched(items: Sequence[Any], batch_size: int) -> List[List[Any]]:
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def alias_match_score(scene: SceneRecord, aliases: Sequence[str], speaker_names: Sequence[str]) -> int:
    title = normalize_text_for_match(scene.scene_title)
    content = normalize_text_for_match(scene.content)
    speaker_set = {normalize_text_for_match(x) for x in speaker_names}
    score = 0
    for alias in aliases:
        norm = normalize_text_for_match(alias)
        if not norm:
            continue
        if norm in title:
            score += 3
        if norm in content:
            score += 3
        pieces = [p for p in norm.split() if len(p) >= 3]
        if any(p in content for p in pieces):
            score += 1
        if norm in speaker_set:
            score += 4
        if pieces and any(p in speaker_set for p in pieces):
            score += 2
    return score


def scene_by_order_map(scenes: Sequence[SceneRecord]) -> Dict[int, SceneRecord]:
    return {scene.scene_order: scene for scene in scenes}


def dynamic_node_budget(cards: Sequence[Dict[str, Any]]) -> int:
    if not cards:
        return 9
    high = sum(1 for c in cards if int(c.get("milestone_score", 0) or 0) >= HIGH_MILESTONE_SCORE)
    medium = sum(1 for c in cards if int(c.get("milestone_score", 0) or 0) >= 3)
    active = sum(1 for c in cards if c.get("scene_role") in {"foreground", "active"})
    budget = 9
    if len(cards) >= 18:
        budget += 1
    if len(cards) >= 30:
        budget += 1
    if medium >= 8:
        budget += 1
    if high >= 5:
        budget += 1
    if active >= 12:
        budget += 1
    return max(9, min(14, budget))


def benchmark_movie_dir(movie_dir: Path) -> Optional[Path]:
    try:
        root = movie_dir.parents[2]
    except Exception:
        return None
    bench_dir = root / "benchmarks" / "STAGEBenchmark" / movie_dir.parent.name / movie_dir.name
    return bench_dir if bench_dir.exists() else None


def compact_match_text(text: str) -> str:
    raw = normalize_text_for_match(text).replace("_", " ")
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", raw)


def provenance_name_matches(candidate: str, aliases: Sequence[str]) -> bool:
    cand = compact_match_text(candidate)
    if not cand:
        return False
    for alias in aliases:
        ali = compact_match_text(alias)
        if not ali:
            continue
        if ali == cand or ali in cand or cand in ali:
            return True
        if len(ali) >= 5 and len(cand) >= 5 and SequenceMatcher(None, ali, cand).ratio() >= 0.86:
            return True
    return False


def provenance_scene_rescue(movie_dir: Path, scenes: Sequence[SceneRecord], character: Dict[str, Any]) -> List[SceneRecord]:
    bench_dir = benchmark_movie_dir(movie_dir)
    if bench_dir is None:
        return []
    extraction_path = bench_dir / "extraction_results.json"
    if not extraction_path.exists():
        return []
    try:
        data = json.loads(extraction_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    aliases = build_aliases(character)
    scene_by_order = {str(scene.scene_order): scene for scene in scenes}
    rescued: List[SceneRecord] = []
    rescued_ids = set()
    matched_aliases: List[str] = []

    for _, entry in data.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get("document_metadata", {}) or {}
        scene_order = clean_text(meta.get("scene_id"))
        if not scene_order or scene_order not in scene_by_order:
            continue
        entities = entry.get("entities", []) or []
        hit = False
        for ent in entities:
            ent_type = clean_text(ent.get("type"))
            ent_name = clean_text(ent.get("name"))
            ent_desc = clean_text(ent.get("description"))
            if ent_type == "Character" and (provenance_name_matches(ent_name, aliases) or provenance_name_matches(ent_desc, aliases)):
                hit = True
                if ent_name and ent_name not in matched_aliases:
                    matched_aliases.append(ent_name)
        if not hit:
            continue
        scene = scene_by_order[scene_order]
        if scene.scene_id not in rescued_ids:
            rescued.append(scene)
            rescued_ids.add(scene.scene_id)

    if matched_aliases:
        character["_provenance_aliases"] = matched_aliases
    return sorted(rescued, key=lambda x: x.scene_order)


def shortlist_scenes_for_character(scenes: Sequence[SceneRecord], character: Dict[str, Any], movie_dir: Optional[Path] = None) -> List[SceneRecord]:
    aliases = build_aliases(character)
    scored: List[Tuple[int, int, SceneRecord]] = []
    order_map = scene_by_order_map(scenes)
    for scene in scenes:
        speakers = scene_speaker_names(scene)
        score = alias_match_score(scene, aliases, speakers)
        if score > 0:
            scored.append((score, scene.scene_order, scene))
    if not scored:
        if movie_dir is not None:
            rescued = provenance_scene_rescue(movie_dir, scenes, character)
            if rescued:
                return rescued
        return []
    selected_ids = set()
    expanded: List[SceneRecord] = []
    for score, _, scene in sorted(scored, key=lambda x: (-x[0], x[1])):
        if scene.scene_id not in selected_ids:
            expanded.append(scene)
            selected_ids.add(scene.scene_id)
        if score >= SHORTLIST_STRONG_HIT:
            for delta in (-2, -1, 1, 2):
                neighbor = order_map.get(scene.scene_order + delta)
                if neighbor and neighbor.scene_id not in selected_ids:
                    expanded.append(neighbor)
                    selected_ids.add(neighbor.scene_id)
    direct_hits = sorted(expanded, key=lambda x: x.scene_order)
    if len(direct_hits) <= SHORTLIST_MAX_SCENES:
        return direct_hits

    max_scene_order = max(scene.scene_order for scene in scenes)
    selected_ids = set()
    by_bucket: Dict[str, List[Tuple[int, int, SceneRecord]]] = defaultdict(list)
    for score, scene_order, scene in scored:
        by_bucket[temporal_bucket(scene_order, max_scene_order)].append((score, scene_order, scene))

    selected: List[SceneRecord] = []
    for bucket in ("early", "middle", "late"):
        rows = sorted(by_bucket.get(bucket, []), key=lambda x: (-x[0], x[1]))
        for _, _, scene in rows[:SHORTLIST_PER_BUCKET]:
            if scene.scene_id in selected_ids:
                continue
            selected.append(scene)
            selected_ids.add(scene.scene_id)

    for _, _, scene in sorted(scored, key=lambda x: (-x[0], x[1])):
        if len(selected) >= SHORTLIST_MAX_SCENES:
            break
        if scene.scene_id in selected_ids:
            continue
        selected.append(scene)
        selected_ids.add(scene.scene_id)
    return sorted(selected, key=lambda x: x.scene_order)


def scene_role_prompt(language: str, character_name: str, scene: SceneRecord) -> List[Dict[str, str]]:
    system = SCENE_ROLE_PROMPT_ZH if language == "zh" else SCENE_ROLE_PROMPT_EN
    user = (
        f"Focal character: {character_name}\n"
        f"Scene title: {scene.scene_title}\n"
        f"Scene order: {scene.scene_order}\n"
        f"Scene text:\n{scene.content[:7000]}"
    )
    return prompt_messages(system, user)


def batch_scene_role_prompt(language: str, character_name: str, scenes: Sequence[SceneRecord]) -> List[Dict[str, str]]:
    system = BATCH_SCENE_ROLE_PROMPT_ZH if language == "zh" else BATCH_SCENE_ROLE_PROMPT_EN
    scene_rows = []
    for scene in scenes:
        scene_rows.append({
            "scene_id": scene.scene_id,
            "scene_order": scene.scene_order,
            "scene_title": scene.scene_title,
            "scene_text": scene.content[:1200],
        })
    user = (
        f"Focal character: {character_name}\n"
        f"Scenes:\n{json.dumps(scene_rows, ensure_ascii=False, indent=2)}"
    )
    return prompt_messages(system, user)


def normalize_scene_role_card(raw: Dict[str, Any], character_name: str, scene: SceneRecord) -> Dict[str, Any]:
    evidence_quotes = []
    for quote in raw.get("evidence_quotes", []) or []:
        quote = clean_text(quote)
        if quote and quote in scene.content and quote not in evidence_quotes:
            evidence_quotes.append(quote)
    try:
        milestone_score = int(raw.get("milestone_score", 0) or 0)
    except Exception:
        milestone_score = 0
    milestone_score = max(0, min(5, milestone_score))
    scene_role = clean_text(raw.get("scene_role")).lower() or "absent"
    if scene_role not in {"foreground", "active", "indirect", "absent"}:
        scene_role = "absent"
    return {
        "character_name": character_name,
        "scene_id": scene.scene_id,
        "scene_order": scene.scene_order,
        "scene_title": scene.scene_title,
        "scene_role": scene_role,
        "role_in_scene": clean_text(raw.get("role_in_scene")),
        "external_change": clean_text(raw.get("external_change")) or None,
        "relation_shift": clean_text(raw.get("relation_shift")) or None,
        "goal_shift": clean_text(raw.get("goal_shift")) or None,
        "state_pressure": clean_text(raw.get("state_pressure")) or None,
        "mission_relevance": clean_text(raw.get("mission_relevance")) or None,
        "milestone_score": milestone_score,
        "milestone_reason": clean_text(raw.get("milestone_reason")),
        "evidence_quotes": evidence_quotes[:4],
    }


def build_role_scene_cards(llm: LLMClient, scenes: Sequence[SceneRecord], character: Dict[str, Any], language: str, movie_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    shortlisted = shortlist_scenes_for_character(scenes, character, movie_dir=movie_dir)
    print(
        json.dumps(
            {
                "stage": "role_scene_shortlist_ready",
                "character": character["character_name"],
                "shortlist_count": len(shortlisted),
                "scene_orders": [scene.scene_order for scene in shortlisted[:24]],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    scene_map = {scene.scene_id: scene for scene in shortlisted}
    cards: List[Dict[str, Any]] = []
    seen_scene_ids = set()
    aliases = build_aliases(character)
    for group in batched(shortlisted, SCENE_ROLE_BATCH_SIZE):
        try:
            raw = llm_json(llm, batch_scene_role_prompt(language, character["character_name"], group), max_tokens=1400)
            raw_cards = raw.get("scene_cards", []) or []
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "stage": "role_scene_batch_failed",
                        "character": character["character_name"],
                        "scene_ids": [scene.scene_id for scene in group],
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            raw_cards = []
        for scene in group:
            matched = None
            for item in raw_cards:
                if clean_text(item.get("scene_id")) == scene.scene_id:
                    matched = item
                    break
            score = alias_match_score(scene, aliases, scene_speaker_names(scene))
            if matched is None:
                try:
                    single_raw = llm_json(llm, scene_role_prompt(language, character["character_name"], scene), max_tokens=700)
                    matched = dict(single_raw)
                    matched["scene_id"] = scene.scene_id
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "stage": "role_scene_single_failed",
                                "character": character["character_name"],
                                "scene_id": scene.scene_id,
                                "scene_order": scene.scene_order,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    matched = {"scene_id": scene.scene_id, "scene_role": "absent", "milestone_score": 0, "milestone_reason": ""}
            card = normalize_scene_role_card(matched, character["character_name"], scene_map[scene.scene_id])
            if card["scene_role"] == "absent" and card["milestone_score"] == 0 and score >= SHORTLIST_STRONG_HIT:
                card = deterministic_scene_role_card(character["character_name"], aliases, scene, language, score)
            if card["scene_role"] != "absent" or card["milestone_score"] > 0:
                cards.append(card)
                seen_scene_ids.add(card["scene_id"])
    for scene in shortlisted:
        if scene.scene_id in seen_scene_ids:
            continue
        score = alias_match_score(scene, aliases, scene_speaker_names(scene))
        if score >= SHORTLIST_STRONG_HIT:
            cards.append(deterministic_scene_role_card(character["character_name"], aliases, scene, language, score))
    cards = sorted(cards, key=lambda x: x["scene_order"])
    if not cards and shortlisted:
        rescue_rows = sorted(
            [
                (alias_match_score(scene, aliases, scene_speaker_names(scene)), scene.scene_order, scene)
                for scene in shortlisted
            ],
            key=lambda x: (-x[0], x[1]),
        )
        for score, _, scene in rescue_rows[: min(12, len(rescue_rows))]:
            if score <= 0:
                continue
            cards.append(deterministic_scene_role_card(character["character_name"], aliases, scene, language, score))
        cards = sorted(cards, key=lambda x: x["scene_order"])
        print(
            json.dumps(
                {
                    "stage": "role_scene_empty_rescued",
                    "character": character["character_name"],
                    "rescued_count": len(cards),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for idx, card in enumerate(cards):
        if int(card.get("milestone_score", 0) or 0) >= 2:
            continue
        neighbors = []
        if idx > 0:
            neighbors.append(cards[idx - 1])
        if idx + 1 < len(cards):
            neighbors.append(cards[idx + 1])
        if any(int(n.get("milestone_score", 0) or 0) >= HIGH_MILESTONE_SCORE for n in neighbors):
            if card.get("scene_role") == "absent":
                card["scene_role"] = "indirect"
            card["milestone_score"] = max(2, int(card.get("milestone_score", 0) or 0))
            if not clean_text(card.get("milestone_reason")):
                card["milestone_reason"] = (
                    "recall-first retention from adjacency to a high-value neighboring scene"
                    if language == "en"
                    else "因邻接高价值场景而进行 recall-first 保留"
                )
    return cards


def temporal_bucket(scene_order: int, max_scene_order: int) -> str:
    if max_scene_order <= 1:
        return "middle"
    ratio = (max(1, scene_order) - 1) / max(1, max_scene_order - 1)
    if ratio < 0.34:
        return "early"
    if ratio < 0.67:
        return "middle"
    return "late"


def summarize_segment_prompt(language: str, character_name: str, bucket: str, cards: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = SEGMENT_SUMMARY_PROMPT_ZH if language == "zh" else SEGMENT_SUMMARY_PROMPT_EN
    user = f"Character: {character_name}\nBucket: {bucket}\nScene-role cards:\n{json.dumps(list(cards), ensure_ascii=False, indent=2)}"
    return prompt_messages(system, user)


def deterministic_segment_summary(language: str, character_name: str, bucket: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scene_orders = [int(row.get("scene_order", 0) or 0) for row in rows]
    start_order = min(scene_orders) if scene_orders else 0
    end_order = max(scene_orders) if scene_orders else 0
    aspect_candidates = [
        ("relation", sum(1 for row in rows if clean_text(row.get("relation_shift")))),
        ("goal", sum(1 for row in rows if clean_text(row.get("goal_shift")))),
        ("external", sum(1 for row in rows if clean_text(row.get("external_change")))),
        ("pressure", sum(1 for row in rows if clean_text(row.get("state_pressure")))),
        ("mission", sum(1 for row in rows if clean_text(row.get("mission_relevance")))),
    ]
    dominant_aspect = max(aspect_candidates, key=lambda item: item[1])[0]
    representative = sorted(
        rows,
        key=lambda row: (int(row.get("milestone_score", 0) or 0), -int(row.get("scene_order", 0) or 0)),
        reverse=True,
    )[:2]
    evidence_bits: List[str] = []
    for row in representative:
        for key in ("external_change", "relation_shift", "goal_shift", "state_pressure", "mission_relevance", "milestone_reason", "role_in_scene"):
            value = clean_text(row.get(key))
            if value and value not in evidence_bits:
                evidence_bits.append(value)
                break
    joined = " / ".join(evidence_bits[:2])
    if language == "zh":
        summary = f"{character_name}在{bucket}阶段主要活动于第{start_order}-{end_order}场，叙事重点偏向{dominant_aspect}变化。"
        if joined:
            summary += f" 代表性线索包括：{joined}。"
    else:
        summary = f"{character_name}'s {bucket} stretch runs mainly across scenes {start_order}-{end_order}, with the main emphasis on {dominant_aspect} change."
        if joined:
            summary += f" Representative signals: {joined}."
    return {
        "segment_summary": summary,
        "dominant_aspect": dominant_aspect,
    }


def build_segments(llm: LLMClient, language: str, character_name: str, cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cards:
        return []
    max_scene_order = max(int(c["scene_order"]) for c in cards)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for card in cards:
        by_bucket[temporal_bucket(int(card["scene_order"]), max_scene_order)].append(card)
    segments = []
    for bucket in ("early", "middle", "late"):
        rows = sorted(by_bucket.get(bucket, []), key=lambda x: x["scene_order"])
        if not rows:
            continue
        try:
            raw = llm_json(llm, summarize_segment_prompt(language, character_name, bucket, rows[:10]), max_tokens=500)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "stage": "segment_summary_fallback",
                        "character": character_name,
                        "bucket": bucket,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            raw = deterministic_segment_summary(language, character_name, bucket, rows)
        segments.append({
            "bucket": bucket,
            "scene_count": len(rows),
            "scene_ids": [c["scene_id"] for c in rows],
            "segment_summary": clean_text(raw.get("segment_summary")),
            "dominant_aspect": clean_text(raw.get("dominant_aspect")),
        })
    return segments


def card_feature_text(card: Dict[str, Any]) -> str:
    parts = [
        clean_text(card.get("role_in_scene")),
        clean_text(card.get("external_change")),
        clean_text(card.get("relation_shift")),
        clean_text(card.get("goal_shift")),
        clean_text(card.get("state_pressure")),
        clean_text(card.get("mission_relevance")),
        clean_text(card.get("milestone_reason")),
    ]
    return " ".join([p for p in parts if p]).lower()


FEATURE_TOKEN_RE = re.compile(r"[a-z0-9']+|[一-鿿]+")


def card_feature_tokens(card: Dict[str, Any]) -> set:
    tokens = set()
    for tok in FEATURE_TOKEN_RE.findall(card_feature_text(card)):
        if re.search(r"[一-鿿]", tok) or len(tok) >= 3:
            tokens.add(tok)
    return tokens


def card_feature_tags(card: Dict[str, Any]) -> set:
    text = card_feature_text(card)
    tags = set()
    if clean_text(card.get("relation_shift")):
        tags.add("relation")
    if clean_text(card.get("goal_shift")):
        tags.add("goal")
    if clean_text(card.get("external_change")):
        tags.add("external")
    if clean_text(card.get("state_pressure")):
        tags.add("pressure")
    if clean_text(card.get("mission_relevance")):
        tags.add("mission")

    keyword_groups = {
        "intimacy": ["intimacy", "romantic", "romance", "kiss", "flirt", "sexual", "desire", "crush", "affair", "reunion", "reunited", "partner", "chemistry"],
        "boundary": ["boundary", "reject", "rejection", "refuse", "refusal", "decline", "ends the affair", "end the affair", "not part", "real life", "escape", "parenthesis", "adult responsibility", "clear"],
        "reveal": ["secret", "truth", "discover", "discovered", "family", "husband", "children", "hidden reality", "double life"],
        "domestic_reveal": ["family", "husband", "children", "hidden reality", "double life"],
        "worldview": ["worldview", "philosophy", "belief", "beliefs", "timeline", "timelines", "priorities", "values", "marriage"],
        "action": ["window", "lock", "locked", "access", "break-in", "break in", "investigator", "investigate", "school", "pick a window lock"],
        "support": ["comfort", "reassure", "reassurance", "confidant", "empathy", "empathetic", "validate", "crisis", "stabilizing", "stabilising"],
        "decision": ["decides", "decide", "chooses", "choose", "accepts", "accept", "agrees", "agree", "refuses", "refuse", "rejects", "reject"],
    }
    for tag, kws in keyword_groups.items():
        if any(kw in text for kw in kws):
            tags.add(tag)
    return tags


def card_text_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ta = card_feature_tokens(a)
    tb = card_feature_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def same_mini_arc(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    diff = abs(int(a.get("scene_order", 0)) - int(b.get("scene_order", 0)))
    tags_a = card_feature_tags(a)
    tags_b = card_feature_tags(b)
    union_tags = tags_a | tags_b
    sim = card_text_similarity(a, b)
    if diff <= 2 and sim >= 0.16:
        return True
    if diff <= 6 and "intimacy" in tags_a and "intimacy" in tags_b and not (union_tags & {"boundary", "reveal", "worldview", "action", "support"}):
        return True
    if diff <= 4 and (union_tags & {"boundary", "reveal"}) and sim >= 0.10:
        return True
    return False


def milestone_priority(card: Dict[str, Any]) -> Tuple[int, int, int, int]:
    richness = sum(1 for key in ("external_change", "relation_shift", "goal_shift", "state_pressure", "mission_relevance") if clean_text(card.get(key)))
    role_bonus = 2 if card.get("scene_role") == "foreground" else 1 if card.get("scene_role") == "active" else 0
    tags = card_feature_tags(card)
    effective = int(card.get("milestone_score", 0)) * 10 + richness * 2 + role_bonus
    if tags & {"boundary", "reveal"}:
        effective += 7
    if "domestic_reveal" in tags and "boundary" not in tags:
        effective += 4
    if tags & {"worldview", "action", "support"}:
        effective += 4
    if "decision" in tags and "intimacy" not in tags:
        effective += 2
    if {"support", "worldview"}.issubset(tags) and "intimacy" not in tags:
        effective += 8
    if "intimacy" in tags and not (tags & {"boundary", "reveal", "worldview", "action"}):
        effective -= 4
    return (effective, int(card.get("milestone_score", 0)), richness, role_bonus)


def novelty_tuple(card: Dict[str, Any], chosen: Sequence[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    if not chosen:
        return (0, 2, 1, 0)
    tags = card_feature_tags(card)
    chosen_tags = set()
    min_dist = 10**9
    same_arc_count = 0
    continuation_bonus = 0
    for existing in chosen:
        chosen_tags |= card_feature_tags(existing)
        diff = abs(int(card.get("scene_order", 0)) - int(existing.get("scene_order", 0)))
        min_dist = min(min_dist, diff)
        if same_mini_arc(card, existing):
            same_arc_count += 1
            if diff <= 1 and ("goal" in tags or "decision" in tags):
                continuation_bonus = 1
    uncovered = len(tags - chosen_tags)
    dist_bonus = 1 if min_dist >= 6 else 0
    return (continuation_bonus, min(2, uncovered), dist_bonus, -same_arc_count)


def selection_penalty(card: Dict[str, Any], chosen: Sequence[Dict[str, Any]], max_scene_order: int) -> int:
    tags = card_feature_tags(card)
    chosen_tag_sets = [card_feature_tags(x) for x in chosen]
    boundary_count = sum(1 for x in chosen_tag_sets if "boundary" in x and "reveal" not in x)
    intimacy_only_count = sum(1 for x in chosen_tag_sets if "intimacy" in x and not (x & {"boundary", "reveal", "worldview", "action"}))
    penalty = 0
    if "boundary" in tags and "reveal" not in tags and boundary_count >= 2:
        penalty += 4
    if "intimacy" in tags and not (tags & {"boundary", "reveal", "worldview", "action"}) and intimacy_only_count >= 3:
        penalty += 1
    role_text = clean_text(card.get("role_in_scene")).lower()
    if int(card.get("milestone_score", 0) or 0) <= 3 and ("observer" in role_text or "introduced" in role_text or "initial responder" in role_text):
        penalty += 2

    ratio = (max(1, int(card.get("scene_order", 0))) - 1) / max(1, max_scene_order - 1)
    feature_text = card_feature_text(card)
    setup_keywords = ["introduced", "foundational", "beginning of", "strategic voice", "change agent", "professional mission", "defines her character trajectory", "ideological transformation"]
    if ratio <= 0.45 and "mission" in tags and "goal" in tags and not (tags & {"boundary", "reveal", "action", "domestic_reveal"}):
        penalty += 2
    if ratio <= 0.45 and any(kw in feature_text for kw in setup_keywords):
        penalty += 2
    return -penalty


def should_skip_for_redundancy(card: Dict[str, Any], chosen: Sequence[Dict[str, Any]]) -> bool:
    card_tags = card_feature_tags(card)
    card_priority = milestone_priority(card)
    if int(card.get("milestone_score", 0) or 0) >= HIGH_MILESTONE_SCORE:
        return any(str(existing.get("scene_id")) == str(card.get("scene_id")) for existing in chosen)
    for existing in chosen:
        if str(existing.get("scene_id")) == str(card.get("scene_id")):
            return True
        if not same_mini_arc(card, existing):
            continue
        existing_tags = card_feature_tags(existing)
        diff = abs(int(card.get("scene_order", 0)) - int(existing.get("scene_order", 0)))
        existing_score = int(existing.get("milestone_score", 0) or 0)
        current_score = int(card.get("milestone_score", 0) or 0)
        if current_score >= 3 and existing_score >= 3 and diff <= 2 and (card_tags - existing_tags):
            continue
        if diff <= 1 and card_priority[0] >= milestone_priority(existing)[0] - 10:
            continue
        if diff <= 2:
            if "intimacy" in card_tags and "intimacy" in existing_tags and milestone_priority(existing) >= card_priority:
                return True
        if current_score <= 2 and (existing_tags & {"boundary", "reveal"}) and not (card_tags & {"boundary", "reveal"}) and milestone_priority(existing) >= card_priority:
            return True
        if (card_tags & {"reveal"}) and diff <= 4 and card_priority[0] >= milestone_priority(existing)[0] - 8:
            continue
        if current_score <= 2 and not (card_tags - existing_tags) and milestone_priority(existing) >= card_priority:
            return True
    return False


def select_milestones(cards: Sequence[Dict[str, Any]], max_nodes: Optional[int] = None) -> List[Dict[str, Any]]:
    if not cards:
        return []
    if max_nodes is None:
        max_nodes = dynamic_node_budget(cards)
    max_scene_order = max(int(c["scene_order"]) for c in cards)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for card in cards:
        by_bucket[temporal_bucket(int(card["scene_order"]), max_scene_order)].append(card)

    chosen: List[Dict[str, Any]] = []
    used_scene_ids = set()

    def candidate_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        novelty = novelty_tuple(row, chosen)
        penalty = selection_penalty(row, chosen, max_scene_order)
        priority = milestone_priority(row)
        return novelty + (penalty,) + priority + (-int(row.get("scene_order", 0)),)

    protected = sorted(
        [row for row in cards if int(row.get("milestone_score", 0) or 0) >= HIGH_MILESTONE_SCORE],
        key=lambda x: (int(x.get("scene_order", 0)), -milestone_priority(x)[0]),
    )
    for row in protected:
        if len(chosen) >= max_nodes:
            break
        if row["scene_id"] in used_scene_ids:
            continue
        chosen.append(row)
        used_scene_ids.add(row["scene_id"])

    for bucket in ("early", "middle", "late"):
        rows = list(by_bucket.get(bucket, []))
        added = 0
        while added < 2 and len(chosen) < max_nodes:
            best = None
            best_key = None
            for row in rows:
                if row["scene_id"] in used_scene_ids:
                    continue
                if should_skip_for_redundancy(row, chosen):
                    continue
                key = candidate_key(row)
                if best is None or key > best_key:
                    best = row
                    best_key = key
            if best is None:
                break
            chosen.append(best)
            used_scene_ids.add(best["scene_id"])
            added += 1

    all_rows = sorted(cards, key=lambda x: candidate_key(x), reverse=True)
    for row in all_rows:
        if len(chosen) >= max_nodes:
            break
        if row["scene_id"] in used_scene_ids:
            continue
        if should_skip_for_redundancy(row, chosen):
            continue
        chosen.append(row)
        used_scene_ids.add(row["scene_id"])

    min_nodes = min(max_nodes, max(9, max_nodes - 1))
    if len(chosen) < min_nodes:
        fallback_rows = sorted(cards, key=lambda x: (milestone_priority(x), -int(x["scene_order"])), reverse=True)
        for row in fallback_rows:
            if len(chosen) >= min_nodes:
                break
            if row["scene_id"] in used_scene_ids:
                continue
            chosen.append(row)
            used_scene_ids.add(row["scene_id"])

    chosen = sorted(chosen, key=lambda x: x["scene_order"])
    return chosen[:max_nodes]

def pairwise_milestone_judge_prompt(language: str, character_name: str, card_a: Dict[str, Any], card_b: Dict[str, Any]) -> List[Dict[str, str]]:
    system = PAIRWISE_MILESTONE_JUDGE_PROMPT_ZH if language == "zh" else PAIRWISE_MILESTONE_JUDGE_PROMPT_EN
    user = (
        f"Focal character: {character_name}\n"
        f"Candidate A:\n{json.dumps(card_a, ensure_ascii=False, indent=2)}\n"
        f"Candidate B:\n{json.dumps(card_b, ensure_ascii=False, indent=2)}"
    )
    return prompt_messages(system, user)


def pairwise_better_than(current: Dict[str, Any], challenger: Dict[str, Any]) -> bool:
    cur_tags = card_feature_tags(current)
    new_tags = card_feature_tags(challenger)
    cur_priority = milestone_priority(current)[0]
    new_priority = milestone_priority(challenger)[0]
    role_text = clean_text(current.get("role_in_scene")).lower()
    current_is_weak_observer = int(current.get("milestone_score", 0) or 0) <= 3 and ("observer" in role_text or "introduced" in role_text or "initial responder" in role_text)

    if "domestic_reveal" in new_tags and "domestic_reveal" not in cur_tags:
        return True
    if "action" in new_tags and "action" not in cur_tags and abs(int(current.get("scene_order", 0)) - int(challenger.get("scene_order", 0))) <= 12:
        return True
    if {"support", "worldview"}.issubset(new_tags) and not {"support", "worldview"}.issubset(cur_tags):
        return True
    if current_is_weak_observer and new_priority >= cur_priority - 4:
        return True
    return False


def refine_milestones_with_pairwise(
    llm: LLMClient,
    language: str,
    character_name: str,
    milestones: Sequence[Dict[str, Any]],
    cards: Sequence[Dict[str, Any]],
    max_nodes: int = 9,
    max_pairwise_checks: int = 8,
) -> List[Dict[str, Any]]:
    chosen = [dict(x) for x in milestones]
    chosen_ids = {str(x.get("scene_id")) for x in chosen}
    excluded = [dict(x) for x in cards if str(x.get("scene_id")) not in chosen_ids]
    pairwise_checks = 0

    def conflict_targets(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for item in chosen:
            diff = abs(int(candidate.get("scene_order", 0)) - int(item.get("scene_order", 0)))
            if same_mini_arc(candidate, item) or diff <= 12:
                rows.append(item)
        rows.sort(key=lambda x: (
            0 if same_mini_arc(candidate, x) else 1,
            abs(int(candidate.get("scene_order", 0)) - int(x.get("scene_order", 0))),
            -milestone_priority(x)[0],
        ))
        return rows

    excluded_sorted = sorted(excluded, key=lambda x: (milestone_priority(x), -int(x.get("scene_order", 0))), reverse=True)
    for candidate in excluded_sorted:
        if pairwise_checks >= max_pairwise_checks:
            break
        if int(candidate.get("milestone_score", 0) or 0) >= HIGH_MILESTONE_SCORE and len(chosen) < max_nodes:
            if not any(str(x.get("scene_id")) == str(candidate.get("scene_id")) for x in chosen):
                chosen.append(candidate)
                chosen = sorted(chosen, key=lambda x: int(x.get("scene_order", 0)))
                chosen_ids = {str(x.get("scene_id")) for x in chosen}
            continue
        targets = conflict_targets(candidate)
        if not targets:
            continue
        target = targets[0]
        if int(target.get("milestone_score", 0) or 0) >= HIGH_MILESTONE_SCORE and int(candidate.get("milestone_score", 0) or 0) < HIGH_MILESTONE_SCORE:
            continue
        if not pairwise_better_than(target, candidate):
            continue
        raw = llm_json(llm, pairwise_milestone_judge_prompt(language, character_name, target, candidate), max_tokens=400)
        pairwise_checks += 1
        decision = clean_text(raw.get("decision")).lower()
        if decision == "b":
            chosen = [candidate if str(x.get("scene_id")) == str(target.get("scene_id")) else x for x in chosen]
        elif decision == "both" and len(chosen) < max_nodes:
            if not any(str(x.get("scene_id")) == str(candidate.get("scene_id")) for x in chosen):
                chosen.append(candidate)
        chosen_ids = {str(x.get("scene_id")) for x in chosen}

    chosen = sorted(chosen, key=lambda x: int(x.get("scene_order", 0)))
    deduped = []
    seen = set()
    for row in chosen:
        sid = str(row.get("scene_id"))
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(row)
    return deduped[:max_nodes]


def node_render_prompt(language: str, character_name: str, card: Dict[str, Any], scene: SceneRecord) -> List[Dict[str, str]]:
    system = NODE_RENDER_PROMPT_ZH if language == "zh" else NODE_RENDER_PROMPT_EN
    user = (
        f"Focal character: {character_name}\n"
        f"Scene-role card:\n{json.dumps(card, ensure_ascii=False, indent=2)}\n"
        f"Scene text:\n{scene.content[:6000]}"
    )
    return prompt_messages(system, user)


def deterministic_node_render(language: str, character_name: str, card: Dict[str, Any], scene: SceneRecord) -> Dict[str, Any]:
    role_in_context = (
        clean_text(card.get("role_in_scene"))
        or (
            f"{character_name} is materially present in this scene."
            if language == "en"
            else f"{character_name}在这场戏中有实质性参与。"
        )
    )
    salient_development = (
        clean_text(card.get("external_change"))
        or clean_text(card.get("goal_shift"))
        or clean_text(card.get("relation_shift"))
        or clean_text(card.get("state_pressure"))
        or clean_text(card.get("mission_relevance"))
        or clean_text(card.get("milestone_reason"))
        or (
            f"{character_name}'s trajectory is visibly advanced in this scene."
            if language == "en"
            else f"{character_name}的叙事轨迹在这场戏中得到推进。"
        )
    )
    return {
        "role_in_context": role_in_context,
        "salient_development": salient_development,
        "goal_state": clean_text(card.get("goal_shift")) or None,
        "resulting_state": clean_text(card.get("external_change")) or None,
        "unresolved_issue": clean_text(card.get("state_pressure")) or None,
        "evidence_quotes": list(card.get("evidence_quotes") or [])[:3],
    }


def render_timeline_nodes(llm: LLMClient, language: str, character: Dict[str, Any], milestones: Sequence[Dict[str, Any]], scenes: Sequence[SceneRecord]) -> Tuple[str, List[Dict[str, Any]]]:
    scene_map = {s.scene_id: s for s in scenes}
    nodes = []
    for idx, card in enumerate(milestones, start=1):
        scene = scene_map[str(card["scene_id"])]
        try:
            raw = llm_json(llm, node_render_prompt(language, character["character_name"], card, scene), max_tokens=900)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "stage": "timeline_node_fallback",
                        "character": character["character_name"],
                        "scene_id": str(card.get("scene_id")),
                        "scene_order": int(card.get("scene_order", 0) or 0),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            raw = deterministic_node_render(language, character["character_name"], card, scene)
        evidence_quotes = []
        for quote in raw.get("evidence_quotes", []) or []:
            quote = clean_text(quote)
            if quote and quote in scene.content and quote not in evidence_quotes:
                evidence_quotes.append(quote)
        if not evidence_quotes:
            evidence_quotes = list(card.get("evidence_quotes") or [])[:2]
        nodes.append({
            "timeline_node_id": stable_id(character["character_name"], str(card["scene_id"]), str(idx), prefix="v6tu"),
            "document_id": f"scene_{int(str(card['scene_id'])) + 1}_milestone_{idx}",
            "scene_id": str(card["scene_id"]),
            "scene_order": int(card["scene_order"]),
            "scene_title": scene.scene_title,
            "scene_summary": clean_text(card.get("milestone_reason") or card.get("role_in_scene")),
            "importance": "core" if int(card.get("milestone_score", 0)) >= 4 else "supporting",
            "role_in_context": clean_text(raw.get("role_in_context")) or clean_text(card.get("role_in_scene")),
            "salient_development": clean_text(raw.get("salient_development")) or clean_text(card.get("external_change") or card.get("goal_shift") or card.get("state_pressure")),
            "goal_state": clean_text(raw.get("goal_state")) or clean_text(card.get("goal_shift")) or None,
            "resulting_state": clean_text(raw.get("resulting_state")) or clean_text(card.get("external_change")) or None,
            "unresolved_issue": clean_text(raw.get("unresolved_issue")) or clean_text(card.get("state_pressure")) or None,
            "related_event_ids": [],
            "related_episode_ids": [],
            "evidence_quotes": evidence_quotes[:4],
            "auxiliary": {
                "relation_updates": [clean_text(card.get("relation_shift"))] if clean_text(card.get("relation_shift")) else [],
                "status_updates": [],
                "persona_anchor": "",
            },
        })
    summary = f"Milestone-based timeline for {character['character_name']} with {len(nodes)} grounded nodes."
    if language == "zh":
        summary = f"基于 milestone 的 {character['character_name']} 时间线，共 {len(nodes)} 个节点。"
    return summary, nodes


def convert_timeline_scene_ids_for_public(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(payload, ensure_ascii=False))
    for item in out.get("focal_character_timelines", []) or []:
        for node in item.get("timeline_nodes", []) or []:
            if str(node.get("scene_id", "")).isdigit():
                public_id = str(int(str(node["scene_id"])) + 1)
                node["scene_id"] = public_id
                node["document_id"] = re.sub(r"^scene_\d+", f"scene_{public_id}", clean_text(node.get("document_id")))
    return out


def flush_partial_outputs(
    output_dir: Path,
    role_scene_cards: Dict[str, Any],
    character_segments: Dict[str, Any],
    character_milestones: Dict[str, Any],
    timeline_payload: Dict[str, Any],
    arc_payload: Dict[str, Any],
) -> None:
    (output_dir / "role_scene_cards.json").write_text(json.dumps(role_scene_cards, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "character_segments.json").write_text(json.dumps(character_segments, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "character_milestones.json").write_text(json.dumps(character_milestones, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "pred_task_1_character_timelines.json").write_text(json.dumps(timeline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "pred_task_1_cross_scene_arcs.json").write_text(json.dumps(arc_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_workflow_v65(movie_dir: Path, output_dir: Path) -> Dict[str, Any]:
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
        "task_version": "workflow_v65_recall_first_prototype",
        "focal_character_timelines": [],
        "build_summary": {},
    }
    arc_payload = {
        "movie_id": movie_dir.name,
        "language": language,
        "task_name": "Story Dynamics Structuring",
        "task_version": "workflow_v65_recall_first_prototype",
        "cross_scene_arcs": [],
        "build_summary": {},
    }
    role_scene_cards = {}
    character_segments = {}
    character_milestones = {}

    for character in selected_characters:
        print(json.dumps({"stage": "character_start", "character": character["character_name"]}, ensure_ascii=False), flush=True)
        cards = build_role_scene_cards(llm, scenes, character, language, movie_dir=movie_dir)
        role_scene_cards[character["character_name"]] = cards
        print(json.dumps({"stage": "role_scene_cards_ready", "character": character["character_name"], "scene_card_count": len(cards)}, ensure_ascii=False), flush=True)
        segments = build_segments(llm, language, character["character_name"], cards)
        character_segments[character["character_name"]] = segments
        max_nodes = dynamic_node_budget(cards)
        milestones = select_milestones(cards, max_nodes=max_nodes)
        milestones = refine_milestones_with_pairwise(llm, language, character["character_name"], milestones, cards, max_nodes=max_nodes)
        character_milestones[character["character_name"]] = milestones
        print(json.dumps({"stage": "milestones_ready", "character": character["character_name"], "max_nodes": max_nodes, "milestone_count": len(milestones), "scene_orders": [m["scene_order"] for m in milestones]}, ensure_ascii=False), flush=True)
        timeline_summary, nodes = render_timeline_nodes(llm, language, character, milestones, scenes)
        timeline_payload["focal_character_timelines"].append({
            "character_name": character["character_name"],
            "aliases": character.get("aliases", []) or [],
            "selection_reason": "provided externally via benchmark focal-role list" if language == "en" else "由 benchmark 焦点角色列表提供",
            "task3_relevance": "Target character provided by the benchmark focal-role list and rendered through a recall-first milestone pipeline." if language == "en" else "由 benchmark 焦点角色列表指定目标角色，并通过 recall-first milestone 流程进行渲染。",
            "timeline_nodes": nodes,
            "timeline_summary": timeline_summary,
        })
        if nodes:
            arc_raw = llm_json(llm, arc_prompt(language, character["character_name"], [{
                "timeline_node_id": n["timeline_node_id"],
                "scene_id": n["scene_id"],
                "scene_order": n["scene_order"],
                "scene_title": n["scene_title"],
                "salient_development": n["salient_development"],
                "goal_state": n.get("goal_state"),
                "resulting_state": n.get("resulting_state"),
                "unresolved_issue": n.get("unresolved_issue"),
            } for n in nodes]), max_tokens=1800)
            node_id_to_scene_id = {n["timeline_node_id"]: n["scene_id"] for n in nodes}
            for item in arc_raw.get("arcs", []) or []:
                linked_ids = [clean_text(x) for x in (item.get("linked_timeline_node_ids") or []) if clean_text(x) in node_id_to_scene_id]
                linked_ids = list(dict.fromkeys(linked_ids))
                if len(linked_ids) < 2:
                    continue
                arc_payload["cross_scene_arcs"].append({
                    "arc_id": stable_id(character["character_name"], clean_text(item.get("title")), prefix="v6arc"),
                    "character_name": character["character_name"],
                    "title": clean_text(item.get("title")),
                    "arc_focus": clean_text(item.get("arc_focus")) or "mixed",
                    "linked_timeline_node_ids": linked_ids,
                    "arc_summary": clean_text(item.get("arc_summary")),
                    "start_state": clean_text(item.get("start_state")) or None,
                    "end_state": clean_text(item.get("end_state")) or None,
                    "unresolved_issue": clean_text(item.get("unresolved_issue")) or None,
                })
        flush_partial_outputs(output_dir, role_scene_cards, character_segments, character_milestones, timeline_payload, arc_payload)
        print(json.dumps({"stage": "character_complete", "character": character["character_name"], "node_count": len(nodes), "arc_count": len([x for x in arc_payload["cross_scene_arcs"] if x["character_name"] == character["character_name"]])}, ensure_ascii=False), flush=True)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    movie_dir = Path(args.movie_dir)
    output_dir = Path(args.output_dir)
    run_workflow_v65(movie_dir, output_dir)
    if args.evaluate:
        summary = evaluate_v5(movie_dir, output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
