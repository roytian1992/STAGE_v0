#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import (
    DEFAULT_EMBED_API_KEY,
    DEFAULT_EMBED_BASE_URL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLMClient,
    OpenAICompatEmbedder,
    SceneRecord,
    arc_prompt,
    clean_text,
    detect_language,
    extract_speaker_candidates,
    llm_json,
    load_scenes,
    normalize_name,
    run_bool_judge,
    stable_id,
)

PAREN_RE = re.compile(r"\s*\((?:O\.S\.|V\.O\.|OS|VO|CONT'D|OFF)\)\s*", re.I)
SLUG_RE = re.compile(r"^(?:[A-Z0-9'\" .-]{0,8})?(?:INT\.|EXT\.|INT/EXT\.|I/E\.|EST\.|OMITTED|TITLE:)", re.I)

BEAT_PROMPT_EN = """You are extracting beat-level narrative evidence for a simplified STAGE Task 1 from screenplay text.

You will receive one beat excerpt from a screenplay scene.
Extract only concrete, benchmark-style key developments that are explicitly supported inside this beat excerpt.
Do not mention later beats, earlier omitted material, or other scenes.
Do not blend multiple developments into one if they are clearly separate. Avoid mood-only or introspection-only updates unless they directly change a decision, relationship, or status.

Return JSON only in this format:
{
  \"beat_summary\": \"1-3 sentence grounded summary of only this beat\",
  \"characters_present\": [\"name\", \"...\"],
  \"character_updates\": [
    {
      \"character_name\": \"string\",
      \"importance\": \"core|supporting\",
      \"durable_change\": true,
      \"change_type\": \"decision|status|goal|relationship|setback|breakthrough|realization|other\",
      \"role_in_context\": \"short grounded role\",
      \"salient_development\": \"specific grounded development from this beat only\",
      \"goal_state\": \"string or null\",
      \"resulting_state\": \"string or null\",
      \"unresolved_issue\": \"string or null\",
      \"evidence_quotes\": [\"exact quote from the beat\", \"...\"]
    }
  ]
}

Rules:
- If a character is merely present but does not undergo a durable development in this beat, omit them from character_updates.
- If the same character undergoes two clearly distinct durable developments in this beat, output two separate character_updates entries for that same character.
- durable_change should be true only for changes that matter beyond the immediate moment.
- evidence_quotes must be exact substrings from the beat excerpt when possible.
- Prefer fewer, cleaner updates over broad or speculative ones. Prioritize concrete decisions, injuries, explicit agreements, conflicts, reconciliations, legal/addiction setbacks, title shots, fight outcomes, and status changes. Reject vague mood descriptions.
"""

BEAT_PROMPT_ZH = """你在从 screenplay 文本中为一个简化版 STAGE Task 1 抽取 beat 级关键发展。

你会收到某一场戏中的一个 beat 片段。
只能抽取这个 beat 片段内部有明确文本支持、并适合作为 benchmark 节点的关键发展。
不要提及后续 beat、前文省略内容或别的场景。
如果有多个发展点，不要混成一个泛化的大总结。不要抽“只是情绪”但没有明确外显变化的内容。

只输出 JSON，格式如下：
{
  \"beat_summary\": \"1-3句、只概括这个 beat 的贴地总结\",
  \"characters_present\": [\"名字\", \"...\"],
  \"character_updates\": [
    {
      \"character_name\": \"字符串\",
      \"importance\": \"core|supporting\",
      \"durable_change\": true,
      \"change_type\": \"decision|status|goal|relationship|setback|breakthrough|realization|other\",
      \"role_in_context\": \"简短且贴地的角色定位\",
      \"salient_development\": \"只来自这个 beat 的具体发展\",
      \"goal_state\": \"字符串或 null\",
      \"resulting_state\": \"字符串或 null\",
      \"unresolved_issue\": \"字符串或 null\",
      \"evidence_quotes\": [\"来自该 beat 的逐字引文\", \"...\"]
    }
  ]
}

要求：
- 如果角色只是出现、但没有产生值得保留的持续性发展，不要放进 character_updates。
- 如果同一角色在这个 beat 中发生了两个明显不同的持续性发展，请为同一角色输出两条独立的 character_updates。
- durable_change 只有在该变化对后续轨迹有持续意义时才写 true。
- evidence_quotes 尽量是该 beat 中可以直接回证的原文。
- 宁可少写，也不要写泛化空话或脑补。优先抽取具体决定、关系变化、伤病、明确协议、法律或成瘾挫折、比赛结果、title shot、身份或状态变化，而不是泛泛的情绪描写。
"""

TIMELINE_REFINE_PROMPT_EN = """You are refining pre-extracted beat-grounded timeline candidates for one focal character.

Each candidate is already tied to one scene_id and one beat_index.
Your job is to select and lightly rewrite the best candidates into a final timeline of benchmark-style key developments.
Aim to keep about 7-9 final nodes unless the candidate list is clearly much smaller.

Rules:
- Keep only candidates that reflect durable, externally verifiable character development.
- Prefer a temporally balanced timeline: include meaningful early, middle, and late developments when the candidate list supports them.
- Do not over-concentrate on the final act or pick many adjacent high-intensity scenes while skipping the character's setup and mid-story turning points.
- Prefer one strong node per scene unless multiple beat_index values in the same scene capture clearly different durable developments.
- Do not invent new scene_ids or beat_index values.
- Keep the writing concrete, conservative, externally verifiable, and benchmark-ready. Avoid mood-only framing.
- Stay close to the supplied evidence and candidate wording.

Return JSON only:
{
  "timeline_summary": "short summary",
  "final_nodes": [
    {
      "scene_id": "string",
      "beat_index": 0,
      "importance": "core|supporting",
      "role_in_context": "string",
      "salient_development": "string",
      "goal_state": "string or null",
      "resulting_state": "string or null",
      "unresolved_issue": "string or null"
    }
  ]
}
"""

TIMELINE_REFINE_PROMPT_ZH = """你在为某个焦点角色精修已经抽取好的 beat-grounded 时间线候选节点。

每个候选节点都已经绑定到一个 scene_id 和一个 beat_index。
你的任务是选择并轻度改写最好的候选节点，形成一条 benchmark 风格的关键发展时间线。
尽量把 final_nodes 控制在 7-9 个左右，除非候选本身就明显更少。

要求：
- 只保留真正体现持续性、而且可被外显验证的角色发展节点。
- 时间线上要尽量覆盖角色的前段、中段、后段发展；如果候选里有对应证据，不要把节点过度堆在结尾高潮段。
- 不要连续选太多相邻的高强度后段场景，却漏掉角色建立、关系转折或中段关键变化。
- 优先一场戏保留一个最强节点；只有当同一 scene_id 下不同 beat_index 代表了明显不同且都重要的持续发展时，才保留多个。
- 不要发明新的 scene_id 或 beat_index。
- 表达要具体、保守、可评测，不要写纯情绪化概括。
- 尽量贴近给定 evidence 和候选内容。

只输出 JSON：
{
  "timeline_summary": "简短总结",
  "final_nodes": [
    {
      "scene_id": "字符串",
      "beat_index": 0,
      "importance": "core|supporting",
      "role_in_context": "字符串",
      "salient_development": "字符串",
      "goal_state": "字符串或 null",
      "resulting_state": "字符串或 null",
      "unresolved_issue": "字符串或 null"
    }
  ]
}
"""

NODE_GROUND_PROMPT_EN = """You are grounding one final Task 1 timeline node against its exact beat text.

Only use the supplied beat excerpt.
Do not import information from other beats or scenes.
Keep the rewrite specific and conservative.

Return JSON only:
{
  \"role_in_context\": \"string\",
  \"salient_development\": \"string\",
  \"goal_state\": \"string or null\",
  \"resulting_state\": \"string or null\",
  \"unresolved_issue\": \"string or null\",
  \"evidence_quotes\": [\"exact quote\", \"...\"]
}
"""

NODE_GROUND_PROMPT_ZH = """你在根据某个精确 beat 文本校正一个最终 Task 1 时间线节点。

只能使用给定 beat 片段。
不要引入别的 beat 或别的场景的信息。
改写时要具体、保守、可回证。

只输出 JSON：
{
  \"role_in_context\": \"字符串\",
  \"salient_development\": \"字符串\",
  \"goal_state\": \"字符串或 null\",
  \"resulting_state\": \"字符串或 null\",
  \"unresolved_issue\": \"字符串或 null\",
  \"evidence_quotes\": [\"逐字引文\", \"...\"]
}
"""

DEV_JUDGE_PROMPT_EN = "You are judging whether two simplified Task 1 nodes for the same character and same screenplay scene are development-compatible. Be fairly permissive: if they capture the same broad externally verifiable change family, output TRUE. Allow paraphrase, abstraction mismatch, and partial detail mismatch. Output TRUE or FALSE only."
STATE_JUDGE_PROMPT_EN = "You are judging whether two simplified Task 1 nodes for the same character and same screenplay scene imply a compatible state transition. Be permissive: missing details are acceptable unless the predicted state is clearly contradictory. Output TRUE or FALSE only."
PRED_TRANSITION_COHERENCE_PROMPT_EN = "You are judging whether two consecutive predicted Task 1 nodes for the same character form a coherent narrative transition. Use the node text and the supplied screenplay scene evidence. Be reasonably permissive about skipped intermediate events, but output FALSE if the later node contradicts the earlier node, ignores an implausible leap in the character trajectory, or is not well supported by the supplied scene evidence. Output TRUE or FALSE only."
PRED_TRANSITION_COHERENCE_PROMPT_ZH = "你在评估同一角色两个相邻预测节点之间是否构成连贯的叙事转移。请结合节点内容和给定的剧本场景证据判断。对中间省略的发展可以适度宽松，但如果后一个节点与前一个节点明显矛盾、人物轨迹跳跃过大且不合理、或与给定场景证据不相符，则输出 FALSE。只输出 TRUE 或 FALSE。"
FACT_TYPES = {"decision", "status", "relationship", "goal", "outcome", "pressure", "other"}
FACT_PHASES = {"early", "middle", "late", "multi"}

TIMELINE_FACT_PROMPT_EN = """You are decomposing one Task 1 character timeline into atomic narrative facts.

Extract a compact set of benchmark-style facts from the full timeline, not one fact per node by default.
Merge redundant nodes when they express the same durable development.
Prefer facts that matter for macro narrative coverage.

Return JSON only:
{
  "facts": [
    {
      "fact_id": "F1",
      "fact_type": "decision|status|relationship|goal|outcome|pressure|other",
      "phase": "early|middle|late|multi",
      "fact_text": "one concrete durable narrative fact",
      "scene_refs": ["scene_id", "..."]
    }
  ]
}
"""

TIMELINE_FACT_PROMPT_ZH = """你在把一个 Task 1 角色时间线拆成原子级叙事实事。

目标是提炼出一组适合做宏观覆盖评测的 benchmark 风格事实，而不是机械地一节点对应一事实。
如果多个节点其实在表达同一条持续发展，可以合并。
优先保留真正重要、能够代表角色主线变化的事实。

只输出 JSON：
{
  "facts": [
    {
      "fact_id": "F1",
      "fact_type": "decision|status|relationship|goal|outcome|pressure|other",
      "phase": "early|middle|late|multi",
      "fact_text": "一句具体、可核对的持续性叙事实事",
      "scene_refs": ["scene_id", "..."]
    }
  ]
}
"""

FACT_SUPPORT_PROMPT_EN = """You are judging macro narrative fact coverage for Task 1.

You will receive:
- a list of atomic facts for one character
- the full reference timeline for the same character

Judge each fact against the full reference timeline, not against a single node.
Be fairly permissive:
- one reference node may cover multiple facts
- multiple reference nodes may jointly support one fact
- paraphrase and granularity mismatch are acceptable

Mark a fact as unsupported only when the reference timeline clearly does not cover that durable development.

Return JSON only:
{
  "supported_fact_ids": ["F1", "F3"],
  "unsupported_fact_ids": ["F2"]
}
"""

FACT_SUPPORT_PROMPT_ZH = """你在评估 Task 1 的宏观叙事实事覆盖情况。

你会收到：
- 某个角色的一组原子事实
- 同一角色的完整参考时间线

判断时要基于整条参考时间线，而不是强行逐节点一一对齐。
要相对宽松：
- 一个参考节点可以同时覆盖多个事实
- 多个参考节点也可以共同支持一个事实
- 改写、概括层级不同都可以接受

只有在参考时间线明显没有覆盖该持续性发展时，才判为 unsupported。

只输出 JSON：
{
  "supported_fact_ids": ["F1", "F3"],
  "unsupported_fact_ids": ["F2"]
}
"""


def prompt_messages(system: str, user: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def strip_speaker_variant(name: str) -> str:
    text = clean_text(name)
    text = PAREN_RE.sub(" ", text)
    return clean_text(text)


def is_slug_line(line: str) -> bool:
    raw = line.strip()
    if not raw:
        return False
    compact = raw.strip(" .:-")
    if len(compact) < 6:
        return False
    alpha_count = sum(1 for ch in compact if ch.isalpha())
    if alpha_count == 0:
        return False
    upper_ratio = sum(1 for ch in compact if ch.isupper()) / alpha_count
    return upper_ratio > 0.7 and bool(SLUG_RE.match(compact))


def split_scene_into_beats(scene: SceneRecord) -> List[Dict[str, Any]]:
    lines = [line.rstrip() for line in scene.content.splitlines()]
    beats: List[Dict[str, Any]] = []
    current: List[str] = []
    current_heading = scene.scene_title

    def flush() -> None:
        nonlocal current, current_heading
        text = "\n".join(current).strip()
        if not text:
            current = []
            return
        beats.append(
            {
                "scene_id": scene.scene_id,
                "scene_order": scene.scene_order,
                "scene_title": scene.scene_title,
                "beat_index": len(beats),
                "beat_heading": current_heading,
                "beat_text": text,
            }
        )
        current = []

    def split_long_beat(beat: Dict[str, Any]) -> List[Dict[str, Any]]:
        beat_lines = beat["beat_text"].splitlines()
        if len(beat_lines) <= 120 and len(beat["beat_text"]) <= 8000:
            return [beat]
        chunks: List[Dict[str, Any]] = []
        start = 0
        while start < len(beat_lines):
            tentative_end = min(len(beat_lines), start + 72)
            end = tentative_end
            for j in range(tentative_end, max(start + 40, tentative_end - 12), -1):
                if j - 1 < len(beat_lines) and not clean_text(beat_lines[j - 1]):
                    end = j
                    break
            chunk_text = "\n".join(beat_lines[start:end]).strip()
            if chunk_text:
                chunk = dict(beat)
                chunk["beat_text"] = chunk_text
                chunks.append(chunk)
            if end >= len(beat_lines):
                break
            start = max(start + 1, end - 10)
        return chunks or [beat]

    for line in lines:
        if is_slug_line(line):
            if len("\n".join(current).strip()) >= 160:
                flush()
            current_heading = clean_text(line)
            current.append(line)
            continue
        current.append(line)
        if len(current) >= 80 and not clean_text(line):
            flush()
    flush()

    if not beats:
        beats = [{"scene_id": scene.scene_id, "scene_order": scene.scene_order, "scene_title": scene.scene_title, "beat_index": 0, "beat_heading": scene.scene_title, "beat_text": scene.content.strip()}]

    merged: List[Dict[str, Any]] = []
    for beat in beats:
        if merged and len(beat["beat_text"]) < 180:
            merged[-1]["beat_text"] = clean_text(merged[-1]["beat_text"] + "\n" + beat["beat_text"])
            continue
        merged.append(dict(beat))

    refined: List[Dict[str, Any]] = []
    for beat in merged:
        refined.extend(split_long_beat(beat))
    for idx, beat in enumerate(refined):
        beat["beat_index"] = idx
    return refined


def beat_prompt(language: str, beat: Dict[str, Any]) -> List[Dict[str, str]]:
    system = BEAT_PROMPT_ZH if language == "zh" else BEAT_PROMPT_EN
    user = (
        f"Scene title: {beat['scene_title']}\n"
        f"Beat heading: {beat.get('beat_heading') or ''}\n"
        f"Beat index: {beat['beat_index']}\n"
        f"Beat text:\n{beat['beat_text'][:7000]}"
    )
    return prompt_messages(system, user)


def normalize_beat_summary(raw: Dict[str, Any], beat: Dict[str, Any]) -> Dict[str, Any]:
    beat_text = beat["beat_text"]
    updates = []
    for item in raw.get("character_updates", []) or []:
        name = strip_speaker_variant(item.get("character_name"))
        if not name:
            continue
        evidence_quotes = []
        for quote in item.get("evidence_quotes", []) or []:
            quote = clean_text(quote)
            if quote and quote in beat_text and quote not in evidence_quotes:
                evidence_quotes.append(quote)
        updates.append(
            {
                "character_name": name,
                "importance": clean_text(item.get("importance")) or "supporting",
                "durable_change": bool(item.get("durable_change")),
                "change_type": clean_text(item.get("change_type")) or "other",
                "role_in_context": clean_text(item.get("role_in_context")),
                "salient_development": clean_text(item.get("salient_development")),
                "goal_state": clean_text(item.get("goal_state")) or None,
                "resulting_state": clean_text(item.get("resulting_state")) or None,
                "unresolved_issue": clean_text(item.get("unresolved_issue")) or None,
                "evidence_quotes": evidence_quotes[:3],
            }
        )
    return {
        "scene_id": beat["scene_id"],
        "scene_order": beat["scene_order"],
        "scene_title": beat["scene_title"],
        "beat_index": beat["beat_index"],
        "beat_heading": beat.get("beat_heading"),
        "beat_text": beat_text,
        "beat_summary": clean_text(raw.get("beat_summary")),
        "characters_present": [strip_speaker_variant(x) for x in (raw.get("characters_present") or []) if strip_speaker_variant(x)],
        "character_updates": updates,
    }


def summarize_beats(llm: LLMClient, scenes: Sequence[SceneRecord], language: str) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for scene in scenes:
        for beat in split_scene_into_beats(scene):
            raw = llm_json(llm, beat_prompt(language, beat), max_tokens=1500)
            normalized = normalize_beat_summary(raw, beat)
            outputs.append(normalized)
            print(json.dumps({"stage": "beat_summarized", "scene_id": beat["scene_id"], "beat_index": beat["beat_index"], "scene_title": beat["scene_title"]}, ensure_ascii=False), flush=True)
    return outputs


def aggregate_candidates(speaker_candidates: Sequence[Dict[str, Any]], beat_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}

    def ensure(name: str) -> Dict[str, Any]:
        base = strip_speaker_variant(name)
        key = normalize_name(base)
        if key not in stats:
            stats[key] = {"character_name": base, "aliases": [], "speaker_utterance_count": 0, "speaker_scene_count": 0, "beat_presence_count": 0, "update_count": 0, "durable_update_count": 0, "core_update_count": 0, "sample_scene_titles": []}
        row = stats[key]
        if base and base not in row["aliases"]:
            row["aliases"].append(base)
        return row

    for item in speaker_candidates:
        row = ensure(item["name"])
        row["speaker_utterance_count"] = max(row["speaker_utterance_count"], int(item.get("utterance_count", 0) or 0))
        row["speaker_scene_count"] = max(row["speaker_scene_count"], int(item.get("scene_count", 0) or 0))
        for title in item.get("sample_scenes", [])[:3]:
            title = clean_text(title)
            if title and title not in row["sample_scene_titles"]:
                row["sample_scene_titles"].append(title)

    for beat in beat_summaries:
        for name in beat.get("characters_present", []) or []:
            row = ensure(name)
            row["beat_presence_count"] += 1
            title = beat["scene_title"]
            if title and title not in row["sample_scene_titles"] and len(row["sample_scene_titles"]) < 4:
                row["sample_scene_titles"].append(title)
        for update in beat.get("character_updates", []) or []:
            row = ensure(update["character_name"])
            row["update_count"] += 1
            if update.get("durable_change"):
                row["durable_update_count"] += 1
            if update.get("importance") == "core":
                row["core_update_count"] += 1

    rows = list(stats.values())
    rows.sort(key=lambda x: (x["durable_update_count"], x["core_update_count"], x["update_count"], x["beat_presence_count"], x["speaker_scene_count"], x["speaker_utterance_count"]), reverse=True)
    return rows[:48]


def select_focal_characters_prompt(language: str, candidates: Sequence[Dict[str, Any]], max_characters: int) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在从 script-only 抽取得到的角色候选里选择 STAGE Task 1 焦点角色。只输出 JSON。"
        user = (
            f"请选择最多 {max_characters} 个焦点角色。\n"
            "优先选择那些跨多场戏有持续发展、重复状态变化、关系变化或关键决策影响的人物。\n"
            "不要合并不同人物，也不要发明新名字。\n"
            "输出格式: {\"selected_characters\": [{\"character_name\":\"...\", \"aliases\":[\"...\"], \"selection_reason\":\"...\"}]}\n"
            f"候选列表:\n{json.dumps(list(candidates), ensure_ascii=False, indent=2)}"
        )
    else:
        system = "You are selecting focal characters for STAGE Task 1 from script-only character candidates. Output JSON only."
        user = (
            f"Choose up to {max_characters} focal characters.\n"
            "Prefer characters with durable cross-scene development, repeated state change, relationship change, or plot-shaping decisions.\n"
            "Do not merge different people and do not invent new names.\n"
            "Return format: {\"selected_characters\": [{\"character_name\":\"...\", \"aliases\":[\"...\"], \"selection_reason\":\"...\"}]}\n"
            f"Candidate list:\n{json.dumps(list(candidates), ensure_ascii=False, indent=2)}"
        )
    return prompt_messages(system, user)


def normalize_target_role_name(name: str) -> str:
    raw = clean_text(name).replace("_", " ")
    return clean_text(raw)


ROLE_NAME_OVERRIDES: Dict[str, Dict[str, str]] = {
    # Residual upstream role-label issue: the screenplay uses "Tracy", while the
    # external role bundle was emitted as "Emily Weaver".
    "en48f2465cf27f49a98a375692d4c2209b": {
        "Emily Weaver": "Tracy",
    },
}


def apply_role_name_override(movie_id: str, name: str) -> str:
    base = normalize_target_role_name(name)
    override_map = ROLE_NAME_OVERRIDES.get(movie_id, {})
    return clean_text(override_map.get(base, base))


def build_target_aliases(name: str) -> List[str]:
    base = normalize_target_role_name(name)
    aliases = [base]
    if " " in base:
        first = clean_text(base.split()[0])
        if first and first not in aliases:
            aliases.append(first)
    compact = clean_text(name)
    if compact and compact not in aliases:
        aliases.append(compact)
    return aliases


def load_target_characters(movie_dir: Path, explicit_file: Optional[str] = None) -> List[Dict[str, Any]]:
    candidates = []
    paths = []
    if explicit_file:
        paths.append(Path(explicit_file))
    paths.extend([
        movie_dir / 'task_3_in_script_character_role_play_single_turn.json',
        movie_dir / 'task_3_in_script_character_role_play_multi_turn.json',
        movie_dir / 'gold_character_timelines_v1.json',
        movie_dir / 'task_1_character_timelines.json',
    ])
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        roles = []
        if isinstance(data, dict):
            if isinstance(data.get('roles'), list):
                roles = data.get('roles', [])
            elif isinstance(data.get('focal_character_timelines'), list):
                roles = [item.get('character_name') for item in data.get('focal_character_timelines', []) if item.get('character_name')]
        for role in roles:
            base = apply_role_name_override(movie_dir.name, str(role))
            if not base:
                continue
            key = normalize_name(base)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                'character_name': base,
                'aliases': build_target_aliases(str(role)),
                'selection_reason': 'provided externally via benchmark focal-role list',
            })
        if candidates:
            break
    return candidates


def select_focal_characters(llm: LLMClient, language: str, candidates: Sequence[Dict[str, Any]], max_characters: int) -> List[Dict[str, Any]]:
    raw = llm_json(llm, select_focal_characters_prompt(language, candidates, max_characters), max_tokens=1200)
    by_name = {normalize_name(item["character_name"]): item for item in candidates}
    selected = []
    seen = set()
    for item in raw.get("selected_characters", []) or []:
        key = normalize_name(item.get("character_name"))
        if key not in by_name or key in seen:
            continue
        seen.add(key)
        base = by_name[key]
        aliases = []
        for alias in base.get("aliases", []) + (item.get("aliases") or []):
            alias = strip_speaker_variant(alias)
            if alias and alias not in aliases:
                aliases.append(alias)
        selected.append({"character_name": base["character_name"], "aliases": aliases[:6], "selection_reason": clean_text(item.get("selection_reason")) or "selected from script-only beat-grounded candidates"})
    return selected


def character_matches(update_name: str, aliases: Sequence[str]) -> bool:
    norm = normalize_name(strip_speaker_variant(update_name))
    alias_norms = {normalize_name(strip_speaker_variant(x)) for x in aliases if strip_speaker_variant(x)}
    return norm in alias_norms


def temporal_bucket(scene_order: int, max_scene_order: int) -> str:
    if max_scene_order <= 1:
        return "middle"
    ratio = (max(1, int(scene_order)) - 1) / max(1, max_scene_order - 1)
    if ratio < 0.34:
        return "early"
    if ratio < 0.67:
        return "middle"
    return "late"


def candidate_rank_score(item: Dict[str, Any]) -> int:
    type_bonus = {
        "status_win": 4,
        "injury_or_loss": 4,
        "career_move": 3,
        "legal_or_addiction_setback": 3,
        "explicit_decision": 3,
        "relationship_shift": 2,
        "family_conflict": 2,
        "reconciliation": 2,
        "other": 0,
    }
    ctype = clean_text(item.get("change_type")).lower()
    score = int(item.get("candidate_score", 0))
    score += type_bonus.get(ctype, 0)
    score += 1 if item.get("evidence_quotes") else 0
    score += 1 if clean_text(item.get("salient_development")) and len(clean_text(item.get("salient_development"))) > 40 else 0
    score += 1 if clean_text(item.get("goal_state")) else 0
    score += 1 if clean_text(item.get("resulting_state")) else 0
    return score


def prune_candidate_nodes(candidate_nodes: Sequence[Dict[str, Any]], max_candidates: int = 14, max_per_scene: int = 2) -> List[Dict[str, Any]]:
    if not candidate_nodes:
        return []
    max_scene_order = max(int(item.get("scene_order", 0) or 0) for item in candidate_nodes) or 1
    ranked = []
    for item in candidate_nodes:
        ctype = clean_text(item.get("change_type")).lower()
        if ctype == "other" and not item.get("evidence_quotes"):
            continue
        score = candidate_rank_score(item)
        bucket = temporal_bucket(int(item.get("scene_order", 0) or 0), max_scene_order)
        ranked.append((score, bucket, item))
    ranked.sort(key=lambda x: (x[0], x[2].get("scene_order", 0), x[2].get("beat_index", 0)), reverse=True)

    bucket_rows: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for score, bucket, item in ranked:
        bucket_rows[bucket].append((score, item))

    picked: List[Dict[str, Any]] = []
    seen_keys = set()
    per_scene = defaultdict(int)

    def try_add(item: Dict[str, Any]) -> bool:
        key = (str(item.get("scene_id")), int(item.get("beat_index", 0) or 0))
        scene_id = str(item.get("scene_id"))
        if key in seen_keys or per_scene[scene_id] >= max_per_scene:
            return False
        picked.append(item)
        seen_keys.add(key)
        per_scene[scene_id] += 1
        return True

    active_buckets = [b for b in ("early", "middle", "late") if bucket_rows.get(b)]
    base_quota = 3 if len(active_buckets) >= 3 else 2
    for bucket in ("early", "middle", "late"):
        quota = base_quota if bucket in active_buckets else 0
        added = 0
        for _, item in bucket_rows.get(bucket, []):
            if len(picked) >= max_candidates or added >= quota:
                break
            if try_add(item):
                added += 1

    for _, _, item in ranked:
        if len(picked) >= max_candidates:
            break
        try_add(item)

    return sorted(picked, key=lambda x: (x.get("scene_order", 0), x.get("beat_index", 0)))


def build_character_candidate_nodes(character: Dict[str, Any], beat_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aliases = list(dict.fromkeys([character["character_name"]] + [a for a in character.get("aliases", []) if a]))
    candidates: List[Dict[str, Any]] = []
    for beat in beat_summaries:
        for update in beat.get("character_updates", []) or []:
            if not character_matches(update["character_name"], aliases):
                continue
            if not update.get("durable_change") and update.get("importance") != "core":
                continue
            candidate = {
                "scene_id": beat["scene_id"],
                "scene_order": beat["scene_order"],
                "scene_title": beat["scene_title"],
                "beat_index": beat["beat_index"],
                "beat_heading": beat.get("beat_heading"),
                "importance": update.get("importance") or "supporting",
                "change_type": update.get("change_type") or "other",
                "role_in_context": update.get("role_in_context"),
                "salient_development": update.get("salient_development"),
                "goal_state": update.get("goal_state"),
                "resulting_state": update.get("resulting_state"),
                "unresolved_issue": update.get("unresolved_issue"),
                "evidence_quotes": list(update.get("evidence_quotes") or [])[:3],
                "beat_summary": beat.get("beat_summary"),
            }
            if not candidate["salient_development"]:
                continue
            score = 0
            score += 3 if update.get("durable_change") else 0
            score += 2 if candidate["importance"] == "core" else 0
            score += min(2, len(candidate["evidence_quotes"]))
            score += 1 if candidate.get("goal_state") else 0
            score += 1 if candidate.get("resulting_state") else 0
            candidate["candidate_score"] = score
            candidates.append(candidate)
    candidates.sort(key=lambda x: (x["scene_order"], x["beat_index"], -x["candidate_score"]))

    deduped: List[Dict[str, Any]] = []
    seen_keys = set()
    for item in candidates:
        key = (item["scene_id"], item["beat_index"], normalize_name(item["salient_development"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)
    return prune_candidate_nodes(deduped)


def timeline_refine_prompt(language: str, character: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = TIMELINE_REFINE_PROMPT_ZH if language == "zh" else TIMELINE_REFINE_PROMPT_EN
    user = f"Character: {character['character_name']}\nAliases: {json.dumps(character.get('aliases', []), ensure_ascii=False)}\nCandidate nodes:\n{json.dumps(list(candidates), ensure_ascii=False, indent=2)}"
    return prompt_messages(system, user)


def validate_nodes_v3(nodes: Sequence[Dict[str, Any]], candidates: Sequence[Dict[str, Any]], scenes: Sequence[SceneRecord]) -> List[Dict[str, Any]]:
    scene_map = {s.scene_id: s for s in scenes}
    cand_map = {(str(item["scene_id"]), int(item["beat_index"])): item for item in candidates}
    out: List[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        scene_id = clean_text(node.get("scene_id"))
        try:
            beat_index = int(node.get("beat_index", 0))
        except Exception:
            beat_index = 0
        key = (scene_id, beat_index)
        if scene_id not in scene_map or key not in cand_map or key in seen:
            continue
        base = cand_map[key]
        scene = scene_map[scene_id]
        cleaned = {
            "scene_id": scene.scene_id,
            "scene_order": scene.scene_order,
            "scene_title": scene.scene_title,
            "beat_index": beat_index,
            "beat_heading": base.get("beat_heading"),
            "importance": clean_text(node.get("importance")) or base.get("importance") or "core",
            "role_in_context": clean_text(node.get("role_in_context")) or base.get("role_in_context"),
            "salient_development": clean_text(node.get("salient_development")) or base.get("salient_development"),
            "goal_state": clean_text(node.get("goal_state")) or base.get("goal_state") or None,
            "resulting_state": clean_text(node.get("resulting_state")) or base.get("resulting_state") or None,
            "unresolved_issue": clean_text(node.get("unresolved_issue")) or base.get("unresolved_issue") or None,
            "evidence_quotes": list(base.get("evidence_quotes") or [])[:3],
            "beat_summary": base.get("beat_summary"),
            "candidate_score": int(base.get("candidate_score", 0) or 0),
            "change_type": clean_text(base.get("change_type")) or "other",
        }
        if not cleaned["salient_development"]:
            continue
        seen.add(key)
        out.append(cleaned)
    return sorted(out, key=lambda x: (x["scene_order"], x["beat_index"]))


def clean_node_from_candidate(candidate: Dict[str, Any], scene_map: Dict[str, SceneRecord]) -> Optional[Dict[str, Any]]:
    scene_id = str(candidate.get("scene_id"))
    if scene_id not in scene_map:
        return None
    scene = scene_map[scene_id]
    cleaned = {
        "scene_id": scene.scene_id,
        "scene_order": scene.scene_order,
        "scene_title": scene.scene_title,
        "beat_index": int(candidate.get("beat_index", 0) or 0),
        "beat_heading": candidate.get("beat_heading"),
        "importance": clean_text(candidate.get("importance")) or "core",
        "role_in_context": clean_text(candidate.get("role_in_context")),
        "salient_development": clean_text(candidate.get("salient_development")),
        "goal_state": clean_text(candidate.get("goal_state")) or None,
        "resulting_state": clean_text(candidate.get("resulting_state")) or None,
        "unresolved_issue": clean_text(candidate.get("unresolved_issue")) or None,
        "evidence_quotes": list(candidate.get("evidence_quotes") or [])[:3],
        "beat_summary": candidate.get("beat_summary"),
        "candidate_score": int(candidate.get("candidate_score", 0) or 0),
        "change_type": clean_text(candidate.get("change_type")) or "other",
    }
    if not cleaned["salient_development"]:
        return None
    return cleaned


def repair_timeline_nodes(final_nodes: Sequence[Dict[str, Any]], candidates: Sequence[Dict[str, Any]], scenes: Sequence[SceneRecord], target_min: int = 7, target_max: int = 9) -> List[Dict[str, Any]]:
    if not candidates:
        return list(final_nodes)
    scene_map = {s.scene_id: s for s in scenes}
    max_scene_order = max(int(c.get("scene_order", 0) or 0) for c in candidates) or 1
    ranked_candidates = sorted(candidates, key=lambda x: (candidate_rank_score(x), x.get("scene_order", 0), x.get("beat_index", 0)), reverse=True)

    repaired = [dict(x) for x in final_nodes]
    seen_keys = {(str(x.get("scene_id")), int(x.get("beat_index", 0) or 0)) for x in repaired}

    def bucket_of_node(node: Dict[str, Any]) -> str:
        return temporal_bucket(int(node.get("scene_order", 0) or 0), max_scene_order)

    def add_candidate(candidate: Dict[str, Any]) -> bool:
        key = (str(candidate.get("scene_id")), int(candidate.get("beat_index", 0) or 0))
        if key in seen_keys:
            return False
        cleaned = clean_node_from_candidate(candidate, scene_map)
        if not cleaned:
            return False
        repaired.append(cleaned)
        seen_keys.add(key)
        return True

    candidate_buckets = {temporal_bucket(int(c.get("scene_order", 0) or 0), max_scene_order) for c in ranked_candidates}
    # Force a small deterministic skeleton anchor from each temporal bucket before free-form filling.
    for bucket in ("early", "middle", "late"):
        if bucket not in candidate_buckets:
            continue
        for candidate in ranked_candidates:
            if temporal_bucket(int(candidate.get("scene_order", 0) or 0), max_scene_order) != bucket:
                continue
            if add_candidate(candidate):
                break

    while len(repaired) < target_min:
        bucket_counts = defaultdict(int)
        for node in repaired:
            bucket_counts[bucket_of_node(node)] += 1
        chosen = None
        chosen_key = None
        for candidate in ranked_candidates:
            key = (str(candidate.get("scene_id")), int(candidate.get("beat_index", 0) or 0))
            if key in seen_keys:
                continue
            bucket = temporal_bucket(int(candidate.get("scene_order", 0) or 0), max_scene_order)
            scene_dup = any(str(n.get("scene_id")) == str(candidate.get("scene_id")) for n in repaired)
            score_key = (
                2 if bucket_counts[bucket] == 0 else 1 if bucket_counts[bucket] == 1 else 0,
                0 if scene_dup else 1,
                candidate_rank_score(candidate),
                -int(candidate.get("scene_order", 0) or 0),
            )
            if chosen is None or score_key > chosen_key:
                chosen = candidate
                chosen_key = score_key
        if chosen is None or not add_candidate(chosen):
            break

    while len(repaired) > target_max:
        bucket_counts = defaultdict(int)
        scene_counts = defaultdict(int)
        for node in repaired:
            bucket_counts[bucket_of_node(node)] += 1
            scene_counts[str(node.get("scene_id"))] += 1
        removable_idx = None
        removable_key = None
        for idx, node in enumerate(repaired):
            bucket = bucket_of_node(node)
            if bucket_counts[bucket] <= 1 and len(bucket_counts) > 1:
                continue
            priority = int(node.get("candidate_score", 0) or 0)
            remove_key = (
                1 if scene_counts[str(node.get("scene_id"))] > 1 else 0,
                bucket_counts[bucket],
                -priority,
                int(node.get("scene_order", 0) or 0),
            )
            if removable_idx is None or remove_key > removable_key:
                removable_idx = idx
                removable_key = remove_key
        if removable_idx is None:
            break
        removed = repaired.pop(removable_idx)
        seen_keys.discard((str(removed.get("scene_id")), int(removed.get("beat_index", 0) or 0)))

    return sorted(repaired, key=lambda x: (x["scene_order"], x["beat_index"]))


def node_ground_prompt(language: str, character_name: str, node: Dict[str, Any], beat: Dict[str, Any]) -> List[Dict[str, str]]:
    system = NODE_GROUND_PROMPT_ZH if language == "zh" else NODE_GROUND_PROMPT_EN
    user = f"Focal character: {character_name}\nDraft node:\n{json.dumps(node, ensure_ascii=False, indent=2)}\nBeat text:\n{beat['beat_text'][:5000]}\n"
    return prompt_messages(system, user)


def build_timeline_nodes(llm: LLMClient, language: str, character: Dict[str, Any], candidate_nodes: Sequence[Dict[str, Any]], beat_map: Dict[Tuple[str, int], Dict[str, Any]], scenes: Sequence[SceneRecord]) -> Tuple[str, List[Dict[str, Any]]]:
    if not candidate_nodes:
        return "", []
    raw = llm_json(llm, timeline_refine_prompt(language, character, candidate_nodes), max_tokens=3200)
    final_nodes = validate_nodes_v3(raw.get("final_nodes", []) or [], candidate_nodes, scenes)
    final_nodes = repair_timeline_nodes(final_nodes, candidate_nodes, scenes)
    grounded_nodes = []
    for ordinal, node in enumerate(final_nodes, start=1):
        beat = beat_map[(node["scene_id"], node["beat_index"])]
        grounded = llm_json(llm, node_ground_prompt(language, character["character_name"], node, beat), max_tokens=1100)
        evidence_quotes = []
        for quote in grounded.get("evidence_quotes", []) or []:
            quote = clean_text(quote)
            if quote and quote in beat["beat_text"] and quote not in evidence_quotes:
                evidence_quotes.append(quote)
        if not evidence_quotes:
            evidence_quotes = list(node.get("evidence_quotes") or [])[:2]
        grounded_nodes.append({"timeline_node_id": stable_id(character["character_name"], node["scene_id"], str(node["beat_index"]), str(ordinal), prefix="ptu3"), "document_id": f"scene_{node['scene_id']}_beat_{node['beat_index']}", "scene_id": node["scene_id"], "scene_order": node["scene_order"], "scene_title": node["scene_title"], "beat_index": node["beat_index"], "beat_heading": node.get("beat_heading"), "scene_summary": clean_text(beat.get("beat_summary")), "importance": node.get("importance") or "core", "role_in_context": clean_text(grounded.get("role_in_context")) or node.get("role_in_context"), "salient_development": clean_text(grounded.get("salient_development")) or node.get("salient_development"), "goal_state": clean_text(grounded.get("goal_state")) or node.get("goal_state"), "resulting_state": clean_text(grounded.get("resulting_state")) or node.get("resulting_state"), "unresolved_issue": clean_text(grounded.get("unresolved_issue")) or node.get("unresolved_issue"), "related_event_ids": [], "related_episode_ids": [], "evidence_quotes": evidence_quotes[:4], "auxiliary": {"relation_updates": [], "status_updates": [], "persona_anchor": ""}})
    return clean_text(raw.get("timeline_summary")), grounded_nodes


def _legacy_scene_pairs(gold_nodes: Sequence[Dict[str, Any]], pred_nodes: Sequence[Dict[str, Any]]) -> Tuple[set, set]:
    return {str(n.get("scene_id")) for n in gold_nodes}, {str(n.get("scene_id")) for n in pred_nodes}


def _node_text(node: Dict[str, Any]) -> str:
    parts = [clean_text(node.get("salient_development")), clean_text(node.get("goal_state")), clean_text(node.get("resulting_state")), clean_text(node.get("unresolved_issue"))]
    return "\n".join([p for p in parts if p])


def greedy_match_same_scene_nodes(gold_nodes: Sequence[Dict[str, Any]], pred_nodes: Sequence[Dict[str, Any]], embedder: OpenAICompatEmbedder) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    by_scene_gold: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_scene_pred: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in gold_nodes:
        by_scene_gold[str(node.get("scene_id"))].append(node)
    for node in pred_nodes:
        by_scene_pred[str(node.get("scene_id"))].append(node)
    matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for scene_id in sorted(set(by_scene_gold) & set(by_scene_pred), key=lambda x: int(x) if x.isdigit() else x):
        g_nodes = by_scene_gold[scene_id]
        p_nodes = by_scene_pred[scene_id]
        texts = [_node_text(x) for x in g_nodes + p_nodes]
        vecs = embedder.embed_documents(texts)
        g_vecs = vecs[: len(g_nodes)]
        p_vecs = vecs[len(g_nodes) :]
        pairs = []
        for gi, g_vec in enumerate(g_vecs):
            for pi, p_vec in enumerate(p_vecs):
                pairs.append((float(np.dot(g_vec, p_vec)), gi, pi))
        used_g = set()
        used_p = set()
        for score, gi, pi in sorted(pairs, reverse=True):
            if gi in used_g or pi in used_p or score < 0.35:
                continue
            used_g.add(gi)
            used_p.add(pi)
            matched.append((g_nodes[gi], p_nodes[pi]))
    return matched


def dev_judge_prompt(character_name: str, gold_node: Dict[str, Any], pred_node: Dict[str, Any]) -> List[Dict[str, str]]:
    user = f"Character: {character_name}\nGold node:\n{json.dumps(gold_node, ensure_ascii=False, indent=2)}\nPredicted node:\n{json.dumps(pred_node, ensure_ascii=False, indent=2)}\nOutput TRUE if the predicted node captures the same broad, externally verifiable change family in this scene, even if details differ. Output FALSE only when the predicted development is about a clearly different change."
    return prompt_messages(DEV_JUDGE_PROMPT_EN, user)


def state_judge_prompt(character_name: str, gold_node: Dict[str, Any], pred_node: Dict[str, Any]) -> List[Dict[str, str]]:
    user = f"Character: {character_name}\nGold node:\n{json.dumps(gold_node, ensure_ascii=False, indent=2)}\nPredicted node:\n{json.dumps(pred_node, ensure_ascii=False, indent=2)}\nFocus on whether the before/after situation is broadly compatible. Missing fields are acceptable; output FALSE only when the implied state transition is clearly incompatible."
    return prompt_messages(STATE_JUDGE_PROMPT_EN, user)


def _scene_excerpt(scene: Optional[SceneRecord], max_chars: int = 900) -> str:
    if scene is None:
        return ""
    return clean_text(scene.content)[:max_chars]


def pred_transition_coherence_prompt(
    language: str,
    character_name: str,
    prev_node: Dict[str, Any],
    next_node: Dict[str, Any],
    prev_scene: Optional[SceneRecord],
    next_scene: Optional[SceneRecord],
) -> List[Dict[str, str]]:
    system = PRED_TRANSITION_COHERENCE_PROMPT_ZH if language == "zh" else PRED_TRANSITION_COHERENCE_PROMPT_EN
    prev_scene_payload = {
        "scene_id": clean_text(prev_node.get("scene_id")),
        "scene_order": prev_node.get("scene_order"),
        "scene_title": clean_text(prev_node.get("scene_title")) or clean_text(prev_scene.scene_title if prev_scene else ""),
        "scene_excerpt": _scene_excerpt(prev_scene),
    }
    next_scene_payload = {
        "scene_id": clean_text(next_node.get("scene_id")),
        "scene_order": next_node.get("scene_order"),
        "scene_title": clean_text(next_node.get("scene_title")) or clean_text(next_scene.scene_title if next_scene else ""),
        "scene_excerpt": _scene_excerpt(next_scene),
    }
    user = (
        f"Character: {character_name}\n"
        "Judge whether the later predicted node is a coherent next step after the earlier predicted node for the same character.\n"
        "Small skipped steps are acceptable, but the overall trajectory should remain plausible and script-grounded.\n\n"
        f"Earlier predicted node:\n{json.dumps(prev_node, ensure_ascii=False, indent=2)}\n\n"
        f"Later predicted node:\n{json.dumps(next_node, ensure_ascii=False, indent=2)}\n\n"
        f"Earlier scene evidence:\n{json.dumps(prev_scene_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Later scene evidence:\n{json.dumps(next_scene_payload, ensure_ascii=False, indent=2)}"
    )
    return prompt_messages(system, user)


def _phase_bucket(scene_order: int, max_scene_order: int) -> str:
    if max_scene_order <= 0:
        return "middle"
    ratio = scene_order / max_scene_order
    if ratio <= 0.34:
        return "early"
    if ratio <= 0.67:
        return "middle"
    return "late"


def _node_transition_salience(node: Dict[str, Any]) -> int:
    score = 0
    if clean_text(node.get("importance")).lower() == "core":
        score += 2
    if clean_text(node.get("goal_state")):
        score += 1
    if clean_text(node.get("resulting_state")):
        score += 1
    if clean_text(node.get("unresolved_issue")):
        score += 1
    if len(clean_text(node.get("salient_development"))) >= 80:
        score += 1
    if len(node.get("evidence_quotes", []) or []) >= 2:
        score += 1
    return score


def _is_sparse_transition_node(node: Dict[str, Any]) -> bool:
    return (
        clean_text(node.get("importance")).lower() != "core"
        and not clean_text(node.get("goal_state"))
        and not clean_text(node.get("resulting_state"))
        and not clean_text(node.get("unresolved_issue"))
        and len(clean_text(node.get("salient_development"))) < 70
    )


def _pair_phase_label(prev_bucket: str, next_bucket: str) -> str:
    return prev_bucket if prev_bucket == next_bucket else f"{prev_bucket}->{next_bucket}"


def build_transition_pair_records(pred_item: Dict[str, Any], gold_name: str, scene_by_id: Dict[str, SceneRecord]) -> List[Dict[str, Any]]:
    pred_name = pred_item["character_name"]
    pred_nodes = sorted(
        pred_item.get("timeline_nodes", []) or [],
        key=lambda x: (int(x.get("scene_order", 0) or 0), int(x.get("beat_index", 0) or 0)),
    )
    if len(pred_nodes) < 2:
        return []
    max_scene_order = max(int(x.get("scene_order", 0) or 0) for x in pred_nodes)
    rows: List[Dict[str, Any]] = []
    for idx in range(len(pred_nodes) - 1):
        prev_node = pred_nodes[idx]
        next_node = pred_nodes[idx + 1]
        prev_order = int(prev_node.get("scene_order", 0) or 0)
        next_order = int(next_node.get("scene_order", 0) or 0)
        prev_bucket = _phase_bucket(prev_order, max_scene_order)
        next_bucket = _phase_bucket(next_order, max_scene_order)
        prev_salience = _node_transition_salience(prev_node)
        next_salience = _node_transition_salience(next_node)
        score = 0
        if clean_text(prev_node.get("importance")).lower() == "core" or clean_text(next_node.get("importance")).lower() == "core":
            score += 2
        if clean_text(prev_node.get("importance")).lower() == "core" and clean_text(next_node.get("importance")).lower() == "core":
            score += 1
        score += max(prev_salience, next_salience)
        if clean_text(prev_node.get("resulting_state")):
            score += 1
        if clean_text(next_node.get("goal_state")):
            score += 1
        if clean_text(next_node.get("resulting_state")):
            score += 1
        if clean_text(next_node.get("unresolved_issue")):
            score += 1
        scene_gap = max(0, next_order - prev_order)
        if prev_bucket != next_bucket:
            score += 1
        if 2 <= scene_gap <= 12:
            score += 1
        if clean_text(prev_node.get("importance")).lower() != "core" and clean_text(next_node.get("importance")).lower() != "core":
            score -= 1
        if scene_gap > 12:
            score -= 1
        if scene_gap <= 1 and prev_bucket == next_bucket:
            score -= 1
        if _is_sparse_transition_node(prev_node) and _is_sparse_transition_node(next_node):
            score -= 1
        rows.append(
            {
                "gold_character_name": gold_name,
                "pred_character_name": pred_name,
                "prev_node": prev_node,
                "next_node": next_node,
                "prev_scene": scene_by_id.get(str(prev_node.get("scene_id"))),
                "next_scene": scene_by_id.get(str(next_node.get("scene_id"))),
                "prev_scene_id": clean_text(prev_node.get("scene_id")),
                "next_scene_id": clean_text(next_node.get("scene_id")),
                "prev_scene_title": clean_text(prev_node.get("scene_title")),
                "next_scene_title": clean_text(next_node.get("scene_title")),
                "prev_bucket": prev_bucket,
                "next_bucket": next_bucket,
                "phase_label": _pair_phase_label(prev_bucket, next_bucket),
                "scene_gap": scene_gap,
                "pair_score": score,
                "selected_as_important": False,
            }
        )
    return rows


def select_important_transition_pairs(rows: Sequence[Dict[str, Any]], max_pairs: int = 4) -> List[int]:
    if not rows:
        return []
    indexed = list(enumerate(rows))
    selected: List[int] = []
    used = set()

    def has_transition_anchor(row: Dict[str, Any]) -> bool:
        prev_node = row["prev_node"]
        next_node = row["next_node"]
        return any(
            clean_text(value)
            for value in (
                prev_node.get("resulting_state"),
                next_node.get("goal_state"),
                next_node.get("resulting_state"),
                next_node.get("unresolved_issue"),
            )
        )

    eligible = []
    fallback = []
    for i, row in indexed:
        scene_gap = int(row.get("scene_gap", 0) or 0)
        phase_label = clean_text(row.get("phase_label"))
        anchored = has_transition_anchor(row)
        if scene_gap > 14:
            continue
        if phase_label == "early->late" and scene_gap > 8:
            continue
        if anchored:
            eligible.append((i, row))
        else:
            fallback.append((i, row))

    candidate_pool = eligible or fallback or indexed

    def scene_gap(row: Dict[str, Any]) -> int:
        return int(row.get("scene_gap", 0) or 0)

    def best_index(candidates: Sequence[Tuple[int, Dict[str, Any]]]) -> Optional[int]:
        best_i = None
        best_key = None
        for i, row in candidates:
            if i in used:
                continue
            scene_gap = int(row.get("scene_gap", 0) or 0)
            phase_change = 1 if row.get("prev_bucket") != row.get("next_bucket") else 0
            anchored = 1 if has_transition_anchor(row) else 0
            key = (
                anchored,
                int(row.get("pair_score", 0) or 0),
                1 if phase_change and scene_gap <= 8 else 0,
                1 if clean_text(row["prev_node"].get("importance")).lower() == "core" else 0,
                1 if clean_text(row["next_node"].get("importance")).lower() == "core" else 0,
                -scene_gap,
            )
            if best_i is None or key > best_key:
                best_i = i
                best_key = key
        return best_i

    cross_candidates = [
        (i, row)
        for i, row in candidate_pool
        if row.get("prev_bucket") != row.get("next_bucket") and scene_gap(row) <= 8
    ]
    idx = best_index(cross_candidates)
    if idx is not None:
        selected.append(idx)
        used.add(idx)

    nontrivial_candidates = [(i, row) for i, row in candidate_pool if scene_gap(row) >= 2]
    idx = best_index(nontrivial_candidates)
    if idx is not None:
        selected.append(idx)
        used.add(idx)

    for target_label in ("early->middle", "middle->late"):
        idx = best_index([(i, row) for i, row in candidate_pool if row.get("phase_label") == target_label])
        if idx is not None:
            selected.append(idx)
            used.add(idx)

    for bucket in ("early", "middle", "late"):
        if len(selected) >= max_pairs:
            break
        idx = best_index(
            [
                (i, row)
                for i, row in candidate_pool
                if row.get("phase_label") == bucket or row.get("prev_bucket") == bucket
            ]
        )
        if idx is not None:
            selected.append(idx)
            used.add(idx)

    while len(selected) < min(max_pairs, len(rows)):
        idx = best_index(candidate_pool)
        if idx is None:
            break
        selected.append(idx)
        used.add(idx)

    selected = sorted(dict.fromkeys(selected))
    return selected[:max_pairs]


def _timeline_text_for_fact_eval(character_item: Dict[str, Any], display_name: Optional[str] = None) -> str:
    lines = [
        f"Character: {clean_text(display_name or character_item.get('character_name'))}",
        f"Timeline summary: {clean_text(character_item.get('timeline_summary'))}",
    ]
    for idx, node in enumerate(character_item.get("timeline_nodes", []) or [], start=1):
        node_parts = [
            f"[{idx}] scene_id={clean_text(node.get('scene_id'))}",
            f"scene_order={clean_text(node.get('scene_order'))}",
            f"scene_title={clean_text(node.get('scene_title'))}",
            f"role={clean_text(node.get('role_in_context'))}",
            f"development={clean_text(node.get('salient_development'))}",
        ]
        if clean_text(node.get("goal_state")):
            node_parts.append(f"goal={clean_text(node.get('goal_state'))}")
        if clean_text(node.get("resulting_state")):
            node_parts.append(f"result={clean_text(node.get('resulting_state'))}")
        if clean_text(node.get("unresolved_issue")):
            node_parts.append(f"pressure={clean_text(node.get('unresolved_issue'))}")
        lines.append(" | ".join(node_parts))
    return "\n".join(lines)


def timeline_fact_prompt(language: str, character_name: str, timeline_item: Dict[str, Any], max_facts: int = 8) -> List[Dict[str, str]]:
    system = TIMELINE_FACT_PROMPT_ZH if language == "zh" else TIMELINE_FACT_PROMPT_EN
    user = (
        f"Character: {character_name}\n"
        f"Extract at most {max_facts} atomic narrative facts from this timeline.\n"
        "Facts should capture durable decisions, status/role changes, relationship shifts, goal shifts, outcomes, or unresolved pressure.\n"
        "Avoid trivial scene-local details.\n"
        "If the timeline uses an alias or alternate surface form for the same matched character, normalize it mentally to the character name above.\n"
        f"Timeline:\n{_timeline_text_for_fact_eval(timeline_item, display_name=character_name)}"
    )
    return prompt_messages(system, user)


def _normalize_fact(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    fact_text = clean_text(item.get("fact_text") or item.get("text"))
    if not fact_text:
        return None
    fact_type = clean_text(item.get("fact_type")).lower() or "other"
    if fact_type not in FACT_TYPES:
        fact_type = "other"
    phase = clean_text(item.get("phase")).lower() or "multi"
    if phase not in FACT_PHASES:
        phase = "multi"
    scene_refs: List[str] = []
    for value in item.get("scene_refs", []) or item.get("scene_ids", []) or []:
        ref = clean_text(value)
        if ref and ref not in scene_refs:
            scene_refs.append(ref)
    return {
        "fact_id": clean_text(item.get("fact_id")) or f"F{idx}",
        "fact_type": fact_type,
        "phase": phase,
        "fact_text": fact_text,
        "scene_refs": scene_refs[:6],
    }


def extract_timeline_facts(llm: LLMClient, language: str, character_item: Dict[str, Any], max_facts: int = 8) -> List[Dict[str, Any]]:
    raw = llm_json(llm, timeline_fact_prompt(language, character_item.get("character_name", ""), character_item, max_facts=max_facts), max_tokens=2200)
    facts: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(raw.get("facts", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_fact(item, idx)
        if not normalized:
            continue
        key = normalize_name(normalized["fact_text"])
        if not key or key in seen:
            continue
        seen.add(key)
        facts.append(normalized)
        if len(facts) >= max_facts:
            break
    return facts


def fact_support_prompt(language: str, character_name: str, facts: Sequence[Dict[str, Any]], reference_timeline: Dict[str, Any]) -> List[Dict[str, str]]:
    system = FACT_SUPPORT_PROMPT_ZH if language == "zh" else FACT_SUPPORT_PROMPT_EN
    user = (
        f"Character: {character_name}\n"
        "The facts and the reference timeline refer to the same matched character. If the timeline uses an alias or alternate surface name, treat it as the same person.\n"
        f"Atomic facts:\n{json.dumps(list(facts), ensure_ascii=False, indent=2)}\n\n"
        f"Reference timeline:\n{_timeline_text_for_fact_eval(reference_timeline, display_name=character_name)}"
    )
    return prompt_messages(system, user)


def judge_supported_fact_ids(
    llm: LLMClient,
    language: str,
    character_name: str,
    facts: Sequence[Dict[str, Any]],
    reference_timeline: Dict[str, Any],
) -> List[str]:
    if not facts:
        return []
    raw = llm_json(
        llm,
        fact_support_prompt(language, character_name, facts, reference_timeline),
        max_tokens=1800,
    )
    valid_ids = {clean_text(item.get("fact_id")) for item in facts if clean_text(item.get("fact_id"))}
    supported: List[str] = []
    for fact_id in raw.get("supported_fact_ids", []) or []:
        fact_id = clean_text(fact_id)
        if fact_id and fact_id in valid_ids and fact_id not in supported:
            supported.append(fact_id)
    return supported


def _arc_text_for_match(arc: Dict[str, Any]) -> str:
    parts = [
        clean_text(arc.get("title")),
        clean_text(arc.get("arc_focus")),
        clean_text(arc.get("arc_summary")),
        clean_text(arc.get("start_state")),
        clean_text(arc.get("end_state")),
        clean_text(arc.get("unresolved_issue")),
    ]
    return "\n".join([p for p in parts if p])


def choose_best_arc_match_narrative(gold_arc: Dict[str, Any], pred_arcs: Sequence[Dict[str, Any]], embedder: OpenAICompatEmbedder) -> Optional[Dict[str, Any]]:
    if not pred_arcs:
        return None
    gold_scenes = set(str(x) for x in (gold_arc.get("linked_scene_ids", []) or []))
    texts = [_arc_text_for_match(gold_arc)] + [_arc_text_for_match(a) for a in pred_arcs]
    vecs = embedder.embed_documents(texts)
    gold_vec = vecs[0]
    best = None
    best_score = -1.0
    for arc, vec in zip(pred_arcs, vecs[1:]):
        pred_scenes = set(str(x) for x in (arc.get("linked_scene_ids", []) or []))
        if gold_scenes or pred_scenes:
            inter = len(gold_scenes & pred_scenes)
            union = len(gold_scenes | pred_scenes)
            jaccard = inter / union if union else 0.0
        else:
            jaccard = 0.0
        sim = float(np.dot(gold_vec, vec))
        score = 0.85 * sim + 0.15 * jaccard
        if score > best_score:
            best_score = score
            best = arc
    return best


def narrative_arc_aspect_judge_prompt(language: str, character_name: str, gold_arc: Dict[str, Any], pred_arc: Dict[str, Any]) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在评估两个角色跨场景 arc 是否体现同一个叙事方面。叙事方面指人物长期变化的核心维度，例如亲密关系、家庭冲突、成瘾与恢复、职业自主、身份认同、忠诚拉扯等。不要要求标题一致，也不要过度依赖 linked_scene_ids。只输出 TRUE 或 FALSE。"
        user = (
            f"角色: {character_name}\n金标 arc:\n{json.dumps(gold_arc, ensure_ascii=False, indent=2)}\n"
            f"预测 arc:\n{json.dumps(pred_arc, ensure_ascii=False, indent=2)}\n"
            "如果两者在讲同一条核心叙事线程或同一类人物长期张力，即使覆盖的节点边界、概括层次或命名不同，也输出 TRUE；只有当两者明显不是同一叙事方面时才输出 FALSE。"
        )
    else:
        system = "You are judging whether two cross-scene arcs express the same narrative aspect for a character. A narrative aspect means the core long-range storyline dimension, such as romance, family conflict, addiction and recovery, career autonomy, identity, or loyalty tension. Do not require the same title, and do not rely heavily on linked_scene_ids. Output TRUE or FALSE only."
        user = (
            f"Character: {character_name}\nGold arc:\n{json.dumps(gold_arc, ensure_ascii=False, indent=2)}\n"
            f"Predicted arc:\n{json.dumps(pred_arc, ensure_ascii=False, indent=2)}\n"
            "Output TRUE if they capture the same underlying narrative thread or long-range tension, even if the node boundaries, abstraction level, or naming differ. Output FALSE only when they are clearly about different narrative aspects."
        )
    return prompt_messages(system, user)


def narrative_arc_progression_judge_prompt(language: str, character_name: str, gold_arc: Dict[str, Any], pred_arc: Dict[str, Any]) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在评估两个角色跨场景 arc 的演进方向是否大体一致。重点看这条叙事线程从什么状态出发，经历了怎样的变化，最终走向什么局面。允许省略部分中间节点，但如果方向相反、阶段判断错误或结局明显不兼容，输出 FALSE。只输出 TRUE 或 FALSE。"
        user = (
            f"角色: {character_name}\n金标 arc:\n{json.dumps(gold_arc, ensure_ascii=False, indent=2)}\n"
            f"预测 arc:\n{json.dumps(pred_arc, ensure_ascii=False, indent=2)}\n"
            "如果预测 arc 在这条叙事方面上的起点、推进方向和结果与金标大体兼容，就输出 TRUE；不要因为少几个节点或概括更粗而判 FALSE。"
        )
    else:
        system = "You are judging whether two cross-scene arcs have broadly compatible narrative progression. Focus on where the thread starts, how it evolves, and where it ends. Missing intermediate beats are acceptable, but reversed direction, wrong phase, or clearly incompatible end states should be FALSE. Output TRUE or FALSE only."
        user = (
            f"Character: {character_name}\nGold arc:\n{json.dumps(gold_arc, ensure_ascii=False, indent=2)}\n"
            f"Predicted arc:\n{json.dumps(pred_arc, ensure_ascii=False, indent=2)}\n"
            "Output TRUE if the predicted arc preserves the broad start-to-end evolution of the same narrative thread, even if it is coarser or omits some intermediate nodes. Output FALSE only when the trajectory is clearly incompatible."
        )
    return prompt_messages(system, user)


def evaluate_v5(movie_dir: Path, output_dir: Path) -> Dict[str, Any]:
    def _load_json_with_fallback(candidates: Sequence[str]) -> Dict[str, Any]:
        for name in candidates:
            path = movie_dir / name
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"Missing any of expected files under {movie_dir}: {candidates}")

    language = detect_language(movie_dir)
    scenes = load_scenes(movie_dir / "script.json", language)
    scene_by_id = {str(scene.scene_id): scene for scene in scenes}

    llm = LLMClient(DEFAULT_LLM_MODEL, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY)
    embedder = OpenAICompatEmbedder(DEFAULT_EMBED_MODEL, DEFAULT_EMBED_BASE_URL, DEFAULT_EMBED_API_KEY)
    pred_timeline = json.loads((output_dir / "pred_task_1_character_timelines.json").read_text(encoding="utf-8"))
    pred_timeline = convert_timeline_scene_ids_for_public(pred_timeline)
    pred_arcs = json.loads((output_dir / "pred_task_1_cross_scene_arcs.json").read_text(encoding="utf-8"))
    gold_timeline = _load_json_with_fallback(["gold_character_timelines_v1.json", "task_1_character_timelines.json"])
    gold_arcs = _load_json_with_fallback(["gold_cross_scene_arcs_v1.json", "task_1_cross_scene_arcs.json"])
    gold_chars = gold_timeline.get("focal_character_timelines", []) or []
    pred_chars = pred_timeline.get("focal_character_timelines", []) or []

    gold_alias_map: Dict[str, str] = {}
    for item in gold_chars:
        canonical = item["character_name"]
        gold_alias_map[normalize_name(canonical)] = canonical
        for alias in item.get("aliases", []) or []:
            gold_alias_map[normalize_name(alias)] = canonical

    matched_pred_to_gold: Dict[str, str] = {}
    used_gold = set()
    for item in pred_chars:
        pred_name = item["character_name"]
        gold_name = gold_alias_map.get(normalize_name(pred_name))
        if gold_name and gold_name not in used_gold:
            matched_pred_to_gold[pred_name] = gold_name
            used_gold.add(gold_name)

    _target_character_coverage = len(matched_pred_to_gold) / max(1, len(pred_chars))

    gold_by_name = {item["character_name"]: item for item in gold_chars}
    matched_pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    node_pred_total = 0
    node_gold_total = 0
    legacy_pred_pairs = set()
    legacy_gold_pairs = set()
    for pred_name, gold_name in matched_pred_to_gold.items():
        pred_item = next(x for x in pred_chars if x["character_name"] == pred_name)
        gold_item = gold_by_name[gold_name]
        pred_nodes = pred_item.get("timeline_nodes", []) or []
        gold_nodes = gold_item.get("timeline_nodes", []) or []
        node_pred_total += len(pred_nodes)
        node_gold_total += len(gold_nodes)
        gold_scene_ids, pred_scene_ids = _legacy_scene_pairs(gold_nodes, pred_nodes)
        for sid in pred_scene_ids:
            legacy_pred_pairs.add((gold_name, sid))
        for sid in gold_scene_ids:
            legacy_gold_pairs.add((gold_name, sid))
        for gold_node, pred_node in greedy_match_same_scene_nodes(gold_nodes, pred_nodes, embedder):
            matched_pairs.append((gold_name, gold_node, pred_node))

    node_precision = len(matched_pairs) / max(1, node_pred_total)
    node_recall = len(matched_pairs) / max(1, node_gold_total)
    node_f1 = 0.0 if node_precision + node_recall == 0 else 2 * node_precision * node_recall / (node_precision + node_recall)

    legacy_inter = legacy_pred_pairs & legacy_gold_pairs
    scene_precision = len(legacy_inter) / max(1, len(legacy_pred_pairs))
    scene_recall = len(legacy_inter) / max(1, len(legacy_gold_pairs))
    scene_f1 = 0.0 if scene_precision + scene_recall == 0 else 2 * scene_precision * scene_recall / (scene_precision + scene_recall)

    dev_votes = []
    state_votes = []
    for character_name, gold_node, pred_node in matched_pairs:
        dev_votes.append(run_bool_judge(llm, dev_judge_prompt(character_name, gold_node, pred_node)))
        state_votes.append(run_bool_judge(llm, state_judge_prompt(character_name, gold_node, pred_node)))

    pred_transition_votes = []
    important_pred_transition_votes = []
    pred_transition_details: List[Dict[str, Any]] = []
    for pred_item in pred_chars:
        pred_name = pred_item["character_name"]
        gold_name = matched_pred_to_gold.get(pred_name, pred_name)
        pair_rows = build_transition_pair_records(pred_item, gold_name, scene_by_id)
        important_indexes = set(select_important_transition_pairs(pair_rows, max_pairs=4))
        for idx, row in enumerate(pair_rows):
            row["selected_as_important"] = idx in important_indexes
            ok = run_bool_judge(
                llm,
                pred_transition_coherence_prompt(
                    language,
                    gold_name,
                    row["prev_node"],
                    row["next_node"],
                    row["prev_scene"],
                    row["next_scene"],
                ),
            )
            pred_transition_votes.append(ok)
            if row["selected_as_important"]:
                important_pred_transition_votes.append(ok)
            pred_transition_details.append(
                {
                    "gold_character_name": row["gold_character_name"],
                    "pred_character_name": row["pred_character_name"],
                    "prev_scene_id": row["prev_scene_id"],
                    "next_scene_id": row["next_scene_id"],
                    "prev_scene_title": row["prev_scene_title"],
                    "next_scene_title": row["next_scene_title"],
                    "prev_bucket": row["prev_bucket"],
                    "next_bucket": row["next_bucket"],
                    "phase_label": row["phase_label"],
                    "scene_gap": row["scene_gap"],
                    "pair_score": row["pair_score"],
                    "selected_as_important": row["selected_as_important"],
                    "coherent": ok,
                }
            )

    gold_fact_total = 0
    pred_fact_total = 0
    supported_gold_fact_total = 0
    supported_pred_fact_total = 0
    fact_details: List[Dict[str, Any]] = []
    for pred_name, gold_name in matched_pred_to_gold.items():
        pred_item = next(x for x in pred_chars if x["character_name"] == pred_name)
        gold_item = gold_by_name[gold_name]
        gold_facts: List[Dict[str, Any]] = []
        pred_facts: List[Dict[str, Any]] = []
        fact_error: Optional[str] = None
        try:
            gold_facts = extract_timeline_facts(llm, language, gold_item)
            pred_facts = extract_timeline_facts(llm, language, pred_item)
        except Exception as exc:
            fact_error = f"fact_extract_failed: {type(exc).__name__}: {exc}"
        supported_gold_ids: List[str] = []
        supported_pred_ids: List[str] = []
        if fact_error is None:
            try:
                supported_gold_ids = judge_supported_fact_ids(llm, language, gold_name, gold_facts, pred_item)
                supported_pred_ids = judge_supported_fact_ids(llm, language, gold_name, pred_facts, gold_item)
            except Exception as exc:
                fact_error = f"fact_judge_failed: {type(exc).__name__}: {exc}"
                supported_gold_ids = []
                supported_pred_ids = []
        gold_fact_total += len(gold_facts)
        pred_fact_total += len(pred_facts)
        supported_gold_fact_total += len(supported_gold_ids)
        supported_pred_fact_total += len(supported_pred_ids)
        fact_details.append(
            {
                "gold_character_name": gold_name,
                "pred_character_name": pred_name,
                "gold_fact_count": len(gold_facts),
                "pred_fact_count": len(pred_facts),
                "supported_gold_fact_ids": supported_gold_ids,
                "supported_pred_fact_ids": supported_pred_ids,
                "gold_facts": gold_facts,
                "pred_facts": pred_facts,
                "error": fact_error,
            }
        )

    gold_fact_recall = supported_gold_fact_total / max(1, gold_fact_total)
    pred_fact_precision = supported_pred_fact_total / max(1, pred_fact_total)
    fact_f1 = 0.0 if gold_fact_recall + pred_fact_precision == 0 else 2 * gold_fact_recall * pred_fact_precision / (gold_fact_recall + pred_fact_precision)

    pred_arc_items = pred_arcs.get("cross_scene_arcs", []) or []
    gold_arc_items = gold_arcs.get("cross_scene_arcs", []) or []
    pred_char_to_nodes = {item["character_name"]: {n["timeline_node_id"]: n for n in item.get("timeline_nodes", []) or []} for item in pred_chars}
    gold_char_to_nodes = {item["character_name"]: {n["timeline_node_id"]: n for n in item.get("timeline_nodes", []) or []} for item in gold_chars}
    pred_arcs_by_gold_char: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for arc in pred_arc_items:
        pred_name = arc["character_name"]
        gold_name = matched_pred_to_gold.get(pred_name)
        if not gold_name:
            continue
        nodes = pred_char_to_nodes.get(pred_name, {})
        linked_scene_ids = [str(nodes[nid]["scene_id"]) for nid in arc.get("linked_timeline_node_ids", []) if nid in nodes]
        item = dict(arc)
        item["linked_scene_ids"] = linked_scene_ids
        pred_arcs_by_gold_char[gold_name].append(item)

    arc_aspect_votes = []
    arc_progression_votes = []
    for gold_arc in gold_arc_items:
        gold_name = gold_arc["character_name"]
        if gold_name not in used_gold:
            continue
        gold_nodes_map = gold_char_to_nodes.get(gold_name, {})
        gold_scene_ids = [str(gold_nodes_map[nid]["scene_id"]) for nid in gold_arc.get("linked_timeline_node_ids", []) if nid in gold_nodes_map]
        gold_arc_item = dict(gold_arc)
        gold_arc_item["linked_scene_ids"] = gold_scene_ids
        best_pred = choose_best_arc_match_narrative(gold_arc_item, pred_arcs_by_gold_char.get(gold_name, []), embedder)
        if not best_pred:
            arc_aspect_votes.append(False)
            arc_progression_votes.append(False)
            continue
        arc_aspect_votes.append(run_bool_judge(llm, narrative_arc_aspect_judge_prompt(language, gold_name, gold_arc_item, best_pred)))
        arc_progression_votes.append(run_bool_judge(llm, narrative_arc_progression_judge_prompt(language, gold_name, gold_arc_item, best_pred)))

    summary = {
        "movie_id": movie_dir.name,
        "arc_eval_protocol": "narrative_aspect_v1",
        "legacy_scene_grounding_precision": round(scene_precision, 4),
        "legacy_scene_grounding_recall": round(scene_recall, 4),
        "legacy_scene_grounding_f1": round(scene_f1, 4),
        "node_grounding_precision": round(node_precision, 4),
        "node_grounding_recall": round(node_recall, 4),
        "node_grounding_f1": round(node_f1, 4),
        "gold_fact_recall": round(gold_fact_recall, 4),
        "pred_fact_precision": round(pred_fact_precision, 4),
        "fact_f1": round(fact_f1, 4),
        "development_correctness": round(sum(dev_votes) / max(1, len(dev_votes)), 4),
        "state_transition_correctness": round(sum(state_votes) / max(1, len(state_votes)), 4),
        "pred_transition_coherence": round(sum(pred_transition_votes) / max(1, len(pred_transition_votes)), 4),
        "important_pred_transition_coherence": round(sum(important_pred_transition_votes) / max(1, len(important_pred_transition_votes)), 4),
        "arc_narrative_aspect_correctness": round(sum(arc_aspect_votes) / max(1, len(arc_aspect_votes)), 4),
        "arc_progression_correctness": round(sum(arc_progression_votes) / max(1, len(arc_progression_votes)), 4),
    }
    summary["overall"] = round((summary["legacy_scene_grounding_f1"] + summary["node_grounding_f1"] + summary["development_correctness"] + summary["state_transition_correctness"] + summary["arc_narrative_aspect_correctness"] + summary["arc_progression_correctness"]) / 6.0, 4)
    (output_dir / "eval_v3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "fact_coverage_details.json").write_text(
        json.dumps(
            {
                "movie_id": movie_dir.name,
                "language": language,
                "gold_fact_count": gold_fact_total,
                "pred_fact_count": pred_fact_total,
                "supported_gold_fact_count": supported_gold_fact_total,
                "supported_pred_fact_count": supported_pred_fact_total,
                "pred_transition_pair_count": len(pred_transition_votes),
                "supported_pred_transition_pair_count": sum(1 for x in pred_transition_votes if x),
                "important_pred_transition_pair_count": len(important_pred_transition_votes),
                "supported_important_pred_transition_pair_count": sum(1 for x in important_pred_transition_votes if x),
                "characters": fact_details,
                "pred_transition_pairs": pred_transition_details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def run_workflow_v5(movie_dir: Path, output_dir: Path, max_characters: Optional[int] = None, target_characters_file: Optional[str] = None) -> Dict[str, Any]:
    started = time.time()
    language = detect_language(movie_dir)
    scenes = load_scenes(movie_dir / "script.json", language)
    llm = LLMClient(DEFAULT_LLM_MODEL, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY)
    print(json.dumps({"stage": "load_complete", "scene_count": len(scenes)}, ensure_ascii=False), flush=True)
    speaker_candidates = extract_speaker_candidates(scenes)
    print(json.dumps({"stage": "speaker_candidates", "count": len(speaker_candidates)}, ensure_ascii=False), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    beat_summary_path = output_dir / "beat_summaries.json"
    if beat_summary_path.exists():
        beat_summaries = json.loads(beat_summary_path.read_text(encoding="utf-8"))
        print(json.dumps({"stage": "beat_summaries_cache_hit", "count": len(beat_summaries)}, ensure_ascii=False), flush=True)
    else:
        beat_summaries = summarize_beats(llm, scenes, language)
        beat_summary_path.write_text(json.dumps(beat_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates = aggregate_candidates(speaker_candidates, beat_summaries)
    print(json.dumps({"stage": "candidate_aggregate_ready", "count": len(candidates)}, ensure_ascii=False), flush=True)
    selected_characters = load_target_characters(movie_dir, target_characters_file)
    effective_max_characters = max_characters if (max_characters is not None and max_characters > 0) else None
    if not selected_characters:
        llm_select_k = effective_max_characters or 3
        selected_characters = select_focal_characters(llm, language, candidates, llm_select_k)
        source = "llm_selected"
    else:
        if effective_max_characters is not None:
            selected_characters = selected_characters[:effective_max_characters]
        source = "provided_task3_roles"
    print(json.dumps({"stage": "characters_ready", "source": source, "count": len(selected_characters), "names": [x["character_name"] for x in selected_characters]}, ensure_ascii=False), flush=True)

    beat_map = {(str(item["scene_id"]), int(item["beat_index"])): item for item in beat_summaries}
    timeline_payload = {"movie_id": movie_dir.name, "language": language, "task_name": "Story Dynamics Structuring", "task_version": "workflow_v5_given_characters_key_developments", "focal_character_timelines": [], "build_summary": {}}
    arc_payload = {"movie_id": movie_dir.name, "language": language, "task_name": "Story Dynamics Structuring", "task_version": "workflow_v5_given_characters_key_developments", "cross_scene_arcs": [], "build_summary": {}}
    diagnostics = {"speaker_candidates": speaker_candidates, "aggregated_candidates": candidates, "selected_characters": selected_characters, "per_character": []}

    for character in selected_characters:
        print(json.dumps({"stage": "character_start", "character": character["character_name"]}, ensure_ascii=False), flush=True)
        candidate_nodes = build_character_candidate_nodes(character, beat_summaries)
        print(json.dumps({"stage": "character_candidate_nodes_ready", "character": character["character_name"], "candidate_count": len(candidate_nodes)}, ensure_ascii=False), flush=True)
        if not candidate_nodes:
            continue
        timeline_summary, grounded_nodes = build_timeline_nodes(llm, language, character, candidate_nodes, beat_map, scenes)
        print(json.dumps({"stage": "timeline_ready", "character": character["character_name"], "final_node_count": len(grounded_nodes)}, ensure_ascii=False), flush=True)
        if not grounded_nodes:
            continue
        selection_reason = character.get("selection_reason")
        if source == "provided_task3_roles":
            selection_reason = "由 Task 3 角色列表提供" if language == "zh" else "provided externally via Task 3 role list"
        task3_relevance = "由 Task 3 角色列表指定目标角色，并基于 beat 级证据追踪其关键发展。" if language == "zh" else "Target character provided by the Task 3 role list and traced through beat-grounded developments."
        fallback_timeline_summary = (f"该时间线基于 beat 级证据刻画了 {character['character_name']} 的 {len(grounded_nodes)} 个关键发展节点。" if language == "zh" else f"The timeline traces {len(grounded_nodes)} beat-grounded developments for {character['character_name']}.")
        timeline_payload["focal_character_timelines"].append({"character_name": character["character_name"], "aliases": character.get("aliases", []), "selection_reason": selection_reason, "task3_relevance": task3_relevance, "timeline_nodes": grounded_nodes, "timeline_summary": timeline_summary or fallback_timeline_summary})
        arc_input_nodes = [{"timeline_node_id": n["timeline_node_id"], "scene_id": n["scene_id"], "scene_order": n["scene_order"], "scene_title": n["scene_title"], "salient_development": n["salient_development"], "goal_state": n.get("goal_state"), "resulting_state": n.get("resulting_state"), "unresolved_issue": n.get("unresolved_issue")} for n in grounded_nodes]
        arc_raw = llm_json(llm, arc_prompt(language, character["character_name"], arc_input_nodes), max_tokens=2600)
        node_id_to_scene_id = {n["timeline_node_id"]: n["scene_id"] for n in grounded_nodes}
        arcs = []
        for item in arc_raw.get("arcs", []) or []:
            linked_ids = [clean_text(x) for x in (item.get("linked_timeline_node_ids") or []) if clean_text(x) in node_id_to_scene_id]
            linked_ids = list(dict.fromkeys(linked_ids))
            if len(linked_ids) < 2:
                continue
            arcs.append({"arc_id": stable_id(character["character_name"], clean_text(item.get("title")), prefix="parc3"), "character_name": character["character_name"], "title": clean_text(item.get("title")), "arc_focus": clean_text(item.get("arc_focus")) or "mixed", "linked_timeline_node_ids": linked_ids, "arc_summary": clean_text(item.get("arc_summary")), "start_state": clean_text(item.get("start_state")) or None, "end_state": clean_text(item.get("end_state")) or None, "unresolved_issue": clean_text(item.get("unresolved_issue")) or None, "linked_scene_ids": [node_id_to_scene_id[x] for x in linked_ids]})
        arc_payload["cross_scene_arcs"].extend(arcs)
        diagnostics["per_character"].append({"character_name": character["character_name"], "candidate_node_count": len(candidate_nodes), "final_node_count": len(grounded_nodes), "arc_count": len(arcs)})

    timeline_payload["build_summary"] = {"candidate_character_count": len(candidates), "selected_focal_character_count": len(timeline_payload["focal_character_timelines"]), "timeline_node_count": sum(len(x["timeline_nodes"]) for x in timeline_payload["focal_character_timelines"]), "elapsed_sec": round(time.time() - started, 2)}
    arc_payload["build_summary"] = {"selected_focal_character_count": len(timeline_payload["focal_character_timelines"]), "arc_count": len(arc_payload["cross_scene_arcs"]), "elapsed_sec": round(time.time() - started, 2)}
    (output_dir / "pred_task_1_character_timelines.json").write_text(json.dumps(timeline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "pred_task_1_cross_scene_arcs.json").write_text(json.dumps(arc_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"timeline": timeline_payload, "arcs": arc_payload, "diagnostics": diagnostics}


def _release_timeline_payload(timeline_payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "movie_id": timeline_payload.get("movie_id"),
        "language": timeline_payload.get("language"),
        "task_name": timeline_payload.get("task_name"),
        "task_version": timeline_payload.get("task_version"),
        "focal_character_timelines": [],
        "build_summary": dict(timeline_payload.get("build_summary") or {}),
    }
    language = clean_text(timeline_payload.get("language")).lower()
    canonical_selection_reason = "由 Task 3 角色列表提供" if language == "zh" else "provided externally via Task 3 role list"
    canonical_task3_relevance = "由 Task 3 角色列表指定目标角色，并基于 beat 级证据追踪其关键发展。" if language == "zh" else "Target character provided by the Task 3 role list and traced through beat-grounded developments."
    for item in timeline_payload.get("focal_character_timelines", []) or []:
        selection_reason = item.get("selection_reason")
        task3_relevance = item.get("task3_relevance")
        if selection_reason and "task3" in selection_reason.lower():
            selection_reason = canonical_selection_reason
        if task3_relevance and ("predicted focal character" in task3_relevance.lower() or "task 3" in task3_relevance.lower()):
            task3_relevance = canonical_task3_relevance
        if not task3_relevance and selection_reason == canonical_selection_reason:
            task3_relevance = canonical_task3_relevance
        timeline_item = {
            "character_name": item.get("character_name"),
            "aliases": item.get("aliases", []) or [],
            "selection_reason": selection_reason,
            "task3_relevance": task3_relevance,
            "timeline_nodes": [],
            "timeline_summary": item.get("timeline_summary"),
        }
        for node in item.get("timeline_nodes", []) or []:
            timeline_item["timeline_nodes"].append({
                "timeline_node_id": node.get("timeline_node_id"),
                "document_id": node.get("document_id"),
                "scene_id": node.get("scene_id"),
                "scene_order": node.get("scene_order"),
                "scene_title": node.get("scene_title"),
                "scene_summary": node.get("scene_summary"),
                "role_in_context": node.get("role_in_context"),
                "salient_development": node.get("salient_development"),
                "goal_state": node.get("goal_state"),
                "resulting_state": node.get("resulting_state"),
                "unresolved_issue": node.get("unresolved_issue"),
                "related_event_ids": node.get("related_event_ids", []) or [],
                "related_episode_ids": node.get("related_episode_ids", []) or [],
                "evidence_quotes": node.get("evidence_quotes", []) or [],
                "auxiliary": node.get("auxiliary") or {"relation_updates": [], "status_updates": [], "persona_anchor": ""},
            })
        out["focal_character_timelines"].append(timeline_item)
    return out


def _release_arc_payload(arc_payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "movie_id": arc_payload.get("movie_id"),
        "language": arc_payload.get("language"),
        "task_name": arc_payload.get("task_name"),
        "task_version": arc_payload.get("task_version"),
        "cross_scene_arcs": [],
        "build_summary": dict(arc_payload.get("build_summary") or {}),
    }
    for item in arc_payload.get("cross_scene_arcs", []) or []:
        out["cross_scene_arcs"].append({
            "arc_id": item.get("arc_id"),
            "character_name": item.get("character_name"),
            "title": item.get("title"),
            "arc_focus": item.get("arc_focus"),
            "linked_timeline_node_ids": item.get("linked_timeline_node_ids", []) or [],
            "arc_summary": item.get("arc_summary"),
            "start_state": item.get("start_state"),
            "end_state": item.get("end_state"),
            "unresolved_issue": item.get("unresolved_issue"),
        })
    return out


def public_scene_id(scene_id: Any) -> str:
    raw = clean_text(scene_id)
    if raw.isdigit():
        return str(int(raw) + 1)
    return raw


def convert_timeline_scene_ids_for_public(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(payload, ensure_ascii=False))
    for item in out.get("focal_character_timelines", []) or []:
        for node in item.get("timeline_nodes", []) or []:
            public_id = public_scene_id(node.get("scene_id"))
            node["scene_id"] = public_id
            beat_index = node.get("beat_index")
            if beat_index is None:
                match = re.match(r"^scene_(\d+)_beat_(\d+)$", clean_text(node.get("document_id")))
                if match:
                    beat_index = int(match.group(2))
            if beat_index is not None:
                node["document_id"] = f"scene_{public_id}_beat_{beat_index}"
    return out


def write_release_task1(movie_dir: Path, workflow_result: Dict[str, Any]) -> None:
    timeline_payload = _release_timeline_payload(workflow_result["timeline"])
    timeline_payload = convert_timeline_scene_ids_for_public(timeline_payload)
    arc_payload = _release_arc_payload(workflow_result["arcs"])
    (movie_dir / "task_1_character_timelines.json").write_text(json.dumps(timeline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (movie_dir / "task_1_cross_scene_arcs.json").write_text(json.dumps(arc_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(movie_dir: Path, output_dir: Path, eval_result: Dict[str, Any]) -> None:
    lines = [f"# Task 1 Workflow V5 Report: {movie_dir.name}", "", "## Key Changes", "", "- Switched from scene-level free summarization to beat-level grounded extraction.", "- Removed LLM-led loose character canonicalization that could merge different people.", "- Used Task 3 role names as the explicit target character set instead of relying on script-only character selection.", "- Built candidate nodes directly from beat-level character updates instead of free node invention.", "- Allowed multiple final nodes for the same scene_id when they come from different beat segments.", "- Grounded final nodes against exact beat text rather than whole-scene text.", "", "## Evaluation", ""]
    for key, value in eval_result.items():
        lines.append(f"- {key}: {value}")
    (output_dir / "task1_workflow_v5_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-characters", type=int, default=0)
    parser.add_argument("--target-characters-file", default=None)
    args = parser.parse_args()
    movie_dir = Path(args.movie_dir)
    output_dir = Path(args.output_dir)
    run_workflow_v5(movie_dir, output_dir, max_characters=args.max_characters, target_characters_file=args.target_characters_file)
    eval_result = evaluate_v5(movie_dir, output_dir)
    write_report(movie_dir, output_dir, eval_result)
    print(json.dumps({"stage": "done", "output_dir": str(output_dir), "eval": eval_result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
