import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from stage_bootstrap_story_dynamics import clean_text, dump_json, load_json, stable_id
from stage_build_character_timelines_v1 import (
    aggregate_character_stats,
    build_episode_indexes,
    chat_json,
    clean_candidates_for_audit,
    consolidate_candidates_with_llm,
    detect_language_from_movie_dir,
    fallback_select_characters,
    filter_relation_updates,
    filter_status_updates,
    keep_scene_update,
    merge_candidate_rows,
    normalize_name,
    related_episode_briefs,
    select_characters_with_llm,
    select_representative_bundles,
    shorten_persona_anchor,
)
from stage_build_story_state_updates import (
    canonical_predicate,
    fact_to_state_type,
    should_keep_fact,
)
from stage_refine_story_dynamics_llm import (
    ChatCompletionsClient,
    DEFAULT_API_BASE as DEFAULT_MIMO_API_BASE,
    DEFAULT_FALLBACK_API_BASE as DEFAULT_LOCAL_API_BASE,
    DEFAULT_FALLBACK_API_KEY as DEFAULT_LOCAL_API_KEY,
    DEFAULT_FALLBACK_MODEL as DEFAULT_LOCAL_MODEL,
    DEFAULT_MODEL as DEFAULT_MIMO_MODEL,
    build_scene_text_map,
)


def load_icrp_character_names(movie_dir: Path) -> List[str]:
    icrp_dir = movie_dir / "ICRP"
    if not icrp_dir.exists():
        return []
    names: List[str] = []
    for child in sorted(icrp_dir.iterdir()):
        if child.is_dir():
            name = clean_text(child.name)
            if name:
                names.append(name)
    return names


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def load_icrp_role_profiles(movie_dir: Path) -> List[Dict[str, Any]]:
    icrp_dir = movie_dir / "ICRP"
    if not icrp_dir.exists():
        return []

    profiles: List[Dict[str, Any]] = []
    for child in sorted(icrp_dir.iterdir()):
        if not child.is_dir():
            continue

        raw_names: List[str] = []
        role_dir_name = clean_text(child.name)
        if role_dir_name:
            raw_names.append(role_dir_name)

        persona_card = _safe_load_json(child / "persona_card.json")
        if isinstance(persona_card, dict):
            for key in ("character", "name"):
                value = clean_text(persona_card.get(key))
                if value:
                    raw_names.append(value)

        for qa_name in ("icrp_qa_v3.json", "icrp_qa_v2.json"):
            qa_payload = _safe_load_json(child / qa_name)
            if isinstance(qa_payload, dict):
                value = clean_text(qa_payload.get("character"))
                if value:
                    raw_names.append(value)

        key_relations = _safe_load_json(child / "key_relations.json")
        key_relation_items: List[Any] = []
        if isinstance(key_relations, dict):
            raw_items = key_relations.get("key_relations")
            if isinstance(raw_items, list):
                key_relation_items = list(raw_items)
            else:
                key_relation_items = []
        elif isinstance(key_relations, list):
            key_relation_items = list(key_relations)
        if key_relation_items:
            role_name_for_pattern = clean_text(role_dir_name).replace("_", " ")
            for item in key_relation_items[:8]:
                if not isinstance(item, dict):
                    continue
                description = clean_text(item.get("description"))
                if description:
                    alias_patterns = [
                        rf"{re.escape(role_name_for_pattern)}\s*\(posing as ([^)]+)\)",
                        rf"{re.escape(role_name_for_pattern)}\s*\(also known as ([^)]+)\)",
                    ]
                    for pattern in alias_patterns:
                        for match in re.findall(pattern, description, flags=re.IGNORECASE):
                            alias = clean_text(match)
                            if alias:
                                raw_names.append(alias)

        names: List[str] = []
        seen_names: Set[str] = set()
        for value in raw_names:
            if value and value not in seen_names:
                names.append(value)
                seen_names.add(value)

        profiles.append(
            {
                "role_dir_name": role_dir_name,
                "names": names,
            }
        )
    return profiles


def _simplify_for_match(text: str) -> str:
    lowered = clean_text(text).replace("_", " ").lower()
    chars: List[str] = []
    previous_space = False
    for ch in lowered:
        if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"):
            chars.append(ch)
            previous_space = False
        else:
            if not previous_space:
                chars.append(" ")
                previous_space = True
    return " ".join("".join(chars).split())


def _name_token_variants(text: str) -> Set[str]:
    simplified = _simplify_for_match(text)
    variants: Set[str] = set()
    if not simplified:
        return variants

    variants.add(simplified)
    variants.add(simplified.replace(" ", ""))

    tokens = [token for token in simplified.split() if token]
    if tokens:
        variants.add(" ".join(tokens))
        variants.add("".join(tokens))
        filtered = [token for token in tokens if token not in {"mr", "mrs", "ms", "miss", "dr", "doctor", "capt", "captain", "officer", "sir"}]
        if filtered:
            variants.add(" ".join(filtered))
            variants.add("".join(filtered))
            variants.add(filtered[-1])
    return {value for value in variants if value}


def _match_score(name_a: str, name_b: str) -> Tuple[int, str]:
    variants_a = _name_token_variants(name_a)
    variants_b = _name_token_variants(name_b)
    if not variants_a or not variants_b:
        return (0, "none")

    shared = variants_a & variants_b
    if shared:
        best = max(shared, key=len)
        if " " in best:
            return (120, "exact_normalized")
        if len(best) >= 4:
            return (115, "exact_compact")
        return (108, "exact_short")

    simplified_a = _simplify_for_match(name_a)
    simplified_b = _simplify_for_match(name_b)
    tokens_a = {token for token in simplified_a.split() if token}
    tokens_b = {token for token in simplified_b.split() if token}
    if tokens_a and tokens_b:
        if tokens_a == tokens_b:
            return (112, "same_token_set")
        if tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a):
            return (104, "token_subset")
        overlap = tokens_a & tokens_b
        if overlap:
            ratio = len(overlap) / max(min(len(tokens_a), len(tokens_b)), 1)
            if ratio >= 0.8:
                return (96, "token_overlap_high")
            if ratio >= 0.5 and len(overlap) >= 1:
                return (88, "token_overlap")

    compact_a = simplified_a.replace(" ", "")
    compact_b = simplified_b.replace(" ", "")
    if compact_a and compact_b:
        if compact_a in compact_b or compact_b in compact_a:
            return (84, "substring")
        if compact_a.endswith(compact_b) or compact_b.endswith(compact_a):
            return (82, "suffix_match")
    return (0, "none")


def build_character_degree_ranking(
    extraction_results: Any,
    rename_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    node_meta: Dict[str, Dict[str, Any]] = {}
    degree_counts: Dict[str, int] = defaultdict(int)

    if isinstance(extraction_results, dict):
        chunks = list(extraction_results.values())
    elif isinstance(extraction_results, list):
        chunks = extraction_results
    else:
        chunks = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        entities = chunk.get("entities", []) or []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            raw_name = clean_text(entity.get("name"))
            if not raw_name:
                continue
            normalized = normalize_character_label(raw_name, rename_map)
            meta = node_meta.setdefault(
                normalized,
                {
                    "normalized_name": normalized,
                    "observed_names": set(),
                    "type_counts": defaultdict(int),
                    "scope_counts": defaultdict(int),
                },
            )
            meta["observed_names"].add(raw_name)
            entity_type = clean_text(entity.get("type"))
            if entity_type:
                meta["type_counts"][entity_type] += 1
            entity_scope = clean_text(entity.get("scope"))
            if entity_scope:
                meta["scope_counts"][entity_scope.lower()] += 1

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for relation in chunk.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            subject = normalize_character_label(clean_text(relation.get("subject")), rename_map)
            obj = normalize_character_label(clean_text(relation.get("object")), rename_map)
            if subject in node_meta and obj in node_meta:
                degree_counts[subject] += 1
                degree_counts[obj] += 1

    ranked: List[Dict[str, Any]] = []
    for normalized_name, meta in node_meta.items():
        type_counts = meta["type_counts"]
        scope_counts = meta["scope_counts"]
        node_type = max(type_counts.items(), key=lambda item: item[1])[0] if type_counts else ""
        node_scope = max(scope_counts.items(), key=lambda item: item[1])[0] if scope_counts else ""
        if node_type != "Character" or node_scope == "local":
            continue
        observed_names = sorted(meta["observed_names"], key=lambda value: (-len(value), value))
        ranked.append(
            {
                "normalized_name": normalized_name,
                "observed_names": observed_names,
                "degree": degree_counts.get(normalized_name, 0),
            }
        )
    ranked.sort(key=lambda item: (-item["degree"], item["normalized_name"]))
    return ranked


def normalize_local_document_for_builder(doc: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(doc)
    atomic_units = normalized.get("atomic_event_units")
    if atomic_units is None:
        atomic_units = doc.get("atomic_event_candidates", [])
    normalized["atomic_event_units"] = list(atomic_units or [])

    occasion_frame = normalized.get("occasion_frame")
    if occasion_frame is None:
        occasion_frame = doc.get("occasion_frame_candidate", {})
    normalized["occasion_frame"] = dict(occasion_frame or {})

    state_updates = normalized.get("scene_state_updates")
    normalized["scene_state_updates"] = list(state_updates or [])
    return normalized


def load_local_story_dynamics_for_builder(movie_dir: Path) -> Dict[str, Any]:
    refined_path = movie_dir / "story_dynamics_local_refined.json"
    legacy_v2_path = movie_dir / "story_dynamics_local_v2.json"
    legacy_v1_path = movie_dir / "story_dynamics_local.json"

    if refined_path.exists():
        payload = load_json(refined_path)
        normalized = dict(payload)
        normalized["documents"] = [
            normalize_local_document_for_builder(doc)
            for doc in (payload.get("documents", []) or [])
        ]
        normalized["_builder_local_source"] = "story_dynamics_local_refined.json"
        return normalized

    if legacy_v2_path.exists():
        payload = load_json(legacy_v2_path)
        payload["_builder_local_source"] = "story_dynamics_local_v2.json"
        return payload

    if legacy_v1_path.exists():
        payload = load_json(legacy_v1_path)
        normalized = dict(payload)
        normalized["documents"] = [
            normalize_local_document_for_builder(doc)
            for doc in (payload.get("documents", []) or [])
        ]
        normalized["_builder_local_source"] = "story_dynamics_local.json"
        return normalized

    raise FileNotFoundError(f"missing local story dynamics file in {movie_dir}")


def normalize_character_label(name: str, rename_map: Dict[str, Any]) -> str:
    cleaned = clean_text(name).replace("_", " ")
    return normalize_name(cleaned, rename_map)


def select_characters_aligned_with_icrp(
    movie_dir: Path,
    candidates: Sequence[Dict[str, Any]],
    rename_map: Dict[str, Any],
    max_characters: int,
    extraction_results: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    role_profiles = load_icrp_role_profiles(movie_dir)
    if not role_profiles:
        return []

    candidate_rows = list(candidates)
    if not candidate_rows:
        return []

    graph_ranked = build_character_degree_ranking(extraction_results, rename_map) if extraction_results is not None else []
    graph_name_rank: Dict[str, int] = {}
    for index, row in enumerate(graph_ranked):
        graph_name_rank[row["normalized_name"]] = index

    candidate_infos: List[Dict[str, Any]] = []
    covered_candidate_names: Set[str] = set()
    for candidate in candidate_rows:
        canonical_name = clean_text(candidate.get("canonical_name"))
        aliases = [clean_text(alias) for alias in (candidate.get("aliases", []) or []) if clean_text(alias)]
        names = [canonical_name] + [alias for alias in aliases if alias != canonical_name]
        normalized_names = [normalize_character_label(name, rename_map) for name in names if name]
        covered_candidate_names.update(name for name in normalized_names if name)
        graph_rank = min(
            (graph_name_rank.get(name) for name in normalized_names if name in graph_name_rank),
            default=10_000,
        )
        candidate_infos.append(
            {
                "candidate": candidate,
                "names": names,
                "normalized_names": normalized_names,
                "graph_rank": graph_rank,
            }
        )

    for row in graph_ranked[:20]:
        normalized_name = row["normalized_name"]
        if normalized_name in covered_candidate_names:
            continue
        observed_names = [clean_text(name) for name in (row.get("observed_names", []) or []) if clean_text(name)]
        if not observed_names:
            continue
        canonical_name = observed_names[0]
        synthetic_candidate = {
            "canonical_name": canonical_name,
            "aliases": sorted(set(observed_names)),
            "score": float(row.get("degree", 0)),
            "scene_count": 0,
            "_scene_ids": [],
            "event_count": 0,
            "update_count": 0,
            "relation_update_count": 0,
            "status_update_count": 0,
            "episode_count": 0,
            "timeline_worthiness": 0.0,
            "task3_readiness": 0.0,
            "entity_types": {"Character": 1},
            "selection_origin": "extraction_graph_fallback",
        }
        normalized_names = [normalize_character_label(name, rename_map) for name in observed_names if name]
        candidate_infos.append(
            {
                "candidate": synthetic_candidate,
                "names": observed_names,
                "normalized_names": normalized_names,
                "graph_rank": graph_name_rank.get(normalized_name, 10_000),
            }
        )

    role_profiles = role_profiles[:max_characters]
    role_candidates: List[List[Tuple[int, int, str, str]]] = []
    for role in role_profiles:
        candidate_scores: List[Tuple[int, int, str, str]] = []
        for candidate_index, info in enumerate(candidate_infos):
            best_score = 0
            best_method = "none"
            best_role_name = ""
            for role_name in (role.get("names", []) or []):
                for candidate_name in info["names"]:
                    score, method = _match_score(role_name, candidate_name)
                    if score > best_score:
                        best_score = score
                        best_method = method
                        best_role_name = role_name
            if best_score >= 80:
                score_with_rank = best_score - min(info["graph_rank"], 500) * 0.01
                candidate_scores.append((candidate_index, int(score_with_rank * 100), best_method, best_role_name))
        candidate_scores.sort(key=lambda row: (-row[1], row[0]))
        role_candidates.append(candidate_scores[:6])

    role_order = sorted(range(len(role_profiles)), key=lambda idx: (len(role_candidates[idx]), idx))
    best_assignment: Optional[Tuple[int, Tuple[int, ...], List[Tuple[int, int, str, str]]]] = None
    candidate_pool = sorted({row[0] for rows in role_candidates for row in rows})
    assignment_size = min(len(role_profiles), len(candidate_pool))
    for perm in permutations(candidate_pool, assignment_size):
        score_total = 0
        assignment_rows: List[Tuple[int, int, str, str]] = []
        valid = True
        for role_offset, role_index in enumerate(role_order[:assignment_size]):
            chosen_candidate_index = perm[role_offset]
            score_row = next((row for row in role_candidates[role_index] if row[0] == chosen_candidate_index), None)
            if score_row is None:
                valid = False
                break
            assignment_rows.append((role_index, *score_row))
            score_total += score_row[1]
        if not valid:
            continue
        if best_assignment is None or score_total > best_assignment[0]:
            best_assignment = (score_total, perm, assignment_rows)

    assigned_by_role: Dict[int, Tuple[int, int, str, str]] = {}
    if best_assignment is not None:
        for role_index, candidate_index, score_value, method, matched_role_name in best_assignment[2]:
            assigned_by_role[role_index] = (candidate_index, score_value, method, matched_role_name)

    selected: List[Dict[str, Any]] = []
    for role_index, role in enumerate(role_profiles):
        assignment = assigned_by_role.get(role_index)
        if assignment is None:
            continue
        candidate_index, score_value, method, matched_role_name = assignment
        candidate = dict(candidate_infos[candidate_index]["candidate"])
        candidate["selection_reason"] = "Aligned with the movie's ICRP character set."
        candidate["task3_relevance"] = "Directly aligned with the same key character used in ICRP."
        candidate["icrp_aligned_name"] = role.get("role_dir_name", "")
        candidate["icrp_role_names"] = list(role.get("names", []))
        candidate["icrp_match_name"] = matched_role_name
        candidate["icrp_match_method"] = method
        candidate["icrp_match_score"] = round(score_value / 100.0, 2)
        selected.append(candidate)
    return selected


LOCAL_NODE_PROMPT_EN = """You are building gold-quality focal-character timeline nodes for a screenplay benchmark.

You are in builder mode, not solver mode.
This means you should use the richer structured evidence to construct a high-quality target timeline.

Goal:
1. inspect each scene packet for whether it contains a meaningful development for the focal character;
2. keep only packets that mark a durable, narratively meaningful development;
3. write compact, scene-grounded node descriptions that will later become benchmark gold.

Keep a scene only when it contributes at least one of:
- a clear goal shift, plan shift, or motivation shift
- a durable status change
- a meaningful relationship update
- a turning point, setback, escalation, disclosure, or decision
- a scene that becomes important for later persona consistency

Do not keep a scene if it only repeats ongoing background with no new development.

Rules:
- Stay grounded in the supplied evidence.
- Do not invent motives, emotions, or interpretations not supported here.
- Prefer concrete narrative changes over abstract praise or literary interpretation.
- Avoid inflated phrases like "symbolic carrier", "national representative", "spiritual elevation", or other broad claims unless the evidence explicitly supports them.
- `salient_development` should describe what changes for the character in this scene.
- `goal_state` may be null.
- `relation_updates` and `status_updates` should be short natural phrases.
- `persona_anchor` should be one short first-person-compatible memory cue.
- Output JSON only.

Return format:
{
  "proposed_nodes": [
    {
      "bundle_id": "string",
      "should_keep": true,
      "importance": "core|supporting|context",
      "role_in_context": "short grounded role",
      "salient_development": "short grounded development",
      "goal_state": "string or null",
      "relation_updates": ["string", "..."],
      "status_updates": ["string", "..."],
      "resulting_state": "string or null",
      "unresolved_issue": "string or null",
      "persona_anchor": "string or null"
    }
  ]
}
"""


LOCAL_NODE_PROMPT_ZH = """你在为 screenplay benchmark 构建金标质量的焦点角色时间线节点。

当前是 builder 模式，不是 solver 模式。
也就是说，你可以利用更丰富的结构化证据，构造高质量的目标时间线。

目标：
1. 逐个检查 scene packet，判断它是否构成该角色的重要发展节点；
2. 只保留那些具有持续叙事意义的发展；
3. 把它们写成紧凑、贴地、可作为 benchmark gold 的 scene-grounded 节点。

保留 scene 的条件至少包括其一：
- 明确的目标变化、计划变化、动机转向
- 持续性的处境或状态变化
- 有意义的关系更新
- 转折、受阻、升级、揭示、决策
- 对后续 persona consistency 很关键的经历

如果一个 scene 只是重复背景、没有新的角色发展，就不要保留。

要求：
- 必须严格基于给定证据。
- 不要脑补没有明确支持的动机、情绪或解释。
- 优先写具体的叙事变化，不要写空泛拔高的评价性大词。
- 避免“精神升华”“全国性代表”“象征性载体”这类过度拔高、难以回证的表述，除非证据非常明确。
- `salient_development` 要写这个 scene 对角色造成了什么变化。
- `goal_state` 可以为 null。
- `relation_updates` 和 `status_updates` 都写成简短自然语言短语。
- `persona_anchor` 写成一句简短、可兼容第一人称记忆的锚点。
- 只输出 JSON。

输出格式：
{
  "proposed_nodes": [
    {
      "bundle_id": "字符串",
      "should_keep": true,
      "importance": "core|supporting|context",
      "role_in_context": "简短且贴地的角色定位",
      "salient_development": "简短且贴地的发展概括",
      "goal_state": "字符串或 null",
      "relation_updates": ["字符串", "..."],
      "status_updates": ["字符串", "..."],
      "resulting_state": "字符串或 null",
      "unresolved_issue": "字符串或 null",
      "persona_anchor": "字符串或 null"
    }
  ]
}
"""


GLOBAL_REFINE_PROMPT_EN = """You are refining candidate focal-character timeline nodes into a compact gold timeline.

You will receive chronologically ordered proposed nodes for one focal character.

Goal:
1. keep a compact but representative set of nodes;
2. remove redundancy;
3. preserve the character's long-range development;
4. rewrite the kept nodes so they are consistent, specific, and benchmark-ready.

Rules:
- Prefer coverage of the full trajectory over dense clustering in one portion of the script.
- Keep scene-grounded nodes. Do not merge multiple scenes into one node.
- Only keep redundant adjacent nodes if each changes the character's state in a distinct way.
- Prioritize nodes that would matter for downstream persona consistency and long-context reasoning.
- Keep the wording conservative and scene-grounded.
- Avoid literary inflation, historical overclaim, and unsupported prestige language.
- `resulting_state` should describe the immediate downstream character state, not a broad retrospective judgment.
- Output between {min_nodes} and {max_nodes} nodes when the evidence supports that many.
- Output JSON only.

Return format:
{
  "timeline_summary": "2-4 sentence summary",
  "final_nodes": [
    {
      "timeline_node_id": "string",
      "role_in_context": "short grounded role",
      "salient_development": "short grounded development",
      "goal_state": "string or null",
      "relation_updates": ["string", "..."],
      "status_updates": ["string", "..."],
      "resulting_state": "string or null",
      "unresolved_issue": "string or null",
      "persona_anchor": "string or null"
    }
  ]
}
"""


GLOBAL_REFINE_PROMPT_ZH = """你在把候选的焦点角色时间线节点精修成紧凑的金标时间线。

你会收到同一角色按时间排序的候选节点。

目标：
1. 保留一组紧凑但有代表性的节点；
2. 去掉冗余；
3. 保住这个角色的长程发展轨迹；
4. 把保留节点改写得更一致、更具体、更适合作为 benchmark gold。

要求：
- 优先覆盖整个角色轨迹，而不是在某一小段过密堆叠。
- 保持 scene-grounded，不要把多个 scene 合成一个节点。
- 相邻节点如果本质上重复，就不要都保留；只有它们分别带来不同状态变化时才同时保留。
- 优先保留那些对后续 persona consistency 和长上下文推理重要的节点。
- 语言保持保守、贴地、可回证，不要写文学化拔高或历史性过度概括。
- `resulting_state` 只写该节点之后的直接角色状态，不要写过大的回顾性判断。
- 如果证据充分，输出 {min_nodes} 到 {max_nodes} 个节点。
- 只输出 JSON。

输出格式：
{
  "timeline_summary": "2-4句概括",
  "final_nodes": [
    {
      "timeline_node_id": "字符串",
      "role_in_context": "简短且贴地的角色定位",
      "salient_development": "简短且贴地的发展概括",
      "goal_state": "字符串或 null",
      "relation_updates": ["字符串", "..."],
      "status_updates": ["字符串", "..."],
      "resulting_state": "字符串或 null",
      "unresolved_issue": "字符串或 null",
      "persona_anchor": "字符串或 null"
    }
  ]
}
"""


ARC_PROMPT_EN = """You are deriving higher-level cross-scene arcs from a refined focal-character timeline.

Goal:
1. connect timeline nodes into a small number of durable development threads;
2. make each arc reflect a meaningful trajectory such as goal progression, relationship evolution, conflict progression, or status change;
3. keep the arcs grounded in the supplied nodes.

Rules:
- Each arc must link at least 2 timeline nodes.
- Do not make one arc per scene.
- Prefer 2-6 arcs depending on evidence.
- Only use the provided timeline node ids.
- Keep the arc wording grounded and restrained. Avoid inflated historical claims and vague philosophical praise.
- Output JSON only.

Return format:
{
  "arcs": [
    {
      "title": "short arc title",
      "arc_focus": "goal|relationship|status|conflict|mixed",
      "linked_timeline_node_ids": ["string", "..."],
      "arc_summary": "short grounded summary",
      "start_state": "string or null",
      "end_state": "string or null",
      "unresolved_issue": "string or null"
    }
  ]
}
"""


ARC_PROMPT_ZH = """你在根据精修后的焦点角色时间线归纳更高层的跨场景 arcs。

目标：
1. 把多个 timeline nodes 连成少量、持续性的角色发展线程；
2. 每条 arc 都应体现有意义的轨迹，例如目标推进、关系演化、冲突推进、处境变化等；
3. arc 必须严格基于给定节点。

要求：
- 每条 arc 至少连接 2 个 timeline nodes。
- 不要一场戏做一条 arc。
- 根据证据强弱，优先输出 2-6 条 arc。
- 只能使用给定的 timeline node ids。
- arc 的表述要克制、贴地，不要写历史性拔高或空泛的哲学赞辞。
- 只输出 JSON。

输出格式：
{
  "arcs": [
    {
      "title": "简短标题",
      "arc_focus": "goal|relationship|status|conflict|mixed",
      "linked_timeline_node_ids": ["字符串", "..."],
      "arc_summary": "简短且贴地的总结",
      "start_state": "字符串或 null",
      "end_state": "字符串或 null",
      "unresolved_issue": "字符串或 null"
    }
  ]
}
"""


FINAL_NODE_GROUNDING_PROMPT_EN = """You are grounding and conservatively rewriting a final focal-character timeline node.

You will receive:
- the focal character
- one draft final node
- the full scene text
- supporting structured cues

Your job:
1. rewrite the node so the wording is concrete, conservative, and scene-grounded;
2. avoid prestige language, literary over-interpretation, and unsupported abstraction;
3. extract 1-4 exact evidence quotes copied verbatim from the scene text when possible.

Rules:
- Do not invent facts outside the scene packet.
- Keep the rewrite compact.
- `resulting_state` should be the immediate state after this scene, not a broad historical verdict.
- `evidence_quotes` must be exact spans from the supplied scene text. If none can be confidently located, return an empty list.
- Output JSON only.

Return format:
{
  "timeline_node_id": "string",
  "role_in_context": "string",
  "salient_development": "string",
  "goal_state": "string or null",
  "resulting_state": "string or null",
  "unresolved_issue": "string or null",
  "persona_anchor": "string or null",
  "evidence_quotes": ["string", "..."]
}
"""


FINAL_NODE_GROUNDING_PROMPT_ZH = """你在对最终的焦点角色 timeline node 做贴地校正。

你会收到：
- focal character
- 一个当前版本的 final node
- 对应 scene 的完整文本
- 一些辅助结构线索

你的任务：
1. 把该节点改写得更具体、更保守、更贴地；
2. 避免空泛拔高、文学化解释和缺乏证据的大词；
3. 尽量从 scene text 中抽取 1-4 条逐字证据片段。

要求：
- 不要引入 scene packet 之外的新事实。
- 改写保持紧凑。
- `resulting_state` 只写该 scene 之后的直接状态，不要写过大的历史总结。
- `evidence_quotes` 必须是从给定 scene text 中逐字拷贝的片段；如果拿不准，就返回空列表。
- 只输出 JSON。

输出格式：
{
  "timeline_node_id": "字符串",
  "role_in_context": "字符串",
  "salient_development": "字符串",
  "goal_state": "字符串或 null",
  "resulting_state": "字符串或 null",
  "unresolved_issue": "字符串或 null",
  "persona_anchor": "字符串或 null",
  "evidence_quotes": ["字符串", "..."]
}
"""


def normalize_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    output: List[str] = []
    seen: Set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def chunked(items: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def build_state_fact_index(state_facts: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fact in state_facts:
        scene_key = clean_text(fact.get("scene_key"))
        if scene_key:
            by_scene[scene_key].append(fact)
    return by_scene


def collect_bundle_evidence_quotes(bundle: Dict[str, Any], limit: int = 4) -> List[str]:
    quotes: List[str] = []
    seen: Set[str] = set()
    scene_text = clean_text(bundle.get("scene_text_full", ""))

    def add(text: str) -> None:
        cleaned = clean_text(text)
        if not cleaned or cleaned in seen or len(quotes) >= limit:
            return
        if scene_text and cleaned not in scene_text:
            return
        if len(cleaned) > 140:
            return
        if cleaned.count("，") + cleaned.count(",") > 3 and len(cleaned) > 60:
            return
        if any(marker in cleaned for marker in ("明确", "表明", "体现", "说明", "establish", "indicating", "showing")) and cleaned not in scene_text:
            return
        if cleaned and cleaned not in seen and len(quotes) < limit:
            seen.add(cleaned)
            quotes.append(cleaned)

    for event in (bundle.get("active_events", []) or []):
        add(event.get("evidence_text", ""))
    for update in (bundle.get("active_updates", []) or []):
        for evidence in (update.get("evidence", []) or []):
            add(evidence)
    return quotes


def normalize_scene_fact(
    fact: Dict[str, Any],
    aliases: Set[str],
    rename_map: Dict[str, str],
    language: str,
) -> Optional[Dict[str, Any]]:
    if not should_keep_fact(fact, language):
        return None
    subject_name = normalize_name(fact.get("subject_name", ""), rename_map)
    value_text = normalize_name(fact.get("value_text", ""), rename_map)
    matched = aliases & {subject_name, value_text}
    if not matched:
        return None
    matched_role = "subject" if subject_name in aliases else "value"
    return {
        "fact_id": clean_text(fact.get("fact_id")),
        "matched_role": matched_role,
        "state_type": fact_to_state_type(fact) or "state",
        "predicate": canonical_predicate(fact),
        "subject_name": subject_name or clean_text(fact.get("subject_name")),
        "value_text": value_text or clean_text(fact.get("value_text")),
        "persistence": clean_text(fact.get("original_persistence")) or clean_text(fact.get("temporal_scope")),
        "evidence_text": clean_text(fact.get("evidence_text"))[:280],
    }


def build_character_scene_bundles(
    movie_dir: Path,
    language: str,
    local_data: Dict[str, Any],
    global_data: Dict[str, Any],
    state_facts: Sequence[Dict[str, Any]],
    rename_map: Dict[str, str],
    selection: Dict[str, Any],
) -> List[Dict[str, Any]]:
    doc2chunks = load_json(movie_dir / "doc2chunks.json")
    scene_text_map = build_scene_text_map(movie_dir, doc2chunks)
    _, episodes_by_doc = build_episode_indexes(global_data)
    facts_by_scene = build_state_fact_index(state_facts)

    aliases = {normalize_name(alias, rename_map) for alias in (selection.get("aliases", []) or []) if clean_text(alias)}
    aliases.add(normalize_name(selection["canonical_name"], rename_map))
    aliases = {alias for alias in aliases if alias}

    bundles: List[Dict[str, Any]] = []
    for doc in sorted(local_data.get("documents", []) or [], key=lambda item: item.get("scene_order", 0)):
        document_id = clean_text(doc.get("document_id"))
        if not document_id:
            continue
        active_events: List[Dict[str, Any]] = []
        active_event_ids: Set[str] = set()
        for event in doc.get("atomic_event_units", []) or []:
            participants = {normalize_name(name, rename_map) for name in (event.get("participants", []) or []) if clean_text(name)}
            targets = {normalize_name(name, rename_map) for name in (event.get("targets", []) or []) if clean_text(name)}
            if aliases & (participants | targets):
                event_id = clean_text(event.get("event_id"))
                active_events.append(
                    {
                        "event_id": event_id,
                        "event_text": clean_text(event.get("event_text")),
                        "event_description": clean_text(event.get("event_description")),
                        "participants": normalize_string_list(event.get("participants", []) or []),
                        "targets": normalize_string_list(event.get("targets", []) or []),
                        "evidence_text": clean_text(event.get("evidence", {}).get("evidence_text")),
                    }
                )
                if event_id:
                    active_event_ids.add(event_id)

        active_updates: List[Dict[str, Any]] = []
        for update in doc.get("scene_state_updates", []) or []:
            subject = normalize_name(update.get("subject", ""), rename_map)
            value = normalize_name(update.get("value", ""), rename_map)
            new_value = normalize_name(update.get("new_value", ""), rename_map)
            if aliases & {subject, value, new_value}:
                if not keep_scene_update(update):
                    continue
                active_updates.append(
                    {
                        "update_id": clean_text(update.get("update_id")),
                        "predicate": clean_text(update.get("predicate")),
                        "subject": clean_text(update.get("subject")),
                        "value": clean_text(update.get("value")),
                        "new_value": clean_text(update.get("new_value")),
                        "evidence": normalize_string_list(update.get("evidence", []) or [])[:3],
                    }
                )

        scene_facts: List[Dict[str, Any]] = []
        for fact in (facts_by_scene.get(document_id, []) or []):
            normalized = normalize_scene_fact(fact, aliases, rename_map, language)
            if normalized:
                scene_facts.append(normalized)

        related_episodes = related_episode_briefs(aliases, document_id, active_event_ids, episodes_by_doc)
        if not active_events and not active_updates and not related_episodes and not scene_facts:
            continue

        scene_text = clean_text(scene_text_map.get(document_id, ""))
        bundles.append(
            {
                "bundle_id": stable_id(selection["canonical_name"], document_id, prefix="tb"),
                "document_id": document_id,
                "scene_id": clean_text(doc.get("scene_id")),
                "scene_order": doc.get("scene_order"),
                "scene_title": clean_text(doc.get("scene_title")),
                "scene_summary": clean_text(doc.get("scene_summary")),
                "scene_text_full": scene_text,
                "scene_text_excerpt": scene_text[:1600],
                "occasion_frame": {
                    "time_space_frame": clean_text(doc.get("occasion_frame", {}).get("time_space_frame")),
                    "interaction_frame": clean_text(doc.get("occasion_frame", {}).get("interaction_frame")),
                    "frame_summary": clean_text(doc.get("occasion_frame", {}).get("frame_summary")),
                },
                "active_events": active_events[:10],
                "active_updates": active_updates[:8],
                "scene_state_facts": scene_facts[:8],
                "related_episodes": related_episodes[:4],
            }
        )
    return bundles


def propose_nodes_for_batch(
    client: ChatCompletionsClient,
    language: str,
    focal_character: str,
    batch: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prompt = LOCAL_NODE_PROMPT_ZH if language == "zh" else LOCAL_NODE_PROMPT_EN
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "focal_character": focal_character,
                    "scene_packets": list(batch),
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    data = chat_json(client, messages, temperature=0.0)
    items = data.get("proposed_nodes", [])
    return items if isinstance(items, list) else []


def build_proposed_timeline(
    client: ChatCompletionsClient,
    language: str,
    selection: Dict[str, Any],
    bundles: Sequence[Dict[str, Any]],
    batch_size: int,
    max_workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    bundle_lookup = {bundle["bundle_id"]: bundle for bundle in bundles}
    raw_items: List[Dict[str, Any]] = []
    print(
        json.dumps(
            {
                "stage": "builder_local_proposals_start",
                "character": selection["canonical_name"],
                "bundle_count": len(bundles),
                "batch_size": batch_size,
                "max_workers": max_workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                propose_nodes_for_batch,
                client,
                language,
                selection["canonical_name"],
                batch,
            ): batch
            for batch in chunked(bundles, batch_size)
        }
        for future in as_completed(futures):
            batch_items = future.result()
            raw_items.extend(batch_items)
            print(
                json.dumps(
                    {
                        "stage": "builder_local_batch_done",
                        "character": selection["canonical_name"],
                        "returned_count": len(batch_items),
                        "accumulated_count": len(raw_items),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    proposed_nodes: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    seen_node_ids: Set[str] = set()
    for item in raw_items:
        bundle_id = clean_text(item.get("bundle_id"))
        if not bundle_id or bundle_id not in bundle_lookup:
            continue
        bundle = bundle_lookup[bundle_id]
        timeline_node_id = stable_id(selection["canonical_name"], bundle["document_id"], prefix="ctu")
        audit_rows.append(
            {
                "bundle_id": bundle_id,
                "timeline_node_id": timeline_node_id,
                "scene_order": bundle.get("scene_order"),
                "scene_title": bundle.get("scene_title"),
                "should_keep": bool(item.get("should_keep", True)),
                "importance": clean_text(item.get("importance")),
                "salient_development": clean_text(item.get("salient_development")),
            }
        )
        if not item.get("should_keep", True):
            continue
        if timeline_node_id in seen_node_ids:
            continue
        seen_node_ids.add(timeline_node_id)
        proposed_nodes.append(
            {
                "timeline_node_id": timeline_node_id,
                "document_id": bundle["document_id"],
                "scene_id": bundle["scene_id"],
                "scene_order": bundle["scene_order"],
                "scene_title": bundle["scene_title"],
                "scene_summary": bundle["scene_summary"],
                "role_in_context": clean_text(item.get("role_in_context")),
                "salient_development": clean_text(item.get("salient_development")),
                "goal_state": item.get("goal_state"),
                "resulting_state": item.get("resulting_state"),
                "unresolved_issue": item.get("unresolved_issue"),
                "importance": clean_text(item.get("importance")) or "supporting",
                "related_event_ids": [clean_text(ev.get("event_id")) for ev in (bundle.get("active_events", []) or []) if clean_text(ev.get("event_id"))],
                "related_episode_ids": [clean_text(ep.get("episode_id")) for ep in (bundle.get("related_episodes", []) or []) if clean_text(ep.get("episode_id"))],
                "evidence_quotes": collect_bundle_evidence_quotes(bundle, limit=4),
                "auxiliary": {
                    "relation_updates": filter_relation_updates(item.get("relation_updates", []), language),
                    "status_updates": filter_status_updates(item.get("status_updates", []), language),
                    "persona_anchor": shorten_persona_anchor(item.get("persona_anchor"), language),
                },
            }
        )
    proposed_nodes.sort(key=lambda item: (item.get("scene_order", 0), item.get("document_id", "")))
    audit_rows.sort(key=lambda item: (item.get("scene_order", 0), item.get("bundle_id", "")))
    return proposed_nodes, audit_rows


def compact_node_for_refine(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timeline_node_id": node.get("timeline_node_id"),
        "scene_order": node.get("scene_order"),
        "scene_title": node.get("scene_title"),
        "scene_summary": node.get("scene_summary"),
        "importance": node.get("importance"),
        "role_in_context": node.get("role_in_context"),
        "salient_development": node.get("salient_development"),
        "goal_state": node.get("goal_state"),
        "resulting_state": node.get("resulting_state"),
        "unresolved_issue": node.get("unresolved_issue"),
        "relation_updates": node.get("auxiliary", {}).get("relation_updates", [])[:3],
        "status_updates": node.get("auxiliary", {}).get("status_updates", [])[:3],
        "evidence_quotes": node.get("evidence_quotes", [])[:2],
    }


MAX_GROUNDING_SCENE_TEXT_CHARS = 12000
MAX_SUPPORT_EVIDENCE_CHARS = 300
MAX_REFINE_NODE_COUNT = 72
MAX_ARC_NODE_COUNT = 24


def truncate_text(value: Any, limit: int) -> str:
    text = clean_text(value)
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def select_nodes_for_refine_payload(
    proposed_nodes: Sequence[Dict[str, Any]],
    max_nodes: int,
) -> List[Dict[str, Any]]:
    ordered = sorted(proposed_nodes, key=lambda item: (item.get("scene_order", 0), item.get("document_id", "")))
    if len(ordered) <= max_nodes:
        return list(ordered)
    sampled: List[Dict[str, Any]] = []
    total = len(ordered)
    for bucket_index in range(max_nodes):
        start = round(bucket_index * total / max_nodes)
        end = round((bucket_index + 1) * total / max_nodes)
        window = ordered[start:end] or [ordered[min(start, total - 1)]]
        best = max(
            window,
            key=lambda item: (
                2 if item.get("importance") == "core" else 1 if item.get("importance") == "supporting" else 0,
                len(item.get("related_event_ids", [])),
                len(item.get("related_episode_ids", [])),
                len(clean_text(item.get("salient_development"))),
            ),
        )
        sampled.append(best)
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in sampled:
        node_id = clean_text(item.get("timeline_node_id"))
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        deduped.append(item)
    return deduped


def validate_exact_quotes(scene_text: str, values: Any, limit: int = 4) -> List[str]:
    cleaned_scene = clean_text(scene_text)
    if not cleaned_scene or not isinstance(values, list):
        return []
    quotes: List[str] = []
    seen: Set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        if cleaned not in cleaned_scene:
            continue
        if len(cleaned) > 140:
            continue
        seen.add(cleaned)
        quotes.append(cleaned)
        if len(quotes) >= limit:
            break
    return quotes


def fallback_refine_nodes(
    proposed_nodes: Sequence[Dict[str, Any]],
    max_final_nodes: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    ordered = sorted(proposed_nodes, key=lambda item: (item.get("scene_order", 0), item.get("document_id", "")))
    if len(ordered) <= max_final_nodes:
        kept = ordered
    else:
        kept = []
        total = len(ordered)
        for bucket_index in range(max_final_nodes):
            start = round(bucket_index * total / max_final_nodes)
            end = round((bucket_index + 1) * total / max_final_nodes)
            window = ordered[start:end] or [ordered[min(start, total - 1)]]
            best = max(
                window,
                key=lambda item: (
                    2 if item.get("importance") == "core" else 1 if item.get("importance") == "supporting" else 0,
                    len(item.get("related_event_ids", [])),
                    len(item.get("related_episode_ids", [])),
                ),
            )
            kept.append(best)
        deduped: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for item in kept:
            node_id = clean_text(item.get("timeline_node_id"))
            if node_id and node_id not in seen:
                seen.add(node_id)
                deduped.append(item)
        kept = deduped
    summary = ""
    if kept:
        summary = f"The timeline traces {len(kept)} key developments across the character's screenplay trajectory."
    return summary, kept


def ground_single_final_node(
    client: ChatCompletionsClient,
    language: str,
    character_name: str,
    node: Dict[str, Any],
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = FINAL_NODE_GROUNDING_PROMPT_ZH if language == "zh" else FINAL_NODE_GROUNDING_PROMPT_EN
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "focal_character": character_name,
                    "draft_node": {
                        "timeline_node_id": node.get("timeline_node_id"),
                        "role_in_context": node.get("role_in_context"),
                        "salient_development": node.get("salient_development"),
                        "goal_state": node.get("goal_state"),
                        "resulting_state": node.get("resulting_state"),
                        "unresolved_issue": node.get("unresolved_issue"),
                        "persona_anchor": node.get("auxiliary", {}).get("persona_anchor"),
                    },
                    "scene_title": bundle.get("scene_title"),
                    "scene_summary": bundle.get("scene_summary"),
                    "scene_text": truncate_text(bundle.get("scene_text_full"), MAX_GROUNDING_SCENE_TEXT_CHARS),
                    "structured_support": {
                        "related_event_ids": node.get("related_event_ids", []),
                        "related_episode_ids": node.get("related_episode_ids", []),
                        "relation_updates": node.get("auxiliary", {}).get("relation_updates", []),
                        "status_updates": node.get("auxiliary", {}).get("status_updates", []),
                        "active_event_evidence": [
                            truncate_text(ev.get("evidence_text"), MAX_SUPPORT_EVIDENCE_CHARS)
                    for ev in (bundle.get("active_events", []) or [])
                            if clean_text(ev.get("evidence_text"))
                        ][:6],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    data = chat_json(client, messages, temperature=0.0)
    scene_text = bundle.get("scene_text_full", "")
    rewritten = dict(node)
    rewritten["role_in_context"] = clean_text(data.get("role_in_context")) or rewritten.get("role_in_context")
    rewritten["salient_development"] = clean_text(data.get("salient_development")) or rewritten.get("salient_development")
    rewritten["goal_state"] = data.get("goal_state")
    rewritten["resulting_state"] = data.get("resulting_state")
    rewritten["unresolved_issue"] = data.get("unresolved_issue")
    rewritten["auxiliary"] = dict(rewritten.get("auxiliary", {}))
    rewritten["auxiliary"]["persona_anchor"] = shorten_persona_anchor(data.get("persona_anchor"), language)
    exact_quotes = validate_exact_quotes(scene_text, data.get("evidence_quotes", []), limit=4)
    rewritten["evidence_quotes"] = exact_quotes
    return rewritten


def ground_final_nodes(
    client: ChatCompletionsClient,
    language: str,
    character_name: str,
    final_nodes: Sequence[Dict[str, Any]],
    bundle_lookup: Dict[str, Dict[str, Any]],
    max_workers: int,
) -> List[Dict[str, Any]]:
    print(
        json.dumps(
            {
                "stage": "builder_final_grounding_start",
                "character": character_name,
                "final_node_count": len(final_nodes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    grounded_by_id: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for node in final_nodes:
            document_id = clean_text(node.get("document_id"))
            bundle = bundle_lookup.get(document_id)
            if not bundle:
                grounded_by_id[clean_text(node.get("timeline_node_id"))] = dict(node)
                continue
            futures[
                executor.submit(
                    ground_single_final_node,
                    client,
                    language,
                    character_name,
                    node,
                    bundle,
                )
            ] = node
        for future in as_completed(futures):
            original = futures[future]
            node_id = clean_text(original.get("timeline_node_id"))
            try:
                grounded_by_id[node_id] = future.result()
            except Exception:
                grounded_by_id[node_id] = dict(original)

    grounded_nodes = []
    for node in final_nodes:
        node_id = clean_text(node.get("timeline_node_id"))
        grounded_nodes.append(grounded_by_id.get(node_id, dict(node)))
    print(
        json.dumps(
            {
                "stage": "builder_final_grounding_done",
                "character": character_name,
                "final_node_count": len(grounded_nodes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return grounded_nodes


def refine_timeline_nodes(
    client: ChatCompletionsClient,
    language: str,
    character_name: str,
    proposed_nodes: Sequence[Dict[str, Any]],
    min_final_nodes: int,
    max_final_nodes: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    prompt = GLOBAL_REFINE_PROMPT_ZH if language == "zh" else GLOBAL_REFINE_PROMPT_EN
    node_lookup = {clean_text(node.get("timeline_node_id")): node for node in proposed_nodes}
    payload_nodes = select_nodes_for_refine_payload(proposed_nodes, MAX_REFINE_NODE_COUNT)
    print(
        json.dumps(
            {
                "stage": "builder_global_refine_start",
                "character": character_name,
                "proposed_node_count": len(proposed_nodes),
                "payload_node_count": len(payload_nodes),
                "min_final_nodes": min_final_nodes,
                "max_final_nodes": max_final_nodes,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    messages = [
        {
            "role": "system",
            "content": prompt.replace("{min_nodes}", str(min_final_nodes)).replace("{max_nodes}", str(max_final_nodes)),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "focal_character": character_name,
                    "proposed_nodes": [compact_node_for_refine(node) for node in payload_nodes],
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    try:
        data = chat_json(client, messages, temperature=0.0)
    except Exception:
        return fallback_refine_nodes(proposed_nodes, max_final_nodes)

    summary = clean_text(data.get("timeline_summary"))
    final_nodes: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in (data.get("final_nodes", []) or []):
        timeline_node_id = clean_text(item.get("timeline_node_id"))
        if not timeline_node_id or timeline_node_id not in node_lookup or timeline_node_id in seen:
            continue
        seen.add(timeline_node_id)
        base = dict(node_lookup[timeline_node_id])
        base["role_in_context"] = clean_text(item.get("role_in_context")) or base.get("role_in_context")
        base["salient_development"] = clean_text(item.get("salient_development")) or base.get("salient_development")
        base["goal_state"] = item.get("goal_state")
        base["resulting_state"] = item.get("resulting_state")
        base["unresolved_issue"] = item.get("unresolved_issue")
        base["auxiliary"] = {
            "relation_updates": filter_relation_updates(item.get("relation_updates", []), language),
            "status_updates": filter_status_updates(item.get("status_updates", []), language),
            "persona_anchor": shorten_persona_anchor(item.get("persona_anchor"), language),
        }
        base.pop("importance", None)
        final_nodes.append(base)

    if len(final_nodes) < min(2, min_final_nodes):
        summary, final_nodes = fallback_refine_nodes(proposed_nodes, max_final_nodes)
    else:
        final_nodes.sort(key=lambda item: (item.get("scene_order", 0), item.get("document_id", "")))
    for node in final_nodes:
        node.pop("importance", None)
    print(
        json.dumps(
            {
                "stage": "builder_global_refine_done",
                "character": character_name,
                "final_node_count": len(final_nodes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary, final_nodes


def fallback_arcs(language: str, character_name: str, timeline_nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    linked_ids = [clean_text(node.get("timeline_node_id")) for node in timeline_nodes if clean_text(node.get("timeline_node_id"))]
    if len(linked_ids) < 2:
        return []
    title = "Overall Development" if language == "en" else "角色主线发展"
    summary = (
        "A broad trajectory connecting the character's major screenplay developments."
        if language == "en"
        else "连接该角色主要剧情发展的总体轨迹。"
    )
    return [
        {
            "arc_id": stable_id(character_name, title, prefix="csa"),
            "character_name": character_name,
            "title": title,
            "arc_focus": "mixed",
            "linked_timeline_node_ids": linked_ids,
            "arc_summary": summary,
            "start_state": timeline_nodes[0].get("resulting_state"),
            "end_state": timeline_nodes[-1].get("resulting_state"),
            "unresolved_issue": timeline_nodes[-1].get("unresolved_issue"),
        }
    ]


def induce_character_arcs(
    client: ChatCompletionsClient,
    language: str,
    character_name: str,
    timeline_nodes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prompt = ARC_PROMPT_ZH if language == "zh" else ARC_PROMPT_EN
    payload_nodes = select_nodes_for_refine_payload(timeline_nodes, MAX_ARC_NODE_COUNT)
    print(
        json.dumps(
            {
                "stage": "builder_arc_induction_start",
                "character": character_name,
                "timeline_node_count": len(timeline_nodes),
                "payload_node_count": len(payload_nodes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "focal_character": character_name,
                    "timeline_nodes": [compact_node_for_refine(node) for node in payload_nodes],
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    try:
        data = chat_json(client, messages, temperature=0.0)
    except Exception:
        return fallback_arcs(language, character_name, timeline_nodes)

    arcs: List[Dict[str, Any]] = []
    valid_ids = {clean_text(node.get("timeline_node_id")) for node in timeline_nodes if clean_text(node.get("timeline_node_id"))}
    for item in (data.get("arcs", []) or []):
        linked_ids = [
            clean_text(node_id)
        for node_id in (item.get("linked_timeline_node_ids", []) or [])
            if clean_text(node_id) in valid_ids
        ]
        linked_ids = list(dict.fromkeys(linked_ids))
        if len(linked_ids) < 2:
            continue
        title = clean_text(item.get("title"))
        if not title:
            continue
        arcs.append(
            {
                "arc_id": stable_id(character_name, title, prefix="csa"),
                "character_name": character_name,
                "title": title,
                "arc_focus": clean_text(item.get("arc_focus")) or "mixed",
                "linked_timeline_node_ids": linked_ids,
                "arc_summary": clean_text(item.get("arc_summary")),
                "start_state": item.get("start_state"),
                "end_state": item.get("end_state"),
                "unresolved_issue": item.get("unresolved_issue"),
            }
        )
    arcs = arcs or fallback_arcs(language, character_name, timeline_nodes)
    print(
        json.dumps(
            {
                "stage": "builder_arc_induction_done",
                "character": character_name,
                "arc_count": len(arcs),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return arcs


def build_client(args: argparse.Namespace) -> ChatCompletionsClient:
    primary_api_base = args.api_base
    primary_api_key = args.api_key
    primary_model = args.model
    fallback_routes: List[Dict[str, str]] = []

    if clean_text(args.fallback_api_key):
        fallback_routes.append(
            {
                "api_base": args.fallback_api_base,
                "api_key": args.fallback_api_key,
                "model": args.fallback_model,
            }
        )

    if not clean_text(primary_api_key):
        primary_api_base = args.fallback_api_base
        primary_api_key = args.fallback_api_key
        primary_model = args.fallback_model
        fallback_routes = []

    return ChatCompletionsClient(
        api_base=primary_api_base,
        api_key=primary_api_key,
        model=primary_model,
        fallback_routes=fallback_routes,
        timeout=args.timeout,
    )


def build_outputs_for_movie(
    movie_dir: Path,
    client: ChatCompletionsClient,
    max_characters: int,
    batch_size: int,
    max_workers: int,
    min_final_nodes: int,
    max_final_nodes: int,
    max_bundles_per_character: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    language = detect_language_from_movie_dir(movie_dir)
    local_data = load_local_story_dynamics_for_builder(movie_dir)
    global_data = load_json(movie_dir / "story_dynamics_global_refined.json")
    extraction_results = load_json(movie_dir / "extraction_results.json")
    state_facts = load_json(movie_dir / "state_facts.json")
    rename_map = load_json(movie_dir / "rename_map.json") if (movie_dir / "rename_map.json").exists() else {}

    candidates = aggregate_character_stats(language, local_data, global_data, extraction_results, rename_map)
    try:
        merge_groups = consolidate_candidates_with_llm(client, language, candidates, limit=20)
        if merge_groups:
            candidates = merge_candidate_rows(candidates, merge_groups)
    except Exception:
        merge_groups = []

    selected = select_characters_aligned_with_icrp(
        movie_dir,
        candidates,
        rename_map,
        max_characters,
        extraction_results=extraction_results,
    )
    if not selected:
        try:
            selected = select_characters_with_llm(client, language, candidates, max_characters)
        except Exception:
            selected = []
    if not selected:
        selected = fallback_select_characters(candidates, max_characters)
    print(
        json.dumps(
            {
                "stage": "builder_character_selection_done",
                "movie_dir": str(movie_dir),
                "selected_characters": [item["canonical_name"] for item in selected],
                "icrp_character_names": load_icrp_character_names(movie_dir),
                "local_source": local_data.get("_builder_local_source"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    timelines: List[Dict[str, Any]] = []
    arcs: List[Dict[str, Any]] = []
    selection_report: List[Dict[str, Any]] = []
    audit_characters: List[Dict[str, Any]] = []

    for selection in selected:
        all_bundles = build_character_scene_bundles(
            movie_dir=movie_dir,
            language=language,
            local_data=local_data,
            global_data=global_data,
            state_facts=state_facts,
            rename_map=rename_map,
            selection=selection,
        )
        bundles = select_representative_bundles(all_bundles, max_bundles_per_character) if max_bundles_per_character else list(all_bundles)
        if not bundles:
            continue
        print(
            json.dumps(
                {
                    "stage": "builder_bundles_ready",
                    "character": selection["canonical_name"],
                    "raw_bundle_count": len(all_bundles),
                    "selected_bundle_count": len(bundles),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        proposed_nodes, proposal_audit = build_proposed_timeline(
            client=client,
            language=language,
            selection=selection,
            bundles=bundles,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        if not proposed_nodes:
            continue
        print(
            json.dumps(
                {
                    "stage": "builder_proposals_ready",
                    "character": selection["canonical_name"],
                    "proposed_node_count": len(proposed_nodes),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        timeline_summary, refined_nodes = refine_timeline_nodes(
            client=client,
            language=language,
            character_name=selection["canonical_name"],
            proposed_nodes=proposed_nodes,
            min_final_nodes=min_final_nodes,
            max_final_nodes=max_final_nodes,
        )
        if not refined_nodes:
            continue
        bundle_lookup = {clean_text(bundle.get("document_id")): bundle for bundle in all_bundles}
        refined_nodes = ground_final_nodes(
            client=client,
            language=language,
            character_name=selection["canonical_name"],
            final_nodes=refined_nodes,
            bundle_lookup=bundle_lookup,
            max_workers=max_workers,
        )
        character_arcs = induce_character_arcs(
            client=client,
            language=language,
            character_name=selection["canonical_name"],
            timeline_nodes=refined_nodes,
        )
        timelines.append(
            {
                "character_name": selection["canonical_name"],
                "aliases": selection.get("aliases", []),
                "selection_reason": selection.get("selection_reason"),
                "task3_relevance": selection.get("task3_relevance"),
                "timeline_nodes": refined_nodes,
                "timeline_summary": timeline_summary,
            }
        )
        arcs.extend(character_arcs)
        selection_report.append(
            {
                "character_name": selection["canonical_name"],
                "aliases": selection.get("aliases", []),
                "selection_reason": selection.get("selection_reason"),
                "task3_relevance": selection.get("task3_relevance"),
                "bundle_count": len(bundles),
                "raw_bundle_count": len(all_bundles),
                "proposed_node_count": len(proposed_nodes),
                "timeline_node_count": len(refined_nodes),
                "arc_count": len(character_arcs),
            }
        )
        audit_characters.append(
            {
                "character_name": selection["canonical_name"],
                "bundle_count": len(bundles),
                "raw_bundle_count": len(all_bundles),
                "proposal_audit": proposal_audit,
                "proposed_timeline_nodes": proposed_nodes,
                "final_timeline_node_ids": [node["timeline_node_id"] for node in refined_nodes],
            }
        )

    timeline_output = {
        "movie_id": clean_text(local_data.get("movie_id")) or movie_dir.name,
        "language": language,
        "task_name": "Story Dynamics Structuring",
        "task_version": "gold_character_timelines_v1",
        "focal_character_timelines": timelines,
        "build_summary": {
            "candidate_character_count": len(candidates),
            "selected_focal_character_count": len(selection_report),
            "timeline_node_count": sum(len(item["timeline_nodes"]) for item in timelines),
            "builder_mode": "gold_builder_v1",
        },
    }

    arc_output = {
        "movie_id": clean_text(local_data.get("movie_id")) or movie_dir.name,
        "language": language,
        "task_name": "Story Dynamics Structuring",
        "task_version": "gold_cross_scene_arcs_v1",
        "cross_scene_arcs": arcs,
        "build_summary": {
            "selected_focal_character_count": len(selection_report),
            "arc_count": len(arcs),
            "builder_mode": "gold_builder_v1",
        },
    }

    audit_output = {
        "movie_id": clean_text(local_data.get("movie_id")) or movie_dir.name,
        "language": language,
        "task_name": "Story Dynamics Structuring",
        "task_version": "gold_builder_audit_v1",
        "focal_character_candidates": clean_candidates_for_audit(candidates[:30]),
        "alias_merge_groups": merge_groups,
        "selected_focal_characters": selection_report,
        "character_audit": audit_characters,
    }
    return timeline_output, arc_output, audit_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold focal-character timelines and cross-scene arcs from existing STAGE JSON files.")
    parser.add_argument("--movie-dir", action="append", required=True, help="Movie directory containing refined or legacy story_dynamics local files and related files.")
    parser.add_argument("--max-characters", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-final-nodes", type=int, default=6)
    parser.add_argument("--max-final-nodes", type=int, default=12)
    parser.add_argument("--max-bundles-per-character", type=int, default=0)
    parser.add_argument("--api-base", default=DEFAULT_MIMO_API_BASE)
    parser.add_argument("--api-key", default=os.environ.get("MIMO_API_KEY", ""))
    parser.add_argument("--model", default=DEFAULT_MIMO_MODEL)
    parser.add_argument("--fallback-api-base", default=DEFAULT_LOCAL_API_BASE)
    parser.add_argument("--fallback-api-key", default=DEFAULT_LOCAL_API_KEY)
    parser.add_argument("--fallback-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client(args)
    max_bundles_per_character = args.max_bundles_per_character if args.max_bundles_per_character > 0 else None

    for movie_dir_str in args.movie_dir:
        movie_dir = Path(movie_dir_str)
        timeline_output, arc_output, audit_output = build_outputs_for_movie(
            movie_dir=movie_dir,
            client=client,
            max_characters=args.max_characters,
            batch_size=args.batch_size,
            max_workers=args.workers,
            min_final_nodes=args.min_final_nodes,
            max_final_nodes=args.max_final_nodes,
            max_bundles_per_character=max_bundles_per_character,
        )
        dump_json(movie_dir / "gold_character_timelines_v1.json", timeline_output)
        dump_json(movie_dir / "gold_cross_scene_arcs_v1.json", arc_output)
        dump_json(movie_dir / "gold_character_timeline_builder_audit_v1.json", audit_output)
        print(
            json.dumps(
                {
                    "movie_dir": str(movie_dir),
                    "selected_focal_characters": audit_output["selected_focal_characters"],
                    "timeline_node_count": timeline_output["build_summary"]["timeline_node_count"],
                    "arc_count": arc_output["build_summary"]["arc_count"],
                    "builder_mode": "gold_builder_v1",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
