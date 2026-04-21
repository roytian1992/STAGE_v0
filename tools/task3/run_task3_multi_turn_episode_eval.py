#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

from task3_llm_fallback import build_clients, build_routes
from task3_runtime_loader import (
    DEFAULT_EMBED_API_KEY,
    DEFAULT_EMBED_BASE_URL,
    DEFAULT_EMBED_MODEL,
    SUPPORTED_MEMORY_MODES,
    MemorySelection,
    Task3RuntimeLoader,
    load_multi_turn_episode,
    normalize_ws,
    render_list,
)


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0")
DEFAULT_BASE_URL = "http://localhost:8002/v1"
DEFAULT_API_KEY = "token-abc123"
DEFAULT_MODEL = "Qwen3-235B"
CORE_METRICS = [
    "character_fidelity",
    "memory_faithfulness",
    "boundary_compliance",
    "response_naturalness",
    "cross_turn_consistency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out and judge one Task3 multi-turn episode against the slim role-asset schema."
    )
    parser.add_argument("--movie-id", required=True)
    parser.add_argument("--language", choices=["Chinese", "English"], required=True)
    parser.add_argument("--episode-instance-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--stage-root", default=str(ROOT))
    parser.add_argument("--memory-mode", choices=list(SUPPORTED_MEMORY_MODES), default="persona_only")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--base-url", default=os.environ.get("TASK3_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("TASK3_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--model", default=os.environ.get("TASK3_MODEL", DEFAULT_MODEL))
    parser.add_argument("--embed-base-url", default=os.environ.get("TASK3_EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL))
    parser.add_argument("--embed-api-key", default=os.environ.get("TASK3_EMBED_API_KEY", DEFAULT_EMBED_API_KEY))
    parser.add_argument("--embed-model", default=os.environ.get("TASK3_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--rollout-temperature", type=float, default=0.2)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--rollout-max-tokens", type=int, default=220)
    parser.add_argument("--judge-max-tokens", type=int, default=520)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_memory_context(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "- none"
    rows = []
    for item in items:
        memory_id = normalize_ws(item.get("memory_id") or "unknown")
        scene_order = item.get("scene_order")
        scene_bits = []
        if scene_order is not None:
            scene_bits.append(f"scene_order={scene_order}")
        scene_id = normalize_ws(item.get("scene_id"))
        if scene_id:
            scene_bits.append(f"scene_id={scene_id}")
        scene_text = ", ".join(scene_bits) if scene_bits else "scene=?"
        memory_text = normalize_ws(item.get("memory_text"))
        rows.append(f"- {memory_id} ({scene_text}): {memory_text}")
    return "\n".join(rows)


def render_relation_context(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "- none"
    rows = []
    for item in items:
        relation_id = normalize_ws(item.get("relation_id") or "unknown")
        target = normalize_ws(item.get("target_character") or "unknown")
        relation = normalize_ws(item.get("relation")) or normalize_ws(item.get("relation_summary")) or "unknown"
        rows.append(f"- {relation_id}: {target} -> {relation}")
    return "\n".join(rows)


def resolve_dialogue_history(
    template: Sequence[Dict[str, Any]],
    response_by_turn: Dict[int, str],
) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for item in template:
        speaker = normalize_ws(item.get("speaker") or "unknown")
        fill_with = normalize_ws(item.get("fill_with") or "")
        source_turn_index = int(item.get("source_turn_index") or 0)
        if fill_with == "previous_model_response":
            text = response_by_turn.get(source_turn_index, "")
        else:
            text = normalize_ws(item.get("text") or "")
        resolved.append(
            {
                "speaker": speaker,
                "text": text,
                "source_turn_index": source_turn_index if source_turn_index else None,
            }
        )
    return resolved


def render_dialogue_history(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {row['speaker']}: {normalize_ws(row['text'])}" for row in items)


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("empty JSON-like output")
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"could not locate JSON object in {stripped[:200]!r}")


def call_text(
    clients: List[Tuple[Any, OpenAI]],
    *,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> Tuple[str, Dict[str, Any], int, str]:
    last_error: Optional[Exception] = None
    for route, client in clients:
        for attempt in range(max_retries + 1):
            started = time.time()
            try:
                response = client.chat.completions.create(
                    model=route.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = int((time.time() - started) * 1000)
                usage = {}
                if getattr(response, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    }
                return (response.choices[0].message.content or "").strip(), usage, latency_ms, route.name
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= max_retries:
                    break
                time.sleep(1.0)
    raise RuntimeError(f"LLM call failed: {type(last_error).__name__}: {last_error}")


def build_actor_messages(
    *,
    episode: Dict[str, Any],
    turn: Dict[str, Any],
    current_user_turn: str,
    resolved_history: Sequence[Dict[str, Any]],
    runtime_loader: Task3RuntimeLoader,
    selection: MemorySelection,
    memory_mode: str,
    language: str,
) -> List[Dict[str, str]]:
    persona = runtime_loader.persona_card
    relation_context = runtime_loader.relation_context()
    character = normalize_ws(episode["character"])
    user_turn = normalize_ws(current_user_turn)
    lang_hint = "Chinese" if normalize_ws(language).lower() == "chinese" else "English"

    system_text = (
        "You are role-playing as a screenplay character in an ongoing conversation.\n"
        "Stay faithful to the provided persona, relations, and released script-grounded context.\n"
        "Answer in the same language as the user's question.\n"
        "Speak as the character, not as a narrator.\n"
        "Prefer 2 to 5 sentences unless the question clearly needs more.\n"
        "Avoid screenplay directions, bullet points, meta commentary, benchmark talk, and scene ids.\n"
        "Do not fabricate concrete facts beyond the provided memory context.\n"
        "If context is incomplete, answer in-character with bounded uncertainty.\n"
        "Preserve continuity with the prior dialogue exactly as given.\n"
        "Later turns should build on what you already committed to instead of resetting to a generic fresh answer.\n"
        "If the user sharpens pressure, your reply should address that sharper pressure while staying consistent."
    )
    user_text = (
        f"Character: {character}\n"
        f"Language: {lang_hint}\n"
        f"Memory mode: {memory_mode}\n"
        f"Episode theme: {normalize_ws(episode.get('episode_theme'))}\n\n"
        "Persona traits:\n"
        f"{render_list(persona.get('traits', []))}\n\n"
        "Speaking style:\n"
        f"{render_list(persona.get('speaking_style', []))}\n\n"
        "Constraints:\n"
        f"{render_list(persona.get('constraints', []))}\n\n"
        "Dialogue exemplars:\n"
        f"{render_list(persona.get('dialogue_exemplars', []))}\n\n"
        "Relation context:\n"
        f"{render_relation_context(relation_context)}\n\n"
        "Memory context:\n"
        f"{render_memory_context(selection.selected_memories)}\n\n"
        "Dialogue history:\n"
        f"{render_dialogue_history(resolved_history)}\n\n"
        f"Current user turn:\n{user_turn}\n\n"
        f"Respond as {character}."
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def render_episode_transcript(rollout_turns: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for row in rollout_turns:
        turn = row["turn"]
        resolved_history = row["resolved_history"]
        rows.append(
            "\n".join(
                [
                    f"Turn {turn['turn_index']}",
                    "Dialogue history:",
                    render_dialogue_history(resolved_history),
                    "Current user turn:",
                    normalize_ws(row["current_user_turn"]),
                    "Candidate response:",
                    normalize_ws(row["response"]),
                ]
            )
        )
    return "\n\n".join(rows)


def build_core_metric_messages(
    metric: str,
    *,
    episode: Dict[str, Any],
    rollout_turns: Sequence[Dict[str, Any]],
    runtime_loader: Task3RuntimeLoader,
    memory_mode: str,
) -> List[Dict[str, str]]:
    persona = runtime_loader.persona_card
    transcript = render_episode_transcript(rollout_turns)
    header = (
        f"Metric: {metric}\n"
        f"Episode ID: {episode['episode_id']}\n"
        f"Character: {episode['character']}\n"
        f"Mode: {memory_mode}\n"
        f"Episode theme: {episode.get('episode_theme')}\n\n"
    )
    if metric == "character_fidelity":
        system_text = (
            "You evaluate only character fidelity for screenplay character role-play across one episode.\n"
            "High scores require the replies to sound specifically like this character, not merely like a plausible generic dramatic speaker.\n"
            "Use lower scores aggressively when the episode drifts into bland emotional prose, generic therapeutic reflection, or interchangeable dialogue that many characters could have said."
        )
        user_text = (
            header
            + "Persona traits:\n"
            + f"{render_list(persona.get('traits', []))}\n\n"
            + "Speaking style:\n"
            + f"{render_list(persona.get('speaking_style', []))}\n\n"
            + "Constraints:\n"
            + f"{render_list(persona.get('constraints', []))}\n\n"
            + "Dialogue exemplars:\n"
            + f"{render_list(persona.get('dialogue_exemplars', []))}\n\n"
            + "Episode transcript:\n"
            + f"{transcript}\n\n"
            + "Scoring rubric:\n"
            + "- 5: every turn sounds distinctly like this character in voice, restraint, stance, and interpersonal framing.\n"
            + "- 4: mostly in-character, with only minor generic wording.\n"
            + "- 3: mixed; some turns are emotionally plausible but could belong to many characters.\n"
            + "- 2: frequent flattening into generic dramatic language, wrong stance, or weakly character-specific voice.\n"
            + "- 1: clearly out of character in voice, attitude, or social framing.\n\n"
            + "Judge only whether the replies stay in-character in voice, stance, restraint, and interpersonal framing.\n"
            + "Do not let factual grounding dominate this score unless it changes whether the replies sound like the character.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    elif metric == "memory_faithfulness":
        system_text = (
            "You evaluate only memory faithfulness for screenplay character role-play across one episode.\n"
            "Score against the selected memory context actually shown at each turn, not against what the character might know in the full movie.\n"
            "A reply must not receive a high score merely for sounding plausible or emotionally fitting.\n"
            "If selected memories are empty or miss the needed support, concrete episodic recall should be penalized unless the answer stays clearly uncertain and non-specific.\n"
            "Penalize fabricated scene details, imported facts absent from selected memories, swapped event details across turns, and confident claims that go beyond the selected evidence.\n"
            "Reward bounded uncertainty when memory evidence is missing.\n"
            "Use lower scores aggressively when later turns become more specific than the selected memories justify.\n"
            "Emotionally coherent answers do not deserve high scores if they smuggle in unsupported scene facts, sensory details, or imported event content.\n"
            "If one turn contains major unsupported concrete recall, the episode should not receive a score above 3.\n"
            "If multiple turns contain major unsupported recall, or one turn clearly contradicts the selected evidence, the score should fall to 1 or 2."
        )
        blocks = []
        for row in rollout_turns:
            turn = row["turn"]
            blocks.append(
                "\n".join(
                    [
                        f"Turn {turn['turn_index']}",
                        "Released source memory IDs:",
                        render_list(turn["reference"].get("source_memory_ids", [])),
                        "Selected memory IDs:",
                        render_list(row["selection"].selected_memory_ids),
                        "Selected memory context:",
                        render_memory_context(row["selection"].selected_memories),
                        "Supporting facts:",
                        render_list(turn["reference"].get("supporting_facts", [])),
                        "Contradicting facts:",
                        render_list(turn["reference"].get("contradicting_facts", [])),
                        "Candidate response:",
                        normalize_ws(row["response"]),
                    ]
                )
            )
        user_text = (
            header
            + "Turn evidence:\n"
            + "\n\n".join(blocks)
            + "\n\nScoring rubric:\n"
            + "- 5: every turn stays concretely grounded in the selected memories or remains carefully bounded when support is missing.\n"
            + "- 4: mostly grounded, only minor overreach or paraphrase drift.\n"
            + "- 3: mixed grounding; at least one turn shows unsupported specificity, weak memory anchoring, or later-turn drift into generic but unsupported claims.\n"
            + "- 2: major unsupported recall, wrong event detail, imported cross-thread specificity, or confident claims not supported by selected memories on one or more turns.\n"
            + "- 1: repeated fabrication or clear contradiction to the selected evidence.\n\n"
            + "Judge whether the replies invent concrete scene recall or memory detail unsupported by the selected memory context and supporting facts.\n"
            + "Penalize answers that become more specific under follow-up pressure without new support.\n"
            + "Do not reward thematic fit, emotional plausibility, or symbolic coherence when concrete scene content is unsupported.\n"
            + "The released source memory IDs are diagnostic context for you, but grounding must be judged against the selected memories actually provided to the actor at each turn.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    elif metric == "boundary_compliance":
        system_text = (
            "You evaluate only knowledge-boundary compliance for screenplay character role-play across one episode.\n"
            "Use lower scores aggressively when the replies smuggle forbidden knowledge through implication, motive claims, interpretive summaries, or concrete details that the released boundary does not support."
        )
        blocks = []
        for row in rollout_turns:
            turn = row["turn"]
            blocks.append(
                "\n".join(
                    [
                        f"Turn {turn['turn_index']}",
                        "Knowledge boundary allowed:",
                        render_list(turn["reference"].get("knowledge_boundary", {}).get("allowed", [])),
                        "Knowledge boundary forbidden:",
                        render_list(turn["reference"].get("knowledge_boundary", {}).get("forbidden", [])),
                        "Supporting facts:",
                        render_list(turn["reference"].get("supporting_facts", [])),
                        "Candidate response:",
                        normalize_ws(row["response"]),
                    ]
                )
            )
        user_text = (
            header
            + "Turn evidence:\n"
            + "\n\n".join(blocks)
            + "\n\nScoring rubric:\n"
            + "- 5: every turn stays cleanly within the released boundary.\n"
            + "- 4: mostly compliant, with only mild implication drift.\n"
            + "- 3: mixed; at least one turn edges beyond the boundary through interpretation or unsupported implication.\n"
            + "- 2: clear forbidden knowledge, hidden assumption, or concrete out-of-bound content appears.\n"
            + "- 1: repeated or severe boundary violations.\n\n"
            + "Judge whether the replies stay inside the released boundary. Do not use negative-answer examples as true story facts.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    elif metric == "response_naturalness":
        system_text = (
            "You evaluate only response naturalness for screenplay character role-play across one episode.\n"
            "Use lower scores aggressively when the replies read like benchmark scaffolding, explanatory summaries, over-polished monologues, or generic mood statements rather than live dialogue."
        )
        user_text = (
            header
            + "Speaking style:\n"
            + f"{render_list(persona.get('speaking_style', []))}\n\n"
            + "Episode transcript:\n"
            + f"{transcript}\n\n"
            + "Scoring rubric:\n"
            + "- 5: the replies consistently read like plausible spoken dialogue in context.\n"
            + "- 4: mostly natural, with only minor stiffness.\n"
            + "- 3: mixed; some turns sound written, over-explained, or benchmark-like.\n"
            + "- 2: multiple turns feel stiff, speechified, or unnaturally expository.\n"
            + "- 1: the episode rarely reads like live dialogue.\n\n"
            + "Judge whether the replies read like plausible spoken dialogue in context. Do not directly lower the score just because a reply is factually weak.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    elif metric == "cross_turn_consistency":
        system_text = (
            "You evaluate only cross-turn consistency for screenplay character role-play across one episode.\n"
            "Use low scores aggressively when later turns dodge, reset, soften, or quietly reverse commitments made earlier in the episode.\n"
            "If later replies retreat into vague symbolism, generic mood language, or broad self-description instead of carrying forward the concrete thread established earlier, do not give a high score."
        )
        blocks = []
        for row in rollout_turns:
            turn = row["turn"]
            blocks.append(
                "\n".join(
                    [
                        f"Turn {turn['turn_index']}",
                        "Cross-turn constraints:",
                        render_list(turn["reference"].get("cross_turn_constraints", [])),
                        "Candidate response:",
                        normalize_ws(row["response"]),
                    ]
                )
            )
        user_text = (
            header
            + "Episode transcript:\n"
            + f"{transcript}\n\n"
            + "Constraint slices:\n"
            + "\n\n".join(blocks)
            + "\n\nScoring rubric:\n"
            + "- 5: later turns clearly build on earlier commitments and maintain stable stance, relation framing, remembered events, and emotional logic.\n"
            + "- 4: mostly stable, with only mild softening or compression.\n"
            + "- 3: mixed; at least one later turn partially resets, weakens, or abstracts away from what was established earlier.\n"
            + "- 2: clear reset, contradiction, selective forgetting, or thread abandonment appears.\n"
            + "- 1: repeated inconsistency or incompatible turn-to-turn behavior.\n\n"
            + "Judge whether the replies stay stable across turns in stance, self-presentation, relation framing, remembered events, and emotional logic under pressure.\n"
            + "Penalize later turns that act as if earlier answers never happened, collapse into generic mood statements, or stop engaging the concrete thread established earlier.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    else:
        raise ValueError(f"unsupported metric: {metric}")
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def build_episode_path_compatibility_messages(
    episode: Dict[str, Any],
    rollout_turns: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    blocks = []
    for row in rollout_turns:
        turn = row["turn"]
        turn_index = int(turn["turn_index"])
        if turn_index <= 1:
            continue
        blocks.append(
            "\n".join(
                [
                    f"Turn {turn_index}",
                    "Actual prior dialogue history:",
                    render_dialogue_history(row["resolved_history"]),
                    "Realized later user turn:",
                    normalize_ws(row["current_user_turn"]),
                ]
            )
        )
    system_text = (
        "You evaluate only whether the released later user turns remain compatible with the actual realized dialogue path in a screenplay role-play episode.\n"
        "Use low scores aggressively when a later user turn remains only thematically related but no longer follows from the specific commitments, topics, emotions, or factual content established in the realized path."
    )
    user_text = (
        f"Episode ID: {episode['episode_id']}\n"
        f"Character: {episode['character']}\n"
        f"Episode theme: {normalize_ws(episode.get('episode_theme'))}\n\n"
        + "For each later turn, judge whether the realized user turn still makes sense after the model's actual earlier responses.\n"
        + "This is part of the multi-turn path quality diagnosis: the later user turn should feel like a valid continuation, not a branch that ignores what just happened.\n"
        + "Low score means the realized dialogue path and the later-turn user question have become mismatched.\n"
        + "Do not reward mere thematic overlap. High scores require direct path compatibility with the actual prior exchange, not just with the broad episode theme.\n"
        + "If a later user turn injects an emotional assumption, concrete premise, or conversational branch that the model's actual earlier replies did not establish, cap the score at 3.\n"
        + "If the later user turn would require ignoring or rewriting the actual prior exchange, use 1 or 2.\n"
        + "Use this rubric:\n"
        + "- 5: fully compatible, natural continuation that directly builds on the realized path with no hidden assumption jump\n"
        + "- 4: mostly compatible, only mild compression or slight path rigidity\n"
        + "- 3: still answerable but noticeably pre-scripted, assumption-heavy, or only weakly connected to the realized path\n"
        + "- 2: clearly mismatched with the realized path or assumes a commitment, emotion, or premise the model did not establish\n"
        + "- 1: incompatible branch; the later user turn no longer follows from the actual dialogue\n\n"
        + "Return strict JSON with schema:\n"
        + "{\"turn_scores\": [{\"turn_index\": 2, \"score\": 1, \"rationale\": \"short\"}], \"average_score\": 1, \"overall_rationale\": \"short\"}\n\n"
        + "Turns to judge:\n"
        + "\n\n".join(blocks)
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def sanitize_metric_result(metric: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    score = int(round(float(raw.get("score", 0)))) if raw.get("score") is not None else 0
    score = max(1, min(5, score))
    flags = raw.get("violation_flags", [])
    if not isinstance(flags, list):
        flags = []
    clean_flags = [normalize_ws(x) for x in flags if normalize_ws(x)]
    return {
        "metric": metric,
        "score": score,
        "rationale": normalize_ws(raw.get("rationale", "")),
        "violation_flags": clean_flags,
    }


def sanitize_episode_path_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    turn_rows = raw.get("turn_scores", [])
    out_rows = []
    if isinstance(turn_rows, list):
        for row in turn_rows:
            if not isinstance(row, dict):
                continue
            try:
                turn_index = int(row.get("turn_index"))
            except Exception:
                continue
            score = int(round(float(row.get("score", 0)))) if row.get("score") is not None else 0
            score = max(1, min(5, score))
            out_rows.append(
                {
                    "turn_index": turn_index,
                    "score": score,
                    "rationale": normalize_ws(row.get("rationale", "")),
                }
            )
    average_score = round(mean(row["score"] for row in out_rows), 4) if out_rows else 0.0
    return {
        "turn_scores": out_rows,
        "average_score": average_score,
        "overall_rationale": normalize_ws(raw.get("overall_rationale", "")),
    }


def summarize_retrieval(rollout_turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rollout_turns:
        return {}
    support_turns = []
    for row in rollout_turns:
        source_ids = [normalize_ws(x) for x in row["turn"]["reference"].get("source_memory_ids", []) or [] if normalize_ws(x)]
        if not source_ids:
            continue
        support_turns.append(row)
    avg_support_hit = (
        round(mean(float(row["selection"].diagnostics["support_hit_at_k"]) for row in support_turns), 4)
        if support_turns
        else None
    )
    avg_support_recall = (
        round(
            mean(
                float(row["selection"].diagnostics["support_recall_at_k"])
                for row in support_turns
                if row["selection"].diagnostics["support_recall_at_k"] is not None
            ),
            4,
        )
        if support_turns
        else None
    )
    return {
        "avg_selected_memory_count": round(
            mean(int(row["selection"].diagnostics["selected_memory_count"]) for row in rollout_turns), 4
        ),
        "avg_selected_memory_tokens": round(
            mean(int(row["selection"].diagnostics["selected_memory_tokens"]) for row in rollout_turns), 4
        ),
        "support_hit_at_k": avg_support_hit,
        "support_recall_at_k": avg_support_recall,
    }


def main() -> None:
    args = parse_args()
    stage_root = Path(args.stage_root)
    _movie_dir, episode, role_asset = load_multi_turn_episode(
        stage_root=stage_root,
        language=args.language,
        movie_id=args.movie_id,
        episode_instance_id=args.episode_instance_id,
    )
    runtime_loader = Task3RuntimeLoader(
        role_asset=role_asset,
        language=args.language,
        embed_base_url=args.embed_base_url,
        embed_api_key=args.embed_api_key,
        embed_model=args.embed_model,
    )

    routes = build_routes(base_url=args.base_url, api_key=args.api_key, model=args.model)
    clients = build_clients(routes, timeout=180)

    rollout_turns: List[Dict[str, Any]] = []
    response_by_turn: Dict[int, str] = {}
    rollout_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    judge_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for turn in episode["turns"]:
        turn_index = int(turn["turn_index"])
        resolved_history = resolve_dialogue_history(
            turn["input"].get("dialogue_history_template", []),
            response_by_turn,
        )
        current_user_turn = normalize_ws(turn["input"].get("current_user_turn"))
        selection = runtime_loader.select_memories(
            mode=args.memory_mode,
            resolved_history=resolved_history,
            current_user_turn=current_user_turn,
            top_k=args.top_k,
            source_memory_ids=turn["reference"].get("source_memory_ids", []),
        )
        messages = build_actor_messages(
            episode=episode,
            turn=turn,
            current_user_turn=current_user_turn,
            resolved_history=resolved_history,
            runtime_loader=runtime_loader,
            selection=selection,
            memory_mode=args.memory_mode,
            language=args.language,
        )
        text, usage, latency_ms, route_name = call_text(
            clients,
            messages=messages,
            temperature=args.rollout_temperature,
            max_tokens=args.rollout_max_tokens,
            max_retries=args.max_retries,
        )
        response_text = normalize_ws(text)
        response_by_turn[turn_index] = response_text
        for key in rollout_usage_totals:
            if isinstance(usage.get(key), int):
                rollout_usage_totals[key] += int(usage[key])
        rollout_turns.append(
            {
                "turn": turn,
                "resolved_history": resolved_history,
                "selection": selection,
                "current_user_turn": current_user_turn,
                "response": response_text,
                "latency_ms": latency_ms,
                "route_name": route_name,
                "usage": usage,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "rollout_turn_complete",
                    "turn_index": turn_index,
                    "route": route_name,
                    "memory_mode": args.memory_mode,
                    "user_turn": current_user_turn,
                    "selected_memory_ids": selection.selected_memory_ids,
                    "response_preview": response_text[:120],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    metric_results: Dict[str, Dict[str, Any]] = {}
    for metric in CORE_METRICS:
        messages = build_core_metric_messages(
            metric,
            episode=episode,
            rollout_turns=rollout_turns,
            runtime_loader=runtime_loader,
            memory_mode=args.memory_mode,
        )
        text, usage, latency_ms, route_name = call_text(
            clients,
            messages=messages,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            max_retries=args.max_retries,
        )
        raw = extract_json_object(text)
        metric_results[metric] = sanitize_metric_result(metric, raw)
        metric_results[metric]["latency_ms"] = latency_ms
        metric_results[metric]["route_name"] = route_name
        metric_results[metric]["usage"] = usage
        for key in judge_usage_totals:
            if isinstance(usage.get(key), int):
                judge_usage_totals[key] += int(usage[key])
        print(
            json.dumps(
                {
                    "stage": "core_metric_complete",
                    "metric": metric,
                    "score": metric_results[metric]["score"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    path_messages = build_episode_path_compatibility_messages(episode, rollout_turns)
    path_text, path_usage, path_latency_ms, path_route_name = call_text(
        clients,
        messages=path_messages,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        max_retries=args.max_retries,
    )
    path_result = sanitize_episode_path_result(extract_json_object(path_text))
    path_result["latency_ms"] = path_latency_ms
    path_result["route_name"] = path_route_name
    path_result["usage"] = path_usage
    for key in judge_usage_totals:
        if isinstance(path_usage.get(key), int):
            judge_usage_totals[key] += int(path_usage[key])

    core_score = round(mean(metric_results[metric]["score"] for metric in CORE_METRICS), 4)
    retrieval_summary = summarize_retrieval(rollout_turns)
    report = {
        "movie_id": args.movie_id,
        "language": args.language,
        "episode_instance_id": args.episode_instance_id,
        "episode_id": normalize_ws(episode.get("episode_id")),
        "memory_mode": args.memory_mode,
        "top_k": args.top_k,
        "character": episode["character"],
        "model": args.model,
        "base_url": args.base_url,
        "embed_model": args.embed_model,
        "embed_base_url": args.embed_base_url,
        "stage_root": args.stage_root,
        "rollout_temperature": args.rollout_temperature,
        "judge_temperature": args.judge_temperature,
        "core_metrics": metric_results,
        "core_multi_turn_score": core_score,
        "episode_path_compatibility": path_result,
        "followup_compatibility": path_result,
        "retrieval_diagnostics": retrieval_summary,
        "rollout_turns": [
            {
                "turn_index": int(row["turn"]["turn_index"]),
                "question_id": row["turn"]["question_id"],
                "current_user_turn": normalize_ws(row["current_user_turn"]),
                "resolved_history": row["resolved_history"],
                "selected_memory_ids": row["selection"].selected_memory_ids,
                "selected_memory_texts": row["selection"].selected_memory_texts,
                "retrieval_query": row["selection"].query,
                "retrieval_scores": row["selection"].score_rows,
                "retrieval_diagnostics": row["selection"].diagnostics,
                "response": row["response"],
                "latency_ms": row["latency_ms"],
                "route_name": row["route_name"],
                "usage": row["usage"],
            }
            for row in rollout_turns
        ],
        "usage": {
            "rollout": rollout_usage_totals,
            "judge": judge_usage_totals,
        },
        "interpretation": {
            "high_core_score_means": "the model stayed in-character, grounded, bounded, natural, and cross-turn consistent under the fixed user path",
            "low_episode_path_compatibility_means": "the released later-turn question no longer fits the realized dialogue path well; this indicates path rigidity or asset/runtime mismatch inside the multi-turn episode",
            "low_followup_compatibility_means": "legacy alias of low_episode_path_compatibility_means",
        },
        "limitations": {
            "unsupported_memory_modes": ["embedding_reranker_topk", "llm_selector_topk"],
        },
    }
    dump_json(Path(args.output_path), report)
    print(
        json.dumps(
            {
                "stage": "complete",
                "output_path": args.output_path,
                "core_multi_turn_score": core_score,
                "episode_path_compatibility": path_result["average_score"],
                "followup_compatibility": path_result["average_score"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
