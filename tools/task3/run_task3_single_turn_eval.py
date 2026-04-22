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
    load_single_turn_instance,
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out and judge one Task3 single-turn instance against the slim role-asset schema."
    )
    parser.add_argument("--movie-id", required=True)
    parser.add_argument("--language", choices=["Chinese", "English"], required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--stage-root", default=str(ROOT))
    parser.add_argument("--memory-mode", choices=list(SUPPORTED_MEMORY_MODES), default="persona_only")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--base-url", default=os.environ.get("TASK3_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("TASK3_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--model", default=os.environ.get("TASK3_MODEL", DEFAULT_MODEL))
    parser.add_argument("--judge-base-url", default=os.environ.get("TASK3_JUDGE_BASE_URL", ""))
    parser.add_argument("--judge-api-key", default=os.environ.get("TASK3_JUDGE_API_KEY", ""))
    parser.add_argument("--judge-model", default=os.environ.get("TASK3_JUDGE_MODEL", ""))
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


def render_dialogue_history(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "- none"
    return "\n".join(
        f"- {normalize_ws(row.get('speaker') or 'unknown')}: {normalize_ws(row.get('text'))}"
        for row in items
    )


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
    instance: Dict[str, Any],
    runtime_loader: Task3RuntimeLoader,
    selection: MemorySelection,
    memory_mode: str,
    language: str,
) -> List[Dict[str, str]]:
    persona = runtime_loader.persona_card
    relation_context = runtime_loader.relation_context()
    character = normalize_ws(instance["character"])
    user_turn = normalize_ws(instance["input"]["current_user_turn"])
    dialogue_history = list(instance["input"].get("dialogue_history", []) or [])
    lang_hint = "Chinese" if normalize_ws(language).lower() == "chinese" else "English"

    system_text = (
        "You are role-playing as a screenplay character in a single-turn conversation.\n"
        "Stay faithful to the provided persona, relations, and released script-grounded context.\n"
        "Answer in the same language as the user's question.\n"
        "Speak as the character, not as a narrator.\n"
        "Prefer 2 to 5 sentences unless the question clearly needs more.\n"
        "Avoid screenplay directions, bullet points, meta commentary, benchmark talk, and scene ids.\n"
        "Do not fabricate concrete facts beyond the provided memory context.\n"
        "If context is incomplete, answer in-character with bounded uncertainty."
    )
    user_text = (
        f"Character: {character}\n"
        f"Language: {lang_hint}\n"
        f"Memory mode: {memory_mode}\n\n"
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
        f"{render_dialogue_history(dialogue_history)}\n\n"
        f"Current user turn:\n{user_turn}\n\n"
        f"Respond as {character}."
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def build_core_metric_messages(
    metric: str,
    *,
    instance: Dict[str, Any],
    response_text: str,
    selection: MemorySelection,
    runtime_loader: Task3RuntimeLoader,
    memory_mode: str,
) -> List[Dict[str, str]]:
    persona = runtime_loader.persona_card
    header = (
        f"Metric: {metric}\n"
        f"Instance ID: {instance['instance_id']}\n"
        f"Character: {instance['character']}\n"
        f"Mode: {memory_mode}\n\n"
    )
    if metric == "character_fidelity":
        system_text = "You evaluate only character fidelity for screenplay single-turn role-play."
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
            + "User turn:\n"
            + normalize_ws(instance["input"]["current_user_turn"])
            + "\n\nCandidate response:\n"
            + normalize_ws(response_text)
            + "\n\nJudge only whether the reply stays in-character in voice, stance, restraint, and interpersonal framing.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    elif metric == "memory_faithfulness":
        system_text = (
            "You evaluate only memory faithfulness for screenplay single-turn role-play.\n"
            "Score against the selected memory context actually shown to the actor, not against what the character might know in the full movie.\n"
            "A response must not receive a high score merely for sounding plausible.\n"
            "If the selected memory context is empty or misses the needed support, specific scene recall should be penalized unless the answer stays clearly uncertain and non-specific.\n"
            "Penalize fabricated episodic details, imported facts absent from selected memories, swapped event details, and confident claims that go beyond the selected evidence.\n"
            "Reward bounded uncertainty when memory evidence is missing.\n"
            "Use lower scores aggressively when the reply contains concrete visual/action details unsupported by the selected memories."
        )
        user_text = (
            header
            + "Released source memory IDs:\n"
            + render_list(instance["reference"].get("source_memory_ids", []))
            + "\n\nSelected memory IDs:\n"
            + render_list(selection.selected_memory_ids)
            + "Selected memory context:\n"
            + render_memory_context(selection.selected_memories)
            + "\n\nSupporting facts:\n"
            + render_list(instance["reference"].get("supporting_facts", []))
            + "\n\nContradicting facts:\n"
            + render_list(instance["reference"].get("contradicting_facts", []))
            + "\n\nCandidate response:\n"
            + normalize_ws(response_text)
            + "\n\nScoring rubric:\n"
            + "- 5: every concrete recalled detail is directly supported by the selected memories or the reply stays carefully bounded.\n"
            + "- 4: mostly grounded, only minor overreach or paraphrase drift.\n"
            + "- 3: mixed grounding; some unsupported specificity or weak memory anchoring.\n"
            + "- 2: major unsupported recall, wrong event detail, or confident claims not supported by selected memories.\n"
            + "- 1: largely fabricated or clearly contradictory to the selected evidence.\n\n"
            + "Judge whether the reply invents concrete scene recall or memory detail unsupported by the selected memory context and supporting facts. The released source memory IDs are diagnostic context for you, but grounding must be judged against the selected memories actually provided to the actor.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    elif metric == "boundary_compliance":
        system_text = "You evaluate only knowledge-boundary compliance for screenplay single-turn role-play."
        user_text = (
            header
            + "Knowledge boundary allowed:\n"
            + render_list(instance["reference"].get("knowledge_boundary", {}).get("allowed", []))
            + "\n\nKnowledge boundary forbidden:\n"
            + render_list(instance["reference"].get("knowledge_boundary", {}).get("forbidden", []))
            + "\n\nSupporting facts:\n"
            + render_list(instance["reference"].get("supporting_facts", []))
            + "\n\nCandidate response:\n"
            + normalize_ws(response_text)
            + "\n\nJudge whether the reply stays inside the released boundary. Do not use negative-answer examples as true story facts.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    elif metric == "response_naturalness":
        system_text = "You evaluate only response naturalness for screenplay single-turn role-play."
        user_text = (
            header
            + "Speaking style:\n"
            + f"{render_list(persona.get('speaking_style', []))}\n\n"
            + "User turn:\n"
            + normalize_ws(instance["input"]["current_user_turn"])
            + "\n\nCandidate response:\n"
            + normalize_ws(response_text)
            + "\n\nJudge whether the reply reads like plausible spoken dialogue in context. Do not directly lower the score just because a reply is factually weak.\n"
            + "Return strict JSON: {\"score\": 1-5, \"rationale\": \"short\", \"violation_flags\": []}"
        )
    else:
        raise ValueError(f"unsupported metric: {metric}")
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


def main() -> None:
    args = parse_args()
    stage_root = Path(args.stage_root)
    _movie_dir, instance, role_asset = load_single_turn_instance(
        stage_root=stage_root,
        language=args.language,
        movie_id=args.movie_id,
        instance_id=args.instance_id,
    )
    runtime_loader = Task3RuntimeLoader(
        role_asset=role_asset,
        language=args.language,
        embed_base_url=args.embed_base_url,
        embed_api_key=args.embed_api_key,
        embed_model=args.embed_model,
    )

    selection = runtime_loader.select_memories(
        mode=args.memory_mode,
        resolved_history=instance["input"].get("dialogue_history", []),
        current_user_turn=instance["input"]["current_user_turn"],
        top_k=args.top_k,
        source_memory_ids=instance["reference"].get("source_memory_ids", []),
    )

    actor_routes = build_routes(base_url=args.base_url, api_key=args.api_key, model=args.model)
    actor_clients = build_clients(actor_routes, timeout=180)
    judge_base_url = args.judge_base_url or args.base_url
    judge_api_key = args.judge_api_key or args.api_key
    judge_model = args.judge_model or args.model
    judge_routes = build_routes(base_url=judge_base_url, api_key=judge_api_key, model=judge_model)
    judge_clients = build_clients(judge_routes, timeout=180)

    actor_messages = build_actor_messages(
        instance=instance,
        runtime_loader=runtime_loader,
        selection=selection,
        memory_mode=args.memory_mode,
        language=args.language,
    )
    response_text, rollout_usage, rollout_latency_ms, rollout_route_name = call_text(
        actor_clients,
        messages=actor_messages,
        temperature=args.rollout_temperature,
        max_tokens=args.rollout_max_tokens,
        max_retries=args.max_retries,
    )
    response_text = normalize_ws(response_text)
    print(
        json.dumps(
            {
                "stage": "rollout_complete",
                "instance_id": instance["instance_id"],
                "memory_mode": args.memory_mode,
                "selected_memory_ids": selection.selected_memory_ids,
                "response_preview": response_text[:120],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    metric_results: Dict[str, Dict[str, Any]] = {}
    judge_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for metric in CORE_METRICS:
        messages = build_core_metric_messages(
            metric,
            instance=instance,
            response_text=response_text,
            selection=selection,
            runtime_loader=runtime_loader,
            memory_mode=args.memory_mode,
        )
        text, usage, latency_ms, route_name = call_text(
            judge_clients,
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

    single_turn_score = round(mean(metric_results[metric]["score"] for metric in CORE_METRICS), 4)
    report = {
        "movie_id": args.movie_id,
        "language": args.language,
        "instance_id": args.instance_id,
        "memory_mode": args.memory_mode,
        "top_k": args.top_k,
        "character": instance["character"],
        "question_id": instance["reference"]["question_id"],
        "model": args.model,
        "base_url": args.base_url,
        "judge_model": judge_model,
        "judge_base_url": judge_base_url,
        "embed_model": args.embed_model,
        "embed_base_url": args.embed_base_url,
        "stage_root": args.stage_root,
        "rollout_temperature": args.rollout_temperature,
        "judge_temperature": args.judge_temperature,
        "response": response_text,
        "core_metrics": metric_results,
        "single_turn_score": single_turn_score,
        "retrieval_diagnostics": selection.diagnostics,
        "selected_memory_ids": selection.selected_memory_ids,
        "selected_memory_texts": selection.selected_memory_texts,
        "retrieval_query": selection.query,
        "retrieval_scores": selection.score_rows,
        "usage": {
            "rollout": rollout_usage,
            "judge": judge_usage_totals,
        },
        "latency_ms": {
            "rollout": rollout_latency_ms,
        },
        "route_name": {
            "rollout": rollout_route_name,
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
                "single_turn_score": single_turn_score,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
