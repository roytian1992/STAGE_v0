#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

try:
    import json_repair  # type: ignore
except Exception:
    json_repair = None

DEFAULT_LLM_MODEL = "Qwen3-235B"
DEFAULT_LLM_BASE_URL = "http://localhost:8002/v1"
DEFAULT_LLM_API_KEY = "token-abc123"
DEFAULT_LLM_FALLBACK_BASE_URL = "http://localhost:8001/v1"
DEFAULT_MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2-pro")
DEFAULT_MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
DEFAULT_MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_EMBED_BASE_URL = "http://localhost:8080/v1"
DEFAULT_EMBED_API_KEY = "not-needed"
DEFAULT_LLM_TIMEOUT_SEC = 45
DEFAULT_EMBED_TIMEOUT_SEC = 180

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
EN_TOKEN_RE = re.compile(r"[a-z0-9']+")
WS_RE = re.compile(r"\s+")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
UPPER_LINE_RE = re.compile(r"^[A-Z][A-Z .\-']{1,40}$")
STOP_SPEAKERS = {
    "INT", "EXT", "CUT TO", "CONTINUED", "ANGLE", "LATER", "MORNING", "NIGHT", "DAY",
    "EVENING", "CLOSE", "VOICE", "OMITTED", "MONTAGE", "FADE OUT", "FADE IN", "TITLE",
    "SUPER", "THE END", "MUSIC", "INSERT", "FLASHBACK", "BACK TO SCENE", "WIDE SHOT",
}


@dataclass
class SceneRecord:
    scene_id: str
    scene_order: int
    scene_title: str
    subtitle: str
    content: str
    language: str


@dataclass
class SceneHit:
    scene_id: str
    scene_order: int
    scene_title: str
    text: str
    score: float
    source: str


def _route_chat_completion(
    base_url: str,
    api_key: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
    queue: Any,
) -> None:
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_sec,
            max_retries=0,
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        queue.put({"ok": True, "content": (resp.choices[0].message.content or "").strip()})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class OpenAICompatEmbedder:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=DEFAULT_EMBED_TIMEOUT_SEC,
            max_retries=2,
        )
        lower = model_name.lower()
        dual = any(x in lower for x in ("bge", "gte", "m3"))
        self.doc_prefix = "passage: " if dual else ""
        self.query_prefix = "query: " if dual else ""

    @staticmethod
    def _normalize(vectors: List[List[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vecs: List[List[float]] = []
        prefixed = [self.doc_prefix + str(t or "") for t in texts]
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            vecs.extend([item.embedding for item in resp.data])
        return self._normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model_name, input=[self.query_prefix + str(text or "")])
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        norm = max(float(np.linalg.norm(vec)), 1e-12)
        return vec / norm


class LLMClient:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.routes: List[Tuple[str, str, str]] = []

        if DEFAULT_MIMO_API_KEY:
            self.routes.append((DEFAULT_MIMO_MODEL, DEFAULT_MIMO_BASE_URL, DEFAULT_MIMO_API_KEY))

        self.routes.append((model_name, base_url, api_key))

        if base_url != DEFAULT_LLM_FALLBACK_BASE_URL:
            self.routes.append((model_name, DEFAULT_LLM_FALLBACK_BASE_URL, api_key))

    def run(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 0.0) -> str:
        last_error: Optional[Exception] = None
        ctx = mp.get_context("fork")
        for route_model, route_base_url, route_api_key in self.routes:
            queue = ctx.Queue()
            proc = ctx.Process(
                target=_route_chat_completion,
                args=(
                    route_base_url,
                    route_api_key,
                    route_model,
                    messages,
                    max_tokens,
                    temperature,
                    DEFAULT_LLM_TIMEOUT_SEC,
                    queue,
                ),
            )
            proc.daemon = True
            try:
                proc.start()
                proc.join(DEFAULT_LLM_TIMEOUT_SEC + 2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(2)
                    raise TimeoutError(f"hard timeout after {DEFAULT_LLM_TIMEOUT_SEC}s")
                if queue.empty():
                    raise RuntimeError("route subprocess exited without result")
                result = queue.get()
                if result.get("ok"):
                    content = str(result.get("content") or "").strip()
                    if not content:
                        raise RuntimeError("empty response")
                    return content
                raise RuntimeError(str(result.get("error") or "unknown route error"))
            except Exception as exc:
                last_error = exc
                print(
                    json.dumps(
                        {
                            "stage": "llm_route_failed",
                            "model": route_model,
                            "base_url": route_base_url,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            finally:
                try:
                    if proc.is_alive():
                        proc.kill()
                        proc.join(2)
                except Exception:
                    pass
        raise RuntimeError(f"All LLM routes failed: {last_error}")


class HybridSceneRetriever:
    def __init__(self, scenes: Sequence[SceneRecord], embedder: OpenAICompatEmbedder):
        self.scenes = list(scenes)
        self.scene_by_id = {s.scene_id: s for s in self.scenes}
        self.index_texts = [scene_index_text(s) for s in self.scenes]
        self.bm25 = BM25Okapi([tokenize(t, s.language) for t, s in zip(self.index_texts, self.scenes)])
        self.embeddings = embedder.embed_documents(self.index_texts)
        self.embedder = embedder

    def vector_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q = self.embedder.embed_query(query)
        scores = self.embeddings @ q
        limit = min(top_k, len(self.scenes))
        if limit <= 0:
            return []
        idxs = np.argpartition(-scores, range(limit))[:limit]
        ranked = sorted(((int(i), float(scores[i])) for i in idxs), key=lambda x: x[1], reverse=True)
        return [(self.scenes[i].scene_id, s) for i, s in ranked]

    def bm25_search(self, query: str, language: str, top_k: int) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query, language))
        limit = min(top_k, len(self.scenes))
        if limit <= 0:
            return []
        idxs = np.argpartition(-scores, range(limit))[:limit]
        ranked = sorted(((int(i), float(scores[i])) for i in idxs), key=lambda x: x[1], reverse=True)
        return [(self.scenes[i].scene_id, s) for i, s in ranked]

    def retrieve(self, queries: Sequence[str], language: str, top_k_per_query: int = 12, final_top_k: int = 12) -> List[SceneHit]:
        fused: Dict[str, float] = defaultdict(float)
        trace: Dict[str, List[str]] = defaultdict(list)
        for query in queries:
            vec_hits = self.vector_search(query, top_k_per_query)
            bm25_hits = self.bm25_search(query, language, top_k_per_query)
            for rank, (scene_id, _) in enumerate(vec_hits, start=1):
                fused[scene_id] += 1.0 / (60 + rank)
                trace[scene_id].append(f"vector:{query}")
            for rank, (scene_id, _) in enumerate(bm25_hits, start=1):
                fused[scene_id] += 1.0 / (60 + rank)
                trace[scene_id].append(f"bm25:{query}")
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:final_top_k]
        hits: List[SceneHit] = []
        for scene_id, score in ranked:
            scene = self.scene_by_id[scene_id]
            hits.append(
                SceneHit(
                    scene_id=scene.scene_id,
                    scene_order=scene.scene_order,
                    scene_title=scene.scene_title,
                    text=scene.content,
                    score=score,
                    source="+".join(sorted(set(trace.get(scene_id, []))))[:400],
                )
            )
        return hits


def clean_text(text: Any) -> str:
    return WS_RE.sub(" ", str(text or "").replace("\u3000", " ").strip())


def normalize_name(text: Any) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return WS_RE.sub(" ", text).strip()


def stable_id(*parts: str, prefix: str) -> str:
    digest = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def tokenize(text: str, language: str) -> List[str]:
    raw = clean_text(text).lower()
    if not raw:
        return []
    if language == "zh":
        toks = TOKEN_RE.findall(raw)
        return toks if toks else list(raw)
    return EN_TOKEN_RE.findall(raw)


def detect_language(movie_dir: Path) -> str:
    return "zh" if movie_dir.parent.name.lower().startswith("ch") or movie_dir.parent.name.lower() == "chinese" else "en"


def load_scenes(script_path: Path, language: str) -> List[SceneRecord]:
    data = json.loads(script_path.read_text(encoding="utf-8"))
    scenes: List[SceneRecord] = []
    for idx, item in enumerate(data):
        scenes.append(
            SceneRecord(
                scene_id=str(item.get("_id", idx)),
                scene_order=idx + 1,
                scene_title=clean_text(item.get("title") or f"scene_{idx+1}"),
                subtitle=clean_text(item.get("subtitle")),
                content=str(item.get("content") or "").strip(),
                language=language,
            )
        )
    return scenes


def scene_index_text(scene: SceneRecord) -> str:
    content = scene.content[:1800]
    subtitle = f"\n{scene.subtitle}" if scene.subtitle else ""
    return f"{scene.scene_title}{subtitle}\n{content}"


def scene_card(scene: SceneRecord, max_chars: int = 700) -> Dict[str, Any]:
    excerpt = clean_text(scene.content)[:max_chars]
    return {
        "scene_id": scene.scene_id,
        "scene_order": scene.scene_order,
        "scene_title": scene.scene_title,
        "scene_excerpt": excerpt,
    }


def extract_speaker_candidates(scenes: Sequence[SceneRecord]) -> List[Dict[str, Any]]:
    utterance_count: Counter[str] = Counter()
    scene_count: Counter[str] = Counter()
    sample_scenes: Dict[str, List[str]] = defaultdict(list)
    for scene in scenes:
        seen_here = set()
        for raw_line in scene.content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = line.replace("(O.S.)", "").replace("(V.O.)", "")
            line = line.strip(" .:-")
            if not UPPER_LINE_RE.match(line):
                continue
            if len(line.split()) > 4:
                continue
            if line in STOP_SPEAKERS:
                continue
            name = clean_text(line.title().replace(" '", "'"))
            if not name:
                continue
            utterance_count[name] += 1
            if name not in seen_here:
                scene_count[name] += 1
                seen_here.add(name)
                if len(sample_scenes[name]) < 3:
                    sample_scenes[name].append(scene.scene_title)
    rows: List[Dict[str, Any]] = []
    for name, ucount in utterance_count.most_common(20):
        rows.append(
            {
                "name": name,
                "utterance_count": int(ucount),
                "scene_count": int(scene_count[name]),
                "sample_scenes": sample_scenes.get(name, []),
            }
        )
    return rows


def _strip_code_fences(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _clean_json_candidate(text: str) -> str:
    text = _strip_code_fences(text)
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def _find_balanced_json_block(text: str) -> Optional[str]:
    starts = [i for i, ch in enumerate(text) if ch == '{']
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _complete_json_object(text: str) -> str:
    text = _clean_json_candidate(text)
    if not text:
        return text
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in '{[':
            stack.append('}' if ch == '{' else ']')
        elif ch in '}]' and stack and ch == stack[-1]:
            stack.pop()
    if in_string:
        text += '"'
    while stack:
        text += stack.pop()
    return text


def extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    candidates = []
    cleaned = _clean_json_candidate(raw)
    if cleaned:
        candidates.append(cleaned)
    completed = _complete_json_object(cleaned)
    if completed and completed not in candidates:
        candidates.append(completed)
    balanced = _find_balanced_json_block(cleaned)
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    matches = JSON_BLOCK_RE.findall(cleaned)
    for match in matches[::-1]:
        cand = _clean_json_candidate(match)
        if cand and cand not in candidates:
            candidates.append(cand)
        completed_match = _complete_json_object(match)
        if completed_match and completed_match not in candidates:
            candidates.append(completed_match)
    last_error: Optional[Exception] = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception as exc:
            last_error = exc
            continue
    if json_repair is not None:
        for cand in candidates or [raw]:
            try:
                repaired = json_repair.repair_json(cand)
                if repaired:
                    return json.loads(repaired)
            except Exception as exc:
                last_error = exc
                continue
        try:
            repaired_obj = json_repair.loads(raw)
            if isinstance(repaired_obj, dict):
                return repaired_obj
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("No JSON object found in LLM output")


def repair_json_prompt(raw_text: str) -> List[Dict[str, str]]:
    system = (
        "You repair malformed JSON produced by another model. "
        "Return exactly one valid JSON object. Preserve keys, structure, and recoverable values whenever possible. "
        "Do not add explanations, markdown fences, or comments."
    )
    user = (
        "Repair the following malformed JSON into one valid JSON object. "
        "If the object appears truncated, complete brackets, quotes, and commas conservatively without inventing unsupported structure.\n\n"
        f"{raw_text[:12000]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_bool(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if raw in {"true", "yes", "1"}:
        return True
    if raw in {"false", "no", "0"}:
        return False
    return "true" in raw and "false" not in raw


def llm_json(llm: LLMClient, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 0.0, retries: int = 3) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    raw = ""
    for _ in range(retries + 1):
        try:
            raw = llm.run(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            last_error = exc
            continue
        if not clean_text(raw):
            last_error = ValueError("Empty LLM response")
            continue
        try:
            return extract_json(raw)
        except Exception as exc:
            last_error = exc
        if json_repair is not None:
            try:
                repaired_obj = json_repair.loads(raw)
                if isinstance(repaired_obj, dict):
                    return repaired_obj
            except Exception as repair_exc:
                last_error = repair_exc
        if clean_text(raw):
            try:
                repaired = llm.run(repair_json_prompt(raw), max_tokens=min(max_tokens, 2200), temperature=0.0)
                if clean_text(repaired):
                    return extract_json(repaired)
                last_error = ValueError("Empty repair response")
            except Exception as repair_exc:
                last_error = repair_exc
                continue
    raise ValueError(f"Failed to parse JSON after retries: {last_error}; raw_excerpt={raw[:400]!r}")

def selection_prompt(language: str, movie_id: str, candidates: Sequence[Dict[str, Any]], max_characters: int) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在为 STAGE Task 1 选择焦点角色。只输出 JSON。"
        user = (
            f"电影ID: {movie_id}\n"
            f"从候选角色里选择最能代表长程叙事发展的 {max_characters} 个焦点角色。\n"
            "优先选择跨多场戏重复出现、状态变化明显、关系或目标变化显著的角色。\n"
            "不要发明新名字，只能从候选列表里选。\n"
            "输出格式: {\"selected_characters\": [{\"character_name\":\"...\", \"aliases\":[\"...\"], \"selection_reason\":\"...\"}]}\n"
            f"候选列表:\n{json.dumps(list(candidates), ensure_ascii=False, indent=2)}"
        )
    else:
        system = "You are selecting focal characters for STAGE Task 1. Output JSON only."
        user = (
            f"Movie id: {movie_id}\n"
            f"Choose up to {max_characters} focal characters from the candidate list.\n"
            "Prefer characters who show durable cross-scene development, relationship change, or goal/conflict evolution.\n"
            "Do not invent new names. Select only from the candidates.\n"
            "Return format: {\"selected_characters\": [{\"character_name\":\"...\", \"aliases\":[\"...\"], \"selection_reason\":\"...\"}]}\n"
            f"Candidate list:\n{json.dumps(list(candidates), ensure_ascii=False, indent=2)}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def node_proposal_prompt(language: str, character: Dict[str, Any], scene_cards: Sequence[Dict[str, Any]], min_nodes: int, max_nodes: int) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在根据检索到的场景卡片，为焦点角色生成 scene-grounded timeline nodes。只输出 JSON。"
        user = (
            f"角色: {character['character_name']}\n"
            "任务：从给定场景中挑选对该角色具有持续叙事意义的发展节点。\n"
            f"尽量输出 {min_nodes}-{max_nodes} 个节点。\n"
            "只保留真正重要的发展，不要把每场戏都列成节点。\n"
            "每个节点必须绑定一个 scene_id。\n"
            "输出格式: {\"proposed_nodes\": [{\"scene_id\":\"...\", \"scene_order\":1, \"scene_title\":\"...\", \"importance\":\"core|supporting\", \"role_in_context\":\"...\", \"salient_development\":\"...\", \"goal_state\":\"...或null\", \"resulting_state\":\"...或null\", \"unresolved_issue\":\"...或null\"}]}\n"
            f"场景卡片:\n{json.dumps(list(scene_cards), ensure_ascii=False, indent=2)}"
        )
    else:
        system = "You are inducing scene-grounded focal-character timeline nodes from retrieved screenplay scene cards. Output JSON only."
        user = (
            f"Focal character: {character['character_name']}\n"
            f"Generate about {min_nodes}-{max_nodes} important timeline nodes from the supplied scene cards.\n"
            "Keep only scenes that contribute durable narrative development for the character.\n"
            "Do not turn every scene into a node. Each node must be tied to one supplied scene_id.\n"
            "Return format: {\"proposed_nodes\": [{\"scene_id\":\"...\", \"scene_order\":1, \"scene_title\":\"...\", \"importance\":\"core|supporting\", \"role_in_context\":\"...\", \"salient_development\":\"...\", \"goal_state\":\"... or null\", \"resulting_state\":\"... or null\", \"unresolved_issue\":\"... or null\"}]}\n"
            f"Scene cards:\n{json.dumps(list(scene_cards), ensure_ascii=False, indent=2)}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def timeline_refine_prompt(language: str, character_name: str, nodes: Sequence[Dict[str, Any]], min_nodes: int, max_nodes: int) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在把候选 timeline nodes 精修成紧凑的正式时间线。只输出 JSON。"
        user = (
            f"角色: {character_name}\n"
            f"请保留 {min_nodes}-{max_nodes} 个最能覆盖长程轨迹的节点。\n"
            "要求：保留 chronology；避免冗余；场景绑定保持不变；语言具体、克制、贴地。\n"
            "输出格式: {\"timeline_summary\":\"...\", \"final_nodes\": [{\"scene_id\":\"...\", \"scene_order\":1, \"scene_title\":\"...\", \"role_in_context\":\"...\", \"salient_development\":\"...\", \"goal_state\":\"...或null\", \"resulting_state\":\"...或null\", \"unresolved_issue\":\"...或null\"}]}\n"
            f"候选节点:\n{json.dumps(list(nodes), ensure_ascii=False, indent=2)}"
        )
    else:
        system = "You are refining candidate focal-character timeline nodes into a compact final timeline. Output JSON only."
        user = (
            f"Character: {character_name}\n"
            f"Keep {min_nodes}-{max_nodes} nodes that best cover the full trajectory.\n"
            "Maintain chronology and scene grounding. Remove redundancy. Keep the writing specific and conservative.\n"
            "Return format: {\"timeline_summary\":\"...\", \"final_nodes\": [{\"scene_id\":\"...\", \"scene_order\":1, \"scene_title\":\"...\", \"role_in_context\":\"...\", \"salient_development\":\"...\", \"goal_state\":\"... or null\", \"resulting_state\":\"... or null\", \"unresolved_issue\":\"... or null\"}]}\n"
            f"Candidate nodes:\n{json.dumps(list(nodes), ensure_ascii=False, indent=2)}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def node_grounding_prompt(language: str, character_name: str, node: Dict[str, Any], scene: SceneRecord) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在对一个最终 timeline node 做 scene-grounded 校正。只输出 JSON。"
        user = (
            f"角色: {character_name}\n"
            f"当前节点:\n{json.dumps(node, ensure_ascii=False, indent=2)}\n"
            f"场景标题: {scene.scene_title}\n"
            f"场景全文:\n{scene.content[:6000]}\n"
            "请把节点改写得更贴地，并抽取 1-4 条逐字 evidence_quotes。\n"
            "输出格式: {\"role_in_context\":\"...\", \"salient_development\":\"...\", \"goal_state\":\"...或null\", \"resulting_state\":\"...或null\", \"unresolved_issue\":\"...或null\", \"evidence_quotes\":[\"...\"]}"
        )
    else:
        system = "You are grounding and conservatively rewriting one final focal-character timeline node. Output JSON only."
        user = (
            f"Focal character: {character_name}\n"
            f"Draft node:\n{json.dumps(node, ensure_ascii=False, indent=2)}\n"
            f"Scene title: {scene.scene_title}\n"
            f"Scene text:\n{scene.content[:6000]}\n"
            "Rewrite the node to stay concrete and scene-grounded, and extract 1-4 exact evidence_quotes from the scene text when possible.\n"
            "Return format: {\"role_in_context\":\"...\", \"salient_development\":\"...\", \"goal_state\":\"... or null\", \"resulting_state\":\"... or null\", \"unresolved_issue\":\"... or null\", \"evidence_quotes\":[\"...\"]}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def arc_prompt(language: str, character_name: str, timeline_nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在根据角色时间线归纳跨场景 arcs。只输出 JSON。"
        user = (
            f"角色: {character_name}\n"
            "根据给定 timeline nodes 归纳 2-5 条高层发展弧线。\n"
            "每条 arc 至少连接 2 个节点。\n"
            "arc_focus 只能是 goal|relationship|status|conflict|mixed。\n"
            "输出格式: {\"arcs\": [{\"title\":\"...\", \"arc_focus\":\"goal|relationship|status|conflict|mixed\", \"linked_timeline_node_ids\":[\"...\"], \"arc_summary\":\"...\", \"start_state\":\"...或null\", \"end_state\":\"...或null\", \"unresolved_issue\":\"...或null\"}]}\n"
            f"timeline_nodes:\n{json.dumps(list(timeline_nodes), ensure_ascii=False, indent=2)}"
        )
    else:
        system = "You are inducing higher-level cross-scene arcs from a focal-character timeline. Output JSON only."
        user = (
            f"Character: {character_name}\n"
            "Induce 2-5 arcs from the timeline nodes. Each arc must connect at least two node ids.\n"
            "arc_focus must be one of goal|relationship|status|conflict|mixed.\n"
            "Return format: {\"arcs\": [{\"title\":\"...\", \"arc_focus\":\"goal|relationship|status|conflict|mixed\", \"linked_timeline_node_ids\":[\"...\"], \"arc_summary\":\"...\", \"start_state\":\"... or null\", \"end_state\":\"... or null\", \"unresolved_issue\":\"... or null\"}]}\n"
            f"Timeline nodes:\n{json.dumps(list(timeline_nodes), ensure_ascii=False, indent=2)}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def dev_judge_prompt(language: str, character_name: str, gold_node: Dict[str, Any], pred_node: Dict[str, Any]) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在评估两个同角色、同场景的 timeline node 是否在核心发展上对齐。只输出 TRUE 或 FALSE。"
        user = (
            f"角色: {character_name}\n金标节点:\n{json.dumps(gold_node, ensure_ascii=False, indent=2)}\n"
            f"预测节点:\n{json.dumps(pred_node, ensure_ascii=False, indent=2)}\n"
            "如果预测节点抓住了相同的核心叙事发展，可允许措辞不同和次要遗漏，输出 TRUE；若核心发展错了或冲突，输出 FALSE。"
        )
    else:
        system = "You are judging whether two timeline nodes for the same character and scene align on the core salient development. Output TRUE or FALSE only."
        user = (
            f"Character: {character_name}\nGold node:\n{json.dumps(gold_node, ensure_ascii=False, indent=2)}\n"
            f"Predicted node:\n{json.dumps(pred_node, ensure_ascii=False, indent=2)}\n"
            "Output TRUE if the predicted node captures the same core development, allowing paraphrase and minor omissions. Output FALSE if it misses or contradicts the key development."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def state_judge_prompt(language: str, character_name: str, gold_node: Dict[str, Any], pred_node: Dict[str, Any]) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在评估两个 timeline node 的状态转移是否大体一致。只输出 TRUE 或 FALSE。"
        user = (
            f"角色: {character_name}\n金标节点:\n{json.dumps(gold_node, ensure_ascii=False, indent=2)}\n"
            f"预测节点:\n{json.dumps(pred_node, ensure_ascii=False, indent=2)}\n"
            "重点看 goal_state、resulting_state、unresolved_issue 是否大体一致。允许部分缺省，但如果核心状态判断错误或相反，输出 FALSE。"
        )
    else:
        system = "You are judging whether two timeline nodes align on state transition. Output TRUE or FALSE only."
        user = (
            f"Character: {character_name}\nGold node:\n{json.dumps(gold_node, ensure_ascii=False, indent=2)}\n"
            f"Predicted node:\n{json.dumps(pred_node, ensure_ascii=False, indent=2)}\n"
            "Focus on goal_state, resulting_state, and unresolved_issue. Minor omissions are acceptable, but a wrong or contradictory state transition should be FALSE."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def arc_link_judge_prompt(language: str, character_name: str, gold_arc: Dict[str, Any], pred_arc: Dict[str, Any]) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在评估两个跨场景 arc 是否代表同一条发展线程。只输出 TRUE 或 FALSE。"
        user = (
            f"角色: {character_name}\n金标 arc:\n{json.dumps(gold_arc, ensure_ascii=False, indent=2)}\n"
            f"预测 arc:\n{json.dumps(pred_arc, ensure_ascii=False, indent=2)}\n"
            "如果预测 arc 覆盖了同一条核心发展线程，即使节点边界不完全相同，也输出 TRUE；否则输出 FALSE。"
        )
    else:
        system = "You are judging whether two cross-scene arcs represent the same underlying development thread. Output TRUE or FALSE only."
        user = (
            f"Character: {character_name}\nGold arc:\n{json.dumps(gold_arc, ensure_ascii=False, indent=2)}\n"
            f"Predicted arc:\n{json.dumps(pred_arc, ensure_ascii=False, indent=2)}\n"
            "Output TRUE if the predicted arc captures the same core development thread, even if the node boundaries differ somewhat. Output FALSE otherwise."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def arc_focus_judge_prompt(language: str, gold_arc: Dict[str, Any], pred_arc: Dict[str, Any]) -> List[Dict[str, str]]:
    if language == "zh":
        system = "你在评估预测 arc 的 focus 是否与金标一致。只输出 TRUE 或 FALSE。"
        user = (
            f"金标 focus: {gold_arc.get('arc_focus')}\n"
            f"金标 arc:\n{json.dumps(gold_arc, ensure_ascii=False, indent=2)}\n"
            f"预测 arc:\n{json.dumps(pred_arc, ensure_ascii=False, indent=2)}\n"
            "如果预测 arc_focus 与金标在大类上对齐，或合理地用 mixed 覆盖金标内容，输出 TRUE；否则 FALSE。"
        )
    else:
        system = "You are judging whether the predicted arc focus aligns with the gold arc focus. Output TRUE or FALSE only."
        user = (
            f"Gold focus: {gold_arc.get('arc_focus')}\n"
            f"Gold arc:\n{json.dumps(gold_arc, ensure_ascii=False, indent=2)}\n"
            f"Predicted arc:\n{json.dumps(pred_arc, ensure_ascii=False, indent=2)}\n"
            "Output TRUE if the predicted arc_focus matches the same broad category, or if mixed is a reasonable umbrella for the gold arc. Otherwise output FALSE."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def choose_best_arc_match(gold_arc: Dict[str, Any], pred_arcs: Sequence[Dict[str, Any]], embedder: OpenAICompatEmbedder) -> Optional[Dict[str, Any]]:
    if not pred_arcs:
        return None
    gold_scenes = set(gold_arc.get("linked_scene_ids", []) or [])
    gold_text = clean_text(gold_arc.get("title")) + "\n" + clean_text(gold_arc.get("arc_summary"))
    texts = [gold_text] + [clean_text(a.get("title")) + "\n" + clean_text(a.get("arc_summary")) for a in pred_arcs]
    vecs = embedder.embed_documents(texts)
    gold_vec = vecs[0]
    best = None
    best_score = -1.0
    for arc, vec in zip(pred_arcs, vecs[1:]):
        pred_scenes = set(arc.get("linked_scene_ids", []) or [])
        if gold_scenes or pred_scenes:
            inter = len(gold_scenes & pred_scenes)
            union = len(gold_scenes | pred_scenes)
            jaccard = inter / union if union else 0.0
        else:
            jaccard = 0.0
        sim = float(np.dot(gold_vec, vec))
        score = 0.7 * jaccard + 0.3 * sim
        if score > best_score:
            best_score = score
            best = arc
    return best


def run_bool_judge(llm: LLMClient, messages: List[Dict[str, str]]) -> bool:
    raw = llm.run(messages, max_tokens=8, temperature=0.0)
    return parse_bool(raw)


def build_character_queries(character: Dict[str, Any]) -> List[str]:
    name = character["character_name"]
    aliases = [a for a in character.get("aliases", []) if clean_text(a) and clean_text(a) != clean_text(name)]
    queries = [
        name,
        f"{name} key turning points major decisions state changes",
        f"{name} relationship conflict goal change",
        f"{name} important scenes personal trajectory",
    ]
    for alias in aliases[:2]:
        queries.append(alias)
    seen = set()
    out = []
    for q in queries:
        q = clean_text(q)
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def validate_nodes(nodes: Sequence[Dict[str, Any]], scenes: Sequence[SceneRecord]) -> List[Dict[str, Any]]:
    scene_map = {s.scene_id: s for s in scenes}
    out: List[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        scene_id = clean_text(node.get("scene_id"))
        if scene_id not in scene_map or scene_id in seen:
            continue
        scene = scene_map[scene_id]
        cleaned = {
            "scene_id": scene.scene_id,
            "scene_order": scene.scene_order,
            "scene_title": scene.scene_title,
            "importance": clean_text(node.get("importance")) or "core",
            "role_in_context": clean_text(node.get("role_in_context")),
            "salient_development": clean_text(node.get("salient_development")),
            "goal_state": clean_text(node.get("goal_state")) or None,
            "resulting_state": clean_text(node.get("resulting_state")) or None,
            "unresolved_issue": clean_text(node.get("unresolved_issue")) or None,
        }
        if not cleaned["salient_development"]:
            continue
        seen.add(scene_id)
        out.append(cleaned)
    return sorted(out, key=lambda x: x["scene_order"])


def run_workflow(movie_dir: Path, output_dir: Path, max_characters: int = 3) -> Dict[str, Any]:
    started = time.time()
    language = detect_language(movie_dir)
    scenes = load_scenes(movie_dir / "script.json", language)
    llm = LLMClient(DEFAULT_LLM_MODEL, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY)
    embedder = OpenAICompatEmbedder(DEFAULT_EMBED_MODEL, DEFAULT_EMBED_BASE_URL, DEFAULT_EMBED_API_KEY)
    retriever = HybridSceneRetriever(scenes, embedder)

    print(json.dumps({"stage":"load_complete","scene_count":len(scenes)}, ensure_ascii=False), flush=True)
    speaker_candidates = extract_speaker_candidates(scenes)
    print(json.dumps({"stage":"speaker_candidates","count":len(speaker_candidates)}, ensure_ascii=False), flush=True)
    print(json.dumps({"stage":"character_selection_start"}, ensure_ascii=False), flush=True)
    selection_raw = llm_json(llm, selection_prompt(language, movie_dir.name, speaker_candidates, max_characters), max_tokens=1200)
    print(json.dumps({"stage":"character_selection_done"}, ensure_ascii=False), flush=True)
    selected_characters = selection_raw.get("selected_characters", []) or []
    normalized_candidates = {normalize_name(item["name"]): item["name"] for item in speaker_candidates}
    characters: List[Dict[str, Any]] = []
    seen_names = set()
    for item in selected_characters:
        raw_name = clean_text(item.get("character_name"))
        key = normalize_name(raw_name)
        canonical = normalized_candidates.get(key, raw_name)
        if not canonical or canonical in seen_names:
            continue
        seen_names.add(canonical)
        aliases = [clean_text(x) for x in (item.get("aliases") or []) if clean_text(x)]
        if canonical not in aliases:
            aliases = [canonical] + aliases
        characters.append(
            {
                "character_name": canonical,
                "aliases": aliases[:4],
                "selection_reason": clean_text(item.get("selection_reason")) or "selected by LLM from speaker-derived candidates",
            }
        )

    timeline_payload = {
        "movie_id": movie_dir.name,
        "language": language,
        "task_name": "Story Dynamics Structuring",
        "task_version": "workflow_baseline_v1",
        "focal_character_timelines": [],
        "build_summary": {},
    }
    arc_payload = {
        "movie_id": movie_dir.name,
        "language": language,
        "task_name": "Story Dynamics Structuring",
        "task_version": "workflow_baseline_v1",
        "cross_scene_arcs": [],
        "build_summary": {},
    }
    diagnostics = {"speaker_candidates": speaker_candidates, "selected_characters": characters, "per_character": []}

    scene_map = {s.scene_id: s for s in scenes}

    print(json.dumps({"stage":"characters_ready","count":len(characters),"names":[c["character_name"] for c in characters]}, ensure_ascii=False), flush=True)

    for character in characters:
        print(json.dumps({"stage":"character_start","character":character["character_name"]}, ensure_ascii=False), flush=True)
        queries = build_character_queries(character)
        hits = retriever.retrieve(queries, language, top_k_per_query=12, final_top_k=min(14, len(scenes)))
        cards = [scene_card(scene_map[h.scene_id]) for h in hits]
        print(json.dumps({"stage":"node_proposal_start","character":character["character_name"],"retrieved_scene_count":len(cards)}, ensure_ascii=False), flush=True)
        proposal_raw = llm_json(llm, node_proposal_prompt(language, character, cards, min_nodes=4, max_nodes=10), max_tokens=4000)
        print(json.dumps({"stage":"node_proposal_done","character":character["character_name"]}, ensure_ascii=False), flush=True)
        proposed = validate_nodes(proposal_raw.get("proposed_nodes", []) or [], scenes)
        if not proposed:
            continue
        print(json.dumps({"stage":"timeline_refine_start","character":character["character_name"],"proposed_count":len(proposed)}, ensure_ascii=False), flush=True)
        refined_raw = llm_json(llm, timeline_refine_prompt(language, character["character_name"], proposed, min_nodes=4, max_nodes=10), max_tokens=4000)
        print(json.dumps({"stage":"timeline_refine_done","character":character["character_name"]}, ensure_ascii=False), flush=True)
        final_nodes = validate_nodes(refined_raw.get("final_nodes", []) or [], scenes)
        grounded_nodes: List[Dict[str, Any]] = []
        for node in final_nodes:
            scene = scene_map[node["scene_id"]]
            grounded = llm_json(llm, node_grounding_prompt(language, character["character_name"], node, scene), max_tokens=1200)
            final_node = dict(node)
            final_node.update(
                {
                    "timeline_node_id": stable_id(character["character_name"], node["scene_id"], prefix="ptu"),
                    "document_id": f"scene_{node['scene_id']}_part_1",
                    "scene_summary": clean_text(scene.content)[:240],
                    "goal_state": clean_text(grounded.get("goal_state")) or node.get("goal_state"),
                    "resulting_state": clean_text(grounded.get("resulting_state")) or node.get("resulting_state"),
                    "unresolved_issue": clean_text(grounded.get("unresolved_issue")) or node.get("unresolved_issue"),
                    "role_in_context": clean_text(grounded.get("role_in_context")) or node.get("role_in_context"),
                    "salient_development": clean_text(grounded.get("salient_development")) or node.get("salient_development"),
                    "related_event_ids": [],
                    "related_episode_ids": [],
                    "evidence_quotes": [clean_text(x) for x in (grounded.get("evidence_quotes") or []) if clean_text(x)][:4],
                    "auxiliary": {"relation_updates": [], "status_updates": [], "persona_anchor": ""},
                }
            )
            grounded_nodes.append(final_node)
        grounded_nodes = sorted(grounded_nodes, key=lambda x: x["scene_order"])
        timeline_summary = clean_text(refined_raw.get("timeline_summary"))
        if not timeline_summary:
            timeline_summary = f"The timeline traces {len(grounded_nodes)} key developments across {character['character_name']}'s screenplay trajectory."
        timeline_item = {
            "character_name": character["character_name"],
            "aliases": character.get("aliases", []),
            "selection_reason": character.get("selection_reason"),
            "task3_relevance": "Predicted focal character from screenplay-only workflow.",
            "timeline_nodes": grounded_nodes,
            "timeline_summary": timeline_summary,
        }
        timeline_payload["focal_character_timelines"].append(timeline_item)

        arc_input_nodes = [
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
            for n in grounded_nodes
        ]
        print(json.dumps({"stage":"arc_start","character":character["character_name"],"final_node_count":len(grounded_nodes)}, ensure_ascii=False), flush=True)
        arc_raw = llm_json(llm, arc_prompt(language, character["character_name"], arc_input_nodes), max_tokens=2800)
        print(json.dumps({"stage":"arc_done","character":character["character_name"]}, ensure_ascii=False), flush=True)
        node_id_to_scene_id = {n["timeline_node_id"]: n["scene_id"] for n in grounded_nodes}
        arcs: List[Dict[str, Any]] = []
        for item in (arc_raw.get("arcs") or []):
            linked_ids = [clean_text(x) for x in (item.get("linked_timeline_node_ids") or []) if clean_text(x) in node_id_to_scene_id]
            linked_ids = list(dict.fromkeys(linked_ids))
            if len(linked_ids) < 2:
                continue
            arcs.append(
                {
                    "arc_id": stable_id(character["character_name"], clean_text(item.get("title")), prefix="parc"),
                    "character_name": character["character_name"],
                    "title": clean_text(item.get("title")),
                    "arc_focus": clean_text(item.get("arc_focus")) or "mixed",
                    "linked_timeline_node_ids": linked_ids,
                    "arc_summary": clean_text(item.get("arc_summary")),
                    "start_state": clean_text(item.get("start_state")) or None,
                    "end_state": clean_text(item.get("end_state")) or None,
                    "unresolved_issue": clean_text(item.get("unresolved_issue")) or None,
                    "linked_scene_ids": [node_id_to_scene_id[x] for x in linked_ids],
                }
            )
        arc_payload["cross_scene_arcs"].extend(arcs)
        diagnostics["per_character"].append(
            {
                "character_name": character["character_name"],
                "queries": queries,
                "retrieved_scenes": [vars(h) for h in hits],
                "proposed_node_count": len(proposed),
                "final_node_count": len(grounded_nodes),
                "arc_count": len(arcs),
            }
        )

    timeline_payload["build_summary"] = {
        "candidate_character_count": len(speaker_candidates),
        "selected_focal_character_count": len(timeline_payload["focal_character_timelines"]),
        "timeline_node_count": sum(len(x["timeline_nodes"]) for x in timeline_payload["focal_character_timelines"]),
        "elapsed_sec": round(time.time() - started, 2),
    }
    arc_payload["build_summary"] = {
        "selected_focal_character_count": len(timeline_payload["focal_character_timelines"]),
        "arc_count": len(arc_payload["cross_scene_arcs"]),
        "elapsed_sec": round(time.time() - started, 2),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pred_task_1_character_timelines.json").write_text(json.dumps(timeline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "pred_task_1_cross_scene_arcs.json").write_text(json.dumps(arc_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"timeline": timeline_payload, "arcs": arc_payload, "diagnostics": diagnostics}


def evaluate(movie_dir: Path, output_dir: Path) -> Dict[str, Any]:
    llm = LLMClient(DEFAULT_LLM_MODEL, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_API_KEY)
    embedder = OpenAICompatEmbedder(DEFAULT_EMBED_MODEL, DEFAULT_EMBED_BASE_URL, DEFAULT_EMBED_API_KEY)
    pred_timeline = json.loads((output_dir / "pred_task_1_character_timelines.json").read_text(encoding="utf-8"))
    pred_arcs = json.loads((output_dir / "pred_task_1_cross_scene_arcs.json").read_text(encoding="utf-8"))
    gold_timeline = json.loads((movie_dir / "task_1_character_timelines.json").read_text(encoding="utf-8"))
    gold_arcs = json.loads((movie_dir / "task_1_cross_scene_arcs.json").read_text(encoding="utf-8"))
    language = gold_timeline.get("language", detect_language(movie_dir))

    gold_chars = gold_timeline.get("focal_character_timelines", []) or []
    pred_chars = pred_timeline.get("focal_character_timelines", []) or []

    gold_alias_map: Dict[str, str] = {}
    for item in gold_chars:
        canonical = item["character_name"]
        gold_alias_map[normalize_name(canonical)] = canonical
        for alias in item.get("aliases", []) or []:
            gold_alias_map[normalize_name(alias)] = canonical

    matched_pred_to_gold: Dict[str, str] = {}
    gold_selected: set[str] = set()
    for item in pred_chars:
        pred_name = item["character_name"]
        gold_name = gold_alias_map.get(normalize_name(pred_name))
        if gold_name and gold_name not in gold_selected:
            matched_pred_to_gold[pred_name] = gold_name
            gold_selected.add(gold_name)

    pred_name_set = set(matched_pred_to_gold.keys())
    gold_name_set = set(matched_pred_to_gold.values())
    char_precision = len(pred_name_set) / max(1, len(pred_chars))
    char_recall = len(gold_name_set) / max(1, len(gold_chars))
    char_f1 = 0.0 if char_precision + char_recall == 0 else 2 * char_precision * char_recall / (char_precision + char_recall)

    gold_by_name = {item["character_name"]: item for item in gold_chars}
    pred_pairs = set()
    gold_pairs = set()
    matched_pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

    for pred_name, gold_name in matched_pred_to_gold.items():
        pred_item = next(x for x in pred_chars if x["character_name"] == pred_name)
        gold_item = gold_by_name[gold_name]
        pred_nodes = pred_item.get("timeline_nodes", []) or []
        gold_nodes = gold_item.get("timeline_nodes", []) or []
        pred_scene_map = {str(n.get("scene_id")): n for n in pred_nodes}
        gold_scene_map = {str(n.get("scene_id")): n for n in gold_nodes}
        for sid in pred_scene_map:
            pred_pairs.add((gold_name, sid))
        for sid in gold_scene_map:
            gold_pairs.add((gold_name, sid))
        for sid in sorted(set(pred_scene_map) & set(gold_scene_map), key=lambda x: int(x) if x.isdigit() else x):
            matched_pairs.append((gold_name, gold_scene_map[sid], pred_scene_map[sid]))

    inter = pred_pairs & gold_pairs
    scene_precision = len(inter) / max(1, len(pred_pairs))
    scene_recall = len(inter) / max(1, len(gold_pairs))
    scene_f1 = 0.0 if scene_precision + scene_recall == 0 else 2 * scene_precision * scene_recall / (scene_precision + scene_recall)

    dev_votes = []
    state_votes = []
    for character_name, gold_node, pred_node in matched_pairs:
        dev_votes.append(run_bool_judge(llm, dev_judge_prompt(language, character_name, gold_node, pred_node)))
        state_votes.append(run_bool_judge(llm, state_judge_prompt(language, character_name, gold_node, pred_node)))

    pred_arc_items = pred_arcs.get("cross_scene_arcs", []) or []
    gold_arc_items = gold_arcs.get("cross_scene_arcs", []) or []
    pred_char_to_nodes = {
        item["character_name"]: {n["timeline_node_id"]: n for n in item.get("timeline_nodes", []) or []}
        for item in pred_chars
    }
    gold_char_to_nodes = {
        item["character_name"]: {n["timeline_node_id"]: n for n in item.get("timeline_nodes", []) or []}
        for item in gold_chars
    }
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

    arc_link_votes = []
    arc_focus_votes = []
    arc_pair_details = []
    for gold_arc in gold_arc_items:
        gold_name = gold_arc["character_name"]
        if gold_name not in gold_selected:
            continue
        gold_nodes_map = gold_char_to_nodes.get(gold_name, {})
        gold_scene_ids = [str(gold_nodes_map[nid]["scene_id"]) for nid in gold_arc.get("linked_timeline_node_ids", []) if nid in gold_nodes_map]
        gold_arc_item = dict(gold_arc)
        gold_arc_item["linked_scene_ids"] = gold_scene_ids
        best_pred = choose_best_arc_match(gold_arc_item, pred_arcs_by_gold_char.get(gold_name, []), embedder)
        if not best_pred:
            arc_link_votes.append(False)
            arc_focus_votes.append(False)
            arc_pair_details.append({"gold_arc_id": gold_arc.get("arc_id"), "matched": False})
            continue
        link_ok = run_bool_judge(llm, arc_link_judge_prompt(language, gold_name, gold_arc_item, best_pred))
        focus_ok = run_bool_judge(llm, arc_focus_judge_prompt(language, gold_arc_item, best_pred))
        arc_link_votes.append(link_ok)
        arc_focus_votes.append(focus_ok)
        arc_pair_details.append(
            {
                "gold_arc_id": gold_arc.get("arc_id"),
                "pred_arc_id": best_pred.get("arc_id"),
                "matched": True,
                "link_ok": link_ok,
                "focus_ok": focus_ok,
                "gold_scene_ids": gold_scene_ids,
                "pred_scene_ids": best_pred.get("linked_scene_ids", []),
            }
        )

    summary = {
        "movie_id": movie_dir.name,
        "language": language,
        "focal_character_metrics": {
            "pred_count": len(pred_chars),
            "gold_count": len(gold_chars),
            "matched_count": len(matched_pred_to_gold),
            "precision": round(char_precision, 4),
            "recall": round(char_recall, 4),
            "f1": round(char_f1, 4),
        },
        "timeline_metrics": {
            "pred_scene_pairs": len(pred_pairs),
            "gold_scene_pairs": len(gold_pairs),
            "matched_scene_pairs": len(inter),
            "scene_grounding_precision": round(scene_precision, 4),
            "scene_grounding_recall": round(scene_recall, 4),
            "scene_grounding_f1": round(scene_f1, 4),
            "matched_node_pair_count": len(matched_pairs),
            "development_correctness": round(sum(1 for x in dev_votes if x) / max(1, len(dev_votes)), 4),
            "state_transition_correctness": round(sum(1 for x in state_votes if x) / max(1, len(state_votes)), 4),
        },
        "arc_metrics": {
            "gold_arc_count_eval": len(arc_link_votes),
            "linkage_correctness": round(sum(1 for x in arc_link_votes if x) / max(1, len(arc_link_votes)), 4),
            "focus_correctness": round(sum(1 for x in arc_focus_votes if x) / max(1, len(arc_focus_votes)), 4),
        },
        "overall": {
            "task1_structuring_score": round((
                char_f1 + scene_f1 + (sum(1 for x in dev_votes if x) / max(1, len(dev_votes))) +
                (sum(1 for x in state_votes if x) / max(1, len(state_votes))) +
                (sum(1 for x in arc_link_votes if x) / max(1, len(arc_link_votes))) +
                (sum(1 for x in arc_focus_votes if x) / max(1, len(arc_focus_votes)))
            ) / 6.0, 4)
        },
        "matching": {
            "pred_to_gold_characters": matched_pred_to_gold,
            "arc_pair_details": arc_pair_details,
        },
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_report(movie_dir: Path, output_dir: Path, run_result: Dict[str, Any], eval_result: Dict[str, Any]) -> None:
    timeline = run_result["timeline"]
    arcs = run_result["arcs"]
    lines = []
    lines.append(f"# Task 1 Workflow Report: {movie_dir.name}")
    lines.append("")
    lines.append("## Workflow")
    lines.append("")
    lines.append("1. Extract speaker-derived character candidates from the screenplay.")
    lines.append("2. Use Qwen3-235B to select focal characters from those candidates.")
    lines.append("3. Build a scene-level hybrid retriever over the screenplay using BM25 + bge-m3 embeddings.")
    lines.append("4. For each focal character, retrieve relevant scenes and induce candidate timeline nodes.")
    lines.append("5. Refine the candidate nodes into a compact chronology and ground each final node back to the source scene.")
    lines.append("6. Derive cross-scene arcs from the final timeline nodes.")
    lines.append("7. Evaluate against released gold with structure-aware Task 1 metrics.")
    lines.append("")
    lines.append("## Generation Summary")
    lines.append("")
    lines.append(f"- Predicted focal characters: {len(timeline['focal_character_timelines'])}")
    lines.append(f"- Predicted timeline nodes: {sum(len(x['timeline_nodes']) for x in timeline['focal_character_timelines'])}")
    lines.append(f"- Predicted cross-scene arcs: {len(arcs['cross_scene_arcs'])}")
    lines.append("")
    lines.append("## Evaluation Summary")
    lines.append("")
    f = eval_result["focal_character_metrics"]
    t = eval_result["timeline_metrics"]
    a = eval_result["arc_metrics"]
    o = eval_result["overall"]
    lines.append(f"- Focal character F1: {f['f1']}")
    lines.append(f"- Timeline scene grounding F1: {t['scene_grounding_f1']}")
    lines.append(f"- Timeline development correctness: {t['development_correctness']}")
    lines.append(f"- Timeline state-transition correctness: {t['state_transition_correctness']}")
    lines.append(f"- Arc linkage correctness: {a['linkage_correctness']}")
    lines.append(f"- Arc focus correctness: {a['focus_correctness']}")
    lines.append(f"- Overall Task 1 structuring score: {o['task1_structuring_score']}")
    lines.append("")
    lines.append("## Predicted Characters")
    lines.append("")
    for item in timeline["focal_character_timelines"]:
        lines.append(f"- {item['character_name']}: {len(item['timeline_nodes'])} nodes")
        lines.append(f"  Summary: {item.get('timeline_summary','')}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This baseline uses screenplay-only input and does not consume Task 1 gold at generation time.")
    lines.append("- The current evaluation focuses on the release-supported Task 1 target: focal-character timelines and cross-scene arcs.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 workflow baseline with hybrid retrieval and structure-aware evaluation.")
    parser.add_argument("--movie_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_characters", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    movie_dir = Path(args.movie_dir)
    output_dir = Path(args.output_dir)
    run_result = run_workflow(movie_dir, output_dir, max_characters=args.max_characters)
    eval_result = evaluate(movie_dir, output_dir)
    write_report(movie_dir, output_dir, run_result, eval_result)
    print(json.dumps(eval_result, ensure_ascii=False, indent=2))
    print(str(output_dir / "report.md"))


if __name__ == "__main__":
    main()
