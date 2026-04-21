#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI
from rank_bm25 import BM25Okapi


DEFAULT_EMBED_BASE_URL = os.environ.get("TASK3_EMBED_BASE_URL", "http://localhost:8080/v1")
DEFAULT_EMBED_API_KEY = os.environ.get("TASK3_EMBED_API_KEY", "not-needed")
DEFAULT_EMBED_MODEL = os.environ.get("TASK3_EMBED_MODEL", "bge-m3")
DEFAULT_EMBED_TIMEOUT_SEC = int(os.environ.get("TASK3_EMBED_TIMEOUT_SEC", "180") or 180)
SUPPORTED_MEMORY_MODES = (
    "persona_only",
    "full_memory_all_in",
    "bm25_topk",
    "embedding_topk",
    "embedding_reranker_topk",
    "llm_selector_topk",
)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def render_list(values: Iterable[str]) -> str:
    rows = [normalize_ws(v) for v in values if normalize_ws(v)]
    if not rows:
        return "- none"
    return "\n".join(f"- {row}" for row in rows)


def tokenize(text: str, language: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    lower = language.lower()
    if lower == "chinese":
        return TOKEN_RE.findall(text.lower())
    return re.findall(r"[a-z0-9']+", text.lower())


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(float(x) * float(y) for x, y in zip(a, b))
    den_a = math.sqrt(sum(float(x) * float(x) for x in a))
    den_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if den_a <= 1e-12 or den_b <= 1e-12:
        return 0.0
    return num / (den_a * den_b)


def approximate_token_count(text: str, language: str) -> int:
    return len(tokenize(text, language))


def build_memory_document(memory: Dict[str, Any]) -> str:
    blocks = [normalize_ws(memory.get("memory_text"))]
    grounded = [normalize_ws(x) for x in memory.get("grounded_facts", []) or [] if normalize_ws(x)]
    if grounded:
        blocks.append("Grounded facts: " + " ".join(grounded))
    tags = [normalize_ws(x) for x in memory.get("tags", []) or [] if normalize_ws(x)]
    if tags:
        blocks.append("Tags: " + ", ".join(tags))
    return "\n".join(blocks)


def build_compact_persona_summary(persona_card: Dict[str, Any]) -> str:
    parts = []
    traits = [normalize_ws(x) for x in persona_card.get("traits", []) or [] if normalize_ws(x)]
    style = [normalize_ws(x) for x in persona_card.get("speaking_style", []) or [] if normalize_ws(x)]
    constraints = [normalize_ws(x) for x in persona_card.get("constraints", []) or [] if normalize_ws(x)]
    if traits:
        parts.append("Traits: " + ", ".join(traits[:6]))
    if style:
        parts.append("Style: " + ", ".join(style[:4]))
    if constraints:
        parts.append("Constraints: " + " ".join(constraints[:3]))
    return "\n".join(parts)


class OpenAICompatEmbedder:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: int = DEFAULT_EMBED_TIMEOUT_SEC):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self.model = model
        lower = model.lower()
        dual = any(x in lower for x in ("bge", "gte", "m3"))
        self.doc_prefix = "passage: " if dual else ""
        self.query_prefix = "query: " if dual else ""

    @staticmethod
    def _normalize(vec: Sequence[float]) -> List[float]:
        den = math.sqrt(sum(float(x) * float(x) for x in vec))
        if den <= 1e-12:
            return [0.0 for _ in vec]
        return [float(x) / den for x in vec]

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        out: List[List[float]] = []
        prefixed = [self.doc_prefix + str(text or "") for text in texts]
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(self._normalize(item.embedding) for item in resp.data)
        return out

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=[self.query_prefix + str(text or "")])
        return self._normalize(resp.data[0].embedding)


@dataclass
class MemorySelection:
    mode: str
    query: str
    selected_memories: List[Dict[str, Any]]
    selected_memory_ids: List[str]
    selected_memory_texts: List[str]
    score_rows: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]


class Task3RuntimeLoader:
    def __init__(
        self,
        *,
        role_asset: Dict[str, Any],
        language: str,
        embed_base_url: str = DEFAULT_EMBED_BASE_URL,
        embed_api_key: str = DEFAULT_EMBED_API_KEY,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ):
        self.role_asset = role_asset
        self.language = normalize_ws(language)
        self.memories = sorted(
            list(role_asset.get("memories", []) or []),
            key=lambda row: (int(row.get("scene_order") or 10**9), normalize_ws(row.get("memory_id"))),
        )
        self.relations = list(role_asset.get("relations", []) or [])
        self.persona_card = dict(role_asset.get("persona_card", {}) or {})
        self._memory_docs = [build_memory_document(memory) for memory in self.memories]
        self._bm25: BM25Okapi | None = None
        self._embedder: OpenAICompatEmbedder | None = None
        self._memory_vectors: List[List[float]] | None = None
        self.embed_base_url = embed_base_url
        self.embed_api_key = embed_api_key
        self.embed_model = embed_model

    def relation_context(self) -> List[Dict[str, Any]]:
        rows = []
        for item in self.relations:
            rows.append(
                {
                    "relation_id": normalize_ws(item.get("relation_id")),
                    "target_character": normalize_ws(item.get("target_character")),
                    "relation": normalize_ws(item.get("relation")) or normalize_ws(item.get("relation_summary")),
                    "relation_summary": normalize_ws(item.get("relation_summary")),
                }
            )
        return rows

    def compact_persona_summary(self) -> str:
        return build_compact_persona_summary(self.persona_card)

    def build_query(self, *, resolved_history: Sequence[Dict[str, Any]], current_user_turn: str) -> str:
        history_text = "\n".join(
            f"{normalize_ws(item.get('speaker'))}: {normalize_ws(item.get('text'))}"
            for item in resolved_history
            if normalize_ws(item.get("text"))
        )
        blocks = [self.compact_persona_summary()]
        if history_text:
            blocks.append("Dialogue history:\n" + history_text)
        blocks.append("Current user turn:\n" + normalize_ws(current_user_turn))
        return "\n\n".join(blocks)

    def _ensure_bm25(self) -> None:
        if self._bm25 is None:
            corpus = [tokenize(text, self.language) for text in self._memory_docs]
            self._bm25 = BM25Okapi(corpus)

    def _ensure_embeddings(self) -> None:
        if self._memory_vectors is None:
            self._embedder = OpenAICompatEmbedder(
                base_url=self.embed_base_url,
                api_key=self.embed_api_key,
                model=self.embed_model,
            )
            self._memory_vectors = self._embedder.embed_documents(self._memory_docs)

    def _materialize_selected_rows(self, ranked_rows: Sequence[tuple[int, float]], top_k: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        selected = []
        score_rows = []
        for index, score in list(ranked_rows)[:top_k]:
            memory = self.memories[index]
            selected.append(memory)
            score_rows.append(
                {
                    "memory_id": normalize_ws(memory.get("memory_id")),
                    "score": round(float(score), 6),
                    "scene_order": int(memory.get("scene_order") or 0),
                }
            )
        return selected, score_rows

    def select_memories(
        self,
        *,
        mode: str,
        resolved_history: Sequence[Dict[str, Any]],
        current_user_turn: str,
        top_k: int,
        source_memory_ids: Sequence[str] | None = None,
    ) -> MemorySelection:
        source_memory_ids = [normalize_ws(x) for x in source_memory_ids or [] if normalize_ws(x)]
        query = self.build_query(resolved_history=resolved_history, current_user_turn=current_user_turn)
        mode = normalize_ws(mode)

        if mode == "persona_only":
            selected = []
            score_rows = []
        elif mode == "full_memory_all_in":
            selected = list(self.memories)
            score_rows = [
                {
                    "memory_id": normalize_ws(memory.get("memory_id")),
                    "score": None,
                    "scene_order": int(memory.get("scene_order") or 0),
                }
                for memory in selected
            ]
        elif mode == "bm25_topk":
            self._ensure_bm25()
            assert self._bm25 is not None
            tokens = tokenize(query, self.language)
            scores = self._bm25.get_scores(tokens) if tokens else [0.0 for _ in self.memories]
            ranked = sorted(
                [(idx, float(score)) for idx, score in enumerate(scores)],
                key=lambda row: (row[1], -int(self.memories[row[0]].get("scene_order") or 0)),
                reverse=True,
            )
            selected, score_rows = self._materialize_selected_rows(ranked, top_k)
        elif mode == "embedding_topk":
            self._ensure_embeddings()
            assert self._embedder is not None and self._memory_vectors is not None
            query_vec = self._embedder.embed_query(query)
            ranked = sorted(
                [
                    (idx, cosine_similarity(query_vec, memory_vec))
                    for idx, memory_vec in enumerate(self._memory_vectors)
                ],
                key=lambda row: (row[1], -int(self.memories[row[0]].get("scene_order") or 0)),
                reverse=True,
            )
            selected, score_rows = self._materialize_selected_rows(ranked, top_k)
        elif mode in {"embedding_reranker_topk", "llm_selector_topk"}:
            raise NotImplementedError(f"memory mode not implemented yet: {mode}")
        else:
            raise ValueError(f"unsupported memory mode: {mode}")

        selected_ids = [normalize_ws(memory.get("memory_id")) for memory in selected]
        selected_texts = [normalize_ws(memory.get("memory_text")) for memory in selected]
        support_overlap = set(selected_ids) & set(source_memory_ids)
        support_hit = 1.0 if source_memory_ids and support_overlap else 0.0
        support_recall = (
            len(support_overlap) / len(set(source_memory_ids))
            if source_memory_ids
            else None
        )
        diagnostics = {
            "retrieval_method": mode,
            "selected_memory_count": len(selected),
            "selected_memory_tokens": sum(approximate_token_count(text, self.language) for text in selected_texts),
            "support_hit_at_k": support_hit,
            "support_recall_at_k": round(float(support_recall), 6) if support_recall is not None else None,
        }
        return MemorySelection(
            mode=mode,
            query=query,
            selected_memories=selected,
            selected_memory_ids=selected_ids,
            selected_memory_texts=selected_texts,
            score_rows=score_rows,
            diagnostics=diagnostics,
        )


def load_multi_turn_episode(
    *,
    stage_root: Path,
    language: str,
    movie_id: str,
    episode_instance_id: str,
) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    movie_dir = stage_root / language / movie_id
    payload = load_json(movie_dir / "task_3_in_script_character_role_play_multi_turn.json")
    for episode in payload.get("episodes", []) or []:
        episode_id = normalize_ws(episode.get("episode_id"))
        legacy_id = normalize_ws(episode.get("episode_instance_id"))
        instance_id = normalize_ws(episode.get("instance_id"))
        target = normalize_ws(episode_instance_id)
        if target in {episode_id, legacy_id, instance_id}:
            asset_ref = dict(episode.get("role_asset_ref", {}) or {})
            asset_file = normalize_ws(asset_ref.get("asset_file")) or "task_3_role_assets.json"
            role_assets = load_json(movie_dir / asset_file)
            character_name = normalize_ws(asset_ref.get("character_name")) or normalize_ws(episode.get("character"))
            for role in role_assets.get("roles", []) or []:
                if normalize_ws(role.get("character_name")) == character_name:
                    return movie_dir, episode, role
            raise KeyError(f"role asset not found for {character_name} in {movie_dir / asset_file}")
    raise KeyError(f"episode_instance_id not found: {episode_instance_id}")


def load_single_turn_instance(
    *,
    stage_root: Path,
    language: str,
    movie_id: str,
    instance_id: str,
) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    movie_dir = stage_root / language / movie_id
    payload = load_json(movie_dir / "task_3_in_script_character_role_play_single_turn.json")
    for instance in payload.get("instances", []) or []:
        current_id = normalize_ws(instance.get("instance_id"))
        if current_id != normalize_ws(instance_id):
            continue
        asset_ref = dict(instance.get("role_asset_ref", {}) or {})
        asset_file = normalize_ws(asset_ref.get("asset_file")) or "task_3_role_assets.json"
        role_assets = load_json(movie_dir / asset_file)
        character_name = normalize_ws(asset_ref.get("character_name")) or normalize_ws(instance.get("character"))
        for role in role_assets.get("roles", []) or []:
            if normalize_ws(role.get("character_name")) == character_name:
                return movie_dir, instance, role
        raise KeyError(f"role asset not found for {character_name} in {movie_dir / asset_file}")
    raise KeyError(f"instance_id not found: {instance_id}")
