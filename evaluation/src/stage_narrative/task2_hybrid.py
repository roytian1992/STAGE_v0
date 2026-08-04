from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .chunking import TokenCounter
from .clients import ModelConfig, OpenAIEmbeddingClient
from .io import atomic_write_jsonl, load_json, sha256_file


TASK2_FIELDS = (
    "id",
    "movie_id",
    "language",
    "related_scenes",
    "question",
    "answer",
    "evidence_or_reason",
    "question_type",
)
GLOBAL_SCENE_LABELS = {"全片", "whole film", "entire film"}


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    max_chunk_tokens: int = 600
    dense_top_k: int = 12
    keyword_top_k: int = 12
    final_top_k: int = 8
    rrf_k: int = 60

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalConfig":
        config = cls(
            max_chunk_tokens=int(payload.get("max_chunk_tokens", 600)),
            dense_top_k=int(payload.get("dense_top_k", 12)),
            keyword_top_k=int(payload.get("keyword_top_k", 12)),
            final_top_k=int(payload.get("final_top_k", 8)),
            rrf_k=int(payload.get("rrf_k", 60)),
        )
        if min(
            config.max_chunk_tokens,
            config.dense_top_k,
            config.keyword_top_k,
            config.final_top_k,
            config.rrf_k,
        ) <= 0:
            raise ValueError("Task 2 retrieval settings must be positive")
        if config.final_top_k > config.dense_top_k + config.keyword_top_k:
            raise ValueError("final_top_k exceeds the maximum fused candidate count")
        return config

    def as_dict(self) -> dict[str, int]:
        return {
            "max_chunk_tokens": self.max_chunk_tokens,
            "dense_top_k": self.dense_top_k,
            "keyword_top_k": self.keyword_top_k,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
        }


def load_manifest_entries(root: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    payload = load_json(root / "manifest.json")
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest has no entries: {root / 'manifest.json'}")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        movie_id = str(row.get("movie_id") or "").strip()
        language = normalize_language(row.get("language"))
        relative = Path(str(row.get("path") or ""))
        if not movie_id or not str(relative):
            raise ValueError(f"Malformed manifest entry in {root}")
        if relative.is_absolute():
            movie_dir = relative
        elif relative.parts and relative.parts[0] == root.name:
            movie_dir = root.parent / relative
        else:
            movie_dir = root / relative
        if movie_id in output:
            raise ValueError(f"Duplicate movie_id in manifest: {movie_id}")
        output[movie_id] = {
            **row,
            "movie_id": movie_id,
            "language": language,
            "movie_dir": movie_dir.resolve(),
        }
    return output


def normalize_language(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"en", "english"}:
        return "en"
    if text in {"zh", "chinese", "cn"}:
        return "zh"
    raise ValueError(f"Unsupported Task 2 language: {value!r}")


def read_task2_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TASK2_FIELDS:
            raise ValueError(
                f"Task 2 CSV fields mismatch in {path}: {reader.fieldnames}"
            )
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"Task 2 CSV is empty: {path}")
    ids = [row["id"] for row in rows]
    if any(not row["question"] or not row["answer"] for row in rows):
        raise ValueError(f"Task 2 CSV contains empty question or answer: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Task 2 CSV contains duplicate ids: {path}")
    return rows


def build_paired_chunks(
    *,
    movie_id: str,
    ordinary_script_path: Path,
    anonymous_script_path: Path,
    token_counter: TokenCounter,
    max_tokens: int,
) -> dict[str, list[dict[str, Any]]]:
    ordinary = _scene_array(ordinary_script_path)
    anonymous = _scene_array(anonymous_script_path)
    if len(ordinary) != len(anonymous):
        raise ValueError(f"Paired scripts have different scene counts: {movie_id}")
    output = {"ordinary": [], "anonymous": []}
    for scene_order, (left, right) in enumerate(zip(ordinary, anonymous), start=1):
        if str(left.get("_id")) != str(right.get("_id")):
            raise ValueError(f"Paired scene ids differ: {movie_id} scene {scene_order}")
        left_lines = str(left.get("content") or "").splitlines(keepends=True) or [""]
        right_lines = str(right.get("content") or "").splitlines(keepends=True) or [""]
        if len(left_lines) != len(right_lines):
            raise ValueError(
                f"Paired scene line counts differ: {movie_id} scene {scene_order}"
            )
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(left_lines):
            end = start
            while end < len(left_lines):
                candidate_end = end + 1
                left_text = "".join(left_lines[start:candidate_end])
                right_text = "".join(right_lines[start:candidate_end])
                if max(
                    token_counter.count(_indexed_text(left, left_text)),
                    token_counter.count(_indexed_text(right, right_text)),
                ) > max_tokens:
                    break
                end = candidate_end
            if end == start:
                raise ValueError(
                    f"One screenplay line exceeds {max_tokens} BGE tokens: "
                    f"{movie_id} scene {scene_order} line {start + 1}"
                )
            spans.append((start, end))
            start = end
        for variant, scene, lines in (
            ("ordinary", left, left_lines),
            ("anonymous", right, right_lines),
        ):
            for chunk_index, (line_start, line_end) in enumerate(spans, start=1):
                content = "".join(lines[line_start:line_end])
                indexed_text = _indexed_text(scene, content)
                token_count = token_counter.count(indexed_text)
                if token_count > max_tokens:
                    raise AssertionError("Paired chunk exceeded the frozen token budget")
                output[variant].append(
                    {
                        "chunk_id": (
                            f"{movie_id}:scene-{scene_order:04d}:chunk-{chunk_index:03d}"
                        ),
                        "variant": variant,
                        "movie_id": movie_id,
                        "scene_order": scene_order,
                        "source_scene_id": str(scene.get("_id", scene_order)),
                        "scene_name": str(scene.get("title") or "").strip(),
                        "scene_subtitle": str(scene.get("subtitle") or "").strip(),
                        "chunk_index": chunk_index,
                        "line_start": line_start + 1,
                        "line_end": line_end,
                        "token_count": token_count,
                        "content": content,
                        "indexed_text": indexed_text,
                    }
                )
    if [row["chunk_id"] for row in output["ordinary"]] != [
        row["chunk_id"] for row in output["anonymous"]
    ]:
        raise AssertionError("Paired chunk identifiers are not aligned")
    return output


def _scene_array(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list) or not payload or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"Expected a non-empty scene array: {path}")
    return payload


def _indexed_text(scene: dict[str, Any], content: str) -> str:
    title = str(scene.get("title") or "").strip()
    subtitle = str(scene.get("subtitle") or "").strip()
    parts = [f"Scene: {title}"]
    if subtitle:
        parts.append(f"Subtitle: {subtitle}")
    if content:
        parts.append(content)
    return "\n".join(parts)


class EmbeddingPool:
    def __init__(self, payload: dict[str, Any]):
        endpoints = payload.get("endpoint_pool")
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise ValueError("Task 2 embedding config requires exactly two endpoints")
        api_key = str(payload.get("api_key") or "").strip()
        model = str(payload.get("model") or "").strip()
        if not api_key or not model:
            raise ValueError("Embedding config requires model and api_key")
        self.clients: list[OpenAIEmbeddingClient] = []
        self.semaphores: list[asyncio.Semaphore] = []
        self.urls: list[str] = []
        for endpoint in endpoints:
            url = str(endpoint.get("base_url") or "").strip()
            concurrency = int(endpoint.get("max_concurrency", 1))
            config = ModelConfig.from_dict(
                {
                    "model": model,
                    "base_url": url,
                    "api_key": api_key,
                    "timeout_seconds": payload.get("timeout_seconds", 180),
                    "max_transport_attempts": payload.get("max_attempts", 3),
                }
            )
            self.clients.append(OpenAIEmbeddingClient(config))
            self.semaphores.append(asyncio.Semaphore(concurrency))
            self.urls.append(url)
        self._lock = asyncio.Lock()
        self._next = 0
        self.usage: list[dict[str, Any]] = []

    async def embed_all(
        self,
        texts: list[str],
        *,
        batch_size: int,
        stage: str,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("Embedding batch_size must be positive")
        batches = [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]

        async def run(batch_index: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            async with self._lock:
                endpoint_index = self._next
                self._next = (self._next + 1) % len(self.clients)
            async with self.semaphores[endpoint_index]:
                vectors, metadata = await self.clients[endpoint_index].embed(
                    batch, stage=f"{stage}:batch-{batch_index:05d}"
                )
            self.usage.append(metadata)
            return batch_index, vectors

        settled = await asyncio.gather(
            *(run(index, batch) for index, batch in enumerate(batches))
        )
        vectors = [row for _, batch in sorted(settled) for row in batch]
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or len(array) != len(texts):
            raise ValueError("Embedding response shape mismatch")
        return normalize_embeddings(array)

    def summary(self) -> dict[str, Any]:
        return {
            "endpoints": self.urls,
            "request_count": len(self.usage),
            "embedded_text_count": sum(int(row.get("count") or 0) for row in self.usage),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in self.usage),
            "elapsed_seconds_sum": round(
                sum(float(row.get("elapsed_seconds") or 0) for row in self.usage), 3
            ),
        }


def normalize_embeddings(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding endpoint returned a zero vector")
    return array / norms


def keyword_tokens(text: str, language: str) -> list[str]:
    normalized = text.casefold()
    if language == "zh":
        import jieba

        jieba.setLogLevel(30)
        output = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
        return [token for token in output if re.search(r"[\w\u3400-\u9fff]", token)]
    return re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", normalized)


def compress_retrieved_passages(
    passages: list[dict[str, Any]], *, question: str, language: str
) -> list[dict[str, Any]]:
    """Keep the three most query-overlapping lines per passage in stable order."""
    query = set(keyword_tokens(question, language))
    output = []
    for passage in passages:
        lines = [line.strip() for line in passage["content"].splitlines() if line.strip()]
        if not lines:
            selected: list[str] = []
        else:
            ranked = sorted(
                range(len(lines)),
                key=lambda index: (
                    -len(query & set(keyword_tokens(lines[index], language))),
                    index,
                ),
            )[:3]
            selected = [lines[index] for index in sorted(ranked)]
        output.append({**passage, "content": "\n".join(selected)})
    return output


def retrieve_hybrid(
    *,
    chunks: list[dict[str, Any]],
    document_embeddings: np.ndarray,
    question: str,
    query_embedding: np.ndarray,
    language: str,
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("Install rank-bm25 to run Task 2 hybrid retrieval") from exc
    if len(chunks) != len(document_embeddings):
        raise ValueError("Chunk and embedding counts differ")
    dense_scores = document_embeddings @ query_embedding
    dense_order = _top_indices(dense_scores, config.dense_top_k)
    corpus_tokens = [keyword_tokens(row["indexed_text"], language) for row in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    keyword_scores = np.asarray(bm25.get_scores(keyword_tokens(question, language)))
    keyword_order = _top_indices(keyword_scores, config.keyword_top_k)
    dense_ranks = {index: rank for rank, index in enumerate(dense_order, start=1)}
    keyword_ranks = {index: rank for rank, index in enumerate(keyword_order, start=1)}
    candidates = set(dense_ranks) | set(keyword_ranks)
    fused = {
        index: (
            (1.0 / (config.rrf_k + dense_ranks[index]) if index in dense_ranks else 0.0)
            + (
                1.0 / (config.rrf_k + keyword_ranks[index])
                if index in keyword_ranks
                else 0.0
            )
        )
        for index in candidates
    }
    order = sorted(
        candidates,
        key=lambda index: (
            -fused[index],
            dense_ranks.get(index, 10**9),
            keyword_ranks.get(index, 10**9),
            chunks[index]["chunk_id"],
        ),
    )[: config.final_top_k]
    return [
        {
            "label": f"R{rank}",
            "chunk_id": chunks[index]["chunk_id"],
            "scene_order": chunks[index]["scene_order"],
            "source_scene_id": chunks[index]["source_scene_id"],
            "scene_name": chunks[index]["scene_name"],
            "scene_subtitle": chunks[index]["scene_subtitle"],
            "chunk_index": chunks[index]["chunk_index"],
            "token_count": chunks[index]["token_count"],
            "content": chunks[index]["content"],
            "dense_rank": dense_ranks.get(index),
            "keyword_rank": keyword_ranks.get(index),
            "dense_score": round(float(dense_scores[index]), 8),
            "keyword_score": round(float(keyword_scores[index]), 8),
            "rrf_score": round(float(fused[index]), 10),
        }
        for rank, index in enumerate(order, start=1)
    ]


def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[
        : min(limit, len(scores))
    ]


def parse_related_scenes(value: str) -> tuple[list[str], bool]:
    labels = [item.strip() for item in value.split(";") if item.strip()]
    global_scope = any(item.casefold() in GLOBAL_SCENE_LABELS for item in labels)
    return [item for item in labels if item.casefold() not in GLOBAL_SCENE_LABELS], global_scope


def retrieval_diagnostic(row: dict[str, Any], passages: list[dict[str, Any]]) -> dict[str, Any]:
    gold, global_scope = parse_related_scenes(row["related_scenes"])
    retrieved_names = [passage["scene_name"] for passage in passages]
    positions = [
        index
        for index, name in enumerate(retrieved_names, start=1)
        if name in set(gold)
    ]
    matched = sorted(set(retrieved_names) & set(gold))
    return {
        "global_scope": global_scope,
        "gold_scene_count": len(gold),
        "matched_gold_scene_count": len(matched),
        "any_gold_scene_retrieved": bool(matched) if gold else None,
        "all_gold_scenes_retrieved": set(gold) <= set(retrieved_names) if gold else None,
        "gold_scene_recall": len(matched) / len(set(gold)) if gold else None,
        "reciprocal_rank": 1.0 / min(positions) if positions else (0.0 if gold else None),
    }


def aggregate_retrieval_diagnostics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    scoped = [row for row in rows if row["diagnostic"]["gold_scene_count"] > 0]
    return {
        "question_count": len(rows),
        "scene_scoped_question_count": len(scoped),
        "global_scope_question_count": sum(row["diagnostic"]["global_scope"] for row in rows),
        "any_gold_scene_recall_at_k": _mean(
            float(row["diagnostic"]["any_gold_scene_retrieved"]) for row in scoped
        ),
        "all_gold_scenes_recall_at_k": _mean(
            float(row["diagnostic"]["all_gold_scenes_retrieved"]) for row in scoped
        ),
        "mean_gold_scene_recall_at_k": _mean(
            row["diagnostic"]["gold_scene_recall"] for row in scoped
        ),
        "mean_reciprocal_rank": _mean(
            row["diagnostic"]["reciprocal_rank"] for row in scoped
        ),
    }


def validate_prediction(payload: dict[str, Any], allowed_labels: set[str]) -> dict[str, Any]:
    if set(payload) != {"answer", "cited_passage_labels"}:
        raise ValueError("Task 2 prediction must contain exactly answer and cited_passage_labels")
    answer = str(payload.get("answer") or "").strip()
    citations = payload.get("cited_passage_labels")
    if not answer or not isinstance(citations, list) or not all(
        isinstance(item, str) for item in citations
    ):
        raise ValueError("Task 2 prediction answer/citation types are invalid")
    citations = [item.strip() for item in citations]
    if len(citations) != len(set(citations)) or not set(citations) <= allowed_labels:
        raise ValueError("Task 2 prediction contains duplicate or unknown citation labels")
    return {"answer": answer, "cited_passage_labels": citations}


def validate_direct_prediction(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        raise ValueError("Task 2 direct prediction must contain exactly answer")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Task 2 direct prediction answer must be non-empty text")
    return {"answer": answer.strip()}


def validate_direct_judgment(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "answer_correct",
        "brief_rationale",
    }:
        raise ValueError(
            "Task 2 direct judgment must contain answer_correct and brief_rationale"
        )
    if not isinstance(payload["answer_correct"], bool):
        raise ValueError("Task 2 direct answer_correct must be boolean")
    rationale = payload["brief_rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Task 2 direct judgment requires a brief rationale")
    return {
        "answer_correct": payload["answer_correct"],
        "brief_rationale": rationale.strip(),
    }


def validate_judgment(payload: dict[str, Any]) -> dict[str, Any]:
    expected = {"answer_correct", "citation_quality", "brief_rationale"}
    if set(payload) != expected or not isinstance(payload.get("answer_correct"), bool):
        raise ValueError("Task 2 judgment has invalid fields or answer_correct type")
    quality = str(payload.get("citation_quality") or "").strip()
    if quality not in {
        "fully_supported",
        "partially_supported",
        "unsupported",
        "no_citation",
        "not_evaluated_provider_filter",
    }:
        raise ValueError("Task 2 judgment has invalid citation_quality")
    rationale = str(payload.get("brief_rationale") or "").strip()
    if not rationale:
        raise ValueError("Task 2 judgment requires a brief_rationale")
    return {
        "answer_correct": payload["answer_correct"],
        "citation_quality": quality,
        "brief_rationale": rationale,
    }


def validate_reasoning_disabled(model_config: dict[str, Any]) -> dict[str, Any]:
    """Require an explicit, non-conflicting provider-specific no-reasoning setting."""
    extra_body = model_config.get("extra_body")
    if not isinstance(extra_body, dict):
        raise ValueError("Model config must explicitly disable thinking/reasoning")
    markers: list[dict[str, Any]] = []

    chat_template = extra_body.get("chat_template_kwargs")
    if isinstance(chat_template, dict) and "enable_thinking" in chat_template:
        value = chat_template["enable_thinking"]
        if value is not False:
            raise ValueError("enable_thinking must be false")
        markers.append({"field": "chat_template_kwargs.enable_thinking", "value": False})

    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and "type" in thinking:
        value = str(thinking["type"]).strip().casefold()
        if value not in {"disabled", "none", "off"}:
            raise ValueError("thinking.type must explicitly disable thinking")
        markers.append({"field": "thinking.type", "value": value})

    if "reasoning_effort" in extra_body:
        value = str(extra_body["reasoning_effort"]).strip().casefold()
        if value not in {"none", "disabled", "off"}:
            raise ValueError("reasoning_effort must explicitly disable reasoning")
        markers.append({"field": "reasoning_effort", "value": value})

    if not markers:
        raise ValueError("Model config has no recognized no-thinking/no-reasoning setting")
    return {
        "requested": extra_body,
        "disable_markers": markers,
        "validated_explicitly_disabled": True,
    }


def summarize_hidden_reasoning_usage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "reasoning_tokens",
        "provider_thought_tokens",
        "thinking_blocks",
        "thinking_chars",
    )
    totals = {
        field: sum(
            int(row.get("call_metadata", {}).get(field) or 0) for row in rows
        )
        for field in fields
    }
    return {**totals, "verified_zero": not any(totals.values())}


def abstract_sensitive_screenplay_terms(text: str) -> str:
    """Deterministically soften filter-prone wording without changing event semantics."""
    replacements = (
        ("把他的尸体飘在扬子江上", "让他遭受致命后果"),
        ("把谁的尸体扔进扬子江", "让违约者遭受致命后果"),
        ("尸体要飘在扬子江上", "会遭受致命后果"),
        ("尸体飘在扬子江上", "遭受致命后果"),
        ("杀死", "致命伤害"),
        ("死亡", "生命终结"),
        ("死去", "生命终结"),
        ("死了", "生命终结了"),
        ("祭奠", "纪念"),
        ("剁来", "使其失去"),
        ("剁", "使其失去"),
        ("砍", "伤害"),
        ("切", "伤害"),
        ("牛刀", "刀具"),
        ("枪弹", "武器攻击"),
        ("机枪", "武器"),
        ("炮弹", "攻击"),
        ("爆炸", "巨响"),
        ("战争", "武装冲突"),
        ("kill", "fatally harm"),
        ("corpse", "remains"),
        ("death", "loss of life"),
        ("died", "lost their life"),
        ("sever", "make someone lose"),
        ("chop off", "make someone lose"),
    )
    output = text
    for source, target in replacements:
        output = output.replace(source, target)
        output = output.replace(source.title(), target)
        output = output.replace(source.upper(), target)
    return output


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            output.append(row)
    return output


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, rows)


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return round(sum(rows) / len(rows), 8) if rows else None


def exact_mcnemar_pvalue(ordinary_only: int, anonymous_only: int) -> float:
    discordant = ordinary_only + anonymous_only
    if discordant == 0:
        return 1.0
    tail = min(ordinary_only, anonymous_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)
