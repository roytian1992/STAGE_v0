from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import sha256_file
from .models import Scene, stable_id


class TokenCounter(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    context_window: int
    max_output_tokens: int
    safety_margin_tokens: int
    target_chunk_tokens: int
    carry_context_max_tokens: int
    reconciliation_max_input_tokens: int
    prompt_headroom_tokens: int = 256

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkingConfig":
        config = cls(
            context_window=int(payload.get("context_window", 24000)),
            max_output_tokens=int(payload.get("max_output_tokens", 8192)),
            safety_margin_tokens=int(payload.get("safety_margin_tokens", 1024)),
            target_chunk_tokens=int(payload.get("target_chunk_tokens", 2400)),
            carry_context_max_tokens=int(payload.get("carry_context_max_tokens", 1200)),
            reconciliation_max_input_tokens=int(
                payload.get("reconciliation_max_input_tokens", 14000)
            ),
            prompt_headroom_tokens=int(payload.get("prompt_headroom_tokens", 256)),
        )
        if min(
            config.context_window,
            config.max_output_tokens,
            config.safety_margin_tokens,
            config.target_chunk_tokens,
            config.reconciliation_max_input_tokens,
        ) <= 0:
            raise ValueError("Chunking token budgets must be positive")
        if config.carry_context_max_tokens < 0:
            raise ValueError("carry_context_max_tokens must be non-negative")
        if config.prompt_headroom_tokens < 0:
            raise ValueError("prompt_headroom_tokens must be non-negative")
        if config.max_output_tokens + config.safety_margin_tokens >= config.context_window:
            raise ValueError("Output and safety budgets leave no model input context")
        if config.reconciliation_max_input_tokens > config.usable_model_input_tokens:
            raise ValueError(
                "reconciliation_max_input_tokens exceeds the available model input budget"
            )
        return config

    @property
    def max_model_input_tokens(self) -> int:
        return self.context_window - self.max_output_tokens - self.safety_margin_tokens

    @property
    def usable_model_input_tokens(self) -> int:
        """Input budget used for packing prompts, with provider overhead headroom.

        The measured tokenizer count does not include every chat-template or
        provider-side wrapper token.  Keeping this margin explicit avoids
        accepting prompts that are only a handful of tokens below the nominal
        limit.  It is a packing budget, not a truncation rule.
        """
        usable = self.max_model_input_tokens - self.prompt_headroom_tokens
        if usable <= 0:
            raise ValueError("Prompt headroom leaves no usable model input budget")
        return usable


@dataclass(frozen=True, slots=True)
class ScreenplayChunk:
    chunk_id: str
    order: int
    char_start: int
    char_end: int
    text: str
    token_count: int

    def prompt_text(self, scene: Scene) -> str:
        header = (
            f"Scene {scene.order} (source id {scene.source_scene_id}), "
            f"chunk {self.order}"
        )
        parts = [header]
        if scene.title:
            parts.append(f"Title: {scene.title}")
        if scene.subtitle:
            parts.append(f"Subtitle: {scene.subtitle}")
        if self.text:
            parts.append(self.text)
        return "\n".join(parts)

    def as_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "order": self.order,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "content": self.text,
        }


class TokenizerJsonCounter:
    def __init__(self, path: Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError(
                "Install the tokenizers dependency to use backend=tokenizer_json"
            ) from exc
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer JSON not found: {path}")
        self.path = path
        self._tokenizer = Tokenizer.from_file(str(path))
        self._metadata = {
            "backend": "tokenizer_json",
            "path": str(path),
            "sha256": sha256_file(path),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class TiktokenCounter:
    def __init__(self, encoding_name: str):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError("Install tiktoken to use backend=tiktoken") from exc
        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "tiktoken", "encoding_name": self.encoding_name}

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text)) if text else 0


def build_token_counter(payload: dict[str, Any]) -> TokenCounter:
    backend = str(payload.get("backend") or "").strip().casefold()
    if backend == "tokenizer_json":
        path = str(payload.get("path") or "").strip()
        if not path:
            raise ValueError("tokenizer backend=tokenizer_json requires path")
        return TokenizerJsonCounter(Path(path))
    if backend == "tiktoken":
        return TiktokenCounter(str(payload.get("encoding_name") or "o200k_base"))
    raise ValueError("tokenizer.backend must be tokenizer_json or tiktoken")


def chunk_scene(
    *,
    movie_id: str,
    scene: Scene,
    token_counter: TokenCounter,
    max_content_tokens: int,
) -> list[ScreenplayChunk]:
    if max_content_tokens <= 0:
        raise ValueError("max_content_tokens must be positive")
    content = scene.content
    if not content:
        return [
            ScreenplayChunk(
                chunk_id=stable_id("chunk", movie_id, scene.scene_id, 1, 0, 0),
                order=1,
                char_start=0,
                char_end=0,
                text="",
                token_count=0,
            )
        ]

    natural_spans = _paragraph_spans(content)
    fine_spans: list[tuple[int, int]] = []
    for start, end in natural_spans:
        fine_spans.extend(
            _split_span_to_budget(
                content,
                start=start,
                end=end,
                token_counter=token_counter,
                max_tokens=max_content_tokens,
            )
        )

    packed_spans: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for start, end in fine_spans:
        if current_start is None:
            current_start, current_end = start, end
            continue
        assert current_end is not None
        projected = content[current_start:end]
        if token_counter.count(projected) <= max_content_tokens:
            current_end = end
        else:
            packed_spans.append((current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        packed_spans.append((current_start, current_end))

    chunks: list[ScreenplayChunk] = []
    for order, (start, end) in enumerate(packed_spans, start=1):
        text = content[start:end]
        token_count = token_counter.count(text)
        if token_count > max_content_tokens:
            raise ValueError(
                f"Chunk exceeds token budget after splitting: scene={scene.scene_id} "
                f"chunk={order} tokens={token_count} budget={max_content_tokens}"
            )
        chunks.append(
            ScreenplayChunk(
                chunk_id=stable_id(
                    "chunk", movie_id, scene.scene_id, order, start, end, text
                ),
                order=order,
                char_start=start,
                char_end=end,
                text=text,
                token_count=token_count,
            )
        )

    _validate_chunk_coverage(content, chunks)
    return chunks


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n[ \t]*\n+", text):
        end = match.end()
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _split_span_to_budget(
    text: str,
    *,
    start: int,
    end: int,
    token_counter: TokenCounter,
    max_tokens: int,
) -> list[tuple[int, int]]:
    if token_counter.count(text[start:end]) <= max_tokens:
        return [(start, end)]
    boundaries = [
        start + match.end()
        for match in re.finditer(
            r"(?:\n+|[。！？；!?;](?:[\"'”’）】》]*)|[，,:：、])",
            text[start:end],
        )
    ]
    boundaries.append(end)
    output: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        fit_end = _largest_fitting_end(
            text,
            start=cursor,
            end=end,
            token_counter=token_counter,
            max_tokens=max_tokens,
        )
        if fit_end <= cursor:
            fit_end = min(end, cursor + 1)
        natural = [boundary for boundary in boundaries if cursor < boundary <= fit_end]
        if natural:
            preferred = natural[-1]
            if preferred - cursor >= max(1, int((fit_end - cursor) * 0.55)):
                fit_end = preferred
        output.append((cursor, fit_end))
        cursor = fit_end
    return output


def _largest_fitting_end(
    text: str,
    *,
    start: int,
    end: int,
    token_counter: TokenCounter,
    max_tokens: int,
) -> int:
    low = start + 1
    high = end
    best = start
    while low <= high:
        middle = (low + high) // 2
        if token_counter.count(text[start:middle]) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _validate_chunk_coverage(content: str, chunks: list[ScreenplayChunk]) -> None:
    if not chunks:
        raise ValueError("Non-empty scene produced no chunks")
    if chunks[0].char_start != 0 or chunks[-1].char_end != len(content):
        raise ValueError("Chunk spans do not cover the full scene")
    for left, right in zip(chunks, chunks[1:]):
        if left.char_end != right.char_start:
            raise ValueError("Chunk spans contain a gap or overlap")
    reconstructed = "".join(chunk.text for chunk in chunks)
    if reconstructed != content:
        raise ValueError("Chunk reconstruction differs from the source scene")
