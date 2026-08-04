from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, TypeVar


class ModelTransportError(RuntimeError):
    pass


class ModelContentFilterError(ModelTransportError):
    pass


class ModelResponseParseError(ValueError):
    """A received model response that could not be parsed as one JSON object."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str,
        metadata: dict[str, Any],
    ):
        super().__init__(message)
        self.raw_text = raw_text
        self.metadata = metadata


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str
    base_url: str
    api_key: str
    api_style: str = "openai"
    timeout_seconds: float = 180.0
    max_transport_attempts: int = 3
    temperature: float = 0.0
    json_response_format: bool = True
    max_tokens: int | None = None
    extra_body: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        model = str(payload.get("model") or "").strip()
        base_url = _direct_or_environment(payload, "base_url")
        api_key = _direct_or_environment(payload, "api_key")
        if not model or not base_url or not api_key:
            raise ValueError("Model config requires non-empty model, base_url, and api_key")
        api_style = str(payload.get("api_style") or "openai").strip().casefold()
        if api_style not in {"openai", "anthropic"}:
            raise ValueError("Model config api_style must be openai or anthropic")
        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_style=api_style,
            timeout_seconds=float(payload.get("timeout_seconds", 180.0)),
            max_transport_attempts=max(
                1,
                int(
                    payload.get(
                        "max_transport_attempts", payload.get("max_attempts", 3)
                    )
                ),
            ),
            temperature=float(payload.get("temperature", 0.0)),
            json_response_format=bool(payload.get("json_response_format", True)),
            max_tokens=(
                int(payload["max_tokens"])
                if payload.get("max_tokens") is not None
                else None
            ),
            extra_body=(
                dict(payload["extra_body"])
                if isinstance(payload.get("extra_body"), dict)
                else None
            ),
        )


def _direct_or_environment(payload: dict[str, Any], key: str) -> str:
    direct = str(payload.get(key) or "").strip()
    env_key = f"{key}_env"
    environment_name = str(payload.get(env_key) or "").strip()
    if direct and environment_name:
        raise ValueError(f"Model config cannot set both {key} and {env_key}")
    if environment_name:
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", environment_name):
            raise ValueError(f"Model config {env_key} is not a valid environment name")
        resolved = str(os.environ.get(environment_name) or "").strip()
        if not resolved:
            raise ValueError(
                f"Model config environment variable is missing or empty: {environment_name}"
            )
        return resolved
    return direct


@dataclass(frozen=True, slots=True)
class JsonCall:
    data: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextCall:
    text: str
    metadata: dict[str, Any]


TCall = TypeVar("TCall", JsonCall, TextCall)


@dataclass(frozen=True, slots=True)
class EndpointPoolEntry:
    base_url: str
    max_concurrency: int


def parse_endpoint_pool(payload: dict[str, Any]) -> tuple[EndpointPoolEntry, ...]:
    raw = payload.get("endpoint_pool")
    if raw is None:
        return ()
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("endpoint_pool must contain at least two endpoint objects")
    entries = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"base_url", "max_concurrency"}:
            raise ValueError(
                "Each endpoint_pool item requires only base_url and max_concurrency"
            )
        base_url = str(item.get("base_url") or "").strip()
        max_concurrency = int(item.get("max_concurrency", 0))
        if not base_url or max_concurrency <= 0:
            raise ValueError("endpoint_pool entries require a URL and positive concurrency")
        if base_url in seen:
            raise ValueError(f"endpoint_pool contains a duplicate URL: {base_url}")
        seen.add(base_url)
        entries.append(
            EndpointPoolEntry(
                base_url=base_url,
                max_concurrency=max_concurrency,
            )
        )
    return tuple(entries)


class EndpointPoolRuntime:
    """Shared endpoint selection, limits, and accounting for one process run."""

    def __init__(self, entries: tuple[EndpointPoolEntry, ...]):
        if not entries:
            raise ValueError("EndpointPoolRuntime requires at least one endpoint")
        self.entries = entries
        self.semaphores = [
            asyncio.Semaphore(item.max_concurrency) for item in entries
        ]
        self._lock = asyncio.Lock()
        self._next_index = 0
        self._attempts = [0 for _ in entries]
        self._successes = [0 for _ in entries]
        self._failures = [0 for _ in entries]

    async def select_index(self) -> int:
        async with self._lock:
            index = self._next_index
            self._next_index = (self._next_index + 1) % len(self.entries)
            self._attempts[index] += 1
            return index

    async def record_result(self, index: int, *, succeeded: bool) -> None:
        async with self._lock:
            target = self._successes if succeeded else self._failures
            target[index] += 1

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "total_attempts": sum(self._attempts),
                "total_successes": sum(self._successes),
                "total_failures": sum(self._failures),
                "endpoints": [
                    {
                        "base_url": entry.base_url,
                        "max_concurrency": entry.max_concurrency,
                        "attempts": self._attempts[index],
                        "successes": self._successes[index],
                        "failures": self._failures[index],
                    }
                    for index, entry in enumerate(self.entries)
                ],
            }


def build_endpoint_pool_runtime(
    payload: dict[str, Any],
) -> EndpointPoolRuntime | None:
    entries = parse_endpoint_pool(payload)
    return EndpointPoolRuntime(entries) if entries else None


class _PooledClient:
    def __init__(
        self,
        *,
        clients: list[Any],
        entries: tuple[EndpointPoolEntry, ...],
        max_transport_attempts: int,
        runtime: EndpointPoolRuntime | None = None,
    ):
        if len(clients) != len(entries) or not clients:
            raise ValueError("Pooled client requires one client per endpoint")
        self._clients = clients
        self._entries = entries
        if runtime is not None and runtime.entries != entries:
            raise ValueError("Shared endpoint runtime does not match client endpoints")
        self.endpoint_runtime = runtime or EndpointPoolRuntime(entries)
        self._max_transport_attempts = max(1, max_transport_attempts)

    async def _call(
        self,
        operation: Callable[[Any], Awaitable[TCall]],
        *,
        stage: str,
    ) -> TCall:
        errors = []
        endpoint_attempts = []
        for pool_attempt in range(1, self._max_transport_attempts + 1):
            index = await self.endpoint_runtime.select_index()
            entry = self._entries[index]
            endpoint_attempts.append(entry.base_url)
            try:
                async with self.endpoint_runtime.semaphores[index]:
                    result = await operation(self._clients[index])
                await self.endpoint_runtime.record_result(index, succeeded=True)
                metadata = {
                    **result.metadata,
                    "endpoint_pool_size": len(self._entries),
                    "endpoint_pool_attempt": pool_attempt,
                    "endpoint_attempts": endpoint_attempts,
                    "endpoint_max_concurrency": entry.max_concurrency,
                }
                if isinstance(result, JsonCall):
                    return JsonCall(data=result.data, metadata=metadata)  # type: ignore[return-value]
                return TextCall(text=result.text, metadata=metadata)  # type: ignore[return-value]
            except ModelTransportError as exc:
                await self.endpoint_runtime.record_result(index, succeeded=False)
                errors.append(f"{entry.base_url}: {exc}")
                if pool_attempt < self._max_transport_attempts:
                    await asyncio.sleep(min(2.0, 0.25 * pool_attempt))
        raise ModelTransportError(
            f"{stage} failed across endpoint pool after "
            f"{self._max_transport_attempts} attempts: {' | '.join(errors)}"
        )


class PooledOpenAIJsonClient(_PooledClient):
    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
    ) -> JsonCall:
        return await self._call(
            lambda client: client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage=stage,
            ),
            stage=stage,
        )


class PooledOpenAITextClient(_PooledClient):
    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
    ) -> TextCall:
        return await self._call(
            lambda client: client.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage=stage,
            ),
            stage=stage,
        )


class OpenAIJsonClient:
    """Strict JSON client for OpenAI-compatible chat-completion servers."""

    def __init__(self, config: ModelConfig):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError("Install the project dependencies before running the pipeline") from exc
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
    ) -> JsonCall:
        last_error: Exception | None = None
        response: Any | None = None
        elapsed_seconds = 0.0
        transport_attempt = 0
        for transport_attempt in range(1, self.config.max_transport_attempts + 1):
            started = time.monotonic()
            try:
                kwargs: dict[str, Any] = {}
                if self.config.json_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.config.max_tokens is not None:
                    kwargs["max_tokens"] = self.config.max_tokens
                if self.config.extra_body:
                    kwargs["extra_body"] = self.config.extra_body
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                )
                elapsed_seconds = time.monotonic() - started
                break
            except Exception as exc:
                if _is_content_filter_error(exc):
                    raise ModelContentFilterError(
                        f"{stage} was rejected by the provider content filter: {exc}"
                    ) from exc
                last_error = exc
                if transport_attempt < self.config.max_transport_attempts:
                    await asyncio.sleep(
                        min(8.0, 0.75 * (2 ** (transport_attempt - 1)))
                    )
        if response is None:
            raise ModelTransportError(
                f"{stage} transport failed after "
                f"{self.config.max_transport_attempts} attempts: {last_error}"
            ) from last_error

        # A received response is one semantic call. Parsing failures are surfaced to the
        # owning stage, which may schedule one explicit, auditable targeted repair.
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        usage_metadata = _openai_usage_metadata(usage)
        metadata = {
            "stage": stage,
            "model": str(getattr(response, "model", None) or self.config.model),
            "requested_model": self.config.model,
            "api_style": "openai",
            "base_url": self.config.base_url,
            "transport_attempt": transport_attempt,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "response_id": getattr(response, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            **usage_metadata,
        }
        try:
            data, repaired = _parse_json_object(text)
        except (json.JSONDecodeError, ValueError) as exc:
            parse_metadata = {
                **metadata,
                "response_parse_failed": True,
                "response_parse_error": str(exc),
                "raw_response_chars": len(text),
                "raw_response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            raise ModelResponseParseError(
                f"{stage} returned invalid JSON: {exc}",
                raw_text=text,
                metadata=parse_metadata,
            ) from exc
        if repaired:
            metadata = {
                **metadata,
                "deterministic_json_repair": True,
                "deterministic_json_repair_method": "json_repair",
            }
        return JsonCall(data=data, metadata=metadata)


class AnthropicJsonClient:
    """Strict JSON client for an Anthropic-compatible single-turn messages endpoint."""

    def __init__(self, config: ModelConfig):
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError("Install the project dependencies before running the pipeline") from exc
        if config.max_tokens is None:
            raise ValueError("Anthropic JSON client requires max_tokens")
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            timeout=config.timeout_seconds,
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
    ) -> JsonCall:
        last_error: Exception | None = None
        response: Any | None = None
        elapsed_seconds = 0.0
        transport_attempt = 0
        request = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            **(self.config.extra_body or {}),
        }
        for transport_attempt in range(1, self.config.max_transport_attempts + 1):
            started = time.monotonic()
            try:
                response = await self._client.post("messages", json=request)
                response.raise_for_status()
                elapsed_seconds = time.monotonic() - started
                break
            except Exception as exc:
                if _is_content_filter_error(exc):
                    raise ModelContentFilterError(
                        f"{stage} was rejected by the provider content filter: {exc}"
                    ) from exc
                last_error = exc
                response = None
                if transport_attempt < self.config.max_transport_attempts:
                    await asyncio.sleep(min(8.0, 0.75 * (2 ** (transport_attempt - 1))))
        if response is None:
            raise ModelTransportError(
                f"{stage} transport failed after "
                f"{self.config.max_transport_attempts} attempts: {last_error}"
            ) from last_error
        try:
            payload = response.json()
        except Exception as exc:
            raise ModelTransportError(f"{stage} returned a non-JSON Anthropic response") from exc
        blocks = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(blocks, list):
            raise ModelTransportError(f"{stage} returned malformed Anthropic content")
        text = "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        thinking_blocks = [
            block
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "thinking"
        ]
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        metadata = {
            "stage": stage,
            "model": str(payload.get("model") or self.config.model),
            "api_style": "anthropic",
            "base_url": self.config.base_url,
            "transport_attempt": transport_attempt,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "response_id": payload.get("id"),
            "finish_reason": payload.get("stop_reason"),
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "thinking_blocks": len(thinking_blocks),
            "thinking_chars": sum(
                len(str(block.get("thinking") or "")) for block in thinking_blocks
            ),
        }
        try:
            data, repaired = _parse_json_object(text)
        except (json.JSONDecodeError, ValueError) as exc:
            parse_metadata = {
                **metadata,
                "response_parse_failed": True,
                "response_parse_error": str(exc),
                "raw_response_chars": len(text),
                "raw_response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            raise ModelResponseParseError(
                f"{stage} returned invalid JSON: {exc}",
                raw_text=text,
                metadata=parse_metadata,
            ) from exc
        if repaired:
            metadata = {
                **metadata,
                "deterministic_json_repair": True,
                "deterministic_json_repair_method": "json_repair",
            }
        return JsonCall(data=data, metadata=metadata)


class OpenAITextClient:
    """Plain-text client for actor responses that must not be forced into JSON."""

    def __init__(self, config: ModelConfig):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError("Install the project dependencies before running the pipeline") from exc
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
    ) -> TextCall:
        last_error: Exception | None = None
        response: Any | None = None
        elapsed_seconds = 0.0
        transport_attempt = 0
        for transport_attempt in range(1, self.config.max_transport_attempts + 1):
            started = time.monotonic()
            try:
                kwargs: dict[str, Any] = {}
                if self.config.max_tokens is not None:
                    kwargs["max_tokens"] = self.config.max_tokens
                if self.config.extra_body:
                    kwargs["extra_body"] = self.config.extra_body
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                )
                elapsed_seconds = time.monotonic() - started
                break
            except Exception as exc:
                if _is_content_filter_error(exc):
                    raise ModelContentFilterError(
                        f"{stage} was rejected by the provider content filter: {exc}"
                    ) from exc
                last_error = exc
                if transport_attempt < self.config.max_transport_attempts:
                    await asyncio.sleep(
                        min(8.0, 0.75 * (2 ** (transport_attempt - 1)))
                    )
        if response is None:
            raise ModelTransportError(
                f"{stage} transport failed after "
                f"{self.config.max_transport_attempts} attempts: {last_error}"
            ) from last_error
        choice = response.choices[0]
        text = str(choice.message.content or "").strip()
        usage = getattr(response, "usage", None)
        usage_metadata = _openai_usage_metadata(usage)
        metadata = {
            "stage": stage,
            "model": str(getattr(response, "model", None) or self.config.model),
            "requested_model": self.config.model,
            "api_style": "openai",
            "base_url": self.config.base_url,
            "transport_attempt": transport_attempt,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "response_id": getattr(response, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            **usage_metadata,
        }
        if not text:
            raise ModelTransportError(f"{stage} returned an empty text response")
        return TextCall(text=text, metadata=metadata)


def _openai_usage_metadata(usage: Any) -> dict[str, Any]:
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    raw: dict[str, Any] = {}
    if usage is not None and callable(getattr(usage, "model_dump", None)):
        candidate = usage.model_dump()
        if isinstance(candidate, dict):
            raw = candidate
    if reasoning_tokens is None:
        raw_details = raw.get("completion_tokens_details")
        if isinstance(raw_details, dict):
            reasoning_tokens = raw_details.get("reasoning_tokens")
    provider_thought_tokens = None
    billing = raw.get("billing_usage")
    if isinstance(billing, dict):
        gemini = billing.get("gemini_usage_metadata")
        if isinstance(gemini, dict):
            provider_thought_tokens = gemini.get("thoughtsTokenCount")
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": reasoning_tokens,
        "provider_thought_tokens": provider_thought_tokens,
    }


def _is_content_filter_error(exc: Exception) -> bool:
    return "content_filter" in str(exc).casefold()


def build_json_client(
    payload: dict[str, Any],
    *,
    endpoint_runtime: EndpointPoolRuntime | None = None,
) -> OpenAIJsonClient | AnthropicJsonClient | PooledOpenAIJsonClient:
    config = ModelConfig.from_dict(payload)
    entries = parse_endpoint_pool(payload)
    if not entries:
        return (
            AnthropicJsonClient(config)
            if config.api_style == "anthropic"
            else OpenAIJsonClient(config)
        )
    client_class = AnthropicJsonClient if config.api_style == "anthropic" else OpenAIJsonClient
    clients = [
        client_class(
            replace(
                config,
                base_url=entry.base_url,
                max_transport_attempts=1,
            )
        )
        for entry in entries
    ]
    pooled = PooledOpenAIJsonClient(
        clients=clients,
        entries=entries,
        max_transport_attempts=config.max_transport_attempts,
        runtime=endpoint_runtime,
    )
    pooled.config = config
    return pooled


def build_text_client(
    payload: dict[str, Any],
    *,
    endpoint_runtime: EndpointPoolRuntime | None = None,
) -> OpenAITextClient | PooledOpenAITextClient:
    config = ModelConfig.from_dict(payload)
    entries = parse_endpoint_pool(payload)
    if not entries:
        return OpenAITextClient(config)
    clients = [
        OpenAITextClient(
            replace(
                config,
                base_url=entry.base_url,
                max_transport_attempts=1,
            )
        )
        for entry in entries
    ]
    pooled = PooledOpenAITextClient(
        clients=clients,
        entries=entries,
        max_transport_attempts=config.max_transport_attempts,
        runtime=endpoint_runtime,
    )
    pooled.config = config
    return pooled


class OpenAIEmbeddingClient:
    def __init__(self, config: ModelConfig):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError("Install the project dependencies before running the pipeline") from exc
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def embed(self, texts: list[str], *, stage: str) -> tuple[list[list[float]], dict[str, Any]]:
        if not texts:
            return [], {"stage": stage, "count": 0}
        last_error: Exception | None = None
        response: Any | None = None
        elapsed_seconds = 0.0
        transport_attempt = 0
        for transport_attempt in range(1, self.config.max_transport_attempts + 1):
            started = time.monotonic()
            try:
                response = await self._client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                )
                elapsed_seconds = time.monotonic() - started
                break
            except Exception as exc:
                last_error = exc
                if transport_attempt < self.config.max_transport_attempts:
                    await asyncio.sleep(
                        min(8.0, 0.75 * (2 ** (transport_attempt - 1)))
                    )
        if response is None:
            raise ModelTransportError(
                f"{stage} transport failed after "
                f"{self.config.max_transport_attempts} attempts: {last_error}"
            ) from last_error
        vectors = [
            item.embedding for item in sorted(response.data, key=lambda item: item.index)
        ]
        if len(vectors) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, received {len(vectors)}")
        usage = getattr(response, "usage", None)
        return vectors, {
            "stage": stage,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "transport_attempt": transport_attempt,
            "count": len(vectors),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "elapsed_seconds": round(elapsed_seconds, 3),
        }


def _strict_json_object(text: str) -> dict[str, Any]:
    payload, _ = _parse_json_object(text)
    return payload


def _parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    """Parse model JSON, allowing one deterministic syntax-only repair.

    The repair is intentionally applied only after the standard parser rejects
    the response. It does not add, remove, or infer narrative content; the
    owning stage still performs its normal schema and grounding validation.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
        repaired = False
    except (json.JSONDecodeError, ValueError) as original_error:
        try:
            from json_repair import repair_json

            repaired_text = repair_json(stripped)
            payload = json.loads(repaired_text)
            repaired = True
        except Exception:
            raise original_error
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object")
    return payload, repaired
