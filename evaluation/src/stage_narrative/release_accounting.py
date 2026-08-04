from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def collect_response_call_metadata(payloads: Iterable[Any]) -> list[dict[str, Any]]:
    """Collect persisted model calls once, even when stages copy prior metadata."""
    calls: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for item in _objects(payload):
            response_id = str(item.get("response_id") or "").strip()
            if not response_id or not _looks_like_call_metadata(item):
                continue
            prior = calls.get(response_id)
            if prior is not None:
                keys = ("stage", "model", "prompt_tokens", "completion_tokens")
                if any(prior.get(key) != item.get(key) for key in keys):
                    raise ValueError(
                        f"Conflicting model-call metadata for response_id={response_id}"
                    )
                continue
            calls[response_id] = dict(item)
    return sorted(
        calls.values(),
        key=lambda item: (str(item.get("stage") or ""), str(item["response_id"])),
    )


def _objects(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _looks_like_call_metadata(item: dict[str, Any]) -> bool:
    return (
        bool(str(item.get("stage") or "").strip())
        and isinstance(item.get("prompt_tokens"), int)
        and isinstance(item.get("completion_tokens"), int)
    )


__all__ = ["collect_response_call_metadata"]
