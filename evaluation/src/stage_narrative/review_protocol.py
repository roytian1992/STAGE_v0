from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .io import atomic_write_json, load_json, sha256_json


CHECKPOINT_SCHEMA = "stage_semantic_review_checkpoint_v1"


def model_identity(client: Any) -> dict[str, Any]:
    config = getattr(client, "config", None)
    if is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = dict(config)
    else:
        return {"client_class": type(client).__name__}
    payload.pop("api_key", None)
    return {
        key: payload.get(key)
        for key in (
            "model",
            "base_url",
            "temperature",
            "json_response_format",
            "max_tokens",
            "extra_body",
        )
        if key in payload
    }


def review_cache_key(
    *,
    namespace: str,
    input_payload: Any,
    system_prompt: str,
    user_prompt: str,
    output_schema: Any,
    client: Any,
) -> tuple[str, dict[str, str]]:
    hashes = {
        "input_sha256": sha256_json(input_payload),
        "prompt_sha256": sha256_json(
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        ),
        "output_schema_sha256": sha256_json(output_schema),
        "model_config_sha256": sha256_json(model_identity(client)),
    }
    return sha256_json({"namespace": namespace, **hashes}), hashes


class ImmutableReviewCache:
    def __init__(self, root: Path | None):
        self.root = root.resolve() if root is not None else None

    def path_for(self, namespace: str, cache_key: str) -> Path | None:
        if self.root is None:
            return None
        safe_namespace = namespace.replace(":", "_").replace("/", "_")
        return self.root / safe_namespace / f"{cache_key}.json"

    def load(
        self,
        *,
        namespace: str,
        cache_key: str,
        hashes: dict[str, str],
    ) -> dict[str, Any] | None:
        path = self.path_for(namespace, cache_key)
        if path is None or not path.is_file():
            return None
        payload = load_json(path)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported semantic checkpoint schema: {path}")
        if payload.get("cache_key") != cache_key:
            raise ValueError(f"Semantic checkpoint key mismatch: {path}")
        for key, value in hashes.items():
            if payload.get(key) != value:
                raise ValueError(f"Semantic checkpoint {key} mismatch: {path}")
        return payload

    def commit(
        self,
        *,
        namespace: str,
        cache_key: str,
        hashes: dict[str, str],
        call_kind: str,
        status: str,
        result: Any,
        call_metadata: list[dict[str, Any]],
        validation_error: str = "",
    ) -> Path | None:
        path = self.path_for(namespace, cache_key)
        if path is None:
            return None
        if path.exists():
            raise FileExistsError(
                f"Immutable semantic checkpoint already exists: {path}"
            )
        atomic_write_json(
            path,
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "namespace": namespace,
                "cache_key": cache_key,
                **hashes,
                "call_kind": call_kind,
                "status": status,
                "result": result,
                "call_metadata": call_metadata,
                "validation_error": validation_error,
            },
        )
        return path


def budget_record(
    *,
    namespace: str,
    cache_key: str,
    hashes: dict[str, str],
    cache: ImmutableReviewCache,
    item_count: int,
    call_kind: str,
) -> dict[str, Any]:
    checkpoint = cache.load(
        namespace=namespace,
        cache_key=cache_key,
        hashes=hashes,
    )
    return {
        "namespace": namespace,
        "cache_key": cache_key,
        **hashes,
        "item_count": item_count,
        "call_kind": call_kind,
        "cache_hit": checkpoint is not None,
        "checkpoint_path": str(cache.path_for(namespace, cache_key))
        if cache.path_for(namespace, cache_key)
        else None,
        "checkpoint_status": checkpoint.get("status") if checkpoint else None,
    }
