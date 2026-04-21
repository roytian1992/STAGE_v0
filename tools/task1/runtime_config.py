#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian")
DEFAULT_RUNTIME_CONFIG_PATH = ROOT / "task1_v1_pilot" / "configs" / "task1_runtime.yaml"
RUNTIME_CONFIG_ENV = "STAGE_TASK1_RUNTIME_CONFIG"
RUNTIME_PROFILE_ENV = "STAGE_TASK1_RUNTIME_PROFILE"


def resolve_runtime_config_path(config_path: str | Path | None = None) -> Path:
    raw = config_path or os.getenv(RUNTIME_CONFIG_ENV) or DEFAULT_RUNTIME_CONFIG_PATH
    return Path(raw).expanduser().resolve()


def resolve_runtime_profile(profile: str | None = None) -> str:
    value = (profile or os.getenv(RUNTIME_PROFILE_ENV) or "default").strip()
    return value or "default"


def export_runtime_env(config_path: str | Path | None = None, profile: str | None = None) -> tuple[Path, str]:
    path = resolve_runtime_config_path(config_path)
    profile_name = resolve_runtime_profile(profile)
    os.environ[RUNTIME_CONFIG_ENV] = str(path)
    os.environ[RUNTIME_PROFILE_ENV] = profile_name
    return path, profile_name


def _expect_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _string_field(data: Dict[str, Any], key: str, context: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"{context}.{key} must be a scalar string-like value")
    return str(value).strip()


def _int_field(data: Dict[str, Any], key: str, context: str, default: int) -> int:
    value = data.get(key, default)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{context}.{key} must be an integer") from exc


def _load_endpoint(data: Dict[str, Any], name: str, *, require_timeout: bool) -> Dict[str, Any]:
    endpoint = _expect_mapping(data.get(name), name)
    out = {
        "model": _string_field(endpoint, "model", name),
        "base_url": _string_field(endpoint, "base_url", name),
        "api_key": _string_field(endpoint, "api_key", name),
    }
    if require_timeout:
        out["timeout_sec"] = _int_field(endpoint, "timeout_sec", name, 0)
        out["transport_retries"] = _int_field(endpoint, "transport_retries", name, 0)
        retry_delay_raw = endpoint.get("retry_delay_sec", 0.0)
        try:
            out["retry_delay_sec"] = float(retry_delay_raw or 0.0)
        except Exception as exc:
            raise ValueError(f"{name}.retry_delay_sec must be numeric") from exc
    fallback_data = endpoint.get("fallback")
    if fallback_data is not None:
        fallback = _expect_mapping(fallback_data, f"{name}.fallback")
        out["fallback"] = {
            "model": _string_field(fallback, "model", f"{name}.fallback"),
            "base_url": _string_field(fallback, "base_url", f"{name}.fallback"),
            "api_key": _string_field(fallback, "api_key", f"{name}.fallback"),
        }
    return out


def load_runtime_profile(config_path: str | Path | None = None, profile: str | None = None) -> Dict[str, Any]:
    path = resolve_runtime_config_path(config_path)
    profile_name = resolve_runtime_profile(profile)
    if not path.exists():
        raise FileNotFoundError(f"Task1 runtime config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = _expect_mapping(raw, "runtime_config")
    profiles = _expect_mapping(data.get("profiles"), "profiles")
    profile_data = _expect_mapping(profiles.get(profile_name), f"profiles.{profile_name}")

    llm = _load_endpoint(profile_data, "llm", require_timeout=True)
    embed = _load_endpoint(profile_data, "embed", require_timeout=True)
    mimo_raw = profile_data.get("mimo") or {}
    mimo = _expect_mapping(mimo_raw, "mimo") if mimo_raw else {}
    mimo_out = {
        "model": _string_field(mimo, "model", "mimo"),
        "base_url": _string_field(mimo, "base_url", "mimo"),
        "api_key": _string_field(mimo, "api_key", "mimo"),
    }

    if not llm["model"] or not llm["base_url"]:
        raise ValueError(f"profiles.{profile_name}.llm.model/base_url must be set")
    if not embed["model"] or not embed["base_url"]:
        raise ValueError(f"profiles.{profile_name}.embed.model/base_url must be set")

    return {
        "config_path": str(path),
        "profile": profile_name,
        "llm": llm,
        "embed": embed,
        "mimo": mimo_out,
    }
