from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .models import Scene, clean_text


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def unwrap_scene_records(payload: Any, *, source: str = "scene payload") -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("scenes"), list):
        records = payload["scenes"]
    else:
        raise ValueError(f"Expected a scene array or object with scenes array: {source}")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Scene records must be objects: {source}")
    return records


def load_scenes(path: Path) -> list[Scene]:
    payload = load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Screenplay must be a non-empty JSON array: {path}")
    scenes: list[Scene] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Scene {index} is not an object in {path}")
        raw_id = item.get("_id", index)
        source_scene_id = clean_text(raw_id) or str(index)
        scene_id = f"S{index:04d}:{source_scene_id}"
        content = str(item.get("content") or "").strip()
        title = clean_text(item.get("title"))
        subtitle = clean_text(item.get("subtitle"))
        if not any((title, subtitle, content)):
            raise ValueError(f"Scene {source_scene_id!r} has no textual fields in {path}")
        scenes.append(
            Scene(
                scene_id=scene_id,
                source_scene_id=source_scene_id,
                order=index,
                title=title,
                subtitle=subtitle,
                content=content,
            )
        )
    return scenes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
