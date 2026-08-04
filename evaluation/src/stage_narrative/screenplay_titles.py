from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


_REDUNDANT_SOURCE_SCENE_NUMBER = re.compile(
    r"^(?P<internal>[0-9]+)、\s*"
    r"(?P<source>[0-9]+(?:[A-Za-z])?)(?:[.．])?\s*"
    r"(?P<body>\S.*)$"
)


def normalize_public_scene_title(record: dict[str, Any]) -> tuple[str, str | None]:
    """Remove a source scene number duplicated after the public internal ID.

    STAGE public titles use ``<record _id>、<scene heading>``. Some source
    headings already began with their own scene number, producing titles such
    as ``4、5 MANOR EXT/NIGHT``. A title is changed only when its first number
    equals the record's ``_id`` and non-empty heading text remains.
    """

    title = str(record.get("title") or "")
    raw_id = str(record.get("_id", "")).strip()
    if not raw_id.isdigit():
        return title, None
    removed: list[str] = []
    current = title
    while True:
        match = _REDUNDANT_SOURCE_SCENE_NUMBER.match(current)
        if match is None or int(match.group("internal")) != int(raw_id):
            break
        body = match.group("body").strip()
        # A numeric-only title such as ``117、106`` can backtrack to source
        # ``10`` plus body ``6``. A one-character alphanumeric remainder is
        # therefore too ambiguous to normalize.
        if not body or (len(body) == 1 and body.isascii() and body.isalnum()):
            break
        removed.append(match.group("source"))
        current = f"{match.group('internal')}、{body}"
    return current, "+".join(removed) if removed else None


def normalize_public_scene_titles(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a normalized copy and an audit row for every changed scene."""

    normalized = deepcopy(records)
    changes: list[dict[str, Any]] = []
    for record in normalized:
        before = str(record.get("title") or "")
        after, source_scene_number = normalize_public_scene_title(record)
        if source_scene_number is None:
            continue
        record["title"] = after
        changes.append(
            {
                "scene_id": record.get("_id"),
                "source_scene_number": source_scene_number,
                "before": before,
                "after": after,
            }
        )
    return normalized, changes


def redundant_public_scene_titles(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return redundant-number findings without changing the input records."""

    _, changes = normalize_public_scene_titles(records)
    return changes
