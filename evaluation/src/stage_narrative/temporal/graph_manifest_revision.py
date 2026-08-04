from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def register_graph_manifest_revision(
    manifest: dict[str, Any],
    *,
    graph_run_dir: Path,
    graph_sha256: str,
    graph_manifest_sha256: str,
    revision_id: str,
    reason: str,
    registered_at: str,
) -> dict[str, Any]:
    output = copy.deepcopy(manifest)
    recorded_dir = Path(str(output.get("graph_run_dir", ""))).resolve()
    if recorded_dir != graph_run_dir.resolve():
        raise ValueError("Temporal run is bound to a different graph run directory")
    if output.get("graph_sha256") != graph_sha256:
        raise ValueError("Narrative graph content changed; a metadata revision is unsafe")
    previous_manifest_sha256 = str(output.get("graph_manifest_sha256", ""))
    if not previous_manifest_sha256:
        raise ValueError("Temporal run has no recorded graph manifest checksum")
    if previous_manifest_sha256 == graph_manifest_sha256:
        raise ValueError("Graph manifest checksum is already current")
    if not revision_id.strip() or not reason.strip():
        raise ValueError("Graph manifest revision requires an ID and reason")
    revisions = output.setdefault("graph_manifest_revisions", [])
    if any(item.get("revision_id") == revision_id for item in revisions):
        raise ValueError(f"Duplicate graph manifest revision ID: {revision_id}")
    revisions.append(
        {
            "revision_id": revision_id,
            "registered_at": registered_at,
            "reason": reason,
            "graph_run_dir": str(graph_run_dir.resolve()),
            "graph_sha256": graph_sha256,
            "from_graph_manifest_sha256": previous_manifest_sha256,
            "to_graph_manifest_sha256": graph_manifest_sha256,
            "content_change": False,
        }
    )
    output["graph_manifest_sha256"] = graph_manifest_sha256
    return output
