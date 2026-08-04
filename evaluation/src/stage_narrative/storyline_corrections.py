from __future__ import annotations

from typing import Any

from .hierarchy import (
    _storyline_component_input_sha256,
    _storyline_component_prompt_assets,
    _validate_storyline_component_payload,
    build_storyline_candidates,
)
from .models import clean_text


def build_storyline_component_corrections(
    *,
    episodes: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    language: str,
    patch: dict[str, Any],
) -> list[dict[str, Any]]:
    if patch.get("schema_version") != "stage_storyline_component_correction_patch_v1":
        raise ValueError("Unsupported Storyline component correction patch schema")
    episode_by_id = {item["episode_id"]: item for item in episodes}
    relation_by_id = {item["relation_id"]: item for item in relations}
    candidates = build_storyline_candidates(episodes, relations)
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for decision in patch.get("decisions", []):
        component_index = int(decision.get("component_index", 0))
        if component_index < 1 or component_index > len(candidates):
            raise ValueError(f"Unknown Storyline component index: {component_index}")
        if component_index in seen:
            raise ValueError(f"Duplicate Storyline component correction: {component_index}")
        seen.add(component_index)
        candidate = candidates[component_index - 1]
        expected_component_id = clean_text(decision.get("component_id"))
        if expected_component_id != candidate["component_id"]:
            raise ValueError(
                f"Storyline component identity mismatch: {expected_component_id} != "
                f"{candidate['component_id']}"
            )
        assets = _storyline_component_prompt_assets(
            candidate,
            episode_by_id=episode_by_id,
            relation_by_id=relation_by_id,
        )
        storylines = _validate_storyline_component_payload(
            {"storylines": decision.get("storylines")},
            episode_by_local_id=assets["episode_by_local_id"],
            relation_by_local_id=assets["relation_by_local_id"],
            entity_by_local_id=assets["entity_by_local_id"],
        )
        output.append(
            {
                "component_index": component_index,
                "component_id": candidate["component_id"],
                "input_sha256": _storyline_component_input_sha256(
                    language=language,
                    candidate=candidate,
                    assets=assets,
                ),
                "result": storylines,
                "call_metadata": {
                    "stage": f"storyline_agent_correction:{component_index:04d}",
                    "call_kind": "agent_component_correction",
                    "reviewer": patch.get("reviewer"),
                    "decision_id": decision.get("decision_id"),
                    "reason": decision.get("reason"),
                    "evidence": decision.get("evidence", []),
                    "model_calls_added": 0,
                },
            }
        )
    return output
