from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..models import clean_text, unique_text


IDENTITY_PATCH_SCHEMA = "stage_character_identity_lineage_patch_v1"


def apply_character_identity_lineages(
    registry: dict[str, Any],
    *,
    patch: dict[str, Any] | None,
    scene_order_by_id: dict[str, int],
) -> dict[str, Any]:
    if not patch:
        return registry
    if patch.get("schema_version") != IDENTITY_PATCH_SCHEMA:
        raise ValueError("Unsupported character identity-lineage patch schema")
    if clean_text(patch.get("movie_id")) != clean_text(registry.get("movie_id")):
        raise ValueError("Character identity-lineage patch movie_id mismatch")

    output = deepcopy(registry)
    rows = {item["character_id"]: item for item in output["characters"]}
    claimed: dict[tuple[str, int], str] = {}
    decisions = []
    for lineage in patch.get("lineages", []):
        tracked_id = clean_text(lineage.get("tracked_character_id"))
        if tracked_id not in rows:
            raise ValueError(f"Identity lineage has unknown tracked character: {tracked_id}")
        segments = _normalize_segments(lineage.get("identity_segments"), rows)
        if not segments or tracked_id not in {
            item["source_character_id"] for item in segments
        }:
            raise ValueError("Identity lineage must include its tracked character")
        for segment in segments:
            for scene_order in range(
                segment["valid_from_scene"], segment["valid_until_scene"] + 1
            ):
                key = (segment["source_character_id"], scene_order)
                if key in claimed:
                    raise ValueError(
                        "Identity lineage source occurrence is claimed more than once: "
                        f"{key[0]} / scene {key[1]}"
                    )
                claimed[key] = tracked_id

        phases = _normalize_phases(lineage.get("identity_phases"), segments)
        source_ids = unique_text(item["source_character_id"] for item in segments)
        scene_ids = sorted(
            {
                scene_id
                for segment in segments
                for scene_id in rows[segment["source_character_id"]].get("scene_ids", [])
                if segment["valid_from_scene"]
                <= scene_order_by_id.get(scene_id, 0)
                <= segment["valid_until_scene"]
            },
            key=scene_order_by_id.__getitem__,
        )
        if not scene_ids:
            raise ValueError("Identity lineage contains no source-backed scenes")
        tracked = rows[tracked_id]
        tracked.update(
            {
                "canonical_name": clean_text(lineage.get("canonical_name"))
                or tracked["canonical_name"],
                "aliases": unique_text(
                    [
                        *(lineage.get("aliases") or []),
                        *(value for source_id in source_ids for value in rows[source_id]["aliases"]),
                        *(phase["name"] for phase in phases),
                    ]
                ),
                "identity_phases": phases,
                "identity_segments": segments,
                "first_scene_order": scene_order_by_id[scene_ids[0]],
                "last_scene_order": scene_order_by_id[scene_ids[-1]],
                "scene_ids": scene_ids,
                "source_entity_ids": source_ids,
                "identity_lineage_revision_id": clean_text(patch.get("revision_id")),
            }
        )
        decisions.append(
            {
                "tracked_character_id": tracked_id,
                "source_character_ids": source_ids,
                "scene_orders": [scene_order_by_id[value] for value in scene_ids],
                "identity_phase_count": len(phases),
            }
        )

    for source_id, row in list(rows.items()):
        if source_id in {item["tracked_character_id"] for item in decisions}:
            continue
        retained = [
            scene_id
            for scene_id in row.get("scene_ids", [])
            if (source_id, scene_order_by_id[scene_id]) not in claimed
        ]
        row["scene_ids"] = retained
        row["first_scene_order"] = scene_order_by_id[retained[0]] if retained else 0
        row["last_scene_order"] = scene_order_by_id[retained[-1]] if retained else 0

    output["characters"] = sorted(
        (item for item in rows.values() if item.get("scene_ids")),
        key=lambda item: (item["first_scene_order"], item["canonical_name"], item["character_id"]),
    )
    output.setdefault("audit", {})["identity_lineage_projection"] = {
        "revision_id": patch.get("revision_id"),
        "reviewer": patch.get("reviewer"),
        "lineage_count": len(decisions),
        "decisions": decisions,
        "model_calls_added": 0,
    }
    return output


def project_evidence_identity_lineages(
    evidence_bank: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    mapping: dict[tuple[str, int], str] = {}
    for character in registry.get("characters", []):
        for segment in character.get("identity_segments", []):
            for scene_order in range(
                int(segment["valid_from_scene"]), int(segment["valid_until_scene"]) + 1
            ):
                mapping[(segment["source_character_id"], scene_order)] = character[
                    "character_id"
                ]
    if not mapping:
        return evidence_bank

    output = deepcopy(evidence_bank)
    changed_fields = 0
    for item in output.get("evidence_units", []):
        scene_order = int(item["scene_order"])
        speaker = clean_text(item.get("speaker_character_id"))
        projected_speaker = mapping.get((speaker, scene_order), speaker)
        if projected_speaker != speaker:
            item["speaker_character_id"] = projected_speaker
            changed_fields += 1
        for key in (
            "participant_character_ids",
            "direct_observer_character_ids",
            "addressee_character_ids",
        ):
            original = list(item.get(key, []))
            projected = unique_text(
                mapping.get((clean_text(value), scene_order), clean_text(value))
                for value in original
            )
            if projected != original:
                item[key] = projected
                changed_fields += 1
    output.setdefault("audit", {})["identity_lineage_projection"] = {
        "mapping_occurrence_count": len(mapping),
        "changed_role_field_count": changed_fields,
        "model_calls_added": 0,
    }
    return output


def source_character_ids_at_scene(
    character: dict[str, Any], scene_order: int
) -> set[str]:
    segments = character.get("identity_segments", [])
    if not segments:
        return {character["character_id"]}
    return {
        segment["source_character_id"]
        for segment in segments
        if int(segment["valid_from_scene"])
        <= scene_order
        <= int(segment["valid_until_scene"])
    }


def character_name_at_scene(character: dict[str, Any], scene_order: int) -> str:
    for phase in character.get("identity_phases", []):
        if int(phase["valid_from_scene"]) <= scene_order <= int(
            phase["valid_until_scene"]
        ):
            return phase["name"]
    return character["canonical_name"]


def _normalize_segments(value: Any, rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("identity_segments must be an array")
    output = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("identity segment must be an object")
        source_id = clean_text(item.get("source_character_id"))
        start = int(item.get("valid_from_scene", 0))
        end = int(item.get("valid_until_scene", 0))
        if source_id not in rows or start <= 0 or end < start:
            raise ValueError("Invalid identity segment")
        output.append(
            {
                "source_character_id": source_id,
                "valid_from_scene": start,
                "valid_until_scene": end,
            }
        )
    return sorted(
        output,
        key=lambda item: (
            item["valid_from_scene"],
            item["valid_until_scene"],
            item["source_character_id"],
        ),
    )


def _normalize_phases(
    value: Any, segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("identity_phases must be a non-empty array")
    output = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("identity phase must be an object")
        name = clean_text(item.get("name"))
        start = int(item.get("valid_from_scene", 0))
        end = int(item.get("valid_until_scene", 0))
        if not name or start <= 0 or end < start:
            raise ValueError("Invalid identity phase")
        if not any(
            segment["valid_from_scene"] <= start <= end <= segment["valid_until_scene"]
            for segment in segments
        ):
            raise ValueError("Identity phase is outside its source segment")
        output.append(
            {
                "phase_id": clean_text(item.get("phase_id")) or f"phase-{len(output) + 1}",
                "name": name,
                "aliases": unique_text(item.get("aliases") or []),
                "valid_from_scene": start,
                "valid_until_scene": end,
            }
        )
    ordered = sorted(output, key=lambda item: (item["valid_from_scene"], item["phase_id"]))
    if any(
        left["valid_until_scene"] >= right["valid_from_scene"]
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("Identity phases overlap")
    return ordered


__all__ = [
    "IDENTITY_PATCH_SCHEMA",
    "apply_character_identity_lineages",
    "character_name_at_scene",
    "project_evidence_identity_lineages",
    "source_character_ids_at_scene",
]
