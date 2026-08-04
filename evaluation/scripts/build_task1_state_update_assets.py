#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.io import (  # noqa: E402
    atomic_write_json,
    load_json,
    load_scenes,
    sha256_file,
)


OUTPUT_SCHEMA = {
    "current_state": {
        "type": "array",
        "item_fields": ["claim", "evidence_scene_orders"],
    },
    "developments_since_previous_checkpoint": {
        "type": "array",
        "item_fields": ["claim", "evidence_scene_orders"],
    },
}

CONTEXT_PROTOCOL = {
    "screenplay_scope": "previous_checkpoint_exclusive_to_current_checkpoint_inclusive",
    "memory_method": "runtime_entity_centric_chunk_extraction",
    "memory_extractor_inputs": ["character", "aliases", "screenplay_interval"],
    "previous_state_visible_to_memory_extractor": False,
    "memory_shared_between_settings": True,
    "full_screenplay_prefix_used": False,
    "previous_state_model_projection": "claim_text_only",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired reference- and autoregressive Task 1 assets."
    )
    parser.add_argument("--public-instances", type=Path, required=True)
    parser.add_argument("--private-evaluator", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference, autoregressive, audit = build_assets(
        public_instances=load_json(args.public_instances.resolve()),
        private_evaluator=load_json(args.private_evaluator.resolve()),
        script_path=args.script.resolve(),
    )
    if audit["status"] != "passed":
        raise ValueError(f"State update Task 1 asset validation failed: {audit}")
    reference_path = output_dir / "task_1_reference_state_update.json"
    autoregressive_path = output_dir / "task_1_autoregressive_state_update.json"
    atomic_write_json(reference_path, reference)
    atomic_write_json(autoregressive_path, autoregressive)
    manifest = {
        "schema": "stage_task1_state_update_asset_manifest",
        "status": "passed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "movie_id": reference["movie_id"],
        "construction": {
            "method": "deterministic_repackaging_of_reviewed_checkpoint_reference",
            "llm_calls": 0,
            "first_checkpoint_policy": "empty_previous_state_and_script_from_scene_1",
            "context_protocol": CONTEXT_PROTOCOL,
        },
        "inputs": {
            "public_instances": _artifact(args.public_instances.resolve()),
            "private_evaluator": _artifact(args.private_evaluator.resolve()),
            "script": _artifact(args.script.resolve()),
        },
        "outputs": {
            "reference_state_update": _artifact(reference_path),
            "autoregressive_state_update": _artifact(autoregressive_path),
        },
        "counts": reference["counts"],
        "validation": audit,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(output_dir / "manifest.json")
    print(manifest["counts"])


def build_assets(
    *,
    public_instances: dict[str, Any],
    private_evaluator: dict[str, Any],
    script_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    movie_id = str(private_evaluator.get("movie_id") or "")
    if not movie_id or public_instances.get("movie_id") != movie_id:
        raise ValueError("Task 1 source movie identities disagree")
    scenes = load_scenes(script_path)
    scene_count = len(scenes)
    public_by_checkpoint = {
        row["checkpoint_id"]: row for row in public_instances.get("instances", [])
    }
    if len(public_by_checkpoint) != int(public_instances.get("instance_count", -1)):
        raise ValueError("Public Task 1 checkpoint IDs are missing or duplicated")

    reference_trajectories: list[dict[str, Any]] = []
    prediction_trajectories: list[dict[str, Any]] = []
    current_state_claims = 0
    development_claims = 0
    corrected_first_boundaries = 0
    seen_checkpoints: set[str] = set()

    for trajectory in private_evaluator.get("trajectories", []):
        checkpoint_ids = list(trajectory.get("checkpoint_ids", []))
        rubric_by_id = {
            row["checkpoint_id"]: row
            for row in trajectory.get("checkpoint_rubrics", [])
        }
        if set(checkpoint_ids) != set(rubric_by_id):
            raise ValueError(f"Checkpoint rubric coverage mismatch: {trajectory['character']}")
        if not checkpoint_ids:
            raise ValueError(f"Empty Task 1 trajectory: {trajectory['character']}")
        aliases = _aliases(checkpoint_ids, public_by_checkpoint)
        reference_checkpoints: list[dict[str, Any]] = []
        prediction_checkpoints: list[dict[str, Any]] = []
        previous_rubric: dict[str, Any] | None = None
        previous_checkpoint_id: str | None = None
        previous_scene_order = 0

        for sequence_index, checkpoint_id in enumerate(checkpoint_ids, start=1):
            if checkpoint_id in seen_checkpoints:
                raise ValueError(f"Checkpoint appears in multiple trajectories: {checkpoint_id}")
            seen_checkpoints.add(checkpoint_id)
            rubric = rubric_by_id[checkpoint_id]
            public = public_by_checkpoint.get(checkpoint_id)
            if public is None:
                raise ValueError(f"Missing public instance for {checkpoint_id}")
            current_scene_order = int(rubric["checkpoint"]["current_scene_order"])
            if current_scene_order <= previous_scene_order or current_scene_order > scene_count:
                raise ValueError(f"Invalid checkpoint order for {checkpoint_id}")
            if sequence_index == 1 and int(
                rubric["checkpoint"].get("previous_scene_order", 0)
            ) != 0:
                corrected_first_boundaries += 1

            interval = {
                "script_file": "script.json",
                "start_scene_order_exclusive": previous_scene_order,
                "end_scene_order_inclusive": current_scene_order,
            }
            previous_reference_state = (
                []
                if previous_rubric is None
                else [_public_claim(row) for row in previous_rubric["current_state_claims"]]
            )
            target = {
                "current_state": [
                    _reference_claim(row) for row in rubric["current_state_claims"]
                ],
                "developments_since_previous_checkpoint": [
                    _reference_claim(row) for row in rubric["development_claims"]
                ],
            }
            _validate_evidence(
                target=target,
                previous_scene_order=previous_scene_order,
                current_scene_order=current_scene_order,
                checkpoint_id=checkpoint_id,
            )
            common = {
                "sequence_index": sequence_index,
                "instance_id": rubric["instance_id"],
                "checkpoint_id": checkpoint_id,
                "previous_checkpoint_id": previous_checkpoint_id,
                "screenplay_interval": interval,
                "entity_memory": {
                    "memory_slot": f"M{sequence_index}",
                    "source": "shared_runtime_interval_extraction",
                    "extractor_inputs": [
                        "character",
                        "aliases",
                        "screenplay_interval",
                    ],
                    "previous_state_visible_to_extractor": False,
                },
            }
            reference_checkpoints.append(
                {
                    **common,
                    "model_input": {
                        "previous_state": previous_reference_state,
                        "previous_state_source": "reference_state",
                    },
                    "evaluation_target": target,
                }
            )
            prediction_checkpoints.append(
                {
                    **common,
                    "model_input": {
                        "previous_state_source": (
                            "empty"
                            if previous_checkpoint_id is None
                            else "autoregressive_prediction"
                        )
                    },
                    "evaluation_target_ref": {
                        "asset_file": "task_1_reference_state_update.json",
                        "checkpoint_id": checkpoint_id,
                    },
                }
            )
            current_state_claims += len(target["current_state"])
            development_claims += len(
                target["developments_since_previous_checkpoint"]
            )
            previous_rubric = rubric
            previous_checkpoint_id = checkpoint_id
            previous_scene_order = current_scene_order

        trajectory_common = {
            "trajectory_id": f"{movie_id}::{trajectory['character_id']}",
            "character_id": trajectory["character_id"],
            "character": trajectory["character"],
            "aliases": aliases,
        }
        reference_trajectories.append(
            {**trajectory_common, "checkpoints": reference_checkpoints}
        )
        prediction_trajectories.append(
            {**trajectory_common, "initial_state": [], "checkpoints": prediction_checkpoints}
        )

    if seen_checkpoints != set(public_by_checkpoint):
        raise ValueError("Private and public Task 1 checkpoint coverage differs")
    counts = {
        "trajectories": len(reference_trajectories),
        "checkpoints": len(seen_checkpoints),
        "current_state_reference_claims": current_state_claims,
        "development_reference_claims": development_claims,
    }
    shared = {
        "movie_id": movie_id,
        "language": private_evaluator["language"],
        "script": {
            "file": "script.json",
            "sha256": sha256_file(script_path),
            "scene_count": scene_count,
        },
        "task": (
            "Track how a focal character's state changes as the story evolves "
            "between consecutive checkpoints."
        ),
        "context_protocol": CONTEXT_PROTOCOL,
        "output_schema": OUTPUT_SCHEMA,
        "counts": counts,
    }
    reference = {
        "schema": "stage_task1_reference_state_update",
        "setting": "reference_state_update",
        **shared,
        "metrics": [
            "current_state_reference_coverage",
            "development_reference_coverage",
            "current_state_unsupported_rate",
            "development_unsupported_rate",
            "current_state_contradiction_rate",
            "development_contradiction_rate",
        ],
        "trajectories": reference_trajectories,
    }
    autoregressive = {
        "schema": "stage_task1_autoregressive_state_update",
        "setting": "autoregressive_state_update",
        **shared,
        "metrics": [
            "current_state_reference_coverage",
            "development_reference_coverage",
            "current_state_unsupported_rate",
            "development_unsupported_rate",
            "current_state_contradiction_rate",
            "development_contradiction_rate",
            "current_state_reference_coverage_accumulation_gap",
            "development_reference_coverage_accumulation_gap",
        ],
        "counts": {
            "trajectories": counts["trajectories"],
            "checkpoints": counts["checkpoints"],
        },
        "trajectories": prediction_trajectories,
    }
    audit = validate_assets(reference=reference, autoregressive=autoregressive)
    audit["corrected_first_checkpoint_boundaries"] = corrected_first_boundaries
    return reference, autoregressive, audit


def validate_assets(
    *, reference: dict[str, Any], autoregressive: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    reference_by_id: dict[str, dict[str, Any]] = {}
    prediction_by_id: dict[str, dict[str, Any]] = {}
    for trajectory in reference["trajectories"]:
        previous_target = None
        for checkpoint in trajectory["checkpoints"]:
            checkpoint_id = checkpoint["checkpoint_id"]
            reference_by_id[checkpoint_id] = checkpoint
            expected_previous = (
                []
                if previous_target is None
                else [
                    _public_claim(row)
                    for row in previous_target["current_state"]
                ]
            )
            if checkpoint["model_input"]["previous_state"] != expected_previous:
                errors.append(f"{checkpoint_id}: previous reference state drift")
            previous_target = checkpoint["evaluation_target"]
    for trajectory in autoregressive["trajectories"]:
        for checkpoint in trajectory["checkpoints"]:
            prediction_by_id[checkpoint["checkpoint_id"]] = checkpoint
    if set(reference_by_id) != set(prediction_by_id):
        errors.append("State update assets have different checkpoint coverage")
    for checkpoint_id, reference_checkpoint in reference_by_id.items():
        predicted = prediction_by_id.get(checkpoint_id, {})
        for key in (
            "instance_id",
            "previous_checkpoint_id",
            "screenplay_interval",
            "entity_memory",
        ):
            if predicted.get(key) != reference_checkpoint.get(key):
                errors.append(f"{checkpoint_id}: setting input drift in {key}")
        memory = reference_checkpoint.get("entity_memory", {})
        if memory.get("previous_state_visible_to_extractor") is not False:
            errors.append(f"{checkpoint_id}: memory extractor can see previous state")
        if "previous_state" in memory.get("extractor_inputs", []):
            errors.append(f"{checkpoint_id}: previous state appears in extractor inputs")
    if reference.get("context_protocol") != autoregressive.get("context_protocol"):
        errors.append("State update assets use different context protocols")
    protocol = reference.get("context_protocol", {})
    if protocol.get("full_screenplay_prefix_used") is not False:
        errors.append("State update protocol unexpectedly uses the full screenplay prefix")
    if protocol.get("memory_shared_between_settings") is not True:
        errors.append("Entity-centric memory is not shared between settings")
    if protocol.get("previous_state_model_projection") != "claim_text_only":
        errors.append("Previous state is not projected to claim text only")
    if _contains_key(autoregressive, "reference_state"):
        errors.append("Autoregressive State Update asset contains a reference-labeled field")
    return {
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "same_checkpoint_coverage": set(reference_by_id) == set(prediction_by_id),
        "same_context_protocol": reference.get("context_protocol")
        == autoregressive.get("context_protocol"),
        "memory_shared_between_settings": protocol.get(
            "memory_shared_between_settings"
        )
        is True,
        "full_screenplay_prefix_used": protocol.get("full_screenplay_prefix_used"),
        "autoregressive_state_update_contains_reference_fields": _contains_key(
            autoregressive, "reference_state"
        ),
    }


def _aliases(
    checkpoint_ids: list[str], public_by_checkpoint: dict[str, dict[str, Any]]
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for checkpoint_id in checkpoint_ids:
        public = public_by_checkpoint[checkpoint_id]
        values = [public.get("focal_character"), *public.get("aliases", [])]
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                output.append(text)
                seen.add(text)
    return output


def _public_claim(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim": row["claim"],
        "evidence_scene_orders": list(row.get("supporting_scene_orders", row.get("evidence_scene_orders", []))),
    }


def _reference_claim(row: dict[str, Any]) -> dict[str, Any]:
    output = _public_claim(row)
    if "stable_state_id" in row:
        output["state_id"] = row["stable_state_id"]
    if "stable_development_id" in row:
        output["development_id"] = row["stable_development_id"]
    return output


def _validate_evidence(
    *,
    target: dict[str, Any],
    previous_scene_order: int,
    current_scene_order: int,
    checkpoint_id: str,
) -> None:
    for row in target["current_state"]:
        evidence = row["evidence_scene_orders"]
        if not evidence or any(value > current_scene_order or value <= 0 for value in evidence):
            raise ValueError(f"Current-state evidence crosses boundary: {checkpoint_id}")
    for row in target["developments_since_previous_checkpoint"]:
        evidence = row["evidence_scene_orders"]
        if not evidence or any(value > current_scene_order or value <= 0 for value in evidence):
            raise ValueError(f"Development evidence crosses boundary: {checkpoint_id}")
        if not any(value > previous_scene_order for value in evidence):
            raise ValueError(f"Development has no evidence in its interval: {checkpoint_id}")


def _contains_key(value: Any, fragment: str) -> bool:
    if isinstance(value, dict):
        return any(fragment in str(key).casefold() or _contains_key(item, fragment) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_key(item, fragment) for item in value)
    return False


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "bytes": path.resolve().stat().st_size,
    }


if __name__ == "__main__":
    main()
