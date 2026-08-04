from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import load_json, sha256_file
from .task1_metrics import aggregate_task1, localize_task1_prediction, score_task1_checkpoint, score_task1_trajectory
from .task1_schemas import (
    validate_task1_checkpoint_judgment,
    validate_task1_prediction,
    validate_task1_private_assets,
    validate_task1_trajectory_judgment,
)


def validate_task1_evaluation_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    contract_path = run_dir / "run_contract.json"
    output_path = run_dir / "task1_evaluation.json"
    manifest = _object(manifest_path)
    contract = _object(contract_path)
    output = _object(output_path)
    if manifest.get("status") != "completed":
        raise ValueError("Task 1 evaluation manifest is not completed")
    if manifest.get("run_contract", {}).get("sha256") != sha256_file(contract_path):
        raise ValueError("Task 1 evaluation run contract hash drift")
    _validate_hashed_paths(contract.get("inputs"), label="input")
    _validate_hashed_paths(contract.get("prompt_artifacts"), label="prompt")
    _validate_hashed_paths(manifest.get("outputs"), label="output")

    input_by_name = {Path(row["path"]).name: Path(row["path"]) for row in contract["inputs"]}
    required_inputs = {
        "task1_public_instances.json",
        "task1_private_evaluator.json",
        "task1_rolling_plans.json",
        "task1_predictions.json",
        "manifest.json",
    }
    missing = required_inputs - set(input_by_name)
    if missing:
        raise ValueError(f"Task 1 evaluation contract lacks required inputs: {sorted(missing)}")
    private = validate_task1_private_assets(
        _object(input_by_name["task1_private_evaluator.json"])
    )
    predictions = _object(input_by_name["task1_predictions.json"])
    prediction_by_id = _predictions_by_instance(predictions)
    trajectory_by_character = {
        row["character_id"]: row for row in private["trajectories"]
    }
    rubric_by_instance = {
        rubric["instance_id"]: (trajectory, rubric)
        for trajectory in private["trajectories"]
        for rubric in trajectory["checkpoint_rubrics"]
    }
    checkpoint_rows = output.get("checkpoint_results")
    if not isinstance(checkpoint_rows, list):
        raise ValueError("Task 1 evaluation checkpoint_results must be an array")
    checkpoint_by_id = {row.get("instance_id"): row for row in checkpoint_rows}
    if len(checkpoint_by_id) != len(checkpoint_rows) or set(checkpoint_by_id) != set(rubric_by_instance):
        raise ValueError("Task 1 evaluation checkpoint result coverage drift")
    recomputed_checkpoints = []
    for instance_id, (trajectory, rubric) in rubric_by_instance.items():
        row = checkpoint_by_id[instance_id]
        localized = localize_task1_prediction(prediction_by_id[instance_id]["prediction"])
        if row.get("localized_prediction") != localized:
            raise ValueError(f"Task 1 localized prediction drift: {instance_id}")
        judgment = validate_task1_checkpoint_judgment(
            row["judgment"],
            gold_ids={
                claim["local_id"]
                for field in ("current_state_claims", "development_claims")
                for claim in rubric[field]
            },
            prediction_ids={claim["local_id"] for claim in localized},
            inactive_state_ids={
                claim["local_id"] for claim in rubric["inactive_state_claims"]
            },
        )
        scoring = score_task1_checkpoint(
            prediction=localized, rubric=rubric, judgment=judgment
        )
        if row.get("scoring") != scoring:
            raise ValueError(f"Task 1 checkpoint metric drift: {instance_id}")
        if row.get("character_id") != trajectory["character_id"]:
            raise ValueError(f"Task 1 checkpoint character drift: {instance_id}")
        recomputed_checkpoints.append({**row, "scoring": scoring})

    raw_judgments = output.get("trajectory_judgments")
    if not isinstance(raw_judgments, list):
        raise ValueError("Task 1 trajectory_judgments must be an array")
    judgments = {row.get("character_id"): row for row in raw_judgments}
    if len(judgments) != len(raw_judgments) or set(judgments) != set(trajectory_by_character):
        raise ValueError("Task 1 trajectory judgment coverage drift")
    recomputed_trajectories = []
    for character_id, trajectory in trajectory_by_character.items():
        judgment = {
            "development_clusters": judgments[character_id]["development_clusters"],
            "adjacent_checks": judgments[character_id]["adjacent_checks"],
        }
        checkpoint_ids = set(trajectory["checkpoint_ids"])
        rows = [
            row
            for row in recomputed_checkpoints
            if row["character_id"] == character_id and row["checkpoint_id"] in checkpoint_ids
        ]
        refs = {
            f"{row['instance_id']}|{claim['local_id']}"
            for row in rows
            for claim in row["localized_prediction"]
            if claim["prediction_type"] == "development"
        }
        validate_task1_trajectory_judgment(
            judgment,
            development_prediction_refs=refs,
            adjacent_instance_pairs=[
                (left["instance_id"], right["instance_id"])
                for left, right in zip(
                    trajectory["checkpoint_rubrics"], trajectory["checkpoint_rubrics"][1:]
                )
            ],
        )
        recomputed_trajectories.append(
            score_task1_trajectory(
                trajectory=trajectory,
                checkpoint_results=rows,
                trajectory_judgment=judgment,
            )
        )
    aggregate = aggregate_task1(recomputed_trajectories)
    if output.get("aggregate") != aggregate:
        raise ValueError("Task 1 aggregate metric drift")
    counts = manifest.get("counts", {})
    expected_counts = {
        "expected_checkpoints": len(rubric_by_instance),
        "evaluated_checkpoints": len(checkpoint_rows),
        "expected_trajectories": len(trajectory_by_character),
        "evaluated_trajectories": len(recomputed_trajectories),
        "expected_adjacent_pairs": sum(
            max(0, len(row["checkpoint_ids"]) - 1) for row in private["trajectories"]
        ),
    }
    for key, value in expected_counts.items():
        if counts.get(key) != value:
            raise ValueError(f"Task 1 evaluation manifest count drift: {key}")
    if counts.get("evaluated_adjacent_pairs") != expected_counts["expected_adjacent_pairs"]:
        raise ValueError("Task 1 evaluated adjacent-pair count drift")
    if counts.get("failure_count") != 0 or counts.get("truncated_count") != 0:
        raise ValueError("Task 1 completed evaluation records failures or truncation")
    return {
        "schema_version": "stage_task1_evaluation_validation",
        "status": "passed",
        "movie_id": private["movie_id"],
        "recomputed_metrics_exact_match": True,
        "counts": expected_counts,
        "run_manifest_sha256": sha256_file(manifest_path),
        "task1_evaluation_sha256": sha256_file(output_path),
    }


def _predictions_by_instance(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for character in payload.get("characters", []):
        for row in character.get("checkpoint_predictions", []):
            validate_task1_prediction(row["prediction"])
            if row["instance_id"] in output:
                raise ValueError("Task 1 prediction input contains duplicate IDs")
            output[row["instance_id"]] = row
    return output


def _validate_hashed_paths(rows: Any, *, label: str) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Task 1 evaluation has no hashed {label} artifacts")
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"Task 1 evaluation {label} artifact hash drift: {path}")


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
