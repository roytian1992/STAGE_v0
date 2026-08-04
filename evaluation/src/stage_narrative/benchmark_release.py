from __future__ import annotations

import copy
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import atomic_write_json, load_json, load_scenes, sha256_file
from .task2_review import validate_task2_movie_review


RELEASE_STATUS = "complete_standard24k_reviewed"
MAX_STANDARD24K_INPUT_TOKENS = 20_928
BASE_RELEASE_SCHEMA = "stage_standard24k_benchmark_release_manifest_v1"
TASK2_REVIEWED_RELEASE_SCHEMA = "stage_standard24k_benchmark_release_manifest_v2"
BASE_ARTIFACT_ROLES = {
    "source_screenplay",
    "pipeline_screenplay",
    "kg_manifest",
    "hierarchy_manifest",
    "temporal_release_manifest",
    "protocol_manifest",
    "checkpoint_anchor_source",
    "human_review_file",
    "gold_manifest",
    "task1_instances",
    "task1_rolling_plans",
    "task1_gold_rubrics",
    "task3_instances",
    "task3_actor_context_packs",
    "task3_gold_rubrics",
    "task3_pair_groups",
    "screenplay_token_manifest",
    "construction_sidecar",
    "task1_prediction_manifest",
    "task3_prediction_manifest",
    "evaluation_manifest",
    "legacy_info",
    "legacy_task2",
    "legacy_task3_multi_turn",
}
TASK2_REVIEW_ARTIFACT_ROLES = {"reviewed_task2", "task2_review_manifest"}


def build_benchmark_release(
    *,
    project_root: Path,
    selection_manifest_path: Path,
    slot_id: str,
    protocol_dir: Path,
    gold_dir: Path,
    task1_prediction_manifest_path: Path,
    task3_prediction_manifest_path: Path,
    evaluation_dir: Path,
    kg_manifest_path: Path,
    hierarchy_manifest_path: Path,
    temporal_manifest_path: Path,
    review_file_path: Path,
    legacy_dir: Path,
    output_dir: Path,
    superseded_runs: list[str] | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite benchmark release: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "provenance").mkdir(parents=True, exist_ok=True)

    selection_manifest_path = selection_manifest_path.resolve()
    selection = _object(selection_manifest_path)
    entry = _selection_entry(selection, slot_id)
    movie_id = str(entry["movie_id"])
    source_script = (project_root / str(entry["script_path"])).resolve()
    pipeline_script = (
        project_root / str(entry.get("pipeline_script_path", entry["script_path"]))
    ).resolve()
    _require_hash(source_script, str(entry["script_sha256"]), "source screenplay")
    _require_hash(
        pipeline_script,
        str(entry.get("pipeline_script_sha256", entry["script_sha256"])),
        "pipeline screenplay",
    )
    source_scene_count = len(load_scenes(source_script))
    pipeline_scene_count = len(load_scenes(pipeline_script))
    if source_scene_count != int(entry["scene_count"]):
        raise ValueError("Selection source scene count does not match screenplay")
    if pipeline_scene_count != int(entry.get("pipeline_scene_count", entry["scene_count"])):
        raise ValueError("Selection pipeline scene count does not match screenplay")

    protocol_dir = protocol_dir.resolve()
    gold_dir = gold_dir.resolve()
    evaluation_dir = evaluation_dir.resolve()
    protocol_manifest_path = protocol_dir / "manifest.json"
    gold_manifest_path = gold_dir / "manifest.json"
    evaluation_manifest_path = evaluation_dir / "manifest.json"
    protocol_manifest = _object(protocol_manifest_path)
    gold_manifest = _object(gold_manifest_path)
    task1_prediction_manifest = _object(task1_prediction_manifest_path.resolve())
    task3_prediction_manifest = _object(task3_prediction_manifest_path.resolve())
    evaluation_manifest = _object(evaluation_manifest_path)
    kg_manifest = _object(kg_manifest_path.resolve())
    hierarchy_manifest = _object(hierarchy_manifest_path.resolve())
    temporal_manifest = _object(temporal_manifest_path.resolve())
    for label, payload in (
        ("protocol", protocol_manifest),
        ("gold", gold_manifest),
        ("Task 1 prediction", task1_prediction_manifest),
        ("Task 3 prediction", task3_prediction_manifest),
        ("evaluation", evaluation_manifest),
        ("KG", kg_manifest),
        ("temporal", temporal_manifest),
    ):
        if payload.get("movie_id") not in (None, "", movie_id):
            raise ValueError(f"{label} manifest belongs to a different movie")
    hierarchy_movie_id = hierarchy_manifest.get("movie_id")
    if hierarchy_movie_id not in (None, "", movie_id):
        raise ValueError("Hierarchy manifest belongs to a different movie")
    if kg_manifest.get("status") != "kg_completed":
        raise ValueError("KG manifest is not complete")
    if hierarchy_manifest.get("status") != "completed":
        raise ValueError("Hierarchy manifest is not complete")
    validation = temporal_manifest.get("validation", {})
    if validation.get("status") != "passed" or int(validation.get("error_count", -1)) != 0:
        raise ValueError("Temporal release has not passed zero-error validation")
    if gold_manifest.get("status") != "human_reviewed_gold":
        raise ValueError("Gold manifest is not human-reviewed")
    if not str(evaluation_manifest.get("status", "")).startswith("completed"):
        raise ValueError("Evaluation manifest is not complete")

    task1_candidates_path = protocol_dir / "task1" / "task1_rubric_candidates.json"
    task1_plans_path = protocol_dir / "task1" / "task1_rolling_plans.json"
    task3_instances_path = (
        protocol_dir / "task3" / "task3_checkpoint_single_turn.anchored.json"
    )
    task3_context_path = protocol_dir / "task3" / "task3_actor_context_packs.json"
    token_manifest_path = protocol_dir / "runtime" / "screenplay_token_manifest.json"
    task1_gold_path = gold_dir / "task1_gold_rubrics.json"
    task3_gold_path = gold_dir / "task3_gold_rubrics.json"
    task1_prediction_path = _manifest_output(
        task1_prediction_manifest, "task1_predictions.json"
    )
    task3_prediction_path = _manifest_output(
        task3_prediction_manifest, "task3_predictions.json"
    )
    task1_evaluation_path = evaluation_dir / "task1_evaluation.json"
    task3_evaluation_path = evaluation_dir / "task3_evaluation.json"

    task1_candidates = _object(task1_candidates_path)
    task1_plans = _object(task1_plans_path)
    task3_instances = _object(task3_instances_path)
    task3_contexts = _object(task3_context_path)
    task1_gold = _object(task1_gold_path)
    task3_gold = _object(task3_gold_path)
    task1_predictions = _object(task1_prediction_path)
    task3_predictions = _object(task3_prediction_path)
    task1_evaluation = _object(task1_evaluation_path)
    task3_evaluation = _object(task3_evaluation_path)
    for label, payload in (
        ("Task 1 prediction payload", task1_predictions),
        ("Task 3 prediction payload", task3_predictions),
        ("Task 1 evaluation payload", task1_evaluation),
        ("Task 3 evaluation payload", task3_evaluation),
    ):
        if payload.get("movie_id") != movie_id:
            raise ValueError(f"{label} belongs to a different movie")

    task1_ids = _exact_ids(
        "Task 1",
        {
            "candidates": _ids(task1_candidates["candidates"]),
            "gold": _ids(task1_gold["rubrics"]),
            "predictions": {
                item["instance_id"]
                for character in task1_predictions["characters"]
                for item in character["checkpoint_predictions"]
            },
            "evaluation": _ids(task1_evaluation["instances"]),
        },
    )
    task3_ids = _exact_ids(
        "Task 3",
        {
            "instances": _ids(task3_instances["instances"]),
            "contexts": _ids(task3_contexts["context_packs"]),
            "gold": _ids(task3_gold["rubrics"]),
            "predictions": _ids(task3_predictions["predictions"]),
            "evaluation": _ids(task3_evaluation["instances"]),
        },
    )
    pair_groups = _pair_groups(task3_instances["instances"], task3_evaluation)

    formal_task1 = _formal_task1(task1_plans, task1_gold)
    formal_task3 = _formal_task3(task3_instances)
    formal_contexts = _formal_task3_contexts(task3_contexts)
    formal_pairs = {
        "schema_version": "stage_task3_standard24k_pair_groups_v1",
        "movie_id": movie_id,
        "pair_group_count": len(pair_groups),
        "pair_groups": pair_groups,
    }
    task1_release_path = output_dir / "task_1_character_development_tracking.json"
    task3_release_path = output_dir / "task_3_checkpoint_single_turn.json"
    context_release_path = output_dir / "task_3_actor_context_packs.json"
    pair_release_path = output_dir / "task_3_pair_groups.json"
    atomic_write_json(task1_release_path, formal_task1)
    atomic_write_json(task3_release_path, formal_task3)
    atomic_write_json(context_release_path, formal_contexts)
    atomic_write_json(pair_release_path, formal_pairs)

    legacy_files = {
        "info": legacy_dir.resolve() / "info.json",
        "task2": legacy_dir.resolve() / "task_2_question_answering.csv",
        "task3_multi_turn": legacy_dir.resolve()
        / "task_3_in_script_character_role_play_multi_turn.json",
    }
    for label, path in legacy_files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty legacy {label} asset: {path}")

    # Keep legacy inputs immutable inside the release. Manifest10 is rebuilt
    # incrementally, so references into its mutable working tree would make an
    # otherwise valid release fail hash validation on the next build.
    legacy_snapshot_dir = output_dir / "provenance" / "legacy"
    legacy_snapshot_dir.mkdir(parents=True, exist_ok=True)
    legacy_snapshots = {
        "info": legacy_snapshot_dir / "info.json",
        "task2": legacy_snapshot_dir / "task_2_question_answering.csv",
        "task3_multi_turn": legacy_snapshot_dir
        / "task_3_in_script_character_role_play_multi_turn.json",
    }
    for label, source in legacy_files.items():
        shutil.copy2(source, legacy_snapshots[label])

    max_tokens = _token_maxima(
        task1_plans=task1_plans,
        task3_contexts=task3_contexts,
        task1_predictions=task1_predictions,
        task3_predictions=task3_predictions,
        task1_evaluation=task1_evaluation,
        task3_evaluation=task3_evaluation,
    )
    if max(max_tokens.values(), default=0) > MAX_STANDARD24K_INPUT_TOKENS:
        raise ValueError(f"A formal prompt exceeds 20,928 tokens: {max_tokens}")

    sidecar_path = output_dir / "provenance" / "construction_sidecar.json"
    sidecar = {
        "schema_version": "stage_standard24k_construction_sidecar_v1",
        "movie_id": movie_id,
        "note": "Opaque construction IDs are retained only in these source artifacts.",
        "task1_instance_ids": sorted(task1_ids),
        "task3_instance_ids": sorted(task3_ids),
        "raw_construction_artifacts": [
            _artifact(task1_candidates_path),
            _artifact(task1_plans_path),
            _artifact(task3_instances_path),
            _artifact(task3_context_path),
            _artifact(review_file_path.resolve()),
        ],
    }
    atomic_write_json(sidecar_path, sidecar)

    artifacts = {
        "source_screenplay": _artifact(source_script),
        "pipeline_screenplay": _artifact(pipeline_script),
        "kg_manifest": _artifact(kg_manifest_path.resolve()),
        "hierarchy_manifest": _artifact(hierarchy_manifest_path.resolve()),
        "temporal_release_manifest": _artifact(temporal_manifest_path.resolve()),
        "protocol_manifest": _artifact(protocol_manifest_path),
        "checkpoint_anchor_source": _artifact(
            Path(protocol_manifest["task3_anchor_source_path"]).resolve()
        ),
        "human_review_file": _artifact(review_file_path.resolve()),
        "gold_manifest": _artifact(gold_manifest_path),
        "task1_instances": _artifact(task1_release_path),
        "task1_rolling_plans": _artifact(task1_plans_path),
        "task1_gold_rubrics": _artifact(task1_gold_path),
        "task3_instances": _artifact(task3_release_path),
        "task3_actor_context_packs": _artifact(context_release_path),
        "task3_gold_rubrics": _artifact(task3_gold_path),
        "task3_pair_groups": _artifact(pair_release_path),
        "screenplay_token_manifest": _artifact(token_manifest_path),
        "construction_sidecar": _artifact(sidecar_path),
        "task1_prediction_manifest": _artifact(
            task1_prediction_manifest_path.resolve()
        ),
        "task3_prediction_manifest": _artifact(
            task3_prediction_manifest_path.resolve()
        ),
        "evaluation_manifest": _artifact(evaluation_manifest_path),
        "legacy_info": _artifact(legacy_snapshots["info"]),
        "legacy_task2": _artifact(legacy_snapshots["task2"]),
        "legacy_task3_multi_turn": _artifact(legacy_snapshots["task3_multi_turn"]),
    }
    manifest = {
        "schema_version": BASE_RELEASE_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": RELEASE_STATUS,
        "review_tier": "human_reviewed_gold_with_agent_validation",
        "slot_id": slot_id,
        "movie_id": movie_id,
        "title": entry["title"],
        "language": entry["language"],
        "selection_manifest": _artifact(selection_manifest_path),
        "screenplay": {
            "source_scene_count": source_scene_count,
            "pipeline_scene_count": pipeline_scene_count,
        },
        "counts": {
            "task1_instances": len(task1_ids),
            "task3_instances": len(task3_ids),
            "task3_pair_groups": len(pair_groups),
            "evaluation_warnings": int(
                evaluation_manifest.get("counts", {}).get(
                    "task3_judge_consistency_warnings", 0
                )
            ),
        },
        "token_validation": {
            "max_input_tokens": MAX_STANDARD24K_INPUT_TOKENS,
            "observed_maxima": max_tokens,
            "minimum_margin": MAX_STANDARD24K_INPUT_TOKENS
            - max(max_tokens.values(), default=0),
            "status": "passed",
        },
        "id_coverage": {
            "task1_exact_match": True,
            "task3_exact_match": True,
            "task1_instance_ids": sorted(task1_ids),
            "task3_instance_ids": sorted(task3_ids),
        },
        "python_executable": protocol_manifest.get("python_executable"),
        "runtime_config_path": protocol_manifest.get("runtime_config_path"),
        "runtime_config_sha256": protocol_manifest.get("runtime_config_sha256"),
        "artifacts": artifacts,
        "superseded_runs": superseded_runs or [],
    }
    manifest_path = output_dir / "standard24k_benchmark_release_manifest.json"
    atomic_write_json(manifest_path, manifest)
    validate_benchmark_release_manifest(manifest_path)
    return manifest_path


def validate_benchmark_release_manifest(path: Path) -> dict[str, Any]:
    manifest = _object(path.resolve())
    schema = manifest.get("schema_version")
    if schema not in {BASE_RELEASE_SCHEMA, TASK2_REVIEWED_RELEASE_SCHEMA}:
        raise ValueError("Unsupported benchmark release manifest schema")
    if manifest.get("status") != RELEASE_STATUS:
        raise ValueError("Benchmark release is not complete_standard24k_reviewed")
    required_roles = set(BASE_ARTIFACT_ROLES)
    if schema == TASK2_REVIEWED_RELEASE_SCHEMA:
        required_roles.update(TASK2_REVIEW_ARTIFACT_ROLES)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != required_roles:
        missing = required_roles - set(artifacts or {})
        extra = set(artifacts or {}) - required_roles
        raise ValueError(f"Release artifact roles mismatch: missing={missing} extra={extra}")
    for role, artifact in artifacts.items():
        if not isinstance(artifact, dict) or set(artifact) < {"path", "sha256", "bytes"}:
            raise ValueError(f"Malformed artifact reference: {role}")
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_file():
            raise ValueError(f"Missing release artifact {role}: {artifact_path}")
        _require_hash(artifact_path, artifact["sha256"], role)
        if artifact_path.stat().st_size != int(artifact["bytes"]):
            raise ValueError(f"Artifact byte count drift: {role}")
    counts = manifest.get("counts", {})
    if int(counts.get("task1_instances", 0)) <= 0:
        raise ValueError("Release has no Task 1 instances")
    if int(counts.get("task3_instances", 0)) <= 0:
        raise ValueError("Release has no Task 3 instances")
    token_validation = manifest.get("token_validation", {})
    if token_validation.get("status") != "passed":
        raise ValueError("Token validation did not pass")
    if max(token_validation.get("observed_maxima", {}).values(), default=0) > int(
        token_validation.get("max_input_tokens", 0)
    ):
        raise ValueError("Recorded prompt maximum exceeds the release limit")
    if not manifest.get("id_coverage", {}).get("task1_exact_match"):
        raise ValueError("Task 1 instance coverage is not exact")
    if not manifest.get("id_coverage", {}).get("task3_exact_match"):
        raise ValueError("Task 3 instance coverage is not exact")
    if schema == TASK2_REVIEWED_RELEASE_SCHEMA:
        task2_path = Path(artifacts["reviewed_task2"]["path"])
        review_path = Path(artifacts["task2_review_manifest"]["path"])
        review = validate_task2_movie_review(
            review_path, reviewed_task2_path=task2_path
        )
        if review.get("movie_id") != manifest.get("movie_id"):
            raise ValueError("Task 2 review belongs to a different movie")
        if review.get("slot_id") != manifest.get("slot_id"):
            raise ValueError("Task 2 review belongs to a different slot")
        if review.get("source_task2_sha256") != artifacts["legacy_task2"]["sha256"]:
            raise ValueError("Task 2 review baseline differs from frozen legacy Task 2")
        if int(counts.get("task2_questions", -1)) != int(
            review["counts"]["reviewed_rows"]
        ):
            raise ValueError("Task 2 release question count mismatch")
        if int(counts.get("task2_replacements", -1)) != int(
            review["counts"]["replacements"]
        ):
            raise ValueError("Task 2 release replacement count mismatch")
        quality = manifest.get("task2_quality_review") or {}
        if quality.get("status") != "human_reviewed":
            raise ValueError("Task 2 release quality review is not human_reviewed")
    return manifest


def promote_task2_reviewed_release(
    *,
    source_release_manifest_path: Path,
    reviewed_task2_path: Path,
    task2_review_manifest_path: Path,
    output_dir: Path,
) -> Path:
    source_release_manifest_path = source_release_manifest_path.resolve()
    source = validate_benchmark_release_manifest(source_release_manifest_path)
    if source.get("schema_version") != BASE_RELEASE_SCHEMA:
        raise ValueError("Task 2 promotion requires an unpromoted v1 benchmark release")
    reviewed_task2_path = reviewed_task2_path.resolve()
    task2_review_manifest_path = task2_review_manifest_path.resolve()
    review = validate_task2_movie_review(
        task2_review_manifest_path, reviewed_task2_path=reviewed_task2_path
    )
    if review.get("movie_id") != source.get("movie_id"):
        raise ValueError("Task 2 review/release movie mismatch")
    if review.get("slot_id") != source.get("slot_id"):
        raise ValueError("Task 2 review/release slot mismatch")
    if review.get("source_task2_sha256") != source["artifacts"]["legacy_task2"]["sha256"]:
        raise ValueError("Task 2 review does not use the release's frozen baseline")

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Task 2 promoted release: {output_dir}")
    snapshot_dir = output_dir / "provenance" / "task2"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    reviewed_snapshot = snapshot_dir / "task_2_question_answering.csv"
    review_snapshot = snapshot_dir / "task2_quality_review.json"
    shutil.copy2(reviewed_task2_path, reviewed_snapshot)
    shutil.copy2(task2_review_manifest_path, review_snapshot)

    manifest = copy.deepcopy(source)
    manifest["schema_version"] = TASK2_REVIEWED_RELEASE_SCHEMA
    manifest["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["review_tier"] = "human_reviewed_gold_with_task2_quality_audit"
    manifest["derived_from_benchmark_release"] = _artifact(
        source_release_manifest_path
    )
    manifest["artifacts"]["reviewed_task2"] = _artifact(reviewed_snapshot)
    manifest["artifacts"]["task2_review_manifest"] = _artifact(review_snapshot)
    manifest["counts"]["task2_questions"] = int(review["counts"]["reviewed_rows"])
    manifest["counts"]["task2_replacements"] = int(review["counts"]["replacements"])
    manifest["task2_quality_review"] = {
        "status": "human_reviewed",
        "coverage_mode": review["review_coverage"]["mode"],
        "baseline_rows": int(review["counts"]["baseline_rows"]),
        "reviewed_rows": int(review["counts"]["reviewed_rows"]),
        "replacement_count": int(review["counts"]["replacements"]),
        "max_replacements": 5,
        "same_schema": True,
        "same_type_distribution": True,
        "semantic_llm_reruns": 0,
    }
    prior = list(manifest.get("superseded_runs") or [])
    source_string = str(source_release_manifest_path)
    if source_string not in prior:
        prior.append(source_string)
    manifest["superseded_runs"] = prior
    manifest_path = output_dir / "standard24k_benchmark_release_manifest.json"
    atomic_write_json(manifest_path, manifest)
    validate_benchmark_release_manifest(manifest_path)
    return manifest_path


def _formal_task1(plans: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    calls_by_instance = {}
    aliases_by_character = {}
    for plan in plans["plans"]:
        aliases_by_character[plan["focal_character"]] = list(plan.get("aliases", []))
        for call in plan["calls"]:
            instance_id = call.get("task_instance_id")
            if instance_id:
                calls_by_instance[instance_id] = {
                    "call_order": call["call_order"],
                    "block_start_scene_order": call["block_start_scene_order"],
                    "block_end_scene_order": call["block_end_scene_order"],
                    "previous_checkpoint_scene_order": call[
                        "previous_checkpoint_scene_order"
                    ],
                }
    instances = []
    for rubric in gold["rubrics"]:
        instance_id = rubric["instance_id"]
        if instance_id not in calls_by_instance:
            raise ValueError(f"Task 1 gold instance has no rolling checkpoint: {instance_id}")
        instances.append(
            {
                "instance_id": instance_id,
                "focal_character": rubric["character"],
                "aliases": aliases_by_character.get(rubric["character"], []),
                "checkpoint": rubric["checkpoint"],
                "rolling_boundary": calls_by_instance[instance_id],
                "instruction": (
                    "Track the focal character through the screenplay up to this "
                    "checkpoint. Report current state, developments since the previous "
                    "checkpoint, and unresolved threads with supporting scene orders."
                ),
                "output_fields": [
                    "current_state",
                    "developments_since_previous_checkpoint",
                    "unresolved_threads",
                ],
            }
        )
    return {
        "schema_version": "stage_task1_standard24k_model_instances_v1",
        "movie_id": plans["movie_id"],
        "task": "character_development_tracking",
        "context_protocol": "standard-24k-sequential-rolling",
        "instance_count": len(instances),
        "instances": instances,
    }


def _formal_task3(raw: dict[str, Any]) -> dict[str, Any]:
    instances = []
    for item in raw["instances"]:
        model_input = item["model_input"]
        anchor = model_input["checkpoint_anchor"]
        instances.append(
            {
                "instance_id": item["instance_id"],
                "movie_id": item["movie_id"],
                "language": item["language"],
                "character": item["character"],
                "interaction_format": "single_turn",
                "actor_context_ref": item["instance_id"],
                "checkpoint_boundary": {
                    "scene_order": anchor["scene_order"],
                    "char_end": anchor["char_end"],
                    "boundary_policy": anchor["boundary_policy"],
                    "review_status": anchor["review_status"],
                },
                "model_input": {
                    "interaction_context": model_input["interaction_context"],
                    "dialogue_history": model_input["dialogue_history"],
                    "current_user_turn": model_input["current_user_turn"],
                },
            }
        )
    return {
        "schema_version": "stage_task3_standard24k_model_instances_v1",
        "movie_id": raw["movie_id"],
        "task": "checkpoint_conditioned_single_turn_role_play",
        "instance_count": len(instances),
        "instances": instances,
    }


def _formal_task3_contexts(raw: dict[str, Any]) -> dict[str, Any]:
    packs = []
    for item in raw["context_packs"]:
        actor_input = item["actor_input"]
        policy = actor_input["input_policy"]
        anchor = policy["checkpoint_anchor"]
        packs.append(
            {
                "instance_id": item["instance_id"],
                "context_pack_id": item["context_pack_id"],
                "memory_mode": item["memory_mode"],
                "raw_prompt_tokens": item["raw_prompt_tokens"],
                "accounted_input_tokens": item["accounted_input_tokens"],
                "max_input_tokens": item["max_input_tokens"],
                "actor_input": {
                    "character": actor_input["character"],
                    "interaction_format": actor_input["interaction_format"],
                    "interaction_context": actor_input["interaction_context"],
                    "dialogue_history": actor_input["dialogue_history"],
                    "current_user_turn": actor_input["current_user_turn"],
                    "role_context": actor_input["role_context"],
                    "input_policy": {
                        "checkpoint_boundary": {
                            "scene_order": anchor["scene_order"],
                            "char_end": anchor["char_end"],
                            "boundary_policy": anchor["boundary_policy"],
                            "review_status": anchor["review_status"],
                        },
                        "memory_mode": policy["memory_mode"],
                        "same_scene_graph_memory": policy[
                            "same_scene_graph_memory"
                        ],
                    },
                },
            }
        )
    return {
        "schema_version": "stage_task3_standard24k_actor_context_packs_release_v1",
        "movie_id": raw["movie_id"],
        "instance_count": len(packs),
        "context_packs": packs,
    }


def _pair_groups(
    instances: list[dict[str, Any]], evaluation: dict[str, Any]
) -> list[dict[str, Any]]:
    members: dict[str, set[str]] = defaultdict(set)
    for item in instances:
        group_id = item["evaluator_reference"].get("paired_prompt_group_id", "")
        if group_id:
            members[group_id].add(item["instance_id"])
    evaluated = {item["pair_group_id"]: item for item in evaluation.get("pairs", [])}
    if set(members) != set(evaluated):
        raise ValueError("Task 3 pair group coverage differs from evaluation")
    output = []
    for group_id in sorted(members):
        row = evaluated[group_id]
        if set(row["instance_ids"]) != members[group_id]:
            raise ValueError(f"Task 3 pair members differ for {group_id}")
        output.append(
            {
                "pair_group_id": group_id,
                "pair_type": row["pair_type"],
                "instance_ids": row["instance_ids"],
            }
        )
    return output


def _token_maxima(**payloads: dict[str, Any]) -> dict[str, int]:
    task1_plans = payloads["task1_plans"]
    task3_contexts = payloads["task3_contexts"]
    task1_predictions = payloads["task1_predictions"]
    task3_predictions = payloads["task3_predictions"]
    task1_evaluation = payloads["task1_evaluation"]
    task3_evaluation = payloads["task3_evaluation"]
    return {
        "task1_plan": max(
            (
                int(call.get("conservative_raw_prompt_tokens", 0))
                for plan in task1_plans["plans"]
                for call in plan["calls"]
            ),
            default=0,
        ),
        "task1_prediction": max(
            (
                int(call.get("accounted_input_tokens", 0))
                for character in task1_predictions["characters"]
                for call in character["calls"]
            ),
            default=0,
        ),
        "task3_actor": max(
            (
                int(item.get("accounted_input_tokens", 0))
                for item in task3_contexts["context_packs"]
            ),
            default=0,
        ),
        "task3_prediction": max(
            (
                int(item.get("accounted_input_tokens", 0))
                for item in task3_predictions["predictions"]
            ),
            default=0,
        ),
        "task1_judge": max(
            (int(item.get("prompt_tokens", 0)) for item in task1_evaluation["instances"]),
            default=0,
        ),
        "task3_response_judge": max(
            (int(item.get("prompt_tokens", 0)) for item in task3_evaluation["instances"]),
            default=0,
        ),
        "task3_pair_judge": max(
            (int(item.get("prompt_tokens", 0)) for item in task3_evaluation.get("pairs", [])),
            default=0,
        ),
    }


def _selection_entry(selection: dict[str, Any], slot_id: str) -> dict[str, Any]:
    matches = [item for item in selection.get("entries", []) if item.get("slot_id") == slot_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one selection slot {slot_id}")
    return matches[0]


def _manifest_output(manifest: dict[str, Any], filename: str) -> Path:
    matches = [Path(item["path"]).resolve() for item in manifest.get("outputs", []) if Path(item["path"]).name == filename]
    if len(matches) != 1:
        raise ValueError(f"Manifest does not contain exactly one {filename}")
    path = matches[0]
    recorded = next(item["sha256"] for item in manifest["outputs"] if Path(item["path"]).name == filename)
    _require_hash(path, recorded, filename)
    return path


def _exact_ids(label: str, groups: dict[str, set[str]]) -> set[str]:
    iterator = iter(groups.items())
    first_label, reference = next(iterator)
    if not reference:
        raise ValueError(f"{label} has no instances")
    for group_label, ids in iterator:
        if ids != reference:
            raise ValueError(
                f"{label} ID coverage differs: {first_label} vs {group_label}; "
                f"missing={sorted(reference - ids)} extra={sorted(ids - reference)}"
            )
    return reference


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    ids = {str(item.get("instance_id", "")) for item in rows}
    if "" in ids or len(ids) != len(rows):
        raise ValueError("Instance IDs must be non-empty and unique")
    return ids


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Missing artifact: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {label}: expected={expected} actual={actual}")


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload
