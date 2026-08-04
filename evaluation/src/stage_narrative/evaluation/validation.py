from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import load_json, sha256_file
from .schemas import (
    validate_task1_judgment,
    validate_task1_sequence_judgment,
    validate_task3_pair_judgment,
    validate_task3_response_judgment,
)
from .task1_checkpoint_metrics import (
    aggregate_checkpoint_task1,
    aggregate_task1_sequences,
    score_task1_instance,
)
from .task3_metrics import aggregate_task3


def validate_evaluation_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = _read(manifest_path)
    contract_path = Path(manifest["run_contract"]["path"])
    _require_hash(contract_path, manifest["run_contract"]["sha256"])
    contract = _read(contract_path)
    if manifest["evaluation_mode"] != contract["evaluation_mode"]:
        raise ValueError("Evaluation manifest/contract mode drift")
    if manifest["model_contract"] != contract["model_contract"]:
        raise ValueError("Evaluation manifest/contract model identity drift")
    task1_output = _read(run_dir / "task1_evaluation.json")
    task3_output = _read(run_dir / "task3_evaluation.json")
    for output in manifest["outputs"]:
        _require_hash(Path(output["path"]), output["sha256"])

    input_paths = [Path(row["path"]) for row in contract["inputs"]]
    for row, path in zip(contract["inputs"], input_paths, strict=True):
        _require_hash(path, row["sha256"])
    task1_prediction_path = _one_named(input_paths, "task1_predictions.json")
    task3_prediction_path = _one_named(input_paths, "task3_predictions.json")
    gold_dir = _one_parent_containing(input_paths, "task1_gold_rubrics.json")
    task1_predictions = _read(task1_prediction_path)
    task3_predictions = _read(task3_prediction_path)
    task1_gold = _read(gold_dir / "task1_gold_rubrics.json")
    task3_gold = _read(gold_dir / "task3_gold_rubrics.json")
    task1_gold_by_id = {row["instance_id"]: row for row in task1_gold["rubrics"]}
    task3_gold_by_id = {row["instance_id"]: row for row in task3_gold["rubrics"]}
    expected_task1_ids = {
        row["instance_id"]
        for character in task1_predictions["characters"]
        for row in character["checkpoint_predictions"]
    }
    expected_task3_ids = {row["instance_id"] for row in task3_predictions["predictions"]}
    observed_task1_ids = [row["instance_id"] for row in task1_output["instances"]]
    observed_task3_ids = [row["instance_id"] for row in task3_output["instances"]]
    _exact_unique_ids(expected_task1_ids, observed_task1_ids, "Task 1 evaluation")
    _exact_unique_ids(expected_task3_ids, observed_task3_ids, "Task 3 evaluation")
    if expected_task1_ids != set(task1_gold_by_id) or expected_task3_ids != set(task3_gold_by_id):
        raise ValueError("Prediction and gold ID coverage differ")

    recomputed_task1 = []
    for row in task1_output["instances"]:
        gold = task1_gold_by_id[row["instance_id"]]
        localized = row["localized_prediction"]
        gold_ids = {
            item["local_id"]
            for field in ("current_state_claims", "development_claims", "invariant_claims")
            for item in gold["rubric"][field]
        }
        prediction_ids = {item["local_id"] for item in localized}
        validate_task1_judgment(
            row["judgment"], gold_ids=gold_ids, prediction_ids=prediction_ids
        )
        scoring = score_task1_instance(
            prediction=localized, rubric=gold["rubric"], judgment=row["judgment"]
        )
        if scoring != row["scoring"]:
            raise ValueError(f"Serialized Task 1 scoring drift: {row['instance_id']}")
        recomputed_task1.append(row)
    for row in task1_output["sequences"]:
        validate_task1_sequence_judgment(row["judgment"])
        expected_consistent = bool(
            row["judgment"]["state_carry_forward"]
            and row["judgment"]["development_to_state_coherent"]
            and not row["judgment"]["contradiction_present"]
            and not row["judgment"]["premature_or_future_information"]
        )
        if row["consistent"] != expected_consistent:
            raise ValueError(f"Task 1 sequence scoring drift: {row['sequence_id']}")
    task1_aggregate = aggregate_checkpoint_task1(recomputed_task1)
    task1_aggregate["longitudinal_consistency"] = aggregate_task1_sequences(
        task1_output["sequences"]
    )
    task1_aggregate["delayed_update"] = {
        "eligible_developments": 0,
        "value": None,
        "reason": "missing_explicit_gold_development_lineage",
    }
    if task1_aggregate != task1_output["aggregate"]:
        raise ValueError("Task 1 aggregate does not equal deterministic recomputation")

    for row in task3_output["instances"]:
        validate_task3_response_judgment(
            row["judgment"],
            allowed_evidence_ids=set(row["judgment"]["evidence_local_ids"]),
        )
    for row in task3_output["pairs"]:
        validate_task3_pair_judgment(
            row["judgment"], expected_pair_type=row["pair_type"]
        )
        if not all(str(label).startswith("T") for label in row["judgment"]["local_evidence_labels"]):
            raise ValueError(f"Task 3 pair evidence label is not pair-local: {row['pair_group_id']}")
        if [item["instance_id"] for item in row["response_prerequisites"]] != row["instance_ids"]:
            raise ValueError(f"Task 3 pair prerequisite coverage drift: {row['pair_group_id']}")
        for prerequisite in row["response_prerequisites"]:
            source = next(
                item
                for item in task3_output["instances"]
                if item["instance_id"] == prerequisite["instance_id"]
            )["judgment"]
            expected_passed = bool(
                source["stance_compatible"]
                and not source["future_leakage"]
                and not source["unknown_fact_hallucination"]
            )
            if prerequisite["passed"] != expected_passed:
                raise ValueError(f"Task 3 pair prerequisite scoring drift: {row['pair_group_id']}")
    task3_aggregate = aggregate_task3(task3_output["instances"], task3_output["pairs"])
    if task3_aggregate != task3_output["aggregate"]:
        raise ValueError("Task 3 aggregate does not equal deterministic recomputation")

    counts = manifest["counts"]
    expected_counts = {
        "expected_task1": len(expected_task1_ids),
        "evaluated_task1": len(task1_output["instances"]),
        "expected_task1_sequences": int(contract["preflight"]["task1_sequence_judges"]),
        "evaluated_task1_sequences": len(task1_output["sequences"]),
        "expected_task3": len(expected_task3_ids),
        "evaluated_task3": len(task3_output["instances"]),
        "expected_task3_pairs": int(contract["preflight"]["task3_pair_judges"]),
        "evaluated_task3_pairs": len(task3_output["pairs"]),
        "failure_count": 0,
        "truncated_count": 0,
    }
    if any(key.startswith("seed_reused_") for key in counts):
        expected_counts.update(
            {
                "seed_reused_task1": sum(
                    "seed_reused_from" in row for row in task1_output["instances"]
                ),
                "seed_reused_task1_sequences": sum(
                    "seed_reused_from" in row for row in task1_output["sequences"]
                ),
                "seed_reused_task3": sum(
                    "seed_reused_from" in row for row in task3_output["instances"]
                ),
                "seed_reused_task3_pairs": sum(
                    "seed_reused_from" in row for row in task3_output["pairs"]
                ),
            }
        )
    if counts != expected_counts:
        raise ValueError(f"Evaluation manifest counts drift: {counts} != {expected_counts}")
    if manifest["status"] != "completed":
        raise ValueError("Evaluation manifest is not completed")
    if manifest["evaluation_mode"] == "formal_independent_evaluation":
        model_contract = manifest.get("model_contract", {})
        if model_contract.get("independent") is not True:
            raise ValueError("Formal evaluation manifest is not independent")
        identity = model_contract.get("judge_identity")
        if not isinstance(identity, dict) or identity.get(
            "weights_and_model_family_distinct_from_actor"
        ) is not True:
            raise ValueError("Formal evaluation manifest lacks an independent judge identity")
    if manifest["evaluation_mode"] == "formal_blinded_human_evaluation":
        model_contract = manifest.get("model_contract", {})
        if (
            model_contract.get("independent") is not True
            or model_contract.get("judge_type") != "blinded_human_consensus"
            or int(model_contract.get("annotator_count", 0)) < 3
        ):
            raise ValueError("Formal human evaluation lacks three-person blinded consensus")
    return {
        "schema_version": "stage_evaluation_run_validation_v1",
        "status": "passed",
        "run_dir": str(run_dir),
        "evaluation_mode": manifest["evaluation_mode"],
        "counts": counts,
        "recomputed_metrics_exact_match": True,
        "input_hashes_valid": True,
        "output_hashes_valid": True,
    }


def validate_evaluation_release(release_dir: Path) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    manifest_path = release_dir / "evaluation_release_manifest.json"
    manifest = _read(manifest_path)
    required_paths = {
        "README.md",
        "metric_spec.md",
        "calibration_summary.md",
        "test_report.json",
    }
    artifacts = manifest["artifacts"]
    observed = set()
    for row in artifacts:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Release artifact path is not relative: {relative}")
        path = release_dir / relative
        _require_hash(path, row["sha256"])
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"Release artifact byte count drift: {relative}")
        observed.add(relative.as_posix())
    if not required_paths <= observed:
        raise ValueError(f"Evaluation release missing required artifacts: {sorted(required_paths - observed)}")
    if manifest["partial_weight"] != 0.5 or manifest["primary_match_variant"] != "strict":
        raise ValueError("Evaluation release metric parameters drift")
    if manifest["formal_benchmark_status"] == "completed_human_calibrated_formal_evaluation":
        human = manifest.get("human_calibration", {})
        reference = manifest.get("reference_evaluation_run") or {}
        model_contract = reference.get("model_contract", {})
        if human.get("status") != "passed" or int(human.get("annotator_count", 0)) < 3:
            raise ValueError("Formal release lacks passed three-annotator calibration")
        if reference.get("evaluation_mode") not in {
            "formal_independent_evaluation",
            "formal_blinded_human_evaluation",
        }:
            raise ValueError("Formal release reference is not independent or blinded-human")
        if model_contract.get("independent") is not True:
            raise ValueError("Formal release reference model contract is not independent")
        if reference.get("evaluation_mode") == "formal_independent_evaluation":
            if model_contract.get("judge_identity", {}).get(
                "weights_and_model_family_distinct_from_actor"
            ) is not True:
                raise ValueError("Formal release lacks independent judge identity")
        elif (
            model_contract.get("judge_type") != "blinded_human_consensus"
            or int(model_contract.get("annotator_count", 0)) < 3
        ):
            raise ValueError("Formal release lacks blinded-human consensus identity")
    return {
        "schema_version": "stage_evaluation_release_validation_v1",
        "status": "passed",
        "release_dir": str(release_dir),
        "artifact_count": len(artifacts),
        "artifact_hashes_valid": True,
        "release_status": manifest["status"],
    }


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def _one_named(paths: list[Path], name: str) -> Path:
    matches = [path for path in paths if path.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} input, found {len(matches)}")
    return matches[0]


def _one_parent_containing(paths: list[Path], name: str) -> Path:
    matches = [path.parent for path in paths if (path.parent / name).is_file()]
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"Expected one input parent containing {name}, found {len(unique)}")
    return unique[0]


def _exact_unique_ids(expected: set[str], observed: list[str], label: str) -> None:
    if len(observed) != len(set(observed)):
        raise ValueError(f"{label} contains duplicate IDs")
    if expected != set(observed):
        raise ValueError(
            f"{label} ID mismatch: missing={sorted(expected - set(observed))} "
            f"extra={sorted(set(observed) - expected)}"
        )


def _read(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
