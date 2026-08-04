from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..io import atomic_write_json, load_json, load_scenes, sha256_file, sha256_json
from ..temporal.benchmark_protocol import BenchmarkRuntimeConfig
from .materialization import (
    materialize_task1_claim_judge,
    materialize_task3_pair_judge,
    materialize_task3_response_judge,
)
from .runner import (
    _artifact_path,
    _read,
    _require_exact_ids,
    _sequence_specs,
    _validate_pair_annotations,
)
from .schemas import (
    TASK3_SCORE_FIELDS,
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


TASK_KINDS = {
    "task1_claim",
    "task1_sequence",
    "task3_response",
    "task3_pair",
}


def build_blind_human_packet(
    *,
    task_release_dir: Path,
    task1_prediction_path: Path,
    task3_prediction_path: Path,
    pair_annotation_path: Path,
    config_path: Path,
    output_dir: Path,
    packet_id: str,
    seed: str,
    task1_per_character: int = 2,
    sequence_per_character: int = 2,
    task3_per_character: int = 4,
    pairs_per_type: int = 2,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite human packet: {output_dir}")
    for value, label in (
        (task1_per_character, "task1_per_character"),
        (sequence_per_character, "sequence_per_character"),
        (task3_per_character, "task3_per_character"),
        (pairs_per_type, "pairs_per_type"),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")

    prepared = _prepare_materialized_inputs(
        task_release_dir=task_release_dir,
        task1_prediction_path=task1_prediction_path,
        task3_prediction_path=task3_prediction_path,
        pair_annotation_path=pair_annotation_path,
        config_path=config_path,
    )
    selected_task1 = _stratified_select(
        prepared["task1_materialized"],
        strata={key: row["character_id"] for key, row in prepared["task1_gold"].items()},
        per_stratum=task1_per_character,
        seed=seed + ":task1",
    )
    selected_sequences = _stratified_select(
        prepared["sequences"],
        strata={key: row["character_id"] for key, row in prepared["sequences"].items()},
        per_stratum=sequence_per_character,
        seed=seed + ":sequence",
    )
    selected_task3 = _stratified_select(
        prepared["task3_materialized"],
        strata={key: row["character_id"] for key, row in prepared["task3_gold"].items()},
        per_stratum=task3_per_character,
        seed=seed + ":task3",
    )
    selected_pairs = _stratified_select(
        prepared["pair_materialized"],
        strata={key: row["pair_type"] for key, row in prepared["pair_annotations"].items()},
        per_stratum=pairs_per_type,
        seed=seed + ":pairs",
    )

    items: list[dict[str, Any]] = []
    for source_id in selected_task1:
        materialized = prepared["task1_materialized"][source_id]
        gold = prepared["task1_gold"][source_id]
        localized = materialized["localized_prediction"]
        gold_ids = [
            row["local_id"]
            for field in ("current_state_claims", "development_claims", "invariant_claims")
            for row in gold["rubric"][field]
        ]
        prediction_ids = [row["local_id"] for row in localized]
        gold_pool = {
            row["local_id"]: ("development" if field == "development_claims" else "state")
            for field in ("current_state_claims", "development_claims", "invariant_claims")
            for row in gold["rubric"][field]
        }
        prediction_pool = {
            row["local_id"]: row["prediction_type"] for row in localized
        }
        allowed_pairs = sorted(
            f"{gold_id}|{prediction_id}"
            for gold_id, pool in gold_pool.items()
            for prediction_id, prediction_type in prediction_pool.items()
            if (pool == "state" and prediction_type == "current_state")
            or (pool == "development" and prediction_type == "development")
        )
        items.append(
            _packet_item(
                task_kind="task1_claim",
                source_id=source_id,
                stratum=gold["character_id"],
                materialized=materialized,
                validation={
                    "gold_ids": sorted(gold_ids),
                    "prediction_ids": sorted(prediction_ids),
                    "allowed_pair_keys": allowed_pairs,
                },
            )
        )
    for source_id in selected_sequences:
        spec = prepared["sequences"][source_id]
        items.append(
            _packet_item(
                task_kind="task1_sequence",
                source_id=source_id,
                stratum=spec["character_id"],
                materialized=spec["materialized"],
                validation={},
            )
        )
    for source_id in selected_task3:
        materialized = prepared["task3_materialized"][source_id]
        gold = prepared["task3_gold"][source_id]
        items.append(
            _packet_item(
                task_kind="task3_response",
                source_id=source_id,
                stratum=gold["character_id"],
                materialized=materialized,
                validation={"allowed_evidence_ids": sorted(materialized["allowed_labels"])},
            )
        )
    for source_id in selected_pairs:
        materialized = prepared["pair_materialized"][source_id]
        annotation = prepared["pair_annotations"][source_id]
        items.append(
            _packet_item(
                task_kind="task3_pair",
                source_id=source_id,
                stratum=annotation["pair_type"],
                materialized=materialized,
                validation={
                    "expected_pair_type": annotation["pair_type"],
                    "responses_by_label": materialized["responses_by_label"],
                },
            )
        )
    items.sort(key=lambda row: _stable_key(seed + ":order", row["item_id"]))

    core = {
        "schema_version": "stage_evaluator_human_packet_v1",
        "packet_id": packet_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "movie_id": prepared["movie_id"],
        "language": prepared["language"],
        "selection": {
            "seed": seed,
            "task1_per_character": task1_per_character,
            "sequence_per_character": sequence_per_character,
            "task3_per_character": task3_per_character,
            "pairs_per_type": pairs_per_type,
            "policy": "deterministic SHA-256 order within character or pair-type stratum",
        },
        "blinding": {
            "actor_model_hidden": True,
            "candidate_judge_outputs_hidden": True,
            "candidate_judge_model_hidden": True,
            "item_order_randomized_deterministically": True,
        },
        "input_hashes": [
            {"role": role, "sha256": sha256_file(path.resolve())}
            for role, path in (
                ("task1_predictions", task1_prediction_path),
                ("task3_predictions", task3_prediction_path),
                ("pair_annotations", pair_annotation_path),
                ("materialization_config", config_path),
            )
        ],
        "counts": {
            "task1_claim": len(selected_task1),
            "task1_sequence": len(selected_sequences),
            "task3_response": len(selected_task3),
            "task3_pair": len(selected_pairs),
            "total": len(items),
        },
        "items": items,
    }
    packet_sha256 = sha256_json(core)
    packet = {**core, "packet_sha256": packet_sha256}
    template = {
        "schema_version": "stage_evaluator_human_annotations_v1",
        "packet_id": packet_id,
        "packet_sha256": packet_sha256,
        "annotator_id": "REPLACE_WITH_PSEUDONYMOUS_ID",
        "completed_at": None,
        "blinding_acknowledgements": {
            "did_not_view_candidate_judgments": False,
            "worked_independently": False,
        },
        "items": [
            {
                "item_id": row["item_id"],
                "source_id": row["source_id"],
                "task_kind": row["task_kind"],
                "judgment": None,
            }
            for row in items
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "blind_items.json"
    atomic_write_json(packet_path, packet)
    atomic_write_json(output_dir / "annotation_template.json", template)
    (output_dir / "README.md").write_text(
        _packet_readme(packet_id=packet_id, counts=core["counts"]),
        encoding="utf-8",
    )
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "stage_evaluator_human_packet_manifest_v1",
            "status": "awaiting_human_annotations",
            "packet_id": packet_id,
            "packet_sha256": packet_sha256,
            "human_annotation_count": 0,
            "input_hashes": core["input_hashes"],
            "artifacts": [
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path.name != "manifest.json"
            ],
        },
    )
    return packet_path


def score_human_annotations(
    *,
    packet_path: Path,
    annotation_paths: list[Path],
    candidate_evaluation_dir: Path,
    policy_path: Path,
    output_path: Path,
) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite human calibration report: {output_path}")
    packet = _read(packet_path.resolve())
    validate_human_packet(packet)
    policy = _read(policy_path.resolve())
    if len(annotation_paths) < int(policy["minimum_annotators"]):
        raise ValueError(
            f"Human calibration requires at least {policy['minimum_annotators']} annotations"
        )
    annotations = [
        validate_human_annotation(packet=packet, payload=_read(path.resolve()))
        for path in annotation_paths
    ]
    annotator_ids = [row["annotator_id"] for row in annotations]
    if len(annotator_ids) != len(set(annotator_ids)):
        raise ValueError("Human annotation files contain duplicate annotator IDs")

    measurements = {
        row["annotator_id"]: _flatten_measurements(packet, row["items"])
        for row in annotations
    }
    human_pairwise = _pairwise_reports(measurements)
    consensus, unresolved = _consensus_measurements(measurements)
    candidate = _candidate_measurements(packet, candidate_evaluation_dir.resolve())
    judge_vs_consensus = _comparison_report(consensus, candidate)
    dimension_correlations = _dimension_correlation_report(consensus)

    checks = []
    for family, threshold in policy["minimum_human_pairwise_kappa"].items():
        values = [
            row["families"][family]["kappa"]
            for row in human_pairwise
            if family in row["families"] and row["families"][family]["kappa"] is not None
        ]
        observed = [
            row["families"][family]["observed_agreement"]
            for row in human_pairwise
            if family in row["families"]
        ]
        minimum = min(values) if values else None
        fallback = min(observed) if observed else None
        passed = bool(
            (minimum is not None and minimum >= float(threshold))
            or (
                minimum is None
                and fallback is not None
                and fallback >= float(policy["constant_label_minimum_agreement"])
            )
        )
        checks.append(
            {
                "check": f"human_pairwise:{family}",
                "threshold": threshold,
                "observed_minimum_kappa": minimum,
                "observed_minimum_agreement": fallback,
                "passed": passed,
            }
        )
    for family, threshold in policy["minimum_judge_consensus_kappa"].items():
        row = judge_vs_consensus["families"].get(family, {})
        kappa = row.get("kappa")
        agreement = row.get("observed_agreement")
        passed = bool(
            (kappa is not None and kappa >= float(threshold))
            or (
                kappa is None
                and agreement is not None
                and agreement >= float(policy["constant_label_minimum_agreement"])
            )
        )
        checks.append(
            {
                "check": f"judge_consensus:{family}",
                "threshold": threshold,
                "observed_kappa": kappa,
                "observed_agreement": agreement,
                "passed": passed,
            }
        )
    for dimension, maximum in policy["maximum_judge_consensus_likert_mae"].items():
        family = f"task3_score:{dimension}"
        row = judge_vs_consensus["families"].get(family, {})
        mae = row.get("mae")
        checks.append(
            {
                "check": f"judge_consensus_mae:{dimension}",
                "maximum": maximum,
                "observed": mae,
                "passed": mae is not None and mae <= float(maximum),
            }
        )
    checks.append(
        {
            "check": "consensus_coverage",
            "maximum_unresolved": int(policy["maximum_unresolved_consensus_items"]),
            "observed": len(unresolved),
            "passed": len(unresolved) <= int(policy["maximum_unresolved_consensus_items"]),
        }
    )
    report = {
        "schema_version": "stage_evaluator_human_calibration_report_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "packet": {
            "path": str(packet_path.resolve()),
            "packet_id": packet["packet_id"],
            "packet_sha256": packet["packet_sha256"],
            "item_count": packet["counts"]["total"],
        },
        "policy": {"path": str(policy_path.resolve()), "sha256": sha256_file(policy_path.resolve())},
        "candidate_evaluation": {
            "path": str(candidate_evaluation_dir.resolve()),
            "manifest_sha256": sha256_file(candidate_evaluation_dir.resolve() / "manifest.json"),
        },
        "human_annotation_count": len(annotations),
        "annotator_ids": sorted(annotator_ids),
        "annotation_files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve()),
                "annotator_id": annotation["annotator_id"],
            }
            for path, annotation in zip(annotation_paths, annotations, strict=True)
        ],
        "human_pairwise": human_pairwise,
        "judge_vs_human_consensus": judge_vs_consensus,
        "unresolved_consensus_measurements": unresolved,
        "task3_dimension_spearman": dimension_correlations,
        "gate_checks": checks,
    }
    atomic_write_json(output_path, report)
    return output_path


def build_manual_annotation(
    *,
    packet_path: Path,
    template_path: Path,
    decisions_path: Path,
    output_path: Path,
) -> Path:
    """Expand explicit reviewer decisions into the strict annotation schema."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite manual annotation: {output_path}")
    packet = validate_human_packet(_read(packet_path.resolve()))
    template = _read(template_path.resolve())
    decisions = _read(decisions_path.resolve())
    if decisions.get("schema_version") != "stage_evaluator_manual_review_decisions_v1":
        raise ValueError("Unknown manual review decision schema")
    required = {"schema_version", "annotator_id", "completed_at", "items"}
    if set(decisions) != required:
        raise ValueError("Manual review decision top-level keys mismatch")
    decision_items = decisions["items"]
    if not isinstance(decision_items, dict):
        raise ValueError("Manual review decisions items must be an object")
    specs = {row["item_id"]: row for row in packet["items"]}
    if set(decision_items) != set(specs):
        raise ValueError("Manual review decisions must cover every packet item exactly once")

    output_items = []
    for template_row in template["items"]:
        item_id = template_row["item_id"]
        spec = specs[item_id]
        decision = decision_items[item_id]
        task_kind = spec["task_kind"]
        if task_kind == "task1_claim":
            expected = {
                "claim_pair_judgments",
                "default_prediction_check",
                "prediction_check_overrides",
                "future_leak_prediction_ids",
                "premature_update_prediction_ids",
                "no_change_false_update",
            }
            if set(decision) != expected:
                raise ValueError(f"Manual Task 1 decision keys mismatch: {item_id}")
            prediction_ids = spec["validation"]["prediction_ids"]
            overrides = decision["prediction_check_overrides"]
            if not isinstance(overrides, dict) or not set(overrides) <= set(prediction_ids):
                raise ValueError(f"Manual Task 1 overrides contain unknown IDs: {item_id}")
            checks = []
            for prediction_id in prediction_ids:
                values = dict(decision["default_prediction_check"])
                values.update(overrides.get(prediction_id, {}))
                checks.append({"prediction_local_id": prediction_id, **values})
            judgment = {
                "claim_pair_judgments": decision["claim_pair_judgments"],
                "prediction_checks": checks,
                "future_leak_prediction_ids": decision["future_leak_prediction_ids"],
                "premature_update_prediction_ids": decision["premature_update_prediction_ids"],
                "no_change_false_update": decision["no_change_false_update"],
            }
        elif task_kind in {"task1_sequence", "task3_response"}:
            judgment = decision
        elif task_kind == "task3_pair":
            expected = {
                "response_support",
                "observed_behavior",
                "expected_direction_present",
                "unsupported_drift",
                "knowledge_boundaries_preserved",
                "local_evidence_labels",
                "brief_rationale",
            }
            if set(decision) != expected:
                raise ValueError(f"Manual Task 3 pair decision keys mismatch: {item_id}")
            paired = _prompt_tag_json(spec["prompt"]["user"], "paired_responses")
            if len(paired) != 2:
                raise ValueError(f"Manual Task 3 pair must contain two source responses: {item_id}")
            pair_type = _prompt_tag_text(spec["prompt"]["user"], "pair_type")
            judgment = {
                "pair_type": pair_type,
                "response_assessments": [
                    {
                        "response_label": label,
                        "response_excerpt": source["response"],
                        "observed_behavior": decision["observed_behavior"][label],
                        "supports_expected_component": decision["response_support"][label],
                    }
                    for label, source in zip(("T1", "T2"), paired, strict=True)
                ],
                "expected_direction_present": decision["expected_direction_present"],
                "unsupported_drift": decision["unsupported_drift"],
                "knowledge_boundaries_preserved": decision["knowledge_boundaries_preserved"],
                "local_evidence_labels": decision["local_evidence_labels"],
                "brief_rationale": decision["brief_rationale"],
            }
        else:  # pragma: no cover - packet validation closes the task-kind enum
            raise ValueError(f"Unsupported manual review task kind: {task_kind}")
        output_items.append({**template_row, "judgment": judgment})

    payload = {
        **template,
        "annotator_id": decisions["annotator_id"],
        "completed_at": decisions["completed_at"],
        "blinding_acknowledgements": {
            "did_not_view_candidate_judgments": True,
            "worked_independently": True,
        },
        "items": output_items,
    }
    validated = validate_human_annotation(packet=packet, payload=payload)
    atomic_write_json(output_path, {**payload, "items": validated["items"]})
    return output_path


def audit_single_manual_annotation(
    *,
    packet_path: Path,
    annotation_path: Path,
    candidate_evaluation_dir: Path,
    policy_path: Path,
    output_path: Path,
) -> Path:
    """Compare one disclosed manual reviewer with a candidate judge diagnostically."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite manual audit report: {output_path}")
    packet = validate_human_packet(_read(packet_path.resolve()))
    annotation = validate_human_annotation(
        packet=packet, payload=_read(annotation_path.resolve())
    )
    policy = _read(policy_path.resolve())
    manual = _flatten_measurements(packet, annotation["items"])
    candidate = _candidate_measurements(packet, candidate_evaluation_dir.resolve())
    comparison = _comparison_report(manual, candidate)
    common = sorted(set(manual) & set(candidate))
    exact = sum(manual[key] == candidate[key] for key in common)
    diagnostics = []
    for family, threshold in policy["minimum_judge_consensus_kappa"].items():
        row = comparison["families"].get(family, {})
        kappa = row.get("kappa")
        agreement = row.get("observed_agreement")
        passed = bool(
            (kappa is not None and kappa >= float(threshold))
            or (
                kappa is None
                and agreement is not None
                and agreement >= float(policy["constant_label_minimum_agreement"])
            )
        )
        diagnostics.append(
            {
                "check": f"candidate_vs_single_manual:{family}",
                "threshold": threshold,
                "observed_kappa": kappa,
                "observed_agreement": agreement,
                "passed": passed,
            }
        )
    for dimension, maximum in policy["maximum_judge_consensus_likert_mae"].items():
        row = comparison["families"].get(f"task3_score:{dimension}", {})
        mae = row.get("mae")
        diagnostics.append(
            {
                "check": f"candidate_vs_single_manual_mae:{dimension}",
                "maximum": maximum,
                "observed": mae,
                "passed": mae is not None and mae <= float(maximum),
            }
        )
    report = {
        "schema_version": "stage_evaluator_single_manual_audit_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "diagnostic_completed",
        "formal_gate_eligible": False,
        "reviewer_classification": "codex_manual_single_reviewer",
        "annotator_id": annotation["annotator_id"],
        "packet": {
            "path": str(packet_path.resolve()),
            "packet_id": packet["packet_id"],
            "packet_sha256": packet["packet_sha256"],
            "item_count": packet["counts"]["total"],
        },
        "annotation": {
            "path": str(annotation_path.resolve()),
            "sha256": sha256_file(annotation_path.resolve()),
        },
        "candidate_evaluation": {
            "path": str(candidate_evaluation_dir.resolve()),
            "manifest_sha256": sha256_file(candidate_evaluation_dir.resolve() / "manifest.json"),
        },
        "policy_reference": {
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path.resolve()),
        },
        "measurement_count": len(common),
        "exact_measurement_agreement": exact / len(common) if common else None,
        "candidate_vs_single_manual": comparison,
        "diagnostic_checks": diagnostics,
        "diagnostic_checks_passed": sum(row["passed"] for row in diagnostics),
        "diagnostic_checks_total": len(diagnostics),
        "limitation": "One Codex manual reviewer is not three independent human annotators and cannot satisfy the formal human calibration gate.",
    }
    atomic_write_json(output_path, report)
    return output_path


def build_human_consensus_evaluation(
    *,
    packet_path: Path,
    annotation_paths: list[Path],
    calibration_report_path: Path,
    structural_evaluation_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite human consensus evaluation: {output_dir}")
    packet = validate_human_packet(_read(packet_path.resolve()))
    report = _read(calibration_report_path.resolve())
    if report.get("schema_version") != "stage_evaluator_human_calibration_report_v1":
        raise ValueError("Unknown human calibration report schema")
    if report.get("status") != "passed":
        raise ValueError("Human consensus evaluation requires a passed calibration report")
    if report.get("packet", {}).get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("Human calibration report belongs to a different packet")
    annotations = [
        validate_human_annotation(packet=packet, payload=_read(path.resolve()))
        for path in annotation_paths
    ]
    expected_hashes = {
        row["annotator_id"]: row["sha256"] for row in report["annotation_files"]
    }
    if len(annotations) < 3 or {
        row["annotator_id"]: sha256_file(path.resolve())
        for row, path in zip(annotations, annotation_paths, strict=True)
    } != expected_hashes:
        raise ValueError("Human annotations do not match the passed calibration report")
    measurements = {
        row["annotator_id"]: _flatten_measurements(packet, row["items"])
        for row in annotations
    }
    consensus, unresolved = _consensus_measurements(measurements)
    if unresolved:
        raise ValueError("Human consensus evaluation has unresolved measurements")

    structural_evaluation_dir = structural_evaluation_dir.resolve()
    structural_manifest = _read(structural_evaluation_dir / "manifest.json")
    structural_contract = _read(Path(structural_manifest["run_contract"]["path"]))
    structural_task1 = _read(structural_evaluation_dir / "task1_evaluation.json")
    structural_task3 = _read(structural_evaluation_dir / "task3_evaluation.json")
    expected_counts = {
        "task1_claim": len(structural_task1["instances"]),
        "task1_sequence": len(structural_task1["sequences"]),
        "task3_response": len(structural_task3["instances"]),
        "task3_pair": len(structural_task3["pairs"]),
        "total": len(structural_task1["instances"])
        + len(structural_task1["sequences"])
        + len(structural_task3["instances"])
        + len(structural_task3["pairs"]),
    }
    if packet["counts"] != expected_counts:
        raise ValueError("Formal human packet does not cover the complete evaluation run")
    item_by_kind_and_source = {
        (row["task_kind"], row["source_id"]): row for row in packet["items"]
    }
    expected_sources = {
        "task1_claim": {row["instance_id"] for row in structural_task1["instances"]},
        "task1_sequence": {row["sequence_id"] for row in structural_task1["sequences"]},
        "task3_response": {row["instance_id"] for row in structural_task3["instances"]},
        "task3_pair": {row["pair_group_id"] for row in structural_task3["pairs"]},
    }
    for kind, source_ids in expected_sources.items():
        observed = {source for item_kind, source in item_by_kind_and_source if item_kind == kind}
        if observed != source_ids:
            raise ValueError(f"Formal human packet source coverage drift: {kind}")

    gold_dir = _gold_dir_from_contract(structural_contract)
    task1_gold = {
        row["instance_id"]: row
        for row in _read(gold_dir / "task1_gold_rubrics.json")["rubrics"]
    }
    task1_results = []
    for source in structural_task1["instances"]:
        item = item_by_kind_and_source[("task1_claim", source["instance_id"])]
        judgment = _consensus_judgment(item, consensus)
        scoring = score_task1_instance(
            prediction=source["localized_prediction"],
            rubric=task1_gold[source["instance_id"]]["rubric"],
            judgment=judgment,
        )
        task1_results.append(
            {
                key: value
                for key, value in source.items()
                if key not in {"judgment", "scoring", "generator_metadata", "seed_reused_from"}
            }
            | {"judgment": judgment, "scoring": scoring, "human_consensus": True}
        )
    sequence_results = []
    for source in structural_task1["sequences"]:
        item = item_by_kind_and_source[("task1_sequence", source["sequence_id"])]
        judgment = _consensus_judgment(item, consensus)
        consistent = bool(
            judgment["state_carry_forward"]
            and judgment["development_to_state_coherent"]
            and not judgment["contradiction_present"]
            and not judgment["premature_or_future_information"]
        )
        sequence_results.append(
            {
                key: value
                for key, value in source.items()
                if key not in {"judgment", "consistent", "generator_metadata", "seed_reused_from"}
            }
            | {"judgment": judgment, "consistent": consistent, "human_consensus": True}
        )
    task3_results = []
    for source in structural_task3["instances"]:
        item = item_by_kind_and_source[("task3_response", source["instance_id"])]
        judgment = _consensus_judgment(item, consensus)
        task3_results.append(
            {
                key: value
                for key, value in source.items()
                if key not in {"judgment", "generator_metadata", "seed_reused_from"}
            }
            | {"judgment": judgment, "human_consensus": True}
        )
    task3_by_id = {row["instance_id"]: row for row in task3_results}
    pair_results = []
    for source in structural_task3["pairs"]:
        item = item_by_kind_and_source[("task3_pair", source["pair_group_id"])]
        judgment = _consensus_judgment(item, consensus)
        prerequisites = []
        for instance_id in source["instance_ids"]:
            response = task3_by_id[instance_id]["judgment"]
            prerequisites.append(
                {
                    "instance_id": instance_id,
                    "stance_compatible": response["stance_compatible"],
                    "future_leakage": response["future_leakage"],
                    "unknown_fact_hallucination": response["unknown_fact_hallucination"],
                    "passed": bool(
                        response["stance_compatible"]
                        and not response["future_leakage"]
                        and not response["unknown_fact_hallucination"]
                    ),
                }
            )
        pair_results.append(
            {
                key: value
                for key, value in source.items()
                if key
                not in {
                    "judgment",
                    "response_prerequisites",
                    "generator_metadata",
                    "seed_reused_from",
                }
            }
            | {
                "judgment": judgment,
                "response_prerequisites": prerequisites,
                "human_consensus": True,
            }
        )

    task1_aggregate = aggregate_checkpoint_task1(task1_results)
    task1_aggregate["longitudinal_consistency"] = aggregate_task1_sequences(sequence_results)
    task1_aggregate["delayed_update"] = {
        "eligible_developments": 0,
        "value": None,
        "reason": "missing_explicit_gold_development_lineage",
    }
    task3_aggregate = aggregate_task3(task3_results, pair_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    task1_path = output_dir / "task1_evaluation.json"
    task3_path = output_dir / "task3_evaluation.json"
    atomic_write_json(
        task1_path,
        {
            "schema_version": "stage_task1_evaluation_v1",
            "movie_id": packet["movie_id"],
            "instance_count": len(task1_results),
            "sequence_count": len(sequence_results),
            "aggregate": task1_aggregate,
            "instances": task1_results,
            "sequences": sequence_results,
        },
    )
    atomic_write_json(
        task3_path,
        {
            "schema_version": "stage_task3_evaluation_v1",
            "movie_id": packet["movie_id"],
            "instance_count": len(task3_results),
            "pair_count": len(pair_results),
            "aggregate": task3_aggregate,
            "instances": task3_results,
            "pairs": pair_results,
        },
    )
    annotator_ids = sorted(row["annotator_id"] for row in annotations)
    model_contract = {
        "actor_models": structural_manifest["model_contract"]["actor_models"],
        "judge_type": "blinded_human_consensus",
        "judge_model": None,
        "independent": True,
        "annotator_count": len(annotator_ids),
        "annotator_ids": annotator_ids,
        "consensus_policy_sha256": report["policy"]["sha256"],
        "semantic_samples_per_call": 0,
    }
    contract = {
        **structural_contract,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_mode": "formal_blinded_human_evaluation",
        "model_contract": model_contract,
        "inputs": [
            *structural_contract["inputs"],
            {"path": str(packet_path.resolve()), "sha256": sha256_file(packet_path.resolve())},
            {
                "path": str(calibration_report_path.resolve()),
                "sha256": sha256_file(calibration_report_path.resolve()),
            },
            *[
                {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
                for path in annotation_paths
            ],
        ],
        "seed_evaluation": None,
        "human_consensus": {
            "packet_id": packet["packet_id"],
            "packet_sha256": packet["packet_sha256"],
            "calibration_report_sha256": sha256_file(calibration_report_path.resolve()),
            "annotator_count": len(annotator_ids),
            "unresolved_measurements": 0,
            "rationale_policy": "metrics use field consensus; generated consensus rationale is non-evidence",
        },
    }
    contract_path = output_dir / "run_contract.json"
    atomic_write_json(contract_path, contract)
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task1_task3_evaluation_run_v1",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "completed",
            "evaluation_mode": "formal_blinded_human_evaluation",
            "movie_id": packet["movie_id"],
            "python_executable": sys.executable,
            "model_contract": model_contract,
            "run_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            "endpoint_pool": None,
            "counts": {
                "expected_task1": len(task1_results),
                "evaluated_task1": len(task1_results),
                "expected_task1_sequences": len(sequence_results),
                "evaluated_task1_sequences": len(sequence_results),
                "expected_task3": len(task3_results),
                "evaluated_task3": len(task3_results),
                "expected_task3_pairs": len(pair_results),
                "evaluated_task3_pairs": len(pair_results),
                "failure_count": 0,
                "truncated_count": 0,
                "seed_reused_task1": 0,
                "seed_reused_task1_sequences": 0,
                "seed_reused_task3": 0,
                "seed_reused_task3_pairs": 0,
            },
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in (task1_path, task3_path)
            ],
        },
    )
    return manifest_path


def validate_human_annotation(
    *, packet: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    validate_human_packet(packet)
    required = {
        "schema_version",
        "packet_id",
        "packet_sha256",
        "annotator_id",
        "completed_at",
        "blinding_acknowledgements",
        "items",
    }
    if set(payload) != required:
        raise ValueError("Human annotation top-level keys mismatch")
    if payload["schema_version"] != "stage_evaluator_human_annotations_v1":
        raise ValueError("Unknown human annotation schema version")
    if payload["packet_id"] != packet["packet_id"] or payload["packet_sha256"] != packet["packet_sha256"]:
        raise ValueError("Human annotation packet identity/hash mismatch")
    annotator_id = str(payload["annotator_id"]).strip()
    if not annotator_id or annotator_id == "REPLACE_WITH_PSEUDONYMOUS_ID":
        raise ValueError("Human annotation requires a pseudonymous annotator ID")
    if not isinstance(payload["completed_at"], str) or not payload["completed_at"].strip():
        raise ValueError("Human annotation requires completed_at")
    acknowledgements = payload["blinding_acknowledgements"]
    if acknowledgements != {
        "did_not_view_candidate_judgments": True,
        "worked_independently": True,
    }:
        raise ValueError("Human annotation requires both blinding acknowledgements")
    expected = {row["item_id"]: row for row in packet["items"]}
    observed_rows = payload["items"]
    if not isinstance(observed_rows, list):
        raise ValueError("Human annotation items must be an array")
    observed = {row.get("item_id"): row for row in observed_rows if isinstance(row, dict)}
    if len(observed) != len(observed_rows) or set(observed) != set(expected):
        raise ValueError("Human annotation must cover each packet item exactly once")
    normalized = []
    for item_id in sorted(expected):
        spec = expected[item_id]
        row = observed[item_id]
        if set(row) != {"item_id", "source_id", "task_kind", "judgment"}:
            raise ValueError(f"Human annotation item keys mismatch: {item_id}")
        if row["source_id"] != spec["source_id"] or row["task_kind"] != spec["task_kind"]:
            raise ValueError(f"Human annotation item identity drift: {item_id}")
        judgment = row["judgment"]
        if not isinstance(judgment, dict):
            raise ValueError(f"Human annotation judgment is incomplete: {item_id}")
        validation = spec["validation"]
        if spec["task_kind"] == "task1_claim":
            judgment = validate_task1_judgment(
                judgment,
                gold_ids=set(validation["gold_ids"]),
                prediction_ids=set(validation["prediction_ids"]),
            )
            allowed = set(validation["allowed_pair_keys"])
            if any(
                f"{row['gold_local_id']}|{row['prediction_local_id']}" not in allowed
                for row in judgment["claim_pair_judgments"]
            ):
                raise ValueError(f"Human Task 1 pair crosses a scoring pool: {item_id}")
        elif spec["task_kind"] == "task1_sequence":
            judgment = validate_task1_sequence_judgment(judgment)
        elif spec["task_kind"] == "task3_response":
            judgment = validate_task3_response_judgment(
                judgment, allowed_evidence_ids=set(validation["allowed_evidence_ids"])
            )
        elif spec["task_kind"] == "task3_pair":
            judgment = validate_task3_pair_judgment(
                judgment, expected_pair_type=validation["expected_pair_type"]
            )
            labels = {row["response_label"] for row in judgment["response_assessments"]}
            if set(judgment["local_evidence_labels"]) - labels:
                raise ValueError(f"Human pair evidence label is unknown: {item_id}")
            for assessment in judgment["response_assessments"]:
                response = validation["responses_by_label"][assessment["response_label"]]
                if _normalize_text(assessment["response_excerpt"]) not in _normalize_text(response):
                    raise ValueError(f"Human pair excerpt is not verbatim: {item_id}")
        else:  # pragma: no cover - packet construction prevents this
            raise ValueError(f"Unknown task kind: {spec['task_kind']}")
        normalized.append({**row, "judgment": judgment})
    return {**payload, "annotator_id": annotator_id, "items": normalized}


def validate_human_packet(packet: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "packet_id",
        "created_at",
        "movie_id",
        "language",
        "selection",
        "blinding",
        "input_hashes",
        "counts",
        "items",
        "packet_sha256",
    }
    if set(packet) != required:
        raise ValueError("Human packet top-level keys mismatch")
    if packet["schema_version"] != "stage_evaluator_human_packet_v1":
        raise ValueError("Unknown human packet schema version")
    expected_hash = str(packet["packet_sha256"])
    observed_hash = sha256_json(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    )
    if observed_hash != expected_hash:
        raise ValueError("Human packet SHA-256 mismatch")
    items = packet["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("Human packet items must be a nonempty array")
    item_ids = [row.get("item_id") for row in items if isinstance(row, dict)]
    if len(item_ids) != len(items) or len(item_ids) != len(set(item_ids)):
        raise ValueError("Human packet item IDs must be unique")
    observed_counts = {kind: 0 for kind in sorted(TASK_KINDS)}
    for row in items:
        kind = row.get("task_kind")
        source_id = row.get("source_id")
        if kind not in TASK_KINDS or row.get("item_id") != f"{kind}:{source_id}":
            raise ValueError("Human packet item identity is invalid")
        observed_counts[kind] += 1
    expected_counts = {**observed_counts, "total": len(items)}
    if packet["counts"] != expected_counts:
        raise ValueError("Human packet counts do not match items")
    return packet


def _prepare_materialized_inputs(
    *,
    task_release_dir: Path,
    task1_prediction_path: Path,
    task3_prediction_path: Path,
    pair_annotation_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    task_release_dir = task_release_dir.resolve()
    release_manifest = _read(task_release_dir / "standard24k_benchmark_release_manifest.json")
    artifacts = release_manifest["artifacts"]
    protocol_manifest_path = _artifact_path(artifacts["protocol_manifest"])
    protocol_dir = protocol_manifest_path.parent
    gold_dir = _artifact_path(artifacts["gold_manifest"]).parent
    config = BenchmarkRuntimeConfig.load(config_path)
    counter = config.build_token_counter()
    task1_gold_payload = _read(gold_dir / "task1_gold_rubrics.json")
    task3_gold_payload = _read(gold_dir / "task3_gold_rubrics.json")
    task1_predictions = _read(task1_prediction_path.resolve())
    task3_predictions = _read(task3_prediction_path.resolve())
    task1_plan = _read(protocol_dir / "task1" / "task1_rolling_plans.json")
    context_packs = _read(protocol_dir / "task3" / "task3_actor_context_packs.json")
    anchored_task3 = _read(protocol_dir / "task3" / "task3_checkpoint_single_turn.anchored.json")
    protocol_manifest = _read(protocol_manifest_path)
    evidence_bank = _read(
        Path(protocol_manifest["temporal_run_dir"]) / "assets" / "source" / "evidence_units.json"
    )
    pair_annotations_payload = _read(pair_annotation_path.resolve())
    source_pairs = _read(task_release_dir / "task_3_pair_groups.json")
    task1_gold = {row["instance_id"]: row for row in task1_gold_payload["rubrics"]}
    task3_gold = {row["instance_id"]: row for row in task3_gold_payload["rubrics"]}
    task1_prediction_by_id = {
        row["instance_id"]: {**row, "character_id": character["character_id"]}
        for character in task1_predictions["characters"]
        for row in character["checkpoint_predictions"]
    }
    task3_prediction_by_id = {row["instance_id"]: row for row in task3_predictions["predictions"]}
    context_by_id = {row["instance_id"]: row for row in context_packs["context_packs"]}
    instance_by_id = {row["instance_id"]: row for row in anchored_task3["instances"]}
    _require_exact_ids(task1_gold, task1_prediction_by_id, "Human packet Task 1 predictions")
    _require_exact_ids(task3_gold, task3_prediction_by_id, "Human packet Task 3 predictions")
    pair_annotations = _validate_pair_annotations(
        pair_annotations_payload, source_pairs=source_pairs, instances=instance_by_id
    )
    scenes = {scene.order: scene for scene in load_scenes(Path(task1_plan["script_path"]))}
    aliases_by_character = {
        row["character_id"]: [row["focal_character"], *row.get("aliases", [])]
        for row in task1_plan["plans"]
    }
    language = str(context_packs["language"])
    task1_materialized = {
        instance_id: materialize_task1_claim_judge(
            gold=gold,
            prediction=task1_prediction_by_id[instance_id],
            scenes=scenes,
            evidence_bank=evidence_bank,
            aliases=aliases_by_character[gold["character_id"]],
            language=language,
            config=config,
            counter=counter,
        )
        for instance_id, gold in task1_gold.items()
    }
    task3_materialized = {
        instance_id: materialize_task3_response_judge(
            gold=gold,
            prediction=task3_prediction_by_id[instance_id],
            context_pack=context_by_id[instance_id],
            instance=instance_by_id[instance_id],
            language=language,
            config=config,
            counter=counter,
        )
        for instance_id, gold in task3_gold.items()
    }
    pair_materialized = {
        pair_id: materialize_task3_pair_judge(
            annotation=annotation,
            predictions=task3_prediction_by_id,
            gold=task3_gold,
            instances=instance_by_id,
            language=language,
            config=config,
            counter=counter,
        )
        for pair_id, annotation in pair_annotations.items()
    }
    sequences = _sequence_specs(
        task1_gold_by_id=task1_gold,
        task1_predictions=task1_prediction_by_id,
        aliases_by_character=aliases_by_character,
        scenes=scenes,
        evidence_bank=evidence_bank,
        language=language,
        config=config,
        counter=counter,
    )
    return {
        "movie_id": release_manifest["movie_id"],
        "language": language,
        "task1_gold": task1_gold,
        "task3_gold": task3_gold,
        "task1_materialized": task1_materialized,
        "task3_materialized": task3_materialized,
        "pair_materialized": pair_materialized,
        "pair_annotations": pair_annotations,
        "sequences": sequences,
    }


def _packet_item(
    *,
    task_kind: str,
    source_id: str,
    stratum: str,
    materialized: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "item_id": f"{task_kind}:{source_id}",
        "source_id": source_id,
        "task_kind": task_kind,
        "stratum": stratum,
        "prompt": {
            "system": materialized["system_prompt"],
            "user": materialized["user_prompt"],
            "sha256": materialized["prompt_sha256"],
            "prompt_tokens": materialized["prompt_tokens"],
            "max_input_tokens": materialized["max_input_tokens"],
        },
        "validation": validation,
    }


def _stratified_select(
    values: dict[str, Any], *, strata: dict[str, str], per_stratum: int, seed: str
) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for identifier in values:
        grouped[strata[identifier]].append(identifier)
    output = []
    for stratum in sorted(grouped):
        ranked = sorted(grouped[stratum], key=lambda value: _stable_key(seed, value))
        output.extend(ranked[: min(per_stratum, len(ranked))])
    return output


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _prompt_tag_text(prompt: str, name: str) -> str:
    opening = f"<{name}>"
    closing = f"</{name}>"
    start = prompt.find(opening)
    end = prompt.find(closing, start + len(opening))
    if start < 0 or end < 0:
        raise ValueError(f"Prompt is missing required tag: {name}")
    return prompt[start + len(opening) : end]


def _prompt_tag_json(prompt: str, name: str) -> Any:
    return json.loads(_prompt_tag_text(prompt, name))


def _flatten_measurements(
    packet: dict[str, Any], annotation_items: list[dict[str, Any]]
) -> dict[tuple[str, str, str], Any]:
    specs = {row["item_id"]: row for row in packet["items"]}
    output: dict[tuple[str, str, str], Any] = {}
    for row in annotation_items:
        item_id = row["item_id"]
        spec = specs[item_id]
        judgment = row["judgment"]
        task_kind = spec["task_kind"]
        if task_kind == "task1_claim":
            labels = {
                f"{item['gold_local_id']}|{item['prediction_local_id']}": item["label"]
                for item in judgment["claim_pair_judgments"]
            }
            for pair_key in spec["validation"]["allowed_pair_keys"]:
                output[("task1_claim_label", item_id, pair_key)] = labels.get(pair_key, "none")
            checks = {item["prediction_local_id"]: item for item in judgment["prediction_checks"]}
            future = set(judgment["future_leak_prediction_ids"])
            premature = set(judgment["premature_update_prediction_ids"])
            for prediction_id in spec["validation"]["prediction_ids"]:
                output[("task1_support", item_id, prediction_id)] = checks[prediction_id]["support"]
                output[("task1_transition", item_id, prediction_id)] = checks[prediction_id]["transition_coherent"]
                output[("task1_evidence", item_id, prediction_id)] = checks[prediction_id]["evidence_grounded"]
                output[("task1_future", item_id, prediction_id)] = prediction_id in future
                output[("task1_premature", item_id, prediction_id)] = prediction_id in premature
            output[("task1_no_change", item_id, "value")] = judgment["no_change_false_update"]
        elif task_kind == "task1_sequence":
            for field in (
                "state_carry_forward",
                "development_to_state_coherent",
                "contradiction_present",
                "premature_or_future_information",
            ):
                output[("task1_sequence", item_id, field)] = judgment[field]
        elif task_kind == "task3_response":
            for field in TASK3_SCORE_FIELDS:
                output[(f"task3_score:{field}", item_id, "value")] = judgment["scores"][field]
            for field in ("future_leakage", "unknown_fact_hallucination", "stance_compatible"):
                output[("task3_response_flags", item_id, field)] = judgment[field]
        elif task_kind == "task3_pair":
            for assessment in judgment["response_assessments"]:
                output[("task3_pair_flags", item_id, f"supports:{assessment['response_label']}")] = assessment[
                    "supports_expected_component"
                ]
            for field in (
                "expected_direction_present",
                "unsupported_drift",
                "knowledge_boundaries_preserved",
            ):
                output[("task3_pair_flags", item_id, field)] = judgment[field]
    return output


def _candidate_measurements(packet: dict[str, Any], run_dir: Path) -> dict[tuple[str, str, str], Any]:
    task1 = _read(run_dir / "task1_evaluation.json")
    task3 = _read(run_dir / "task3_evaluation.json")
    source = {
        "task1_claim": {row["instance_id"]: row["judgment"] for row in task1["instances"]},
        "task1_sequence": {row["sequence_id"]: row["judgment"] for row in task1["sequences"]},
        "task3_response": {row["instance_id"]: row["judgment"] for row in task3["instances"]},
        "task3_pair": {row["pair_group_id"]: row["judgment"] for row in task3["pairs"]},
    }
    rows = [
        {
            "item_id": item["item_id"],
            "source_id": item["source_id"],
            "task_kind": item["task_kind"],
            "judgment": source[item["task_kind"]][item["source_id"]],
        }
        for item in packet["items"]
    ]
    return _flatten_measurements(packet, rows)


def _pairwise_reports(
    measurements: dict[str, dict[tuple[str, str, str], Any]]
) -> list[dict[str, Any]]:
    output = []
    ids = sorted(measurements)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            output.append(
                {
                    "annotators": [left, right],
                    "families": _comparison_report(measurements[left], measurements[right])["families"],
                }
            )
    return output


def _comparison_report(
    left: dict[tuple[str, str, str], Any],
    right: dict[tuple[str, str, str], Any],
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    grouped: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for key in common:
        grouped[key[0]].append((left[key], right[key]))
    families = {}
    for family, pairs in sorted(grouped.items()):
        a = [row[0] for row in pairs]
        b = [row[1] for row in pairs]
        if family.startswith("task3_score:"):
            families[family] = {
                "count": len(pairs),
                "observed_agreement": _observed_agreement(a, b),
                "kappa": _cohen_kappa(a, b, ordinal_values=range(1, 6)),
                "mae": sum(abs(float(x) - float(y)) for x, y in pairs) / len(pairs),
                "spearman": _spearman(a, b),
            }
        else:
            families[family] = {
                "count": len(pairs),
                "observed_agreement": _observed_agreement(a, b),
                "kappa": _cohen_kappa(a, b),
            }
    return {"families": families}


def _consensus_measurements(
    measurements: dict[str, dict[tuple[str, str, str], Any]]
) -> tuple[dict[tuple[str, str, str], Any], list[dict[str, Any]]]:
    keys = sorted(set.intersection(*(set(row) for row in measurements.values())))
    consensus = {}
    unresolved = []
    for key in keys:
        values = [row[key] for row in measurements.values()]
        if key[0].startswith("task3_score:"):
            consensus[key] = int(statistics.median(values))
            continue
        counts = defaultdict(int)
        for value in values:
            counts[json.dumps(value, sort_keys=True)] += 1
        winner, count = max(counts.items(), key=lambda row: (row[1], row[0]))
        if list(counts.values()).count(count) > 1 or count <= len(values) / 2:
            unresolved.append({"family": key[0], "item_id": key[1], "field": key[2], "values": values})
        else:
            consensus[key] = json.loads(winner)
    return consensus, unresolved


def _dimension_correlation_report(
    consensus: dict[tuple[str, str, str], Any]
) -> list[dict[str, Any]]:
    by_dimension: dict[str, dict[str, int]] = defaultdict(dict)
    for (family, item_id, _), value in consensus.items():
        if family.startswith("task3_score:"):
            by_dimension[family.split(":", 1)[1]][item_id] = int(value)
    output = []
    dimensions = sorted(by_dimension)
    for index, left in enumerate(dimensions):
        for right in dimensions[index + 1 :]:
            ids = sorted(set(by_dimension[left]) & set(by_dimension[right]))
            output.append(
                {
                    "dimensions": [left, right],
                    "count": len(ids),
                    "spearman": _spearman(
                        [by_dimension[left][item] for item in ids],
                        [by_dimension[right][item] for item in ids],
                    ),
                }
            )
    return output


def _observed_agreement(left: list[Any], right: list[Any]) -> float:
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def _cohen_kappa(
    left: list[Any], right: list[Any], *, ordinal_values: Iterable[int] | None = None
) -> float | None:
    if not left:
        return None
    categories = list(ordinal_values) if ordinal_values is not None else sorted(
        set(left) | set(right), key=lambda value: json.dumps(value, sort_keys=True)
    )
    index = {value: position for position, value in enumerate(categories)}
    size = len(categories)
    if size <= 1:
        return None
    observed = [[0.0 for _ in categories] for _ in categories]
    for a, b in zip(left, right, strict=True):
        observed[index[a]][index[b]] += 1.0
    total = float(len(left))
    row_marginal = [sum(row) for row in observed]
    col_marginal = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    if ordinal_values is None:
        observed_agreement = sum(observed[i][i] for i in range(size)) / total
        expected_agreement = sum(row_marginal[i] * col_marginal[i] for i in range(size)) / (total * total)
        denominator = 1.0 - expected_agreement
        return None if denominator == 0 else (observed_agreement - expected_agreement) / denominator
    maximum = float((size - 1) ** 2)
    observed_disagreement = sum(
        observed[i][j] * ((i - j) ** 2 / maximum)
        for i in range(size)
        for j in range(size)
    ) / total
    expected_disagreement = sum(
        (row_marginal[i] * col_marginal[j] / total) * ((i - j) ** 2 / maximum)
        for i in range(size)
        for j in range(size)
    ) / total
    return None if expected_disagreement == 0 else 1.0 - observed_disagreement / expected_disagreement


def _spearman(left: list[Any], right: list[Any]) -> float | None:
    if len(left) < 2:
        return None
    a = _ranks([float(value) for value in left])
    b = _ranks([float(value) for value in right])
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    denominator = math.sqrt(
        sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    )
    return None if denominator == 0 else numerator / denominator


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in ordered[cursor:end]:
            ranks[position] = average
        cursor = end
    return ranks


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _consensus_judgment(
    item: dict[str, Any], consensus: dict[tuple[str, str, str], Any]
) -> dict[str, Any]:
    item_id = item["item_id"]
    kind = item["task_kind"]

    def value(family: str, field: str) -> Any:
        return consensus[(family, item_id, field)]

    if kind == "task1_claim":
        pairs = []
        for pair_key in item["validation"]["allowed_pair_keys"]:
            label = value("task1_claim_label", pair_key)
            if label != "none":
                gold_id, prediction_id = pair_key.split("|", 1)
                pairs.append(
                    {
                        "gold_local_id": gold_id,
                        "prediction_local_id": prediction_id,
                        "label": label,
                    }
                )
        prediction_ids = item["validation"]["prediction_ids"]
        return {
            "claim_pair_judgments": pairs,
            "prediction_checks": [
                {
                    "prediction_local_id": prediction_id,
                    "support": value("task1_support", prediction_id),
                    "transition_coherent": value("task1_transition", prediction_id),
                    "evidence_grounded": value("task1_evidence", prediction_id),
                }
                for prediction_id in prediction_ids
            ],
            "future_leak_prediction_ids": [
                prediction_id
                for prediction_id in prediction_ids
                if value("task1_future", prediction_id)
            ],
            "premature_update_prediction_ids": [
                prediction_id
                for prediction_id in prediction_ids
                if value("task1_premature", prediction_id)
            ],
            "no_change_false_update": value("task1_no_change", "value"),
        }
    if kind == "task1_sequence":
        return {
            field: value("task1_sequence", field)
            for field in (
                "state_carry_forward",
                "development_to_state_coherent",
                "contradiction_present",
                "premature_or_future_information",
            )
        } | {
            "local_evidence_labels": [],
            "brief_rationale": "Blinded human field consensus; qualitative rationales are excluded from metric evidence.",
        }
    if kind == "task3_response":
        return {
            "scores": {
                field: value(f"task3_score:{field}", "value")
                for field in TASK3_SCORE_FIELDS
            },
            "future_leakage": value("task3_response_flags", "future_leakage"),
            "unknown_fact_hallucination": value(
                "task3_response_flags", "unknown_fact_hallucination"
            ),
            "stance_compatible": value("task3_response_flags", "stance_compatible"),
            "evidence_local_ids": [],
            "brief_rationale": "Blinded human field consensus; qualitative rationales are excluded from metric evidence.",
        }
    if kind == "task3_pair":
        responses = item["validation"]["responses_by_label"]
        return {
            "pair_type": item["validation"]["expected_pair_type"],
            "response_assessments": [
                {
                    "response_label": label,
                    "response_excerpt": " ".join(str(responses[label]).split()[:8]),
                    "observed_behavior": "Blinded human field consensus; qualitative rationales excluded.",
                    "supports_expected_component": value(
                        "task3_pair_flags", f"supports:{label}"
                    ),
                }
                for label in ("T1", "T2")
            ],
            "expected_direction_present": value(
                "task3_pair_flags", "expected_direction_present"
            ),
            "unsupported_drift": value("task3_pair_flags", "unsupported_drift"),
            "knowledge_boundaries_preserved": value(
                "task3_pair_flags", "knowledge_boundaries_preserved"
            ),
            "local_evidence_labels": [],
            "brief_rationale": "Blinded human field consensus; qualitative rationales are excluded from metric evidence.",
        }
    raise ValueError(f"Unknown human packet task kind: {kind}")


def _gold_dir_from_contract(contract: dict[str, Any]) -> Path:
    parents = {
        Path(row["path"]).parent
        for row in contract["inputs"]
        if (Path(row["path"]).parent / "task1_gold_rubrics.json").is_file()
    }
    if len(parents) != 1:
        raise ValueError("Structural evaluation contract does not identify one gold directory")
    return next(iter(parents))


def _packet_readme(*, packet_id: str, counts: dict[str, int]) -> str:
    return f"""# Blind Evaluator Human Annotation Packet

Packet: `{packet_id}`

This packet contains {counts['total']} items: {counts['task1_claim']} Task 1 claim,
{counts['task1_sequence']} Task 1 sequence, {counts['task3_response']} Task 3 response,
and {counts['task3_pair']} Task 3 pair judgments.

Work only from `blind_items.json`. Do not inspect model manifests, candidate judge
outputs, aggregate metrics, or another annotator's file. Copy
`annotation_template.json` to a new file, replace the pseudonymous annotator ID,
record an ISO-8601 completion time, set both acknowledgements to true only when
accurate, and replace every null judgment with JSON matching the prompt contract.

Do not change item IDs, source IDs, task kinds, packet hashes, or prompt text.
The scoring CLI rejects incomplete coverage, illegal local IDs, non-verbatim pair
excerpts, boundary-score contradictions, duplicate annotators, and missing blinding
acknowledgements.
"""
