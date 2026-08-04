from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..io import atomic_write_json, load_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_evaluation_release(
    *,
    output_dir: Path,
    task_release_manifest: Path,
    calibration_summary: Path,
    test_report: Path,
    runtime_config: Path,
    pair_annotations: Path,
    reference_evaluation_run: Path | None = None,
    human_calibration_report: Path | None = None,
    human_packet_manifest: Path | None = None,
    independent_smoke_report: Path | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation release: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy(PROJECT_ROOT / "docs" / "evaluation_release_readme_v1.md", output_dir / "README.md")
    _copy(PROJECT_ROOT / "docs" / "task1_task3_metric_spec_v1.md", output_dir / "metric_spec.md")
    _copy(calibration_summary.resolve(), output_dir / "calibration_summary.md")
    _copy(test_report.resolve(), output_dir / "test_report.json")
    _copy_tree(PROJECT_ROOT / "configs" / "schemas" / "evaluation_v1", output_dir / "schemas")
    _copy_tree(PROJECT_ROOT / "configs" / "prompts" / "en" / "evaluation_v1", output_dir / "prompts" / "en")
    _copy_tree(PROJECT_ROOT / "configs" / "prompts" / "zh" / "evaluation_v1", output_dir / "prompts" / "zh")
    _copy(pair_annotations.resolve(), output_dir / "configs" / "task3_pair_annotations_evaluation_v1.json")
    _copy(
        PROJECT_ROOT / "configs" / "no_context_control_v1.json",
        output_dir / "configs" / "no_context_control_v1.json",
    )
    _write_public_config(runtime_config.resolve(), output_dir / "configs" / "evaluation_standard24k_v1_public.json")
    _copy(
        PROJECT_ROOT / "configs" / "human_calibration_policy_v1.json",
        output_dir / "configs" / "human_calibration_policy_v1.json",
    )
    _write_public_config(
        PROJECT_ROOT / "configs" / "evaluation_standard24k_v1_independent_deepseek_example.json",
        output_dir / "configs" / "evaluation_standard24k_v1_independent_example_public.json",
    )
    for relative in (
        "src/stage_narrative/evaluation/__init__.py",
        "src/stage_narrative/evaluation/aggregation.py",
        "src/stage_narrative/evaluation/human_calibration.py",
        "src/stage_narrative/evaluation/materialization.py",
        "src/stage_narrative/evaluation/release.py",
        "src/stage_narrative/evaluation/runner.py",
        "src/stage_narrative/evaluation/schemas.py",
        "src/stage_narrative/evaluation/task1_alignment.py",
        "src/stage_narrative/evaluation/task1_metrics.py",
        "src/stage_narrative/evaluation/task3_metrics.py",
        "src/stage_narrative/evaluation/validation.py",
        "src/stage_narrative/temporal/benchmark_protocol.py",
        "src/stage_narrative/clients.py",
        "src/stage_narrative/io.py",
        "src/stage_narrative/prompt_loader.py",
        "scripts/run_standard24k_predictions.py",
        "scripts/run_standard24k_evaluation.py",
        "scripts/build_evaluation_release.py",
        "scripts/build_evaluator_human_packet.py",
        "scripts/build_human_consensus_evaluation.py",
        "scripts/check_independent_judge.py",
        "scripts/score_evaluator_human_annotations.py",
        "scripts/validate_evaluation_release.py",
    ):
        _copy(PROJECT_ROOT / relative, output_dir / "code" / relative)
    task_release_manifest = task_release_manifest.resolve()
    reference = None
    reference_payload = None
    if reference_evaluation_run is not None:
        reference_manifest = reference_evaluation_run.resolve() / "manifest.json"
        reference_payload = load_json(reference_manifest)
        from .validation import validate_evaluation_run

        reference_validation = validate_evaluation_run(reference_evaluation_run.resolve())
        reference = {
            "path": str(reference_manifest),
            "sha256": sha256_file(reference_manifest),
            "evaluation_mode": reference_payload["evaluation_mode"],
            "model_contract": reference_payload["model_contract"],
            "validation_status": reference_validation["status"],
        }
    human_calibration = {"instance_count": 0, "annotator_count": 0, "agreement": None, "status": "pending"}
    if human_calibration_report is not None:
        human_payload = load_json(human_calibration_report.resolve())
        if human_payload.get("schema_version") != "stage_evaluator_human_calibration_report_v1":
            raise ValueError("Unknown human calibration report schema")
        _copy(human_calibration_report.resolve(), output_dir / "human_calibration_report.json")
        human_calibration = {
            "instance_count": int(human_payload["packet"]["item_count"]),
            "annotator_count": int(human_payload["human_annotation_count"]),
            "agreement": {
                "human_pairwise": human_payload["human_pairwise"],
                "judge_vs_human_consensus": human_payload["judge_vs_human_consensus"],
            },
            "status": human_payload["status"],
            "report_sha256": sha256_file(human_calibration_report.resolve()),
        }
    if human_packet_manifest is not None:
        _copy(human_packet_manifest.resolve(), output_dir / "human_packet_manifest.json")
    if independent_smoke_report is not None:
        _copy(independent_smoke_report.resolve(), output_dir / "independent_judge_smoke.json")
    formal_complete = bool(
        human_calibration["status"] == "passed"
        and human_calibration["annotator_count"] >= 3
        and reference_payload is not None
        and reference_payload.get("evaluation_mode")
        in {"formal_independent_evaluation", "formal_blinded_human_evaluation"}
        and reference_payload.get("model_contract", {}).get("independent") is True
        and (
            reference_payload.get("evaluation_mode") == "formal_blinded_human_evaluation"
            or reference_payload.get("model_contract", {})
            .get("judge_identity", {})
            .get("weights_and_model_family_distinct_from_actor")
            is True
        )
    )
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "evaluation_release_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": "stage_evaluation_release_manifest_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "formal_evaluation_release_v1" if formal_complete else "frozen_evaluator_v1",
        "formal_benchmark_status": (
            "completed_human_calibrated_formal_evaluation"
            if formal_complete
            else "pending_human_calibration_and_independent_judge"
        ),
        "track_id": "standard-24k-evaluation-v1",
        "metric_spec_sha256": sha256_file(output_dir / "metric_spec.md"),
        "primary_metrics": [
            "task1_current_state_quality_f1",
            "task1_development_quality_f1",
            "task1_state_retention_strict_rate",
            "task3_character_fidelity",
            "task3_memory_faithfulness",
            "task3_boundary_compliance",
            "task3_response_naturalness"
        ],
        "diagnostic_metrics": [
            "task1_soft_partial_weighted_metrics",
            "task1_closed_gold_strict_precision_recall_f1",
            "task1_explicit_gold_state_coverage",
            "task1_evidence_grounding",
            "task1_transition_coherence",
            "task1_future_leakage",
            "task1_premature_update",
            "task1_no_change_false_update",
            "task1_longitudinal_consistency",
            "task3_future_leakage",
            "task3_unknown_fact_hallucination",
            "task3_stance_incompatibility",
            "task3_typed_pair_accuracy"
        ],
        "primary_match_variant": "strict",
        "partial_weight": 0.5,
        "aggregation_order": "checkpoint within character; movie macro primary; character macro secondary",
        "empty_denominator": "null with valid-count reporting",
        "prediction_schema_version": "stage_task1/task3_standard24k_predictions_v1",
        "evaluation_schema_version": "stage_task1/task3_evaluation_v1",
        "allowed_context_protocols": ["standard-24k-sequential-rolling", "standard-24k-frozen-actor-context"],
        "retry_repair_contract": {
            "semantic_samples_per_call": 1,
            "transport_retry_from_runtime_config": True,
            "semantic_voting": False,
            "format_repair": "at most one recorded format-only repair; runner v1 currently fail-closed"
        },
        "human_calibration": human_calibration,
        "task_asset_release": {
            "manifest_path": str(task_release_manifest),
            "manifest_sha256": sha256_file(task_release_manifest),
        },
        "reference_evaluation_run": reference,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "evaluation_release_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            _copy(path, destination / path.relative_to(source))


def _write_public_config(source: Path, destination: Path) -> None:
    payload = load_json(source)
    if "api_key" in payload["llm"]:
        payload["llm"]["api_key"] = "<configured-locally>"
    if "base_url" in payload["llm"]:
        payload["llm"]["base_url"] = "<configured-locally>"
    for endpoint in payload["llm"].get("endpoint_pool", []):
        endpoint["base_url"] = "<configured-locally>"
    payload["tokenizer"]["path"] = "<configured-locally>"
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload)
