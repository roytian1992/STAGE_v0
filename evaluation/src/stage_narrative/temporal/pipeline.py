from __future__ import annotations

import hashlib
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..chunking import ChunkingConfig, TokenCounter, build_token_counter
from ..clients import ModelConfig, build_json_client
from ..io import (
    atomic_write_json,
    load_config,
    load_json,
    load_scenes,
    sha256_file,
    sha256_json,
    unwrap_scene_records,
)
from ..models import clean_text
from ..prompt_loader import PROMPTS
from .materialization import (
    materialize_role_snapshots,
    materialize_task1_assets,
    materialize_task1_state_update_assets,
    materialize_task3_single_turn,
    validate_task1_state_update_assets,
)
from .builder import TemporalAssetBuilder
from .identity import (
    apply_character_identity_lineages,
    project_evidence_identity_lineages,
)
from .models import TemporalBuildConfig, build_graph_index
from .validation import require_temporal_valid, validate_temporal_release


def _prompt_parts(prompt_id: str) -> tuple[str, str]:
    spec = PROMPTS.get(prompt_id)
    return spec.system, spec.user


EVIDENCE_SYSTEM, EVIDENCE_USER = _prompt_parts("evidence")
EVIDENCE_REPAIR_SYSTEM, EVIDENCE_REPAIR_USER = _prompt_parts("evidence_repair")
STATE_OBSERVATION_SYSTEM, STATE_OBSERVATION_USER = _prompt_parts("state_observation")
STATE_TARGET_RESOLUTION_SYSTEM, STATE_TARGET_RESOLUTION_USER = _prompt_parts(
    "state_target_resolution"
)
STATE_RECONCILIATION_SYSTEM, STATE_RECONCILIATION_USER = _prompt_parts(
    "state_reconciliation"
)
DEVELOPMENT_SYSTEM, DEVELOPMENT_USER = _prompt_parts("development")
EPISTEMIC_SYSTEM, EPISTEMIC_USER = _prompt_parts("epistemic")
PERSONA_SYSTEM, PERSONA_USER = _prompt_parts("persona")
CHECKPOINT_SYSTEM, CHECKPOINT_USER = _prompt_parts("checkpoint")
TASK3_PROMPT_SYSTEM, TASK3_PROMPT_USER = _prompt_parts("task3_prompt")


class CharacterTemporalPipeline:
    def __init__(
        self,
        *,
        script_path: Path,
        graph_run_dir: Path,
        config_path: Path,
        output_root: Path,
        run_id: str | None = None,
        resume: bool = False,
        llm_client: Any | None = None,
        token_counter: TokenCounter | None = None,
    ):
        self.script_path = script_path.resolve()
        self.graph_run_dir = graph_run_dir.resolve()
        self.config_path = config_path.resolve()
        self.output_root = output_root.resolve()
        self.config = load_config(self.config_path)
        self.scenes = load_scenes(self.script_path)
        self.movie_id = self.script_path.parent.name
        self.language = _resolve_language(
            self.config.get("language", "auto"), self.script_path
        )
        self.run_id = run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.output_root / self.movie_id / self.run_id
        self.stages_dir = self.run_dir / "stages"
        self.call_checkpoint_dir = self.stages_dir / "llm_call_checkpoints"
        self.assets_dir = self.run_dir / "assets"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.resume = resume
        self._llm_client = llm_client

        chunking_payload = _required_dict(self.config, "chunking")
        self.chunking_config = ChunkingConfig.from_dict(chunking_payload)
        self.token_counter = token_counter or build_token_counter(
            _required_dict(chunking_payload, "tokenizer")
        )
        self.temporal_config = TemporalBuildConfig.from_dict(
            _required_dict(self.config, "temporal"), language=self.language
        )

        self.graph_manifest_path = self.graph_run_dir / "run_manifest.json"
        self.graph_path = self.graph_run_dir / "narrative_graph.json"
        self.graph_validation_path = self.graph_run_dir / "validation_report.json"
        self.graph_manifest = load_json(self.graph_manifest_path)
        self.source_kg_run = _resolve_source_kg_run(
            self.graph_run_dir, self.graph_manifest
        )
        self.source_kg_manifest_path = self.source_kg_run / "run_manifest.json"
        self.source_kg_manifest = load_json(self.source_kg_manifest_path)
        self.canonical_extraction_path = _resolve_canonical_extraction_path(
            self.graph_run_dir, self.source_kg_run
        )
        local_quality = self.graph_run_dir / "kg_quality_report.json"
        source_quality = self.source_kg_run / "kg_quality_report.json"
        self.graph_kg_quality_path = (
            local_quality if local_quality.is_file() else source_quality
        )
        self._source_kg_manifest_drift: dict[str, Any] | None = None
        self._validate_graph_run_inputs()
        self.graph = load_json(self.graph_path)
        self.graph_validation = load_json(self.graph_validation_path)
        self.canonical_scene_records = unwrap_scene_records(
            load_json(self.canonical_extraction_path),
            source=str(self.canonical_extraction_path),
        )
        self.graph_index = build_graph_index(self.graph)

        self._script_hash = sha256_file(self.script_path)
        self._config_hash = sha256_json(self.config)
        self._graph_hash = sha256_file(self.graph_path)
        self._graph_manifest_hash = sha256_file(self.graph_manifest_path)
        self.identity_lineage_patch_path, self.identity_lineage_patch = (
            self._load_identity_lineage_patch()
        )
        self._identity_lineage_patch_hash = (
            sha256_file(self.identity_lineage_patch_path)
            if self.identity_lineage_patch_path
            else None
        )
        package_dir = Path(__file__).resolve().parent
        core_dir = package_dir.parent
        code_paths = [
            core_dir / "chunking.py",
            core_dir / "clients.py",
            core_dir / "io.py",
            core_dir / "models.py",
            *sorted(package_dir.glob("*.py")),
        ]
        self._code_hashes = {
            str(path.relative_to(core_dir.parent)): sha256_file(path)
            for path in code_paths
        }
        self._code_hash = sha256_json(self._code_hashes)

    async def run(self) -> Path:
        self._prepare_run()
        try:
            llm_client = self._build_llm_client()
            builder = TemporalAssetBuilder(
                movie_id=self.movie_id,
                llm_client=llm_client,
                token_counter=self.token_counter,
                max_input_tokens=self.chunking_config.usable_model_input_tokens,
                max_output_tokens=self.chunking_config.max_output_tokens,
                config=self.temporal_config,
                call_checkpoint_dir=self.call_checkpoint_dir,
            )

            base_registry = self._stage(
                "01_character_registry_base",
                lambda: builder.build_character_registry(self.graph_index),
            )
            registry = self._stage(
                "01a_character_identity_registry",
                lambda: apply_character_identity_lineages(
                    base_registry,
                    patch=self.identity_lineage_patch,
                    scene_order_by_id=self.graph_index.scene_order_by_id,
                ),
            )
            raw_evidence_bank = await self._async_stage(
                "02_evidence_bank",
                lambda: builder.build_evidence_bank(
                    scenes=self.scenes,
                    canonical_scene_records=self.canonical_scene_records,
                    character_registry=base_registry,
                    index=self.graph_index,
                ),
            )
            evidence_bank = self._stage(
                "02a_identity_projected_evidence_bank",
                lambda: project_evidence_identity_lineages(raw_evidence_bank, registry),
            )
            _attach_dialogue_scenes(registry, evidence_bank, self.graph_index.scene_order_by_id)
            atomic_write_json(self.stages_dir / "03_registry_with_evidence.json", registry)
            self._complete_stage("03_registry_with_evidence")

            observations = await self._async_stage(
                "04_state_observations",
                lambda: builder.build_state_observations(
                    registry=registry,
                    evidence_bank=evidence_bank,
                    index=self.graph_index,
                ),
            )
            observations = self._stage(
                "04a_asserted_state_observations",
                lambda: builder.filter_state_observations_for_ledger(
                    observations=observations,
                    index=self.graph_index,
                ),
            )
            observations = await self._async_stage(
                "04c_state_target_resolution",
                lambda: builder.resolve_state_targets(
                    registry=registry,
                    observations=observations,
                ),
            )
            state_ledger = await self._async_stage(
                "05_state_ledger",
                lambda: builder.build_state_ledger(
                    registry=registry,
                    observations=observations,
                    scene_count=len(self.scenes),
                ),
            )
            development_graph = await self._async_stage(
                "06_development_graph",
                lambda: builder.build_development_graph(
                    registry=registry,
                    state_ledger=state_ledger,
                    evidence_bank=evidence_bank,
                    index=self.graph_index,
                ),
            )
            epistemic_ledger = await self._async_stage(
                "07_epistemic_ledger",
                lambda: builder.build_epistemic_ledger(
                    registry=registry,
                    evidence_bank=evidence_bank,
                    index=self.graph_index,
                ),
            )
            persona_bank = await self._async_stage(
                "08_persona_evidence_bank",
                lambda: builder.build_persona_evidence_bank(
                    registry=registry,
                    evidence_bank=evidence_bank,
                    index=self.graph_index,
                ),
            )
            registry = self._stage(
                "09_character_registry",
                lambda: builder.apply_eligibility(
                    registry=registry,
                    state_ledger=state_ledger,
                    development_graph=development_graph,
                    epistemic_ledger=epistemic_ledger,
                    persona_bank=persona_bank,
                    evidence_bank=evidence_bank,
                ),
            )
            checkpoints = await self._async_stage(
                "10_checkpoint_manifest",
                lambda: builder.build_checkpoint_manifest(
                    scenes=self.scenes,
                    registry=registry,
                    state_ledger=state_ledger,
                    development_graph=development_graph,
                    epistemic_ledger=epistemic_ledger,
                    persona_bank=persona_bank,
                    evidence_bank=evidence_bank,
                ),
            )
            # Checkpoint construction may reconcile a KG span with later
            # canonically linked temporal evidence on a resumed run. Keep the
            # stage snapshot aligned with the in-memory registry used below.
            atomic_write_json(self.stages_dir / "09_character_registry.json", registry)
            task1 = self._stage(
                "11_task1",
                lambda: materialize_task1_assets(
                    movie_id=self.movie_id,
                    language=self.language,
                    script_path=str(self.script_path),
                    script_sha256=self._script_hash,
                    scenes=self.scenes,
                    registry=registry,
                    checkpoints=checkpoints,
                    state_ledger=state_ledger,
                    development_graph=development_graph,
                ),
            )
            task1_state_updates = materialize_task1_state_update_assets(
                legacy_task1=task1,
                script_path=str(self.script_path),
                script_sha256=self._script_hash,
                scene_count=len(self.scenes),
                state_ledger=state_ledger,
                development_graph=development_graph,
                evidence_bank=evidence_bank,
            )
            task1_state_update_audit = validate_task1_state_update_assets(task1_state_updates)
            if task1_state_update_audit["status"] != "passed":
                raise ValueError(f"Task 1 state-update asset validation failed: {task1_state_update_audit}")
            task3_prompt_candidates = await self._async_stage(
                "13_task3_prompt_candidates",
                lambda: builder.build_task3_prompt_candidates(
                    registry=registry,
                    checkpoints=checkpoints,
                    state_ledger=state_ledger,
                    development_graph=development_graph,
                    epistemic_ledger=epistemic_ledger,
                    persona_bank=persona_bank,
                    evidence_bank=evidence_bank,
                    index=self.graph_index,
                ),
            )
            role_snapshots = self._stage(
                "12_role_snapshots",
                lambda: materialize_role_snapshots(
                    movie_id=self.movie_id,
                    registry=registry,
                    checkpoints=checkpoints,
                    state_ledger=state_ledger,
                    evidence_bank=evidence_bank,
                    persona_bank=persona_bank,
                    prompt_candidates=task3_prompt_candidates,
                ),
            )
            task3 = self._stage(
                "14_task3_single_turn",
                lambda: materialize_task3_single_turn(
                    movie_id=self.movie_id,
                    language=self.language,
                    registry=registry,
                    role_snapshots=role_snapshots,
                    prompt_candidates=task3_prompt_candidates,
                    epistemic_ledger=epistemic_ledger,
                ),
            )

            report = validate_temporal_release(
                scene_count=len(self.scenes),
                graph=self.graph,
                registry=registry,
                evidence_bank=evidence_bank,
                state_ledger=state_ledger,
                development_graph=development_graph,
                epistemic_ledger=epistemic_ledger,
                persona_bank=persona_bank,
                checkpoints=checkpoints,
                task1=task1,
                task1_state_updates=task1_state_updates,
                role_snapshots=role_snapshots,
                task3=task3,
            )
            atomic_write_json(self.run_dir / "validation_report.json", report)
            require_temporal_valid(report)
            task1_eligible = int(report["counts"]["characters_task1_eligible"])
            task3_eligible = int(report["counts"]["characters_task3_eligible"])
            if task1_eligible and task3_eligible:
                eligibility_status = "completed"
            elif task1_eligible or task3_eligible:
                eligibility_status = "completed_partial_eligibility"
            else:
                eligibility_status = "insufficient_eligibility"
            eligibility_report = {
                "schema_version": "stage_temporal_eligibility_report_v1",
                "status": eligibility_status,
                "task1_eligible_character_count": task1_eligible,
                "task3_single_turn_eligible_character_count": task3_eligible,
                "characters": [
                    {
                        "character_id": item["character_id"],
                        "canonical_name": item["canonical_name"],
                        "construction_selected": item["construction_selected"],
                        "task1_eligible": item["task1_eligible"],
                        "task1_exclusion_reasons": item["task1_exclusion_reasons"],
                        "task3_single_turn_eligible": item[
                            "task3_single_turn_eligible"
                        ],
                        "task3_exclusion_reasons": item[
                            "task3_exclusion_reasons"
                        ],
                    }
                    for item in registry["characters"]
                    if item["construction_selected"]
                ],
            }
            atomic_write_json(
                self.run_dir / "eligibility_report.json", eligibility_report
            )
            self._write_canonical_assets(
                registry=registry,
                evidence_bank=evidence_bank,
                state_ledger=state_ledger,
                development_graph=development_graph,
                epistemic_ledger=epistemic_ledger,
                persona_bank=persona_bank,
                checkpoints=checkpoints,
                task1=task1,
                task1_state_updates=task1_state_updates,
                role_snapshots=role_snapshots,
                task3=task3,
            )

            manifest = self._load_manifest()
            manifest["status"] = eligibility_status
            manifest["completed_at"] = _now()
            manifest["completed_stages"] = [
                "01_character_registry_base",
                "02_evidence_bank",
                "03_registry_with_evidence",
                "04_state_observations",
                "04a_asserted_state_observations",
                "04c_state_target_resolution",
                "05_state_ledger",
                "06_development_graph",
                "07_epistemic_ledger",
                "08_persona_evidence_bank",
                "09_character_registry",
                "10_checkpoint_manifest",
                "11_task1",
                "12_role_snapshots",
                "13_task3_prompt_candidates",
                "14_task3_single_turn",
                "15_validation_and_release",
            ]
            manifest["counts"] = report["counts"]
            manifest["outputs"] = {
                "assets": str(self.assets_dir),
                "validation": str(self.run_dir / "validation_report.json"),
                "eligibility": str(self.run_dir / "eligibility_report.json"),
                "stages": str(self.stages_dir),
            }
            self._write_manifest(manifest)
            atomic_write_json(
                self.output_root / self.movie_id / "latest.json",
                {
                    "movie_id": self.movie_id,
                    "run_id": self.run_id,
                    "run_dir": str(self.run_dir),
                    "completed_at": manifest["completed_at"],
                    "script_sha256": self._script_hash,
                    "graph_sha256": self._graph_hash,
                    "config_sha256": self._config_hash,
                    "code_sha256": self._code_hash,
                },
            )
            return self.run_dir
        except Exception as exc:
            manifest = self._load_manifest()
            manifest["status"] = "failed"
            manifest["failed_at"] = _now()
            manifest["error"] = str(exc)
            manifest["traceback"] = traceback.format_exc()
            self._write_manifest(manifest)
            raise

    def _validate_graph_run_inputs(self) -> None:
        required = [
            self.graph_manifest_path,
            self.graph_path,
            self.graph_validation_path,
            self.canonical_extraction_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Completed graph run is missing required files: {missing}")
        manifest = load_json(self.graph_manifest_path)
        validation = load_json(self.graph_validation_path)
        if manifest.get("status") != "completed":
            raise ValueError("Character temporal build requires a completed narrative graph run")
        if validation.get("status") != "passed":
            raise ValueError("Character temporal build requires a validated narrative graph")
        if self.graph_kg_quality_path.is_file():
            kg_quality = load_json(self.graph_kg_quality_path)
            if kg_quality.get("status") != "passed":
                raise ValueError("Character temporal build requires a passed KG quality gate")
        if not clean_text(self.source_kg_manifest.get("status")).endswith("completed"):
            raise ValueError("Character temporal build requires a completed source KG run")
        expected_source_manifest_hash = clean_text(
            manifest.get("source_kg_manifest_sha256")
        )
        current_source_manifest_hash = sha256_file(self.source_kg_manifest_path)
        if (
            expected_source_manifest_hash
            and expected_source_manifest_hash != current_source_manifest_hash
        ):
            entity_registry_path = self.source_kg_run / "stages" / "03_entity_registry.json"
            expected_content_hashes = {
                "canonical_extraction_sha256": clean_text(
                    manifest.get("canonical_scenes_sha256")
                ),
                "entity_registry_sha256": clean_text(
                    manifest.get("entity_registry_sha256")
                ),
            }
            content_paths = {
                "canonical_extraction_sha256": self.canonical_extraction_path,
                "entity_registry_sha256": entity_registry_path,
            }
            mismatches = []
            observed_content_hashes: dict[str, str] = {}
            for key, expected_hash in expected_content_hashes.items():
                path = content_paths[key]
                if not expected_hash or not path.is_file():
                    mismatches.append(f"{key} is unavailable for metadata-only validation")
                    continue
                observed_hash = sha256_file(path)
                observed_content_hashes[key] = observed_hash
                if observed_hash != expected_hash:
                    mismatches.append(f"{key} changed")
            if mismatches:
                raise ValueError(
                    "Narrative graph source KG manifest checksum changed and content "
                    f"identity could not be preserved: {mismatches}"
                )
            self._source_kg_manifest_drift = {
                "classification": "metadata_only",
                "expected_manifest_sha256": expected_source_manifest_hash,
                "observed_manifest_sha256": current_source_manifest_hash,
                "verified_content_sha256": observed_content_hashes,
                "reason": (
                    "The source KG manifest changed after hierarchy construction, while the "
                    "canonical extraction and entity registry consumed by the hierarchy remain "
                    "byte-identical."
                ),
            }
        script_hash = sha256_file(self.script_path)
        expected_script_hash = clean_text(
            manifest.get("input_sha256")
            or self.source_kg_manifest.get("input_sha256")
        )
        if expected_script_hash != script_hash:
            raise ValueError("Screenplay checksum differs from the narrative graph run input")
        expected_movie_id = clean_text(
            manifest.get("movie_id") or self.source_kg_manifest.get("movie_id")
        )
        if expected_movie_id != self.movie_id:
            raise ValueError("Screenplay movie_id differs from the narrative graph run")

    def _build_llm_client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        llm_config = ModelConfig.from_dict(_required_dict(self.config, "llm"))
        if (
            llm_config.max_tokens is not None
            and llm_config.max_tokens != self.chunking_config.max_output_tokens
        ):
            raise ValueError(
                "llm.max_tokens must equal chunking.max_output_tokens for exact context budgets"
            )
        return build_json_client(_required_dict(self.config, "llm"))

    def _stage(self, name: str, build: Callable[[], Any]) -> Any:
        path = self.stages_dir / f"{name}.json"
        if path.exists():
            return load_json(path)
        payload = build()
        atomic_write_json(path, payload)
        self._complete_stage(name)
        return payload

    async def _async_stage(self, name: str, build: Callable[[], Any]) -> Any:
        path = self.stages_dir / f"{name}.json"
        if path.exists():
            return load_json(path)
        payload = await build()
        atomic_write_json(path, payload)
        self._complete_stage(name)
        return payload

    def _prepare_run(self) -> None:
        if self.resume and not self.run_dir.exists():
            raise FileNotFoundError(f"Cannot resume missing temporal run: {self.run_dir}")
        if self.run_dir.exists() and not self.resume:
            raise FileExistsError(
                f"Temporal run already exists; use --resume or a new --run-id: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.call_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest = self._load_manifest()
            checks = {
                "script_sha256": self._script_hash,
                "graph_sha256": self._graph_hash,
                "graph_manifest_sha256": self._graph_manifest_hash,
                "config_sha256": self._config_hash,
                "code_sha256": self._code_hash,
                "identity_lineage_patch_sha256": self._identity_lineage_patch_hash,
            }
            for key, expected in checks.items():
                if manifest.get(key) != expected:
                    if key != "code_sha256":
                        raise ValueError(f"Cannot resume temporal run: {key} changed")
                    current_prompt_hashes = _prompt_hashes()
                    manifest.setdefault("code_revisions", []).append(
                        {
                            "registered_at": _now(),
                            "from_code_sha256": manifest.get("code_sha256"),
                            "to_code_sha256": self._code_hash,
                            "checkpoint_reuse_policy": (
                                "Reuse successful LLM checkpoints only when their prompt/input "
                                "hash matches; new or changed prompt inputs use new checkpoints; "
                                "rerun deterministic validation and incomplete stages."
                            ),
                            "from_prompt_hashes": manifest.get("prompt_hashes"),
                            "to_prompt_hashes": current_prompt_hashes,
                        }
                    )
                    manifest["code_sha256"] = self._code_hash
                    manifest["code_file_sha256"] = self._code_hashes
                    manifest["prompt_hashes"] = current_prompt_hashes
            if manifest.get("status") in {"failed", "paused"}:
                manifest.setdefault("attempt_history", []).append(
                    {
                        "status": manifest.get("status"),
                        "failed_at": manifest.get("failed_at"),
                        "error": manifest.get("error"),
                    }
                )
            for key in ("failed_at", "error", "traceback"):
                manifest.pop(key, None)
            manifest["status"] = "running"
            manifest["resumed_at"] = _now()
            self._write_manifest(manifest)
            return
        manifest = {
            "schema_version": "stage_character_temporal_run_v1",
            "status": "running",
            "started_at": _now(),
            "movie_id": self.movie_id,
            "language": self.language,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "script_path": str(self.script_path),
            "script_sha256": self._script_hash,
            "scene_count": len(self.scenes),
            "graph_run_dir": str(self.graph_run_dir),
            "graph_sha256": self._graph_hash,
            "graph_manifest_sha256": self._graph_manifest_hash,
            "source_kg_run": str(self.source_kg_run),
            "source_kg_manifest_sha256": sha256_file(
                self.source_kg_manifest_path
            ),
            "source_kg_manifest_drift": self._source_kg_manifest_drift,
            "canonical_extraction_path": str(self.canonical_extraction_path),
            "canonical_extraction_sha256": sha256_file(
                self.canonical_extraction_path
            ),
            "config_path": str(self.config_path),
            "config_sha256": self._config_hash,
            "config_snapshot": str(self.run_dir / "config_snapshot.json"),
            "code_sha256": self._code_hash,
            "code_file_sha256": self._code_hashes,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tokenizer": self.token_counter.metadata,
            "max_model_input_tokens": self.chunking_config.max_model_input_tokens,
            "usable_model_input_tokens": self.chunking_config.usable_model_input_tokens,
            "prompt_headroom_tokens": self.chunking_config.prompt_headroom_tokens,
            "prompt_hashes": _prompt_hashes(),
            "identity_lineage_patch": (
                str(self.identity_lineage_patch_path)
                if self.identity_lineage_patch_path
                else None
            ),
            "identity_lineage_patch_sha256": self._identity_lineage_patch_hash,
            "completed_stages": [],
        }
        atomic_write_json(self.run_dir / "config_snapshot.json", self.config)
        self._write_manifest(manifest)

    def _load_identity_lineage_patch(self) -> tuple[Path | None, dict[str, Any] | None]:
        temporal = _required_dict(self.config, "temporal")
        raw_path = str(temporal.get("identity_lineage_patch") or "").strip()
        if not raw_path:
            return None, None
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self.config_path.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = load_json(path)
        if payload.get("source_graph_sha256") != self._graph_hash:
            raise ValueError("Identity-lineage patch source graph hash mismatch")
        return path, payload

    def _complete_stage(self, stage: str) -> None:
        manifest = self._load_manifest()
        completed = list(manifest.get("completed_stages", []))
        if stage not in completed:
            completed.append(stage)
        manifest["completed_stages"] = completed
        manifest["updated_at"] = _now()
        self._write_manifest(manifest)

    def _write_canonical_assets(self, **payloads: dict[str, Any]) -> None:
        paths = {
            "registry": self.assets_dir / "characters" / "character_registry.json",
            "evidence_bank": self.assets_dir / "source" / "evidence_units.json",
            "state_ledger": self.assets_dir / "characters" / "state_ledger.json",
            "development_graph": self.assets_dir / "characters" / "development_graph.json",
            "epistemic_ledger": self.assets_dir / "characters" / "epistemic_ledger.json",
            "persona_bank": self.assets_dir / "characters" / "persona_evidence_bank.json",
            "checkpoints": self.assets_dir / "checkpoints" / "checkpoint_manifest.json",
            "role_snapshots": self.assets_dir / "checkpoints" / "role_snapshot_index.json",
            "task1": self.assets_dir / "tasks" / "task1_character_development_tracking.json",
            "task1_reference_state_update": self.assets_dir / "tasks" / "task_1_reference_state_update.json",
            "task1_autoregressive_state_update": self.assets_dir / "tasks" / "task_1_autoregressive_state_update.json",
            "task3": self.assets_dir / "tasks" / "task3_checkpoint_single_turn.json",
        }
        for key, path in paths.items():
            if key == "task1_reference_state_update":
                atomic_write_json(path, payloads["task1_state_updates"]["reference"])
            elif key == "task1_autoregressive_state_update":
                atomic_write_json(path, payloads["task1_state_updates"]["autoregressive"])
            else:
                atomic_write_json(path, payloads[key])
        atomic_write_json(
            self.assets_dir / "source" / "screenplay_manifest.json",
            {
                "movie_id": self.movie_id,
                "script_path": str(self.script_path),
                "script_sha256": self._script_hash,
                "scene_count": len(self.scenes),
            },
        )
        atomic_write_json(
            self.assets_dir / "narrative" / "narrative_graph_ref.json",
            {
                "movie_id": self.movie_id,
                "graph_run_dir": str(self.graph_run_dir),
                "graph_path": str(self.graph_path),
                "graph_sha256": self._graph_hash,
                "graph_manifest_sha256": self._graph_manifest_hash,
            },
        )

    def _load_manifest(self) -> dict[str, Any]:
        return load_json(self.manifest_path)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)


def _attach_dialogue_scenes(
    registry: dict[str, Any],
    evidence_bank: dict[str, Any],
    scene_order_by_id: dict[str, int],
) -> None:
    by_character: dict[str, set[str]] = {}
    for item in evidence_bank["evidence_units"]:
        character_id = item.get("speaker_character_id")
        if character_id and item.get("evidence_type") == "dialogue":
            by_character.setdefault(character_id, set()).add(item["scene_id"])
    for character in registry["characters"]:
        character["dialogue_scene_ids"] = sorted(
            by_character.get(character["character_id"], set()),
            key=lambda scene_id: scene_order_by_id[scene_id],
        )


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config requires object field: {key}")
    return value


def _resolve_language(value: Any, script_path: Path) -> str:
    language = clean_text(value) or "auto"
    if language.casefold() != "auto":
        return language
    parents = {parent.name.casefold() for parent in script_path.parents}
    if "chinese" in parents:
        return "Chinese"
    if "english" in parents:
        return "English"
    raise ValueError("language=auto requires screenplay under Chinese/ or English/")


def _resolve_source_kg_run(
    graph_run_dir: Path, graph_manifest: dict[str, Any]
) -> Path:
    configured = clean_text(graph_manifest.get("source_kg_run"))
    if configured:
        return Path(configured).expanduser().resolve()
    return graph_run_dir


def _resolve_canonical_extraction_path(
    graph_run_dir: Path, source_kg_run: Path
) -> Path:
    candidates = [
        graph_run_dir / "stages" / "04_canonicalized_extraction.json",
        graph_run_dir / "stages" / "03_canonicalized_extraction.json",
        source_kg_run / "stages" / "04_canonicalized_extraction.json",
        source_kg_run / "stages" / "03_canonicalized_extraction.json",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _prompt_hashes() -> dict[str, str]:
    prompts = {
        "evidence": (
            EVIDENCE_SYSTEM
            + EVIDENCE_USER
            + EVIDENCE_REPAIR_SYSTEM
            + EVIDENCE_REPAIR_USER
        ),
        "state_observation": STATE_OBSERVATION_SYSTEM + STATE_OBSERVATION_USER,
        "state_target_resolution": (
            STATE_TARGET_RESOLUTION_SYSTEM + STATE_TARGET_RESOLUTION_USER
        ),
        "state_reconciliation": STATE_RECONCILIATION_SYSTEM + STATE_RECONCILIATION_USER,
        "development": DEVELOPMENT_SYSTEM + DEVELOPMENT_USER,
        "epistemic": EPISTEMIC_SYSTEM + EPISTEMIC_USER,
        "persona": PERSONA_SYSTEM + PERSONA_USER,
        "checkpoint": CHECKPOINT_SYSTEM + CHECKPOINT_USER,
        "task3_prompt": TASK3_PROMPT_SYSTEM + TASK3_PROMPT_USER,
    }
    return {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in prompts.items()
    }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
