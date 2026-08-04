from __future__ import annotations

import asyncio
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import atomic_write_json, load_config, load_json, sha256_file, sha256_json
from .pipeline import NarrativeGraphPipeline
from .temporal.pipeline import CharacterTemporalPipeline


THROUGH_ORDER = {
    "kg": 1,
    "episodes": 2,
    "relations": 3,
    "storylines": 4,
    "task-assets": 5,
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class StageBuildOrchestrator:
    def __init__(
        self,
        *,
        script_path: Path,
        config_path: Path,
        output_root: Path,
        run_id: str,
        through: str,
        resume: bool = False,
        plan_only: bool = False,
        project_root: Path | None = None,
        benchmark_selection: dict[str, Any] | None = None,
    ):
        if through not in THROUGH_ORDER:
            raise ValueError(f"Unsupported through stage: {through}")
        self.script_path = script_path.resolve()
        self.config_path = config_path.resolve()
        self.output_root = output_root.resolve()
        self.run_id = run_id
        self.through = through
        self.resume = resume
        self.plan_only = plan_only
        self.project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        self.benchmark_selection = benchmark_selection
        self.config = load_config(self.config_path)
        if self.config.get("schema_version") != "stage_end_to_end_config_v1":
            raise ValueError("Unified build config must use stage_end_to_end_config_v1")
        self.movie_id = self.script_path.parent.name
        self.run_dir = self.output_root / self.movie_id / self.run_id
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.stage_refs_dir = self.run_dir / "stages"
        self.stage_runs_dir = self.run_dir / "stage_runs"
        self.logs_dir = self.run_dir / "logs"
        self.stage_configs = self._resolve_stage_configs()
        self.config_hash = sha256_json(self.config)
        self.script_hash = sha256_file(self.script_path)

    def _resolve_stage_configs(self) -> dict[str, Path]:
        values = self.config.get("stage_configs")
        if not isinstance(values, dict):
            raise ValueError("Unified build config requires stage_configs")
        required = ["kg"]
        if THROUGH_ORDER[self.through] >= THROUGH_ORDER["episodes"]:
            required.append("hierarchy")
        if THROUGH_ORDER[self.through] >= THROUGH_ORDER["task-assets"]:
            required.append("temporal")
        output = {}
        for stage in required:
            raw = values.get(stage)
            if not raw:
                raise ValueError(f"Unified build config lacks {stage} config")
            path = Path(str(raw))
            if not path.is_absolute():
                path = self.config_path.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            output[stage] = path
        return output

    @property
    def kg_run_dir(self) -> Path:
        return self.stage_runs_dir / "kg" / self.movie_id / "kg"

    @property
    def hierarchy_run_dir(self) -> Path:
        return self.stage_runs_dir / "hierarchy"

    @property
    def temporal_run_dir(self) -> Path:
        return self.stage_runs_dir / "temporal" / self.movie_id / "temporal"

    async def run(self) -> Path:
        self._prepare()
        if self.plan_only:
            return self.run_dir
        manifest = load_json(self.manifest_path)
        try:
            kg_dir = await self._run_kg()
            self._record_stage("kg", kg_dir)
            if self.through == "kg":
                return self._complete()

            hierarchy_dir = await self._run_hierarchy(kg_dir)
            self._record_stage(self.through if self.through != "task-assets" else "storylines", hierarchy_dir)
            if self.through != "task-assets":
                return self._complete()

            temporal_dir = await self._run_temporal(hierarchy_dir)
            self._record_stage("task-assets", temporal_dir)
            return self._complete()
        except Exception as exc:
            manifest = load_json(self.manifest_path)
            manifest.update(
                {
                    "status": "failed",
                    "failed_at": _now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            atomic_write_json(self.manifest_path, manifest)
            raise

    def _prepare(self) -> None:
        if self.resume and not self.run_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing lineage run: {self.run_dir}")
        if self.run_dir.exists() and not self.resume:
            raise FileExistsError(f"Lineage run already exists: {self.run_dir}")
        self.stage_refs_dir.mkdir(parents=True, exist_ok=self.resume)
        self.stage_runs_dir.mkdir(parents=True, exist_ok=self.resume)
        self.logs_dir.mkdir(parents=True, exist_ok=self.resume)
        stage_config_meta = {
            stage: {"path": str(path), "sha256": sha256_file(path)}
            for stage, path in self.stage_configs.items()
        }
        if self.manifest_path.is_file():
            manifest = load_json(self.manifest_path)
            checks = {
                "script_sha256": self.script_hash,
                "config_sha256": self.config_hash,
            }
            for key, expected in checks.items():
                if manifest.get(key) != expected:
                    raise ValueError(f"Cannot resume lineage run after {key} changed")
            prior_through = str(manifest.get("through"))
            if THROUGH_ORDER[self.through] < THROUGH_ORDER[prior_through]:
                raise ValueError("Cannot resume a lineage run to an earlier through stage")
            prior_configs = manifest.get("stage_configs", {})
            if any(stage_config_meta.get(key) != value for key, value in prior_configs.items()):
                raise ValueError("Cannot resume lineage run after a stage config changed")
            manifest["through"] = self.through
            manifest["stage_configs"] = stage_config_meta
            manifest["status"] = "running"
            manifest["resumed_at"] = _now()
            manifest.pop("failed_at", None)
            manifest.pop("error", None)
            manifest.pop("traceback", None)
        else:
            manifest = {
                "schema_version": "stage_end_to_end_lineage_run_v1",
                "status": "planned" if self.plan_only else "running",
                "started_at": _now(),
                "movie_id": self.movie_id,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "script_path": str(self.script_path),
                "script_sha256": self.script_hash,
                "config_path": str(self.config_path),
                "config_sha256": self.config_hash,
                "stage_configs": stage_config_meta,
                "through": self.through,
                "completed_stages": [],
                "stage_refs": {},
                **(
                    {"benchmark_selection": self.benchmark_selection}
                    if self.benchmark_selection is not None
                    else {}
                ),
            }
        atomic_write_json(self.manifest_path, manifest)
        atomic_write_json(self.run_dir / "config_snapshot.json", self.config)
        atomic_write_json(self.run_dir / "call_budget.json", self._build_call_budget())

    def _build_call_budget(self) -> dict[str, Any]:
        stages = {}
        for name, path in self.stage_configs.items():
            config = load_config(path)
            if name == "kg":
                concurrency_values = [
                    int(config.get(section_name, {}).get("max_concurrency", 0))
                    for section_name in (
                        "extraction",
                        "entity_resolution",
                        "kg_quality",
                    )
                ]
                max_concurrency = max(concurrency_values, default=0) or None
            else:
                section = config.get("temporal" if name == "temporal" else name, {})
                max_concurrency = section.get("max_concurrency")
            endpoint_pool = config.get("llm", {}).get("endpoint_pool", [])
            stages[name] = {
                "config_path": str(path),
                "config_sha256": sha256_file(path),
                "model": config.get("llm", {}).get("model"),
                "max_concurrency": max_concurrency,
                "endpoint_pool_total_concurrency": sum(
                    int(item.get("max_concurrency", 0))
                    for item in endpoint_pool
                    if isinstance(item, dict)
                ),
                "exact_call_count_known": False,
            }
        return {
            "schema_version": "stage_end_to_end_call_budget_v1",
            "created_at": _now(),
            "plan_only": self.plan_only,
            "through": self.through,
            "formal_calls_per_item": 1,
            "semantic_resampling": False,
            "stages": stages,
            "note": "Exact item counts are written by stage-local preflight/accounting artifacts when available.",
        }

    async def _run_kg(self) -> Path:
        if self._stage_complete("kg"):
            return Path(load_json(self._stage_ref_path("kg"))["run_dir"])
        if self._completed_stage_run("kg", self.kg_run_dir):
            return self.kg_run_dir
        pipeline = NarrativeGraphPipeline(
            script_path=self.script_path,
            config_path=self.stage_configs["kg"],
            output_root=self.stage_runs_dir / "kg",
            run_id="kg",
            resume=self.kg_run_dir.exists(),
            kg_only=True,
        )
        return await pipeline.run()

    async def _run_hierarchy(self, kg_dir: Path) -> Path:
        target = "storylines" if self.through == "task-assets" else self.through
        if self._stage_complete(target):
            return Path(load_json(self._stage_ref_path(target))["run_dir"])
        if self._completed_stage_run(target, self.hierarchy_run_dir):
            return self.hierarchy_run_dir
        command = [
            sys.executable,
            str(self.project_root / "scripts" / "build_evolving_hierarchy.py"),
            "--source-kg-run",
            str(kg_dir),
            "--config",
            str(self.stage_configs["hierarchy"]),
            "--output-run-dir",
            str(self.hierarchy_run_dir),
            "--stop-after",
            target,
        ]
        if self.hierarchy_run_dir.exists():
            command.append("--resume")
        await self._run_subprocess(command, self.logs_dir / f"hierarchy-{target}.log")
        return self.hierarchy_run_dir

    async def _run_temporal(self, hierarchy_dir: Path) -> Path:
        if self._stage_complete("task-assets"):
            return Path(load_json(self._stage_ref_path("task-assets"))["run_dir"])
        if self._completed_stage_run("task-assets", self.temporal_run_dir):
            return self.temporal_run_dir
        pipeline = CharacterTemporalPipeline(
            script_path=self.script_path,
            graph_run_dir=hierarchy_dir,
            config_path=self.stage_configs["temporal"],
            output_root=self.stage_runs_dir / "temporal",
            run_id="temporal",
            resume=self.temporal_run_dir.exists(),
        )
        return await pipeline.run()

    async def _run_subprocess(self, command: list[str], log_path: Path) -> None:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        log_path.write_bytes(stdout)
        if process.returncode:
            raise RuntimeError(
                f"Stage command failed with exit {process.returncode}; see {log_path}"
            )

    def _stage_ref_path(self, stage: str) -> Path:
        return self.stage_refs_dir / stage / "run_ref.json"

    def _stage_complete(self, stage: str) -> bool:
        path = self._stage_ref_path(stage)
        if not path.is_file():
            return False
        ref = load_json(path)
        manifest_path = Path(ref["manifest_path"])
        return manifest_path.is_file() and sha256_file(manifest_path) == ref["manifest_sha256"]

    def _completed_stage_run(self, stage: str, run_dir: Path) -> bool:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = load_json(manifest_path)
        if stage == "kg":
            return (
                manifest.get("status") == "kg_completed"
                and (manifest.get("script_sha256") or manifest.get("input_sha256"))
                == self.script_hash
                and manifest.get("config_sha256")
                == sha256_json(load_config(self.stage_configs["kg"]))
            )
        validation_path = run_dir / "validation_report.json"
        if not validation_path.is_file() or load_json(validation_path).get("status") != "passed":
            return False
        if stage == "task-assets":
            graph_path = self.hierarchy_run_dir / "narrative_graph.json"
            return (
                manifest.get("status")
                in {"completed", "completed_partial_eligibility", "insufficient_eligibility"}
                and manifest.get("script_sha256") == self.script_hash
                and manifest.get("config_sha256")
                == sha256_json(load_config(self.stage_configs["temporal"]))
                and graph_path.is_file()
                and manifest.get("graph_sha256") == sha256_file(graph_path)
            )
        return (
            manifest.get("status") == "completed"
            and manifest.get("config_sha256")
            == sha256_json(load_config(self.stage_configs["hierarchy"]))
        )

    def _record_stage(self, stage: str, run_dir: Path) -> None:
        stage_manifest_path = run_dir / "run_manifest.json"
        stage_manifest = load_json(stage_manifest_path)
        ref = {
            "schema_version": "stage_end_to_end_stage_ref_v1",
            "stage": stage,
            "run_dir": str(run_dir),
            "manifest_path": str(stage_manifest_path),
            "manifest_sha256": sha256_file(stage_manifest_path),
            "stage_status": stage_manifest.get("status"),
            "recorded_at": _now(),
        }
        path = self._stage_ref_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, ref)
        manifest = load_json(self.manifest_path)
        if stage not in manifest["completed_stages"]:
            manifest["completed_stages"].append(stage)
        manifest["stage_refs"][stage] = str(path)
        atomic_write_json(self.manifest_path, manifest)

    def _complete(self) -> Path:
        manifest = load_json(self.manifest_path)
        manifest.update(
            {
                "status": f"completed_through_{self.through}",
                "completed_at": _now(),
            }
        )
        atomic_write_json(self.manifest_path, manifest)
        refs = {
            stage: load_json(Path(path))
            for stage, path in manifest["stage_refs"].items()
        }
        call_budget_path = self.run_dir / "call_budget.json"
        call_budget = load_json(call_budget_path)
        actual_accounting = self._collect_actual_call_accounting(refs)
        call_budget.update(
            {
                "schema_version": "stage_end_to_end_call_budget_v2",
                "actual_accounting": actual_accounting,
                "actual_accounting_complete": all(
                    item["available"] for item in actual_accounting.values()
                ),
            }
        )
        atomic_write_json(call_budget_path, call_budget)
        release = {
            "schema_version": "stage_end_to_end_release_manifest_v1",
            "created_at": _now(),
            "release_status": manifest["status"],
            "movie_id": self.movie_id,
            "run_id": self.run_id,
            "through": self.through,
            "script_path": str(self.script_path),
            "script_sha256": self.script_hash,
            "config_sha256": self.config_hash,
            "stage_refs": refs,
            "stage_release_statuses": {
                stage: load_json(Path(ref["manifest_path"])).get("release_status")
                for stage, ref in refs.items()
            },
            "call_budget_path": str(call_budget_path),
            "call_budget_sha256": sha256_file(call_budget_path),
        }
        release_path = self.run_dir / "release_manifest.json"
        atomic_write_json(release_path, release)
        manifest["release_manifest"] = str(release_path)
        atomic_write_json(self.manifest_path, manifest)
        return self.run_dir

    def _collect_actual_call_accounting(
        self, refs: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        output = {}
        for stage, ref in refs.items():
            run_dir = Path(ref["run_dir"])
            candidates = [
                run_dir / "call_budget.json",
                run_dir / "kg_call_budget.json",
                run_dir / "hierarchy_call_budget.json",
            ]
            budget_path = next((path for path in candidates if path.is_file()), None)
            if budget_path is None:
                output[stage] = {
                    "available": False,
                    "reason": "stage_local_actual_call_budget_not_available",
                }
                continue
            output[stage] = {
                "available": True,
                "path": str(budget_path),
                "sha256": sha256_file(budget_path),
                "accounting": load_json(budget_path),
            }
        return output


__all__ = ["StageBuildOrchestrator", "THROUGH_ORDER"]
