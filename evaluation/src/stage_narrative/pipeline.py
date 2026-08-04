from __future__ import annotations

import asyncio
import hashlib
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .clients import ModelConfig, OpenAIEmbeddingClient, build_json_client
from .chunking import ChunkingConfig, TokenCounter, build_token_counter
from .entity_resolution import (
    EntityResolver,
    ResolutionConfig,
    canonicalize_scene_records,
    finalize_narrative_references,
)
from .extraction import ExtractionConfig, ScreenplayExtractor
from .graph import build_narrative_graph
from .hierarchy import (
    HierarchyConfig,
    NarrativeHierarchyBuilder,
    build_entity_neighbor_index,
)
from .io import atomic_write_json, load_config, load_json, load_scenes, sha256_file, sha256_json
from .kg_quality import (
    KGQualityConfig,
    KGQualityReviewer,
    build_named_character_review_queue,
    build_kg_quality_report,
    build_quality_issues,
    build_semantic_kg,
    require_quality,
)
from .kg_review import SceneGraphReviewer, SceneReviewConfig
from .prompt_loader import PROMPTS
from .prompt_assets import prompt_asset_paths
from .relation_resolution import RelationResolutionConfig, RelationResolver
from .validation import require_valid, validate_graph


class NarrativeGraphPipeline:
    def __init__(
        self,
        *,
        script_path: Path,
        config_path: Path,
        output_root: Path,
        run_id: str | None = None,
        resume: bool = False,
        llm_client: Any | None = None,
        embedding_client: Any | None = None,
        token_counter: TokenCounter | None = None,
        kg_only: bool = False,
    ):
        self.script_path = script_path.resolve()
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        self.scenes = load_scenes(self.script_path)
        self.movie_id = self.script_path.parent.name
        self.language = _resolve_language(self.config.get("language", "auto"), self.script_path)
        self.output_root = output_root.resolve()
        self.run_id = run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.output_root / self.movie_id / self.run_id
        self.resume = resume
        self.kg_only = kg_only
        self._llm_client = llm_client
        self._embedding_client = embedding_client
        chunking_payload = _required_dict(self.config, "chunking")
        self.chunking_config = ChunkingConfig.from_dict(chunking_payload)
        self.token_counter = token_counter or build_token_counter(
            _required_dict(chunking_payload, "tokenizer")
        )
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.stages_dir = self.run_dir / "stages"
        self._config_hash = sha256_json(self.config)
        self._input_hash = sha256_file(self.script_path)
        package_dir = Path(__file__).resolve().parent
        self._code_hashes = {
            str(path.relative_to(package_dir.parent)): sha256_file(path)
            for path in sorted(package_dir.glob("*.py"))
        }
        self._code_hash = sha256_json(self._code_hashes)

    async def run(self) -> Path:
        self._prepare_run()
        manifest = self._load_manifest()
        try:
            if self._llm_client is None:
                llm_config = ModelConfig.from_dict(
                    _required_dict(self.config, "llm")
                )
                if (
                    llm_config.max_tokens is not None
                    and llm_config.max_tokens != self.chunking_config.max_output_tokens
                ):
                    raise ValueError(
                        "llm.max_tokens must equal chunking.max_output_tokens so context "
                        "budgets match the actual generation limit"
                    )
                llm_client = build_json_client(_required_dict(self.config, "llm"))
            else:
                llm_client = self._llm_client
            if self._embedding_client is None:
                embedding_config = ModelConfig.from_dict(
                    _required_dict(self.config, "embedding")
                )
                embedding_client = OpenAIEmbeddingClient(embedding_config)
            else:
                embedding_client = self._embedding_client

            extraction = await self._run_extraction(llm_client)
            self._complete_stage("01_extraction")

            reviewed_records = await self._run_scene_graph_review(extraction, llm_client)
            self._complete_stage("02_scene_graph_review")

            registry_path = self.stages_dir / "03_entity_registry.json"
            canonical_path = self.stages_dir / "04_canonicalized_extraction.json"
            if registry_path.exists() and canonical_path.exists():
                entity_registry = load_json(registry_path)
                canonical_records = load_json(canonical_path)
            else:
                resolution_cfg = _required_dict(self.config, "entity_resolution")
                resolver = EntityResolver(
                    movie_id=self.movie_id,
                    llm_client=llm_client,
                    embedding_client=embedding_client,
                    config=ResolutionConfig(
                        lexical_threshold=float(resolution_cfg.get("lexical_threshold", 0.88)),
                        embedding_threshold=float(resolution_cfg.get("embedding_threshold", 0.86)),
                        embedding_top_k=int(resolution_cfg.get("embedding_top_k", 8)),
                        embedding_batch_size=int(resolution_cfg.get("embedding_batch_size", 128)),
                        decision_batch_size=int(resolution_cfg.get("decision_batch_size", 16)),
                        cluster_decision_batch_size=int(
                            resolution_cfg.get("cluster_decision_batch_size", 4)
                        ),
                        decision_min_confidence=float(
                            resolution_cfg.get("decision_min_confidence", 0.75)
                        ),
                        same_identity_probability_threshold=float(
                            resolution_cfg.get(
                                "same_identity_probability_threshold", 0.70
                            )
                        ),
                        different_identity_probability_threshold=float(
                            resolution_cfg.get(
                                "different_identity_probability_threshold", 0.25
                            )
                        ),
                        cluster_review_rounds=int(
                            resolution_cfg.get("cluster_review_rounds", 2)
                        ),
                        semantic_attempts=int(resolution_cfg.get("semantic_attempts", 2)),
                        max_concurrency=int(
                            resolution_cfg.get("max_concurrency", 4)
                        ),
                    ),
                    token_counter=self.token_counter,
                    max_input_tokens=self.chunking_config.usable_model_input_tokens,
                    checkpoint_dir=self.stages_dir / "03_entity_resolution_batches",
                )
                entity_registry = await resolver.resolve(reviewed_records)
                canonical_records = canonicalize_scene_records(
                    reviewed_records, entity_registry
                )
                atomic_write_json(registry_path, entity_registry)
                atomic_write_json(canonical_path, canonical_records)
            canonical_records, reference_finalization_audit = (
                finalize_narrative_references(canonical_records, entity_registry)
            )
            atomic_write_json(canonical_path, canonical_records)
            atomic_write_json(
                self.stages_dir / "04_reference_finalization_audit.json",
                reference_finalization_audit,
            )
            self._complete_stage("03_entity_resolution")

            quality_cfg = _required_dict(self.config, "kg_quality")
            relation_registry_path = self.stages_dir / "05_relation_registry.json"
            if relation_registry_path.exists():
                relation_registry = load_json(relation_registry_path)
            else:
                relation_resolver = RelationResolver(
                    movie_id=self.movie_id,
                    llm_client=llm_client,
                    config=RelationResolutionConfig(
                        batch_size=int(quality_cfg.get("relation_batch_size", 16)),
                        semantic_attempts=int(quality_cfg.get("semantic_attempts", 2)),
                        max_concurrency=int(quality_cfg.get("max_concurrency", 4)),
                    ),
                    token_counter=self.token_counter,
                    max_input_tokens=self.chunking_config.usable_model_input_tokens,
                    checkpoint_dir=self.stages_dir / "05_relation_review_batches",
                )
                relation_registry = await relation_resolver.resolve(canonical_records)
                atomic_write_json(relation_registry_path, relation_registry)
            self._complete_stage("04_relation_resolution")

            issues = build_quality_issues(
                entity_registry=entity_registry,
                relation_registry=relation_registry,
                scene_records=canonical_records,
            )
            global_review_path = self.stages_dir / "06_global_kg_review.json"
            if global_review_path.exists():
                global_review = load_json(global_review_path)
            else:
                quality_reviewer = KGQualityReviewer(
                    llm_client=llm_client,
                    config=KGQualityConfig(
                        review_batch_size=int(quality_cfg.get("review_batch_size", 16)),
                        semantic_attempts=int(quality_cfg.get("semantic_attempts", 2)),
                        max_concurrency=int(quality_cfg.get("max_concurrency", 4)),
                    ),
                    token_counter=self.token_counter,
                    max_input_tokens=self.chunking_config.usable_model_input_tokens,
                )
                global_review = await quality_reviewer.review(issues)
                atomic_write_json(global_review_path, global_review)
            self._complete_stage("05_global_kg_review")

            semantic_kg = build_semantic_kg(
                movie_id=self.movie_id,
                scene_records=canonical_records,
                entity_registry=entity_registry,
                relation_registry=relation_registry,
            )
            quality_report = build_kg_quality_report(
                raw_scene_records=extraction,
                reviewed_scene_records=reviewed_records,
                canonical_scene_records=canonical_records,
                entity_registry=entity_registry,
                relation_registry=relation_registry,
                issues=issues,
                global_review=global_review,
            )
            atomic_write_json(self.run_dir / "kg_quality_report.json", quality_report)
            human_review_decisions = [
                item
                for item in global_review.get("decisions", [])
                if item.get("disposition") == "human_review_required"
            ]
            issue_by_id = {item["issue_id"]: item for item in issues}
            atomic_write_json(
                self.run_dir / "human_review_required.json",
                {
                    "schema_version": "stage_human_review_queue_v1",
                    "items": [
                        {
                            "issue": issue_by_id.get(item.get("issue_id")),
                            "decision": item,
                        }
                        for item in human_review_decisions
                    ],
                },
            )
            atomic_write_json(
                self.run_dir / "named_character_review_queue.json",
                build_named_character_review_queue(entity_registry),
            )
            atomic_write_json(
                self.run_dir / "semantic_kg_candidate.json", semantic_kg
            )
            require_quality(quality_report)
            atomic_write_json(self.run_dir / "semantic_kg.json", semantic_kg)
            self._write_kg_release_provenance()
            self._complete_stage("06_kg_quality_gate")

            if self.kg_only:
                self._finalize_kg_only(quality_report)
                return self.run_dir

            hierarchy_cfg = _required_dict(self.config, "hierarchy")
            hierarchy = NarrativeHierarchyBuilder(
                movie_id=self.movie_id,
                llm_client=llm_client,
                config=HierarchyConfig(
                    language=self.language,
                    relation_candidate_window=int(
                        hierarchy_cfg.get("relation_candidate_window", 8)
                    ),
                    relation_batch_size=int(hierarchy_cfg.get("relation_batch_size", 12)),
                    relation_min_confidence=float(
                        hierarchy_cfg.get("relation_min_confidence", 0.55)
                    ),
                    semantic_attempts=int(hierarchy_cfg.get("semantic_attempts", 2)),
                    max_concurrency=int(hierarchy_cfg.get("max_concurrency", 4)),
                ),
                token_counter=self.token_counter,
                max_input_tokens=self.chunking_config.usable_model_input_tokens,
                checkpoint_root=self.stages_dir,
                entity_neighbors=build_entity_neighbor_index(relation_registry),
            )

            episode_path = self.stages_dir / "07_episodes.json"
            if episode_path.exists():
                episode_stage = load_json(episode_path)
                episodes = episode_stage["episodes"]
            else:
                episodes, audit = await hierarchy.build_episodes(canonical_records)
                episode_stage = {"episodes": episodes, "audit": audit}
                atomic_write_json(episode_path, episode_stage)
            self._complete_stage("07_episodes")

            relation_path = self.stages_dir / "08_episode_dag.json"
            if relation_path.exists():
                relation_stage = load_json(relation_path)
                episode_relations = relation_stage["relations"]
            else:
                episode_relations, audit = await hierarchy.build_episode_relations(episodes)
                relation_stage = {"relations": episode_relations, "audit": audit}
                atomic_write_json(relation_path, relation_stage)
            self._complete_stage("08_episode_dag")

            storyline_path = self.stages_dir / "09_storylines.json"
            if storyline_path.exists():
                storyline_stage = load_json(storyline_path)
                storylines = storyline_stage["storylines"]
            else:
                storylines, audit = await hierarchy.build_storylines(
                    episodes, episode_relations
                )
                storyline_stage = {"storylines": storylines, "audit": audit}
                atomic_write_json(storyline_path, storyline_stage)
            self._complete_stage("09_storylines")

            graph = build_narrative_graph(
                movie_id=self.movie_id,
                scene_records=canonical_records,
                entity_registry=entity_registry,
                episodes=episodes,
                episode_relations=episode_relations,
                storylines=storylines,
                relation_registry=relation_registry,
            )
            report = validate_graph(
                graph=graph,
                scene_records=canonical_records,
                episodes=episodes,
                episode_relations=episode_relations,
                storylines=storylines,
            )
            atomic_write_json(self.run_dir / "validation_report.json", report)
            require_valid(report)
            atomic_write_json(self.run_dir / "narrative_graph.json", graph)

            manifest = self._load_manifest()
            manifest["status"] = "completed"
            manifest["completed_at"] = _now()
            manifest["counts"] = report["counts"]
            manifest["outputs"] = {
                "graph": str(self.run_dir / "narrative_graph.json"),
                "validation": str(self.run_dir / "validation_report.json"),
                "semantic_kg": str(self.run_dir / "semantic_kg.json"),
                "kg_quality": str(self.run_dir / "kg_quality_report.json"),
                "kg_release_provenance": str(
                    self.run_dir / "kg_release_provenance.json"
                ),
                "named_character_review_queue": str(
                    self.run_dir / "named_character_review_queue.json"
                ),
                "stages": str(self.stages_dir),
            }
            self._write_manifest(manifest)
            latest_path = self.output_root / self.movie_id / "latest.json"
            atomic_write_json(
                latest_path,
                {
                    "movie_id": self.movie_id,
                    "run_id": self.run_id,
                    "run_dir": str(self.run_dir),
                    "completed_at": manifest["completed_at"],
                    "input_sha256": self._input_hash,
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

    async def _run_extraction(self, llm_client: Any) -> list[dict[str, Any]]:
        combined_path = self.stages_dir / "01_extraction.json"
        if combined_path.exists():
            payload = load_json(combined_path)
            return payload["scenes"]
        extraction_cfg = _required_dict(self.config, "extraction")
        extractor = ScreenplayExtractor(
            client=llm_client,
            config=ExtractionConfig(
                language=self.language,
                semantic_attempts=int(extraction_cfg.get("semantic_attempts", 2)),
                max_concurrency=int(extraction_cfg.get("max_concurrency", 4)),
            ),
            movie_id=self.movie_id,
            token_counter=self.token_counter,
            chunking_config=self.chunking_config,
        )
        scene_dir = self.stages_dir / "01_extraction_scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)

        async def extract_or_load(scene: Any) -> dict[str, Any]:
            checkpoint = scene_dir / f"{scene.order:04d}.json"
            if checkpoint.exists():
                record = load_json(checkpoint)
                if record.get("scene", {}).get("scene_id") != scene.scene_id:
                    raise ValueError(f"Scene checkpoint mismatch: {checkpoint}")
                return record
            record = await extractor.extract_scene(scene)
            atomic_write_json(checkpoint, record)
            return record

        records = await asyncio.gather(*(extract_or_load(scene) for scene in self.scenes))
        records.sort(key=lambda record: int(record["scene"]["order"]))
        atomic_write_json(combined_path, {"scenes": records})
        return records

    async def _run_scene_graph_review(
        self, extraction: list[dict[str, Any]], llm_client: Any
    ) -> list[dict[str, Any]]:
        combined_path = self.stages_dir / "02_scene_graph_review.json"
        if combined_path.exists():
            return load_json(combined_path)["scenes"]
        quality_cfg = _required_dict(self.config, "kg_quality")
        reviewer = SceneGraphReviewer(
            movie_id=self.movie_id,
            llm_client=llm_client,
            token_counter=self.token_counter,
            max_input_tokens=self.chunking_config.usable_model_input_tokens,
            config=SceneReviewConfig(
                language=self.language,
                semantic_attempts=int(quality_cfg.get("semantic_attempts", 2)),
            ),
        )
        checkpoint_dir = self.stages_dir / "02_scene_graph_review_scenes"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(max(1, int(quality_cfg.get("max_concurrency", 4))))

        async def review_or_load(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            order = int(record["scene"]["order"])
            checkpoint = checkpoint_dir / f"{order:04d}.json"
            if checkpoint.exists():
                payload = load_json(checkpoint)
                if payload.get("record", {}).get("scene", {}).get("scene_id") != record["scene"]["scene_id"]:
                    raise ValueError(f"Scene review checkpoint mismatch: {checkpoint}")
                return payload["record"], payload["audit"]
            async with semaphore:
                reviewed, audit = await reviewer.review_scene(record)
            atomic_write_json(checkpoint, {"record": reviewed, "audit": audit})
            return reviewed, audit

        results = await asyncio.gather(*(review_or_load(record) for record in extraction))
        results.sort(key=lambda item: int(item[0]["scene"]["order"]))
        records = [item[0] for item in results]
        audits = [item[1] for item in results]
        atomic_write_json(combined_path, {"scenes": records, "audits": audits})
        return records

    def _finalize_kg_only(self, quality_report: dict[str, Any]) -> None:
        manifest = self._load_manifest()
        manifest["status"] = "kg_completed"
        manifest["completed_at"] = _now()
        manifest["counts"] = quality_report["counts"]
        manifest["outputs"] = {
            "semantic_kg": str(self.run_dir / "semantic_kg.json"),
            "kg_quality": str(self.run_dir / "kg_quality_report.json"),
            "kg_release_provenance": str(
                self.run_dir / "kg_release_provenance.json"
            ),
            "named_character_review_queue": str(
                self.run_dir / "named_character_review_queue.json"
            ),
            "stages": str(self.stages_dir),
        }
        self._write_manifest(manifest)
        atomic_write_json(
            self.output_root / self.movie_id / "latest_kg.json",
            {
                "movie_id": self.movie_id,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "completed_at": manifest["completed_at"],
                "input_sha256": self._input_hash,
                "config_sha256": self._config_hash,
                "code_sha256": self._code_hash,
                "quality_status": quality_report["status"],
            },
        )

    def _write_kg_release_provenance(self) -> None:
        artifact_paths = {
            "01_extraction": self.stages_dir / "01_extraction.json",
            "02_scene_graph_review": self.stages_dir / "02_scene_graph_review.json",
            "03_entity_registry": self.stages_dir / "03_entity_registry.json",
            "04_canonicalized_extraction": self.stages_dir
            / "04_canonicalized_extraction.json",
            "05_relation_registry": self.stages_dir / "05_relation_registry.json",
            "06_global_kg_review": self.stages_dir / "06_global_kg_review.json",
            "kg_quality_report": self.run_dir / "kg_quality_report.json",
            "semantic_kg": self.run_dir / "semantic_kg.json",
        }
        existing = {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
            if path.is_file()
        }
        llm_config = self.config.get("llm")
        if not isinstance(llm_config, dict):
            llm_config = {"model": "injected_test_or_runtime_client"}
        provenance = {
            "schema_version": "stage_kg_release_provenance_v1",
            "created_at": _now(),
            "movie_id": self.movie_id,
            "run_id": self.run_id,
            "input": {
                "script_path": str(self.script_path),
                "script_sha256": self._input_hash,
            },
            "config": {
                "path": str(self.config_path),
                "sha256": self._config_hash,
                "llm": {
                    key: llm_config.get(key)
                    for key in (
                        "model",
                        "base_url",
                        "temperature",
                        "max_tokens",
                    )
                    if key in llm_config
                },
                "api_key_excluded": True,
            },
            "code_sha256": self._code_hash,
            "prompt_hashes": _prompt_hashes(),
            "prompt_source_files": {
                str(path): sha256_file(path)
                for path in (*PROMPTS.source_paths, *prompt_asset_paths())
            },
            "local_id_policies": {
                "scene_review": "E#### entities, R#### relations, X#### references",
                "entity_pair_review": "P###### mention pairs",
                "entity_cluster_review": "C##-###### cluster pairs",
                "predicate_review": "stable relation-observation-* ids",
                "global_review": "stable kg-issue-* ids",
            },
            "stages": {
                "01_extraction": {
                    "input_sha256": self._input_hash,
                    "output": existing.get("01_extraction"),
                    "prompt_hashes": [
                        "narrative_extraction",
                        "entity_extraction",
                        "relation_extraction",
                    ],
                    "call_metadata_path": "stages/01_extraction_scenes/*.json",
                },
                "02_scene_graph_review": {
                    "input_sha256": existing.get("01_extraction", {}).get("sha256"),
                    "output": existing.get("02_scene_graph_review"),
                    "prompt_hashes": ["scene_graph_review"],
                    "call_metadata_path": "stages/02_scene_graph_review_scenes/*.json",
                    "input_policy": "minimal item records and evidence snippets only",
                },
                "03_entity_resolution": {
                    "input_sha256": existing.get("02_scene_graph_review", {}).get(
                        "sha256"
                    ),
                    "outputs": [
                        existing.get("03_entity_registry"),
                        existing.get("04_canonicalized_extraction"),
                    ],
                    "prompt_hashes": ["entity_resolution"],
                    "call_metadata_path": "stages/03_entity_registry.json#audit",
                },
                "04_relation_resolution": {
                    "input_sha256": existing.get(
                        "04_canonicalized_extraction", {}
                    ).get("sha256"),
                    "output": existing.get("05_relation_registry"),
                    "prompt_hashes": ["predicate_review"],
                    "call_metadata_path": "stages/05_relation_registry.json#audit",
                },
                "05_global_kg_review": {
                    "input_sha256": sha256_json(
                        {
                            "entities": existing.get("03_entity_registry", {}).get(
                                "sha256"
                            ),
                            "relations": existing.get("05_relation_registry", {}).get(
                                "sha256"
                            ),
                        }
                    ),
                    "output": existing.get("06_global_kg_review"),
                    "prompt_hashes": ["global_kg_review"],
                    "call_metadata_path": "stages/06_global_kg_review.json#llm_calls",
                },
                "06_kg_quality_gate": {
                    "input_sha256": sha256_json(
                        {
                            name: item["sha256"]
                            for name, item in existing.items()
                            if name.startswith(("03_", "04_", "05_", "06_"))
                        }
                    ),
                    "outputs": [
                        existing.get("kg_quality_report"),
                        existing.get("semantic_kg"),
                    ],
                    "deterministic": True,
                },
            },
        }
        atomic_write_json(self.run_dir / "kg_release_provenance.json", provenance)

    def _prepare_run(self) -> None:
        if self.resume and not self.run_dir.exists():
            raise FileNotFoundError(f"Cannot resume missing run directory: {self.run_dir}")
        if self.run_dir.exists() and not self.resume:
            raise FileExistsError(
                f"Run directory already exists; use --resume or a new --run-id: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest = self._load_manifest()
            if manifest.get("input_sha256") != self._input_hash:
                raise ValueError("Cannot resume: screenplay checksum changed")
            if manifest.get("config_sha256") != self._config_hash:
                raise ValueError("Cannot resume: config checksum changed")
            if manifest.get("code_sha256") != self._code_hash:
                manifest.setdefault("code_revisions", []).append(
                    {
                        "registered_at": _now(),
                        "from_code_sha256": manifest.get("code_sha256"),
                        "to_code_sha256": self._code_hash,
                        "checkpoint_reuse_policy": (
                            "Reuse successful scene/stage checkpoints when their input and "
                            "prompt hashes remain valid; rerun incomplete stages and deterministic "
                            "validation under the current code."
                        ),
                        "from_code_file_sha256": manifest.get("code_file_sha256", {}),
                        "to_code_file_sha256": self._code_hashes,
                    }
                )
                manifest["code_sha256"] = self._code_hash
                manifest["code_file_sha256"] = self._code_hashes
            if manifest.get("prompt_hashes") != _prompt_hashes():
                raise ValueError("Cannot resume: prompt content checksum changed")
            if manifest.get("run_mode") != ("kg_only" if self.kg_only else "full"):
                raise ValueError("Cannot resume: run mode changed")
            prior_status = manifest.get("status")
            if prior_status in {"failed", "paused"}:
                attempt_history = list(manifest.get("attempt_history", []))
                attempt_history.append(
                    {
                        "status": prior_status,
                        "resumed_at": manifest.get("resumed_at"),
                        "paused_at": manifest.get("paused_at"),
                        "failed_at": manifest.get("failed_at"),
                        "error": manifest.get("error"),
                    }
                )
                manifest["attempt_history"] = attempt_history
            for stale_key in ("failed_at", "error", "traceback"):
                manifest.pop(stale_key, None)
            manifest["status"] = "running"
            manifest["resumed_at"] = _now()
            self._write_manifest(manifest)
            return
        manifest = {
            "schema_version": "stage_narrative_run_v1",
            "status": "running",
            "started_at": _now(),
            "movie_id": self.movie_id,
            "language": self.language,
            "script_path": str(self.script_path),
            "input_sha256": self._input_hash,
            "scene_count": len(self.scenes),
            "config_path": str(self.config_path),
            "config_snapshot": str(self.run_dir / "config_snapshot.json"),
            "config_sha256": self._config_hash,
            "code_sha256": self._code_hash,
            "code_file_sha256": self._code_hashes,
            "run_id": self.run_id,
            "run_mode": "kg_only" if self.kg_only else "full",
            "run_dir": str(self.run_dir),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "prompt_hashes": _prompt_hashes(),
            "prompt_source_files": {
                str(path): sha256_file(path)
                for path in (*PROMPTS.source_paths, *prompt_asset_paths())
            },
            "tokenizer": self.token_counter.metadata,
            "chunking_budgets": {
                "context_window": self.chunking_config.context_window,
                "max_output_tokens": self.chunking_config.max_output_tokens,
                "safety_margin_tokens": self.chunking_config.safety_margin_tokens,
                "max_model_input_tokens": self.chunking_config.max_model_input_tokens,
                "usable_model_input_tokens": self.chunking_config.usable_model_input_tokens,
                "prompt_headroom_tokens": self.chunking_config.prompt_headroom_tokens,
                "target_chunk_tokens": self.chunking_config.target_chunk_tokens,
                "carry_context_max_tokens": self.chunking_config.carry_context_max_tokens,
                "reconciliation_max_input_tokens": self.chunking_config.reconciliation_max_input_tokens,
            },
            "completed_stages": [],
        }
        atomic_write_json(self.run_dir / "config_snapshot.json", self.config)
        self._write_manifest(manifest)

    def _complete_stage(self, stage: str) -> None:
        manifest = self._load_manifest()
        stages = list(manifest.get("completed_stages", []))
        if stage not in stages:
            stages.append(stage)
        manifest["completed_stages"] = stages
        manifest["updated_at"] = _now()
        self._write_manifest(manifest)

    def _load_manifest(self) -> dict[str, Any]:
        return load_json(self.manifest_path)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config requires object field: {key}")
    return value


def _resolve_language(value: Any, script_path: Path) -> str:
    language = str(value or "auto").strip()
    if language.casefold() != "auto":
        return language
    parents = {parent.name.casefold() for parent in script_path.parents}
    if "chinese" in parents:
        return "Chinese"
    if "english" in parents:
        return "English"
    raise ValueError("language=auto requires the screenplay to be under Chinese/ or English/")


def _prompt_hashes() -> dict[str, str]:
    def prompt_text(*prompt_ids: str) -> str:
        return "".join(
            PROMPTS.get(prompt_id).system + PROMPTS.get(prompt_id).user
            for prompt_id in prompt_ids
        )

    extraction_assets = "".join(
        path.read_text(encoding="utf-8") for path in prompt_asset_paths()
    )
    prompts = {
        "entity_extraction": prompt_text("entity_extraction", "entity_repair")
        + extraction_assets,
        "relation_extraction": prompt_text(
            "relation_extraction", "relation_repair"
        ) + extraction_assets,
        "narrative_extraction": prompt_text(
            "narrative_extraction", "narrative_repair"
        ) + extraction_assets,
        "scene_reconciliation": prompt_text("scene_reconciliation"),
        "scene_graph_review": prompt_text("scene_graph_review"),
        "entity_resolution": prompt_text(
            "entity_pair_review", "entity_cluster_review"
        ),
        "predicate_review": prompt_text(
            "predicate_review", "predicate_schema_repair"
        ),
        "global_kg_review": prompt_text("global_kg_review"),
        "episode": prompt_text("episode", "episode_repair"),
        "episode_relation": prompt_text("episode_relation"),
        "storyline": prompt_text("storyline"),
    }
    return {
        name: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for name, prompt in prompts.items()
    }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
