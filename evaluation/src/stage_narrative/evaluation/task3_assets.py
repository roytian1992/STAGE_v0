from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..clients import build_endpoint_pool_runtime, build_json_client
from ..io import atomic_write_json, load_json, sha256_file, sha256_json
from ..prompt_loader import PROMPTS
from ..temporal.benchmark_protocol import BenchmarkRuntimeConfig, normalize_language
from .schemas import PAIR_TYPES


async def build_task3_pair_annotation_draft(
    *,
    instances_path: Path,
    gold_rubrics_path: Path,
    pair_groups_path: Path,
    config_path: Path,
    output_dir: Path,
    workers: int = 16,
    resume: bool = False,
    preflight_only: bool = False,
    client: Any | None = None,
) -> Path:
    if workers <= 0 or workers > 64:
        raise ValueError("Task 3 pair-annotation workers must be in 1..64")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite Task 3 pair annotation draft: {output_dir}")
    partial_dir = output_dir / "partials"
    failure_dir = output_dir / "failures"
    for path in (output_dir, partial_dir, failure_dir):
        path.mkdir(parents=True, exist_ok=True)
    instances = _object(instances_path)
    gold = _object(gold_rubrics_path)
    pairs = _object(pair_groups_path)
    movie_ids = {str(payload["movie_id"]) for payload in (instances, gold, pairs)}
    if len(movie_ids) != 1:
        raise ValueError(f"Task 3 pair-annotation inputs differ: {sorted(movie_ids)}")
    movie_id = next(iter(movie_ids))
    instance_by_id = {row["instance_id"]: row for row in instances["instances"]}
    gold_by_id = {row["instance_id"]: row for row in gold["rubrics"]}
    if set(instance_by_id) != set(gold_by_id):
        raise ValueError("Task 3 pair annotation requires exact instance/gold coverage")
    config = BenchmarkRuntimeConfig.load(config_path)
    counter = config.build_token_counter()
    language = normalize_language(instances.get("language") or _infer_language(instances))
    specs = {
        row["pair_group_id"]: _materialize_pair_annotation(
            row=row,
            instance_by_id=instance_by_id,
            gold_by_id=gold_by_id,
            language=language,
            config=config,
            counter=counter,
        )
        for row in pairs["pair_groups"]
    }
    contract = {
        "schema_version": "stage_task3_pair_annotation_contract",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "movie_id": movie_id,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for path in (instances_path, gold_rubrics_path, pair_groups_path)
        ],
        "config": {"path": str(config.source_path), "sha256": config.source_sha256},
        "model": config.evaluation_llm["model"],
        "prompt_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in PROMPTS.get(f"{language}/evaluation/task3_pair_annotation").source_paths
        ],
        "preflight": {
            "pair_count": len(specs),
            "max_input_tokens": max(row["prompt_tokens"] for row in specs.values()),
            "truncated_count": 0,
        },
    }
    contract_path = output_dir / "run_contract.json"
    if resume and contract_path.is_file():
        existing = _object(contract_path)
        if _without_created_at(existing) != _without_created_at(contract):
            raise ValueError("Task 3 pair annotation resume contract drift")
    else:
        atomic_write_json(contract_path, contract)
    if preflight_only:
        path = output_dir / "preflight.json"
        atomic_write_json(
            path,
            {
                "schema_version": "stage_task3_pair_annotation_preflight",
                "status": "passed_zero_call",
                "run_contract_sha256": sha256_file(contract_path),
                **contract["preflight"],
            },
        )
        return path

    endpoint_runtime = None
    if client is None:
        endpoint_runtime = build_endpoint_pool_runtime(config.evaluation_llm)
        client = build_json_client(
            {**config.evaluation_llm, "json_response_format": True, "max_tokens": 1024},
            endpoint_runtime=endpoint_runtime,
        )
    semaphore = asyncio.Semaphore(workers)

    async def one(pair_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        path = partial_dir / f"{pair_id}.json"
        if resume and path.is_file():
            existing = _object(path)
            if existing.get("prompt_sha256") != spec["prompt_sha256"]:
                raise ValueError(f"Task 3 pair annotation prompt drift: {path}")
            return existing
        try:
            async with semaphore:
                response = await client.generate_json(
                    system_prompt=spec["system_prompt"],
                    user_prompt=spec["user_prompt"],
                    stage=f"task3_pair_annotation:{pair_id}",
                )
            judgment = _validate_pair_annotation(response.data)
            result = {
                "pair_group_id": pair_id,
                "pair_type": judgment["pair_type"],
                "ordered_instance_ids": spec["ordered_instance_ids"],
                "checkpoint_scene_orders": spec["checkpoint_scene_orders"],
                "expected_direction": judgment["expected_direction"],
                "comparability": judgment["comparability"],
                "brief_rationale": judgment["brief_rationale"],
                "prompt_sha256": spec["prompt_sha256"],
                "prompt_tokens": spec["prompt_tokens"],
                "generator_metadata": response.metadata,
            }
            atomic_write_json(path, result)
            return result
        except Exception as exc:
            atomic_write_json(
                failure_dir / f"{pair_id}.json",
                {
                    "pair_group_id": pair_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "prompt_sha256": spec["prompt_sha256"],
                    "parsed_response": getattr(exc, "parsed_response", None),
                },
            )
            raise

    settled = await asyncio.gather(
        *(one(pair_id, spec) for pair_id, spec in specs.items()), return_exceptions=True
    )
    failures = [row for row in settled if isinstance(row, BaseException)]
    if failures:
        raise RuntimeError(
            f"{len(failures)} Task 3 pair annotations failed; first: "
            f"{type(failures[0]).__name__}: {failures[0]}"
        )
    rows = [row for row in settled if isinstance(row, dict)]
    draft_path = output_dir / "task3_pair_annotations_draft.json"
    atomic_write_json(
        draft_path,
        {
            "schema_version": "stage_task3_pair_annotations_draft",
            "movie_id": movie_id,
            "review_status": "requires_manual_review",
            "pair_count": len(rows),
            "pairs": sorted(rows, key=lambda row: row["pair_group_id"]),
        },
    )
    manifest_path = output_dir / "manifest.json"
    endpoint_snapshot = await endpoint_runtime.snapshot() if endpoint_runtime else None
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task3_pair_annotation_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "machine_draft_requires_manual_review",
            "movie_id": movie_id,
            "python_executable": sys.executable,
            "model": config.evaluation_llm["model"],
            "endpoint_pool": endpoint_snapshot,
            "counts": {"pairs": len(rows), "failure_count": 0, "truncated_count": 0},
            "run_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            "outputs": [{"path": str(draft_path), "sha256": sha256_file(draft_path)}],
        },
    )
    return manifest_path


def promote_task3_pair_annotations(
    *, draft_path: Path, reviewer_id: str, output_path: Path
) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Task 3 pair annotations: {output_path}")
    if not reviewer_id.strip():
        raise ValueError("Task 3 pair annotations require a reviewer ID")
    draft = _object(draft_path)
    pairs = []
    for row in draft["pairs"]:
        _validate_pair_annotation(row)
        pairs.append(
            {
                key: row[key]
                for key in (
                    "pair_group_id",
                    "pair_type",
                    "ordered_instance_ids",
                    "checkpoint_scene_orders",
                    "expected_direction",
                    "comparability",
                )
            }
        )
    atomic_write_json(
        output_path,
        {
            "schema_version": "stage_task3_pair_annotations",
            "movie_id": draft["movie_id"],
            "reviewer_id": reviewer_id.strip(),
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "pair_count": len(pairs),
            "pairs": pairs,
        },
    )
    return output_path


def _materialize_pair_annotation(
    *, row: dict[str, Any], instance_by_id: dict[str, Any], gold_by_id: dict[str, Any],
    language: str, config: BenchmarkRuntimeConfig, counter: Any
) -> dict[str, Any]:
    ordered = sorted(
        row["instance_ids"],
        key=lambda instance_id: int(
            _checkpoint_boundary(instance_by_id[instance_id])["scene_order"]
        ),
    )
    cases = []
    for instance_id in ordered:
        instance = instance_by_id[instance_id]
        cases.append(
            {
                "instance_id": instance_id,
                "checkpoint": _checkpoint_boundary(instance),
                "user_turn": instance["model_input"]["current_user_turn"],
                "rubric": gold_by_id[instance_id]["rubric"],
            }
        )
    characters = {gold_by_id[value]["character"] for value in ordered}
    if len(characters) != 1:
        raise ValueError(f"Task 3 pair spans characters: {row['pair_group_id']}")
    path = f"{language}/evaluation/task3_pair_annotation"
    system, user = PROMPTS.render(
        path,
        character=next(iter(characters)),
        source_pair_type=row["pair_type"],
        earlier_case=cases[0],
        later_case=cases[1],
    )
    tokens = counter.count(system) + counter.count(user) + config.reserved_chat_template_tokens
    maximum = config.call_budgets["task3_pair_judge"].max_input_tokens
    if tokens > maximum:
        raise ValueError(f"Task 3 pair annotation exceeds budget: {tokens}>{maximum}")
    return {
        "system_prompt": system,
        "user_prompt": user,
        "prompt_tokens": tokens,
        "prompt_sha256": sha256_json({"system": system, "user": user}),
        "ordered_instance_ids": ordered,
        "checkpoint_scene_orders": [
            int(_checkpoint_boundary(instance_by_id[value])["scene_order"])
            for value in ordered
        ],
    }


def _validate_pair_annotation(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"pair_type", "expected_direction", "comparability", "brief_rationale"}
    optional = {
        "pair_group_id", "ordered_instance_ids", "checkpoint_scene_orders",
        "prompt_sha256", "prompt_tokens", "generator_metadata"
    }
    if not isinstance(payload, dict) or not required <= set(payload) or set(payload) - required - optional:
        raise ValueError("Task 3 pair annotation fields are invalid")
    if payload["pair_type"] not in PAIR_TYPES:
        raise ValueError("Task 3 pair annotation has an invalid pair type")
    for field in ("expected_direction", "comparability", "brief_rationale"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"Task 3 pair annotation requires nonempty {field}")
    return payload


def _infer_language(instances: dict[str, Any]) -> str:
    first = instances["instances"][0]
    text = str(first["model_input"]["current_user_turn"])
    return "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en"


def _checkpoint_boundary(instance: dict[str, Any]) -> dict[str, Any]:
    return instance.get("checkpoint_boundary") or instance["model_input"]["checkpoint_anchor"]


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _without_created_at(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at"}
