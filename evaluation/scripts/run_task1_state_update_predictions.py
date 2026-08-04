#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_task1_state_update_assets import validate_assets  # noqa: E402
from stage_narrative.chunking import chunk_scene  # noqa: E402
from stage_narrative.clients import (  # noqa: E402
    ModelResponseParseError,
    build_endpoint_pool_runtime,
    build_json_client,
)
from stage_narrative.evaluation.task1_schemas import (  # noqa: E402
    validate_task1_prediction,
)
from stage_narrative.io import (  # noqa: E402
    atomic_write_json,
    load_json,
    load_scenes,
    sha256_file,
    sha256_json,
)
from stage_narrative.models import stable_id  # noqa: E402
from stage_narrative.prompt_loader import PROMPTS  # noqa: E402
from stage_narrative.temporal.benchmark_protocol import (  # noqa: E402
    BenchmarkRuntimeConfig,
)


MEMORY_PROMPT_NAME = "task1_entity_memory_extraction"
MEMORY_CONSOLIDATION_PROMPT_NAME = "task1_entity_memory_consolidation"
PREDICTION_PROMPT_NAME = "task1_state_update_prediction"
DEFAULT_CHUNK_CONTENT_TOKENS = 600
MAX_CONSOLIDATED_OBSERVATIONS = 64
MAX_BATCH_CONSOLIDATED_OBSERVATIONS = 32
CONSOLIDATION_BATCH_INPUT_TARGET = 14000
CONSOLIDATION_TRIGGER_INPUT_TOKENS = 12000
INTERNAL_ID = re.compile(
    r"(?:task1-(?:state|development|instance)|checkpoint|entity)-[0-9a-f]{8,}",
    re.IGNORECASE,
)


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Task 1 reference- and autoregressive predictions with "
            "one shared runtime entity-centric memory."
        )
    )
    parser.add_argument("--reference-asset", type=Path, required=True)
    parser.add_argument("--autoregressive-asset", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--chunk-content-tokens",
        type=int,
        default=DEFAULT_CHUNK_CONTENT_TOKENS,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("--workers must be in 1..64")
    if args.chunk_content_tokens <= 0:
        raise ValueError("--chunk-content-tokens must be positive")

    reference_path = args.reference_asset.resolve()
    autoregressive_path = args.autoregressive_asset.resolve()
    script_path = args.script.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite prediction run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    reference_asset = _object(reference_path)
    autoregressive_asset = _object(autoregressive_path)
    audit = validate_assets(
        reference=reference_asset, autoregressive=autoregressive_asset
    )
    if audit["status"] != "passed":
        raise ValueError(f"State update assets failed validation: {audit}")
    language = str(reference_asset.get("language") or "")
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported Task 1 language: {language}")
    if sha256_file(script_path) != reference_asset["script"]["sha256"]:
        raise ValueError("Runtime screenplay differs from the state_update assets")

    config = BenchmarkRuntimeConfig.load(args.config.resolve())
    counter = config.build_token_counter()
    scenes = load_scenes(script_path)
    memory_jobs = _memory_jobs(
        reference_asset,
        scenes=scenes,
        counter=counter,
        chunk_content_tokens=args.chunk_content_tokens,
    )
    preflight = _preflight(
        reference_asset=reference_asset,
        autoregressive_asset=autoregressive_asset,
        memory_jobs=memory_jobs,
        config=config,
        counter=counter,
    )
    if args.dry_run:
        payload = {
            "schema": "stage_task1_state_update_prediction_preflight",
            "status": "passed_zero_call",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "assets": {
                "reference": _artifact(reference_path),
                "autoregressive": _artifact(autoregressive_path),
                "script": _artifact(script_path),
            },
            "config": _artifact(args.config.resolve()),
            "prompt_artifacts": _prompt_artifacts(language),
            "workers": args.workers,
            "chunk_content_tokens": args.chunk_content_tokens,
            **preflight,
            "semantic_calls": 0,
        }
        atomic_write_json(output_root / "preflight.json", payload)
        print(output_root / "preflight.json")
        print(payload["counts"])
        return

    model_config = {
        **config.prediction_llm,
        "json_response_format": True,
        "max_tokens": config.call_budgets["task1_prediction"].max_output_tokens,
    }
    endpoint_runtime = build_endpoint_pool_runtime(model_config)
    client = build_json_client(model_config, endpoint_runtime=endpoint_runtime)
    semaphore = asyncio.Semaphore(args.workers)
    memory_pack = await _extract_shared_memory(
        jobs=memory_jobs,
        client=client,
        semaphore=semaphore,
        config=config,
        counter=counter,
        output_root=output_root,
        resume=args.resume,
    )
    memory_path = output_root / "shared_entity_memory.json"
    atomic_write_json(memory_path, memory_pack)

    reference_predictions = await _run_setting(
        asset=reference_asset,
        memory_pack=memory_pack,
        client=client,
        semaphore=semaphore,
        config=config,
        counter=counter,
        output_root=output_root / "reference_state_update",
        resume=args.resume,
    )
    shared_initial_predictions = {
        trajectory["checkpoints"][0]["checkpoint_id"]: trajectory["checkpoints"][0]
        for trajectory in reference_predictions["trajectories"]
    }
    autoregressive_predictions = await _run_setting(
        asset=autoregressive_asset,
        memory_pack=memory_pack,
        client=client,
        semaphore=semaphore,
        config=config,
        counter=counter,
        output_root=output_root / "autoregressive_state_update",
        resume=args.resume,
        seed_predictions=shared_initial_predictions,
    )
    reference_output = output_root / "reference_state_update_predictions.json"
    autoregressive_output = output_root / "autoregressive_state_update_predictions.json"
    atomic_write_json(reference_output, reference_predictions)
    atomic_write_json(autoregressive_output, autoregressive_predictions)
    endpoint_snapshot = (
        await endpoint_runtime.snapshot() if endpoint_runtime is not None else None
    )
    manifest = {
        "schema": "stage_task1_state_update_prediction_run",
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prediction_model": config.prediction_llm["model"],
        "evaluation_model_configured_but_unused": config.evaluation_llm["model"],
        "thinking_disabled": True,
        "workers": args.workers,
        "chunk_content_tokens": args.chunk_content_tokens,
        "previous_state_projection": "claim_text_only",
        "previous_state_provenance_policy": "runner_only_union_carry_forward",
        "context_protocol": reference_asset["context_protocol"],
        "shared_memory_sha256": sha256_file(memory_path),
        "counts": {
            "trajectories_per_setting": reference_asset["counts"]["trajectories"],
            "checkpoints_per_setting": reference_asset["counts"]["checkpoints"],
            "memory_slots": len(memory_pack["slots"]),
            "memory_extraction_calls": memory_pack["chunk_count"],
            "memory_consolidation_calls": memory_pack["consolidation_call_count"],
            "memory_consolidation_semantic_calls": memory_pack[
                "consolidation_semantic_call_count"
            ],
            "memory_raw_observations": memory_pack["raw_observation_count"],
            "memory_final_observations": memory_pack["final_observation_count"],
            "memory_chunk_tokens": _chunk_token_statistics(memory_pack),
            "prediction_calls": (
                reference_asset["counts"]["checkpoints"]
                + autoregressive_asset["counts"]["checkpoints"]
                - len(shared_initial_predictions)
            ),
            "shared_initial_checkpoint_predictions": len(shared_initial_predictions),
            "failure_count": 0,
        },
        "inputs": {
            "reference_asset": _artifact(reference_path),
            "autoregressive_asset": _artifact(autoregressive_path),
            "script": _artifact(script_path),
            "config": _artifact(args.config.resolve()),
        },
        "outputs": {
            "shared_entity_memory": _artifact(memory_path),
            "reference_state_update_predictions": _artifact(reference_output),
            "autoregressive_state_update_predictions": _artifact(autoregressive_output),
        },
        "prompt_artifacts": _prompt_artifacts(language),
        "endpoint_pool": endpoint_snapshot,
        "fallbacks": {"content_filter": "none", "semantic": "none"},
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(output_root / "manifest.json")
    print(manifest["counts"])


def _memory_jobs(
    asset: dict[str, Any], *, scenes: list[Any], counter: Any, chunk_content_tokens: int
) -> list[dict[str, Any]]:
    jobs = []
    for trajectory in asset["trajectories"]:
        for checkpoint in trajectory["checkpoints"]:
            interval = checkpoint["screenplay_interval"]
            selected = scenes[
                int(interval["start_scene_order_exclusive"]) : int(
                    interval["end_scene_order_inclusive"]
                )
            ]
            if not selected:
                raise ValueError(f"Empty screenplay interval: {checkpoint['checkpoint_id']}")
            chunks = _pack_interval(
                movie_id=asset["movie_id"],
                scenes=selected,
                counter=counter,
                maximum=chunk_content_tokens,
            )
            jobs.append(
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "character": trajectory["character"],
                    "aliases": trajectory["aliases"],
                    "language": asset["language"],
                    "memory_slot": checkpoint["entity_memory"]["memory_slot"],
                    "interval": interval,
                    "chunks": chunks,
                }
            )
    return jobs


def _pack_interval(
    *, movie_id: str, scenes: list[Any], counter: Any, maximum: int
) -> list[dict[str, Any]]:
    units = []
    for scene in scenes:
        for part in chunk_scene(
            movie_id=movie_id,
            scene=scene,
            token_counter=counter,
            max_content_tokens=maximum,
        ):
            pieces = [f"Scene {scene.order}: {scene.title}"]
            if scene.subtitle:
                pieces.append(scene.subtitle)
            if part.text:
                pieces.append(part.text)
            text = "\n".join(pieces)
            units.append({"scene_order": scene.order, "text": text})
    packed: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for unit in units:
        candidate = "\n\n".join(row["text"] for row in [*current, unit])
        if current and counter.count(candidate) > maximum:
            packed.append(_packed_chunk(current, len(packed) + 1, counter))
            current = [unit]
        else:
            current.append(unit)
    if current:
        packed.append(_packed_chunk(current, len(packed) + 1, counter))
    return packed


def _packed_chunk(rows: list[dict[str, Any]], index: int, counter: Any) -> dict[str, Any]:
    text = "\n\n".join(row["text"] for row in rows)
    return {
        "chunk_index": index,
        "scene_orders": sorted({row["scene_order"] for row in rows}),
        "text": text,
        "content_tokens": counter.count(text),
    }


def _preflight(
    *,
    reference_asset: dict[str, Any],
    autoregressive_asset: dict[str, Any],
    memory_jobs: list[dict[str, Any]],
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> dict[str, Any]:
    maximum_memory = {"tokens": -1}
    chunk_count = 0
    for job in memory_jobs:
        for chunk in job["chunks"]:
            rendered = _render_memory(job, chunk, config=config, counter=counter)
            chunk_count += 1
            if rendered["accounted_input_tokens"] > maximum_memory["tokens"]:
                maximum_memory = {
                    "tokens": rendered["accounted_input_tokens"],
                    "character": job["character"],
                    "scene_orders": chunk["scene_orders"],
                }
    for asset in (reference_asset, autoregressive_asset):
        for trajectory in asset["trajectories"]:
            previous_prediction: list[dict[str, Any]] = []
            for checkpoint in trajectory["checkpoints"]:
                previous_state_rows = (
                    checkpoint["model_input"].get("previous_state", [])
                    if asset["setting"] == "reference_state_update"
                    else previous_prediction
                )
                previous_state = _previous_state_content(previous_state_rows)
                _assert_previous_state_safe(previous_state)
                previous_prediction = []
    return {
        "counts": {
            "trajectories_per_setting": reference_asset["counts"]["trajectories"],
            "checkpoints_per_setting": reference_asset["counts"]["checkpoints"],
            "memory_slots": len(memory_jobs),
            "memory_chunks": chunk_count,
            "planned_prediction_calls": (
                reference_asset["counts"]["checkpoints"]
                + autoregressive_asset["counts"]["checkpoints"]
                - autoregressive_asset["counts"]["trajectories"]
            ),
            "shared_initial_checkpoint_predictions": autoregressive_asset["counts"][
                "trajectories"
            ],
        },
        "maximum_memory_prompt": maximum_memory,
        "shared_memory_between_settings": True,
        "previous_state_internal_id_leakage": False,
        "previous_state_projection": "claim_text_only",
        "previous_state_provenance_policy": "runner_only_union_carry_forward",
    }


async def _extract_shared_memory(
    *, jobs: list[dict[str, Any]], client: Any, semaphore: asyncio.Semaphore,
    config: BenchmarkRuntimeConfig, counter: Any, output_root: Path, resume: bool,
) -> dict[str, Any]:
    async def run_job(job: dict[str, Any]) -> dict[str, Any]:
        settled = await asyncio.gather(
            *[
                _extract_chunk(
                    job=job, chunk=chunk, client=client, semaphore=semaphore,
                    config=config, counter=counter, output_root=output_root, resume=resume,
                )
                for chunk in job["chunks"]
            ]
        )
        observations = []
        seen = set()
        for row in settled:
            for observation in row["observations"]:
                key = " ".join(observation["claim"].casefold().split())
                if key not in seen:
                    observations.append(observation)
                    seen.add(key)
        raw_observation_count = len(observations)
        observations, consolidation = await _consolidate_memory(
            job=job,
            observations=observations,
            client=client,
            semaphore=semaphore,
            config=config,
            counter=counter,
            output_root=output_root,
            resume=resume,
        )
        return {
            "checkpoint_id": job["checkpoint_id"],
            "trajectory_id": job["trajectory_id"],
            "character": job["character"],
            "memory_slot": job["memory_slot"],
            "screenplay_interval": job["interval"],
            "raw_observation_count": raw_observation_count,
            "observations": observations,
            "consolidation": consolidation,
            "chunks": settled,
        }
    slots = await asyncio.gather(*(run_job(job) for job in jobs))
    return {
        "schema": "stage_task1_shared_entity_memory",
        "status": "completed",
        "slot_count": len(slots),
        "chunk_count": sum(len(row["chunks"]) for row in slots),
        "consolidation_call_count": sum(
            int(row["consolidation"]["call_count"]) for row in slots
        ),
        "consolidation_semantic_call_count": sum(
            int(row["consolidation"]["semantic_call_count"]) for row in slots
        ),
        "raw_observation_count": sum(row["raw_observation_count"] for row in slots),
        "final_observation_count": sum(len(row["observations"]) for row in slots),
        "previous_state_visible_to_extractor": False,
        "slots": slots,
    }


async def _consolidate_memory(
    *, job: dict[str, Any], observations: list[dict[str, Any]], client: Any,
    semaphore: asyncio.Semaphore, config: BenchmarkRuntimeConfig, counter: Any,
    output_root: Path, resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        trigger_tokens = _render_consolidation(
            job=job,
            observations=observations,
            max_observations=MAX_CONSOLIDATED_OBSERVATIONS,
            config=config,
            counter=counter,
        )["accounted_input_tokens"]
    except ValueError:
        trigger_tokens = config.call_budgets["task1_prediction"].max_input_tokens + 1
    if trigger_tokens <= CONSOLIDATION_TRIGGER_INPUT_TOKENS:
        return observations, {
            "applied": False,
            "policy": "hierarchical_qwen_state_salience_consolidation",
            "trigger_input_tokens": CONSOLIDATION_TRIGGER_INPUT_TOKENS,
            "observed_input_tokens": trigger_tokens,
            "raw_observation_count": len(observations),
            "final_observation_count": len(observations),
            "batch_count": 0,
            "call_count": 0,
            "semantic_call_count": 0,
        }
    batches = _consolidation_batches(
        job=job,
        observations=observations,
        config=config,
        counter=counter,
    )
    settled = await asyncio.gather(
        *[
            _consolidate_batch(
                job=job,
                observations=batch,
                max_observations=MAX_BATCH_CONSOLIDATED_OBSERVATIONS,
                level=1,
                batch_index=index,
                client=client,
                semaphore=semaphore,
                config=config,
                counter=counter,
                output_root=output_root,
                resume=resume,
            )
            for index, batch in enumerate(batches, start=1)
        ]
    )
    merged = _balanced_merge_observations(
        [record["observations"] for record in settled],
        limit=MAX_CONSOLIDATED_OBSERVATIONS,
    )
    calls = len(settled)
    semantic_calls = sum(
        int(record["generator_metadata"].get("semantic_attempt", 1))
        for record in settled
    )
    return merged, {
        "applied": True,
        "policy": "hierarchical_qwen_state_salience_consolidation",
        "trigger_input_tokens": CONSOLIDATION_TRIGGER_INPUT_TOKENS,
        "observed_input_tokens": trigger_tokens,
        "raw_observation_count": len(observations),
        "final_observation_count": len(merged),
        "batch_count": len(batches),
        "call_count": calls,
        "semantic_call_count": semantic_calls,
        "merge_policy": "round_robin_by_batch_preserving_qwen_priority",
    }


def _consolidation_batches(
    *, job: dict[str, Any], observations: list[dict[str, Any]],
    config: BenchmarkRuntimeConfig, counter: Any,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for observation in observations:
        candidate = [*current, observation]
        try:
            rendered = _render_consolidation(
                job=job,
                observations=candidate,
                max_observations=MAX_BATCH_CONSOLIDATED_OBSERVATIONS,
                config=config,
                counter=counter,
            )
        except ValueError:
            if not current:
                raise
            batches.append(current)
            current = [observation]
            continue
        if current and rendered["accounted_input_tokens"] > CONSOLIDATION_BATCH_INPUT_TARGET:
            batches.append(current)
            current = [observation]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


async def _consolidate_batch(
    *, job: dict[str, Any], observations: list[dict[str, Any]], max_observations: int,
    level: int, batch_index: int, client: Any, semaphore: asyncio.Semaphore,
    config: BenchmarkRuntimeConfig, counter: Any, output_root: Path, resume: bool,
) -> dict[str, Any]:
    rendered = _render_consolidation(
        job=job,
        observations=observations,
        max_observations=max_observations,
        config=config,
        counter=counter,
    )
    record_id = stable_id(
        "task1-memory-consolidation",
        job["trajectory_id"],
        job["memory_slot"],
        level,
        batch_index,
        rendered["prompt_sha256"],
    )
    checkpoint_path = (
        output_root / "memory_consolidation_checkpoints" / f"{record_id}.json"
    )
    if resume and checkpoint_path.is_file():
        record = _object(checkpoint_path)
        if record.get("prompt_sha256") != rendered["prompt_sha256"]:
            raise ValueError(f"Memory consolidation checkpoint drift: {checkpoint_path}")
        return record
    rejections = []
    semantic_attempts = max(1, int(config.prediction_llm.get("semantic_attempts", 3)))
    for semantic_attempt in range(1, semantic_attempts + 1):
        repair = ""
        if rejections:
            repair = (
                "\n\nThe previous response failed validation: "
                f"{rejections[-1]['error']}. Return a corrected JSON object with no more than "
                f"{max_observations} unique selected_observation_ids; rank and drop "
                "lower-priority IDs if needed."
            )
        user_prompt = rendered["user_prompt"] + repair
        try:
            async with semaphore:
                response = await client.generate_json(
                    system_prompt=rendered["system_prompt"],
                    user_prompt=user_prompt,
                    stage=f"task1_entity_memory_consolidation:{record_id}",
                )
        except ModelResponseParseError as exc:
            _assert_no_reasoning(exc.metadata)
            rejections.append(
                {
                    "semantic_attempt": semantic_attempt,
                    "error": str(exc),
                    "raw_response_chars": exc.metadata.get("raw_response_chars"),
                    "raw_response_sha256": exc.metadata.get("raw_response_sha256"),
                }
            )
            continue
        _assert_no_reasoning(response.metadata)
        try:
            selected_indexes = _validate_consolidation_selection(
                response.data,
                observation_count=len(observations),
                max_observations=max_observations,
            )
            consolidated = [observations[index] for index in selected_indexes]
        except ValueError as exc:
            rejections.append(
                {"semantic_attempt": semantic_attempt, "error": str(exc)}
            )
            continue
        break
    else:
        raise ValueError(
            f"Memory consolidation failed after {semantic_attempts} semantic attempts: "
            f"{rejections}"
        )
    record = {
        "level": level,
        "batch_index": batch_index,
        "input_observation_count": len(observations),
        "max_observations": max_observations,
        "observations": consolidated,
        "prompt_sha256": rendered["prompt_sha256"],
        "final_prompt_sha256": sha256_json(
            {"system": rendered["system_prompt"], "user": user_prompt}
        ),
        "accounted_input_tokens": rendered["accounted_input_tokens"],
        "generator_metadata": {
            **response.metadata,
            "semantic_attempt": semantic_attempt,
            "rejected_semantic_attempts": rejections,
        },
    }
    atomic_write_json(checkpoint_path, record)
    return record


def _validate_consolidation_selection(
    payload: dict[str, Any], *, observation_count: int, max_observations: int,
) -> list[int]:
    if set(payload) != {"selected_observation_ids"} or not isinstance(
        payload["selected_observation_ids"], list
    ):
        raise ValueError("Memory consolidation must return selected_observation_ids only")
    values = payload["selected_observation_ids"]
    if not values or len(values) > max_observations or len(values) != len(set(values)):
        raise ValueError(
            f"selection count must be unique and within 1..{max_observations}"
        )
    output = []
    for value in values:
        match = re.fullmatch(r"O([1-9][0-9]*)", str(value))
        if match is None or not 1 <= int(match.group(1)) <= observation_count:
            raise ValueError(f"unknown observation selection: {value!r}")
        output.append(int(match.group(1)) - 1)
    return output


def _render_consolidation(
    *, job: dict[str, Any], observations: list[dict[str, Any]], max_observations: int,
    config: BenchmarkRuntimeConfig, counter: Any,
) -> dict[str, Any]:
    allowed = sorted(
        {order for row in observations for order in row["evidence_scene_orders"]}
    )
    localized = [
        {"observation_id": f"O{index}", **row}
        for index, row in enumerate(observations, start=1)
    ]
    interval = job["interval"]
    system, user = PROMPTS.render(
        _prompt_key(job["language"], MEMORY_CONSOLIDATION_PROMPT_NAME),
        character=job["character"],
        aliases=job["aliases"],
        previous_scene_order=interval["start_scene_order_exclusive"],
        current_scene_order=interval["end_scene_order_inclusive"],
        observations=localized,
        allowed_observation_ids=[row["observation_id"] for row in localized],
        allowed_scene_orders=allowed,
        max_observations=max_observations,
    )
    return _budgeted(system, user, config=config, counter=counter)


def _balanced_merge_observations(
    batches: list[list[dict[str, Any]]], *, limit: int,
) -> list[dict[str, Any]]:
    output = []
    seen = set()
    position = 0
    while len(output) < limit:
        added = False
        for batch in batches:
            if position >= len(batch):
                continue
            observation = batch[position]
            key = " ".join(observation["claim"].casefold().split())
            if key not in seen:
                output.append(observation)
                seen.add(key)
                added = True
                if len(output) == limit:
                    break
        if not added and all(position >= len(batch) - 1 for batch in batches):
            break
        position += 1
    return output


async def _extract_chunk(
    *, job: dict[str, Any], chunk: dict[str, Any], client: Any,
    semaphore: asyncio.Semaphore, config: BenchmarkRuntimeConfig, counter: Any,
    output_root: Path, resume: bool,
) -> dict[str, Any]:
    rendered = _render_memory(job, chunk, config=config, counter=counter)
    record_id = stable_id(
        "task1-entity-memory", job["trajectory_id"], job["memory_slot"], chunk["chunk_index"]
    )
    checkpoint_path = output_root / "memory_checkpoints" / f"{record_id}.json"
    if resume and checkpoint_path.is_file():
        record = _object(checkpoint_path)
        if record.get("prompt_sha256") != rendered["prompt_sha256"]:
            raise ValueError(f"Memory checkpoint prompt drift: {checkpoint_path}")
        return record
    async with semaphore:
        response = await client.generate_json(
            system_prompt=rendered["system_prompt"],
            user_prompt=rendered["user_prompt"],
            stage=f"task1_entity_memory:{record_id}",
        )
    _assert_no_reasoning(response.metadata)
    observations = _validate_observations(
        response.data, allowed_scene_orders=set(chunk["scene_orders"])
    )
    record = {
        "chunk_index": chunk["chunk_index"],
        "scene_orders": chunk["scene_orders"],
        "content_tokens": chunk["content_tokens"],
        "observations": observations,
        "prompt_sha256": rendered["prompt_sha256"],
        "accounted_input_tokens": rendered["accounted_input_tokens"],
        "generator_metadata": response.metadata,
    }
    atomic_write_json(checkpoint_path, record)
    return record


def _render_memory(
    job: dict[str, Any], chunk: dict[str, Any], *, config: BenchmarkRuntimeConfig, counter: Any
) -> dict[str, Any]:
    system, user = PROMPTS.render(
        _prompt_key(job["language"], MEMORY_PROMPT_NAME),
        character=job["character"], aliases=job["aliases"],
        screenplay_chunk=chunk["text"], allowed_scene_orders=chunk["scene_orders"],
    )
    return _budgeted(system, user, config=config, counter=counter)


async def _run_setting(
    *, asset: dict[str, Any], memory_pack: dict[str, Any], client: Any,
    semaphore: asyncio.Semaphore, config: BenchmarkRuntimeConfig, counter: Any,
    output_root: Path, resume: bool,
    seed_predictions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    memory_by_checkpoint = {row["checkpoint_id"]: row for row in memory_pack["slots"]}

    async def run_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
        previous_prediction: list[dict[str, Any]] = []
        previous_prediction_provenance: list[int] = []
        rows = []
        for checkpoint in trajectory["checkpoints"]:
            source = checkpoint["model_input"]["previous_state_source"]
            previous_state_rows = (
                checkpoint["model_input"].get("previous_state", [])
                if source == "reference_state"
                else [] if source == "empty" else previous_prediction
            )
            previous_state = _previous_state_content(previous_state_rows)
            previous_state_evidence = (
                previous_prediction_provenance
                if source == "autoregressive_prediction"
                else _previous_state_evidence(previous_state_rows)
            )
            _assert_previous_state_safe(previous_state)
            memory = memory_by_checkpoint[checkpoint["checkpoint_id"]]
            rendered = _render_prediction(
                trajectory=trajectory, checkpoint=checkpoint,
                previous_state=previous_state, memory=memory["observations"],
                language=asset["language"],
                config=config, counter=counter,
            )
            record_id = stable_id("task1-state_update-prediction", asset["setting"], checkpoint["checkpoint_id"])
            checkpoint_path = output_root / "checkpoints" / f"{record_id}.json"
            seed = (seed_predictions or {}).get(checkpoint["checkpoint_id"])
            if source == "empty" and seed is not None:
                record = {
                    "instance_id": checkpoint["instance_id"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "previous_state_source": source,
                    "previous_state": [],
                    "previous_state_evidence_scene_orders": [],
                    "memory_slot": checkpoint["entity_memory"]["memory_slot"],
                    "shared_entity_memory_sha256": sha256_json(memory["observations"]),
                    "prediction": seed["prediction"],
                    "state_provenance_scene_orders": seed[
                        "state_provenance_scene_orders"
                    ],
                    "prompt_sha256": rendered["prompt_sha256"],
                    "accounted_input_tokens": rendered["accounted_input_tokens"],
                    "generator_metadata": {
                        **seed["generator_metadata"],
                        "shared_initial_checkpoint_prediction": True,
                        "source_setting": "reference_state_update",
                    },
                }
                atomic_write_json(checkpoint_path, record)
            elif resume and checkpoint_path.is_file():
                record = _object(checkpoint_path)
                if record.get("prompt_sha256") != rendered["prompt_sha256"]:
                    superseded = output_root / "superseded_checkpoints"
                    superseded.mkdir(parents=True, exist_ok=True)
                    old_hash = str(record.get("prompt_sha256") or "unknown")[:12]
                    checkpoint_path.replace(
                        superseded / f"{checkpoint_path.stem}-{old_hash}.json"
                    )
                    record = None
            else:
                record = None
            if record is None:
                async with semaphore:
                    response = await client.generate_json(
                        system_prompt=rendered["system_prompt"],
                        user_prompt=rendered["user_prompt"],
                        stage=f"task1_state_update_prediction:{asset['setting']}:{record_id}",
                    )
                _assert_no_reasoning(response.metadata)
                prediction = validate_task1_prediction(response.data)
                _validate_prediction_boundary(
                    prediction,
                    previous=int(checkpoint["screenplay_interval"]["start_scene_order_exclusive"]),
                    current=int(checkpoint["screenplay_interval"]["end_scene_order_inclusive"]),
                )
                record = {
                    "instance_id": checkpoint["instance_id"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "previous_state_source": source,
                    "previous_state": previous_state,
                    "previous_state_evidence_scene_orders": previous_state_evidence,
                    "memory_slot": checkpoint["entity_memory"]["memory_slot"],
                    "shared_entity_memory_sha256": sha256_json(memory["observations"]),
                    "prediction": prediction,
                    "state_provenance_scene_orders": _state_provenance(
                        previous_state_evidence=previous_state_evidence,
                        prediction=prediction,
                    ),
                    "prompt_sha256": rendered["prompt_sha256"],
                    "accounted_input_tokens": rendered["accounted_input_tokens"],
                    "generator_metadata": response.metadata,
                }
                atomic_write_json(checkpoint_path, record)
            previous_prediction = record["prediction"]["current_state"]
            previous_prediction_provenance = record["state_provenance_scene_orders"]
            rows.append(record)
        return {
            "trajectory_id": trajectory["trajectory_id"],
            "character_id": trajectory["character_id"],
            "character": trajectory["character"],
            "checkpoints": rows,
        }
    trajectories = await asyncio.gather(
        *(run_trajectory(trajectory) for trajectory in asset["trajectories"])
    )
    return {
        "schema": "stage_task1_state_update_predictions",
        "setting": asset["setting"],
        "movie_id": asset["movie_id"],
        "prediction_model": config.prediction_llm["model"],
        "trajectory_count": len(trajectories),
        "checkpoint_count": sum(len(row["checkpoints"]) for row in trajectories),
        "trajectories": trajectories,
    }


def _render_prediction(
    *, trajectory: dict[str, Any], checkpoint: dict[str, Any],
    previous_state: list[dict[str, Any]], memory: list[dict[str, Any]],
    language: str, config: BenchmarkRuntimeConfig, counter: Any,
) -> dict[str, Any]:
    interval = checkpoint["screenplay_interval"]
    system, user = PROMPTS.render(
        _prompt_key(language, PREDICTION_PROMPT_NAME),
        character=trajectory["character"], aliases=trajectory["aliases"],
        previous_scene_order=interval["start_scene_order_exclusive"],
        current_scene_order=interval["end_scene_order_inclusive"],
        previous_state=previous_state, entity_memory=memory,
    )
    if INTERNAL_ID.search(system) or INTERNAL_ID.search(user):
        raise ValueError("Internal ID leaked into Task 1 prediction prompt")
    return _budgeted(system, user, config=config, counter=counter)


def _budgeted(system: str, user: str, *, config: BenchmarkRuntimeConfig, counter: Any) -> dict[str, Any]:
    raw = counter.count(system) + counter.count(user)
    accounted = raw + config.reserved_chat_template_tokens
    maximum = config.call_budgets["task1_prediction"].max_input_tokens
    if accounted > maximum:
        raise ValueError(f"Task 1 state_update prompt exceeds budget: {accounted}>{maximum}")
    return {
        "system_prompt": system, "user_prompt": user,
        "prompt_sha256": sha256_json({"system": system, "user": user}),
        "accounted_input_tokens": accounted, "max_input_tokens": maximum,
    }


def _validate_observations(payload: dict[str, Any], *, allowed_scene_orders: set[int]) -> list[dict[str, Any]]:
    if set(payload) != {"observations"} or not isinstance(payload["observations"], list):
        raise ValueError("Entity memory response must contain observations only")
    output = []
    for row in payload["observations"]:
        if not isinstance(row, dict) or set(row) != {"claim", "evidence_scene_orders"}:
            raise ValueError("Entity memory observation schema mismatch")
        claim = str(row["claim"] or "").strip()
        scenes = sorted(set(row["evidence_scene_orders"]))
        if not claim or INTERNAL_ID.search(claim):
            raise ValueError("Entity memory contains empty text or an internal ID")
        if not scenes or any(not isinstance(value, int) or value not in allowed_scene_orders for value in scenes):
            raise ValueError("Entity memory cites a scene outside its chunk")
        output.append({"claim": claim, "evidence_scene_orders": scenes})
    return output


def _previous_state_content(rows: list[dict[str, Any]]) -> list[str]:
    if not isinstance(rows, list):
        raise ValueError("Previous state source must be an array")
    output = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("claim"), str):
            raise ValueError("Previous state source rows must contain claim text")
        claim = row["claim"].strip()
        if not claim:
            raise ValueError("Previous state contains an empty claim")
        output.append(claim)
    return output


def _previous_state_evidence(rows: list[dict[str, Any]]) -> list[int]:
    return sorted(
        {
            value
            for row in rows
            for value in row.get("evidence_scene_orders", [])
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
    )


def _state_provenance(
    *, previous_state_evidence: list[int], prediction: dict[str, Any]
) -> list[int]:
    values = set(previous_state_evidence)
    for row in prediction["current_state"]:
        values.update(row.get("evidence_scene_orders", []))
    return sorted(
        value
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    )


def _assert_previous_state_safe(rows: list[str]) -> None:
    if not isinstance(rows, list):
        raise ValueError("Previous state must be an array")
    for claim in rows:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Previous state must contain claim text only")
        if INTERNAL_ID.search(claim):
            raise ValueError("Previous state contains an internal ID")


def _chunk_token_statistics(memory_pack: dict[str, Any]) -> dict[str, Any]:
    values = [
        int(chunk["content_tokens"])
        for slot in memory_pack["slots"]
        for chunk in slot["chunks"]
    ]
    return {
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def _validate_prediction_boundary(payload: dict[str, Any], *, previous: int, current: int) -> None:
    for field in ("current_state", "developments_since_previous_checkpoint"):
        for row in payload[field]:
            if any(value > current for value in row["evidence_scene_orders"]):
                raise ValueError("Prediction cites a scene after the current checkpoint")
    for row in payload["developments_since_previous_checkpoint"]:
        if not any(previous < value <= current for value in row["evidence_scene_orders"]):
            raise ValueError("Development lacks evidence in the current checkpoint interval")


def _assert_no_reasoning(metadata: dict[str, Any]) -> None:
    for key in ("reasoning_tokens", "provider_thought_tokens", "thinking_chars"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and value > 0:
            raise ValueError(f"Reasoning was not disabled: {key}={value}")


def _prompt_key(language: str, name: str) -> str:
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported prompt language: {language}")
    return f"{language}/evaluation/{name}"


def _prompt_artifacts(language: str) -> list[dict[str, Any]]:
    paths = set(
        PROMPTS.get(_prompt_key(language, MEMORY_PROMPT_NAME)).source_paths
    ) | set(
        PROMPTS.get(
            _prompt_key(language, MEMORY_CONSOLIDATION_PROMPT_NAME)
        ).source_paths
    ) | set(
        PROMPTS.get(_prompt_key(language, PREDICTION_PROMPT_NAME)).source_paths
    )
    return [_artifact(path) for path in sorted(paths)]


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path.resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
