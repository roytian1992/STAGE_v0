#!/usr/bin/env python3
"""Run Task 3 single-turn predictions across a complete STAGE condition."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.clients import (  # noqa: E402
    ModelConfig,
    build_endpoint_pool_runtime,
    build_text_client,
)
from stage_narrative.io import (  # noqa: E402
    atomic_write_json,
    load_json,
    sha256_file,
    sha256_json,
)
from stage_narrative.models import stable_id  # noqa: E402
from stage_narrative.prompt_loader import PROMPTS  # noqa: E402
from stage_narrative.task3.prediction import (  # noqa: E402
    materialize_single_actor_input as materialize_legacy_actor_input,
    role_index as legacy_role_index,
)
from stage_narrative.task3.release_validation import (  # noqa: E402
    validate_task3_release as validate_legacy_release,
)
from stage_narrative.task3.validation import (  # noqa: E402
    validate_role_assets as validate_modern_roles,
    validate_task3_release as validate_modern_release,
)
from stage_narrative.task3.visibility import (  # noqa: E402
    materialize_role_at_boundary as materialize_modern_role,
)
from stage_narrative.task2_hybrid import keyword_tokens  # noqa: E402
from stage_narrative.temporal.benchmark_protocol import (  # noqa: E402
    BenchmarkRuntimeConfig,
)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cohort", choices=("all", "core10", "expansion50", "completion91"), default="all"
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    release_root = args.release_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite prediction run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    collection_path = release_root / "manifest.json"
    collection = load_json(collection_path)
    if collection.get("schema_version") == "stage_complete_condition_release_v1":
        collection_entries = collection.get("entries", [])
    elif collection.get("schema") == "stage_manifest50_release":
        collection_entries = [
            {**item, "cohort": "expansion50"}
            for item in collection.get("entries", [])
        ]
    else:
        raise ValueError("Unsupported STAGE condition release manifest")
    selected = [
        item
        for item in collection_entries
        if args.cohort == "all" or item.get("cohort") == args.cohort
    ]
    if not selected:
        raise ValueError("The selected Task 3 cohort is empty")
    movies = [_load_movie(release_root / item["path"], item) for item in selected]
    jobs = [
        (movie, instance)
        for movie in movies
        for instance in movie["single_turn"]["instances"]
    ]
    config = BenchmarkRuntimeConfig.load(args.config.resolve())
    counter = config.build_token_counter()
    materialized = []
    for movie, instance in jobs:
        _, rendered, retrieval = _materialize_prompt(
            movie, instance, config=config, counter=counter
        )
        materialized.append((rendered, retrieval))
    preflight = {
        "schema_version": "stage_complete_task3_single_prediction_preflight_v1",
        "status": "passed_zero_call",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "condition": collection["condition"],
        "cohort": args.cohort,
        "movies": len(movies),
        "instances": len(jobs),
        "maximum_accounted_input_tokens": max(
            item[0]["accounted_input_tokens"] for item in materialized
        ),
        "maximum_input_tokens": config.call_budgets["task3_actor"].max_input_tokens,
        "retrieval_fallback_instances": sum(
            item[1]["mode"] == "query_only_visible_memory_retrieval"
            for item in materialized
        ),
        "minimum_selected_memories": min(
            item[1]["selected_memories"] for item in materialized
        ),
        "semantic_calls": 0,
    }
    atomic_write_json(output_root / "preflight.json", preflight)
    if args.dry_run:
        print(output_root / "preflight.json")
        print(preflight)
        return

    model_config = {
        **config.prediction_llm,
        "json_response_format": False,
        "max_tokens": config.call_budgets["task3_actor"].max_output_tokens,
    }
    resolved = ModelConfig.from_dict(model_config)
    endpoint_runtime = build_endpoint_pool_runtime(model_config)
    client = build_text_client(model_config, endpoint_runtime=endpoint_runtime)
    semaphore = asyncio.Semaphore(args.workers)
    settled = await asyncio.gather(
        *[
            _run_one(
                movie,
                instance,
                config=config,
                counter=counter,
                client=client,
                semaphore=semaphore,
                output_root=output_root,
                prediction_model=resolved.model,
            )
            for movie, instance in jobs
        ],
        return_exceptions=True,
    )
    predictions, failures = [], []
    for (_, instance), result in zip(jobs, settled, strict=True):
        if isinstance(result, BaseException):
            failures.append(
                {
                    "instance_id": instance["instance_id"],
                    "error_type": type(result).__name__,
                    "error": str(result),
                }
            )
        else:
            predictions.append(result)
    payload = {
        "schema_version": "stage_complete_task3_single_predictions_v1",
        "condition": collection["condition"],
        "cohort": args.cohort,
        "prediction_model": resolved.model,
        "prediction_count": len(predictions),
        "predictions": predictions,
    }
    prediction_path = output_root / "predictions.json"
    atomic_write_json(prediction_path, payload)
    manifest = {
        "schema_version": "stage_complete_task3_single_prediction_run_v1",
        "status": "completed" if not failures else "completed_with_failures",
        "condition": collection["condition"],
        "cohort": args.cohort,
        "prediction_model": resolved.model,
        "counts": {
            "movies": len(movies),
            "expected_instances": len(jobs),
            "completed_instances": len(predictions),
            "failures": len(failures),
        },
        "inputs": {
            "collection_manifest": _artifact(collection_path),
            "config": _artifact(args.config.resolve()),
        },
        "output": _artifact(prediction_path),
        "failures": failures,
        "endpoint_pool": await endpoint_runtime.snapshot() if endpoint_runtime else None,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(output_root / "manifest.json")
    print(manifest["counts"])
    if failures:
        raise RuntimeError(f"Task 3 prediction run has {len(failures)} failures")


def _load_movie(movie_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    roles = load_json(movie_dir / "task_3_role_assets.json")
    single = load_json(movie_dir / "task_3_single_turn.json")
    first_role = (roles.get("roles") or [{}])[0]
    modern = "canonical_name" in first_role
    if modern:
        validate_modern_roles(roles)
        empty_multi = {
            "schema_version": "stage_task3_multi_turn",
            "movie_id": roles["movie_id"],
            "episode_count": 0,
            "turn_count": 0,
            "episodes": [],
        }
        validate_modern_release(
            role_assets=roles, single_turn=single, multi_turn=empty_multi
        )
        role_by_id = {item["role_id"]: item for item in roles["roles"]}
    else:
        multi_path = movie_dir / "task_3_multi_turn.json"
        if not multi_path.is_file():
            raise ValueError(f"Legacy Task 3 role schema lacks multi-turn validation asset: {movie_dir}")
        validate_legacy_release(
            role_assets=roles,
            single_turn=single,
            multi_turn=load_json(multi_path),
        )
        role_by_id = legacy_role_index(roles)
    return {
        "movie_id": roles["movie_id"],
        "language": roles["language"],
        "cohort": entry["cohort"],
        "modern": modern,
        "roles": role_by_id,
        "single_turn": single,
    }


def _actor_input(movie: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    if not movie["modern"]:
        return materialize_legacy_actor_input(instance, movie["roles"])
    role = movie["roles"].get(instance["role_ref"])
    if role is None:
        raise ValueError(f"Unknown modern Task 3 role: {instance['role_ref']}")
    visible = materialize_modern_role(role, instance["checkpoint"])
    identity = visible.get("identity") or {}
    return {
        "character": identity.get("name") or visible["canonical_name"],
        "role_context": visible,
        "interaction_context": {
            "setting": instance["interaction_context"],
            "dialogue_history": instance.get("dialogue_history", []),
        },
        "current_user_turn": instance["current_user_turn"],
    }


async def _run_one(
    movie: dict[str, Any],
    instance: dict[str, Any],
    *,
    config: BenchmarkRuntimeConfig,
    counter: Any,
    client: Any,
    semaphore: asyncio.Semaphore,
    output_root: Path,
    prediction_model: str,
) -> dict[str, Any]:
    actor_input, rendered, retrieval = _materialize_prompt(
        movie, instance, config=config, counter=counter
    )
    checkpoint_id = stable_id(
        "stage-task3-single-prediction", movie["movie_id"], instance["instance_id"]
    )
    checkpoint_path = output_root / "checkpoints" / f"{checkpoint_id}.json"
    if checkpoint_path.is_file():
        checkpoint = load_json(checkpoint_path)
        if (
            checkpoint.get("instance_id") != instance["instance_id"]
            or checkpoint.get("prompt_sha256") != rendered["prompt_sha256"]
            or checkpoint.get("prediction_model") != prediction_model
        ):
            raise ValueError(f"Task 3 prediction checkpoint drift: {checkpoint_path}")
        return {**checkpoint, "checkpoint_reused": True}
    async with semaphore:
        response = await client.generate_text(
            system_prompt=rendered["system_prompt"],
            user_prompt=rendered["user_prompt"],
            stage=f"stage_task3_single:{instance['instance_id']}",
        )
    if response.metadata.get("finish_reason") != "stop":
        raise ValueError(f"Task 3 actor did not finish normally: {instance['instance_id']}")
    record = {
        "movie_id": movie["movie_id"],
        "instance_id": instance["instance_id"],
        "response": response.text,
        "prediction_model": prediction_model,
        "prompt_sha256": rendered["prompt_sha256"],
        "actor_input_sha256": sha256_json(actor_input),
        "accounted_input_tokens": rendered["accounted_input_tokens"],
        "memory_context": retrieval,
        "checkpoint_reused": False,
    }
    atomic_write_json(checkpoint_path, record)
    return record


def _render_prompt(
    actor_input: dict[str, Any],
    *,
    language: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> dict[str, Any]:
    if "evaluator_reference" in repr(actor_input):
        raise ValueError("Evaluator reference leaked into Task 3 actor input")
    prompt_path = config.prompt_path("task3_actor", language)
    system, user = PROMPTS.render(prompt_path, **actor_input)
    raw_tokens = counter.count(system) + counter.count(user)
    accounted = raw_tokens + config.reserved_chat_template_tokens
    maximum = config.call_budgets["task3_actor"].max_input_tokens
    if accounted > maximum:
        raise ValueError(f"Task 3 actor prompt exceeds budget: {accounted}>{maximum}")
    return {
        "system_prompt": system,
        "user_prompt": user,
        "accounted_input_tokens": accounted,
        "prompt_sha256": sha256_json(
            {"system_prompt": system, "user_prompt": user}
        ),
    }


def _materialize_prompt(
    movie: dict[str, Any],
    instance: dict[str, Any],
    *,
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    actor_input = _actor_input(movie, instance)
    memories = actor_input.get("role_context", {}).get("memories", [])
    if not isinstance(memories, list):
        raise ValueError(f"Task 3 actor memories are not an array: {instance['instance_id']}")
    try:
        rendered = _render_prompt(
            actor_input,
            language=movie["language"],
            config=config,
            counter=counter,
        )
        return actor_input, rendered, {
            "mode": "full_visible_memory",
            "visible_memories": len(memories),
            "selected_memories": len(memories),
            "query_fields": [],
        }
    except ValueError as exc:
        if "exceeds budget" not in str(exc):
            raise
    ranked = _rank_memories(
        memories,
        query=(
            f"{actor_input.get('current_user_turn', '')}\n"
            f"{actor_input.get('interaction_context', {}).get('setting', '')}"
        ),
        language=movie["language"],
    )
    low, high = 0, len(ranked)
    selected_actor = None
    selected_rendered = None
    while low <= high:
        middle = (low + high) // 2
        candidate = deepcopy(actor_input)
        candidate["role_context"]["memories"] = ranked[:middle]
        try:
            candidate_rendered = _render_prompt(
                candidate,
                language=movie["language"],
                config=config,
                counter=counter,
            )
        except ValueError as exc:
            if "exceeds budget" not in str(exc):
                raise
            high = middle - 1
        else:
            selected_actor = candidate
            selected_rendered = candidate_rendered
            low = middle + 1
    if selected_actor is None or selected_rendered is None:
        raise ValueError(
            f"Task 3 prompt cannot fit even without memories: {instance['instance_id']}"
        )
    return selected_actor, selected_rendered, {
        "mode": "query_only_visible_memory_retrieval",
        "visible_memories": len(memories),
        "selected_memories": len(selected_actor["role_context"]["memories"]),
        "query_fields": ["current_user_turn", "interaction_context.setting"],
        "ranking": "lexical_overlap_then_importance_then_recency",
    }


def _rank_memories(
    memories: list[dict[str, Any]], *, query: str, language: str
) -> list[dict[str, Any]]:
    language_code = "zh" if language in {"zh", "Chinese"} else "en"
    query_tokens = set(keyword_tokens(query, language_code))
    ranked = []
    for index, memory in enumerate(memories):
        text = " ".join(
            [
                str(memory.get("memory_text") or ""),
                " ".join(str(item) for item in memory.get("tags", [])),
            ]
        )
        overlap = len(query_tokens & set(keyword_tokens(text, language_code)))
        importance = 1 if memory.get("importance") in {"core", "high", 3} else 0
        ranked.append((overlap, importance, index, memory))
    return [
        item[3]
        for item in sorted(
            ranked,
            key=lambda item: (-item[0], -item[1], -item[2]),
        )
    ]


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
