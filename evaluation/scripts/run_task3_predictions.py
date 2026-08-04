#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.clients import (  # noqa: E402
    ModelConfig,
    ModelResponseParseError,
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
    materialize_multi_actor_input,
    materialize_single_actor_input,
    role_index,
)
from stage_narrative.task3.release_validation import (  # noqa: E402
    validate_multi_turn_release,
    validate_task3_release,
)
from stage_narrative.temporal.benchmark_protocol import (  # noqa: E402
    BenchmarkRuntimeConfig,
)


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Task 3 single- and multi-turn predictions from a formal task release."
        )
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", choices=("ordinary", "anonymous"), required=True)
    parser.add_argument("--mode", choices=("single", "multi", "both"), default="both")
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
    config = BenchmarkRuntimeConfig.load(args.config.resolve())
    counter = config.build_token_counter()
    release_manifest_path = release_root / "manifest.json"
    release_manifest = load_json(release_manifest_path)
    if release_manifest.get("schema") == "stage_manifest10_task_release":
        release_entries = release_manifest["entries"]
    elif (
        release_manifest.get("schema_version")
        == "stage_complete_condition_release_v1"
    ):
        release_entries = release_manifest.get("entries", [])
        if args.mode != "multi":
            release_entries = [
                item for item in release_entries if item.get("cohort") == "core10"
            ]
    else:
        raise ValueError("Unsupported formal Task 3 release manifest")
    if release_manifest.get("condition") != args.condition:
        raise ValueError(
            f"Release condition is {release_manifest.get('condition')!r}, "
            f"not {args.condition!r}"
        )
    release_dirs = [release_root / row["path"] for row in release_entries]
    expected_movies = 151 if args.mode == "multi" and len(release_entries) > 10 else 10
    if len(release_dirs) != expected_movies:
        raise ValueError(
            f"Expected {expected_movies} Task 3 release directories, "
            f"found {len(release_dirs)}"
        )

    movies = [
        _load_movie(path, release_manifest_path, mode=args.mode)
        for path in release_dirs
    ]
    expected_single = sum(item["single_turn"]["instance_count"] for item in movies)
    expected_episodes = sum(item["multi_turn"]["episode_count"] for item in movies)
    expected_turns = sum(item["multi_turn"]["turn_count"] for item in movies)

    if args.dry_run:
        audit = _preflight(
            movies=movies,
            config=config,
            counter=counter,
            mode=args.mode,
        )
        payload = {
            "schema_version": "stage_task3_role_memory_prediction_preflight",
            "status": "passed",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "condition": args.condition,
            "mode": args.mode,
            "config": config.public_snapshot(),
            "release_root": str(release_root),
            "counts": {
                "movies": len(movies),
                "expected_single_turn": expected_single,
                "expected_multi_turn_episodes": expected_episodes,
                "expected_multi_turn_turns": expected_turns,
                **audit["counts"],
            },
            "maximum_prompt": audit["maximum_prompt"],
            "prompt_artifacts": _prompt_artifacts(movies, config),
            "evaluator_fields_in_actor_input": 0,
            "semantic_calls": 0,
        }
        atomic_write_json(output_root / "preflight.json", payload)
        print(output_root / "preflight.json")
        print(payload["counts"])
        return

    model_config = {
        **config.prediction_llm,
        "json_response_format": False,
        "max_tokens": config.call_budgets["task3_actor"].max_output_tokens,
    }
    resolved_model = ModelConfig.from_dict(model_config)
    endpoint_runtime = build_endpoint_pool_runtime(model_config)
    client = build_text_client(model_config, endpoint_runtime=endpoint_runtime)
    semaphore = asyncio.Semaphore(args.workers)

    single_results: list[dict[str, Any]] = []
    multi_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if args.mode in {"single", "both"}:
        jobs = [
            (movie, instance)
            for movie in movies
            for instance in movie["single_turn"]["instances"]
        ]
        settled = await asyncio.gather(
            *(
                _run_single(
                    movie,
                    instance,
                    config=config,
                    counter=counter,
                    client=client,
                    semaphore=semaphore,
                    output_root=output_root,
                    prediction_model=resolved_model.model,
                )
                for movie, instance in jobs
            ),
            return_exceptions=True,
        )
        single_results, single_failures = _split(
            settled, [item[1]["instance_id"] for item in jobs]
        )
        failures.extend({"mode": "single", **item} for item in single_failures)

    if args.mode in {"multi", "both"}:
        jobs = [
            (movie, episode)
            for movie in movies
            for episode in movie["multi_turn"]["episodes"]
        ]
        settled = await asyncio.gather(
            *(
                _run_multi_episode(
                    movie,
                    episode,
                    config=config,
                    counter=counter,
                    client=client,
                    semaphore=semaphore,
                    output_root=output_root,
                    prediction_model=resolved_model.model,
                )
                for movie, episode in jobs
            ),
            return_exceptions=True,
        )
        multi_results, multi_failures = _split(
            settled, [item[1]["episode_id"] for item in jobs]
        )
        failures.extend({"mode": "multi", **item} for item in multi_failures)

    single_payload = {
        "schema_version": "stage_task3_role_memory_single_predictions",
        "condition": args.condition,
        "prediction_model": resolved_model.model,
        "prediction_count": len(single_results),
        "predictions": single_results,
    }
    multi_payload = {
        "schema_version": "stage_task3_role_memory_multi_predictions",
        "condition": args.condition,
        "prediction_model": resolved_model.model,
        "episode_count": len(multi_results),
        "turn_count": sum(len(item["turns"]) for item in multi_results),
        "episodes": multi_results,
    }
    atomic_write_json(output_root / "single_turn_predictions.json", single_payload)
    atomic_write_json(output_root / "multi_turn_predictions.json", multi_payload)
    endpoint_snapshot = (
        await endpoint_runtime.snapshot() if endpoint_runtime is not None else None
    )
    manifest = {
        "schema_version": "stage_task3_role_memory_prediction_run",
        "status": "completed" if not failures else "completed_with_failures",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "condition": args.condition,
        "mode": args.mode,
        "prediction_model": resolved_model.model,
        "evaluation_model_configured_but_unused": config.evaluation_llm.get("model"),
        "workers": args.workers,
        "config": config.public_snapshot(),
        "release_root": str(release_root),
        "release_manifests": [
            {"path": str(item["release_manifest"]), "sha256": sha256_file(item["release_manifest"])}
            for item in movies
        ],
        "prompt_artifacts": _prompt_artifacts(movies, config),
        "counts": {
            "expected_single_turn": expected_single if args.mode in {"single", "both"} else 0,
            "completed_single_turn": len(single_results),
            "expected_multi_turn_episodes": expected_episodes if args.mode in {"multi", "both"} else 0,
            "completed_multi_turn_episodes": len(multi_results),
            "expected_multi_turn_turns": expected_turns if args.mode in {"multi", "both"} else 0,
            "completed_multi_turn_turns": multi_payload["turn_count"],
            "failure_count": len(failures),
        },
        "failures": failures,
        "endpoint_pool": endpoint_snapshot,
        "checkpoint_policy": {
            "resume_requires_exact_prompt_hash": True,
            "semantic_fallback": "none",
            "content_filter_fallback": "none",
        },
        "outputs": {
            "single_turn_predictions": _artifact(output_root / "single_turn_predictions.json"),
            "multi_turn_predictions": _artifact(output_root / "multi_turn_predictions.json"),
        },
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(output_root / "manifest.json")
    print(manifest["counts"])
    if failures:
        raise RuntimeError(f"Prediction run has {len(failures)} incomplete units")


def _load_movie(
    release_dir: Path, release_manifest: Path, *, mode: str
) -> dict[str, Any]:
    role_assets = load_json(release_dir / "task_3_role_assets.json")
    single_turn = load_json(release_dir / "task_3_single_turn.json")
    multi_turn = load_json(release_dir / "task_3_multi_turn.json")
    if mode == "multi":
        validate_multi_turn_release(
            role_assets=role_assets,
            multi_turn=multi_turn,
        )
    else:
        validate_task3_release(
            role_assets=role_assets,
            single_turn=single_turn,
            multi_turn=multi_turn,
        )
    return {
        "movie_id": role_assets["movie_id"],
        "language": role_assets["language"],
        "roles": role_index(role_assets),
        "single_turn": single_turn,
        "multi_turn": multi_turn,
        "release_manifest": release_manifest,
    }


def _preflight(
    *,
    movies: list[dict[str, Any]],
    config: BenchmarkRuntimeConfig,
    counter: Any,
    mode: str,
) -> dict[str, Any]:
    counts = {"materialized_single_turn": 0, "materialized_multi_turn_turns": 0}
    maximum = {"accounted_input_tokens": -1}
    if mode in {"single", "both"}:
        for movie in movies:
            for instance in movie["single_turn"]["instances"]:
                actor_input = materialize_single_actor_input(instance, movie["roles"])
                record = _render_prompt(
                    actor_input, language=movie["language"], config=config, counter=counter
                )
                maximum = _larger(maximum, record, movie["movie_id"], instance["instance_id"])
                counts["materialized_single_turn"] += 1
    if mode in {"multi", "both"}:
        for movie in movies:
            for episode in movie["multi_turn"]["episodes"]:
                placeholders = {
                    index: f"<model response from turn {index}>"
                    for index in range(1, len(episode["turns"]) + 1)
                }
                for turn in episode["turns"]:
                    actor_input = materialize_multi_actor_input(
                        episode, turn, movie["roles"], placeholders
                    )
                    record = _render_prompt(
                        actor_input,
                        language=movie["language"],
                        config=config,
                        counter=counter,
                    )
                    identifier = f"{episode['episode_id']}:{turn['turn_index']}"
                    maximum = _larger(maximum, record, movie["movie_id"], identifier)
                    counts["materialized_multi_turn_turns"] += 1
    return {"counts": counts, "maximum_prompt": maximum}


async def _run_single(
    movie: dict[str, Any],
    instance: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    actor_input = materialize_single_actor_input(instance, movie["roles"])
    record = await _run_call(
        actor_input=actor_input,
        language=movie["language"],
        movie_id=movie["movie_id"],
        identifier=instance["instance_id"],
        stage=f"task3_role_memory_single:{instance['instance_id']}",
        checkpoint_kind="single",
        **kwargs,
    )
    return {
        "movie_id": movie["movie_id"],
        "instance_id": instance["instance_id"],
        "character_id": instance["character_id"],
        "checkpoint_boundary": instance["checkpoint_boundary"],
        **record,
    }


async def _run_multi_episode(
    movie: dict[str, Any],
    episode: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    prior_responses: dict[int, str] = {}
    turns = []
    for turn in episode["turns"]:
        turn_index = int(turn["turn_index"])
        actor_input = materialize_multi_actor_input(
            episode, turn, movie["roles"], prior_responses
        )
        identifier = f"{episode['episode_id']}:{turn_index}"
        record = await _run_call(
            actor_input=actor_input,
            language=movie["language"],
            movie_id=movie["movie_id"],
            identifier=identifier,
            stage=f"task3_role_memory_multi:{identifier}",
            checkpoint_kind="multi",
            **kwargs,
        )
        prior_responses[turn_index] = record["response"]
        turns.append(
            {
                "turn_index": turn_index,
                "question_id": turn["question_id"],
                **record,
            }
        )
    return {
        "movie_id": movie["movie_id"],
        "episode_id": episode["episode_id"],
        "character_id": episode["character_id"],
        "turns": turns,
    }


async def _run_call(
    *,
    actor_input: dict[str, Any],
    language: str,
    movie_id: str,
    identifier: str,
    stage: str,
    checkpoint_kind: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
    client: Any,
    semaphore: asyncio.Semaphore,
    output_root: Path,
    prediction_model: str,
) -> dict[str, Any]:
    rendered = _render_prompt(
        actor_input, language=language, config=config, counter=counter
    )
    checkpoint_id = stable_id(
        "task3-role-memory-prediction", movie_id, checkpoint_kind, identifier
    )
    checkpoint_path = output_root / "checkpoints" / checkpoint_kind / f"{checkpoint_id}.json"
    if checkpoint_path.is_file():
        checkpoint = load_json(checkpoint_path)
        if (
            checkpoint.get("identifier") != identifier
            or checkpoint.get("prompt_sha256") != rendered["prompt_sha256"]
            or checkpoint.get("prediction_model") != prediction_model
        ):
            raise ValueError(f"Prediction checkpoint drift: {checkpoint_path}")
        return {**checkpoint, "checkpoint_reused": True}

    async with semaphore:
        response = await client.generate_text(
            system_prompt=rendered["system_prompt"],
            user_prompt=rendered["user_prompt"],
            stage=stage,
        )
    if response.metadata.get("finish_reason") != "stop":
        raise ModelResponseParseError(
            f"{stage} did not finish normally",
            raw_text=response.text,
            metadata=response.metadata,
        )
    record = {
        "identifier": identifier,
        "response": response.text,
        "prediction_model": prediction_model,
        "prompt_path": rendered["prompt_path"],
        "prompt_sha256": rendered["prompt_sha256"],
        "actor_input_sha256": sha256_json(actor_input),
        "raw_prompt_tokens": rendered["raw_prompt_tokens"],
        "accounted_input_tokens": rendered["accounted_input_tokens"],
        "max_input_tokens": rendered["max_input_tokens"],
        "generator_metadata": response.metadata,
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
        "prompt_path": prompt_path,
        "system_prompt": system,
        "user_prompt": user,
        "raw_prompt_tokens": raw_tokens,
        "accounted_input_tokens": accounted,
        "max_input_tokens": maximum,
        "prompt_sha256": sha256_json(
            {"system_prompt": system, "user_prompt": user}
        ),
    }


def _larger(
    current: dict[str, Any],
    rendered: dict[str, Any],
    movie_id: str,
    identifier: str,
) -> dict[str, Any]:
    if rendered["accounted_input_tokens"] <= int(current["accounted_input_tokens"]):
        return current
    return {
        "accounted_input_tokens": rendered["accounted_input_tokens"],
        "raw_prompt_tokens": rendered["raw_prompt_tokens"],
        "max_input_tokens": rendered["max_input_tokens"],
        "movie_id": movie_id,
        "identifier": identifier,
    }


def _prompt_artifacts(
    movies: list[dict[str, Any]], config: BenchmarkRuntimeConfig
) -> list[dict[str, str]]:
    paths = set()
    for language in {item["language"] for item in movies}:
        paths.update(PROMPTS.get(config.prompt_path("task3_actor", language)).source_paths)
    return [_artifact(path) for path in sorted(paths)]


def _split(
    settled: list[Any], identifiers: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    completed, failures = [], []
    for identifier, result in zip(identifiers, settled, strict=True):
        if isinstance(result, BaseException):
            failures.append(
                {
                    "identifier": identifier,
                    "error_type": type(result).__name__,
                    "error": str(result),
                }
            )
        else:
            completed.append(result)
    return completed, failures


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
