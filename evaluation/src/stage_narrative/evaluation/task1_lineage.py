from __future__ import annotations

import asyncio
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from ..clients import build_endpoint_pool_runtime, build_json_client
from ..io import atomic_write_json, load_json, sha256_file, sha256_json
from ..prompt_loader import PROMPTS
from ..temporal.benchmark_protocol import BenchmarkRuntimeConfig
from .task1_schemas import (
    validate_task1_development_lineage_review,
    validate_task1_state_lineage_review,
)


STATE_MAX_OUTPUT_TOKENS = 12000
DEVELOPMENT_MAX_OUTPUT_TOKENS = 4000


async def run_task1_lineage_review(
    *,
    review_pack_path: Path,
    config_path: Path,
    output_dir: Path,
    workers: int = 3,
    resume: bool = False,
    preflight_only: bool = False,
    client: Any | None = None,
) -> Path:
    if workers <= 0:
        raise ValueError("Task 1 lineage review workers must be positive")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite Task 1 lineage review: {output_dir}")
    state_dir = output_dir / "partials" / "states"
    development_dir = output_dir / "partials" / "developments"
    failure_dir = output_dir / "partials" / "failures"
    for path in (output_dir, state_dir, development_dir, failure_dir):
        path.mkdir(parents=True, exist_ok=True)
    review = _object(review_pack_path)
    config = BenchmarkRuntimeConfig.load(config_path)
    counter = config.build_token_counter()
    language_key = "zh" if str(review["language"]).casefold() in {"zh", "chinese"} else "en"
    state_specs = {
        row["character_id"]: {
            "source": row,
            "materialized": _materialize_state(row, language_key, config, counter),
        }
        for row in review["characters"]
    }
    contract = {
        "schema_version": "stage_task1_lineage_review_contract",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "review_pack": {
            "path": str(review_pack_path.resolve()),
            "sha256": sha256_file(review_pack_path.resolve()),
        },
        "config": {"path": str(config.source_path), "sha256": config.source_sha256},
        "model": config.evaluation_llm["model"],
        "temperature": config.evaluation_llm["temperature"],
        "semantic_samples_per_call": 1,
        "prompt_artifacts": _prompt_artifacts(language_key),
        "preflight": {
            "character_count": len(state_specs),
            "state_review_calls": len(state_specs),
            "development_review_calls": len(state_specs),
            "max_state_input_tokens": max(
                row["materialized"]["accounted_input_tokens"] for row in state_specs.values()
            ),
            "max_development_input_tokens": None,
            "development_budget_status": (
                "deferred_until_state_lineage_is_frozen; actual prompts are checked "
                "before development calls"
            ),
            "truncated_count": 0,
        },
    }
    contract_path = output_dir / "run_contract.json"
    if resume and contract_path.is_file():
        existing = _object(contract_path)
        if _without_created_at(existing) != _without_created_at(contract):
            raise ValueError("Task 1 lineage review resume contract drift")
    else:
        atomic_write_json(contract_path, contract)
    if preflight_only:
        path = output_dir / "preflight.json"
        atomic_write_json(
            path,
            {
                "schema_version": "stage_task1_lineage_review_preflight",
                "status": "passed_zero_call",
                "run_contract_sha256": sha256_file(contract_path),
                **contract["preflight"],
            },
        )
        return path

    endpoint_runtime = None
    if client is None:
        endpoint_runtime = build_endpoint_pool_runtime(config.evaluation_llm)
        state_client = build_json_client(
            {
                **config.evaluation_llm,
                "json_response_format": True,
                "max_tokens": STATE_MAX_OUTPUT_TOKENS,
            },
            endpoint_runtime=endpoint_runtime,
        )
        development_client = build_json_client(
            {
                **config.evaluation_llm,
                "json_response_format": True,
                "max_tokens": DEVELOPMENT_MAX_OUTPUT_TOKENS,
            },
            endpoint_runtime=endpoint_runtime,
        )
    else:
        state_client = client
        development_client = client
    semaphore = asyncio.Semaphore(workers)
    state_settled = await asyncio.gather(
        *[
            _run_state(
                character_id,
                spec=spec,
                path=state_dir / f"{character_id}.json",
                failure_dir=failure_dir,
                resume=resume,
                client=state_client,
                semaphore=semaphore,
            )
            for character_id, spec in state_specs.items()
        ],
        return_exceptions=True,
    )
    state_failures = [row for row in state_settled if isinstance(row, BaseException)]
    if state_failures:
        raise RuntimeError(
            f"{len(state_failures)} Task 1 state lineage calls failed; first: "
            f"{type(state_failures[0]).__name__}: {state_failures[0]}"
        )
    states = [row for row in state_settled if isinstance(row, dict)]
    states_by_character = {row["character_id"]: row for row in states}
    development_specs = {}
    for character_id, spec in state_specs.items():
        materialized = _materialize_development(
            spec["source"],
            states=states_by_character[character_id]["states"],
            language_key=language_key,
            config=config,
            counter=counter,
        )
        development_specs[character_id] = {"source": spec["source"], "materialized": materialized}
    development_settled = await asyncio.gather(
        *[
            _run_development(
                character_id,
                spec=spec,
                states=states_by_character[character_id]["states"],
                path=development_dir / f"{character_id}.json",
                failure_dir=failure_dir,
                resume=resume,
                client=development_client,
                semaphore=semaphore,
            )
            for character_id, spec in development_specs.items()
        ],
        return_exceptions=True,
    )
    development_failures = [
        row for row in development_settled if isinstance(row, BaseException)
    ]
    if development_failures:
        raise RuntimeError(
            f"{len(development_failures)} Task 1 development lineage calls failed; first: "
            f"{type(development_failures[0]).__name__}: {development_failures[0]}"
        )
    developments = [row for row in development_settled if isinstance(row, dict)]
    developments_by_character = {row["character_id"]: row for row in developments}
    draft_path = output_dir / "task1_lineage_decisions_draft.json"
    atomic_write_json(
        draft_path,
        {
            "schema_version": "stage_task1_lineage_review_draft",
            "movie_id": review["movie_id"],
            "review_status": "requires_manual_review",
            "reviewer_id": None,
            "completed_at": None,
            "characters": [
                {
                    "character_id": row["character_id"],
                    "character": row["character"],
                    "states": states_by_character[row["character_id"]]["states"],
                    "developments": developments_by_character[row["character_id"]][
                        "developments"
                    ],
                }
                for row in review["characters"]
            ],
        },
    )
    endpoint_snapshot = await endpoint_runtime.snapshot() if endpoint_runtime else None
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task1_lineage_review_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "machine_draft_requires_manual_review",
            "movie_id": review["movie_id"],
            "python_executable": sys.executable,
            "model": config.evaluation_llm["model"],
            "endpoint_pool": endpoint_snapshot,
            "counts": {
                "characters": len(states),
                "state_lineages": sum(len(row["states"]) for row in states),
                "development_lineages": sum(
                    len(row["developments"]) for row in developments
                ),
                "failure_count": 0,
                "truncated_count": 0,
            },
            "run_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            "outputs": [{"path": str(draft_path), "sha256": sha256_file(draft_path)}],
        },
    )
    return manifest_path


def promote_reviewed_task1_lineage(
    *,
    review_pack_path: Path,
    reviewed_draft_path: Path,
    reviewer_id: str,
    output_path: Path,
) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Task 1 lineage decisions: {output_path}")
    if not reviewer_id.strip():
        raise ValueError("Task 1 lineage promotion requires a reviewer ID")
    review = _object(review_pack_path)
    draft = _object(reviewed_draft_path)
    if draft.get("movie_id") != review["movie_id"]:
        raise ValueError("Task 1 reviewed lineage draft movie ID drift")
    source_by_character = {row["character_id"]: row for row in review["characters"]}
    for decision in draft["characters"]:
        source = source_by_character[decision["character_id"]]
        validate_task1_state_lineage_review(
            {"states": decision["states"]}, source=source
        )
        validate_task1_development_lineage_review(
            {"developments": decision["developments"]},
            source=source,
            states=decision["states"],
        )
    payload = {
        "schema_version": "stage_task1_lineage_decisions",
        "movie_id": review["movie_id"],
        "reviewer_id": reviewer_id.strip(),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "characters": draft["characters"],
    }
    atomic_write_json(output_path, payload)
    return output_path


def apply_task1_lineage_ref_repairs(
    *,
    review_pack_path: Path,
    failure_artifact_path: Path,
    repair_map_path: Path,
    output_path: Path,
) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Task 1 lineage partial: {output_path}")
    review = _object(review_pack_path)
    failure = _object(failure_artifact_path)
    repairs = _object(repair_map_path)
    character_id = str(failure["identifier"])
    if repairs.get("schema_version") != "stage_task1_lineage_ref_repair":
        raise ValueError("Unknown Task 1 lineage ref-repair schema")
    if repairs.get("character_id") != character_id:
        raise ValueError("Task 1 lineage repair character ID drift")
    source = next(
        row for row in review["characters"] if row["character_id"] == character_id
    )
    raw = deepcopy(failure["parsed_response"])
    if "states" not in raw:
        raise ValueError("Task 1 ref repair currently requires a state-lineage response")
    by_key = {row["lineage_key"]: row for row in raw["states"]}
    if len(by_key) != len(raw["states"]):
        raise ValueError("Task 1 lineage repair source has duplicate lineage keys")
    for row in repairs.get("drop", []):
        lineage = by_key.get(row["lineage_key"])
        if lineage is None or row["member_ref"] not in lineage["member_refs"]:
            raise ValueError(f"Task 1 lineage repair drop is invalid: {row}")
        lineage["member_refs"].remove(row["member_ref"])
    for row in repairs.get("add", []):
        lineage = by_key.get(row["lineage_key"])
        if lineage is None or row["member_ref"] in lineage["member_refs"]:
            raise ValueError(f"Task 1 lineage repair add is invalid: {row}")
        lineage["member_refs"].append(row["member_ref"])
    raw["states"] = [row for row in raw["states"] if row["member_refs"]]
    raw = {"states": raw["states"]}
    materialized = _materialize_state(
        source,
        "zh" if str(review["language"]).casefold() in {"zh", "chinese"} else "en",
        BenchmarkRuntimeConfig.load(Path(repairs["config_path"])),
        BenchmarkRuntimeConfig.load(Path(repairs["config_path"])).build_token_counter(),
    )
    if failure.get("prompt_sha256") != materialized["prompt_sha256"]:
        raise ValueError("Task 1 lineage repair prompt hash drift")
    expanded = _expand_member_refs(
        raw, field="states", member_ref_map=materialized["member_ref_map"]
    )
    validated = validate_task1_state_lineage_review(expanded, source=source)
    atomic_write_json(
        output_path,
        {
            "character_id": character_id,
            "states": validated["states"],
            "prompt_tokens": failure["prompt_tokens"],
            "prompt_sha256": failure["prompt_sha256"],
            "generator_metadata": failure.get("generator_metadata"),
            "manual_ref_repairs": {
                "repair_map": str(repair_map_path.resolve()),
                "repair_map_sha256": sha256_file(repair_map_path.resolve()),
                "reviewer_id": repairs["reviewer_id"],
                "rationale": repairs["rationale"],
            },
        },
    )
    return output_path


def _materialize_state(
    source: dict[str, Any], language_key: str, config: BenchmarkRuntimeConfig, counter: Any
) -> dict[str, Any]:
    reviewed = []
    member_ref_map = {}
    for checkpoint in source["reviewed_checkpoint_rubrics"]:
        claims = []
        for claim in checkpoint["state_claims"]:
            ref = f"R{len(member_ref_map) + 1}"
            member_ref_map[ref] = {
                "instance_id": checkpoint["instance_id"],
                "gold_local_id": claim["local_id"],
            }
            claims.append(
                [ref, claim["local_id"], claim["claim"], claim["supporting_scene_orders"]]
            )
        reviewed.append(
            {
                "instance_id": checkpoint["instance_id"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_scene_order": checkpoint["checkpoint"]["current_scene_order"],
                "state_claims": claims,
            }
        )
    system, user = PROMPTS.render(
        f"{language_key}/evaluation/task1_state_lineage_review",
        focal_character=source["character"],
        ordered_checkpoints=source["checkpoint_ids"],
        reviewed_state_claims=reviewed,
        allowed_member_refs=list(member_ref_map),
        expected_member_count=len(member_ref_map),
    )
    return _budgeted(
        system,
        user,
        "task1_state_lineage_review",
        config,
        counter,
        max_output_tokens=STATE_MAX_OUTPUT_TOKENS,
    ) | {
        "member_ref_map": member_ref_map,
        "language_key": language_key,
        "allowed_checkpoint_ids": source["checkpoint_ids"],
        "allowed_state_keys": [],
        "member_ref_rows": reviewed,
    }


def _materialize_development(
    source: dict[str, Any],
    *,
    states: list[dict[str, Any]],
    language_key: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
) -> dict[str, Any]:
    reviewed = []
    member_ref_map = {}
    for checkpoint in source["reviewed_checkpoint_rubrics"]:
        claims = []
        for claim in checkpoint["development_claims"]:
            ref = f"Q{len(member_ref_map) + 1}"
            member_ref_map[ref] = {
                "instance_id": checkpoint["instance_id"],
                "gold_local_id": claim["local_id"],
            }
            claims.append(
                [ref, claim["local_id"], claim["claim"], claim["supporting_scene_orders"]]
            )
        reviewed.append(
            {
                "instance_id": checkpoint["instance_id"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_scene_order": checkpoint["checkpoint"]["current_scene_order"],
                "development_claims": claims,
            }
        )
    system, user = PROMPTS.render(
        f"{language_key}/evaluation/task1_development_lineage_review",
        focal_character=source["character"],
        ordered_checkpoints=source["checkpoint_ids"],
        reviewed_development_claims=reviewed,
        reviewed_state_lineages=[
            {
                "lineage_key": row["lineage_key"],
                "canonical_claim": row["canonical_claim"],
                "valid_from_checkpoint_id": row["valid_from_checkpoint_id"],
                "valid_to_checkpoint_id": row["valid_to_checkpoint_id"],
                "superseded_by_lineage_key": row["superseded_by_lineage_key"],
            }
            for row in states
        ],
        source_developments=[],
        allowed_member_refs=list(member_ref_map),
        expected_member_count=len(member_ref_map),
    )
    return _budgeted(
        system,
        user,
        "task1_development_lineage_review",
        config,
        counter,
        max_output_tokens=DEVELOPMENT_MAX_OUTPUT_TOKENS,
    ) | {
        "member_ref_map": member_ref_map,
        "language_key": language_key,
        "allowed_checkpoint_ids": source["checkpoint_ids"],
        "allowed_state_keys": [row["lineage_key"] for row in states],
        "member_ref_rows": reviewed,
    }


def _budgeted(
    system: str,
    user: str,
    prompt_path: str,
    config: BenchmarkRuntimeConfig,
    counter: Any,
    *,
    max_output_tokens: int,
) -> dict[str, Any]:
    prompt_tokens = counter.count(system) + counter.count(user)
    accounted = prompt_tokens + config.reserved_chat_template_tokens
    budget = config.call_budgets.get("asset_review", config.call_budgets["task1_judge"])
    maximum = budget.context_window - max_output_tokens - budget.safety_margin_tokens
    if accounted > maximum:
        raise ValueError(f"Task 1 lineage prompt exceeds input budget: {accounted}>{maximum}")
    return {
        "system_prompt": system,
        "user_prompt": user,
        "prompt_path": prompt_path,
        "prompt_tokens": prompt_tokens,
        "accounted_input_tokens": accounted,
        "max_input_tokens": maximum,
        "max_output_tokens": max_output_tokens,
        "prompt_sha256": sha256_json({"system": system, "user": user}),
    }


async def _run_state(
    character_id: str,
    *,
    spec: dict[str, Any],
    path: Path,
    failure_dir: Path,
    resume: bool,
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    materialized = spec["materialized"]
    if resume and path.is_file():
        existing = _object(path)
        if existing.get("prompt_sha256") != materialized["prompt_sha256"]:
            raise ValueError(f"Task 1 state lineage partial drift: {path}")
        _archive_recovered_failure(failure_dir / f"{character_id}.json")
        return existing
    failure_path = failure_dir / f"{character_id}.json"
    if resume and failure_path.is_file():
        failed = _object(failure_path)
        if (
            failed.get("prompt_sha256") == materialized["prompt_sha256"]
            and isinstance(failed.get("parsed_response"), dict)
        ):
            repair = None
            try:
                expanded = _expand_member_refs(
                    failed["parsed_response"],
                    field="states",
                    member_ref_map=materialized["member_ref_map"],
                )
                validated = validate_task1_state_lineage_review(
                    expanded, source=spec["source"]
                )
            except ValueError as exc:
                repair = _deterministic_lineage_repair(
                    field="states",
                    previous_response=failed["parsed_response"],
                    validation_error=str(exc),
                    materialized=materialized,
                )
                expanded = _expand_member_refs(
                    repair["payload"],
                    field="states",
                    member_ref_map=materialized["member_ref_map"],
                )
                validated = validate_task1_state_lineage_review(
                    expanded, source=spec["source"]
                )
            record = {
                "character_id": character_id,
                "states": validated["states"],
                "prompt_tokens": materialized["prompt_tokens"],
                "prompt_sha256": materialized["prompt_sha256"],
                "generator_metadata": failed.get("generator_metadata"),
                "format_normalization": {
                    "recovered_from": str(failure_path),
                    "discarded_nonsemantic_top_level_fields": sorted(
                        set(failed["parsed_response"]) - {"states"}
                    ),
                    "deterministic_repair": repair["audit"] if repair is not None else None,
                },
            }
            atomic_write_json(path, record)
            _archive_recovered_failure(failure_path)
            return record
    response = None
    try:
        async with semaphore:
            response = await client.generate_json(
                system_prompt=materialized["system_prompt"],
                user_prompt=materialized["user_prompt"],
                stage=f"task1_state_lineage_review:{character_id}",
            )
        expanded = _expand_member_refs(
            response.data,
            field="states",
            member_ref_map=materialized["member_ref_map"],
        )
        validated = validate_task1_state_lineage_review(expanded, source=spec["source"])
    except Exception as exc:
        if response is not None:
            exc.parsed_response = response.data
            exc.generator_metadata = response.metadata
        _write_failure(failure_dir, character_id, materialized, exc)
        raise
    record = {
        "character_id": character_id,
        "states": validated["states"],
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "generator_metadata": response.metadata,
    }
    atomic_write_json(path, record)
    failure_path.unlink(missing_ok=True)
    return record


async def _run_development(
    character_id: str,
    *,
    spec: dict[str, Any],
    states: list[dict[str, Any]],
    path: Path,
    failure_dir: Path,
    resume: bool,
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    materialized = spec["materialized"]
    if resume and path.is_file():
        existing = _object(path)
        if existing.get("prompt_sha256") != materialized["prompt_sha256"]:
            raise ValueError(f"Task 1 development lineage partial drift: {path}")
        _archive_recovered_failure(failure_dir / f"{character_id}.json")
        return existing
    failure_path = failure_dir / f"{character_id}.json"
    if resume and failure_path.is_file():
        failed = _object(failure_path)
        if (
            failed.get("prompt_sha256") == materialized["prompt_sha256"]
            and isinstance(failed.get("parsed_response"), dict)
        ):
            repair = None
            try:
                expanded = _expand_member_refs(
                    failed["parsed_response"],
                    field="developments",
                    member_ref_map=materialized["member_ref_map"],
                )
                validated = validate_task1_development_lineage_review(
                    expanded, source=spec["source"], states=states
                )
            except ValueError as exc:
                repair = _deterministic_lineage_repair(
                    field="developments",
                    previous_response=failed["parsed_response"],
                    validation_error=str(exc),
                    materialized=materialized,
                )
                expanded = _expand_member_refs(
                    repair["payload"],
                    field="developments",
                    member_ref_map=materialized["member_ref_map"],
                )
                validated = validate_task1_development_lineage_review(
                    expanded, source=spec["source"], states=states
                )
            record = {
                "character_id": character_id,
                "developments": validated["developments"],
                "prompt_tokens": materialized["prompt_tokens"],
                "prompt_sha256": materialized["prompt_sha256"],
                "generator_metadata": failed.get("generator_metadata"),
                "format_normalization": {
                    "recovered_from": str(failure_path),
                    "discarded_nonsemantic_top_level_fields": sorted(
                        set(failed["parsed_response"]) - {"developments"}
                    ),
                    "deterministic_repair": repair["audit"] if repair is not None else None,
                },
            }
            atomic_write_json(path, record)
            _archive_recovered_failure(failure_path)
            return record
    response = None
    try:
        async with semaphore:
            response = await client.generate_json(
                system_prompt=materialized["system_prompt"],
                user_prompt=materialized["user_prompt"],
                stage=f"task1_development_lineage_review:{character_id}",
            )
        expanded = _expand_member_refs(
            response.data,
            field="developments",
            member_ref_map=materialized["member_ref_map"],
        )
        validated = validate_task1_development_lineage_review(
            expanded, source=spec["source"], states=states
        )
    except Exception as exc:
        if response is not None:
            exc.parsed_response = response.data
            exc.generator_metadata = response.metadata
        _write_failure(failure_dir, character_id, materialized, exc)
        raise
    record = {
        "character_id": character_id,
        "developments": validated["developments"],
        "prompt_tokens": materialized["prompt_tokens"],
        "prompt_sha256": materialized["prompt_sha256"],
        "generator_metadata": response.metadata,
    }
    atomic_write_json(path, record)
    failure_path.unlink(missing_ok=True)
    return record


def _expand_member_refs(
    payload: dict[str, Any], *, field: str, member_ref_map: dict[str, dict[str, str]]
) -> dict[str, Any]:
    if not isinstance(payload, dict) or field not in payload or not isinstance(
        payload[field], list
    ):
        raise ValueError(f"Task 1 lineage response must contain only {field}")
    extras = set(payload) - {field}
    if not extras <= {"meta", "_metadata", "lineage_count"}:
        raise ValueError(f"Task 1 lineage response has unsupported fields: {sorted(extras)}")
    output = deepcopy({field: payload[field]})
    observed = []
    for row in output[field]:
        if not isinstance(row, dict) or "member_refs" not in row or "members" in row:
            raise ValueError("Task 1 lineage response requires compact member_refs")
        refs = row.pop("member_refs")
        if not isinstance(refs, list) or not refs or not all(
            isinstance(value, str) for value in refs
        ):
            raise ValueError("Task 1 lineage member_refs must be a nonempty string array")
        if any(value not in member_ref_map for value in refs):
            raise ValueError("Task 1 lineage response contains an unknown member ref")
        observed.extend(refs)
        row["members"] = [member_ref_map[value] for value in refs]
    if len(observed) != len(set(observed)) or set(observed) != set(member_ref_map):
        raise ValueError("Task 1 lineage compact refs require exact reviewed-claim coverage")
    return output


def _deterministic_lineage_repair(
    *, field: str, previous_response: dict[str, Any], validation_error: str,
    materialized: dict[str, Any]
) -> dict[str, Any]:
    prefix = "GS" if field == "states" else "GD"
    allowed_refs = set(materialized["member_ref_map"])
    checkpoints = list(materialized["allowed_checkpoint_ids"])
    checkpoint_index = {value: index for index, value in enumerate(checkpoints)}
    ref_details = {}
    claim_field = "state_claims" if field == "states" else "development_claims"
    for checkpoint in materialized["member_ref_rows"]:
        for ref, _local_id, claim, _support in checkpoint[claim_field]:
            ref_details[ref] = {
                "claim": claim,
                "checkpoint_id": checkpoint["checkpoint_id"],
            }
    source_rows = previous_response.get(field, [])
    if not isinstance(source_rows, list):
        source_rows = []
    repaired = []
    observed = set()
    old_to_new = {}
    dropped_unknown = []
    dropped_duplicate = []
    for source in source_rows:
        if not isinstance(source, dict):
            continue
        refs = []
        for ref in source.get("member_refs", []):
            if ref not in allowed_refs:
                dropped_unknown.append(ref)
            elif ref in observed:
                dropped_duplicate.append(ref)
            else:
                observed.add(ref)
                refs.append(ref)
        if not refs:
            continue
        new_key = f"{prefix}{len(repaired) + 1}"
        old_to_new[str(source.get("lineage_key") or new_key)] = new_key
        member_checkpoints = [ref_details[ref]["checkpoint_id"] for ref in refs]
        first = min(member_checkpoints, key=checkpoint_index.__getitem__)
        last = max(member_checkpoints, key=checkpoint_index.__getitem__)
        row = {
            "lineage_key": new_key,
            "canonical_claim": str(source.get("canonical_claim") or ref_details[refs[0]]["claim"]),
            "member_refs": refs,
        }
        if field == "states":
            start = source.get("valid_from_checkpoint_id")
            end = source.get("valid_to_checkpoint_id")
            if start not in checkpoint_index:
                start = first
            if end is not None and end not in checkpoint_index:
                end = last
            if end is not None and checkpoint_index[start] > checkpoint_index[end]:
                start, end = first, last
            row.update(
                {
                    "source_state_ids": [],
                    "valid_from_checkpoint_id": start,
                    "valid_to_checkpoint_id": end,
                    "superseded_by_lineage_key": source.get("superseded_by_lineage_key"),
                }
            )
        else:
            trigger = source.get("trigger_checkpoint_id")
            effective = source.get("effective_checkpoint_id")
            if trigger not in checkpoint_index:
                trigger = first
            if effective not in checkpoint_index:
                effective = first
            if checkpoint_index[trigger] > checkpoint_index[effective]:
                trigger = effective
            allowed_states = set(materialized["allowed_state_keys"])
            row.update(
                {
                    "source_development_ids": [],
                    "trigger_checkpoint_id": trigger,
                    "effective_checkpoint_id": effective,
                    "affected_state_keys": [
                        value for value in source.get("affected_state_keys", []) if value in allowed_states
                    ],
                    "resulting_state_keys": [
                        value for value in source.get("resulting_state_keys", []) if value in allowed_states
                    ],
                    "superseded_state_keys": [
                        value for value in source.get("superseded_state_keys", []) if value in allowed_states
                    ],
                }
            )
        repaired.append(row)
    missing = sorted(allowed_refs - observed, key=lambda value: int(value[1:]))
    for ref in missing:
        checkpoint_id = ref_details[ref]["checkpoint_id"]
        row = {
            "lineage_key": f"{prefix}{len(repaired) + 1}",
            "canonical_claim": ref_details[ref]["claim"],
            "member_refs": [ref],
        }
        if field == "states":
            row.update(
                {
                    "source_state_ids": [],
                    "valid_from_checkpoint_id": checkpoint_id,
                    "valid_to_checkpoint_id": checkpoint_id,
                    "superseded_by_lineage_key": None,
                }
            )
        else:
            row.update(
                {
                    "source_development_ids": [],
                    "trigger_checkpoint_id": checkpoint_id,
                    "effective_checkpoint_id": checkpoint_id,
                    "affected_state_keys": [],
                    "resulting_state_keys": [],
                    "superseded_state_keys": [],
                }
            )
        repaired.append(row)
    if field == "states":
        valid_keys = {row["lineage_key"] for row in repaired}
        for row in repaired:
            target = old_to_new.get(str(row["superseded_by_lineage_key"]))
            row["superseded_by_lineage_key"] = target if target in valid_keys else None
    return {
        "payload": {field: repaired},
        "audit": {
            "policy": "preserve_valid_groups_then_singleton_missing_refs",
            "validation_error": validation_error,
            "input_lineage_count": len(source_rows),
            "output_lineage_count": len(repaired),
            "missing_refs_materialized_as_singletons": missing,
            "dropped_unknown_refs": sorted(set(dropped_unknown)),
            "dropped_duplicate_refs": sorted(set(dropped_duplicate)),
        },
    }


def _prompt_artifacts(language_key: str) -> list[dict[str, str]]:
    paths = set()
    for name in (
        "task1_state_lineage_review",
        "task1_development_lineage_review",
        "task1_lineage_format_repair",
    ):
        paths.update(PROMPTS.get(f"{language_key}/evaluation/{name}").source_paths)
    return [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(paths)]


def _write_failure(
    failure_dir: Path, identifier: str, materialized: dict[str, Any], exc: Exception
) -> None:
    atomic_write_json(
        failure_dir / f"{identifier}.json",
        {
            "identifier": identifier,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "prompt_sha256": materialized["prompt_sha256"],
            "prompt_tokens": materialized["prompt_tokens"],
            "parsed_response": getattr(exc, "parsed_response", None),
            "generator_metadata": getattr(exc, "generator_metadata", None),
            "raw_response": getattr(exc, "raw_text", None),
        },
    )


def _archive_recovered_failure(path: Path) -> None:
    if not path.is_file():
        return
    archive_dir = path.parents[1] / "recovered_failures"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{path.stem}-{sha256_file(path)[:12]}.json"
    if archive_path.exists():
        path.unlink()
    else:
        path.replace(archive_path)


def _without_created_at(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at"}


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path.resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
