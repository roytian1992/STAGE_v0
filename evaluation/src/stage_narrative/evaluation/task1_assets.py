from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..io import atomic_write_json, load_json, sha256_file
from ..models import stable_id
from .task1_schemas import validate_task1_private_assets


def build_task1_review_pack(
    *,
    task1_instances_path: Path,
    rolling_plans_path: Path,
    reviewed_gold_path: Path,
    rubric_candidates_path: Path,
    state_ledger_path: Path,
    development_graph_path: Path,
    output_dir: Path,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Task 1 review pack: {output_dir}")
    instances_payload = _object(task1_instances_path)
    plans_payload = _object(rolling_plans_path)
    gold_payload = _object(reviewed_gold_path)
    candidate_payload = _object(rubric_candidates_path)
    state_payload = _object(state_ledger_path)
    development_payload = _object(development_graph_path)
    movie_ids = {
        str(instances_payload["movie_id"]),
        str(gold_payload["movie_id"]),
        str(candidate_payload["movie_id"]),
    }
    if len(movie_ids) != 1:
        raise ValueError(f"Task 1 review pack movie IDs differ: {sorted(movie_ids)}")
    movie_id = next(iter(movie_ids))
    instances = instances_payload["instances"]
    gold_by_id = {row["instance_id"]: row for row in gold_payload["rubrics"]}
    candidates = {row["instance_id"]: row for row in candidate_payload["candidates"]}
    if {row["instance_id"] for row in instances} != set(gold_by_id) or set(gold_by_id) != set(candidates):
        raise ValueError("Task 1 inputs require exact instance ID coverage")
    checkpoint_id_by_instance = {
        instance_id: row["construction_provenance"]["checkpoint_id"]
        for instance_id, row in candidates.items()
    }
    public_instances = []
    for row in instances:
        public_instances.append(
            {
                "instance_id": row["instance_id"],
                "checkpoint_id": checkpoint_id_by_instance[row["instance_id"]],
                "focal_character": row["focal_character"],
                "aliases": row["aliases"],
                "checkpoint": {
                    "previous_scene_order": row["checkpoint"]["previous_scene_order"],
                    "current_scene_order": row["checkpoint"]["current_scene_order"],
                },
                "rolling_boundary": row["rolling_boundary"],
                "instruction": (
                    "Track the focal character through the screenplay up to this checkpoint. "
                    "Report current state and developments since the previous checkpoint with "
                    "supporting scene orders."
                ),
                "output_fields": [
                    "current_state",
                    "developments_since_previous_checkpoint",
                ],
            }
        )
    language = str(plans_payload["language"])
    public_payload = {
        "schema_version": "stage_task1_public_instances",
        "movie_id": movie_id,
        "language": language,
        "instance_count": len(public_instances),
        "instances": public_instances,
    }
    public_path = output_dir / "task1_public_instances.json"
    atomic_write_json(public_path, public_payload)

    rolling_plans = {
        **plans_payload,
        "schema_version": "stage_task1_standard_24k_rolling_plan",
        "prompt_path": f"{language}/evaluation/task1_prediction",
        "prompt_source_paths": [],
        "scored_output_fields": [
            "current_state",
            "developments_since_previous_checkpoint",
        ],
        "unresolved_threads_policy": "private_unscored_memory_only",
    }
    plan_path = output_dir / "task1_rolling_plans.json"
    atomic_write_json(plan_path, rolling_plans)

    state_by_id = {row["state_id"]: row for row in state_payload["states"]}
    development_by_id = {
        row["development_id"]: row for row in development_payload["developments"]
    }
    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in public_instances:
        gold = gold_by_id[instance["instance_id"]]
        candidate = candidates[instance["instance_id"]]
        by_character[gold["character_id"]].append(
            {
                "instance": instance,
                "gold": gold,
                "candidate": candidate,
            }
        )
    review_characters = []
    for character_id, rows in sorted(by_character.items()):
        rows.sort(key=lambda value: value["instance"]["checkpoint"]["current_scene_order"])
        source_state_ids = sorted(
            {
                value
                for row in rows
                for value in row["candidate"]["construction_provenance"]["state_ids"]
                if value in state_by_id
            }
        )
        source_development_ids = sorted(
            {
                value
                for row in rows
                for value in row["candidate"]["construction_provenance"]["development_ids"]
                if value in development_by_id
            }
        )
        reviewed = []
        state_member_refs = []
        development_member_refs = []
        for row in rows:
            gold = row["gold"]
            rubric = gold["rubric"]
            states = [
                {**claim, "source_pool": "current_state_claims"}
                for claim in rubric["current_state_claims"]
            ]
            state_member_refs.extend(
                f"{gold['instance_id']}|{claim['local_id']}" for claim in states
            )
            developments = list(rubric["development_claims"])
            development_member_refs.extend(
                f"{gold['instance_id']}|{claim['local_id']}" for claim in developments
            )
            reviewed.append(
                {
                    "instance_id": gold["instance_id"],
                    "checkpoint_id": row["instance"]["checkpoint_id"],
                    "checkpoint": {
                        **row["instance"]["checkpoint"],
                        "control_types": gold["checkpoint"]["control_types"],
                    },
                    "state_claims": states,
                    "invariant_claims": rubric["invariant_claims"],
                    "development_claims": developments,
                    "salient_future_negatives": rubric["salient_future_negatives"],
                }
            )
        review_characters.append(
            {
                "character_id": character_id,
                "character": rows[0]["gold"]["character"],
                "checkpoint_ids": [row["instance"]["checkpoint_id"] for row in rows],
                "reviewed_checkpoint_rubrics": reviewed,
                "source_states": [_compact_state(state_by_id[value]) for value in source_state_ids],
                "source_developments": [
                    _compact_development(development_by_id[value])
                    for value in source_development_ids
                ],
                "required_state_member_refs": sorted(state_member_refs),
                "required_development_member_refs": sorted(development_member_refs),
            }
        )
    review_payload = {
        "schema_version": "stage_task1_lineage_review_pack",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "movie_id": movie_id,
        "language": language,
        "public_instances_path": str(public_path),
        "public_instances_sha256": sha256_file(public_path),
        "character_count": len(review_characters),
        "checkpoint_count": len(public_instances),
        "characters": review_characters,
    }
    review_path = output_dir / "task1_lineage_review_pack.json"
    atomic_write_json(review_path, review_payload)
    template_path = output_dir / "task1_lineage_decisions_template.json"
    atomic_write_json(
        template_path,
        {
            "schema_version": "stage_task1_lineage_decisions",
            "movie_id": movie_id,
            "reviewer_id": "REPLACE_WITH_REVIEWER_ID",
            "completed_at": None,
            "characters": [
                {
                    "character_id": row["character_id"],
                    "character": row["character"],
                    "states": None,
                    "developments": None,
                }
                for row in review_characters
            ],
        },
    )
    manifest_path = output_dir / "manifest.json"
    inputs = [
        task1_instances_path,
        rolling_plans_path,
        reviewed_gold_path,
        rubric_candidates_path,
        state_ledger_path,
        development_graph_path,
    ]
    outputs = [public_path, plan_path, review_path, template_path]
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task1_review_pack_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "awaiting_explicit_lineage_review",
            "movie_id": movie_id,
            "counts": {
                "characters": len(review_characters),
                "checkpoints": len(public_instances),
                "state_claim_members": sum(
                    len(row["required_state_member_refs"]) for row in review_characters
                ),
                "development_claim_members": sum(
                    len(row["required_development_member_refs"]) for row in review_characters
                ),
            },
            "inputs": [
                {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
                for path in inputs
            ],
            "outputs": [
                {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
                for path in outputs
            ],
        },
    )
    return manifest_path


def finalize_task1_assets(
    *, review_pack_path: Path, decisions_path: Path, output_dir: Path
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Task 1 assets: {output_dir}")
    review = _object(review_pack_path)
    decisions = _object(decisions_path)
    _validate_lineage_decisions(review, decisions)
    public_path = Path(review["public_instances_path"])
    public = _object(public_path)
    public_by_id = {row["instance_id"]: row for row in public["instances"]}
    review_by_character = {row["character_id"]: row for row in review["characters"]}
    trajectories = []
    for decision in decisions["characters"]:
        source = review_by_character[decision["character_id"]]
        checkpoint_ids = source["checkpoint_ids"]
        checkpoint_index = {value: index for index, value in enumerate(checkpoint_ids)}
        reviewed_by_checkpoint = {
            row["checkpoint_id"]: row for row in source["reviewed_checkpoint_rubrics"]
        }
        state_id_by_key = {
            row["lineage_key"]: stable_id(
                "task1-state",
                review["movie_id"],
                decision["character_id"],
                sorted(f"{m['instance_id']}|{m['gold_local_id']}" for m in row["members"]),
            )
            for row in decision["states"]
        }
        development_id_by_key = {
            row["lineage_key"]: stable_id(
                "task1-development",
                review["movie_id"],
                decision["character_id"],
                sorted(f"{m['instance_id']}|{m['gold_local_id']}" for m in row["members"]),
            )
            for row in decision["developments"]
        }
        state_members = _members_by_ref(decision["states"], "lineage_key")
        development_members = _members_by_ref(decision["developments"], "lineage_key")
        states = []
        for row in decision["states"]:
            member_claims = [
                _reviewed_claim(source, member["instance_id"], member["gold_local_id"], "state")
                for member in row["members"]
            ]
            states.append(
                {
                    "stable_state_id": state_id_by_key[row["lineage_key"]],
                    "claim": row["canonical_claim"],
                    "source_state_ids": sorted(set(row["source_state_ids"])),
                    "valid_from_checkpoint_id": row["valid_from_checkpoint_id"],
                    "valid_to_checkpoint_id": row["valid_to_checkpoint_id"],
                    "superseded_by_state_id": state_id_by_key.get(
                        row["superseded_by_lineage_key"]
                    ),
                    "supporting_scene_orders": sorted(
                        {
                            value
                            for claim in member_claims
                            for value in claim["supporting_scene_orders"]
                        }
                    ),
                }
            )
        developments = []
        for row in decision["developments"]:
            member_claims = [
                _reviewed_claim(
                    source, member["instance_id"], member["gold_local_id"], "development"
                )
                for member in row["members"]
            ]
            developments.append(
                {
                    "stable_development_id": development_id_by_key[row["lineage_key"]],
                    "claim": row["canonical_claim"],
                    "source_development_ids": sorted(set(row["source_development_ids"])),
                    "trigger_checkpoint_id": row["trigger_checkpoint_id"],
                    "effective_checkpoint_id": row["effective_checkpoint_id"],
                    "affected_state_ids": [state_id_by_key[value] for value in row["affected_state_keys"]],
                    "resulting_state_ids": [state_id_by_key[value] for value in row["resulting_state_keys"]],
                    "superseded_state_ids": [state_id_by_key[value] for value in row["superseded_state_keys"]],
                    "supporting_scene_orders": sorted(
                        {
                            value
                            for claim in member_claims
                            for value in claim["supporting_scene_orders"]
                        }
                    ),
                }
            )
        state_by_key = {row["lineage_key"]: row for row in decision["states"]}
        checkpoint_rubrics = []
        for checkpoint_id in checkpoint_ids:
            reviewed = reviewed_by_checkpoint[checkpoint_id]
            current_scene = reviewed["checkpoint"]["current_scene_order"]
            active_rows_by_id: dict[str, dict[str, Any]] = {}
            inactive_rows = []
            for claim in reviewed["state_claims"]:
                # Invariants are useful construction context, but they are not
                # checkpoint-specific current-state obligations.
                if claim["source_pool"] != "current_state_claims":
                    continue
                ref = f"{reviewed['instance_id']}|{claim['local_id']}"
                stable_state_id = state_id_by_key[state_members[ref]]
                active_rows_by_id.setdefault(
                    stable_state_id,
                    {
                        "stable_state_id": stable_state_id,
                        "claim": claim["claim"],
                        "supporting_scene_orders": [],
                    },
                )
                active_rows_by_id[stable_state_id]["supporting_scene_orders"] = sorted(
                    set(active_rows_by_id[stable_state_id]["supporting_scene_orders"])
                    | {
                        value
                        for value in claim["supporting_scene_orders"]
                        if value <= current_scene
                    }
                )
            active_rows = list(active_rows_by_id.values())

            for key, row in sorted(state_by_key.items()):
                end = (
                    checkpoint_index[row["valid_to_checkpoint_id"]]
                    if row["valid_to_checkpoint_id"] is not None
                    else len(checkpoint_ids) - 1
                )
                if checkpoint_index[checkpoint_id] <= end:
                    continue
                stable_state_id = state_id_by_key[key]
                if stable_state_id in active_rows_by_id:
                    continue
                claim = _member_claim_at_checkpoint(
                    source, row["members"], checkpoint_id, row["canonical_claim"], "state"
                )
                inactive_rows.append(
                    {
                        "stable_state_id": stable_state_id,
                        "claim": claim["claim"],
                        "supporting_scene_orders": [
                            value
                            for value in claim["supporting_scene_orders"]
                            if value <= current_scene
                        ],
                    }
                )

            development_rows_by_id: dict[str, dict[str, Any]] = {}
            for claim in reviewed["development_claims"]:
                ref = f"{reviewed['instance_id']}|{claim['local_id']}"
                stable_development_id = development_id_by_key[development_members[ref]]
                development_rows_by_id.setdefault(
                    stable_development_id,
                    {
                        "stable_development_id": stable_development_id,
                        "claim": claim["claim"],
                        "supporting_scene_orders": [],
                    },
                )
                development_rows_by_id[stable_development_id][
                    "supporting_scene_orders"
                ] = sorted(
                    set(
                        development_rows_by_id[stable_development_id][
                            "supporting_scene_orders"
                        ]
                    )
                    | {
                        value
                        for value in claim["supporting_scene_orders"]
                        if value <= current_scene
                    }
                )
            development_rows = list(development_rows_by_id.values())
            checkpoint_rubrics.append(
                {
                    "instance_id": reviewed["instance_id"],
                    "checkpoint_id": checkpoint_id,
                    "checkpoint": reviewed["checkpoint"],
                    "current_state_claims": [
                        {"local_id": f"S{index}", **row}
                        for index, row in enumerate(active_rows, start=1)
                    ],
                    "development_claims": [
                        {"local_id": f"D{index}", **row}
                        for index, row in enumerate(development_rows, start=1)
                    ],
                    "inactive_state_claims": [
                        {"local_id": f"X{index}", **row}
                        for index, row in enumerate(inactive_rows, start=1)
                    ],
                    "salient_future_negatives": reviewed["salient_future_negatives"],
                }
            )
        trajectories.append(
            {
                "character_id": decision["character_id"],
                "character": decision["character"],
                "checkpoint_ids": checkpoint_ids,
                "states": states,
                "developments": developments,
                "checkpoint_rubrics": checkpoint_rubrics,
            }
        )
    private = {
        "schema_version": "stage_task1_private_evaluator",
        "movie_id": review["movie_id"],
        "language": review["language"],
        "public_instances_sha256": sha256_file(public_path),
        "trajectory_count": len(trajectories),
        "checkpoint_count": sum(len(row["checkpoint_ids"]) for row in trajectories),
        "trajectories": trajectories,
    }
    validate_task1_private_assets(private)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_output = output_dir / "task1_public_instances.json"
    private_output = output_dir / "task1_private_evaluator.json"
    atomic_write_json(public_output, public)
    atomic_write_json(private_output, private)
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "stage_task1_asset_manifest",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "lineage_reviewed_task1_assets",
            "movie_id": review["movie_id"],
            "counts": {
                "characters": len(trajectories),
                "checkpoints": private["checkpoint_count"],
                "stable_states": sum(len(row["states"]) for row in trajectories),
                "stable_developments": sum(len(row["developments"]) for row in trajectories),
            },
            "inputs": [
                {"path": str(review_pack_path.resolve()), "sha256": sha256_file(review_pack_path.resolve())},
                {"path": str(decisions_path.resolve()), "sha256": sha256_file(decisions_path.resolve())},
            ],
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in (public_output, private_output)
            ],
        },
    )
    return manifest_path


def _validate_lineage_decisions(review: dict[str, Any], decisions: dict[str, Any]) -> None:
    if decisions.get("schema_version") != "stage_task1_lineage_decisions":
        raise ValueError("Unknown Task 1 lineage decision schema")
    if decisions.get("movie_id") != review["movie_id"]:
        raise ValueError("Task 1 lineage decision movie ID drift")
    if not str(decisions.get("reviewer_id") or "").strip() or not str(
        decisions.get("completed_at") or ""
    ).strip():
        raise ValueError("Task 1 lineage decisions require reviewer and completion time")
    review_by_character = {row["character_id"]: row for row in review["characters"]}
    decision_by_character = {row["character_id"]: row for row in decisions["characters"]}
    if len(decision_by_character) != len(decisions["characters"]) or set(decision_by_character) != set(
        review_by_character
    ):
        raise ValueError("Task 1 lineage decisions require exact character coverage")
    for character_id, source in review_by_character.items():
        decision = decision_by_character[character_id]
        if decision["character"] != source["character"]:
            raise ValueError("Task 1 lineage character name drift")
        state_keys = {row["lineage_key"] for row in decision["states"]}
        development_keys = {row["lineage_key"] for row in decision["developments"]}
        if len(state_keys) != len(decision["states"]) or len(development_keys) != len(
            decision["developments"]
        ):
            raise ValueError("Task 1 lineage keys must be unique")
        observed_states = _decision_member_refs(decision["states"])
        observed_developments = _decision_member_refs(decision["developments"])
        if observed_states != set(source["required_state_member_refs"]):
            raise ValueError(f"Task 1 state lineage coverage differs: {character_id}")
        if observed_developments != set(source["required_development_member_refs"]):
            raise ValueError(f"Task 1 development lineage coverage differs: {character_id}")
        checkpoints = set(source["checkpoint_ids"])
        source_states = {row["state_id"] for row in source["source_states"]}
        source_developments = {
            row["development_id"] for row in source["source_developments"]
        }
        for row in decision["states"]:
            if (
                row["valid_from_checkpoint_id"] not in checkpoints
                or row["valid_to_checkpoint_id"] is not None
                and row["valid_to_checkpoint_id"] not in checkpoints
                or row["superseded_by_lineage_key"] is not None
                and row["superseded_by_lineage_key"] not in state_keys
                or not set(row["source_state_ids"]) <= source_states
            ):
                raise ValueError(f"Task 1 state lineage reference is invalid: {character_id}")
        for row in decision["developments"]:
            if (
                row["trigger_checkpoint_id"] not in checkpoints
                or row["effective_checkpoint_id"] not in checkpoints
                or not set(row["source_development_ids"]) <= source_developments
                or not set(row["affected_state_keys"]) <= state_keys
                or not set(row["resulting_state_keys"]) <= state_keys
                or not set(row["superseded_state_keys"]) <= state_keys
            ):
                raise ValueError(f"Task 1 development lineage reference is invalid: {character_id}")


def _object(path: Path) -> dict[str, Any]:
    payload = load_json(path.resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _compact_state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "state_id",
            "dimension",
            "target_id_or_text",
            "polarity",
            "state_value",
            "valid_from_scene",
            "valid_until_scene",
            "source_unit_ids",
        )
    }


def _compact_development(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "development_id",
            "dimension",
            "target_id_or_text",
            "operation",
            "before_state_ids",
            "resulting_state_ids",
            "invariant_state_ids",
            "effective_from_scene",
            "consequence_visible_from_scene",
            "catalyst_event_ids",
        )
    }


def _decision_member_refs(rows: list[dict[str, Any]]) -> set[str]:
    refs = [
        f"{member['instance_id']}|{member['gold_local_id']}"
        for row in rows
        for member in row["members"]
    ]
    if len(refs) != len(set(refs)):
        raise ValueError("Task 1 lineage member appears more than once")
    return set(refs)


def _members_by_ref(rows: list[dict[str, Any]], value_key: str) -> dict[str, str]:
    return {
        f"{member['instance_id']}|{member['gold_local_id']}": row[value_key]
        for row in rows
        for member in row["members"]
    }


def _reviewed_claim(
    source: dict[str, Any], instance_id: str, local_id: str, kind: str
) -> dict[str, Any]:
    checkpoint = next(
        row for row in source["reviewed_checkpoint_rubrics"] if row["instance_id"] == instance_id
    )
    field = "state_claims" if kind == "state" else "development_claims"
    matches = [row for row in checkpoint[field] if row["local_id"] == local_id]
    if len(matches) != 1:
        raise ValueError(f"Task 1 reviewed claim reference is not unique: {instance_id}|{local_id}")
    return matches[0]


def _member_claim_at_checkpoint(
    source: dict[str, Any],
    members: list[dict[str, Any]],
    checkpoint_id: str,
    canonical_claim: str,
    kind: str,
) -> dict[str, Any]:
    instance_ids = {
        row["instance_id"]
        for row in source["reviewed_checkpoint_rubrics"]
        if row["checkpoint_id"] == checkpoint_id
    }
    matching = [member for member in members if member["instance_id"] in instance_ids]
    if matching:
        claims = [
            _reviewed_claim(source, row["instance_id"], row["gold_local_id"], kind)
            for row in matching
        ]
        return {
            "claim": claims[0]["claim"],
            "supporting_scene_orders": sorted(
                {value for claim in claims for value in claim["supporting_scene_orders"]}
            ),
        }
    all_claims = [
        _reviewed_claim(source, row["instance_id"], row["gold_local_id"], kind)
        for row in members
    ]
    return {
        "claim": canonical_claim,
        "supporting_scene_orders": sorted(
            {value for claim in all_claims for value in claim["supporting_scene_orders"]}
        ),
    }
