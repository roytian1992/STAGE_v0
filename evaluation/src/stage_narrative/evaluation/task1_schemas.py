from __future__ import annotations

import re
from typing import Any


PREDICTION_FIELDS = ("current_state", "developments_since_previous_checkpoint")
ROLLING_MEMORY_FIELDS = (*PREDICTION_FIELDS, "unresolved_threads")
ROLLING_MEMORY_LIMITS = {
    "current_state": 8,
    "developments_since_previous_checkpoint": 4,
    "unresolved_threads": 6,
}
PAIR_LABELS = {"full", "partial", "contradiction"}
SUPPORT_LABELS = {"supported", "partial", "unsupported"}
_CLUSTER_ID = re.compile(r"PD[1-9][0-9]*")
_STATE_LINEAGE_ID = re.compile(r"GS[1-9][0-9]*")
_DEVELOPMENT_LINEAGE_ID = re.compile(r"GD[1-9][0-9]*")


def validate_task1_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(payload, set(PREDICTION_FIELDS), "Task 1 prediction")
    _validate_prediction_rows(payload, fields=PREDICTION_FIELDS)
    return payload


def validate_task1_rolling_memory(payload: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(payload, set(ROLLING_MEMORY_FIELDS), "Task 1 rolling memory")
    _validate_prediction_rows(payload, fields=ROLLING_MEMORY_FIELDS)
    return payload


def _validate_prediction_rows(
    payload: dict[str, Any], *, fields: tuple[str, ...]
) -> None:
    for field in fields:
        rows = payload[field]
        if not isinstance(rows, list):
            raise ValueError(f"Task 1 prediction field must be an array: {field}")
        maximum = ROLLING_MEMORY_LIMITS[field]
        if len(rows) > maximum:
            raise ValueError(
                f"Task 1 prediction field exceeds maximum: {field}/{len(rows)}>{maximum}"
            )
        normalized_claims: set[str] = set()
        for row in rows:
            _exact_keys(row, {"claim", "evidence_scene_orders"}, f"Task 1 {field} row")
            if not isinstance(row["claim"], str) or not row["claim"].strip():
                raise ValueError(f"Task 1 claim must be nonempty: {field}")
            normalized = " ".join(row["claim"].casefold().split())
            if normalized in normalized_claims:
                raise ValueError(f"Task 1 prediction repeats a normalized claim: {field}")
            normalized_claims.add(normalized)
            scenes = row["evidence_scene_orders"]
            if (
                not isinstance(scenes, list)
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in scenes)
            ):
                raise ValueError(f"Task 1 evidence scenes must be positive integers: {field}")
            row["evidence_scene_orders"] = sorted(set(scenes))


def validate_task1_checkpoint_judgment(
    payload: dict[str, Any],
    *,
    gold_ids: set[str],
    prediction_ids: set[str],
    inactive_state_ids: set[str],
) -> dict[str, Any]:
    required = {
        "claim_pair_judgments",
        "prediction_checks",
        "future_leak_prediction_ids",
        "premature_update_prediction_ids",
        "false_persistence_pairs",
        "no_change_false_update",
    }
    _exact_keys(payload, required, "Task 1 checkpoint judgment")
    pair_keys = set()
    for row in _array(payload["claim_pair_judgments"], "claim_pair_judgments"):
        _exact_keys(row, {"gold_local_id", "prediction_local_id", "label"}, "Task 1 pair")
        if row["gold_local_id"] not in gold_ids or row["prediction_local_id"] not in prediction_ids:
            raise ValueError("Task 1 pair contains an unknown local ID")
        if row["label"] not in PAIR_LABELS:
            raise ValueError("Task 1 pair label is invalid")
        key = (row["gold_local_id"], row["prediction_local_id"])
        if key in pair_keys:
            raise ValueError("Task 1 pair judgment contains a duplicate pair")
        pair_keys.add(key)
    checks = {}
    for row in _array(payload["prediction_checks"], "prediction_checks"):
        _exact_keys(
            row,
            {
                "prediction_local_id",
                "support",
                "transition_coherent",
                "evidence_grounded",
                "checkpoint_valid",
                "salient",
            },
            "Task 1 prediction check",
        )
        local_id = row["prediction_local_id"]
        if local_id not in prediction_ids or local_id in checks:
            raise ValueError("Task 1 prediction checks must cover unique known IDs")
        if row["support"] not in SUPPORT_LABELS:
            raise ValueError("Task 1 support label is invalid")
        _booleans(
            row,
            ("transition_coherent", "evidence_grounded", "checkpoint_valid", "salient"),
            "Task 1 prediction check",
        )
        checks[local_id] = row
    if set(checks) != prediction_ids:
        raise ValueError("Task 1 prediction checks require exact ID coverage")
    for field in ("future_leak_prediction_ids", "premature_update_prediction_ids"):
        values = _array(payload[field], field)
        if not set(values) <= prediction_ids:
            raise ValueError(f"Task 1 {field} contains an unknown prediction ID")
        payload[field] = sorted(set(values))
    false_pairs = set()
    for row in _array(payload["false_persistence_pairs"], "false_persistence_pairs"):
        _exact_keys(
            row,
            {"inactive_state_local_id", "prediction_local_id"},
            "Task 1 false-persistence pair",
        )
        if (
            row["inactive_state_local_id"] not in inactive_state_ids
            or row["prediction_local_id"] not in prediction_ids
        ):
            raise ValueError("Task 1 false-persistence pair contains an unknown ID")
        key = (row["inactive_state_local_id"], row["prediction_local_id"])
        if key in false_pairs:
            raise ValueError("Task 1 false-persistence pair is duplicated")
        false_pairs.add(key)
    if not isinstance(payload["no_change_false_update"], bool):
        raise ValueError("Task 1 no-change flag must be boolean")
    return payload


def validate_task1_alignment_judgment(
    payload: dict[str, Any], *, gold_ids: set[str], prediction_ids: set[str],
    inactive_state_ids: set[str]
) -> dict[str, Any]:
    required = {"claim_pair_judgments", "false_persistence_pairs", "no_change_false_update"}
    _exact_keys(payload, required, "Task 1 alignment judgment")
    combined = {
        **payload,
        "prediction_checks": [
            {
                "prediction_local_id": value,
                "support": "unsupported",
                "transition_coherent": True,
                "evidence_grounded": False,
                "checkpoint_valid": False,
                "salient": False,
            }
            for value in sorted(prediction_ids)
        ],
        "future_leak_prediction_ids": [],
        "premature_update_prediction_ids": [],
    }
    validate_task1_checkpoint_judgment(
        combined,
        gold_ids=gold_ids,
        prediction_ids=prediction_ids,
        inactive_state_ids=inactive_state_ids,
    )
    return payload


def validate_task1_evidence_judgment(
    payload: dict[str, Any], *, prediction_ids: set[str]
) -> dict[str, Any]:
    required = {
        "prediction_checks", "future_leak_prediction_ids", "premature_update_prediction_ids"
    }
    _exact_keys(payload, required, "Task 1 evidence judgment")
    combined = {
        "claim_pair_judgments": [],
        **payload,
        "false_persistence_pairs": [],
        "no_change_false_update": False,
    }
    validate_task1_checkpoint_judgment(
        combined,
        gold_ids=set(),
        prediction_ids=prediction_ids,
        inactive_state_ids=set(),
    )
    return payload


def validate_task1_trajectory_judgment(
    payload: dict[str, Any],
    *,
    development_prediction_refs: set[str],
    adjacent_instance_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    _exact_keys(payload, {"development_clusters", "adjacent_checks"}, "Task 1 trajectory judgment")
    observed_refs = set()
    cluster_ids = set()
    for row in _array(payload["development_clusters"], "development_clusters"):
        _exact_keys(row, {"cluster_id", "members", "brief_rationale"}, "Task 1 development cluster")
        cluster_id = row["cluster_id"]
        if not isinstance(cluster_id, str) or not _CLUSTER_ID.fullmatch(cluster_id):
            raise ValueError("Task 1 cluster ID must match PD[1-9][0-9]*")
        if cluster_id in cluster_ids:
            raise ValueError("Task 1 cluster IDs must be unique")
        cluster_ids.add(cluster_id)
        members = _array(row["members"], "development cluster members")
        if not members or not all(isinstance(value, str) for value in members):
            raise ValueError("Task 1 development cluster must have string members")
        if observed_refs & set(members):
            raise ValueError("Task 1 development reference appears in multiple clusters")
        observed_refs.update(members)
        if not isinstance(row["brief_rationale"], str):
            raise ValueError("Task 1 cluster rationale must be text")
    if observed_refs != development_prediction_refs:
        raise ValueError("Task 1 development clusters require exact prediction coverage")
    expected_pairs = set(adjacent_instance_pairs)
    observed_pairs = set()
    for row in _array(payload["adjacent_checks"], "adjacent_checks"):
        required = {
            "earlier_instance_id",
            "later_instance_id",
            "state_carry_forward",
            "development_to_state_coherent",
            "contradiction_present",
            "premature_or_future_information",
            "brief_rationale",
        }
        _exact_keys(row, required, "Task 1 adjacent check")
        pair = (row["earlier_instance_id"], row["later_instance_id"])
        if pair not in expected_pairs or pair in observed_pairs:
            raise ValueError("Task 1 adjacent check coverage is invalid")
        observed_pairs.add(pair)
        _booleans(
            row,
            (
                "state_carry_forward",
                "development_to_state_coherent",
                "contradiction_present",
                "premature_or_future_information",
            ),
            "Task 1 adjacent check",
        )
        if not isinstance(row["brief_rationale"], str):
            raise ValueError("Task 1 adjacent rationale must be text")
    if observed_pairs != expected_pairs:
        raise ValueError("Task 1 adjacent checks require exact pair coverage")
    return payload


def validate_task1_development_cluster_judgment(
    payload: dict[str, Any], *, development_prediction_refs: set[str]
) -> dict[str, Any]:
    _exact_keys(payload, {"development_clusters"}, "Task 1 cluster judgment")
    validate_task1_trajectory_judgment(
        {"development_clusters": payload["development_clusters"], "adjacent_checks": []},
        development_prediction_refs=development_prediction_refs,
        adjacent_instance_pairs=[],
    )
    return payload


def validate_task1_adjacent_judgment(
    payload: dict[str, Any], *, earlier_instance_id: str, later_instance_id: str
) -> dict[str, Any]:
    validate_task1_trajectory_judgment(
        {"development_clusters": [], "adjacent_checks": [payload]},
        development_prediction_refs=set(),
        adjacent_instance_pairs=[(earlier_instance_id, later_instance_id)],
    )
    return payload


def validate_task1_private_assets(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "movie_id",
        "language",
        "public_instances_sha256",
        "trajectory_count",
        "checkpoint_count",
        "trajectories",
    }
    _exact_keys(payload, required, "Task 1 private assets")
    if payload["schema_version"] != "stage_task1_private_evaluator":
        raise ValueError("Unknown Task 1 private asset schema")
    trajectories = _array(payload["trajectories"], "trajectories")
    if payload["trajectory_count"] != len(trajectories):
        raise ValueError("Task 1 trajectory count drift")
    if payload["checkpoint_count"] != sum(len(row["checkpoint_ids"]) for row in trajectories):
        raise ValueError("Task 1 checkpoint count drift")
    seen_instances = set()
    for trajectory in trajectories:
        _validate_private_trajectory(trajectory, seen_instances)
    return payload


def validate_task1_state_lineage_review(
    payload: dict[str, Any], *, source: dict[str, Any]
) -> dict[str, Any]:
    _exact_keys(payload, {"states"}, "Task 1 state lineage review")
    checkpoints = list(source["checkpoint_ids"])
    checkpoint_index = {value: index for index, value in enumerate(checkpoints)}
    allowed_source_ids = {row["state_id"] for row in source["source_states"]}
    keys = set()
    refs = []
    for row in _array(payload["states"], "state lineages"):
        _exact_keys(
            row,
            {
                "lineage_key",
                "canonical_claim",
                "members",
                "source_state_ids",
                "valid_from_checkpoint_id",
                "valid_to_checkpoint_id",
                "superseded_by_lineage_key",
            },
            "Task 1 state lineage",
        )
        key = row["lineage_key"]
        if not isinstance(key, str) or not _STATE_LINEAGE_ID.fullmatch(key) or key in keys:
            raise ValueError("Task 1 state lineage keys must be unique GS IDs")
        keys.add(key)
        _nonempty_text(row["canonical_claim"], "Task 1 state canonical claim")
        member_instance_ids = set()
        for member in _array(row["members"], "state lineage members"):
            _exact_keys(member, {"instance_id", "gold_local_id"}, "Task 1 state member")
            if member["instance_id"] in member_instance_ids:
                raise ValueError(
                    "Task 1 state lineage cannot merge multiple claims from one checkpoint"
                )
            member_instance_ids.add(member["instance_id"])
            refs.append(f"{member['instance_id']}|{member['gold_local_id']}")
        if not row["members"]:
            raise ValueError("Task 1 state lineage cannot be empty")
        if not set(_array(row["source_state_ids"], "source state IDs")) <= allowed_source_ids:
            raise ValueError("Task 1 state lineage references an unknown source state")
        start = row["valid_from_checkpoint_id"]
        end = row["valid_to_checkpoint_id"]
        if start not in checkpoint_index or (end is not None and end not in checkpoint_index):
            raise ValueError("Task 1 state lineage interval references an unknown checkpoint")
        if end is not None and checkpoint_index[end] < checkpoint_index[start]:
            raise ValueError("Task 1 state lineage interval is reversed")
    if len(refs) != len(set(refs)) or set(refs) != set(source["required_state_member_refs"]):
        raise ValueError("Task 1 state lineages require exact reviewed-claim coverage")
    for row in payload["states"]:
        successor = row["superseded_by_lineage_key"]
        if successor is not None and (successor not in keys or successor == row["lineage_key"]):
            raise ValueError("Task 1 state supersession references an invalid lineage")
    return payload


def validate_task1_development_lineage_review(
    payload: dict[str, Any], *, source: dict[str, Any], states: list[dict[str, Any]]
) -> dict[str, Any]:
    _exact_keys(payload, {"developments"}, "Task 1 development lineage review")
    checkpoints = list(source["checkpoint_ids"])
    checkpoint_index = {value: index for index, value in enumerate(checkpoints)}
    state_keys = {row["lineage_key"] for row in states}
    allowed_source_ids = {
        row["development_id"] for row in source["source_developments"]
    }
    keys = set()
    refs = []
    for row in _array(payload["developments"], "development lineages"):
        _exact_keys(
            row,
            {
                "lineage_key",
                "canonical_claim",
                "members",
                "source_development_ids",
                "trigger_checkpoint_id",
                "effective_checkpoint_id",
                "affected_state_keys",
                "resulting_state_keys",
                "superseded_state_keys",
            },
            "Task 1 development lineage",
        )
        key = row["lineage_key"]
        if (
            not isinstance(key, str)
            or not _DEVELOPMENT_LINEAGE_ID.fullmatch(key)
            or key in keys
        ):
            raise ValueError("Task 1 development lineage keys must be unique GD IDs")
        keys.add(key)
        _nonempty_text(row["canonical_claim"], "Task 1 development canonical claim")
        for member in _array(row["members"], "development lineage members"):
            _exact_keys(
                member,
                {"instance_id", "gold_local_id"},
                "Task 1 development member",
            )
            refs.append(f"{member['instance_id']}|{member['gold_local_id']}")
        if not row["members"]:
            raise ValueError("Task 1 development lineage cannot be empty")
        if not set(
            _array(row["source_development_ids"], "source development IDs")
        ) <= allowed_source_ids:
            raise ValueError("Task 1 development references an unknown source development")
        trigger = row["trigger_checkpoint_id"]
        effective = row["effective_checkpoint_id"]
        if trigger not in checkpoint_index or effective not in checkpoint_index:
            raise ValueError("Task 1 development references an unknown checkpoint")
        if checkpoint_index[trigger] > checkpoint_index[effective]:
            raise ValueError("Task 1 development trigger follows its effective checkpoint")
        for field in ("affected_state_keys", "resulting_state_keys", "superseded_state_keys"):
            values = _array(row[field], field)
            if len(values) != len(set(values)) or not set(values) <= state_keys:
                raise ValueError(f"Task 1 development {field} has invalid state keys")
    if len(refs) != len(set(refs)) or set(refs) != set(
        source["required_development_member_refs"]
    ):
        raise ValueError("Task 1 development lineages require exact reviewed-claim coverage")
    return payload


def _validate_private_trajectory(trajectory: dict[str, Any], seen_instances: set[str]) -> None:
    required = {
        "character_id",
        "character",
        "checkpoint_ids",
        "states",
        "developments",
        "checkpoint_rubrics",
    }
    _exact_keys(trajectory, required, "Task 1 private trajectory")
    checkpoint_ids = trajectory["checkpoint_ids"]
    if not isinstance(checkpoint_ids, list) or len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("Task 1 checkpoint IDs must be a unique array")
    checkpoint_index = {value: index for index, value in enumerate(checkpoint_ids)}
    states = {row["stable_state_id"]: row for row in _array(trajectory["states"], "states")}
    if len(states) != len(trajectory["states"]):
        raise ValueError("Task 1 stable state IDs must be unique")
    for row in states.values():
        required_state = {
            "stable_state_id",
            "claim",
            "source_state_ids",
            "valid_from_checkpoint_id",
            "valid_to_checkpoint_id",
            "superseded_by_state_id",
            "supporting_scene_orders",
        }
        _exact_keys(row, required_state, "Task 1 stable state")
        start = row["valid_from_checkpoint_id"]
        end = row["valid_to_checkpoint_id"]
        if start not in checkpoint_index or (end is not None and end not in checkpoint_index):
            raise ValueError("Task 1 state interval references an unknown checkpoint")
        if end is not None and checkpoint_index[end] < checkpoint_index[start]:
            raise ValueError("Task 1 state interval is reversed")
        successor = row["superseded_by_state_id"]
        if successor is not None and successor not in states:
            raise ValueError("Task 1 state supersession references an unknown state")
    developments = {
        row["stable_development_id"]: row
        for row in _array(trajectory["developments"], "developments")
    }
    if len(developments) != len(trajectory["developments"]):
        raise ValueError("Task 1 stable development IDs must be unique")
    for row in developments.values():
        required_development = {
            "stable_development_id",
            "claim",
            "source_development_ids",
            "trigger_checkpoint_id",
            "effective_checkpoint_id",
            "affected_state_ids",
            "resulting_state_ids",
            "superseded_state_ids",
            "supporting_scene_orders",
        }
        _exact_keys(row, required_development, "Task 1 stable development")
        if (
            row["trigger_checkpoint_id"] not in checkpoint_index
            or row["effective_checkpoint_id"] not in checkpoint_index
        ):
            raise ValueError("Task 1 development references an unknown checkpoint")
        if checkpoint_index[row["trigger_checkpoint_id"]] > checkpoint_index[row["effective_checkpoint_id"]]:
            raise ValueError("Task 1 development trigger occurs after its effective checkpoint")
        for field in ("affected_state_ids", "resulting_state_ids", "superseded_state_ids"):
            if not set(row[field]) <= set(states):
                raise ValueError(f"Task 1 development {field} references an unknown state")
    rubric_by_checkpoint = {}
    for rubric in _array(trajectory["checkpoint_rubrics"], "checkpoint_rubrics"):
        required_rubric = {
            "instance_id",
            "checkpoint_id",
            "checkpoint",
            "current_state_claims",
            "development_claims",
            "inactive_state_claims",
            "salient_future_negatives",
        }
        _exact_keys(rubric, required_rubric, "Task 1 checkpoint rubric")
        if rubric["instance_id"] in seen_instances:
            raise ValueError("Task 1 instance ID is duplicated")
        seen_instances.add(rubric["instance_id"])
        checkpoint_id = rubric["checkpoint_id"]
        if checkpoint_id not in checkpoint_index or checkpoint_id in rubric_by_checkpoint:
            raise ValueError("Task 1 rubric checkpoint coverage is invalid")
        rubric_by_checkpoint[checkpoint_id] = rubric
        _validate_local_claims(rubric["current_state_claims"], "S", "stable_state_id", states)
        _validate_local_claims(
            rubric["development_claims"], "D", "stable_development_id", developments
        )
        _validate_local_claims(rubric["inactive_state_claims"], "X", "stable_state_id", states)
    if set(rubric_by_checkpoint) != set(checkpoint_ids):
        raise ValueError("Task 1 private rubrics require exact checkpoint coverage")
    for rubric in rubric_by_checkpoint.values():
        current = rubric["current_state_claims"]
        developments_at_checkpoint = rubric["development_claims"]
        if len(current) > 8 or len(developments_at_checkpoint) > 4:
            raise ValueError("Task 1 checkpoint rubric exceeds compactness limits")
        for rows, stable_key, label in (
            (current, "stable_state_id", "current state"),
            (developments_at_checkpoint, "stable_development_id", "development"),
        ):
            stable_ids = [row[stable_key] for row in rows]
            normalized_claims = [" ".join(row["claim"].casefold().split()) for row in rows]
            if len(stable_ids) != len(set(stable_ids)):
                raise ValueError(f"Task 1 checkpoint repeats a stable {label} ID")
            if len(normalized_claims) != len(set(normalized_claims)):
                raise ValueError(f"Task 1 checkpoint repeats a normalized {label} claim")


def _validate_local_claims(
    rows: Any, prefix: str, stable_key: str, stable: dict[str, Any]
) -> None:
    observed = set()
    for row in _array(rows, f"{prefix} claims"):
        _exact_keys(
            row,
            {"local_id", stable_key, "claim", "supporting_scene_orders"},
            f"Task 1 {prefix} claim",
        )
        if row["local_id"] in observed or not re.fullmatch(f"{prefix}[1-9][0-9]*", row["local_id"]):
            raise ValueError(f"Task 1 {prefix} local IDs must be unique and contiguous-style")
        observed.add(row["local_id"])
        if row[stable_key] not in stable:
            raise ValueError(f"Task 1 {prefix} claim references an unknown stable ID")


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Task 1 field must be an array: {label}")
    return value


def _booleans(payload: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    if any(not isinstance(payload[field], bool) for field in fields):
        raise ValueError(f"{label} flags must be booleans")


def _exact_keys(payload: Any, expected: set[str], label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        observed = set(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"{label} keys mismatch: {observed} != {expected}")


def _nonempty_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
