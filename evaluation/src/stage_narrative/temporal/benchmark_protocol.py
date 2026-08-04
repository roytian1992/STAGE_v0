from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..chunking import TokenCounter, build_token_counter, chunk_scene
from ..io import load_config, load_scenes, sha256_file, sha256_json
from ..models import Scene, clean_text, normalize_name, stable_id
from ..prompt_loader import PROMPTS, YamlPromptRegistry
from .runtime import materialize_task3_actor_input


STANDARD_CALL_NAMES = {
    "task1_prediction",
    "task1_judge",
    "task3_actor",
    "task3_response_judge",
    "task3_pair_judge",
}
OPTIONAL_CALL_NAMES = {"asset_review"}


def restrict_tasks_to_release_roles(
    *,
    task1: dict[str, Any],
    task3: dict[str, Any],
    registry: dict[str, Any],
    role_assets: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Restrict benchmark-facing tasks to uniquely matched legacy release roles."""
    movie_ids = {
        clean_text(payload.get("movie_id"))
        for payload in (task1, task3, registry, role_assets)
    }
    if len(movie_ids) != 1 or not next(iter(movie_ids)):
        raise ValueError("Release-role inputs belong to different or missing movies")
    characters = registry.get("characters")
    roles = role_assets.get("roles")
    if not isinstance(characters, list) or not isinstance(roles, list) or not roles:
        raise ValueError("Release-role alignment requires characters and non-empty roles")

    character_surfaces: dict[str, set[str]] = {}
    for character in characters:
        character_id = clean_text(character.get("character_id"))
        surfaces = {
            normalize_name(value)
            for value in [
                character.get("canonical_name"),
                *(character.get("aliases") or []),
            ]
            if clean_text(value)
        }
        if character_id and surfaces:
            character_surfaces[character_id] = surfaces

    mappings = []
    selected_ids: set[str] = set()
    for role in roles:
        role_name = clean_text(role.get("character_name"))
        role_surfaces = {
            normalize_name(value)
            for value in [role_name, *(role.get("aliases") or [])]
            if clean_text(value)
        }
        matches = [
            character_id
            for character_id, surfaces in character_surfaces.items()
            if surfaces & role_surfaces
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Release role must match exactly one canonical character: "
                f"{role_name!r} -> {matches}"
            )
        character_id = matches[0]
        if character_id in selected_ids:
            raise ValueError(
                f"Multiple release roles resolve to one character: {character_id}"
            )
        selected_ids.add(character_id)
        character = next(
            item for item in characters if item.get("character_id") == character_id
        )
        mappings.append(
            {
                "release_role": role_name,
                "character_id": character_id,
                "canonical_name": character.get("canonical_name"),
                "matched_surfaces": sorted(
                    role_surfaces & character_surfaces[character_id]
                ),
            }
        )

    def filtered(payload: dict[str, Any]) -> dict[str, Any]:
        output = deepcopy(payload)
        original = output.get("instances")
        if not isinstance(original, list):
            raise ValueError("Task payload instances must be an array")
        output["instances"] = [
            item for item in original if item.get("character_id") in selected_ids
        ]
        output["instance_count"] = len(output["instances"])
        output["character_count"] = len(
            {item.get("character_id") for item in output["instances"]}
        )
        if "pair_group_count" in output:
            output["pair_group_count"] = len(
                {
                    clean_text(
                        item.get("evaluator_reference", {}).get(
                            "paired_prompt_group_id"
                        )
                    )
                    for item in output["instances"]
                    if clean_text(
                        item.get("evaluator_reference", {}).get(
                            "paired_prompt_group_id"
                        )
                    )
                }
            )
        return output

    filtered_task1 = filtered(task1)
    filtered_task3 = filtered(task3)
    if not filtered_task1["instances"] or not filtered_task3["instances"]:
        raise ValueError("Release-role alignment removed every Task 1 or Task 3 instance")
    audit = {
        "schema_version": "stage_release_role_alignment_v1",
        "movie_id": next(iter(movie_ids)),
        "release_role_count": len(roles),
        "matched_character_count": len(selected_ids),
        "mappings": mappings,
        "task1_input_count": len(task1["instances"]),
        "task1_output_count": len(filtered_task1["instances"]),
        "task3_input_count": len(task3["instances"]),
        "task3_output_count": len(filtered_task3["instances"]),
        "filtered_character_ids": sorted(
            {
                item.get("character_id")
                for item in [*task1["instances"], *task3["instances"]]
                if item.get("character_id") not in selected_ids
            }
        ),
    }
    return filtered_task1, filtered_task3, audit


@dataclass(frozen=True, slots=True)
class CallBudget:
    context_window: int
    max_output_tokens: int
    safety_margin_tokens: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, label: str) -> "CallBudget":
        budget = cls(
            context_window=int(payload.get("context_window", 0)),
            max_output_tokens=int(payload.get("max_output_tokens", 0)),
            safety_margin_tokens=int(payload.get("safety_margin_tokens", 0)),
        )
        if min(
            budget.context_window,
            budget.max_output_tokens,
            budget.safety_margin_tokens,
        ) <= 0:
            raise ValueError(f"Benchmark call budget must be positive: {label}")
        if budget.max_output_tokens + budget.safety_margin_tokens >= budget.context_window:
            raise ValueError(f"Benchmark call budget leaves no input space: {label}")
        return budget

    @property
    def max_input_tokens(self) -> int:
        return self.context_window - self.max_output_tokens - self.safety_margin_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "max_input_tokens": self.max_input_tokens,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRuntimeConfig:
    source_path: Path
    source_sha256: str
    schema_version: str
    track_id: str
    prediction_llm: dict[str, Any]
    evaluation_llm: dict[str, Any]
    tokenizer: dict[str, Any]
    call_budgets: dict[str, CallBudget]
    reserved_chat_template_tokens: int
    task1: dict[str, Any]
    task3: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "BenchmarkRuntimeConfig":
        resolved = path.resolve()
        payload = load_config(resolved)
        shared = {
            "schema_version",
            "track_id",
            "tokenizer",
            "call_budgets",
            "token_accounting",
            "task1",
            "task3",
        }
        formal = shared | {"prediction_llm", "evaluation_llm"}
        legacy = shared | {"llm"}
        if set(payload) == formal:
            prediction_llm = dict(payload["prediction_llm"])
            evaluation_llm = dict(payload["evaluation_llm"])
        elif set(payload) == legacy:
            # Historical configs remain readable, but every new run should use the
            # explicit role-specific fields above.
            prediction_llm = dict(payload["llm"])
            evaluation_llm = dict(payload["llm"])
        else:
            raise ValueError(
                "Benchmark runtime config keys mismatch: "
                f"formal_missing={sorted(formal - set(payload))} "
                f"legacy_missing={sorted(legacy - set(payload))} "
                f"extra={sorted(set(payload) - formal - legacy)}"
            )
        raw_budgets = payload["call_budgets"]
        if (
            not isinstance(raw_budgets, dict)
            or not STANDARD_CALL_NAMES <= set(raw_budgets)
            or set(raw_budgets) - STANDARD_CALL_NAMES - OPTIONAL_CALL_NAMES
        ):
            raise ValueError(
                "Benchmark runtime config requires the five frozen call budgets "
                "and permits only the optional asset_review budget"
            )
        budgets = {
            name: CallBudget.from_dict(raw_budgets[name], label=name)
            for name in sorted(raw_budgets)
        }
        windows = {budgets[name].context_window for name in STANDARD_CALL_NAMES}
        if len(windows) != 1:
            raise ValueError("standard-24k requires one shared context window")
        accounting = payload["token_accounting"]
        if accounting.get("method") != (
            "raw_system_plus_user_plus_reserved_chat_overhead"
        ):
            raise ValueError("Unsupported benchmark token accounting method")
        reserved = int(accounting.get("reserved_chat_template_tokens", 0))
        if reserved < 0:
            raise ValueError("reserved_chat_template_tokens must be non-negative")
        task1 = dict(payload["task1"])
        if int(task1.get("max_scene_part_content_tokens", 0)) <= 0:
            raise ValueError("Task 1 scene-part budget must be positive")
        if int(task1.get("reserved_prior_memory_tokens", 0)) <= 0:
            raise ValueError("Task 1 prior-memory reserve must be positive")
        evidence_batch_limit = int(
            task1.get(
                "max_evidence_batch_input_tokens",
                budgets["task1_judge"].max_input_tokens,
            )
        )
        if not 0 < evidence_batch_limit <= budgets["task1_judge"].max_input_tokens:
            raise ValueError(
                "Task 1 evidence batch limit must fit the Task 1 judge input budget"
            )
        return cls(
            source_path=resolved,
            source_sha256=sha256_file(resolved),
            schema_version=str(payload["schema_version"]),
            track_id=str(payload["track_id"]),
            prediction_llm=prediction_llm,
            evaluation_llm=evaluation_llm,
            tokenizer=dict(payload["tokenizer"]),
            call_budgets=budgets,
            reserved_chat_template_tokens=reserved,
            task1=task1,
            task3=dict(payload["task3"]),
        )

    def build_token_counter(self) -> TokenCounter:
        return build_token_counter(self.tokenizer)

    @property
    def llm(self) -> dict[str, Any]:
        """Compatibility view for non-evaluation construction utilities."""
        return self.prediction_llm

    def prompt_path(self, task: str, language: str) -> str:
        language_key = normalize_language(language)
        key_by_task = {
            "task1_prediction": (self.task1, "prompt_paths"),
            "task1_judge": (self.task1, "judge_prompt_paths"),
            "task3_actor": (self.task3, "actor_prompt_paths"),
            "task3_response_judge": (self.task3, "judge_prompt_paths"),
            "task3_pair_judge": (self.task3, "pair_judge_prompt_paths"),
            "task1_rubric_construction": (
                self.task1,
                "rubric_construction_prompt_paths",
            ),
            "task3_rubric_construction": (
                self.task3,
                "rubric_construction_prompt_paths",
            ),
        }
        try:
            section, key = key_by_task[task]
            return str(section[key][language_key])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing prompt path for {task}/{language_key}") from exc

    def public_snapshot(self) -> dict[str, Any]:
        def redacted(value: dict[str, Any]) -> dict[str, Any]:
            output = dict(value)
            if "api_key" in output:
                output["api_key"] = "<configured-locally>"
            return output

        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "prediction_llm": redacted(self.prediction_llm),
            "evaluation_llm": redacted(self.evaluation_llm),
            "tokenizer": self.tokenizer,
            "call_budgets": {
                name: budget.as_dict() for name, budget in self.call_budgets.items()
            },
            "token_accounting": {
                "method": "raw_system_plus_user_plus_reserved_chat_overhead",
                "reserved_chat_template_tokens": self.reserved_chat_template_tokens,
            },
            "task1": self.task1,
            "task3": self.task3,
        }


def normalize_language(value: Any) -> str:
    normalized = clean_text(value).casefold()
    if normalized in {"zh", "chinese", "中文"}:
        return "zh"
    if normalized in {"en", "english", "英文"}:
        return "en"
    raise ValueError(f"Unsupported benchmark language: {value!r}")


def apply_task3_checkpoint_anchors(
    *,
    task3: dict[str, Any],
    anchor_manifest: dict[str, Any],
    evidence_bank: dict[str, Any],
    script_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach reviewed within-scene cutoffs without changing the temporal source run."""
    if anchor_manifest.get("schema_version") != "stage_task3_checkpoint_anchors_v1":
        raise ValueError("Unsupported Task 3 checkpoint-anchor schema")
    if anchor_manifest.get("movie_id") != task3.get("movie_id"):
        raise ValueError("Task 3 checkpoint anchors belong to another movie")
    raw_anchors = anchor_manifest.get("anchors")
    if not isinstance(raw_anchors, list):
        raise ValueError("Task 3 checkpoint anchors must be a list")
    anchors = {item.get("prompt_id"): item for item in raw_anchors}
    if len(anchors) != len(raw_anchors) or None in anchors:
        raise ValueError("Task 3 checkpoint anchors contain duplicate or missing prompt IDs")

    instances = task3.get("instances", [])
    prompt_ids = {
        item["evaluator_reference"]["prompt_id"] for item in instances
    }
    if set(anchors) != prompt_ids:
        raise ValueError(
            "Task 3 checkpoint-anchor coverage mismatch: "
            f"missing={sorted(prompt_ids - set(anchors))} "
            f"extra={sorted(set(anchors) - prompt_ids)}"
        )
    scenes = {scene.order: scene for scene in load_scenes(script_path.resolve())}
    evidence = {
        item["evidence_id"]: item for item in evidence_bank.get("evidence_units", [])
    }
    anchored = deepcopy(task3)
    records = []
    for instance in anchored["instances"]:
        prompt_id = instance["evaluator_reference"]["prompt_id"]
        raw = anchors[prompt_id]
        scene_order = int(raw.get("scene_order", 0))
        char_end = int(raw.get("char_end", -1))
        scene = scenes.get(scene_order)
        if scene is None or char_end < 0 or char_end > len(scene.content):
            raise ValueError(f"Invalid Task 3 checkpoint anchor: {prompt_id}")
        anchor_evidence_ids = list(raw.get("anchor_evidence_ids", []))
        if not anchor_evidence_ids:
            raise ValueError(f"Task 3 anchor lacks source evidence: {prompt_id}")
        for evidence_id in anchor_evidence_ids:
            item = evidence.get(evidence_id)
            if item is None:
                raise ValueError(f"Task 3 anchor references unknown evidence: {evidence_id}")
            if int(item["scene_order"]) != scene_order or int(item["char_end"]) > char_end:
                raise ValueError(f"Task 3 anchor evidence lies after its cutoff: {prompt_id}")
        future_negative_evidence_ids = list(
            raw.get("future_negative_evidence_ids", [])
        )
        for evidence_id in future_negative_evidence_ids:
            item = evidence.get(evidence_id)
            if item is None:
                raise ValueError(
                    f"Task 3 future negative references unknown evidence: {evidence_id}"
                )
            item_scene = int(item["scene_order"])
            if item_scene < scene_order or (
                item_scene == scene_order and int(item["char_start"]) < char_end
            ):
                raise ValueError(
                    f"Task 3 future negative does not lie after its cutoff: {prompt_id}"
                )
        overrides = raw.get("model_input_overrides", {})
        if not isinstance(overrides, dict) or not set(overrides) <= {
            "interaction_context",
            "current_user_turn",
        }:
            raise ValueError(f"Unsupported Task 3 model-input override: {prompt_id}")
        instance["model_input"].update(overrides)
        checkpoint_anchor = {
            "scene_order": scene_order,
            "char_end": char_end,
            "anchor_evidence_ids": anchor_evidence_ids,
            "future_negative_evidence_ids": future_negative_evidence_ids,
            "boundary_policy": "source_visible_through_char_end_before_answer",
            "review_status": anchor_manifest.get("status", ""),
        }
        instance["model_input"]["checkpoint_anchor"] = checkpoint_anchor
        instance["evaluator_reference"]["checkpoint_anchor"] = checkpoint_anchor
        records.append(
            {
                "instance_id": instance["instance_id"],
                "prompt_id": prompt_id,
                **checkpoint_anchor,
                "model_input_overrides": overrides,
                "review_note": raw.get("review_note", ""),
            }
        )
    snapshot = {
        "schema_version": "stage_task3_applied_checkpoint_anchors_v1",
        "movie_id": task3.get("movie_id"),
        "status": anchor_manifest.get("status", ""),
        "anchor_count": len(records),
        "anchors": records,
    }
    return anchored, snapshot


def build_screenplay_token_manifest(
    *,
    movie_id: str,
    script_path: Path,
    token_counter: TokenCounter,
    config: BenchmarkRuntimeConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = script_path.resolve()
    scenes = load_scenes(resolved)
    max_part_tokens = int(config.task1["max_scene_part_content_tokens"])
    scene_records: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for scene in scenes:
        chunks = chunk_scene(
            movie_id=movie_id,
            scene=scene,
            token_counter=token_counter,
            max_content_tokens=max_part_tokens,
        )
        unit_records = []
        for chunk in chunks:
            prompt_text = chunk.prompt_text(scene)
            record = {
                "unit_id": stable_id(
                    "task1-scene-unit",
                    movie_id,
                    scene.scene_id,
                    chunk.order,
                    chunk.char_start,
                    chunk.char_end,
                ),
                "scene_id": scene.scene_id,
                "source_scene_id": scene.source_scene_id,
                "scene_order": scene.order,
                "part_order": chunk.order,
                "part_count": len(chunks),
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "content_token_count": chunk.token_count,
                "prompt_token_count": token_counter.count(prompt_text),
                "prompt_text_sha256": sha256_json(prompt_text),
            }
            units.append(record)
            unit_records.append(record)
        scene_prompt = scene.prompt_text()
        scene_records.append(
            {
                "scene_id": scene.scene_id,
                "source_scene_id": scene.source_scene_id,
                "scene_order": scene.order,
                "content_char_count": len(scene.content),
                "content_token_count": token_counter.count(scene.content),
                "prompt_token_count": token_counter.count(scene_prompt),
                "prompt_text_sha256": sha256_json(scene_prompt),
                "unit_count": len(unit_records),
                "unit_ids": [item["unit_id"] for item in unit_records],
            }
        )
    manifest = {
        "schema_version": "stage_screenplay_token_manifest_v1",
        "track_id": config.track_id,
        "movie_id": movie_id,
        "script_path": str(resolved),
        "script_sha256": sha256_file(resolved),
        "scene_count": len(scenes),
        "scene_unit_count": len(units),
        "tokenizer": token_counter.metadata,
        "max_scene_part_content_tokens": max_part_tokens,
        "coverage_policy": "scene_order_exact_once_with_lossless_scene_parts",
        "scenes": scene_records,
        "scene_units": units,
    }
    return manifest, units


def materialize_task1_rolling_prompt(
    *,
    character_plan: dict[str, Any],
    call: dict[str, Any],
    prior_memory: dict[str, Any],
    script_path: Path,
    token_manifest: dict[str, Any],
    language: str,
    token_counter: TokenCounter,
    config: BenchmarkRuntimeConfig,
    prompt_registry: YamlPromptRegistry = PROMPTS,
) -> dict[str, Any]:
    validate_task1_memory(
        prior_memory,
        maximum_evidence_scene_order=max(
            0, int(call["block_start_scene_order"])
        ),
    )
    resolved_script = script_path.resolve()
    if sha256_file(resolved_script) != token_manifest.get("script_sha256"):
        raise ValueError("Task 1 runtime script differs from its token manifest")
    scenes = {scene.scene_id: scene for scene in load_scenes(resolved_script)}
    units = {
        item["unit_id"]: item for item in token_manifest.get("scene_units", [])
    }
    refs = list(call.get("scene_unit_refs", []))
    if not refs or any(item not in units for item in refs):
        raise ValueError("Task 1 rolling call contains unknown or empty scene-unit refs")
    block_payload = [
        _resolve_scene_unit(units[item], scenes[units[item]["scene_id"]])
        for item in refs
    ]
    prompt_path = config.prompt_path("task1_prediction", language)
    system, user = prompt_registry.render(
        prompt_path,
        focal_character=character_plan["focal_character"],
        aliases=character_plan.get("aliases", []),
        previous_checkpoint_scene_order=call["previous_checkpoint_scene_order"],
        block_end_scene_order=call["block_end_scene_order"],
        checkpoint_at_block_end=call["checkpoint_at_block_end"],
        prior_memory=prior_memory,
        screenplay_block=block_payload,
    )
    raw_tokens = token_counter.count(system) + token_counter.count(user)
    accounted = raw_tokens + config.reserved_chat_template_tokens
    maximum = config.call_budgets["task1_prediction"].max_input_tokens
    if accounted > maximum:
        raise ValueError(
            f"Task 1 rolling prompt exceeds input budget: {accounted}>{maximum}"
        )
    return {
        "call_id": call["call_id"],
        "system_prompt": system,
        "user_prompt": user,
        "raw_prompt_tokens": raw_tokens,
        "accounted_input_tokens": accounted,
        "max_input_tokens": maximum,
        "checkpoint_at_block_end": bool(call["checkpoint_at_block_end"]),
        "checkpoint_id": call.get("checkpoint_id", ""),
        "task_instance_id": call.get("task_instance_id", ""),
    }


def validate_task1_memory(
    payload: dict[str, Any], *, maximum_evidence_scene_order: int
) -> None:
    required = {
        "current_state",
        "developments_since_previous_checkpoint",
        "unresolved_threads",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Task 1 memory must contain exactly the three public fields")
    for field in sorted(required):
        rows = payload[field]
        if not isinstance(rows, list):
            raise ValueError(f"Task 1 memory field must be a list: {field}")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "claim",
                "evidence_scene_orders",
            }:
                raise ValueError(f"Task 1 memory row has invalid fields: {field}")
            if not clean_text(row["claim"]):
                raise ValueError(f"Task 1 memory row has an empty claim: {field}")
            scene_orders = row["evidence_scene_orders"]
            if (
                not isinstance(scene_orders, list)
                or any(not isinstance(item, int) or isinstance(item, bool) for item in scene_orders)
                or len(scene_orders) != len(set(scene_orders))
                or any(item <= 0 or item > maximum_evidence_scene_order for item in scene_orders)
            ):
                raise ValueError(
                    f"Task 1 memory cites an invalid or unread scene order: {field}"
                )


def reset_task1_interval_memory(payload: dict[str, Any]) -> dict[str, Any]:
    validate_task1_memory(payload, maximum_evidence_scene_order=10**9)
    return {
        "current_state": list(payload["current_state"]),
        "developments_since_previous_checkpoint": [],
        "unresolved_threads": list(payload["unresolved_threads"]),
    }


def build_task1_rolling_plans(
    *,
    movie_id: str,
    language: str,
    script_path: Path,
    task1: dict[str, Any],
    token_counter: TokenCounter,
    config: BenchmarkRuntimeConfig,
    prompt_registry: YamlPromptRegistry = PROMPTS,
) -> dict[str, Any]:
    scenes = load_scenes(script_path.resolve())
    _, units = build_screenplay_token_manifest(
        movie_id=movie_id,
        script_path=script_path,
        token_counter=token_counter,
        config=config,
    )
    prompt_path = config.prompt_path("task1_prediction", language)
    budget = config.call_budgets["task1_prediction"]
    effective_limit = (
        budget.max_input_tokens
        - config.reserved_chat_template_tokens
        - int(config.task1["reserved_prior_memory_tokens"])
    )
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    plans = []
    by_character: dict[str, list[dict[str, Any]]] = {}
    for instance in task1.get("instances", []):
        by_character.setdefault(instance["character_id"], []).append(instance)
    for character_id, instances in sorted(by_character.items()):
        instances.sort(
            key=lambda item: int(item["model_input"]["current_checkpoint_scene_order"])
        )
        checkpoint_by_order = {
            int(item["model_input"]["current_checkpoint_scene_order"]): item
            for item in instances
        }
        final_order = max(checkpoint_by_order)
        relevant_units = [item for item in units if int(item["scene_order"]) <= final_order]
        calls: list[dict[str, Any]] = []
        cursor = 0
        previous_checkpoint = 0
        while cursor < len(relevant_units):
            start = cursor
            block_units: list[dict[str, Any]] = []
            next_checkpoint = min(
                order
                for order in checkpoint_by_order
                if order >= int(relevant_units[cursor]["scene_order"])
            )
            best_tokens = 0
            while cursor < len(relevant_units):
                candidate = relevant_units[cursor]
                if int(candidate["scene_order"]) > next_checkpoint:
                    break
                projected = [*block_units, candidate]
                block_payload = [
                    _resolve_scene_unit(item, scene_by_id[item["scene_id"]])
                    for item in projected
                ]
                values = {
                    "focal_character": instances[0]["model_input"]["focal_character"],
                    "aliases": instances[0]["model_input"].get("aliases", []),
                    "previous_checkpoint_scene_order": previous_checkpoint,
                    "block_end_scene_order": int(candidate["scene_order"]),
                    "checkpoint_at_block_end": (
                        int(candidate["scene_order"]) == next_checkpoint
                        and int(candidate["part_order"]) == int(candidate["part_count"])
                    ),
                    "screenplay_block": block_payload,
                }
                prompt_variables = set(prompt_registry.get(prompt_path).variables)
                if "prior_scored_memory" in prompt_variables:
                    values.update(
                        {
                            "prior_scored_memory": _empty_task1_memory(),
                            "private_unscored_memory": _empty_task1_memory(),
                        }
                    )
                else:
                    values["prior_memory"] = _empty_task1_memory()
                system, user = prompt_registry.render(prompt_path, **values)
                measured = token_counter.count(system) + token_counter.count(user)
                if measured > effective_limit:
                    if not block_units:
                        raise ValueError(
                            "One Task 1 scene unit exceeds the conservative prompt budget: "
                            f"character={character_id} unit={candidate['unit_id']} "
                            f"tokens={measured}>{effective_limit}"
                        )
                    break
                block_units = projected
                best_tokens = measured
                cursor += 1
                if (
                    int(candidate["scene_order"]) == next_checkpoint
                    and int(candidate["part_order"]) == int(candidate["part_count"])
                ):
                    break
            if not block_units or cursor == start:
                raise ValueError(f"Task 1 rolling planner made no progress: {character_id}")
            last = block_units[-1]
            at_checkpoint = (
                int(last["scene_order"]) == next_checkpoint
                and int(last["part_order"]) == int(last["part_count"])
            )
            checkpoint_instance = checkpoint_by_order.get(next_checkpoint) if at_checkpoint else None
            call = {
                "call_id": stable_id(
                    "task1-rolling-call",
                    movie_id,
                    character_id,
                    len(calls) + 1,
                    block_units[0]["unit_id"],
                    block_units[-1]["unit_id"],
                ),
                "call_order": len(calls) + 1,
                "previous_checkpoint_scene_order": previous_checkpoint,
                "block_start_scene_order": int(block_units[0]["scene_order"]),
                "block_end_scene_order": int(last["scene_order"]),
                "scene_unit_refs": [item["unit_id"] for item in block_units],
                "scene_unit_count": len(block_units),
                "checkpoint_at_block_end": at_checkpoint,
                "checkpoint_id": (
                    checkpoint_instance["evaluator_reference"]["checkpoint_id"]
                    if checkpoint_instance
                    else ""
                ),
                "task_instance_id": (
                    checkpoint_instance["instance_id"] if checkpoint_instance else ""
                ),
                "conservative_raw_prompt_tokens": best_tokens,
                "reserved_prior_memory_tokens": int(
                    config.task1["reserved_prior_memory_tokens"]
                ),
                "reserved_chat_template_tokens": config.reserved_chat_template_tokens,
                "max_input_tokens": budget.max_input_tokens,
            }
            calls.append(call)
            if at_checkpoint:
                previous_checkpoint = next_checkpoint
        _validate_task1_plan_coverage(relevant_units, calls)
        plans.append(
            {
                "character_id": character_id,
                "focal_character": instances[0]["model_input"]["focal_character"],
                "aliases": instances[0]["model_input"].get("aliases", []),
                "checkpoint_count": len(instances),
                "call_count": len(calls),
                "calls": calls,
            }
        )
    return {
        "schema_version": "stage_task1_standard_24k_rolling_plan_v1",
        "track_id": config.track_id,
        "movie_id": movie_id,
        "language": normalize_language(language),
        "script_path": str(script_path.resolve()),
        "script_sha256": sha256_file(script_path.resolve()),
        "prompt_path": prompt_path,
        "prompt_source_paths": [
            str(path) for path in prompt_registry.get(prompt_path).source_paths
        ],
        "character_count": len(plans),
        "call_count": sum(item["call_count"] for item in plans),
        "plans": plans,
    }


def build_task1_rubric_candidates(
    *,
    task1: dict[str, Any],
    state_ledger: dict[str, Any],
    development_graph: dict[str, Any],
    evidence_bank: dict[str, Any],
    graph: dict[str, Any],
) -> dict[str, Any]:
    states = {item["state_id"]: item for item in state_ledger.get("states", [])}
    developments = {
        item["development_id"]: item
        for item in development_graph.get("developments", [])
    }
    evidence = {
        item["evidence_id"]: item for item in evidence_bank.get("evidence_units", [])
    }
    facts = {item["id"]: item for item in graph.get("nodes", [])}
    candidates = []
    for instance in task1.get("instances", []):
        reference = instance["evaluator_reference"]
        checkpoint_order = int(
            instance["model_input"]["current_checkpoint_scene_order"]
        )
        state_rows = [
            states[item]
            for item in reference["gold_current_state_ids"]
            if item in states
            and _evidence_scene_orders(
                states[item].get("supporting_evidence_ids", []),
                evidence,
                maximum_scene_order=checkpoint_order,
            )
        ]
        development_rows = [
            developments[item]
            for item in reference["gold_development_ids"]
            if item in developments
            and int(developments[item].get("effective_from_scene", 10**9))
            <= checkpoint_order
            and _development_evidence_scene_orders(
                developments[item],
                evidence,
                maximum_scene_order=checkpoint_order,
            )
        ]
        evidence_ids = [
            item
            for item in reference.get("supporting_evidence_ids", [])
            if item in evidence
            and int(evidence[item]["scene_order"]) <= checkpoint_order
        ]
        candidates.append(
            {
                "instance_id": instance["instance_id"],
                "character_id": instance["character_id"],
                "focal_character": instance["model_input"]["focal_character"],
                "checkpoint": {
                    "previous_scene_order": instance["model_input"][
                        "previous_checkpoint_scene_order"
                    ],
                    "current_scene_order": instance["model_input"][
                        "current_checkpoint_scene_order"
                    ],
                    "control_types": reference.get("checkpoint_control_types", []),
                },
                "state_candidates": [
                    {
                        "local_id": f"S{position}",
                        "dimension": item.get("dimension", ""),
                        "target": item.get("target_id_or_text", ""),
                        "claim": item.get("state_value", ""),
                        "valid_from_scene": item.get("valid_from_scene"),
                        "valid_until_scene": item.get("valid_until_scene"),
                        "supporting_scene_orders": _evidence_scene_orders(
                            item.get("supporting_evidence_ids", []),
                            evidence,
                            maximum_scene_order=checkpoint_order,
                        ),
                    }
                    for position, item in enumerate(state_rows, start=1)
                ],
                "development_candidates": [
                    _development_candidate(
                        position,
                        item,
                        states,
                        evidence,
                        maximum_scene_order=checkpoint_order,
                    )
                    for position, item in enumerate(development_rows, start=1)
                ],
                "supporting_evidence": _localize_evidence(evidence_ids, evidence),
                "future_fact_candidates": _future_fact_candidates(
                    reference.get("future_forbidden_fact_ids", []),
                    facts,
                    character_id=instance["character_id"],
                ),
                "future_negative_pool_count": len(
                    reference.get("future_forbidden_fact_ids", [])
                ),
                "requires_semantic_consolidation": True,
                "review_status": "candidate_not_gold",
                "construction_provenance": {
                    "checkpoint_id": reference["checkpoint_id"],
                    "state_ids": reference["gold_current_state_ids"],
                    "development_ids": reference["gold_development_ids"],
                    "future_forbidden_fact_ids": reference.get(
                        "future_forbidden_fact_ids", []
                    ),
                },
            }
        )
    return {
        "schema_version": "stage_task1_rubric_candidates_v1",
        "movie_id": task1.get("movie_id"),
        "candidate_count": len(candidates),
        "status": "requires_llm_consolidation_and_human_review",
        "candidates": candidates,
    }


def build_task3_actor_context_packs(
    *,
    task3: dict[str, Any],
    role_snapshots: dict[str, Any],
    evidence_bank: dict[str, Any],
    persona_bank: dict[str, Any],
    graph: dict[str, Any],
    token_counter: TokenCounter,
    config: BenchmarkRuntimeConfig,
    prompt_registry: YamlPromptRegistry = PROMPTS,
) -> dict[str, Any]:
    language = normalize_language(task3.get("language"))
    prompt_path = config.prompt_path("task3_actor", language)
    budget = config.call_budgets["task3_actor"]
    role_snapshots, backfilled_snapshot_count = _backfill_role_snapshot_scene_orders(
        role_snapshots, task3
    )
    packs = []
    for instance in task3.get("instances", []):
        actor_input = materialize_task3_actor_input(
            instance,
            role_snapshots=role_snapshots,
            evidence_bank=evidence_bank,
            persona_bank=persona_bank,
            graph=graph,
            memory_mode="full_visible_memory",
        )
        system, user = prompt_registry.render(
            prompt_path,
            character=actor_input["character"],
            role_context=actor_input["role_context"],
            interaction_context=actor_input["interaction_context"],
            current_user_turn=actor_input["current_user_turn"],
        )
        raw_tokens = token_counter.count(system) + token_counter.count(user)
        accounted_tokens = raw_tokens + config.reserved_chat_template_tokens
        fits = accounted_tokens <= budget.max_input_tokens
        packs.append(
            {
                "context_pack_id": stable_id(
                    "task3-context-pack",
                    instance["instance_id"],
                    sha256_json(actor_input),
                ),
                "instance_id": instance["instance_id"],
                "character_id": instance["character_id"],
                "checkpoint_id": instance["evaluator_reference"]["checkpoint_id"],
                "checkpoint_anchor": instance["model_input"]["checkpoint_anchor"],
                "memory_mode": "full_visible_memory",
                "actor_input": actor_input,
                "prompt_path": prompt_path,
                "raw_prompt_tokens": raw_tokens,
                "accounted_input_tokens": accounted_tokens,
                "max_input_tokens": budget.max_input_tokens,
                "materialization_status": (
                    "frozen_full_visible_memory"
                    if fits
                    else "requires_frozen_retrieval"
                ),
            }
        )
    return {
        "schema_version": "stage_task3_actor_context_packs_v2",
        "track_id": config.track_id,
        "movie_id": task3.get("movie_id"),
        "language": language,
        "prompt_path": prompt_path,
        "instance_count": len(packs),
        "frozen_count": sum(
            item["materialization_status"] == "frozen_full_visible_memory"
            for item in packs
        ),
        "requires_retrieval_count": sum(
            item["materialization_status"] == "requires_frozen_retrieval"
            for item in packs
        ),
        "filtered_source_item_count": sum(
            sum(item["actor_input"]["input_policy"]["filtered_counts"].values())
            for item in packs
        ),
        "backfilled_role_snapshot_scene_order_count": backfilled_snapshot_count,
        "context_packs": packs,
    }


def _backfill_role_snapshot_scene_orders(
    role_snapshots: dict[str, Any], task3: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """Migrate reviewed snapshots whose checkpoint scene was stored separately."""
    checkpoint_orders: dict[str, set[int]] = defaultdict(set)
    for instance in task3.get("instances", []):
        reference = instance.get("evaluator_reference", {})
        anchor = instance.get("model_input", {}).get("checkpoint_anchor", {})
        checkpoint_id = str(reference.get("checkpoint_id", ""))
        if checkpoint_id and anchor.get("scene_order") is not None:
            checkpoint_orders[checkpoint_id].add(int(anchor["scene_order"]))
    migrated = deepcopy(role_snapshots)
    count = 0
    for snapshot in migrated.get("role_snapshots", []):
        checkpoint_id = str(snapshot.get("checkpoint_id", ""))
        orders = checkpoint_orders.get(checkpoint_id, set())
        if snapshot.get("scene_order") is None:
            if len(orders) != 1:
                raise ValueError(
                    "Cannot uniquely recover role snapshot scene_order for "
                    f"checkpoint {checkpoint_id}: {sorted(orders)}"
                )
            snapshot["scene_order"] = next(iter(orders))
            snapshot["scene_order_provenance"] = "reviewed_task3_checkpoint_anchor"
            count += 1
        elif orders and int(snapshot["scene_order"]) not in orders:
            raise ValueError(
                f"Role snapshot scene_order conflicts with reviewed anchor: {checkpoint_id}"
            )
    return migrated, count


def build_task3_rubric_candidates(
    *,
    task3: dict[str, Any],
    state_ledger: dict[str, Any],
    evidence_bank: dict[str, Any],
    graph: dict[str, Any],
) -> dict[str, Any]:
    states = {item["state_id"]: item for item in state_ledger.get("states", [])}
    evidence = {
        item["evidence_id"]: item for item in evidence_bank.get("evidence_units", [])
    }
    facts = {item["id"]: item for item in graph.get("nodes", [])}
    candidates = []
    for instance in task3.get("instances", []):
        reference = instance["evaluator_reference"]
        anchor = reference.get("checkpoint_anchor")
        if not isinstance(anchor, dict):
            raise ValueError("Task 3 rubric construction requires checkpoint anchors")
        anchor_scene_order = int(anchor["scene_order"])
        anchor_char_end = int(anchor["char_end"])
        acceptable_state_ids = [
            item_id
            for item_id in reference.get("acceptable_state_fact_ids", [])
            if item_id in states
            and _record_visible_at_anchor(
                states[item_id],
                evidence,
                scene_order=anchor_scene_order,
                char_end=anchor_char_end,
                scene_field="valid_from_scene",
            )
        ]
        visible_memory_ids = [
            item_id
            for item_id in reference.get("required_memory_fact_ids", [])
            if item_id in facts
            and facts[item_id].get("source_scene_order") is not None
            and int(facts[item_id]["source_scene_order"]) < anchor_scene_order
        ]
        future_candidates = _selected_future_evidence_candidates(
            anchor.get("future_negative_evidence_ids", []), evidence
        )
        candidates.append(
            {
                "instance_id": instance["instance_id"],
                "character_id": instance["character_id"],
                "character": instance["character"],
                "checkpoint_id": reference["checkpoint_id"],
                "checkpoint_anchor": anchor,
                "prompt_family": reference["prompt_family"],
                "expected_stances": list(reference.get("expected_stances", [])),
                "acceptable_state_claims": _localize_records(
                    acceptable_state_ids,
                    states,
                    prefix="S",
                    text_key="state_value",
                ),
                "required_or_relevant_memories": _localize_records(
                    visible_memory_ids,
                    facts,
                    prefix="M",
                ),
                "contradictions": _localize_records(
                    reference.get("contradicting_fact_ids", []),
                    facts,
                    prefix="C",
                ),
                "unknown_at_checkpoint": _localize_records(
                    reference.get("unknown_fact_ids", []),
                    facts,
                    prefix="U",
                ),
                "supporting_evidence": _localize_evidence(
                    reference.get("supporting_evidence_ids", []), evidence
                ),
                "style_evidence": _localize_evidence(
                    reference.get("style_evidence_ids", []), evidence, prefix="T"
                ),
                "boundary_risk_type": reference.get("boundary_risk_type", ""),
                "paired_prompt_group_id": reference.get("paired_prompt_group_id", ""),
                "future_negative_pool_count": len(future_candidates),
                "future_fact_candidates": future_candidates,
                "requires_future_negative_selection": bool(
                    future_candidates
                ),
                "review_status": "candidate_not_gold",
                "construction_provenance": {
                    "prompt_id": reference["prompt_id"],
                    "reviewed_future_negative_evidence_ids": anchor.get(
                        "future_negative_evidence_ids", []
                    ),
                    "future_forbidden_fact_ids": reference.get(
                        "future_forbidden_fact_ids", []
                    ),
                },
            }
        )
    return {
        "schema_version": "stage_task3_rubric_candidates_v2",
        "movie_id": task3.get("movie_id"),
        "candidate_count": len(candidates),
        "status": "requires_human_review_and_future_negative_selection",
        "candidates": candidates,
    }


def _resolve_scene_unit(unit: dict[str, Any], scene: Scene) -> dict[str, Any]:
    text = scene.content[int(unit["char_start"]): int(unit["char_end"])]
    return {
        "scene_order": scene.order,
        "source_scene_id": scene.source_scene_id,
        "part_order": unit["part_order"],
        "part_count": unit["part_count"],
        "title": scene.title,
        "subtitle": scene.subtitle,
        "content": text,
    }


def _empty_task1_memory() -> dict[str, list[Any]]:
    return {
        "current_state": [],
        "developments_since_previous_checkpoint": [],
        "unresolved_threads": [],
    }


def _validate_task1_plan_coverage(
    units: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> None:
    expected = [item["unit_id"] for item in units]
    actual = [unit_id for call in calls for unit_id in call["scene_unit_refs"]]
    if actual != expected:
        raise ValueError("Task 1 rolling plan does not cover scene units exactly once in order")


def _evidence_scene_orders(
    evidence_ids: list[str],
    evidence: dict[str, dict[str, Any]],
    *,
    maximum_scene_order: int | None = None,
) -> list[int]:
    return sorted(
        {
            int(evidence[item]["scene_order"])
            for item in evidence_ids
            if item in evidence
            and (
                maximum_scene_order is None
                or int(evidence[item]["scene_order"]) <= maximum_scene_order
            )
        }
    )


def _localize_evidence(
    evidence_ids: list[str],
    evidence: dict[str, dict[str, Any]],
    *,
    prefix: str = "E",
) -> list[dict[str, Any]]:
    output = []
    for item_id in evidence_ids:
        item = evidence.get(item_id)
        if item is None:
            continue
        output.append(
            {
                "local_id": f"{prefix}{len(output) + 1}",
                "scene_order": int(item["scene_order"]),
                "evidence_text": item["evidence_text"],
            }
        )
    return output


def _development_candidate(
    position: int,
    development: dict[str, Any],
    states: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    *,
    maximum_scene_order: int,
) -> dict[str, Any]:
    return {
        "local_id": f"D{position}",
        "dimension": development.get("dimension", ""),
        "target": development.get("target_id_or_text", ""),
        "operation": development.get("operation", ""),
        "before_state_claims": [
            states[item]["state_value"]
            for item in development.get("before_state_ids", [])
            if item in states
        ],
        "resulting_state_claims": [
            states[item]["state_value"]
            for item in development.get("resulting_state_ids", [])
            if item in states
        ],
        "effective_from_scene": development.get("effective_from_scene"),
        "consequence_visible_from_scene": development.get(
            "consequence_visible_from_scene"
        ),
        "supporting_scene_orders": _development_evidence_scene_orders(
            development,
            evidence,
            maximum_scene_order=maximum_scene_order,
        ),
    }


def _development_evidence_scene_orders(
    development: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    *,
    maximum_scene_order: int,
) -> list[int]:
    return _evidence_scene_orders(
        [
            *development.get("evidence_before_ids", []),
            *development.get("evidence_catalyst_ids", []),
            *development.get("evidence_after_ids", []),
        ],
        evidence,
        maximum_scene_order=maximum_scene_order,
    )


def _fact_text(item: dict[str, Any]) -> str:
    return clean_text(
        item.get("fact")
        or item.get("description")
        or item.get("name")
        or item.get("state_value")
    )


def _localize_records(
    ids: list[str],
    records: dict[str, dict[str, Any]],
    *,
    prefix: str,
    text_key: str | None = None,
) -> list[dict[str, str]]:
    output = []
    seen: set[str] = set()
    for item_id in ids:
        item = records.get(item_id)
        if item is None:
            continue
        text = clean_text(item.get(text_key)) if text_key else _fact_text(item)
        normalized = text.casefold()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        output.append({"local_id": f"{prefix}{len(output) + 1}", "claim": text})
    return output


def _future_fact_candidates(
    ids: list[str],
    facts: dict[str, dict[str, Any]],
    *,
    character_id: str,
) -> list[dict[str, Any]]:
    selected = []
    seen: set[str] = set()
    for item_id in ids:
        item = facts.get(item_id)
        if item is None or not _fact_involves_character(item, character_id):
            continue
        claim = _fact_text(item)
        normalized = claim.casefold()
        if not claim or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(
            {
                "local_id": f"F{len(selected) + 1}",
                "claim": claim,
                "source_scene_order": item.get("source_scene_order"),
            }
        )
    selected.sort(
        key=lambda item: (
            int(item["source_scene_order"] or 10**9),
            item["local_id"],
        )
    )
    for position, item in enumerate(selected, start=1):
        item["local_id"] = f"F{position}"
    return selected


def _selected_future_evidence_candidates(
    evidence_ids: list[str], evidence: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = []
    seen: set[str] = set()
    for evidence_id in evidence_ids:
        item = evidence.get(evidence_id)
        if item is None:
            continue
        claim = clean_text(item.get("evidence_text"))
        normalized = claim.casefold()
        if not claim or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(
            {
                "local_id": f"F{len(selected) + 1}",
                "claim": claim,
                "source_scene_order": int(item["scene_order"]),
                "source_char_start": int(item["char_start"]),
                "source_char_end": int(item["char_end"]),
                "source_kind": "post_anchor_evidence",
            }
        )
    return selected


def _record_visible_at_anchor(
    item: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    *,
    scene_order: int,
    char_end: int,
    scene_field: str,
) -> bool:
    established = item.get(scene_field)
    if established is None:
        return False
    established_order = int(established)
    if established_order < scene_order:
        return True
    if established_order > scene_order:
        return False
    support_ids = item.get("supporting_evidence_ids", [])
    return any(
        evidence_id in evidence
        and (
            int(evidence[evidence_id].get("scene_order", 0)) < scene_order
            or (
                int(evidence[evidence_id].get("scene_order", 0)) == scene_order
                and int(evidence[evidence_id].get("char_end", 10**18)) <= char_end
            )
        )
        for evidence_id in support_ids
    )


def _fact_involves_character(item: dict[str, Any], character_id: str) -> bool:
    direct = {
        clean_text(item.get("subject_entity_id")),
        clean_text(item.get("object_entity_id")),
        clean_text(item.get("agent_entity_id")),
        clean_text(item.get("patient_entity_id")),
    }
    participants = {
        clean_text(value) for value in item.get("participant_entity_ids", [])
    }
    return character_id in direct or character_id in participants
