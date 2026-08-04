from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from ..chunking import TokenCounter, chunk_scene
from ..io import atomic_write_json, load_json, sha256_file
from ..models import Scene, clean_text, normalize_name, stable_id, unique_text
from ..prompt_loader import PROMPTS
from .models import (
    ACCESS_TYPES,
    BOUNDARY_RISK_TYPES,
    CHECKPOINT_TYPES,
    FACT_NODE_TYPES,
    PERSONA_EVIDENCE_KINDS,
    PERSONA_STABILITY,
    STATE_DIMENSIONS,
    STATE_DURABILITY,
    STATE_OPERATIONS,
    TASK3_PROMPT_FAMILIES,
    GraphIndex,
    TemporalBuildConfig,
    normalize_state_durability,
    normalize_state_dimension,
    normalize_state_polarity,
    normalize_boundary_risk_type,
    normalize_task3_prompt_family,
)
from .identity import source_character_ids_at_scene


def _prompt_parts(prompt_id: str) -> tuple[str, str]:
    spec = PROMPTS.get(prompt_id)
    return spec.system, spec.user


EVIDENCE_SYSTEM, EVIDENCE_USER = _prompt_parts("evidence")
EVIDENCE_REPAIR_SYSTEM, EVIDENCE_REPAIR_USER = _prompt_parts("evidence_repair")
STATE_OBSERVATION_SYSTEM, STATE_OBSERVATION_USER = _prompt_parts("state_observation")
STATE_TARGET_RESOLUTION_SYSTEM, STATE_TARGET_RESOLUTION_USER = _prompt_parts(
    "state_target_resolution"
)
STATE_RECONCILIATION_SYSTEM, STATE_RECONCILIATION_USER = _prompt_parts(
    "state_reconciliation"
)
DEVELOPMENT_SYSTEM, DEVELOPMENT_USER = _prompt_parts("development")
EPISTEMIC_SYSTEM, EPISTEMIC_USER = _prompt_parts("epistemic")
PERSONA_SYSTEM, PERSONA_USER = _prompt_parts("persona")
CHECKPOINT_SYSTEM, CHECKPOINT_USER = _prompt_parts("checkpoint")
TASK3_PROMPT_SYSTEM, TASK3_PROMPT_USER = _prompt_parts("task3_prompt")


def _task3_checkpoint_visible_local_ids(
    *,
    field: str,
    local_ids: list[str],
    mapping: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
) -> list[str]:
    """Restrict model-selected references to assets visible at one checkpoint."""
    if field == "state_ids":
        allowed = set(checkpoint["active_state_ids"])
        return [
            local_id
            for local_id in local_ids
            if mapping[local_id]["state_id"] in allowed
        ]
    if field == "required_access_ids":
        allowed = set(checkpoint["accessible_fact_ids"])
        return [
            local_id
            for local_id in local_ids
            if mapping[local_id]["fact_or_event_id"] in allowed
        ]
    if field == "supporting_evidence_ids":
        scene_order = int(checkpoint["scene_order"])
        return [
            local_id
            for local_id in local_ids
            if int(mapping[local_id]["scene_order"]) <= scene_order
        ]
    if field == "contradicting_fact_ids":
        allowed = set(checkpoint["accessible_fact_ids"])
        return [
            local_id
            for local_id in local_ids
            if mapping[local_id]["id"] in allowed
        ]
    if field == "style_evidence_ids":
        allowed = set(checkpoint["persona_evidence_ids"])
        return [
            local_id
            for local_id in local_ids
            if mapping[local_id]["persona_evidence_id"] in allowed
        ]
    raise ValueError(f"Unsupported Task 3 checkpoint-local field: {field}")


def _fit_task3_prompt_packages(
    packages: list[dict[str, Any]],
    *,
    token_counter: TokenCounter,
    max_input_tokens: int,
    render: Callable[[list[dict[str, Any]]], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit checkpoint evidence into the model budget with auditable packing.

    Task 3 only needs a small, checkpoint-visible evidence view.  Large films can
    nevertheless produce many future/unknown facts or repeated dialogue evidence.
    Pack those assets deterministically before the single semantic call, retaining
    at least one item from each useful evidence family where possible.
    """
    fitted = copy.deepcopy(packages)
    audit: list[dict[str, Any]] = []
    text_fields = {
        "active_states": ("state_value", "target"),
        "visible_access": ("fact_text",),
        "unknown_facts": ("text",),
        "future_facts": ("text",),
        "persona_evidence": ("value",),
        "dialogue_exemplars": ("text",),
    }

    def prompt_tokens(value: list[dict[str, Any]]) -> int:
        return token_counter.count(render(value))

    current_tokens = prompt_tokens(fitted)
    if current_tokens <= max_input_tokens:
        return fitted, audit

    # First bound individual evidence strings. This preserves the item and its
    # local ID while avoiding one pathological screenplay line dominating input.
    for package_index, package in enumerate(fitted):
        for field, keys in text_fields.items():
            for item_index, item in enumerate(package.get(field, [])):
                for key in keys:
                    value = clean_text(item.get(key))
                    limit = 900
                    if len(value) <= limit:
                        continue
                    item[key] = value[:limit].rstrip() + "..."
                    audit.append(
                        {
                            "action": "truncate_task3_prompt_asset_text",
                            "package_index": package_index,
                            "field": field,
                            "item_index": item_index,
                            "key": key,
                            "original_characters": len(value),
                            "kept_characters": limit,
                        }
                    )
    current_tokens = prompt_tokens(fitted)

    # Remove the least essential repeated assets first. A single item is kept
    # per family where possible so the model can still ground every prompt.
    removal_order = (
        "future_facts",
        "unknown_facts",
        "visible_access",
        "dialogue_exemplars",
        "persona_evidence",
        "active_states",
    )
    minimum_items = {
        "future_facts": 1,
        "unknown_facts": 1,
        "visible_access": 1,
        "dialogue_exemplars": 1,
        "persona_evidence": 1,
        "active_states": 1,
    }
    while current_tokens > max_input_tokens:
        best: tuple[int, int, str] | None = None
        for package_index, package in enumerate(fitted):
            for field in removal_order:
                values = package.get(field, [])
                if len(values) <= minimum_items[field]:
                    continue
                trial = copy.deepcopy(fitted)
                removed = trial[package_index][field].pop()
                reduction = current_tokens - prompt_tokens(trial)
                candidate = (reduction, package_index, field)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            break
        _, package_index, field = best
        removed = fitted[package_index][field].pop()
        audit.append(
            {
                "action": "drop_task3_prompt_asset_for_input_budget",
                "package_index": package_index,
                "field": field,
                "local_id": next(
                    (
                        clean_text(value)
                        for value in removed.values()
                        if isinstance(value, str) and value.startswith(("F", "W", "A", "S", "P"))
                    ),
                    "",
                ),
            }
        )
        current_tokens = prompt_tokens(fitted)

    # If a single retained item is still large, progressively shorten only its
    # natural-language fields. IDs and checkpoint structure remain untouched.
    if current_tokens > max_input_tokens:
        for limit in (512, 256, 128, 64):
            for package in fitted:
                for field, keys in text_fields.items():
                    for item in package.get(field, []):
                        for key in keys:
                            value = clean_text(item.get(key))
                            if len(value) > limit:
                                item[key] = value[:limit].rstrip() + "..."
                                audit.append(
                                    {
                                        "action": "compress_task3_prompt_asset_text_for_input_budget",
                                        "field": field,
                                        "key": key,
                                        "kept_characters": limit,
                                    }
                                )
            current_tokens = prompt_tokens(fitted)
            if current_tokens <= max_input_tokens:
                break
    if current_tokens > max_input_tokens:
        raise ValueError(
            "Task 3 checkpoint packages cannot fit input budget after deterministic packing: "
            f"tokens={current_tokens} budget={max_input_tokens}"
        )
    audit.append(
        {
            "action": "fit_task3_prompt_packages_to_input_budget",
            "final_input_tokens": current_tokens,
            "max_input_tokens": max_input_tokens,
        }
    )
    return fitted, audit


def assign_task3_longitudinal_pairs(
    prompts: list[dict[str, Any]],
    *,
    state_by_id: dict[str, dict[str, Any]],
    access_by_id: dict[str, dict[str, Any]],
    checkpoint_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assign disjoint cross-checkpoint pairs from shared semantic anchors."""
    anchors: dict[str, tuple[set[tuple[str, ...]], set[str]]] = {}
    for prompt in prompts:
        state_anchors = {
            (
                state["dimension"],
                normalize_name(state["target_id_or_text"]),
            )
            for state_id in prompt.get("state_ids", [])
            for state in [state_by_id.get(state_id)]
            if state and normalize_name(state.get("target_id_or_text"))
        }
        fact_anchors = {
            access["fact_or_event_id"]
            for access_id in prompt.get("required_access_ids", [])
            for access in [access_by_id.get(access_id)]
            if access
        }
        fact_anchors.update(prompt.get("contradicting_fact_ids", []))
        fact_anchors.update(prompt.get("unknown_fact_ids", []))
        fact_anchors.update(prompt.get("future_forbidden_fact_ids", []))
        anchors[prompt["prompt_id"]] = (state_anchors, fact_anchors)

    candidates = []
    ordered = sorted(prompts, key=lambda item: item["prompt_id"])
    for position, left in enumerate(ordered):
        left_checkpoint = checkpoint_by_id[left["checkpoint_id"]]
        left_states, left_facts = anchors[left["prompt_id"]]
        for right in ordered[position + 1 :]:
            if (
                left["character_id"] != right["character_id"]
                or left["prompt_family"] != right["prompt_family"]
                or left["checkpoint_id"] == right["checkpoint_id"]
            ):
                continue
            shared_states = left_states & anchors[right["prompt_id"]][0]
            shared_facts = left_facts & anchors[right["prompt_id"]][1]
            if not shared_states and not shared_facts:
                continue
            gap = abs(
                int(left_checkpoint["scene_order"])
                - int(checkpoint_by_id[right["checkpoint_id"]]["scene_order"])
            )
            candidates.append(
                (
                    -len(shared_states),
                    -len(shared_facts),
                    gap,
                    left["prompt_id"],
                    right["prompt_id"],
                    sorted(shared_states),
                    sorted(shared_facts),
                )
            )

    selected = []
    assigned: set[str] = set()
    prompts_by_id = {item["prompt_id"]: item for item in prompts}
    for _, _, _, left_id, right_id, shared_states, shared_facts in sorted(candidates):
        if left_id in assigned or right_id in assigned:
            continue
        group = stable_id("task3-longitudinal-pair", left_id, right_id)
        for prompt_id in (left_id, right_id):
            prompt = prompts_by_id[prompt_id]
            prompt["model_pair_group"] = prompt.get("pair_group", "")
            prompt["pair_group"] = group
        assigned.update((left_id, right_id))
        selected.append(
            {
                "pair_group": group,
                "prompt_ids": [left_id, right_id],
                "shared_state_anchors": [list(value) for value in shared_states],
                "shared_fact_ids": shared_facts,
            }
        )
    for prompt in prompts:
        if prompt["prompt_id"] not in assigned:
            prompt["model_pair_group"] = prompt.get("pair_group", "")
            prompt["pair_group"] = ""
    return {
        "policy": "disjoint_cross_checkpoint_shared_anchor_v1",
        "candidate_edge_count": len(candidates),
        "pair_count": len(selected),
        "paired_prompt_count": len(assigned),
        "pairs": selected,
    }


class TemporalAssetBuilder:
    def __init__(
        self,
        *,
        movie_id: str,
        llm_client: Any,
        token_counter: TokenCounter,
        max_input_tokens: int,
        config: TemporalBuildConfig,
        max_output_tokens: int = 8192,
        call_checkpoint_dir: Path | None = None,
    ):
        self.movie_id = movie_id
        self.llm_client = llm_client
        self.token_counter = token_counter
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max(1, max_output_tokens)
        self.config = config
        self.call_checkpoint_dir = call_checkpoint_dir
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    def build_character_registry(self, index: GraphIndex) -> dict[str, Any]:
        include_names = {normalize_name(name) for name in self.config.include_character_names}
        rows: list[dict[str, Any]] = []
        for entity in index.person_entities:
            name = clean_text(entity.get("canonical_name") or entity.get("name"))
            aliases = unique_text([name, *(entity.get("aliases") or [])])
            scene_ids = sorted(
                {
                    clean_text(value)
                    for value in entity.get("source_scene_ids", [])
                    if clean_text(value) in index.scene_order_by_id
                },
                key=lambda scene_id: index.scene_order_by_id[scene_id],
            )
            eligible_by_count = len(scene_ids) >= self.config.min_source_scenes
            eligible_by_name = not include_names or bool(
                include_names
                & {normalize_name(value) for value in aliases if normalize_name(value)}
            )
            rows.append(
                {
                    "character_id": entity["id"],
                    "canonical_name": name,
                    "aliases": aliases,
                    "identity_phases": [],
                    "first_scene_order": (
                        index.scene_order_by_id[scene_ids[0]] if scene_ids else 0
                    ),
                    "last_scene_order": (
                        index.scene_order_by_id[scene_ids[-1]] if scene_ids else 0
                    ),
                    "scene_ids": scene_ids,
                    "dialogue_scene_ids": [],
                    "construction_selected": eligible_by_count and eligible_by_name,
                    "task1_eligible": False,
                    "task1_exclusion_reasons": [],
                    "task3_single_turn_eligible": False,
                    "task3_exclusion_reasons": [],
                    "source_entity_ids": [entity["id"]],
                    "validation_status": "silver_candidate",
                }
            )
        selected = [row for row in rows if row["construction_selected"]]
        selected.sort(
            key=lambda row: (-len(row["scene_ids"]), row["canonical_name"], row["character_id"])
        )
        if self.config.max_characters:
            selected_ids = {
                row["character_id"] for row in selected[: self.config.max_characters]
            }
            for row in rows:
                row["construction_selected"] = row["character_id"] in selected_ids
        if not any(row["construction_selected"] for row in rows):
            raise ValueError("Character selection retained no person entities")
        return {
            "schema_version": "stage_character_registry_v1",
            "movie_id": self.movie_id,
            "characters": rows,
            "audit": {
                "person_entity_count": len(rows),
                "construction_selected_count": sum(
                    bool(row["construction_selected"]) for row in rows
                ),
                "selection_config": {
                    "min_source_scenes": self.config.min_source_scenes,
                    "max_characters": self.config.max_characters,
                    "include_character_names": list(
                        self.config.include_character_names
                    ),
                },
            },
        }

    async def build_evidence_bank(
        self,
        *,
        scenes: list[Scene],
        canonical_scene_records: list[dict[str, Any]],
        character_registry: dict[str, Any],
        index: GraphIndex,
    ) -> dict[str, Any]:
        scene_by_id = {scene.scene_id: scene for scene in scenes}
        records_by_scene = {
            clean_text(record.get("scene", {}).get("scene_id")): record
            for record in canonical_scene_records
        }
        characters = character_registry["characters"]
        jobs = []
        for scene in scenes:
            if scene.scene_id not in records_by_scene:
                raise ValueError(f"Canonical extraction missing scene: {scene.scene_id}")
            record = records_by_scene[scene.scene_id]
            chunks = record.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                raise ValueError(f"Canonical extraction has no chunks: {scene.scene_id}")
            reconstructed = "".join(str(chunk.get("content") or "") for chunk in chunks)
            if reconstructed != scene.content:
                raise ValueError(f"Canonical chunks do not reconstruct scene: {scene.scene_id}")
            relevant = [
                character
                for character in characters
                if scene.scene_id in character.get("scene_ids", [])
            ]
            if not relevant:
                continue
            for chunk in chunks:
                for evidence_chunk in self._evidence_source_chunks(scene, chunk):
                    jobs.append(
                        self._extract_evidence_chunk(
                            scene=scene,
                            chunk=evidence_chunk,
                            characters=relevant,
                        )
                    )
        results = await _gather_all_settled(jobs, label="evidence extraction")
        evidence = [item for result, _ in results for item in result]
        calls = [metadata for _, metadata in results]
        evidence.sort(
            key=lambda item: (
                int(item["scene_order"]),
                int(item["char_start"]),
                item["evidence_id"],
            )
        )
        ids = [item["evidence_id"] for item in evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence extraction produced duplicate stable IDs")
        dialogue_by_character: dict[str, set[str]] = defaultdict(set)
        for item in evidence:
            speaker = clean_text(item.get("speaker_character_id"))
            if speaker and item["evidence_type"] == "dialogue":
                dialogue_by_character[speaker].add(item["scene_id"])
        for character in characters:
            character["dialogue_scene_ids"] = sorted(
                dialogue_by_character.get(character["character_id"], set()),
                key=lambda scene_id: index.scene_order_by_id[scene_id],
            )
        if set(scene_by_id) != set(records_by_scene):
            raise ValueError("Screenplay and canonical extraction scene sets differ")
        return {
            "schema_version": "stage_evidence_bank_v1",
            "movie_id": self.movie_id,
            "evidence_units": evidence,
            "audit": {
                "llm_calls": calls,
                "evidence_count": len(evidence),
                "source_scene_count": len(scenes),
            },
        }

    def _evidence_source_chunks(
        self, scene: Scene, source_chunk: dict[str, Any]
    ) -> list[dict[str, Any]]:
        content = str(source_chunk.get("content") or "")
        budget = self.config.evidence_source_chunk_tokens
        if self.token_counter.count(content) <= budget:
            return [dict(source_chunk)]
        spans = self._evidence_line_spans(scene, source_chunk, content, budget)
        base = int(source_chunk.get("char_start", 0))
        return [
            {
                **source_chunk,
                "content": content[start:end],
                "char_start": base + start,
                "char_end": base + end,
                "temporal_subchunk_order": order,
                "temporal_subchunk_count": len(spans),
                "temporal_subchunk_scheme": "screenplay_lines_v1",
                "temporal_source_tokens": self.token_counter.count(
                    content[start:end]
                ),
            }
            for order, (start, end) in enumerate(spans, start=1)
        ]

    def _evidence_line_spans(
        self,
        scene: Scene,
        source_chunk: dict[str, Any],
        content: str,
        budget: int,
    ) -> list[tuple[int, int]]:
        line_ends = [match.end() for match in re.finditer(r"\n+", content)]
        if not line_ends or line_ends[-1] < len(content):
            line_ends.append(len(content))
        line_spans: list[tuple[int, int]] = []
        start = 0
        for end in line_ends:
            if end > start:
                line_spans.append((start, end))
            start = end

        packed: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        for line_start, line_end in line_spans:
            if self.token_counter.count(content[line_start:line_end]) > budget:
                if current_start is not None and current_end is not None:
                    packed.append((current_start, current_end))
                    current_start = current_end = None
                local_scene = Scene(
                    scene_id=(
                        f"{scene.scene_id}:evidence-line:"
                        f"{int(source_chunk.get('order', 0))}:{line_start}"
                    ),
                    source_scene_id=scene.source_scene_id,
                    order=scene.order,
                    title="",
                    subtitle="",
                    content=content[line_start:line_end],
                )
                for subchunk in chunk_scene(
                    movie_id=self.movie_id,
                    scene=local_scene,
                    token_counter=self.token_counter,
                    max_content_tokens=budget,
                ):
                    packed.append(
                        (
                            line_start + subchunk.char_start,
                            line_start + subchunk.char_end,
                        )
                    )
                continue
            if current_start is None:
                current_start, current_end = line_start, line_end
                continue
            if self.token_counter.count(content[current_start:line_end]) <= budget:
                current_end = line_end
            else:
                assert current_end is not None
                packed.append((current_start, current_end))
                current_start, current_end = line_start, line_end
        if current_start is not None and current_end is not None:
            packed.append((current_start, current_end))
        if "".join(content[start:end] for start, end in packed) != content:
            raise ValueError(
                f"Evidence subchunks do not reconstruct source chunk: {scene.scene_id}"
            )
        if any(
            self.token_counter.count(content[start:end]) > budget
            for start, end in packed
        ):
            raise ValueError(
                f"Evidence subchunk exceeds source budget: {scene.scene_id}"
            )
        return packed

    async def _extract_evidence_chunk(
        self,
        *,
        scene: Scene,
        chunk: dict[str, Any],
        characters: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local_to_character = {
            f"C{position:03d}": character
            for position, character in enumerate(characters, start=1)
        }
        prompt_characters = [
            {
                "local_character_id": local_id,
                "name": character["canonical_name"],
                "aliases": character["aliases"],
            }
            for local_id, character in local_to_character.items()
        ]
        chunk_order = int(chunk.get("order", 0))
        subchunk_order = chunk.get("temporal_subchunk_order")
        first_source_piece = subchunk_order is None or int(subchunk_order) == 1
        source_fields = {
            "title": scene.title if chunk_order == 1 and first_source_piece else "",
            "subtitle": scene.subtitle if chunk_order == 1 and first_source_piece else "",
            "content": str(chunk.get("content") or ""),
        }
        # Evidence prompts are chunked at screenplay-line boundaries, but a
        # model may quote a line that crosses one boundary. Keep the chunk as
        # the prompt source while allowing deterministic grounding against the
        # complete scene content; this avoids another semantic call and keeps
        # global character offsets stable.
        full_scene_content = scene.content
        scene_chunk = {
            "local_scene_id": "SCENE_CURRENT",
            "scene_order": scene.order,
            "chunk_order": chunk_order,
            **source_fields,
        }
        user_prompt = EVIDENCE_USER.format(
            language=self.config.language,
            characters=json.dumps(prompt_characters, ensure_ascii=False, indent=2),
            scene_chunk=json.dumps(scene_chunk, ensure_ascii=False, indent=2),
        )

        def normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
            _exact_object(payload, {"evidence_units"}, "evidence extraction")
            raw_items = _array(payload, "evidence_units")
            output: list[dict[str, Any]] = []
            seen_locations: set[tuple[str, int, int]] = set()
            dropped_ungrounded: list[dict[str, Any]] = []
            allowed_local = set(local_to_character)
            known_speaker_names = {
                normalize_name(value)
                for character in local_to_character.values()
                for value in [
                    character.get("canonical_name", ""),
                    *(character.get("aliases") or []),
                ]
                if normalize_name(value)
            }

            def dialogue_body_after_known_speaker(
                text: str, speaker_local_id: str = "", source_text: str = ""
            ) -> tuple[str, bool]:
                """Strip a model-added speaker prefix only when it is grounded."""
                # Qwen may render a screenplay speaker prefix with either a
                # colon or a line break (``BRUCE\nAh!``).  Both forms are
                # accepted only after the prefix is grounded in the source.
                match = re.match(r"^\s*([^:\n：]{1,80})\s*(?:[:：]|\n)\s*", text)
                if match is None:
                    return text.strip(), False
                # Parenthetical delivery directions can be inserted between a
                # speaker name and the colon.  Resolve only the base name so
                # ``母亲（看着姐姐）`` still matches the canonical ``母亲``.
                prefix_text = re.split(r"[（(]", match.group(1), maxsplit=1)[0]
                prefix = normalize_name(prefix_text)
                # A model-added colon is not enough evidence that the prefix
                # is a screenplay speaker label. Require the source chunk to
                # contain the same label on its own line (optionally followed
                # by a parenthetical direction) before stripping it.
                if source_text and not re.search(
                    r"(?im)^\s*"
                    + re.escape(prefix_text.strip())
                    + r"\s*(?:\*+\s*)?(?:\([^\n)]{0,80}\)\s*)?[:：\n]",
                    source_text,
                ):
                    return text.strip(), False
                speaker_names = set(known_speaker_names)
                speaker = local_to_character.get(speaker_local_id)
                if speaker is not None:
                    speaker_names.update(
                        normalize_name(value)
                        for value in [
                            speaker.get("canonical_name", ""),
                            *(speaker.get("aliases") or []),
                        ]
                        if normalize_name(value)
                    )
                    # Screenplay labels often shorten a canonical role name
                    # (e.g. ``母亲`` for ``诗人母亲``).  Only allow a short
                    # suffix when the model has supplied a valid local ID.
                    if any(
                        len(prefix) >= 2 and name.endswith(prefix)
                        for name in speaker_names
                    ):
                        return text[match.end() :].strip(), True
                if not prefix or prefix not in speaker_names:
                    return text.strip(), False
                return text[match.end() :].strip(), True

            for raw in raw_items:
                _object(raw, "evidence unit")
                source_field = clean_text(raw.get("source_field")).casefold()
                raw_evidence_type = clean_text(raw.get("evidence_type")).casefold()
                if source_field not in source_fields or not source_fields[source_field]:
                    raise ValueError(f"Invalid or unavailable evidence source_field: {source_field}")
                quote = str(raw.get("evidence_text") or "")
                if not quote:
                    raise ValueError("Evidence quote must be non-empty")
                occurrence = int(raw.get("occurrence_index", 0))
                if occurrence <= 0:
                    raise ValueError("Evidence occurrence_index must be positive")
                match_source = source_fields[source_field]
                match_base = int(chunk.get("char_start", 0)) if source_field == "content" else 0
                source_hash_text = match_source
                match_scope = "chunk"
                # Some repair responses wrap an otherwise exact screenplay
                # quote in an extra pair of ASCII or smart quotation marks.
                # Remove that wrapper only when the inner text is an exact
                # occurrence at the requested index; otherwise retain the
                # original quote and let the stricter grounding checks fail.
                quote_wrappers = {"'": "'", '"': '"', "‘": "’", "“": "”"}
                if (
                    len(quote) >= 2
                    and quote[0] in quote_wrappers
                    and quote[-1] == quote_wrappers[quote[0]]
                ):
                    inner_quote = quote[1:-1]
                    if _nth_occurrence(match_source, inner_quote, occurrence) >= 0:
                        occurrence_corrections = [
                            {
                                "action": "strip_deterministic_quote_wrapper",
                                "wrapper": quote[0],
                            }
                        ]
                        quote = inner_quote
                    else:
                        occurrence_corrections = []
                else:
                    occurrence_corrections = []
                local_start = _nth_occurrence(match_source, quote, occurrence)
                if local_start < 0:
                    # A model can label a line that is present in screenplay
                    # content as title/subtitle (notably title graphics). If
                    # the quote is uniquely grounded in content, repair only
                    # the source-field label deterministically.
                    alternate_sources: list[tuple[str, str, int, str]] = []
                    content_source = source_fields.get("content", "")
                    if source_field != "content" and content_source:
                        alternate_sources.append(
                            (
                                "content",
                                content_source,
                                int(chunk.get("char_start", 0)),
                                "chunk",
                            )
                        )
                    if full_scene_content and full_scene_content != content_source:
                        alternate_sources.append(("content", full_scene_content, 0, "full_scene"))
                    for alternate_field, alternate_source, alternate_base, alternate_scope in alternate_sources:
                        alternate_start = _nth_occurrence(
                            alternate_source, quote, occurrence
                        )
                        if alternate_start < 0:
                            normalized_alternate = _unique_normalized_source_span(
                                alternate_source, quote, minimum_length=4
                            )
                            if normalized_alternate is None:
                                continue
                            alternate_start, alternate_end = normalized_alternate
                        else:
                            alternate_end = alternate_start + len(quote)
                        occurrence_corrections.append(
                            {
                                "action": "repair_mislabeled_evidence_source_field",
                                "requested_source_field": source_field,
                                "resolved_source_field": alternate_field,
                                "match_scope": alternate_scope,
                            }
                        )
                        source_field = alternate_field
                        match_source = alternate_source
                        match_base = alternate_base
                        source_hash_text = alternate_source
                        match_scope = alternate_scope
                        local_start = alternate_start
                        if alternate_start != alternate_end - len(quote):
                            quote = alternate_source[alternate_start:alternate_end]
                        break
                if local_start < 0:
                    exact_positions = _occurrence_positions(
                        match_source, quote
                    )
                    if len(exact_positions) == 1:
                        occurrence_corrections.append(
                            {
                                "action": "normalize_unique_exact_occurrence",
                                "requested_occurrence_index": occurrence,
                                "resolved_occurrence_index": 1,
                            }
                        )
                        occurrence = 1
                        local_start = exact_positions[0]
                    elif exact_positions and occurrence == len(exact_positions) + 1:
                        # Qwen occasionally overcounts a repeated short quote by
                        # one. Resolve only this bounded off-by-one case; larger
                        # mismatches remain invalid rather than guessing a span.
                        occurrence_corrections.append(
                            {
                                "action": "normalize_repeated_exact_occurrence_off_by_one",
                                "requested_occurrence_index": occurrence,
                                "resolved_occurrence_index": len(exact_positions),
                                "exact_occurrence_count": len(exact_positions),
                            }
                        )
                        occurrence = len(exact_positions)
                        local_start = exact_positions[-1]
                    else:
                        # Short screenplay lines are often uniquely grounded even
                        # after the export inserts spaces around punctuation. Keep
                        # the match deterministic and unique, but allow four or
                        # more normalized characters instead of forcing a repair
                        # call for every short Chinese utterance.
                        normalized_span = _unique_normalized_source_span(
                            match_source, quote, minimum_length=4
                        )
                        correction_action = "restore_unique_normalized_source_span"
                        if normalized_span is None:
                            # Screenplay dialogue may omit a parenthetical stage
                            # direction such as ``(THEN)`` between the speaker
                            # label and the quoted line. Ignore only bounded
                            # parenthetical spans and still return the exact
                            # source substring for auditability.
                            normalized_span = _unique_normalized_source_span(
                                match_source,
                                quote,
                                minimum_length=4,
                                ignore_parentheticals=True,
                            )
                            correction_action = (
                                "restore_unique_normalized_source_span_ignoring_parentheticals"
                            )
                            if normalized_span is None:
                                normalized_span = _longest_unique_normalized_clause_span(
                                    match_source, quote, minimum_length=8
                                )
                                correction_action = (
                                    "restore_longest_unique_normalized_source_clause"
                                )
                            if normalized_span is None and raw_evidence_type == "dialogue":
                                # A screenplay may place a parenthetical or a
                                # continuation marker between the speaker label
                                # and the spoken line. If the model supplied a
                                # speaker prefix, accept only a unique short
                                # dialogue body as a grounded span.
                                dialogue_body, known_speaker = (
                                    dialogue_body_after_known_speaker(
                                        quote,
                                        clean_text(raw.get("speaker_character_id")),
                                        match_source,
                                    )
                                )
                                normalized_span = _normalized_occurrence_source_span(
                                    match_source,
                                    dialogue_body,
                                    occurrence=occurrence,
                                    minimum_length=1 if known_speaker else 4,
                                )
                                correction_action = (
                                    "restore_dialogue_body_occurrence_after_speaker_label"
                                )
                        if normalized_span is None:
                            # Retry only against the complete scene for content
                            # chunks. This handles quotes spanning adjacent
                            # temporal subchunks without accepting ungrounded
                            # text: the same exact/normalized matching rules are
                            # retained and the resulting span is scene-global.
                            if source_field == "content" and full_scene_content != match_source:
                                match_source = full_scene_content
                                match_base = 0
                                source_hash_text = match_source
                                match_scope = "full_scene"
                                exact_positions = _occurrence_positions(match_source, quote)
                                if 0 < occurrence <= len(exact_positions):
                                    resolved_position = exact_positions[occurrence - 1]
                                    normalized_span = (
                                        resolved_position,
                                        resolved_position + len(quote),
                                    )
                                    correction_action = "restore_full_scene_occurrence"
                                elif exact_positions and occurrence == len(exact_positions) + 1:
                                    # Keep the same bounded off-by-one rule used
                                    # for the chunk source when a model counts a
                                    # repeated quote one position too far.
                                    resolved_position = exact_positions[-1]
                                    normalized_span = (
                                        resolved_position,
                                        resolved_position + len(quote),
                                    )
                                    correction_action = (
                                        "restore_full_scene_repeated_occurrence_off_by_one"
                                    )
                                else:
                                    normalized_span = _unique_normalized_source_span(
                                        match_source, quote, minimum_length=4
                                    )
                                    correction_action = "restore_unique_full_scene_normalized_span"
                                    if normalized_span is None:
                                        normalized_span = _unique_normalized_source_span(
                                            match_source,
                                            quote,
                                            minimum_length=4,
                                            ignore_parentheticals=True,
                                        )
                                        correction_action = (
                                            "restore_unique_full_scene_normalized_span_ignoring_parentheticals"
                                        )
                                    if normalized_span is None:
                                        normalized_span = _longest_unique_normalized_clause_span(
                                            match_source, quote, minimum_length=8
                                        )
                                        correction_action = (
                                            "restore_longest_unique_full_scene_normalized_clause"
                                        )
                                    if normalized_span is None and raw_evidence_type == "dialogue":
                                        dialogue_body, known_speaker = (
                                            dialogue_body_after_known_speaker(
                                                quote,
                                                clean_text(raw.get("speaker_character_id")),
                                                match_source,
                                            )
                                        )
                                        normalized_span = _normalized_occurrence_source_span(
                                            match_source,
                                            dialogue_body,
                                            occurrence=occurrence,
                                            minimum_length=1 if known_speaker else 4,
                                        )
                                        correction_action = (
                                            "restore_full_scene_dialogue_body_occurrence_after_speaker_label"
                                        )
                            if normalized_span is None:
                                # A model can summarize an action instead of
                                # quoting the screenplay. Do not invent a
                                # source span or spend another repair call:
                                # discard only this unit and preserve the
                                # independently grounded units in the same
                                # chunk, with an auditable correction record.
                                correction = {
                                    "action": "drop_ungrounded_evidence_unit",
                                    "source_field": source_field,
                                    "requested_occurrence_index": occurrence,
                                    "exact_occurrence_count": len(exact_positions),
                                    "evidence_text": quote,
                                }
                                occurrence_corrections.append(correction)
                                dropped_ungrounded.append(correction)
                                continue
                        local_start, local_end = normalized_span
                        quote = match_source[local_start:local_end]
                        occurrence = 1
                        occurrence_corrections.append(
                            {
                                "action": correction_action,
                                "normalization": "nfkc_casefold_drop_whitespace_and_punctuation_v1",
                                "match_scope": match_scope,
                            }
                        )
                char_start = match_base + local_start
                char_end = char_start + len(quote)
                location = (source_field, char_start, char_end)
                if location in seen_locations:
                    prior = next(
                        item
                        for item in output
                        if (
                            item["source_field"],
                            item["char_start"],
                            item["char_end"],
                        )
                        == location
                    )
                    prior["deterministic_payload_corrections"].append(
                        {"action": "drop_duplicate_evidence_span"}
                    )
                    continue
                seen_locations.add(location)
                evidence_type = clean_text(raw.get("evidence_type")).casefold()
                if evidence_type not in {"dialogue", "action", "narration", "scene_context"}:
                    raise ValueError(f"Unsupported evidence_type: {evidence_type}")
                speaker_local = clean_text(raw.get("speaker_character_id"))
                if speaker_local and speaker_local not in allowed_local:
                    occurrence_corrections.append(
                        {
                            "action": "drop_unknown_speaker_character_id",
                            "field": "speaker_character_id",
                            "dropped_value": speaker_local,
                        }
                    )
                    speaker_local = ""
                roles = {}
                for key in (
                    "participant_character_ids",
                    "direct_observer_character_ids",
                    "addressee_character_ids",
                ):
                    values = raw.get(key)
                    if isinstance(values, list):
                        pass
                    elif values is None or not clean_text(values):
                        values = []
                        occurrence_corrections.append(
                            {
                                "action": "restore_missing_character_id_array",
                                "field": key,
                            }
                        )
                    else:
                        values = [values]
                        occurrence_corrections.append(
                            {
                                "action": "wrap_scalar_character_id_as_array",
                                "field": key,
                            }
                        )
                    local_values = unique_text(values)
                    unknown_values = [
                        value for value in local_values if value not in allowed_local
                    ]
                    if unknown_values:
                        occurrence_corrections.append(
                            {
                                "action": "drop_unknown_character_ids",
                                "field": key,
                                "dropped_values": unknown_values,
                            }
                        )
                    local_values = [
                        value for value in local_values if value in allowed_local
                    ]
                    roles[key] = [
                        local_to_character[value]["character_id"] for value in local_values
                    ]
                speaker_id = (
                    local_to_character[speaker_local]["character_id"]
                    if speaker_local
                    else ""
                )
                evidence_id = stable_id(
                    "evidence",
                    self.movie_id,
                    scene.scene_id,
                    source_field,
                    char_start,
                    char_end,
                    quote,
                )
                output.append(
                    {
                        "evidence_id": evidence_id,
                        "movie_id": self.movie_id,
                        "scene_id": scene.scene_id,
                        "scene_order": scene.order,
                        "chunk_id": clean_text(chunk.get("chunk_id")),
                        "source_field": source_field,
                        "source_occurrence_index": occurrence,
                        "char_start": char_start,
                        "char_end": char_end,
                        "evidence_type": evidence_type,
                        "speaker_character_id": speaker_id,
                        **roles,
                        "evidence_text": quote,
                        "source_sha256": hashlib.sha256(
                            source_hash_text.encode("utf-8")
                        ).hexdigest(),
                        "validation_status": "silver_candidate",
                        "deterministic_payload_corrections": occurrence_corrections,
                    }
                )
            if raw_items and not output and dropped_ungrounded:
                first = dropped_ungrounded[0]
                raise ValueError(
                    "Evidence quote/occurrence is not an exact source substring: "
                    f"source_field={first['source_field']}, "
                    f"occurrence_index={first['requested_occurrence_index']}, "
                    f"exact_occurrence_count={first['exact_occurrence_count']}, "
                    f"evidence_text={first['evidence_text']!r}"
                )
            return output

        stage = f"temporal_evidence:{scene.scene_id}:{chunk_order:04d}"
        if subchunk_order is not None:
            scheme = clean_text(chunk.get("temporal_subchunk_scheme")) or "legacy"
            stage += f":{scheme}:{int(subchunk_order):04d}"
        async def repair(
            candidate: dict[str, Any], validation_error: Exception, semantic_attempt: int
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            return await self._repair_evidence_payload(
                candidate=candidate,
                validation_error=validation_error,
                semantic_attempt=semantic_attempt,
                stage=stage,
                prompt_characters=prompt_characters,
                scene_chunk=scene_chunk,
                normalize=normalize,
            )

        return await self._semantic_call(
            system_prompt=EVIDENCE_SYSTEM,
            user_prompt=user_prompt,
            stage=stage,
            normalize=normalize,
            repair=repair,
        )

    async def _repair_evidence_payload(
        self,
        *,
        candidate: dict[str, Any],
        validation_error: Exception,
        semantic_attempt: int,
        stage: str,
        prompt_characters: list[dict[str, Any]],
        scene_chunk: dict[str, Any],
        normalize: Callable[[dict[str, Any]], Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        current = candidate
        last_error = validation_error
        calls: list[dict[str, Any]] = []
        repair_prompt = EVIDENCE_REPAIR_USER.format(
            language=self.config.language,
            validation_error=clean_text(last_error),
            candidate_payload=json.dumps(current, ensure_ascii=False, indent=2),
            characters=json.dumps(prompt_characters, ensure_ascii=False, indent=2),
            scene_chunk=json.dumps(scene_chunk, ensure_ascii=False, indent=2),
        )
        prompt_tokens = self.token_counter.count(
            EVIDENCE_REPAIR_SYSTEM + repair_prompt
        )
        if prompt_tokens > self.max_input_tokens:
            raise ValueError(
                f"Evidence repair prompt exceeds input budget: stage={stage} "
                f"tokens={prompt_tokens} budget={self.max_input_tokens}"
            )
        async with self._semaphore:
            call = await self.llm_client.generate_json(
                system_prompt=EVIDENCE_REPAIR_SYSTEM,
                user_prompt=repair_prompt,
                stage=f"temporal_evidence_repair:{stage}:01",
            )
        current = call.data
        calls.append(
            {
                **call.metadata,
                "semantic_attempt": semantic_attempt,
                "repair_attempt": 1,
                "prompt_tokens_measured": prompt_tokens,
                "validation_error": clean_text(last_error),
            }
        )
        normalize(current)
        return current, calls

    async def build_state_observations(
        self,
        *,
        registry: dict[str, Any],
        evidence_bank: dict[str, Any],
        index: GraphIndex,
    ) -> dict[str, Any]:
        selected = {
            item["character_id"]: item
            for item in registry["characters"]
            if item["construction_selected"]
        }
        evidence = evidence_bank["evidence_units"]
        jobs = []
        skipped_without_aligned_evidence: list[dict[str, Any]] = []
        for episode in index.episodes:
            scene_ids = set(index.node_scene_ids(episode))
            scene_evidence = [item for item in evidence if item["scene_id"] in scene_ids]
            facts = _episode_facts(index, episode)
            episode_evidence = _fact_aligned_evidence(
                facts=facts,
                evidence=scene_evidence,
            )
            relevant_characters = [
                character
                for character in selected.values()
                if _character_participates_in_episode(character, episode)
            ]
            if not episode_evidence:
                skipped_without_aligned_evidence.append(
                    {
                        "episode_id": episode["id"],
                        "scene_ids": sorted(scene_ids),
                        "relevant_character_ids": [
                            item["character_id"] for item in relevant_characters
                        ],
                        "reason": "no_primary_child_evidence_alignment",
                    }
                )
                continue
            for character in relevant_characters:
                jobs.append(
                    self._extract_state_observations(
                        character=character,
                        episode=episode,
                        evidence=episode_evidence,
                        facts=facts,
                        index=index,
                    )
                )
        results = await _gather_all_settled(jobs, label="state observations")
        observations = [item for result, _ in results for item in result]
        observations.sort(
            key=lambda item: (
                item["character_id"],
                int(item["observed_from_scene"]),
                item["observation_id"],
            )
        )
        return {
            "schema_version": "stage_state_observations_v1",
            "movie_id": self.movie_id,
            "observations": observations,
            "audit": {
                "llm_calls": [meta for _, meta in results],
                "evidence_selection_policy": (
                    "exact_primary_child_fact_evidence_alignment_v1"
                ),
                "skipped_without_aligned_evidence": skipped_without_aligned_evidence,
                "skipped_without_aligned_evidence_count": len(
                    skipped_without_aligned_evidence
                ),
            },
        }

    async def _extract_state_observations(
        self,
        *,
        character: dict[str, Any],
        episode: dict[str, Any],
        evidence: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        index: GraphIndex,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Extract state observations with a budgeted, evidence-preserving split.

        Most episodes fit in one request.  A very long scene/episode is split
        only when the measured prompt exceeds the configured input budget.  A
        partition contains evidence units together with the source facts they
        align to, so the model never receives a blind text fragment and every
        returned local ID remains grounded.
        """

        def prompt_tokens(
            partition_evidence: list[dict[str, Any]],
            partition_facts: list[dict[str, Any]],
        ) -> int:
            prompt_evidence = [
                {
                    "local_evidence_id": f"W{position:04d}",
                    "scene_order": item["scene_order"],
                    "type": item["evidence_type"],
                    "text": item["evidence_text"],
                    "speaker_is_target": (
                        item["speaker_character_id"] == character["character_id"]
                    ),
                }
                for position, item in enumerate(partition_evidence, start=1)
            ]
            prompt_facts = [
                {
                    "local_fact_id": f"F{position:04d}",
                    "type": item["node_type"],
                    "text": index.fact_text(item),
                    "scene_orders": index.node_scene_orders(item),
                }
                for position, item in enumerate(partition_facts, start=1)
            ]
            user_prompt = STATE_OBSERVATION_USER.format(
                language=self.config.language,
                character=json.dumps(_character_prompt(character), ensure_ascii=False),
                episode=json.dumps(_episode_prompt(episode), ensure_ascii=False, indent=2),
                evidence=json.dumps(prompt_evidence, ensure_ascii=False, indent=2),
                facts=json.dumps(prompt_facts, ensure_ascii=False, indent=2),
            )
            return self.token_counter.count(STATE_OBSERVATION_SYSTEM + user_prompt)

        # First preserve the normal one-call behavior.  The additional
        # partition bookkeeping is only activated for an over-budget prompt.
        if prompt_tokens(evidence, facts) <= self.max_input_tokens:
            return await self._extract_state_observations_partition(
                character=character,
                episode=episode,
                evidence=evidence,
                facts=facts,
                index=index,
            )

        fact_to_evidence_ids: dict[str, set[str]] = {}
        for fact in facts:
            fact_to_evidence_ids[fact["id"]] = {
                item["evidence_id"]
                for item in _fact_aligned_evidence(facts=[fact], evidence=evidence)
            }

        # Build small semantic units rather than splitting raw JSON in the
        # middle of a fact/evidence record.  A fact can align to multiple
        # evidence spans; assign it to its first span so partitioning does not
        # manufacture duplicate state observations.
        units: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        assigned_facts: set[str] = set()
        for item in evidence:
            aligned_facts = [
                fact
                for fact in facts
                if item["evidence_id"] in fact_to_evidence_ids.get(fact["id"], set())
                and fact["id"] not in assigned_facts
            ]
            assigned_facts.update(fact["id"] for fact in aligned_facts)
            units.append(([item], aligned_facts))
        for fact in facts:
            if fact["id"] not in assigned_facts:
                units.append(([], [fact]))

        partitions: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        current_evidence: list[dict[str, Any]] = []
        current_facts: list[dict[str, Any]] = []
        for unit_evidence, unit_facts in units:
            trial_evidence = [*current_evidence, *unit_evidence]
            trial_facts = [*current_facts, *unit_facts]
            if current_evidence or current_facts:
                if prompt_tokens(trial_evidence, trial_facts) > self.max_input_tokens:
                    partitions.append((current_evidence, current_facts))
                    current_evidence = []
                    current_facts = []
                    trial_evidence = list(unit_evidence)
                    trial_facts = list(unit_facts)
            if prompt_tokens(trial_evidence, trial_facts) > self.max_input_tokens:
                raise ValueError(
                    "One state observation evidence unit exceeds the input budget: "
                    f"character={character['character_id']} episode={episode['id']}"
                )
            current_evidence = trial_evidence
            current_facts = trial_facts
        if current_evidence or current_facts:
            partitions.append((current_evidence, current_facts))

        results = await _gather_all_settled(
            (
                self._extract_state_observations_partition(
                    character=character,
                    episode=episode,
                    evidence=partition_evidence,
                    facts=partition_facts,
                    index=index,
                    partition_index=partition_index,
                    partition_count=len(partitions),
                )
                for partition_index, (partition_evidence, partition_facts) in enumerate(
                    partitions, start=1
                )
            ),
            label=f"state observation partitions for {character['character_id']}:{episode['id']}",
        )
        observations = [item for rows, _ in results for item in rows]
        metadata = {
            "stage": f"temporal_state_observation:{character['character_id']}:{episode['id']}",
            "call_kind": "budgeted_evidence_fact_partitions",
            "partition_count": len(partitions),
            "partition_prompt_tokens": [
                prompt_tokens(partition_evidence, partition_facts)
                for partition_evidence, partition_facts in partitions
            ],
            "partitions": [
                {
                    "partition_index": partition_index,
                    "evidence_count": len(partition_evidence),
                    "fact_count": len(partition_facts),
                    "scene_orders": sorted(
                        {
                            int(item["scene_order"])
                            for item in partition_evidence
                        }
                    ),
                }
                for partition_index, (partition_evidence, partition_facts) in enumerate(
                    partitions, start=1
                )
            ],
            "llm_calls": [meta for _, meta in results],
        }
        return observations, metadata

    async def _extract_state_observations_partition(
        self,
        *,
        character: dict[str, Any],
        episode: dict[str, Any],
        evidence: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        index: GraphIndex,
        partition_index: int | None = None,
        partition_count: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local_evidence = {
            f"W{position:04d}": item for position, item in enumerate(evidence, start=1)
        }
        local_facts = {
            f"F{position:04d}": item for position, item in enumerate(facts, start=1)
        }
        prompt_evidence = [
            {
                "local_evidence_id": local_id,
                "scene_order": item["scene_order"],
                "type": item["evidence_type"],
                "text": item["evidence_text"],
                "speaker_is_target": (
                    item["speaker_character_id"] == character["character_id"]
                ),
            }
            for local_id, item in local_evidence.items()
        ]
        prompt_facts = [
            {
                "local_fact_id": local_id,
                "type": item["node_type"],
                "text": index.fact_text(item),
                "scene_orders": index.node_scene_orders(item),
            }
            for local_id, item in local_facts.items()
        ]
        user_prompt = STATE_OBSERVATION_USER.format(
            language=self.config.language,
            character=json.dumps(_character_prompt(character), ensure_ascii=False),
            episode=json.dumps(_episode_prompt(episode), ensure_ascii=False, indent=2),
            evidence=json.dumps(prompt_evidence, ensure_ascii=False, indent=2),
            facts=json.dumps(prompt_facts, ensure_ascii=False, indent=2),
        )
        episode_orders = set(index.node_scene_orders(episode))
        normalization_corrections: list[dict[str, Any]] = []

        def normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
            normalization_corrections.clear()
            _exact_object(payload, {"observations"}, "state observation")
            output = []
            for position, raw in enumerate(_array(payload, "observations"), start=1):
                _object(raw, "state observation")
                dimension = normalize_state_dimension(raw.get("dimension"))
                if dimension in {"emotion", "action", "behavior", "behaviour"}:
                    normalization_corrections.append(
                        {
                            "action": "drop_non_state_observation",
                            "observation_position": position,
                            "dimension": dimension,
                            "state_value": clean_text(raw.get("state_value")),
                        }
                    )
                    continue
                if dimension not in STATE_DIMENSIONS:
                    # The fixed state taxonomy is closed.  A model may still
                    # emit a near-miss label (for example ``mood`` or
                    # ``movement``); dropping it keeps the call useful while
                    # making the correction explicit instead of failing the
                    # whole episode.
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_state_dimension",
                            "observation_position": position,
                            "dimension": dimension,
                            "state_value": clean_text(raw.get("state_value")),
                        }
                    )
                    continue
                target = clean_text(raw.get("target"))
                state_value = clean_text(raw.get("state_value"))
                if not target or not state_value:
                    normalization_corrections.append(
                        {
                            "action": "drop_incomplete_state_observation",
                            "observation_position": position,
                            "dimension": dimension,
                            "missing_target": not bool(target),
                            "missing_state_value": not bool(state_value),
                        }
                    )
                    continue
                polarity = normalize_state_polarity(raw.get("polarity"))
                if polarity not in {
                    "positive",
                    "negative",
                    "neutral",
                    "mixed",
                    "uncertain",
                }:
                    raise ValueError(f"Unsupported state polarity: {polarity}")
                certainty = _probability(raw.get("certainty"), "state certainty")
                durability = normalize_state_durability(raw.get("durability"))
                if durability not in STATE_DURABILITY:
                    raise ValueError(f"Unsupported state durability: {durability}")
                scene_order = int(raw.get("observed_from_scene", 0))
                scene_order_corrections: list[dict[str, Any]] = []
                if scene_order not in episode_orders:
                    if len(episode_orders) == 1:
                        corrected_scene_order = next(iter(episode_orders))
                        scene_order_corrections.append(
                            {
                                "action": "normalize_to_unique_episode_scene",
                                "requested_scene_order": scene_order,
                                "resolved_scene_order": corrected_scene_order,
                            }
                        )
                        scene_order = corrected_scene_order
                    else:
                        raise ValueError("State observation scene is outside its episode")
                raw_evidence_ids = raw.get("supporting_evidence_ids")
                if not isinstance(raw_evidence_ids, list):
                    raise ValueError("state supporting evidence must be an array")
                evidence_local = unique_text(raw_evidence_ids)
                unknown_evidence_ids = [
                    value for value in evidence_local if value not in local_evidence
                ]
                evidence_local = [
                    value for value in evidence_local if value in local_evidence
                ]
                if unknown_evidence_ids:
                    normalization_corrections.append(
                        {
                            "action": "drop_unknown_state_evidence_ids",
                            "observation_position": position,
                            "unknown_local_ids": unknown_evidence_ids,
                        }
                    )
                if not evidence_local:
                    normalization_corrections.append(
                        {
                            "action": "drop_state_without_valid_evidence",
                            "observation_position": position,
                        }
                    )
                    continue
                evidence_ids = [
                    local_evidence[value]["evidence_id"] for value in evidence_local
                ]
                raw_fact_ids = raw.get("source_fact_ids")
                if not isinstance(raw_fact_ids, list):
                    raise ValueError("state source facts must be an array")
                fact_local = unique_text(raw_fact_ids)
                unknown_fact_ids = [
                    value for value in fact_local if value not in local_facts
                ]
                if unknown_fact_ids:
                    normalization_corrections.append(
                        {
                            "action": "drop_unknown_state_fact_ids",
                            "observation_position": position,
                            "unknown_local_ids": unknown_fact_ids,
                        }
                    )
                fact_ids = [
                    local_facts[value]["id"]
                    for value in fact_local
                    if value in local_facts
                ]
                observation_id = stable_id(
                    "state-observation",
                    self.movie_id,
                    character["character_id"],
                    episode["id"],
                    *([partition_index] if partition_index is not None else []),
                    position,
                    dimension,
                    target,
                    state_value,
                    scene_order,
                )
                output.append(
                    {
                        "observation_id": observation_id,
                        "character_id": character["character_id"],
                        "source_episode_id": episode["id"],
                        "dimension": dimension,
                        "target_id_or_text": target,
                        "state_value": state_value,
                        "polarity": polarity,
                        "certainty": certainty,
                        "durability": durability,
                        "observed_from_scene": scene_order,
                        "supporting_evidence_ids": evidence_ids,
                        "source_fact_ids": fact_ids,
                        "validation_status": "silver_candidate",
                        "deterministic_payload_corrections": scene_order_corrections,
                    }
                )
            return output

        observations, metadata = await self._semantic_call(
            system_prompt=STATE_OBSERVATION_SYSTEM,
            user_prompt=user_prompt,
            stage=(
                f"temporal_state_observation:{character['character_id']}:{episode['id']}"
                + (
                    f":{partition_index:04d}"
                    if partition_index is not None
                    else ""
                )
            ),
            normalize=normalize,
        )
        metadata = {
            **metadata,
            "deterministic_payload_corrections": list(normalization_corrections),
            "input_fact_count": len(facts),
            "selected_evidence_count": len(evidence),
            "evidence_selection_policy": (
                "exact_primary_child_fact_evidence_alignment_v1"
            ),
        }
        if partition_index is not None:
            metadata.update(
                {
                    "partition_index": partition_index,
                    "partition_count": partition_count,
                }
            )
        return observations, metadata

    async def resolve_state_targets(
        self,
        *,
        registry: dict[str, Any],
        observations: dict[str, Any],
    ) -> dict[str, Any]:
        character_by_id = {
            item["character_id"]: item for item in registry["characters"]
        }
        by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in observations["observations"]:
            by_character[item["character_id"]].append(item)
        known_characters = [
            {
                "canonical_name": item["canonical_name"],
                "aliases": item["aliases"],
            }
            for item in registry["characters"]
        ]

        async def resolve_character(
            character_id: str,
            items: list[dict[str, Any]],
        ) -> tuple[str, dict[tuple[str, str], dict[str, str]], dict[str, Any]]:
            character = character_by_id[character_id]
            unique_targets: dict[tuple[str, str], str] = {}
            for item in sorted(
                items,
                key=lambda value: (
                    value["dimension"],
                    normalize_name(value["target_id_or_text"]),
                    value["target_id_or_text"],
                ),
            ):
                key = (item["dimension"], normalize_name(item["target_id_or_text"]))
                unique_targets.setdefault(key, item["target_id_or_text"])
            local_targets = {
                f"T{position:04d}": {
                    "dimension": dimension,
                    "surface_target": surface,
                    "normalized_surface": normalized,
                }
                for position, ((dimension, normalized), surface) in enumerate(
                    unique_targets.items(), start=1
                )
            }
            output_budget = _usable_output_budget(self.max_output_tokens)
            packs: list[dict[str, dict[str, str]]] = []
            current: dict[str, dict[str, str]] = {}
            for local_id, target in local_targets.items():
                trial = {**current, local_id: target}
                prompt = _state_target_resolution_prompt(
                    character=character,
                    known_characters=known_characters,
                    local_targets=trial,
                    language=self.config.language,
                )
                estimated_output = _state_target_resolution_output_estimate(trial)
                if current and (
                    self.token_counter.count(
                        STATE_TARGET_RESOLUTION_SYSTEM + prompt
                    )
                    > self.max_input_tokens
                    or self.token_counter.count(estimated_output) > output_budget
                ):
                    packs.append(current)
                    current = {local_id: target}
                else:
                    current = trial
            if current:
                packs.append(current)

            async def resolve_pack(
                pack_index: int,
                pack_targets: dict[str, dict[str, str]],
            ) -> tuple[list[dict[str, str]], dict[str, Any]]:
                user_prompt = _state_target_resolution_prompt(
                    character=character,
                    known_characters=known_characters,
                    local_targets=pack_targets,
                    language=self.config.language,
                )
                normalization_corrections: list[dict[str, Any]] = []

                def normalize(payload: dict[str, Any]) -> list[dict[str, str]]:
                    normalization_corrections.clear()
                    _exact_object(payload, {"resolutions"}, "state target resolution")
                    output: list[dict[str, str]] = []
                    seen: list[str] = []
                    allowed_kinds = {
                        "character",
                        "self",
                        "proposition",
                        "goal",
                        "object",
                        "organization",
                        "location",
                        "other",
                    }
                    known_character_names = {
                        normalize_name(value)
                        for item in known_characters
                        for value in [
                            item.get("canonical_name"),
                            *(item.get("aliases") or []),
                        ]
                        if normalize_name(value)
                    }
                    seen_ids: set[str] = set()
                    for raw in _array(payload, "resolutions"):
                        _object(raw, "state target resolution item")
                        local_id = clean_text(raw.get("target_id"))
                        if local_id not in pack_targets:
                            normalization_corrections.append(
                                {
                                    "action": "drop_unknown_state_target_id",
                                    "target_id": local_id,
                                }
                            )
                            continue
                        if local_id in seen_ids:
                            normalization_corrections.append(
                                {
                                    "action": "drop_duplicate_state_target_resolution",
                                    "target_id": local_id,
                                }
                            )
                            continue
                        seen_ids.add(local_id)
                        seen.append(local_id)
                        canonical_target = clean_text(raw.get("canonical_target"))
                        target_kind = clean_text(raw.get("target_kind")).casefold()
                        source = pack_targets[local_id]
                        if not canonical_target:
                            canonical_target = source["surface_target"]
                            normalization_corrections.append(
                                {
                                    "action": "fallback_to_surface_target",
                                    "target_id": local_id,
                                }
                            )
                        if target_kind not in allowed_kinds:
                            normalization_corrections.append(
                                {
                                    "action": "normalize_unknown_target_kind",
                                    "target_id": local_id,
                                    "requested_target_kind": target_kind,
                                    "resolved_target_kind": "other",
                                }
                            )
                            target_kind = "other"
                        output.append(
                            {
                                "dimension": source["dimension"],
                                "normalized_surface": source["normalized_surface"],
                                "canonical_target": canonical_target,
                                "target_kind": target_kind,
                            }
                        )
                    missing_ids = [
                        local_id for local_id in pack_targets if local_id not in seen_ids
                    ]
                    for local_id in missing_ids:
                        source = pack_targets[local_id]
                        surface = source["surface_target"]
                        normalized_surface = source["normalized_surface"]
                        if normalized_surface in known_character_names:
                            fallback_kind = "character"
                        else:
                            fallback_kind = "other"
                        output.append(
                            {
                                "dimension": source["dimension"],
                                "normalized_surface": normalized_surface,
                                "canonical_target": surface,
                                "target_kind": fallback_kind,
                            }
                        )
                        normalization_corrections.append(
                            {
                                "action": "add_conservative_missing_state_target_resolution",
                                "target_id": local_id,
                                "canonical_target": surface,
                                "target_kind": fallback_kind,
                            }
                        )
                    return output

                rows, metadata = await self._semantic_call(
                    system_prompt=STATE_TARGET_RESOLUTION_SYSTEM,
                    user_prompt=user_prompt,
                    stage=(
                        f"temporal_state_target_resolution:{character_id}:"
                        f"{pack_index:04d}"
                    ),
                    normalize=normalize,
                )
                return rows, {
                    **metadata,
                    "deterministic_payload_corrections": list(
                        normalization_corrections
                    ),
                }

            pack_results = await _gather_all_settled(
                (
                    resolve_pack(pack_index, pack)
                    for pack_index, pack in enumerate(packs, start=1)
                ),
                label=f"state target resolution packs for {character_id}",
            )
            resolved_rows = [
                item for rows, _ in pack_results for item in rows
            ]
            metadata = {
                "stage": f"temporal_state_target_resolution:{character_id}",
                "call_kind": "token_budgeted_target_packs",
                "pack_count": len(packs),
                "pack_target_counts": [len(pack) for pack in packs],
                "llm_calls": [metadata for _, metadata in pack_results],
            }
            resolved = {
                (item["dimension"], item["normalized_surface"]): {
                    "canonical_target": item["canonical_target"],
                    "target_kind": item["target_kind"],
                }
                for item in resolved_rows
            }
            return character_id, resolved, metadata

        results = await _gather_all_settled(
            (
                resolve_character(character_id, items)
                for character_id, items in sorted(by_character.items())
            ),
            label="state target resolution characters",
        )
        resolution_by_character = {
            character_id: resolved
            for character_id, resolved, _ in results
        }
        resolved_observations = []
        self_identity_target_normalization_count = 0
        for item in observations["observations"]:
            key = (item["dimension"], normalize_name(item["target_id_or_text"]))
            resolution = resolution_by_character[item["character_id"]][key]
            canonical_target = resolution["canonical_target"]
            target_kind = resolution["target_kind"]
            character = character_by_id[item["character_id"]]
            self_names = {
                normalize_name(value)
                for value in [
                    character["canonical_name"],
                    *character.get("aliases", []),
                    *(
                        phase.get("name", "")
                        for phase in character.get("identity_phases", [])
                    ),
                ]
                if normalize_name(value)
            }
            if item["dimension"] == "status_identity" and (
                target_kind == "self"
                or normalize_name(canonical_target) in self_names
            ):
                canonical_target = "self"
                target_kind = "self"
                self_identity_target_normalization_count += 1
            resolved_observations.append(
                {
                    **item,
                    "raw_target_id_or_text": item["target_id_or_text"],
                    "target_id_or_text": canonical_target,
                    "target_kind": target_kind,
                }
            )
        return {
            "schema_version": "stage_state_observations_resolved_targets_v1",
            "movie_id": self.movie_id,
            "observations": resolved_observations,
            "audit": {
                "input_observation_count": len(observations["observations"]),
                "output_observation_count": len(resolved_observations),
                "resolved_surface_target_count": sum(
                    len(resolved) for resolved in resolution_by_character.values()
                ),
                "self_identity_target_normalization_count": (
                    self_identity_target_normalization_count
                ),
                "llm_calls": [metadata for _, _, metadata in results],
            },
        }

    def filter_state_observations_for_ledger(
        self,
        *,
        observations: dict[str, Any],
        index: GraphIndex,
    ) -> dict[str, Any]:
        episode_by_id = {item["id"]: item for item in index.episodes}
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for item in observations["observations"]:
            episode = episode_by_id.get(item["source_episode_id"])
            if episode is None:
                raise ValueError(
                    f"State observation references unknown Episode: {item['observation_id']}"
                )
            modality = clean_text(episode.get("modality")) or "asserted"
            if modality != "asserted":
                dropped.append(
                    {
                        "observation_id": item["observation_id"],
                        "character_id": item["character_id"],
                        "source_episode_id": item["source_episode_id"],
                        "episode_modality": modality,
                        "reason": "non_asserted_episode_does_not_update_reality_state_ledger",
                    }
                )
                continue
            kept.append(item)
        return {
            "schema_version": "stage_state_observations_asserted_ledger_input_v1",
            "movie_id": self.movie_id,
            "observations": kept,
            "audit": {
                "input_observation_count": len(observations["observations"]),
                "kept_observation_count": len(kept),
                "dropped_non_asserted_observation_count": len(dropped),
                "dropped": dropped,
                "model_calls": 0,
            },
        }

    async def build_state_ledger(
        self,
        *,
        registry: dict[str, Any],
        observations: dict[str, Any],
        scene_count: int,
    ) -> dict[str, Any]:
        by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in observations["observations"]:
            by_character[item["character_id"]].append(item)
        jobs = []
        for character in registry["characters"]:
            if not character["construction_selected"] or not by_character[character["character_id"]]:
                continue
            jobs.append(
                self._reconcile_character_states(
                    character=character,
                    observations=by_character[character["character_id"]],
                    scene_count=scene_count,
                )
            )
        results = await _gather_all_settled(jobs, label="state ledger characters")
        raw_states = [item for result, _ in results for item in result]
        states_by_id: dict[str, dict[str, Any]] = {}
        duplicate_merges: list[dict[str, Any]] = []
        for state in raw_states:
            state_id = state["state_id"]
            if state_id not in states_by_id:
                states_by_id[state_id] = state
                continue
            existing = states_by_id[state_id]
            identity_fields = (
                "character_id",
                "dimension",
                "target_id_or_text",
                "state_value",
                "polarity",
                "durability",
                "valid_from_scene",
                "valid_until_scene",
            )
            if any(existing[field] != state[field] for field in identity_fields):
                raise ValueError(f"State stable ID collision has different content: {state_id}")
            for field in (
                "source_observation_ids",
                "source_episode_ids",
                "source_unit_ids",
                "supporting_evidence_ids",
            ):
                existing[field] = unique_text([*existing[field], *state[field]])
            existing["certainty"] = max(
                float(existing["certainty"]), float(state["certainty"])
            )
            duplicate_merges.append(
                {
                    "state_id": state_id,
                    "merged_source_observation_ids": state["source_observation_ids"],
                }
            )
        states = list(states_by_id.values())
        states.sort(
            key=lambda item: (
                item["character_id"],
                int(item["valid_from_scene"]),
                item["state_id"],
            )
        )
        return {
            "schema_version": "stage_character_state_ledger_v1",
            "movie_id": self.movie_id,
            "states": states,
            "audit": {
                "llm_calls": [meta for _, meta in results],
                "raw_state_count": len(raw_states),
                "state_count": len(states),
                "duplicate_stable_id_merges": duplicate_merges,
            },
        }

    async def _reconcile_character_states(
        self,
        *,
        character: dict[str, Any],
        observations: list[dict[str, Any]],
        scene_count: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in sorted(
            observations,
            key=lambda value: (
                int(value["observed_from_scene"]),
                value["observation_id"],
            ),
        ):
            groups[(item["dimension"], normalize_name(item["target_id_or_text"]))].append(
                item
            )
        ordered_groups = sorted(
            groups.values(),
            key=lambda items: (
                min(int(item["observed_from_scene"]) for item in items),
                items[0]["dimension"],
                normalize_name(items[0]["target_id_or_text"]),
            ),
        )
        def group_budget(group: list[dict[str, Any]]) -> tuple[int, int]:
            assets = _state_reconciliation_prompt_assets(
                character=character,
                observations=group,
                language=self.config.language,
            )
            return (
                self.token_counter.count(
                    STATE_RECONCILIATION_SYSTEM + assets["user_prompt"]
                ),
                self.token_counter.count(
                    _state_reconciliation_output_estimate(group)
                ),
            )

        bounded_groups: list[dict[str, Any]] = []
        output_budget = _usable_output_budget(self.max_output_tokens)

        def append_budget_bounded_partitions(
            group: list[dict[str, Any]],
            *,
            maximum_valid_until_scene: int | None,
            identity_phase_partition: bool,
        ) -> int:
            partitions: list[list[dict[str, Any]]] = []
            current_partition: list[dict[str, Any]] = []
            for observation in group:
                trial = [*current_partition, observation]
                input_tokens, output_tokens = group_budget(trial)
                if current_partition and (
                    input_tokens > self.max_input_tokens
                    or output_tokens > output_budget
                ):
                    partitions.append(current_partition)
                    current_partition = [observation]
                    input_tokens, output_tokens = group_budget(current_partition)
                else:
                    current_partition = trial
                if (
                    input_tokens > self.max_input_tokens
                    or output_tokens > output_budget
                ):
                    raise ValueError(
                        "One state observation exceeds reconciliation budget: "
                        f"character={character['character_id']} "
                        f"observation={observation['observation_id']} "
                        f"input_tokens={input_tokens}>{self.max_input_tokens} "
                        f"estimated_output_tokens={output_tokens}>{output_budget}"
                    )
            if current_partition:
                partitions.append(current_partition)
            was_budget_partitioned = len(partitions) > 1
            for partition in partitions:
                bounded_groups.append(
                    {
                        "observations": partition,
                        "maximum_valid_until_scene": maximum_valid_until_scene,
                        "identity_phase_partition": identity_phase_partition,
                        "budget_partition": was_budget_partitioned,
                    }
                )
            return len(partitions)

        oversized_target_group_count = 0
        for group in ordered_groups:
            input_tokens, output_tokens = group_budget(group)
            if input_tokens <= self.max_input_tokens and output_tokens <= output_budget:
                bounded_groups.append(
                    {
                        "observations": group,
                        "maximum_valid_until_scene": None,
                        "identity_phase_partition": False,
                        "budget_partition": False,
                    }
                )
                continue
            oversized_target_group_count += 1
            is_self_identity = (
                group[0]["dimension"] == "status_identity"
                and normalize_name(group[0]["target_id_or_text"])
                == normalize_name("self")
            )
            phases = character.get("identity_phases", []) if is_self_identity else []
            if not phases:
                append_budget_bounded_partitions(
                    group,
                    maximum_valid_until_scene=None,
                    identity_phase_partition=False,
                )
                continue
            covered_ids: list[str] = []
            for phase in phases:
                start = int(phase["valid_from_scene"])
                end = int(phase["valid_until_scene"])
                subgroup = [
                    item
                    for item in group
                    if start <= int(item["observed_from_scene"]) <= end
                ]
                if not subgroup:
                    continue
                covered_ids.extend(item["observation_id"] for item in subgroup)
                append_budget_bounded_partitions(
                    subgroup,
                    maximum_valid_until_scene=(
                        end if end < scene_count else None
                    ),
                    identity_phase_partition=True,
                )
            if len(covered_ids) != len(group) or set(covered_ids) != {
                item["observation_id"] for item in group
            }:
                raise ValueError(
                    "Identity phase reconciliation does not exactly cover observations: "
                    f"character={character['character_id']}"
                )

        packs: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        for bounded_group in bounded_groups:
            group = bounded_group["observations"]
            is_standalone_partition = (
                bounded_group["identity_phase_partition"]
                or bounded_group["budget_partition"]
            )
            if is_standalone_partition:
                if current:
                    packs.append(
                        {
                            "observations": current,
                            "maximum_valid_until_scene": None,
                            "identity_phase_partition": False,
                            "budget_partition": False,
                        }
                    )
                    current = []
                packs.append(
                    {
                        "observations": group,
                        "maximum_valid_until_scene": bounded_group[
                            "maximum_valid_until_scene"
                        ],
                        "identity_phase_partition": bounded_group[
                            "identity_phase_partition"
                        ],
                        "budget_partition": bounded_group["budget_partition"],
                    }
                )
                continue
            trial = [*current, *group]
            tokens, estimated_output_tokens = group_budget(trial)
            if current and (
                tokens > self.max_input_tokens
                or estimated_output_tokens > output_budget
            ):
                packs.append(
                    {
                        "observations": current,
                        "maximum_valid_until_scene": None,
                        "identity_phase_partition": False,
                        "budget_partition": False,
                    }
                )
                current = list(group)
            else:
                current = trial
        if current:
            packs.append(
                {
                "observations": current,
                "maximum_valid_until_scene": None,
                "identity_phase_partition": False,
                "budget_partition": False,
            }
            )
        results = await _gather_all_settled(
            (
                self._reconcile_state_pack(
                    character=character,
                    observations=pack["observations"],
                    scene_count=scene_count,
                    pack_index=pack_index,
                    maximum_valid_until_scene=pack["maximum_valid_until_scene"],
                )
                for pack_index, pack in enumerate(packs, start=1)
            ),
            label=f"state reconciliation packs for {character['character_id']}",
        )
        reconciled = [item for states, _ in results for item in states]
        covered = [
            observation_id
            for state in reconciled
            for observation_id in state["source_observation_ids"]
        ]
        expected = [item["observation_id"] for item in observations]
        if len(covered) != len(expected) or set(covered) != set(expected):
            raise ValueError("Packed state reconciliation coverage is not exact")
        return reconciled, {
            "stage": f"temporal_state_reconciliation:{character['character_id']}",
            "call_kind": "token_budgeted_target_group_packs",
            "pack_count": len(packs),
            "pack_observation_counts": [len(pack["observations"]) for pack in packs],
            "identity_phase_partition_count": sum(
                bool(pack["identity_phase_partition"]) for pack in packs
            ),
            "oversized_target_group_count": oversized_target_group_count,
            "budget_partition_count": sum(
                bool(pack["budget_partition"]) for pack in packs
            ),
            "budget_partition_observation_counts": [
                len(pack["observations"])
                for pack in packs
                if pack["budget_partition"]
            ],
            "llm_calls": [metadata for _, metadata in results],
        }

    async def _reconcile_state_pack(
        self,
        *,
        character: dict[str, Any],
        observations: list[dict[str, Any]],
        scene_count: int,
        pack_index: int,
        maximum_valid_until_scene: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assets = _state_reconciliation_prompt_assets(
            character=character,
            observations=observations,
            language=self.config.language,
        )
        ordered = assets["ordered"]
        local = assets["local"]
        local_evidence = assets["local_evidence"]
        user_prompt = assets["user_prompt"]
        normalization_corrections: list[dict[str, Any]] = []

        def normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
            normalization_corrections.clear()
            _exact_object(payload, {"states"}, "state reconciliation")
            output = []
            seen_observations: list[str] = []
            seen_observation_set: set[str] = set()
            expanded: list[tuple[dict[str, Any], list[str], bool]] = []
            for raw in _array(payload, "states"):
                _object(raw, "state")
                local_ids = _local_id_list(
                    raw.get("source_observation_ids"), local, "state observations", True
                )
                duplicate_ids = [
                    local_id
                    for local_id in local_ids
                    if local_id in seen_observation_set
                ]
                local_ids = [
                    local_id
                    for local_id in local_ids
                    if local_id not in seen_observation_set
                ]
                if duplicate_ids:
                    normalization_corrections.append(
                        {
                            "action": "drop_duplicate_state_observation_references",
                            "source_observation_ids": duplicate_ids,
                        }
                    )
                if not local_ids:
                    continue
                seen_observations.extend(local_ids)
                seen_observation_set.update(local_ids)
                groups: dict[tuple[str, str], list[str]] = defaultdict(list)
                for local_id in local_ids:
                    item = local[local_id]
                    groups[
                        (
                            item["dimension"],
                            normalize_name(item["target_id_or_text"]),
                        )
                    ].append(local_id)
                if len(groups) > 1:
                    normalization_corrections.append(
                        {
                            "action": "split_cross_dimension_or_target_state",
                            "source_observation_ids": local_ids,
                            "group_count": len(groups),
                        }
                    )
                    expanded.extend((raw, [local_id], True) for local_id in local_ids)
                else:
                    expanded.append((raw, local_ids, False))

            missing_ids = [
                local_id
                for local_id in local
                if local_id not in seen_observation_set
            ]
            if missing_ids:
                normalization_corrections.append(
                    {
                        "action": "restore_unmentioned_observations_from_source",
                        "source_observation_ids": missing_ids,
                    }
                )
                expanded.extend(({}, [local_id], True) for local_id in missing_ids)
                seen_observations.extend(missing_ids)
                seen_observation_set.update(missing_ids)

            for raw, local_ids, source_fallback in expanded:
                source = [local[value] for value in local_ids]
                requested_dimension = normalize_state_dimension(raw.get("dimension"))
                dimension = source[0]["dimension"]
                requested_target = clean_text(raw.get("target"))
                if requested_dimension != dimension:
                    normalization_corrections.append(
                        {
                            "action": "restore_source_dimension",
                            "source_observation_ids": local_ids,
                            "requested_dimension": requested_dimension,
                            "resolved_dimension": dimension,
                        }
                    )
                target = source[0]["target_id_or_text"]
                if normalize_name(requested_target) != normalize_name(target):
                    normalization_corrections.append(
                        {
                            "action": "restore_resolved_source_target",
                            "source_observation_ids": local_ids,
                            "requested_target": requested_target,
                            "resolved_target": target,
                        }
                    )
                state_value = (
                    source[0]["state_value"]
                    if source_fallback
                    else clean_text(raw.get("state_value"))
                )
                requested_polarity = normalize_state_polarity(raw.get("polarity"))
                polarity = (
                    source[0]["polarity"] if source_fallback else requested_polarity
                )
                requested_durability = normalize_state_durability(raw.get("durability"))
                durability = (
                    source[0]["durability"]
                    if source_fallback
                    else requested_durability
                )
                if not target or not state_value:
                    state_value = source[0]["state_value"]
                    normalization_corrections.append(
                        {
                            "action": "restore_source_state_value",
                            "source_observation_ids": local_ids,
                        }
                    )
                if polarity not in {
                    "positive",
                    "negative",
                    "neutral",
                    "mixed",
                    "uncertain",
                }:
                    polarity = source[0]["polarity"]
                    normalization_corrections.append(
                        {
                            "action": "restore_source_polarity",
                            "source_observation_ids": local_ids,
                            "requested_polarity": requested_polarity,
                            "resolved_polarity": polarity,
                        }
                    )
                if durability not in STATE_DURABILITY:
                    durability = source[0]["durability"]
                    normalization_corrections.append(
                        {
                            "action": "restore_source_durability",
                            "source_observation_ids": local_ids,
                            "requested_durability": requested_durability,
                            "resolved_durability": durability,
                        }
                    )
                valid_from = (
                    int(source[0]["observed_from_scene"])
                    if source_fallback
                    else int(raw.get("valid_from_scene", 0))
                )
                valid_until_raw = (
                    int(source[0]["observed_from_scene"])
                    if source_fallback and durability == "transient"
                    else (0 if source_fallback else int(raw.get("valid_until_scene", 0)))
                )
                valid_until = valid_until_raw or None
                if not 1 <= valid_from <= scene_count:
                    valid_from = min(int(item["observed_from_scene"]) for item in source)
                    normalization_corrections.append(
                        {
                            "action": "restore_source_valid_from_scene",
                            "source_observation_ids": local_ids,
                            "resolved_valid_from_scene": valid_from,
                        }
                    )
                if valid_until is not None and not valid_from <= valid_until <= scene_count:
                    valid_until = (
                        max(int(item["observed_from_scene"]) for item in source)
                        if durability == "transient"
                        else None
                    )
                    normalization_corrections.append(
                        {
                            "action": "restore_source_valid_until_scene",
                            "source_observation_ids": local_ids,
                            "resolved_valid_until_scene": valid_until or 0,
                        }
                    )
                if (
                    maximum_valid_until_scene is not None
                    and valid_from <= maximum_valid_until_scene
                    and (
                        valid_until is None
                        or valid_until > maximum_valid_until_scene
                    )
                ):
                    valid_until = maximum_valid_until_scene
                    normalization_corrections.append(
                        {
                            "action": "clamp_identity_state_to_phase_boundary",
                            "source_observation_ids": local_ids,
                            "resolved_valid_until_scene": valid_until,
                        }
                    )
                allowed_evidence = {
                    evidence_id for item in source for evidence_id in item["supporting_evidence_ids"]
                }
                requested_evidence = raw.get("supporting_evidence_ids")
                requested_evidence = (
                    unique_text(requested_evidence)
                    if isinstance(requested_evidence, list)
                    else []
                )
                evidence_ids = [
                    local_evidence[value]
                    for value in requested_evidence
                    if value in local_evidence
                    and local_evidence[value] in allowed_evidence
                ]
                if source_fallback or not evidence_ids:
                    evidence_ids = sorted(allowed_evidence)
                if set(evidence_ids) != {
                    local_evidence[value]
                    for value in requested_evidence
                    if value in local_evidence
                }:
                    normalization_corrections.append(
                        {
                            "action": "restrict_to_source_evidence",
                            "source_observation_ids": local_ids,
                            "resolved_evidence_count": len(evidence_ids),
                        }
                    )
                try:
                    certainty = _probability(raw.get("certainty"), "state certainty")
                except Exception:
                    certainty = round(
                        sum(float(item["certainty"]) for item in source) / len(source),
                        6,
                    )
                    normalization_corrections.append(
                        {
                            "action": "restore_source_certainty",
                            "source_observation_ids": local_ids,
                            "resolved_certainty": certainty,
                        }
                    )
                state_id = stable_id(
                    "character-state",
                    self.movie_id,
                    character["character_id"],
                    dimension,
                    target,
                    valid_from,
                    valid_until or 0,
                    state_value,
                    polarity,
                    durability,
                )
                output.append(
                    {
                        "state_id": state_id,
                        "character_id": character["character_id"],
                        "dimension": dimension,
                        "target_id_or_text": target,
                        "state_value": state_value,
                        "polarity": polarity,
                        "certainty": certainty,
                        "durability": durability,
                        "valid_from_scene": valid_from,
                        "valid_until_scene": valid_until,
                        "supporting_evidence_ids": evidence_ids,
                        "source_observation_ids": [item["observation_id"] for item in source],
                        "source_unit_ids": unique_text(
                            fact_id for item in source for fact_id in item["source_fact_ids"]
                        ),
                        "source_episode_ids": unique_text(
                            item["source_episode_id"] for item in source
                        ),
                        "validation_status": "silver_candidate",
                    }
                )
            if set(seen_observations) != set(local) or len(seen_observations) != len(local):
                raise ValueError("State reconciliation observation coverage is not exact")
            return output

        stage = (
            f"temporal_state_reconciliation:{character['character_id']}:"
            f"{pack_index:04d}"
        )
        try:
            states, metadata = await self._semantic_call(
                system_prompt=STATE_RECONCILIATION_SYSTEM,
                user_prompt=user_prompt,
                stage=stage,
                normalize=normalize,
            )
        except Exception as exc:
            if not any(
                marker in clean_text(exc)
                for marker in (
                    "observation coverage is not exact",
                    "state observations must be non-empty",
                )
            ):
                raise
            evidence_to_local = {
                evidence_id: local_id
                for local_id, evidence_id in local_evidence.items()
            }
            fallback_payload = {
                "states": [
                    {
                        "source_observation_ids": [local_id],
                        "dimension": item["dimension"],
                        "target": item["target_id_or_text"],
                        "state_value": item["state_value"],
                        "polarity": item["polarity"],
                        "certainty": item["certainty"],
                        "durability": item["durability"],
                        "valid_from_scene": item["observed_from_scene"],
                        "valid_until_scene": (
                            item["observed_from_scene"]
                            if item["durability"] == "transient"
                            else 0
                        ),
                        "supporting_evidence_ids": [
                            evidence_to_local[evidence_id]
                            for evidence_id in item["supporting_evidence_ids"]
                        ],
                    }
                    for local_id, item in local.items()
                ]
            }
            states = normalize(fallback_payload)
            metadata = {
                "stage": stage,
                "call_kind": "deterministic_source_observation_fallback",
                "trigger_error": clean_text(exc),
                "source_observation_count": len(local),
                "model_calls_added": 0,
            }
        return states, {
            **metadata,
            "deterministic_payload_corrections": list(normalization_corrections),
        }

    async def build_development_graph(
        self,
        *,
        registry: dict[str, Any],
        state_ledger: dict[str, Any],
        evidence_bank: dict[str, Any],
        index: GraphIndex,
    ) -> dict[str, Any]:
        states_by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for state in state_ledger["states"]:
            states_by_character[state["character_id"]].append(state)
        evidence_by_id = {
            item["evidence_id"]: item for item in evidence_bank["evidence_units"]
        }
        jobs = []
        for character in registry["characters"]:
            character_states = states_by_character[character["character_id"]]
            if not character["construction_selected"] or not character_states:
                continue
            jobs.append(
                self._extract_character_developments(
                    character=character,
                    states=character_states,
                    evidence_by_id=evidence_by_id,
                    index=index,
                )
            )
        results = await _gather_all_settled(jobs, label="development extraction")
        raw_developments = [item for result, _ in results for item in result]
        developments, sanitization = sanitize_developments(
            raw_developments, state_ledger["states"]
        )
        developments.sort(
            key=lambda item: (
                item["character_id"],
                int(item["effective_from_scene"]),
                item["development_id"],
            )
        )
        return {
            "schema_version": "stage_character_development_graph_v1",
            "movie_id": self.movie_id,
            "developments": developments,
            "audit": {
                "llm_calls": [meta for _, meta in results],
                "development_sanitization": sanitization,
            },
        }

    async def _extract_character_developments(
        self,
        *,
        character: dict[str, Any],
        states: list[dict[str, Any]],
        evidence_by_id: dict[str, dict[str, Any]],
        index: GraphIndex,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for state in states:
            if state["durability"] != "durable":
                continue
            groups[
                (state["dimension"], normalize_name(state["target_id_or_text"]))
            ].append(state)
        focus_groups = [
            sorted(
                items,
                key=lambda item: (int(item["valid_from_scene"]), item["state_id"]),
            )
            for items in groups.values()
            if len(items) >= 2
        ]
        focus_groups.sort(
            key=lambda items: (
                int(items[0]["valid_from_scene"]),
                items[0]["dimension"],
                normalize_name(items[0]["target_id_or_text"]),
            )
        )
        jobs = []
        group_audits = []
        for group_index, focus_states in enumerate(focus_groups, start=1):
            def context_facts(
                partition_states: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                fact_ids = unique_text(
                    fact_id
                    for state in partition_states
                    for fact_id in state["source_unit_ids"]
                )
                source_episode_ids = unique_text(
                    episode_id
                    for state in partition_states
                    for episode_id in state["source_episode_ids"]
                )
                fact_ids = unique_text(
                    [
                        *fact_ids,
                        *_immediate_development_fact_ids(
                            index=index,
                            character=character,
                            source_episode_ids=source_episode_ids,
                        ),
                    ]
                )
                return [
                    index.nodes_by_id[fact_id]
                    for fact_id in fact_ids
                    if fact_id in index.nodes_by_id
                    and index.nodes_by_id[fact_id].get("node_type")
                    in FACT_NODE_TYPES
                ]

            def fits_input_budget(
                partition_states: list[dict[str, Any]],
            ) -> bool:
                assets = _development_prompt_assets(
                    character=character,
                    states=partition_states,
                    focus_state_ids={
                        item["state_id"] for item in partition_states
                    },
                    facts=context_facts(partition_states),
                    evidence_by_id=evidence_by_id,
                    index=index,
                    language=self.config.language,
                )
                return self.token_counter.count(
                    DEVELOPMENT_SYSTEM + assets["user_prompt"]
                ) <= self.max_input_tokens

            partitions = _partition_development_sequence(
                focus_states,
                fits=fits_input_budget,
            )
            partition_audits = []
            for partition_index, partition_states in enumerate(
                partitions, start=1
            ):
                facts = context_facts(partition_states)
                prompt_assets = _development_prompt_assets(
                    character=character,
                    states=partition_states,
                    focus_state_ids={
                        item["state_id"] for item in partition_states
                    },
                    facts=facts,
                    evidence_by_id=evidence_by_id,
                    index=index,
                    language=self.config.language,
                )
                prompt_tokens = self.token_counter.count(
                    DEVELOPMENT_SYSTEM + prompt_assets["user_prompt"]
                )
                jobs.append(
                    self._extract_developments(
                        character=character,
                        states=partition_states,
                        focus_state_ids={
                            item["state_id"] for item in partition_states
                        },
                        facts=facts,
                        evidence_by_id=evidence_by_id,
                        index=index,
                        group_index=group_index,
                        partition_index=(
                            partition_index if len(partitions) > 1 else None
                        ),
                    )
                )
                partition_audits.append(
                    {
                        "partition_index": partition_index,
                        "focus_state_count": len(partition_states),
                        "fact_count": len(facts),
                        "prompt_tokens_measured": prompt_tokens,
                        "first_scene": min(
                            int(item["valid_from_scene"])
                            for item in partition_states
                        ),
                        "last_scene": max(
                            int(item["valid_from_scene"])
                            for item in partition_states
                        ),
                    }
                )
            group_audits.append(
                {
                    "group_index": group_index,
                    "dimension": focus_states[0]["dimension"],
                    "target": focus_states[0]["target_id_or_text"],
                    "focus_state_count": len(focus_states),
                    "partition_count": len(partitions),
                    "partitions": partition_audits,
                }
            )
        if not jobs:
            return [], {
                "stage": f"temporal_development:{character['character_id']}",
                "call_kind": "durable_target_sequences",
                "group_count": 0,
                "llm_calls": [],
            }
        results = await _gather_all_settled(
            jobs,
            label=f"development groups for {character['character_id']}",
        )
        return [item for rows, _ in results for item in rows], {
            "stage": f"temporal_development:{character['character_id']}",
            "call_kind": "durable_target_sequences",
            "group_count": len(focus_groups),
            "groups": group_audits,
            "llm_calls": [metadata for _, metadata in results],
        }

    async def _extract_developments(
        self,
        *,
        character: dict[str, Any],
        states: list[dict[str, Any]],
        focus_state_ids: set[str],
        facts: list[dict[str, Any]],
        evidence_by_id: dict[str, dict[str, Any]],
        index: GraphIndex,
        group_index: int,
        partition_index: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assets = _development_prompt_assets(
            character=character,
            states=states,
            focus_state_ids=focus_state_ids,
            facts=facts,
            evidence_by_id=evidence_by_id,
            index=index,
            language=self.config.language,
        )
        local_states = assets["local_states"]
        local_facts = assets["local_facts"]
        local_evidence = assets["local_evidence"]
        user_prompt = assets["user_prompt"]

        normalization_corrections: list[dict[str, Any]] = []

        def normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
            normalization_corrections.clear()
            _exact_object(payload, {"developments"}, "development extraction")
            output = []
            for position, raw in enumerate(
                _array(payload, "developments"), start=1
            ):
                _object(raw, "development")
                dimension = normalize_state_dimension(raw.get("dimension"))
                target = clean_text(raw.get("target"))
                operation = clean_text(raw.get("operation")).casefold()
                if dimension not in STATE_DIMENSIONS or operation not in STATE_OPERATIONS:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "unsupported_dimension_or_operation",
                        }
                    )
                    continue

                def known_ids(value: Any, mapping: dict[str, Any]) -> list[str]:
                    return [
                        item
                        for item in unique_text(value if isinstance(value, list) else [])
                        if item in mapping
                    ]

                before_local = known_ids(raw.get("before_state_ids"), local_states)
                result_local = known_ids(raw.get("resulting_state_ids"), local_states)
                invariant_local = known_ids(raw.get("invariant_state_ids"), local_states)
                catalyst_local = known_ids(raw.get("catalyst_fact_ids"), local_facts)
                consequence_local = known_ids(
                    raw.get("downstream_consequence_ids"), local_facts
                )
                missing_required = [
                    name
                    for name, values in {
                        "resulting_states": result_local,
                        "catalyst_facts": catalyst_local,
                        "consequence_facts": consequence_local,
                    }.items()
                    if not values
                ]
                if missing_required:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "missing_grounded_required_fields",
                            "missing_fields": missing_required,
                        }
                    )
                    continue
                before = [local_states[value] for value in before_local]
                resulting = [local_states[value] for value in result_local]
                focus_local_ids = {
                    local_id
                    for local_id, item in local_states.items()
                    if item["state_id"] in focus_state_ids
                }
                if not set(before_local + result_local) <= focus_local_ids:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "before_or_result_outside_focus_sequence",
                        }
                    )
                    continue
                if invariant_local:
                    normalization_corrections.append(
                        {
                            "action": "clear_model_selected_invariants",
                            "development_position": position,
                            "state_ids": [
                                local_states[value]["state_id"]
                                for value in invariant_local
                            ],
                            "reason": "invariants_are_derived_from_checkpoint_continuity",
                        }
                    )
                    invariant_local = []
                compared = [*before, *resulting]
                resolved_dimension = resulting[0]["dimension"]
                resolved_target = resulting[0]["target_id_or_text"]
                if any(item["dimension"] != resolved_dimension for item in compared):
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "source_state_dimension_mismatch",
                        }
                    )
                    continue
                if any(
                    normalize_name(item["target_id_or_text"])
                    != normalize_name(resolved_target)
                    for item in compared
                ):
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "source_state_target_mismatch",
                        }
                    )
                    continue
                if dimension != resolved_dimension or normalize_name(target) != normalize_name(
                    resolved_target
                ):
                    normalization_corrections.append(
                        {
                            "action": "restore_focus_dimension_and_target",
                            "development_position": position,
                            "requested_dimension": dimension,
                            "resolved_dimension": resolved_dimension,
                            "requested_target": target,
                            "resolved_target": resolved_target,
                        }
                    )
                    dimension = resolved_dimension
                    target = resolved_target
                if operation in {"update", "revoke"} and not before:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "update_or_revoke_without_before_state",
                        }
                    )
                    continue
                effective = int(raw.get("effective_from_scene", 0))
                consequence_visible = int(raw.get("consequence_visible_from_scene", 0))
                if effective <= 0 or consequence_visible < effective:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "invalid_temporal_boundary",
                        }
                    )
                    continue
                latest_before_scene = max(
                    (int(item["valid_from_scene"]) for item in before),
                    default=0,
                )
                earliest_result_scene = min(
                    int(item["valid_from_scene"]) for item in resulting
                )
                if before and latest_before_scene >= earliest_result_scene:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "non_forward_state_transition",
                            "latest_before_scene": latest_before_scene,
                            "earliest_result_scene": earliest_result_scene,
                        }
                    )
                    continue
                if not latest_before_scene <= effective <= earliest_result_scene:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "effective_scene_outside_state_transition",
                        }
                    )
                    continue
                if consequence_visible < earliest_result_scene:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "consequence_precedes_resulting_state",
                        }
                    )
                    continue

                def fact_has_scene_in_range(
                    local_id: str, *, minimum: int, maximum: int | None = None
                ) -> bool:
                    orders = index.node_scene_orders(local_facts[local_id])
                    return any(
                        order >= minimum and (maximum is None or order <= maximum)
                        for order in orders
                    )

                supported_catalyst_local = [
                    local_id
                    for local_id in catalyst_local
                    if fact_has_scene_in_range(
                        local_id,
                        minimum=latest_before_scene,
                        maximum=earliest_result_scene,
                    )
                ]
                supported_consequence_local = [
                    local_id
                    for local_id in consequence_local
                    if fact_has_scene_in_range(
                        local_id,
                        minimum=consequence_visible,
                    )
                ]
                if not supported_catalyst_local or not supported_consequence_local:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "fact_support_outside_transition_window",
                        }
                    )
                    continue
                if (
                    supported_catalyst_local != catalyst_local
                    or supported_consequence_local != consequence_local
                ):
                    normalization_corrections.append(
                        {
                            "action": "filter_temporally_misaligned_fact_support",
                            "development_position": position,
                            "dropped_catalyst_ids": [
                                local_facts[value]["id"]
                                for value in catalyst_local
                                if value not in supported_catalyst_local
                            ],
                            "dropped_consequence_ids": [
                                local_facts[value]["id"]
                                for value in consequence_local
                                if value not in supported_consequence_local
                            ],
                        }
                    )
                    catalyst_local = supported_catalyst_local
                    consequence_local = supported_consequence_local
                evidence_fields = {}
                for key in (
                    "evidence_before_ids",
                    "evidence_catalyst_ids",
                    "evidence_after_ids",
                ):
                    local_values = known_ids(raw.get(key), local_evidence)
                    evidence_fields[key] = [
                        local_evidence[value]["evidence_id"] for value in local_values
                    ]
                if not evidence_fields["evidence_catalyst_ids"] or not evidence_fields["evidence_after_ids"]:
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_development",
                            "development_position": position,
                            "reason": "missing_catalyst_or_after_evidence",
                        }
                    )
                    continue
                before_ids = [item["state_id"] for item in before]
                resulting_ids = [item["state_id"] for item in resulting]
                catalyst_ids = [local_facts[value]["id"] for value in catalyst_local]
                consequence_ids = [local_facts[value]["id"] for value in consequence_local]
                invariant_ids = [local_states[value]["state_id"] for value in invariant_local]
                development_id = stable_id(
                    "character-development",
                    self.movie_id,
                    character["character_id"],
                    dimension,
                    target,
                    operation,
                    *before_ids,
                    *resulting_ids,
                    effective,
                )
                output.append(
                    {
                        "development_id": development_id,
                        "character_id": character["character_id"],
                        "dimension": dimension,
                        "target_id_or_text": target,
                        "operation": operation,
                        "before_state_ids": before_ids,
                        "catalyst_event_ids": catalyst_ids,
                        "resulting_state_ids": resulting_ids,
                        "downstream_consequence_ids": consequence_ids,
                        "invariant_state_ids": invariant_ids,
                        **evidence_fields,
                        "effective_from_scene": effective,
                        "consequence_visible_from_scene": consequence_visible,
                        "validation_status": "silver_candidate",
                    }
                )
            return output

        stage_suffix = f"{group_index:04d}"
        if partition_index is not None:
            stage_suffix = f"{stage_suffix}:{partition_index:04d}"
        developments, metadata = await self._semantic_call(
            system_prompt=DEVELOPMENT_SYSTEM,
            user_prompt=user_prompt,
            stage=(
                f"temporal_development:{character['character_id']}:"
                f"{stage_suffix}"
            ),
            normalize=normalize,
        )
        return developments, {
            **metadata,
            "deterministic_payload_corrections": list(normalization_corrections),
        }

    async def build_epistemic_ledger(
        self,
        *,
        registry: dict[str, Any],
        evidence_bank: dict[str, Any],
        index: GraphIndex,
    ) -> dict[str, Any]:
        selected = {
            item["character_id"]: item
            for item in registry["characters"]
            if item["construction_selected"]
        }
        evidence = evidence_bank["evidence_units"]
        fact_assignment = _assign_facts_to_episodes(index)
        jobs = []
        for episode in index.episodes:
            scene_ids = set(index.node_scene_ids(episode))
            scene_evidence = [item for item in evidence if item["scene_id"] in scene_ids]
            episode_facts = [
                fact
                for fact in fact_assignment.get(episode["id"], [])
                if (clean_text(fact.get("modality")) or "asserted") == "asserted"
            ]
            if not episode_facts:
                continue
            episode_evidence = _fact_aligned_evidence(
                facts=episode_facts,
                evidence=scene_evidence,
            )
            relevant_characters = [
                character
                for character in selected.values()
                if _character_relevant_in_evidence(character["character_id"], episode_evidence)
                or _character_participates_in_episode(character, episode)
            ]
            known_characters = [
                item
                for item in registry["characters"]
                if set(item["scene_ids"]) & scene_ids
            ]
            for character in relevant_characters:
                jobs.append(
                    self._classify_epistemic_access(
                        character=character,
                        characters=known_characters,
                        episode=episode,
                        evidence=episode_evidence,
                        facts=episode_facts,
                        index=index,
                    )
                )
        results = await _gather_all_settled(jobs, label="epistemic extraction")
        records = [item for result, _ in results for item in result]
        records.sort(
            key=lambda item: (
                item["character_id"],
                int(item["acquired_at_scene"] or 10**9),
                item["access_id"],
            )
        )
        return {
            "schema_version": "stage_epistemic_ledger_v1",
            "movie_id": self.movie_id,
            "access_records": records,
            "audit": {"llm_calls": [meta for _, meta in results]},
        }

    async def _classify_epistemic_access(
        self,
        *,
        character: dict[str, Any],
        characters: list[dict[str, Any]],
        episode: dict[str, Any],
        evidence: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        index: GraphIndex,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local_characters = {
            f"C{position:03d}": item for position, item in enumerate(characters, start=1)
        }
        character_id_to_local = {
            item["character_id"]: local_id
            for local_id, item in local_characters.items()
        }
        local_evidence = {
            f"W{position:04d}": item for position, item in enumerate(evidence, start=1)
        }
        local_facts = {f"F{position:04d}": item for position, item in enumerate(facts, start=1)}
        prompt_characters = [
            {"local_character_id": local_id, "name": item["canonical_name"]}
            for local_id, item in local_characters.items()
        ]
        prompt_evidence = [
            {
                "local_evidence_id": local_id,
                "scene_order": item["scene_order"],
                "text": item["evidence_text"],
                "speaker": character_id_to_local.get(item["speaker_character_id"], ""),
                "participants": [
                    character_id_to_local[value]
                    for value in item["participant_character_ids"]
                    if value in character_id_to_local
                ],
                "observers": [
                    character_id_to_local[value]
                    for value in item["direct_observer_character_ids"]
                    if value in character_id_to_local
                ],
                "addressees": [
                    character_id_to_local[value]
                    for value in item["addressee_character_ids"]
                    if value in character_id_to_local
                ],
            }
            for local_id, item in local_evidence.items()
        ]
        prompt_facts = [
            {
                "local_fact_id": local_id,
                "type": item["node_type"],
                "text": index.fact_text(item),
                "scene_orders": index.node_scene_orders(item),
            }
            for local_id, item in local_facts.items()
        ]
        user_prompt = EPISTEMIC_USER.format(
            character=json.dumps(_character_prompt(character), ensure_ascii=False),
            characters=json.dumps(prompt_characters, ensure_ascii=False, indent=2),
            evidence=json.dumps(prompt_evidence, ensure_ascii=False, indent=2),
            facts=json.dumps(prompt_facts, ensure_ascii=False, indent=2),
        )
        allowed_scene_orders = set(index.node_scene_orders(episode))
        normalization_corrections: list[dict[str, Any]] = []

        def normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
            _exact_object(payload, {"decisions"}, "epistemic classification")
            decisions = _array(payload, "decisions")
            by_fact: dict[str, dict[str, Any]] = {}
            for position, raw in enumerate(decisions, start=1):
                if not isinstance(raw, dict):
                    normalization_corrections.append(
                        {
                            "action": "drop_non_object_decision",
                            "decision_position": position,
                        }
                    )
                    continue
                fact_local = clean_text(raw.get("fact_id"))
                if fact_local not in local_facts or fact_local in by_fact:
                    normalization_corrections.append(
                        {
                            "action": "drop_unknown_or_duplicate_fact_decision",
                            "decision_position": position,
                            "fact_id": fact_local,
                        }
                    )
                    continue
                access_type = clean_text(raw.get("access_type")).casefold()
                if access_type not in ACCESS_TYPES:
                    normalization_corrections.append(
                        {
                            "action": "downgrade_unsupported_access_to_unknown",
                            "fact_id": fact_local,
                            "requested_access_type": access_type,
                        }
                    )
                    access_type = "unknown"
                try:
                    acquired_raw = int(raw.get("acquired_at_scene", 0))
                except (TypeError, ValueError):
                    acquired_raw = 0
                acquired = acquired_raw or None
                if access_type == "unknown" and acquired is not None:
                    normalization_corrections.append(
                        {
                            "action": "clear_unknown_acquisition_scene",
                            "fact_id": fact_local,
                            "requested_scene": acquired,
                        }
                    )
                    acquired = None
                if access_type != "unknown" and acquired not in allowed_scene_orders:
                    source_orders = [
                        value
                        for value in index.node_scene_orders(local_facts[fact_local])
                        if value in allowed_scene_orders
                    ]
                    if source_orders:
                        normalized_scene = min(source_orders)
                        normalization_corrections.append(
                            {
                                "action": "align_acquisition_to_fact_scene",
                                "fact_id": fact_local,
                                "requested_scene": acquired,
                                "resolved_scene": normalized_scene,
                            }
                        )
                        acquired = normalized_scene
                    else:
                        normalization_corrections.append(
                            {
                                "action": "downgrade_unlocated_access_to_unknown",
                                "fact_id": fact_local,
                            }
                        )
                        access_type = "unknown"
                        acquired = None
                source_local = clean_text(raw.get("source_character_id"))
                if source_local and source_local not in local_characters:
                    normalization_corrections.append(
                        {
                            "action": "clear_unknown_source_character",
                            "fact_id": fact_local,
                            "source_character_id": source_local,
                        }
                    )
                    source_local = ""
                requested_evidence = raw.get("supporting_evidence_ids")
                evidence_local = unique_text(
                    value
                    for value in (
                        requested_evidence if isinstance(requested_evidence, list) else []
                    )
                    if clean_text(value) in local_evidence
                )
                if access_type != "unknown" and not evidence_local:
                    normalization_corrections.append(
                        {
                            "action": "downgrade_ungrounded_access_to_unknown",
                            "fact_id": fact_local,
                        }
                    )
                    access_type = "unknown"
                    acquired = None
                    source_local = ""
                fact = local_facts[fact_local]
                fact_id = fact["id"]
                try:
                    certainty = _probability(raw.get("certainty"), "epistemic certainty")
                except (TypeError, ValueError):
                    certainty = 0.0 if access_type == "unknown" else 0.5
                    normalization_corrections.append(
                        {
                            "action": "normalize_invalid_certainty",
                            "fact_id": fact_local,
                            "resolved_certainty": certainty,
                        }
                    )
                record = {
                    "access_id": stable_id(
                        "epistemic-access",
                        self.movie_id,
                        character["character_id"],
                        fact_id,
                    ),
                    "character_id": character["character_id"],
                    "fact_or_event_id": fact_id,
                    "fact_source_scene_orders": index.node_scene_orders(fact),
                    "access_type": access_type,
                    "acquired_at_scene": acquired,
                    "source_character_id": (
                        local_characters[source_local]["character_id"]
                        if source_local
                        else ""
                    ),
                    "certainty": certainty,
                    "superseded_at_scene": None,
                    "supporting_evidence_ids": [
                        local_evidence[value]["evidence_id"] for value in evidence_local
                    ],
                    "generated_rationale_hint": clean_text(
                        raw.get("generated_rationale_hint")
                    ),
                    "validation_status": "silver_candidate",
                }
                by_fact[fact_local] = record
            for fact_local, fact in local_facts.items():
                if fact_local in by_fact:
                    continue
                fact_id = fact["id"]
                normalization_corrections.append(
                    {
                        "action": "add_conservative_unknown_for_missing_fact",
                        "fact_id": fact_local,
                    }
                )
                by_fact[fact_local] = {
                    "access_id": stable_id(
                        "epistemic-access",
                        self.movie_id,
                        character["character_id"],
                        fact_id,
                    ),
                    "character_id": character["character_id"],
                    "fact_or_event_id": fact_id,
                    "fact_source_scene_orders": index.node_scene_orders(fact),
                    "access_type": "unknown",
                    "acquired_at_scene": None,
                    "source_character_id": "",
                    "certainty": 0.0,
                    "superseded_at_scene": None,
                    "supporting_evidence_ids": [],
                    "generated_rationale_hint": "",
                    "validation_status": "silver_candidate",
                }
            return [by_fact[local_id] for local_id in local_facts]

        records, metadata = await self._semantic_call(
            system_prompt=EPISTEMIC_SYSTEM,
            user_prompt=user_prompt,
            stage=f"temporal_epistemic:{character['character_id']}:{episode['id']}",
            normalize=normalize,
        )
        return records, {
            **metadata,
            "deterministic_payload_corrections": list(normalization_corrections),
        }

    async def build_persona_evidence_bank(
        self,
        *,
        registry: dict[str, Any],
        evidence_bank: dict[str, Any],
        index: GraphIndex,
    ) -> dict[str, Any]:
        evidence = evidence_bank["evidence_units"]
        jobs = []
        for character in registry["characters"]:
            if not character["construction_selected"]:
                continue
            for episode in index.episodes:
                scene_ids = set(index.node_scene_ids(episode))
                relevant = [
                    item
                    for item in evidence
                    if item["scene_id"] in scene_ids
                    and _character_relevant_in_evidence(character["character_id"], [item])
                ]
                if relevant:
                    partitions = _partition_persona_evidence(
                        character=character,
                        episode=episode,
                        evidence=relevant,
                        token_counter=self.token_counter,
                        max_input_tokens=self.max_input_tokens,
                        language=self.config.language,
                    )
                    for partition_index, partition in enumerate(partitions, start=1):
                        jobs.append(
                            self._extract_persona_evidence(
                                character=character,
                                episode=episode,
                                evidence=partition,
                                partition_index=(
                                    partition_index if len(partitions) > 1 else None
                                ),
                            )
                        )
        results = await _gather_all_settled(jobs, label="persona extraction")
        raw_items = [item for result, _ in results for item in result]
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in raw_items:
            key = (
                item["character_id"],
                item["evidence_kind"],
                clean_text(item["value"]).casefold(),
            )
            if key not in merged:
                merged[key] = item
                continue
            existing = merged[key]
            existing["supporting_evidence_ids"] = unique_text(
                [*existing["supporting_evidence_ids"], *item["supporting_evidence_ids"]]
            )
            existing["established_from_scene"] = min(
                existing["established_from_scene"], item["established_from_scene"]
            )
            if existing["stability"] != item["stability"]:
                existing["stability"] = "phase_specific"
        items = sorted(
            merged.values(),
            key=lambda item: (
                item["character_id"],
                int(item["established_from_scene"]),
                item["persona_evidence_id"],
            ),
        )
        return {
            "schema_version": "stage_persona_evidence_bank_v1",
            "movie_id": self.movie_id,
            "persona_evidence": items,
            "audit": {"llm_calls": [meta for _, meta in results]},
        }

    async def _extract_persona_evidence(
        self,
        *,
        character: dict[str, Any],
        episode: dict[str, Any],
        evidence: list[dict[str, Any]],
        partition_index: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local = {f"W{position:04d}": item for position, item in enumerate(evidence, start=1)}
        prompt_evidence = [
            {
                "local_evidence_id": local_id,
                "scene_order": item["scene_order"],
                "type": item["evidence_type"],
                "speaker_is_target": (
                    item["speaker_character_id"] == character["character_id"]
                ),
                "text": item["evidence_text"],
            }
            for local_id, item in local.items()
        ]
        user_prompt = PERSONA_USER.format(
            language=self.config.language,
            character=json.dumps(_character_prompt(character), ensure_ascii=False),
            episode=json.dumps(_persona_episode_prompt(episode), ensure_ascii=False),
            evidence=json.dumps(prompt_evidence, ensure_ascii=False, indent=2),
        )
        episode_orders = {item["scene_order"] for item in evidence}
        normalization_corrections: list[dict[str, Any]] = []

        def normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
            _exact_object(payload, {"persona_evidence"}, "persona extraction")
            output = []
            for position, raw in enumerate(
                _array(payload, "persona_evidence"), start=1
            ):
                if not isinstance(raw, dict):
                    normalization_corrections.append(
                        {
                            "action": "drop_non_object_persona_candidate",
                            "candidate_position": position,
                        }
                    )
                    continue
                kind = clean_text(raw.get("evidence_kind")).casefold()
                stability = clean_text(raw.get("stability")).casefold()
                value = clean_text(raw.get("value"))
                if (
                    kind not in PERSONA_EVIDENCE_KINDS
                    or stability not in PERSONA_STABILITY
                    or not value
                ):
                    normalization_corrections.append(
                        {
                            "action": "drop_unsupported_or_empty_persona_candidate",
                            "candidate_position": position,
                            "kind": kind,
                            "stability": stability,
                        }
                    )
                    continue
                try:
                    established = int(raw.get("established_from_scene", 0))
                    superseded_raw = int(raw.get("superseded_at_scene", 0))
                except (TypeError, ValueError):
                    normalization_corrections.append(
                        {
                            "action": "drop_persona_candidate_with_invalid_scene",
                            "candidate_position": position,
                        }
                    )
                    continue
                superseded = superseded_raw or None
                if established not in episode_orders:
                    normalization_corrections.append(
                        {
                            "action": "drop_persona_candidate_outside_episode",
                            "candidate_position": position,
                            "established_from_scene": established,
                        }
                    )
                    continue
                if superseded is not None and superseded < established:
                    normalization_corrections.append(
                        {
                            "action": "clear_invalid_persona_supersession",
                            "candidate_position": position,
                            "superseded_at_scene": superseded,
                        }
                    )
                    superseded = None
                requested_ids = raw.get("supporting_evidence_ids")
                local_ids = unique_text(
                    local_id
                    for local_id in (
                        requested_ids if isinstance(requested_ids, list) else []
                    )
                    if clean_text(local_id) in local
                )
                if not local_ids:
                    normalization_corrections.append(
                        {
                            "action": "drop_ungrounded_persona_candidate",
                            "candidate_position": position,
                        }
                    )
                    continue
                selected = [local[local_id] for local_id in local_ids]
                if kind == "speaking_style" and not any(
                    item["evidence_type"] == "dialogue"
                    and item["speaker_character_id"] == character["character_id"]
                    for item in selected
                ):
                    normalization_corrections.append(
                        {
                            "action": "drop_speaking_style_without_target_dialogue",
                            "candidate_position": position,
                        }
                    )
                    continue
                persona_id = stable_id(
                    "persona-evidence",
                    self.movie_id,
                    character["character_id"],
                    kind,
                    value,
                    established,
                )
                output.append(
                    {
                        "persona_evidence_id": persona_id,
                        "character_id": character["character_id"],
                        "evidence_kind": kind,
                        "value": value,
                        "established_from_scene": established,
                        "superseded_at_scene": superseded,
                        "supporting_evidence_ids": [item["evidence_id"] for item in selected],
                        "stability": stability,
                        "validation_status": "silver_candidate",
                    }
                )
            return output

        stage = f"temporal_persona:{character['character_id']}:{episode['id']}"
        if partition_index is not None:
            stage += f":{partition_index:04d}"
        records, metadata = await self._semantic_call(
            system_prompt=PERSONA_SYSTEM,
            user_prompt=user_prompt,
            stage=stage,
            normalize=normalize,
        )
        return records, {
            **metadata,
            "deterministic_payload_corrections": list(normalization_corrections),
        }

    def _extend_temporal_character_spans(
        self,
        *,
        registry: dict[str, Any],
        states: dict[str, list[dict[str, Any]]],
        developments: dict[str, list[dict[str, Any]]],
        accesses: dict[str, list[dict[str, Any]]],
        persona: dict[str, list[dict[str, Any]]],
        evidence_bank: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extend KG-derived spans only for linked, canonically grounded evidence."""
        evidence_by_id = {
            item["evidence_id"]: item for item in evidence_bank["evidence_units"]
        }
        temporal_span_corrections: list[dict[str, Any]] = []
        for character in registry["characters"]:
            character_id = character["character_id"]
            linked_evidence_ids: set[str] = set()
            linked_development_rows: list[dict[str, Any]] = list(
                developments[character_id]
            )
            for item in (
                *states[character_id],
                *developments[character_id],
                *accesses[character_id],
                *persona[character_id],
            ):
                for key in (
                    "supporting_evidence_ids",
                    "evidence_before_ids",
                    "evidence_catalyst_ids",
                    "evidence_after_ids",
                ):
                    values = item.get(key, [])
                    if isinstance(values, list):
                        linked_evidence_ids.update(
                            value for value in values if value in evidence_by_id
                        )
            aliases = []
            for raw_alias in unique_text(
                [character.get("canonical_name"), *(character.get("aliases") or [])]
            ):
                normalized_alias = normalize_name(raw_alias)
                has_cjk = any("\u4e00" <= char <= "\u9fff" for char in raw_alias)
                if len(normalized_alias) >= (2 if has_cjk else 3):
                    aliases.append(normalized_alias)
            current_first = int(character.get("first_scene_order") or 0)
            current_last = int(character.get("last_scene_order") or 0)
            candidate_rows: list[dict[str, Any]] = []
            # A development is already owned by this canonical character. If it
            # has grounded evidence and falls outside the KG-derived span, the
            # span must include its scene so checkpoint construction can keep a
            # valid baseline -> change -> final chain. This also handles visual
            # introductions that precede the first named/dialogue mention.
            for development in linked_development_rows:
                scene_order = int(development.get("effective_from_scene") or 0)
                evidence_ids = unique_text(
                    value
                    for key in (
                        "evidence_before_ids",
                        "evidence_catalyst_ids",
                        "evidence_after_ids",
                        "supporting_evidence_ids",
                    )
                    for value in (development.get(key) or [])
                )
                if (
                    scene_order
                    and (scene_order < current_first or scene_order > current_last)
                    and any(value in evidence_by_id for value in evidence_ids)
                ):
                    candidate_rows.append(
                        {
                            "development_id": development.get("development_id", ""),
                            "scene_order": scene_order,
                            "match": "character_owned_development",
                        }
                    )
            for evidence_id in sorted(linked_evidence_ids):
                evidence = evidence_by_id[evidence_id]
                scene_order = int(evidence.get("scene_order") or 0)
                if current_first <= scene_order <= current_last:
                    continue
                referenced_ids = {
                    value
                    for key in (
                        "speaker_character_id",
                        "participant_character_ids",
                        "direct_observer_character_ids",
                        "addressee_character_ids",
                    )
                    for value in (
                        evidence.get(key, [])
                        if isinstance(evidence.get(key), list)
                        else [evidence.get(key)]
                    )
                    if clean_text(value)
                }
                explicit_match = character_id in referenced_ids
                evidence_text = normalize_name(evidence.get("evidence_text"))
                lexical_match = bool(
                    evidence_text
                    and any(alias in evidence_text for alias in aliases)
                )
                if not (explicit_match or lexical_match):
                    continue
                candidate_rows.append(
                    {
                        "evidence_id": evidence_id,
                        "scene_id": evidence.get("scene_id", ""),
                        "scene_order": scene_order,
                        "match": "canonical_id" if explicit_match else "canonical_alias",
                    }
                )
            if not candidate_rows:
                continue
            new_first = min(
                current_first,
                *(int(item["scene_order"]) for item in candidate_rows),
            )
            new_last = max(
                current_last,
                *(int(item["scene_order"]) for item in candidate_rows),
            )
            if new_first >= current_first and new_last <= current_last:
                continue
            previous_first = current_first
            character["last_scene_order"] = new_last
            character["first_scene_order"] = new_first
            correction = {
                "character_id": character_id,
                "canonical_name": character.get("canonical_name", ""),
                "previous_first_scene_order": previous_first,
                "new_first_scene_order": new_first,
                "previous_last_scene_order": current_last,
                "new_last_scene_order": new_last,
                "linked_evidence": candidate_rows,
                "policy": (
                    "temporal_asset_linked_canonical_span_extension_v2"
                    if new_first < previous_first
                    else "temporal_asset_linked_canonical_span_extension_v1"
                ),
            }
            character["temporal_span_correction"] = correction
            temporal_span_corrections.append(correction)
        audit = registry.setdefault("audit", {})
        existing_corrections = list(audit.get("temporal_span_corrections") or [])
        known_corrections = {
            (
                item.get("character_id"),
                item.get("previous_first_scene_order"),
                item.get("new_first_scene_order"),
                item.get("previous_last_scene_order"),
                item.get("new_last_scene_order"),
            )
            for item in existing_corrections
        }
        for correction in temporal_span_corrections:
            key = (
                correction.get("character_id"),
                correction.get("previous_first_scene_order"),
                correction.get("new_first_scene_order"),
                correction.get("previous_last_scene_order"),
                correction.get("new_last_scene_order"),
            )
            if key not in known_corrections:
                existing_corrections.append(correction)
                known_corrections.add(key)
        audit["temporal_span_corrections"] = existing_corrections
        return temporal_span_corrections

    def apply_eligibility(
        self,
        *,
        registry: dict[str, Any],
        state_ledger: dict[str, Any],
        development_graph: dict[str, Any],
        epistemic_ledger: dict[str, Any],
        persona_bank: dict[str, Any],
        evidence_bank: dict[str, Any],
    ) -> dict[str, Any]:
        states: dict[str, list[dict[str, Any]]] = defaultdict(list)
        developments: dict[str, list[dict[str, Any]]] = defaultdict(list)
        accesses: dict[str, list[dict[str, Any]]] = defaultdict(list)
        persona: dict[str, list[dict[str, Any]]] = defaultdict(list)
        dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in state_ledger["states"]:
            states[item["character_id"]].append(item)
        for item in development_graph["developments"]:
            developments[item["character_id"]].append(item)
        for item in epistemic_ledger["access_records"]:
            accesses[item["character_id"]].append(item)
        for item in persona_bank["persona_evidence"]:
            persona[item["character_id"]].append(item)
        for item in evidence_bank["evidence_units"]:
            if item["evidence_type"] == "dialogue" and item["speaker_character_id"]:
                dialogue[item["speaker_character_id"]].append(item)

        temporal_span_corrections = self._extend_temporal_character_spans(
            registry=registry,
            states=states,
            developments=developments,
            accesses=accesses,
            persona=persona,
            evidence_bank=evidence_bank,
        )
        for character in registry["characters"]:
            character_id = character["character_id"]
            task1_reasons = []
            durable = [item for item in states[character_id] if item["durability"] == "durable"]
            dimensions = {item["dimension"] for item in durable}
            if len(durable) < self.config.task1_min_durable_states:
                task1_reasons.append("insufficient_durable_states")
            if len(developments[character_id]) < self.config.task1_min_developments:
                task1_reasons.append("insufficient_developments")
            if len(dimensions) < self.config.task1_min_dimensions:
                task1_reasons.append("insufficient_state_dimensions")
            if not character["construction_selected"]:
                task1_reasons.append("not_selected_for_construction")
            character["task1_exclusion_reasons"] = task1_reasons
            character["task1_eligible"] = not task1_reasons

            task3_reasons = []
            accessible = [item for item in accesses[character_id] if item["access_type"] != "unknown"]
            if len(dialogue[character_id]) < self.config.task3_min_dialogue_evidence:
                task3_reasons.append("insufficient_dialogue_evidence")
            if len(persona[character_id]) < self.config.task3_min_persona_evidence:
                task3_reasons.append("insufficient_persona_evidence")
            if len(accessible) < self.config.task3_min_accessible_facts:
                task3_reasons.append("insufficient_accessible_facts")
            if not character["construction_selected"]:
                task3_reasons.append("not_selected_for_construction")
            character["task3_exclusion_reasons"] = task3_reasons
            character["task3_single_turn_eligible"] = not task3_reasons
        registry["audit"]["task1_eligible_count"] = sum(
            bool(item["task1_eligible"]) for item in registry["characters"]
        )
        registry["audit"]["task3_single_turn_eligible_count"] = sum(
            bool(item["task3_single_turn_eligible"]) for item in registry["characters"]
        )
        return registry

    async def build_checkpoint_manifest(
        self,
        *,
        scenes: list[Scene],
        registry: dict[str, Any],
        state_ledger: dict[str, Any],
        development_graph: dict[str, Any],
        epistemic_ledger: dict[str, Any],
        persona_bank: dict[str, Any],
        evidence_bank: dict[str, Any],
    ) -> dict[str, Any]:
        states_by_character = _group(state_ledger["states"], "character_id")
        developments_by_character = _group(
            development_graph["developments"], "character_id"
        )
        access_by_character = _group(epistemic_ledger["access_records"], "character_id")
        persona_by_character = _group(persona_bank["persona_evidence"], "character_id")
        self._extend_temporal_character_spans(
            registry=registry,
            states=states_by_character,
            developments=developments_by_character,
            accesses=access_by_character,
            persona=persona_by_character,
            evidence_bank=evidence_bank,
        )
        dialogue_by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in evidence_bank["evidence_units"]:
            if item["evidence_type"] == "dialogue" and item["speaker_character_id"]:
                dialogue_by_character[item["speaker_character_id"]].append(item)
        jobs = []
        for character in registry["characters"]:
            if not (character["task1_eligible"] or character["task3_single_turn_eligible"]):
                continue
            jobs.append(
                self._select_checkpoints(
                    character=character,
                    scenes=scenes,
                    states=states_by_character[character["character_id"]],
                    developments=developments_by_character[character["character_id"]],
                    access_records=access_by_character[character["character_id"]],
                    persona=persona_by_character[character["character_id"]],
                    dialogue=dialogue_by_character[character["character_id"]],
                )
            )
        results = await _gather_all_settled(jobs, label="checkpoint selection")
        checkpoints = [item for result, _ in results for item in result]
        checkpoints.sort(
            key=lambda item: (
                item["character_id"],
                int(item["scene_order"]),
                item["checkpoint_id"],
            )
        )
        return {
            "schema_version": "stage_checkpoint_manifest_v1",
            "movie_id": self.movie_id,
            "checkpoints": checkpoints,
            "audit": {"llm_calls": [meta for _, meta in results]},
        }

    async def _select_checkpoints(
        self,
        *,
        character: dict[str, Any],
        scenes: list[Scene],
        states: list[dict[str, Any]],
        developments: list[dict[str, Any]],
        access_records: list[dict[str, Any]],
        persona: list[dict[str, Any]],
        dialogue: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        scene_by_order = {scene.order: scene for scene in scenes}
        if not scene_by_order:
            raise ValueError("Checkpoint materialization requires screenplay scenes")
        first_observed_orders = [
            int(character.get("first_scene_order") or min(scene_by_order)),
            *(int(item["scene_order"]) for item in dialogue),
        ]
        baseline_order = min(first_observed_orders)
        final_order = int(character.get("last_scene_order") or max(scene_by_order))
        if baseline_order not in scene_by_order:
            raise ValueError("Character baseline scene is outside the screenplay")
        if final_order not in scene_by_order or final_order < baseline_order:
            raise ValueError("Character final scene is outside its screenplay span")
        developments_by_scene: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for development in developments:
            scene_order = int(development["effective_from_scene"])
            if scene_order not in scene_by_order:
                raise ValueError("Development checkpoint is outside the screenplay")
            # A development grounded in the character's first observed scene is
            # part of the initial observed state. Keep it on the baseline
            # checkpoint so every validated development remains covered exactly
            # once, while later transitions remain change checkpoints.
            if scene_order > final_order:
                raise ValueError("Development is later than the character final checkpoint")
            developments_by_scene[scene_order].append(development)
        baseline_developments = developments_by_scene.pop(baseline_order, [])
        final_developments = developments_by_scene.pop(final_order, [])
        definitions = [
            {
                "scene_order": baseline_order,
                "checkpoint_type": "baseline",
                "developments": sorted(
                    baseline_developments, key=lambda item: item["development_id"]
                ),
            },
            *[
                {
                    "scene_order": scene_order,
                    "checkpoint_type": "change",
                    "developments": sorted(
                        values, key=lambda item: item["development_id"]
                    ),
                }
                for scene_order, values in sorted(developments_by_scene.items())
            ],
            {
                "scene_order": final_order,
                "checkpoint_type": "final",
                "developments": sorted(
                    final_developments, key=lambda item: item["development_id"]
                ),
            },
        ]
        output = []
        previous_id = ""
        for definition in definitions:
            scene_order = definition["scene_order"]
            checkpoint_id = stable_id(
                "checkpoint",
                self.movie_id,
                character["character_id"],
                scene_order,
                definition["checkpoint_type"],
            )
            accessible = []
            unknown = []
            future = []
            for access in access_records:
                source_order = min(access["fact_source_scene_orders"], default=10**9)
                acquired = access["acquired_at_scene"]
                if source_order > scene_order:
                    future.append(access["fact_or_event_id"])
                elif (
                    access["access_type"] != "unknown"
                    and acquired is not None
                    and acquired <= scene_order
                ):
                    accessible.append(access["fact_or_event_id"])
                else:
                    unknown.append(access["fact_or_event_id"])
            checkpoint_developments = definition["developments"]
            control_types = []
            if not checkpoint_developments:
                control_types.append("no_change")
            if any(
                access["access_type"] == "unknown"
                and min(access["fact_source_scene_orders"], default=10**9)
                <= scene_order
                for access in access_records
            ):
                control_types.append("inaccessible")
            if any(
                int(item["consequence_visible_from_scene"]) > scene_order
                for item in checkpoint_developments
            ):
                control_types.append("delayed_consequence")
            output.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "movie_id": self.movie_id,
                    "character_id": character["character_id"],
                    "scene_id": scene_by_order[scene_order].scene_id,
                    "scene_order": scene_order,
                    "previous_checkpoint_id": previous_id,
                    "checkpoint_type": definition["checkpoint_type"],
                    "control_types": control_types,
                    "active_state_ids": [
                        state["state_id"]
                        for state in states
                        if _state_active(state, scene_order)
                    ],
                    "new_development_ids": [
                        item["development_id"] for item in checkpoint_developments
                    ],
                    "invariant_state_ids": unique_text(
                        state_id
                        for development in checkpoint_developments
                        for state_id in development["invariant_state_ids"]
                    ),
                    "accessible_fact_ids": unique_text(accessible),
                    "unknown_fact_ids": unique_text(unknown),
                    "future_forbidden_fact_ids": unique_text(future),
                    "persona_evidence_ids": [
                        item["persona_evidence_id"]
                        for item in persona
                        if item["established_from_scene"] <= scene_order
                        and (
                            item["superseded_at_scene"] is None
                            or item["superseded_at_scene"] >= scene_order
                        )
                    ],
                    "dialogue_exemplar_ids": [
                        item["evidence_id"]
                        for item in dialogue
                        if item["scene_order"] <= scene_order
                    ],
                    "selection_metadata": {
                        "policy": "character_span_plus_development_changes_v4",
                        "model_calls_added": 0,
                    },
                    "validation_status": "silver_candidate",
                }
            )
            previous_id = checkpoint_id
        return output, {
            "stage": f"temporal_checkpoint:{character['character_id']}",
            "call_kind": "deterministic_checkpoint_materialization",
            "selection_policy": "character_span_plus_development_changes_v5",
            "baseline_development_count": len(baseline_developments),
            "model_calls_added": 0,
            "checkpoint_count": len(output),
        }

    async def build_task3_prompt_candidates(
        self,
        *,
        registry: dict[str, Any],
        checkpoints: dict[str, Any],
        state_ledger: dict[str, Any],
        development_graph: dict[str, Any],
        epistemic_ledger: dict[str, Any],
        persona_bank: dict[str, Any],
        evidence_bank: dict[str, Any],
        index: GraphIndex,
    ) -> dict[str, Any]:
        checkpoint_by_character = _group(checkpoints["checkpoints"], "character_id")
        state_by_character = _group(state_ledger["states"], "character_id")
        access_by_character = _group(epistemic_ledger["access_records"], "character_id")
        persona_by_character = _group(persona_bank["persona_evidence"], "character_id")
        evidence_by_id = {item["evidence_id"]: item for item in evidence_bank["evidence_units"]}
        state_by_id = {item["state_id"]: item for item in state_ledger["states"]}
        development_by_id = {
            item["development_id"]: item
            for item in development_graph["developments"]
        }
        jobs = []
        for character in registry["characters"]:
            if not character["task3_single_turn_eligible"]:
                continue
            for checkpoint in checkpoint_by_character[character["character_id"]]:
                package = _task3_checkpoint_assets(
                    character_id=character["character_id"],
                    checkpoint=checkpoint,
                    states=state_by_character[character["character_id"]],
                    access_records=access_by_character[character["character_id"]],
                    persona=persona_by_character[character["character_id"]],
                    evidence_by_id=evidence_by_id,
                    state_by_id=state_by_id,
                    development_by_id=development_by_id,
                    index=index,
                )
                jobs.append(
                    self._construct_task3_prompts(
                        character=character,
                        checkpoints=[package["checkpoint"]],
                        states=package["states"],
                        access_records=package["access_records"],
                        persona=package["persona"],
                        evidence_by_id=package["evidence_by_id"],
                        index=index,
                    )
                )
        results = await _gather_all_settled(jobs, label="task3 prompt generation")
        prompts = [item for result, _ in results for item in result]
        prompts.sort(key=lambda item: (item["character_id"], item["checkpoint_id"], item["prompt_id"]))
        pairing_audit = assign_task3_longitudinal_pairs(
            prompts,
            state_by_id=state_by_id,
            access_by_id={
                item["access_id"]: item
                for item in epistemic_ledger["access_records"]
            },
            checkpoint_by_id={
                item["checkpoint_id"]: item for item in checkpoints["checkpoints"]
            },
        )
        return {
            "schema_version": "stage_task3_prompt_candidates_v1",
            "movie_id": self.movie_id,
            "prompts": prompts,
            "audit": {
                "llm_calls": [meta for _, meta in results],
                "longitudinal_pairing": pairing_audit,
            },
        }

    async def _construct_task3_prompts(
        self,
        *,
        character: dict[str, Any],
        checkpoints: list[dict[str, Any]],
        states: list[dict[str, Any]],
        access_records: list[dict[str, Any]],
        persona: list[dict[str, Any]],
        evidence_by_id: dict[str, dict[str, Any]],
        index: GraphIndex,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        local_checkpoints = {
            f"Q{position:03d}": item for position, item in enumerate(checkpoints, start=1)
        }
        local_states = {f"S{position:04d}": item for position, item in enumerate(states, start=1)}
        local_access = {
            f"A{position:04d}": item for position, item in enumerate(access_records, start=1)
        }
        fact_ids = unique_text(
            fact_id
            for checkpoint in checkpoints
            for fact_id in [
                *checkpoint["accessible_fact_ids"],
                *checkpoint["unknown_fact_ids"],
                *checkpoint["future_forbidden_fact_ids"],
            ]
        )
        local_facts = {
            f"F{position:04d}": index.nodes_by_id[fact_id]
            for position, fact_id in enumerate(fact_ids, start=1)
            if fact_id in index.nodes_by_id
        }
        fact_id_to_local = {item["id"]: key for key, item in local_facts.items()}
        state_id_to_local = {item["state_id"]: key for key, item in local_states.items()}
        access_id_to_local = {item["access_id"]: key for key, item in local_access.items()}
        local_persona = {
            f"P{position:04d}": item for position, item in enumerate(persona, start=1)
        }
        persona_id_to_local = {
            item["persona_evidence_id"]: key for key, item in local_persona.items()
        }
        evidence_ids = unique_text(
            [
                *(
                    evidence_id
                    for item in [*states, *persona, *access_records]
                    for evidence_id in item.get("supporting_evidence_ids", [])
                    if evidence_id in evidence_by_id
                ),
                *(
                    evidence_id
                    for checkpoint in checkpoints
                    for evidence_id in checkpoint["dialogue_exemplar_ids"]
                    if evidence_id in evidence_by_id
                ),
            ]
        )
        local_evidence = {
            f"W{position:04d}": evidence_by_id[evidence_id]
            for position, evidence_id in enumerate(evidence_ids, start=1)
        }
        evidence_id_to_local = {item["evidence_id"]: key for key, item in local_evidence.items()}
        packages = []
        for local_id, checkpoint in local_checkpoints.items():
            packages.append(
                {
                    "local_checkpoint_id": local_id,
                    "scene_order": checkpoint["scene_order"],
                    "active_states": [
                        {
                            "local_state_id": state_id_to_local[state_id],
                            **_state_prompt(
                                index_state,
                                evidence_to_local=evidence_id_to_local,
                            ),
                        }
                        for state_id in checkpoint["active_state_ids"]
                        if state_id in state_id_to_local
                        for index_state in [local_states[state_id_to_local[state_id]]]
                    ],
                    "visible_access": [
                        {
                            "local_access_id": access_id_to_local[item["access_id"]],
                            "local_fact_id": fact_id_to_local.get(item["fact_or_event_id"], ""),
                            "fact_text": index.fact_text(index.nodes_by_id[item["fact_or_event_id"]]),
                            "access_type": item["access_type"],
                        }
                        for item in access_records
                        if item["fact_or_event_id"] in checkpoint["accessible_fact_ids"]
                        and item["access_id"] in access_id_to_local
                        and item["fact_or_event_id"] in index.nodes_by_id
                    ],
                    "unknown_facts": [
                        {"local_fact_id": fact_id_to_local[fact_id], "text": index.fact_text(index.nodes_by_id[fact_id])}
                        for fact_id in checkpoint["unknown_fact_ids"]
                        if fact_id in fact_id_to_local and fact_id in index.nodes_by_id
                    ],
                    "future_facts": [
                        {"local_fact_id": fact_id_to_local[fact_id], "text": index.fact_text(index.nodes_by_id[fact_id])}
                        for fact_id in checkpoint["future_forbidden_fact_ids"]
                        if fact_id in fact_id_to_local and fact_id in index.nodes_by_id
                    ],
                    "persona_evidence": [
                        {
                            "local_persona_id": persona_id_to_local[persona_id],
                            "kind": local_persona[persona_id_to_local[persona_id]]["evidence_kind"],
                            "value": local_persona[persona_id_to_local[persona_id]]["value"],
                        }
                        for persona_id in checkpoint["persona_evidence_ids"]
                        if persona_id in persona_id_to_local
                    ],
                    "dialogue_exemplars": [
                        {
                            "local_evidence_id": evidence_id_to_local[evidence_id],
                            "text": evidence_by_id[evidence_id]["evidence_text"],
                        }
                        for evidence_id in checkpoint["dialogue_exemplar_ids"]
                        if evidence_id in evidence_id_to_local
                    ],
                }
            )
        character_json = json.dumps(_character_prompt(character), ensure_ascii=False)

        def render_prompt(values: list[dict[str, Any]]) -> str:
            return TASK3_PROMPT_USER.format(
                language=self.config.language,
                prompts_per_checkpoint=self.config.task3_prompts_per_checkpoint,
                character=character_json,
                checkpoint_packages=json.dumps(
                    values, ensure_ascii=False, indent=2
                ),
            )

        packages, package_budget_audit = _fit_task3_prompt_packages(
            packages,
            token_counter=self.token_counter,
            max_input_tokens=self.max_input_tokens,
            render=lambda values: TASK3_PROMPT_SYSTEM + render_prompt(values),
        )
        user_prompt = render_prompt(packages)
        normalization_corrections: list[dict[str, Any]] = []

        def normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
            normalization_corrections.clear()
            _exact_object(payload, {"prompts"}, "Task 3 prompt construction")
            output = []
            for position, raw in enumerate(_array(payload, "prompts"), start=1):
                _object(raw, "Task 3 prompt")
                checkpoint_local = clean_text(raw.get("checkpoint_id"))
                if checkpoint_local not in local_checkpoints:
                    raise ValueError("Task 3 prompt uses unknown checkpoint ID")
                requested_family = clean_text(raw.get("prompt_family")).casefold()
                family = normalize_task3_prompt_family(requested_family)
                requested_risk = clean_text(raw.get("boundary_risk_type")).casefold()
                risk = normalize_boundary_risk_type(requested_risk)
                if family != requested_family:
                    normalization_corrections.append(
                        {
                            "action": "normalize_state_dimension_as_task3_prompt_family",
                            "prompt_position": position,
                            "requested_family": requested_family,
                            "resolved_family": family,
                        }
                    )
                if risk != requested_risk:
                    normalization_corrections.append(
                        {
                            "action": "normalize_task3_boundary_risk_type",
                            "prompt_position": position,
                            "requested_risk": requested_risk,
                            "resolved_risk": risk,
                        }
                    )
                if family not in TASK3_PROMPT_FAMILIES or risk not in BOUNDARY_RISK_TYPES:
                    raise ValueError("Task 3 prompt family or boundary risk is unsupported")
                context = clean_text(raw.get("interaction_context"))
                user_turn = clean_text(raw.get("current_user_turn"))
                stances = unique_text(raw.get("expected_stances") or [])
                if not user_turn or not stances:
                    raise ValueError("Task 3 prompt requires user turn and expected stances")
                def grounded_ids(
                    value: Any, mapping: dict[str, Any], field: str
                ) -> list[str]:
                    requested = unique_text(value if isinstance(value, list) else [])
                    known = [local_id for local_id in requested if local_id in mapping]
                    dropped = [local_id for local_id in requested if local_id not in mapping]
                    if dropped:
                        normalization_corrections.append(
                            {
                                "action": "drop_unknown_task3_local_ids",
                                "prompt_position": position,
                                "field": field,
                                "dropped_ids": dropped,
                            }
                        )
                    return known

                state_local = grounded_ids(raw.get("state_ids"), local_states, "state_ids")
                access_local = grounded_ids(
                    raw.get("required_access_ids"), local_access, "required_access_ids"
                )
                support_local = grounded_ids(
                    raw.get("supporting_evidence_ids"),
                    local_evidence,
                    "supporting_evidence_ids",
                )
                contradiction_local = grounded_ids(
                    raw.get("contradicting_fact_ids"),
                    local_facts,
                    "contradicting_fact_ids",
                )
                unknown_local = grounded_ids(
                    raw.get("unknown_fact_ids"), local_facts, "unknown_fact_ids"
                )
                future_local = grounded_ids(
                    raw.get("future_forbidden_fact_ids"),
                    local_facts,
                    "future_forbidden_fact_ids",
                )
                style_local = grounded_ids(
                    raw.get("style_evidence_ids"),
                    local_persona,
                    "style_evidence_ids",
                )
                checkpoint = local_checkpoints[checkpoint_local]
                checkpoint_local_fields = (
                    ("state_ids", state_local, local_states),
                    ("required_access_ids", access_local, local_access),
                    ("supporting_evidence_ids", support_local, local_evidence),
                    ("contradicting_fact_ids", contradiction_local, local_facts),
                    ("style_evidence_ids", style_local, local_persona),
                )
                restricted: dict[str, list[str]] = {}
                for field, local_ids, mapping in checkpoint_local_fields:
                    visible = _task3_checkpoint_visible_local_ids(
                        field=field,
                        local_ids=local_ids,
                        mapping=mapping,
                        checkpoint=checkpoint,
                    )
                    dropped = [local_id for local_id in local_ids if local_id not in visible]
                    if dropped:
                        normalization_corrections.append(
                            {
                                "action": "drop_task3_references_not_visible_at_checkpoint",
                                "prompt_position": position,
                                "checkpoint_id": checkpoint["checkpoint_id"],
                                "field": field,
                                "dropped_ids": dropped,
                            }
                        )
                    restricted[field] = visible
                state_local = restricted["state_ids"]
                access_local = restricted["required_access_ids"]
                support_local = restricted["supporting_evidence_ids"]
                contradiction_local = restricted["contradicting_fact_ids"]
                style_local = restricted["style_evidence_ids"]
                if not any(
                    (
                        state_local,
                        access_local,
                        support_local,
                        contradiction_local,
                        unknown_local,
                        future_local,
                        style_local,
                    )
                ):
                    # Keep a semantically usable prompt when the one permitted
                    # model call omits every reference field. Select an asset
                    # that is valid at this checkpoint; this is normalization,
                    # not a second semantic call.
                    active_state_local = [
                        local_id
                        for local_id, state in local_states.items()
                        if state.get("state_id") in checkpoint["active_state_ids"]
                    ]
                    accessible_fact_ids = set(checkpoint["accessible_fact_ids"])
                    accessible_access_local = [
                        local_id
                        for local_id, access in local_access.items()
                        if access.get("fact_or_event_id") in accessible_fact_ids
                    ]
                    unknown_fact_local = [
                        fact_id_to_local[fact_id]
                        for fact_id in checkpoint["unknown_fact_ids"]
                        if fact_id in fact_id_to_local
                    ]
                    future_fact_local = [
                        fact_id_to_local[fact_id]
                        for fact_id in checkpoint["future_forbidden_fact_ids"]
                        if fact_id in fact_id_to_local
                    ]
                    if active_state_local:
                        state_local = active_state_local[:1]
                        fallback_field = "state_ids"
                    elif accessible_access_local:
                        access_local = accessible_access_local[:1]
                        fallback_field = "required_access_ids"
                    elif unknown_fact_local:
                        unknown_local = unknown_fact_local[:1]
                        fallback_field = "unknown_fact_ids"
                    elif future_fact_local:
                        future_local = future_fact_local[:1]
                        fallback_field = "future_forbidden_fact_ids"
                    elif local_evidence:
                        support_local = [next(iter(local_evidence))]
                        fallback_field = "supporting_evidence_ids"
                    elif local_persona:
                        style_local = [next(iter(local_persona))]
                        fallback_field = "style_evidence_ids"
                    else:
                        fallback_field = "none"
                    normalization_corrections.append(
                        {
                            "action": "add_checkpoint_grounded_anchor_when_model_omits_references",
                            "prompt_position": position,
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "field": fallback_field,
                        }
                    )
                if not any(
                    (
                        state_local,
                        access_local,
                        support_local,
                        contradiction_local,
                        unknown_local,
                        future_local,
                        style_local,
                    )
                ):
                    raise ValueError("Task 3 prompt requires at least one evaluator evidence anchor")
                unknown_ids = [local_facts[value]["id"] for value in unknown_local]
                future_ids = [local_facts[value]["id"] for value in future_local]
                valid_unknown_ids = set(checkpoint["unknown_fact_ids"])
                invalid_unknown_ids = [
                    fact_id for fact_id in unknown_ids if fact_id not in valid_unknown_ids
                ]
                if invalid_unknown_ids:
                    normalization_corrections.append(
                        {
                            "action": "drop_task3_unknown_facts_not_unknown_at_checkpoint",
                            "prompt_position": position,
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "dropped_fact_ids": invalid_unknown_ids,
                        }
                    )
                    unknown_ids = [
                        fact_id for fact_id in unknown_ids if fact_id in valid_unknown_ids
                    ]
                valid_future_ids = set(checkpoint["future_forbidden_fact_ids"])
                invalid_future_ids = [
                    fact_id for fact_id in future_ids if fact_id not in valid_future_ids
                ]
                if invalid_future_ids:
                    normalization_corrections.append(
                        {
                            "action": "drop_task3_future_facts_not_future_at_checkpoint",
                            "prompt_position": position,
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "dropped_fact_ids": invalid_future_ids,
                        }
                    )
                    future_ids = [
                        fact_id for fact_id in future_ids if fact_id in valid_future_ids
                    ]
                checkpoint_id = checkpoint["checkpoint_id"]
                prompt_id = stable_id(
                    "task3-prompt",
                    self.movie_id,
                    character["character_id"],
                    checkpoint_id,
                    position,
                    family,
                    user_turn,
                )
                pair_group = clean_text(raw.get("pair_group"))
                output.append(
                    {
                        "prompt_id": prompt_id,
                        "character_id": character["character_id"],
                        "checkpoint_id": checkpoint_id,
                        "prompt_family": family,
                        "interaction_context": context,
                        "current_user_turn": user_turn,
                        "expected_stances": stances,
                        "state_ids": [local_states[value]["state_id"] for value in state_local],
                        "required_access_ids": [local_access[value]["access_id"] for value in access_local],
                        "supporting_evidence_ids": [local_evidence[value]["evidence_id"] for value in support_local],
                        "contradicting_fact_ids": [local_facts[value]["id"] for value in contradiction_local],
                        "unknown_fact_ids": unknown_ids,
                        "future_forbidden_fact_ids": future_ids,
                        "style_evidence_ids": [local_persona[value]["persona_evidence_id"] for value in style_local],
                        "boundary_risk_type": risk,
                        "pair_group": pair_group,
                        "validation_status": "silver_candidate",
                    }
                )
            expected_count = self.config.task3_prompts_per_checkpoint
            by_checkpoint = _group(output, "checkpoint_id")
            selected_output: list[dict[str, Any]] = []
            for local_id, checkpoint in local_checkpoints.items():
                checkpoint_id = checkpoint["checkpoint_id"]
                candidates = by_checkpoint.get(checkpoint_id, [])
                if len(candidates) > expected_count:
                    selected = _select_task3_prompt_quota(
                        candidates, expected_count=expected_count
                    )
                    normalization_corrections.append(
                        {
                            "action": "select_exact_task3_prompt_quota",
                            "checkpoint_id": checkpoint_id,
                            "candidate_count": len(candidates),
                            "selected_count": len(selected),
                            "dropped_prompt_ids": [
                                item["prompt_id"]
                                for item in candidates
                                if item not in selected
                            ],
                        }
                    )
                    candidates = selected
                selected_output.extend(candidates)
            per_checkpoint = {
                local_id: len(
                    by_checkpoint.get(local_checkpoints[local_id]["checkpoint_id"], [])
                )
                for local_id in local_checkpoints
            }
            for local_id, checkpoint in local_checkpoints.items():
                checkpoint_id = checkpoint["checkpoint_id"]
                per_checkpoint[local_id] = sum(
                    item["checkpoint_id"] == checkpoint_id
                    for item in selected_output
                )
            mismatched_counts = {
                local_id: per_checkpoint[local_id]
                for local_id in local_checkpoints
                if per_checkpoint[local_id] != expected_count
            }
            if mismatched_counts:
                # A single model response can legally contain fewer prompts
                # than requested (including zero) even when all fields are
                # otherwise valid.  Fill only the missing quota from assets
                # already visible at this checkpoint; this is deterministic
                # materialization, not a second semantic call.
                is_chinese = clean_text(self.config.language).casefold() in {
                    "zh",
                    "chinese",
                    "中文",
                }

                def fallback_prompt(
                    *,
                    local_id: str,
                    checkpoint: dict[str, Any],
                    ordinal: int,
                    template: dict[str, Any] | None,
                ) -> dict[str, Any]:
                    checkpoint_id = checkpoint["checkpoint_id"]
                    if template is not None:
                        prompt = copy.deepcopy(template)
                        prompt["prompt_id"] = stable_id(
                            "task3-prompt-fallback",
                            self.movie_id,
                            character["character_id"],
                            checkpoint_id,
                            ordinal,
                        )
                        prompt["current_user_turn"] = (
                            "在当前情况下，你会如何回应？"
                            if is_chinese
                            else "How would you respond in this situation?"
                        )
                        prompt["pair_group"] = f"fallback-{checkpoint_id}"
                        return prompt

                    active_state_local = [
                        state_local_id
                        for state_local_id, state in local_states.items()
                        if state.get("state_id") in checkpoint["active_state_ids"]
                    ]
                    access_local = [
                        access_local_id
                        for access_local_id, access in local_access.items()
                        if access.get("fact_or_event_id")
                        in checkpoint["accessible_fact_ids"]
                    ]
                    unknown_local = [
                        fact_id_to_local[fact_id]
                        for fact_id in checkpoint["unknown_fact_ids"]
                        if fact_id in fact_id_to_local
                    ]
                    future_local = [
                        fact_id_to_local[fact_id]
                        for fact_id in checkpoint["future_forbidden_fact_ids"]
                        if fact_id in fact_id_to_local
                    ]
                    support_local = [
                        evidence_id_to_local[evidence_id]
                        for evidence_id in checkpoint["dialogue_exemplar_ids"]
                        if evidence_id in evidence_id_to_local
                    ]
                    style_local = [
                        persona_id_to_local[persona_id]
                        for persona_id in checkpoint["persona_evidence_ids"]
                        if persona_id in persona_id_to_local
                    ]
                    state_local = active_state_local[:1]
                    access_local = access_local[:1]
                    unknown_local = unknown_local[:1]
                    future_local = future_local[:1]
                    support_local = support_local[:1]
                    style_local = style_local[:1]
                    if unknown_local or future_local:
                        family = "knowledge_boundary_probe"
                        risk = "unknown_information" if unknown_local else "future_information"
                    elif state_local:
                        dimension = local_states[state_local[0]].get("dimension")
                        family = {
                            "relationship": "relationship_stance",
                            "goal_plan": "goal_decision_pressure",
                            "constraint_resource": "goal_decision_pressure",
                            "status_identity": "persona_invariant",
                        }.get(dimension, "memory_grounded_reflection")
                        risk = "none"
                    elif style_local:
                        family = "persona_invariant"
                        risk = "persona_inconsistency"
                    else:
                        family = "memory_grounded_reflection"
                        risk = "none"
                    if is_chinese:
                        turns = {
                            "knowledge_boundary_probe": "关于这件事，你现在知道多少？只说你此刻能确认的部分。",
                            "relationship_stance": "你现在如何看待这段关系？请说说你的立场。",
                            "goal_decision_pressure": "面对现在的处境，你准备怎么做？",
                            "persona_invariant": "在这件事上，你会坚持怎样的原则？",
                            "memory_grounded_reflection": "结合你目前经历的事情，你现在怎么看？",
                        }
                        stance = "回答应与当前可见证据一致，不使用后续信息，并保持角色视角。"
                        context = "围绕当前情境的自然追问"
                    else:
                        turns = {
                            "knowledge_boundary_probe": "What do you know about this right now? State only what you can confirm.",
                            "relationship_stance": "How do you see this relationship now? Explain your stance.",
                            "goal_decision_pressure": "Given the situation now, what are you going to do?",
                            "persona_invariant": "What principle would you stand by in this situation?",
                            "memory_grounded_reflection": "Looking at what you have experienced so far, how do you see this now?",
                        }
                        stance = "The answer should remain consistent with visible evidence, avoid future information, and stay in character."
                        context = "A natural follow-up in the current situation"
                    prompt = {
                        "prompt_id": stable_id(
                            "task3-prompt-fallback",
                            self.movie_id,
                            character["character_id"],
                            checkpoint_id,
                            ordinal,
                        ),
                        "character_id": character["character_id"],
                        "checkpoint_id": checkpoint_id,
                        "prompt_family": family,
                        "interaction_context": context,
                        "current_user_turn": turns[family],
                        "expected_stances": [stance],
                        "state_ids": [local_states[value]["state_id"] for value in state_local],
                        "required_access_ids": [local_access[value]["access_id"] for value in access_local],
                        "supporting_evidence_ids": [local_evidence[value]["evidence_id"] for value in support_local],
                        "contradicting_fact_ids": [],
                        "unknown_fact_ids": [local_facts[value]["id"] for value in unknown_local],
                        "future_forbidden_fact_ids": [local_facts[value]["id"] for value in future_local],
                        "style_evidence_ids": [local_persona[value]["persona_evidence_id"] for value in style_local],
                        "boundary_risk_type": risk,
                        "pair_group": f"fallback-{checkpoint_id}",
                        "validation_status": "silver_candidate",
                    }
                    return prompt

                for local_id, count in mismatched_counts.items():
                    checkpoint = local_checkpoints[local_id]
                    existing = [
                        item
                        for item in selected_output
                        if item["checkpoint_id"] == checkpoint["checkpoint_id"]
                    ]
                    missing = max(0, expected_count - count)
                    if missing:
                        template = existing[0] if existing else None
                        for ordinal in range(1, missing + 1):
                            selected_output.append(
                                fallback_prompt(
                                    local_id=local_id,
                                    checkpoint=checkpoint,
                                    ordinal=ordinal,
                                    template=template,
                                )
                            )
                        normalization_corrections.append(
                            {
                                "action": "fill_missing_task3_prompt_quota",
                                "checkpoint_id": checkpoint["checkpoint_id"],
                                "existing_count": count,
                                "added_count": missing,
                                "strategy": "reuse_checkpoint_candidate_or_visible_asset_anchor",
                            }
                        )
                selected_output.sort(key=lambda item: (item["checkpoint_id"], item["prompt_id"]))
                per_checkpoint = {
                    local_id: sum(
                        item["checkpoint_id"] == local_checkpoints[local_id]["checkpoint_id"]
                        for item in selected_output
                    )
                    for local_id in local_checkpoints
                }
                mismatched_counts = {
                    local_id: count
                    for local_id, count in per_checkpoint.items()
                    if count != expected_count
                }
                if mismatched_counts:
                    raise ValueError(
                        "Task 3 prompt construction count differs from the configured exact "
                        f"count {expected_count}: {mismatched_counts}"
                    )
            return selected_output

        prompts, metadata = await self._semantic_call(
            system_prompt=TASK3_PROMPT_SYSTEM,
            user_prompt=user_prompt,
            stage=(
                f"temporal_task3_prompts:{character['character_id']}:"
                f"{checkpoints[0]['checkpoint_id']}"
            ),
            normalize=normalize,
        )
        return prompts, {
            **metadata,
            "deterministic_payload_corrections": list(normalization_corrections),
            "input_budget_packing": package_budget_audit,
        }

    async def _semantic_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
        normalize: Callable[[dict[str, Any]], Any],
        repair: Callable[
            [dict[str, Any], Exception, int], Any
        ] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        checkpoint_path = None
        input_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema_version": "stage_temporal_call_input_v1",
                    "stage": stage,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self.call_checkpoint_dir is not None:
            checkpoint_path = self.call_checkpoint_dir / (
                stable_id("temporal-call", self.movie_id, stage) + ".json"
            )
            if checkpoint_path.exists():
                cached = load_json(checkpoint_path)
                if cached.get("stage") != stage:
                    raise ValueError(f"Temporal call checkpoint stage mismatch: {checkpoint_path}")
                cached_input_sha256 = clean_text(cached.get("input_sha256"))
                if cached_input_sha256 and cached_input_sha256 != input_sha256:
                    superseded_dir = (
                        self.call_checkpoint_dir.parent
                        / "llm_call_checkpoints_superseded"
                    )
                    superseded_dir.mkdir(parents=True, exist_ok=True)
                    suffix = cached_input_sha256[:12] or "legacy"
                    superseded_path = superseded_dir / (
                        f"{checkpoint_path.stem}.{suffix}.json"
                    )
                    if superseded_path.exists():
                        raise FileExistsError(superseded_path)
                    checkpoint_path.replace(superseded_path)
                else:
                    metadata = {
                        **cached["metadata"],
                        "checkpoint_reused": True,
                        "checkpoint_path": str(checkpoint_path),
                        "legacy_input_hash_backfilled": not bool(cached_input_sha256),
                    }
                    cached_result = cached["result"]
                    candidate_payload = cached.get("candidate_payload")
                    can_renormalize = (
                        candidate_payload is not None
                        and not bool(cached.get("metadata", {}).get("repaired"))
                    )
                    result = normalize(candidate_payload) if can_renormalize else cached_result
                    result_changed = result != cached_result
                    if result_changed:
                        result_sha256 = hashlib.sha256(
                            json.dumps(
                                cached_result,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        archive_dir = (
                            self.call_checkpoint_dir.parent
                            / "llm_call_checkpoints_renormalized"
                        )
                        archive_dir.mkdir(parents=True, exist_ok=True)
                        archive_path = archive_dir / (
                            f"{checkpoint_path.stem}.{result_sha256[:12]}.json"
                        )
                        if not archive_path.exists():
                            atomic_write_json(archive_path, cached)
                        metadata["checkpoint_renormalized"] = True
                        metadata["prior_result_sha256"] = result_sha256
                    if not cached_input_sha256 or result_changed:
                        atomic_write_json(
                            checkpoint_path,
                            {
                                "schema_version": "stage_temporal_call_checkpoint_v2",
                                "stage": stage,
                                "input_sha256": input_sha256,
                                "candidate_payload": candidate_payload,
                                "result": result,
                                "metadata": metadata,
                            },
                        )
                    return result, metadata
            override_path = (
                self.call_checkpoint_dir.parent
                / "llm_call_overrides"
                / checkpoint_path.name
            )
            if override_path.is_file():
                override = load_json(override_path)
                if (
                    override.get("schema_version")
                    != "stage_temporal_call_override_v1"
                    or override.get("stage") != stage
                ):
                    raise ValueError(f"Invalid temporal call override: {override_path}")
                recorded_input_sha256 = clean_text(
                    override.get("candidate_recorded_input_sha256")
                )
                requires_exact_input = bool(
                    override.get("requires_current_input_validation")
                )
                source_failure = Path(clean_text(override.get("source_failure")))
                if requires_exact_input and not recorded_input_sha256:
                    expected_source_hash = clean_text(
                        override.get("source_failure_sha256")
                    )
                    if (
                        source_failure.is_file()
                        and expected_source_hash
                        and sha256_file(source_failure) == expected_source_hash
                    ):
                        recorded_input_sha256 = clean_text(
                            load_json(source_failure).get("input_sha256")
                        )
                if requires_exact_input and recorded_input_sha256 != input_sha256:
                    superseded_dir = (
                        self.call_checkpoint_dir.parent
                        / "llm_call_overrides_superseded"
                    )
                    superseded_dir.mkdir(parents=True, exist_ok=True)
                    suffix = recorded_input_sha256[:12] or "unbound"
                    superseded_path = superseded_dir / (
                        f"{override_path.stem}.{suffix}.json"
                    )
                    if superseded_path.exists():
                        override_path.unlink()
                    else:
                        override_path.replace(superseded_path)
                else:
                    normalized = normalize(override.get("candidate_payload"))
                    metadata = {
                        "stage": stage,
                        "call_kind": "agent_manual_override",
                        "reviewer": override.get("reviewer"),
                        "decision_id": override.get("decision_id"),
                        "reason": override.get("reason"),
                        "evidence": override.get("evidence", []),
                        "model_calls_added": 0,
                        "override_path": str(override_path),
                        "candidate_recorded_input_sha256": recorded_input_sha256,
                    }
                    atomic_write_json(
                        checkpoint_path,
                        {
                            "schema_version": "stage_temporal_call_checkpoint_v2",
                            "stage": stage,
                            "input_sha256": input_sha256,
                            "result": normalized,
                            "metadata": metadata,
                        },
                    )
                    return normalized, metadata
        prompt_tokens = self.token_counter.count(system_prompt + user_prompt)
        if prompt_tokens > self.max_input_tokens:
            raise ValueError(
                f"Temporal prompt exceeds input budget: stage={stage} "
                f"tokens={prompt_tokens} budget={self.max_input_tokens}"
            )
        call = None
        try:
            async with self._semaphore:
                call = await self.llm_client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    stage=stage,
                )
            repaired = False
            repair_calls: list[dict[str, Any]] = []
            try:
                normalized = normalize(call.data)
            except Exception as validation_error:
                if repair is None:
                    raise
                repaired_payload, repair_calls = await repair(
                    call.data, validation_error, 1
                )
                normalized = normalize(repaired_payload)
                repaired = True
            metadata = {
                **call.metadata,
                "semantic_attempt": 1,
                "formal_call_count": 1,
                "maximum_targeted_repair_calls": 1 if repair is not None else 0,
                "prompt_tokens_measured": prompt_tokens,
                "repaired": repaired,
                "repair_calls": repair_calls,
            }
            if isinstance(normalized, list):
                for item in normalized:
                    if isinstance(item, dict):
                        item.setdefault("generator_metadata", metadata)
            if checkpoint_path is not None:
                atomic_write_json(
                    checkpoint_path,
                    {
                        "schema_version": "stage_temporal_call_checkpoint_v2",
                        "stage": stage,
                        "input_sha256": input_sha256,
                        "candidate_payload": call.data,
                        "result": normalized,
                        "metadata": metadata,
                    },
                )
            return normalized, metadata
        except Exception as exc:
            if self.call_checkpoint_dir is not None:
                failure_dir = self.call_checkpoint_dir.parent / "llm_call_failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_path = failure_dir / (
                    stable_id("temporal-call", self.movie_id, stage) + ".json"
                )
                failure = load_json(failure_path) if failure_path.is_file() else {
                    "schema_version": "stage_temporal_call_failure_audit_v1",
                    "stage": stage,
                    "input_sha256": input_sha256,
                    "attempts": [],
                }
                failure["attempts"].append(
                    {
                        "input_sha256": input_sha256,
                        "validation_error": clean_text(exc),
                        "candidate_payload": call.data if call is not None else None,
                        "call_metadata": call.metadata if call is not None else {},
                    }
                )
                atomic_write_json(failure_path, failure)
            raise RuntimeError(
                f"{stage} failed after one formal call"
                + (" and one targeted repair" if repair is not None else "")
                + f": {exc}"
            ) from exc


def _persona_prompt_token_count(
    *,
    character: dict[str, Any],
    episode: dict[str, Any],
    evidence: list[dict[str, Any]],
    token_counter: TokenCounter,
    language: str,
) -> int:
    prompt_evidence = [
        {
            "local_evidence_id": f"W{position:04d}",
            "scene_order": item["scene_order"],
            "type": item["evidence_type"],
            "speaker_is_target": item.get("speaker_character_id")
            == character["character_id"],
            "text": item["evidence_text"],
        }
        for position, item in enumerate(evidence, start=1)
    ]
    prompt = PERSONA_USER.format(
        language=language,
        character=json.dumps(_character_prompt(character), ensure_ascii=False),
        episode=json.dumps(_persona_episode_prompt(episode), ensure_ascii=False, indent=2),
        evidence=json.dumps(prompt_evidence, ensure_ascii=False, indent=2),
    )
    return token_counter.count(PERSONA_SYSTEM + prompt)


def _partition_persona_evidence(
    *,
    character: dict[str, Any],
    episode: dict[str, Any],
    evidence: list[dict[str, Any]],
    token_counter: TokenCounter,
    max_input_tokens: int,
    language: str,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        evidence,
        key=lambda item: (int(item["scene_order"]), item["evidence_id"]),
    )
    partitions: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in ordered:
        candidate = [*current, item]
        if current and _persona_prompt_token_count(
            character=character,
            episode=episode,
            evidence=candidate,
            token_counter=token_counter,
            language=language,
        ) > max_input_tokens:
            partitions.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        partitions.append(current)
    if any(
        _persona_prompt_token_count(
            character=character,
            episode=episode,
            evidence=partition,
            token_counter=token_counter,
            language=language,
        ) > max_input_tokens
        for partition in partitions
    ):
        raise ValueError(
            "Persona evidence item exceeds the temporal input budget even after partitioning"
        )
    return partitions


def _exact_object(payload: Any, keys: set[str], label: str) -> None:
    _object(payload, label)
    if set(payload) != keys:
        raise ValueError(f"{label} must return exactly {sorted(keys)}")


def _object(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")


def _array(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _probability(value: Any, label: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0,1]")
    return round(number, 6)


def _nth_occurrence(source: str, quote: str, occurrence: int) -> int:
    start = 0
    found = -1
    for _ in range(occurrence):
        found = source.find(quote, start)
        if found < 0:
            return -1
        start = found + len(quote)
    return found


def _occurrence_positions(source: str, quote: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        position = source.find(quote, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + max(1, len(quote))


def _unique_normalized_source_span(
    source: str,
    quote: str,
    *,
    minimum_length: int = 8,
    ignore_parentheticals: bool = False,
) -> tuple[int, int] | None:
    def normalized_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
        characters: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        ignored_positions = {
            position
            for marker in re.finditer(r"\?\s*\d+\s*\?", text)
            for position in range(marker.start(), marker.end())
        }
        if ignore_parentheticals:
            ignored_positions.update(
                position
                for marker in re.finditer(r"\([^()\n]{1,80}\)", text)
                for position in range(marker.start(), marker.end())
            )
        for start, value in enumerate(text):
            if start in ignored_positions:
                continue
            normalized = unicodedata.normalize("NFKC", value).casefold()
            for character in normalized:
                if character.isspace() or unicodedata.category(character).startswith(
                    ("P", "S")
                ):
                    continue
                characters.append(character)
                starts.append(start)
                ends.append(start + 1)
        return "".join(characters), starts, ends

    normalized_source, starts, ends = normalized_with_offsets(source)
    normalized_quote, _, _ = normalized_with_offsets(quote)
    if len(normalized_quote) < minimum_length:
        return None
    positions = _occurrence_positions(normalized_source, normalized_quote)
    if len(positions) != 1:
        return None
    position = positions[0]
    end = ends[position + len(normalized_quote) - 1]
    stripped_quote = quote.rstrip()
    if stripped_quote and unicodedata.category(stripped_quote[-1]).startswith(
        ("P", "S")
    ):
        while end < len(source) and unicodedata.category(source[end]).startswith(
            ("P", "S")
        ):
            end += 1
    return starts[position], end


def _normalized_occurrence_source_span(
    source: str,
    quote: str,
    *,
    occurrence: int,
    minimum_length: int = 8,
    ignore_parentheticals: bool = False,
) -> tuple[int, int] | None:
    """Resolve a normalized quote by its bounded occurrence index.

    This is used only for dialogue bodies when a speaker prefix and an
    intervening screenplay parenthetical make the full quote non-contiguous.
    Selecting the requested occurrence preserves the model's source-local
    ordering without accepting a paraphrase or silently choosing a span.
    """

    def normalized_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
        characters: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        ignored_positions = {
            position
            for marker in re.finditer(r"\?\s*\d+\s*\?", text)
            for position in range(marker.start(), marker.end())
        }
        if ignore_parentheticals:
            ignored_positions.update(
                position
                for marker in re.finditer(r"\([^()\n]{1,80}\)", text)
                for position in range(marker.start(), marker.end())
            )
        for start, value in enumerate(text):
            if start in ignored_positions:
                continue
            normalized = unicodedata.normalize("NFKC", value).casefold()
            for character in normalized:
                if character.isspace() or unicodedata.category(character).startswith(
                    ("P", "S")
                ):
                    continue
                characters.append(character)
                starts.append(start)
                ends.append(start + 1)
        return "".join(characters), starts, ends

    normalized_source, starts, ends = normalized_with_offsets(source)
    normalized_quote, _, _ = normalized_with_offsets(quote)
    if occurrence <= 0 or len(normalized_quote) < minimum_length:
        return None
    positions = _occurrence_positions(normalized_source, normalized_quote)
    if occurrence > len(positions):
        return None
    position = positions[occurrence - 1]
    end = ends[position + len(normalized_quote) - 1]
    stripped_quote = quote.rstrip()
    if stripped_quote and unicodedata.category(stripped_quote[-1]).startswith(
        ("P", "S")
    ):
        while end < len(source) and unicodedata.category(source[end]).startswith(
            ("P", "S")
        ):
            end += 1
    return starts[position], end


def _longest_unique_normalized_clause_span(
    source: str, quote: str, *, minimum_length: int = 8
) -> tuple[int, int] | None:
    clauses = sorted(
        unique_text(
            re.split(r"[\s\n]*[，。！？；：,.!?;:]+[\s\n]*|\n+", quote)
        ),
        key=lambda value: (-len(value), value),
    )
    for clause in clauses:
        span = _unique_normalized_source_span(
            source, clause, minimum_length=minimum_length
        )
        if span is not None:
            return span
    # Screenplay exports may interleave two columns or mix dialogue and action
    # on adjacent source lines. Search contiguous token fragments so the
    # fallback can retain a real, auditable source span without accepting a
    # paraphrase as evidence.
    tokens = re.findall(r"\S+", quote)
    for width in range(len(tokens), 1, -1):
        for start in range(0, len(tokens) - width + 1):
            fragment = " ".join(tokens[start : start + width])
            span = _unique_normalized_source_span(
                source, fragment, minimum_length=minimum_length
            )
            if span is not None:
                return span
    return None


def _local_id_list(
    value: Any,
    mapping: dict[str, Any],
    label: str,
    require_nonempty: bool,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    values = unique_text(value)
    if require_nonempty and not values:
        raise ValueError(f"{label} must be non-empty")
    if any(item not in mapping for item in values):
        raise ValueError(f"{label} contains an unknown local ID")
    return values


def _resolve_local_ids(
    value: Any,
    mapping: dict[str, dict[str, Any]],
    label: str,
    *,
    value_key: str,
    require_nonempty: bool,
) -> list[str]:
    return [
        mapping[local_id][value_key]
        for local_id in _local_id_list(value, mapping, label, require_nonempty)
    ]


def _character_prompt(character: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": character["canonical_name"],
        "aliases": character["aliases"],
        "first_scene_order": character["first_scene_order"],
        "last_scene_order": character["last_scene_order"],
        **(
            {"identity_phases": character["identity_phases"]}
            if character.get("identity_phases")
            else {}
        ),
    }


def _state_reconciliation_prompt_assets(
    *,
    character: dict[str, Any],
    observations: list[dict[str, Any]],
    language: str,
) -> dict[str, Any]:
    ordered = sorted(
        observations,
        key=lambda item: (
            int(item["observed_from_scene"]),
            item["observation_id"],
        ),
    )
    local = {
        f"O{position:04d}": item
        for position, item in enumerate(ordered, start=1)
    }
    evidence_ids = unique_text(
        evidence_id
        for item in ordered
        for evidence_id in item["supporting_evidence_ids"]
    )
    local_evidence = {
        f"W{position:04d}": evidence_id
        for position, evidence_id in enumerate(evidence_ids, start=1)
    }
    evidence_to_local = {
        evidence_id: local_id for local_id, evidence_id in local_evidence.items()
    }
    prompt_items = [
        {
            "local_observation_id": local_id,
            "dimension": item["dimension"],
            "target": item["target_id_or_text"],
            "state_value": item["state_value"],
            "polarity": item["polarity"],
            "certainty": item["certainty"],
            "durability": item["durability"],
            "scene_order": item["observed_from_scene"],
            "supporting_evidence_ids": [
                evidence_to_local[evidence_id]
                for evidence_id in item["supporting_evidence_ids"]
            ],
        }
        for local_id, item in local.items()
    ]
    user_prompt = STATE_RECONCILIATION_USER.format(
        language=language,
        character=json.dumps(_character_prompt(character), ensure_ascii=False),
        observations=json.dumps(prompt_items, ensure_ascii=False, indent=2),
    )
    return {
        "ordered": ordered,
        "local": local,
        "local_evidence": local_evidence,
        "user_prompt": user_prompt,
    }


def _state_reconciliation_output_estimate(
    observations: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "states": [
                {
                    "source_observation_ids": [item["observation_id"]],
                    "dimension": item["dimension"],
                    "target": item["target_id_or_text"],
                    "state_value": item["state_value"],
                    "polarity": item["polarity"],
                    "certainty": item["certainty"],
                    "durability": item["durability"],
                    "valid_from_scene": item["observed_from_scene"],
                    "valid_until_scene": 0,
                    "supporting_evidence_ids": item["supporting_evidence_ids"],
                }
                for item in observations
            ]
        },
        ensure_ascii=False,
        indent=2,
    )


def _usable_output_budget(max_output_tokens: int) -> int:
    margin = min(1024, max(64, max_output_tokens // 8))
    return max(1, max_output_tokens - margin)


def _state_target_resolution_prompt(
    *,
    character: dict[str, Any],
    known_characters: list[dict[str, Any]],
    local_targets: dict[str, dict[str, str]],
    language: str,
) -> str:
    return STATE_TARGET_RESOLUTION_USER.format(
        language=language,
        character=json.dumps(_character_prompt(character), ensure_ascii=False),
        known_characters=json.dumps(known_characters, ensure_ascii=False, indent=2),
        targets=json.dumps(
            [
                {"target_id": local_id, **payload}
                for local_id, payload in local_targets.items()
            ],
            ensure_ascii=False,
            indent=2,
        ),
    )


def _state_target_resolution_output_estimate(
    local_targets: dict[str, dict[str, str]],
) -> str:
    return json.dumps(
        {
            "resolutions": [
                {
                    "target_id": local_id,
                    "canonical_target": payload["surface_target"],
                    "target_kind": "other",
                }
                for local_id, payload in local_targets.items()
            ]
        },
        ensure_ascii=False,
        indent=2,
    )


async def _gather_all_settled(awaitables, *, label: str) -> list[Any]:
    results = await asyncio.gather(*list(awaitables), return_exceptions=True)
    failures = [
        (index, result)
        for index, result in enumerate(results, start=1)
        if isinstance(result, BaseException)
    ]
    if failures:
        details = "; ".join(
            f"item {index}: {clean_text(error)}"
            for index, error in failures
        )
        raise RuntimeError(f"{label} failed after all calls settled: {details}")
    return results


def _episode_prompt(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": episode.get("name", ""),
        "description": episode.get("description", ""),
        "source_scene_orders": episode.get("source_scene_orders", []),
        "progression_steps": episode.get("progression_steps", []),
        "state_changes": episode.get("state_changes", []),
    }


def _persona_episode_prompt(episode: dict[str, Any]) -> dict[str, Any]:
    """Keep persona context local without repeating the full episode graph."""
    return {
        "name": episode.get("name", ""),
        "description": episode.get("description", ""),
        "source_scene_orders": episode.get("source_scene_orders", []),
        "outcome": episode.get("outcome", ""),
    }


def _partition_development_sequence(
    states: list[dict[str, Any]],
    *,
    fits: Callable[[list[dict[str, Any]]], bool],
) -> list[list[dict[str, Any]]]:
    if len(states) < 2:
        return []
    if fits(states):
        return [states]
    partitions: list[list[dict[str, Any]]] = []
    current = [states[0]]
    for state in states[1:]:
        trial = [*current, state]
        if fits(trial):
            current = trial
            continue
        if len(current) < 2:
            raise ValueError(
                "One adjacent state transition exceeds the development input budget: "
                f"before_state={current[0]['state_id']} "
                f"resulting_state={state['state_id']}"
            )
        partitions.append(current)
        current = [current[-1], state]
        if not fits(current):
            raise ValueError(
                "One adjacent state transition exceeds the development input budget: "
                f"before_state={current[0]['state_id']} "
                f"resulting_state={current[1]['state_id']}"
            )
    partitions.append(current)
    return partitions


def _select_task3_prompt_quota(
    candidates: list[dict[str, Any]],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    if len(candidates) <= expected_count:
        return list(candidates)
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for item in candidates:
        family = item["prompt_family"]
        if family in seen_families:
            continue
        selected.append(item)
        seen_families.add(family)
        if len(selected) == expected_count:
            return selected
    for item in candidates:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) == expected_count:
            break
    return selected


def _development_prompt_assets(
    *,
    character: dict[str, Any],
    states: list[dict[str, Any]],
    focus_state_ids: set[str],
    facts: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    index: GraphIndex,
    language: str,
) -> dict[str, Any]:
    local_states = {
        f"S{position:04d}": item
        for position, item in enumerate(states, start=1)
    }
    local_facts = {
        f"F{position:04d}": item
        for position, item in enumerate(facts, start=1)
    }
    development_evidence_ids = unique_text(
        evidence_id
        for state in states
        for evidence_id in state["supporting_evidence_ids"]
    )
    local_evidence = {
        f"W{position:04d}": evidence_by_id[evidence_id]
        for position, evidence_id in enumerate(
            development_evidence_ids, start=1
        )
    }
    evidence_to_local = {
        item["evidence_id"]: local_id
        for local_id, item in local_evidence.items()
    }
    prompt_states = [
        {
            "local_state_id": local_id,
            "candidate_role": (
                "focus_sequence"
                if item["state_id"] in focus_state_ids
                else "invariant_only"
            ),
            **_state_prompt(item, evidence_to_local=evidence_to_local),
        }
        for local_id, item in local_states.items()
    ]
    prompt_facts = [
        {
            "local_fact_id": local_id,
            "type": item["node_type"],
            "text": index.fact_text(item),
            "scene_orders": index.node_scene_orders(item),
        }
        for local_id, item in local_facts.items()
    ]
    user_prompt = DEVELOPMENT_USER.format(
        language=language,
        character=json.dumps(_character_prompt(character), ensure_ascii=False),
        states=json.dumps(prompt_states, ensure_ascii=False, indent=2),
        facts=json.dumps(prompt_facts, ensure_ascii=False, indent=2),
        evidence=json.dumps(
            [
                {
                    "local_evidence_id": local_id,
                    "scene_order": item["scene_order"],
                    "type": item["evidence_type"],
                    "text": item["evidence_text"],
                }
                for local_id, item in local_evidence.items()
            ],
            ensure_ascii=False,
            indent=2,
        ),
    )
    return {
        "local_states": local_states,
        "local_facts": local_facts,
        "local_evidence": local_evidence,
        "user_prompt": user_prompt,
    }


def _state_prompt(
    state: dict[str, Any], *, evidence_to_local: dict[str, str] | None = None
) -> dict[str, Any]:
    evidence_ids: list[str] = []
    if evidence_to_local is not None:
        evidence_ids = [
            evidence_to_local[evidence_id]
            for evidence_id in state["supporting_evidence_ids"]
            if evidence_id in evidence_to_local
        ]
    return {
        "dimension": state["dimension"],
        "target": state["target_id_or_text"],
        "state_value": state["state_value"],
        "durability": state["durability"],
        "valid_from_scene": state["valid_from_scene"],
        "valid_until_scene": state["valid_until_scene"] or 0,
        "supporting_evidence_ids": evidence_ids,
    }


def _episode_facts(index: GraphIndex, episode: dict[str, Any]) -> list[dict[str, Any]]:
    child_ids = set(episode.get("child_unit_ids", []))
    return [fact for fact in index.facts if fact["id"] in child_ids]


def _fact_aligned_evidence(
    *,
    facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for fact in facts:
        fact_anchor = _evidence_anchor_text(fact.get("evidence"))
        fact_spans = [
            span
            for span in fact.get("source_spans", [])
            if isinstance(span, dict)
            and clean_text(span.get("chunk_id"))
            and span.get("char_start") is not None
            and span.get("char_end") is not None
        ]
        matched: list[dict[str, Any]] = []
        for item in evidence:
            evidence_anchor = _evidence_anchor_text(item.get("evidence_text"))
            if not evidence_anchor:
                continue
            span_match = any(
                clean_text(span.get("chunk_id")) == clean_text(item.get("chunk_id"))
                and int(span.get("char_start", 0)) < int(item.get("char_end", 0))
                and int(item.get("char_start", 0)) < int(span.get("char_end", 0))
                for span in fact_spans
            )
            if (
                (
                    span_match
                    and fact_anchor
                    and _meaningful_text_overlap(
                        item.get("evidence_text"), fact.get("evidence")
                    )
                )
                or (
                    fact_anchor
                    and (
                        evidence_anchor in fact_anchor
                        or fact_anchor in evidence_anchor
                    )
                )
            ):
                matched.append(item)
        # Participant overlap is only a fallback for source facts without their
        # own evidence text. Otherwise it widens a precise fact to every mention
        # of the same people in the scene and contaminates epistemic access.
        if not matched and not fact_anchor:
            participant_ids = set(fact.get("participant_entity_ids", []))
            participant_ids.update(
                item.get("entity_id")
                for item in fact.get("participant_entities", [])
                if isinstance(item, dict) and item.get("entity_id")
            )
            matched = [
                item
                for item in evidence
                if participant_ids
                & {
                    item.get("speaker_character_id"),
                    *item.get("participant_character_ids", []),
                    *item.get("direct_observer_character_ids", []),
                    *item.get("addressee_character_ids", []),
                }
            ]
        for item in matched:
            output[item["evidence_id"]] = item
    return sorted(
        output.values(),
        key=lambda item: (
            int(item["scene_order"]),
            int(item["char_start"]),
            item["evidence_id"],
        ),
    )


def _evidence_anchor_text(value: Any) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", str(value or ""))
        if character.isalnum()
    )


def _meaningful_text_overlap(left: Any, right: Any) -> bool:
    left_tokens = set(re.findall(r"[^\W_]+", clean_text(left).casefold()))
    right_tokens = set(re.findall(r"[^\W_]+", clean_text(right).casefold()))
    if not left_tokens or not right_tokens:
        return False
    shorter = min(len(left_tokens), len(right_tokens))
    overlap = len(left_tokens & right_tokens)
    return overlap >= 3 and overlap / shorter >= 0.25


def _immediate_development_fact_ids(
    *,
    index: GraphIndex,
    character: dict[str, Any],
    source_episode_ids: list[str],
) -> list[str]:
    episode_by_id = {item["id"]: item for item in index.episodes}
    source_ids = {item for item in source_episode_ids if item in episode_by_id}
    related_episode_ids = {
        clean_text(edge.get("target"))
        for edge in index.graph.get("edges", [])
        if clean_text(edge.get("source")) in source_ids
        and clean_text(edge.get("target")) in episode_by_id
        and clean_text(edge.get("relation_type"))
        in {"causes", "enables", "continues", "resolves", "reverses", "reveals"}
    }
    if not related_episode_ids and source_ids:
        last_order = max(int(episode_by_id[item]["order"]) for item in source_ids)
        next_episode = next(
            (
                episode
                for episode in index.episodes
                if int(episode["order"]) > last_order
                and _character_participates_in_episode(character, episode)
            ),
            None,
        )
        if next_episode is not None:
            related_episode_ids.add(next_episode["id"])
    return unique_text(
        fact_id
        for episode_id in sorted(
            related_episode_ids,
            key=lambda item: (int(episode_by_id[item]["order"]), item),
        )
        for fact_id in episode_by_id[episode_id].get("child_unit_ids", [])
    )


def _character_relevant_in_evidence(
    character_id: str, evidence: list[dict[str, Any]]
) -> bool:
    for item in evidence:
        if character_id == item.get("speaker_character_id"):
            return True
        if any(
            character_id in item.get(key, [])
            for key in (
                "participant_character_ids",
                "direct_observer_character_ids",
                "addressee_character_ids",
            )
        ):
            return True
    return False


def _character_participates_in_episode(
    character: dict[str, Any], episode: dict[str, Any]
) -> bool:
    participants = {
        item.get("entity_id")
        for item in episode.get("participant_entities", [])
        if isinstance(item, dict)
    }
    orders = [int(value) for value in episode.get("source_scene_orders", [])]
    if not orders and episode.get("source_scene_order") is not None:
        orders = [int(episode["source_scene_order"])]
    if not orders:
        return character["character_id"] in participants
    return any(
        bool(source_character_ids_at_scene(character, order) & participants)
        for order in orders
    )


def _fact_relevant_to_character(
    fact: dict[str, Any], character: dict[str, Any], index: GraphIndex
) -> bool:
    source_ids = set().union(
        *(source_character_ids_at_scene(character, order) for order in index.node_scene_orders(fact))
    ) or {character["character_id"]}
    if fact.get("subject_entity_id") in source_ids or fact.get("object_entity_id") in source_ids:
        return True
    if any(
        item.get("entity_id") in source_ids
        for item in fact.get("participant_entities", [])
        if isinstance(item, dict)
    ):
        return True
    return bool(set(index.node_scene_ids(fact)) & set(character["scene_ids"]))


def _assign_facts_to_episodes(index: GraphIndex) -> dict[str, list[dict[str, Any]]]:
    episode_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in index.episodes:
        for scene_id in index.node_scene_ids(episode):
            episode_by_scene[scene_id].append(episode)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in index.facts:
        candidates = {
            episode["id"]: episode
            for scene_id in index.node_scene_ids(fact)
            for episode in episode_by_scene.get(scene_id, [])
        }
        if fact["id"] in {
            child_id for episode in index.episodes for child_id in episode.get("child_unit_ids", [])
        }:
            candidates = {
                episode["id"]: episode
                for episode in index.episodes
                if fact["id"] in episode.get("child_unit_ids", [])
            }
        if not candidates:
            continue
        assigned = min(
            candidates.values(), key=lambda episode: (int(episode.get("order", 0)), episode["id"])
        )
        output[assigned["id"]].append(fact)
    return output


def sanitize_developments(
    developments: list[dict[str, Any]],
    states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states_by_id = {item["state_id"]: item for item in states}
    output = []
    decisions = []
    seen_development_ids: set[str] = set()
    for development in developments:
        development_id = clean_text(development.get("development_id"))
        if development_id and development_id in seen_development_ids:
            decisions.append(
                {
                    "development_id": development_id,
                    "action": "drop_duplicate_development",
                    "reason": "same_transition_emitted_by_multiple_partitions",
                }
            )
            continue
        if development_id:
            seen_development_ids.add(development_id)
        before = [
            states_by_id[state_id]
            for state_id in development.get("before_state_ids", [])
            if state_id in states_by_id
        ]
        resulting = [
            states_by_id[state_id]
            for state_id in development.get("resulting_state_ids", [])
            if state_id in states_by_id
        ]
        before_values = {
            normalize_name(item["state_value"]) for item in before
        }
        resulting_values = {
            normalize_name(item["state_value"]) for item in resulting
        }
        if before and resulting and (
            set(development.get("before_state_ids", []))
            == set(development.get("resulting_state_ids", []))
            or before_values == resulting_values
        ):
            decisions.append(
                {
                    "development_id": development["development_id"],
                    "action": "drop_noop_development",
                    "reason": "before_and_resulting_states_are_equivalent",
                }
            )
            continue
        dropped_invariants = list(development.get("invariant_state_ids", []))
        normalized = {**development, "invariant_state_ids": []}
        output.append(normalized)
        if dropped_invariants:
            decisions.append(
                {
                    "development_id": development["development_id"],
                    "action": "clear_model_selected_invariants",
                    "state_ids": dropped_invariants,
                    "reason": "invariants_are_derived_from_checkpoint_continuity",
                }
            )
    return output, {
        "policy": "drop_noop_and_derive_invariants_from_checkpoint_continuity_v2",
        "input_count": len(developments),
        "output_count": len(output),
        "decisions": decisions,
        "model_calls_added": 0,
    }


def _task3_checkpoint_assets(
    *,
    character_id: str,
    checkpoint: dict[str, Any],
    states: list[dict[str, Any]],
    access_records: list[dict[str, Any]],
    persona: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    state_by_id: dict[str, dict[str, Any]],
    development_by_id: dict[str, dict[str, Any]],
    index: GraphIndex,
) -> dict[str, Any]:
    scene_order = int(checkpoint["scene_order"])
    developments = [
        development_by_id[development_id]
        for development_id in checkpoint["new_development_ids"]
        if development_id in development_by_id
    ]
    focal_state_ids = set(
        unique_text(
            state_id
            for development in developments
            for state_id in [
                *development["before_state_ids"],
                *development["resulting_state_ids"],
                *development["invariant_state_ids"],
            ]
        )
    )
    focal_fact_ids = set(
        unique_text(
            fact_id
            for development in developments
            for fact_id in [
                *development["catalyst_event_ids"],
                *development["downstream_consequence_ids"],
            ]
        )
    )
    focal_evidence_ids = set(
        unique_text(
            evidence_id
            for development in developments
            for key in (
                "evidence_before_ids",
                "evidence_catalyst_ids",
                "evidence_after_ids",
            )
            for evidence_id in development[key]
            if evidence_id in evidence_by_id
        )
    )
    scene_character_evidence = [
        item
        for item in evidence_by_id.values()
        if int(item["scene_order"]) == scene_order
        and _character_relevant_in_evidence(character_id, [item])
    ]
    focal_evidence_ids.update(item["evidence_id"] for item in scene_character_evidence)
    if not developments:
        focal_state_ids.update(
            state["state_id"]
            for state in states
            if state["state_id"] in checkpoint["active_state_ids"]
            and (
                int(state["valid_from_scene"]) == scene_order
                or bool(set(state["supporting_evidence_ids"]) & focal_evidence_ids)
            )
        )
        focal_fact_ids.update(
            access["fact_or_event_id"]
            for access in access_records
            if scene_order in access["fact_source_scene_orders"]
            or bool(set(access["supporting_evidence_ids"]) & focal_evidence_ids)
        )
    if not focal_state_ids and not focal_fact_ids:
        # A baseline checkpoint can legitimately have no active state or
        # accessible memory. Retain one future fact as a negative boundary
        # anchor so Task 3 still has an evaluator-visible cutoff asset.
        focal_fact_ids.update(
            fact_id
            for fact_id in checkpoint["future_forbidden_fact_ids"]
            if fact_id in index.nodes_by_id
        )
        if not focal_fact_ids and scene_character_evidence:
            focal_evidence_ids.add(scene_character_evidence[0]["evidence_id"])
    selected_states = [
        state_by_id[state_id]
        for state_id in sorted(focal_state_ids)
        if state_id in state_by_id
        and state_by_id[state_id]["character_id"] == character_id
    ]
    focal_evidence_ids.update(
        evidence_id
        for state in selected_states
        for evidence_id in state["supporting_evidence_ids"]
        if evidence_id in evidence_by_id
    )
    selected_access = [
        access
        for access in access_records
        if access["fact_or_event_id"] in focal_fact_ids
        or bool(set(access["supporting_evidence_ids"]) & focal_evidence_ids)
    ]
    focal_fact_ids.update(access["fact_or_event_id"] for access in selected_access)
    focal_evidence_ids.update(
        evidence_id
        for access in selected_access
        for evidence_id in access["supporting_evidence_ids"]
        if evidence_id in evidence_by_id
    )
    selected_persona = [
        item
        for item in persona
        if int(item["established_from_scene"]) <= scene_order
        and (
            item["superseded_at_scene"] is None
            or int(item["superseded_at_scene"]) >= scene_order
        )
        and bool(set(item["supporting_evidence_ids"]) & focal_evidence_ids)
    ]
    if not selected_persona:
        active_stable = [
            item
            for item in persona
            if item["stability"] == "stable"
            and int(item["established_from_scene"]) <= scene_order
            and (
                item["superseded_at_scene"] is None
                or int(item["superseded_at_scene"]) >= scene_order
            )
        ]
        earliest_by_kind = {
            kind: min(
                int(item["established_from_scene"])
                for item in active_stable
                if item["evidence_kind"] == kind
            )
            for kind in {item["evidence_kind"] for item in active_stable}
        }
        selected_persona = [
            item
            for item in active_stable
            if int(item["established_from_scene"])
            == earliest_by_kind[item["evidence_kind"]]
        ]
    focal_evidence_ids.update(
        evidence_id
        for item in selected_persona
        for evidence_id in item["supporting_evidence_ids"]
        if evidence_id in evidence_by_id
    )
    selected_checkpoint = {
        **checkpoint,
        "active_state_ids": [
            state_id
            for state_id in checkpoint["active_state_ids"]
            if state_id in focal_state_ids
        ],
        "accessible_fact_ids": [
            fact_id
            for fact_id in checkpoint["accessible_fact_ids"]
            if fact_id in focal_fact_ids
        ],
        "unknown_fact_ids": [
            fact_id
            for fact_id in checkpoint["unknown_fact_ids"]
            if fact_id in focal_fact_ids
        ],
        "future_forbidden_fact_ids": [
            fact_id
            for fact_id in checkpoint["future_forbidden_fact_ids"]
            if fact_id in focal_fact_ids
        ],
        "persona_evidence_ids": [
            item["persona_evidence_id"] for item in selected_persona
        ],
        "dialogue_exemplar_ids": [
            item["evidence_id"]
            for item in scene_character_evidence
            if item["evidence_type"] == "dialogue"
            and item["speaker_character_id"] == character_id
        ],
    }
    return {
        "checkpoint": selected_checkpoint,
        "states": selected_states,
        "access_records": selected_access,
        "persona": selected_persona,
        "evidence_by_id": {
            evidence_id: evidence_by_id[evidence_id]
            for evidence_id in sorted(focal_evidence_ids)
            if evidence_id in evidence_by_id
        },
    }


def _group(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        output[item[key]].append(item)
    return output


def _state_active(state: dict[str, Any], scene_order: int) -> bool:
    return int(state["valid_from_scene"]) <= scene_order and (
        state["valid_until_scene"] is None
        or scene_order <= int(state["valid_until_scene"])
    )
