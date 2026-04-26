# STAGE_v0 Scene Reference Repair Log (2026-04-26)

## Scope

- Source benchmark path: `/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0`
- Repair target:
  - `script.json` scene title prefixes
  - `task_1_character_timelines.json` `scene_title`
  - `task_2_question_answering.csv` structured scene references
  - `task_3_role_assets.json` `scene_title`
- This repair does **not** change benchmark article counts or task counts.

## Base Counts

- Article directories with core files: `151`
- `script.json`: `151`
- `task_1_character_timelines.json`: `151`
- `task_2_question_answering.csv`: `151`
- `task_3_role_assets.json`: `151`

## Problem Summary

The working tree already contained a prior global renumber pass that left three classes of data inconsistency:

1. `script.json` titles had inconsistent or duplicated numeric prefixes.
2. `task1` / `task3` `scene_title` fields no longer matched the authoritative scene title in `script.json`.
3. `task2` contained corrupted structured scene references such as duplicated prefixes like `20、20、...`, wrong copied scene titles, and scene-grounding answers missing stable scene numbering.

Special convention requested in this repair:

- `序` should use prefix `0`.

## Repair Rules Applied

Implemented in:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0/tools/repair_scene_references_20260426.py`

Rules:

1. Canonicalize scene titles from `script.json` with authoritative prefix format `"{scene_order}、{body}"`.
2. If the scene body begins with `序`, force prefix `0`.
3. Conservatively remove only redundant repeated same-id prefixes after the canonical prefix, for example:
   - `17、17 .堕落街，夜，外` -> `17、堕落街，夜，外`
   - `70、70 KTV 厅 内 日` -> `70、KTV 厅 内 日`
4. Rebuild `task1` / `task3` `scene_title` directly from `scene_order`.
5. Repair `task2`:
   - `related_scenes` by article-local scene lookup
   - scene-grounding and scene-like answer fields by deterministic scene-title resolution
   - explicit broken scene mentions in `question` / `evidence_or_reason` when they can be safely resolved from article-local scene references
6. Do **not** globally rewrite arbitrary free text when a scene reference cannot be resolved safely.

## Files Written

Final modified file categories in the current worktree:

- `script.json`: `142`
- `task_1_character_timelines.json`: `144`
- `task_2_question_answering.csv`: `151`
- `task_3_role_assets.json`: `144`
- repair script: `1`

Notes:

- Some files were already dirty before this repair.
- `task_1_cross_scene_arcs.json` was inspected but not rewritten because it does not carry mutable `scene_title` payloads in this repair flow.

## Verification

Final structural check command:

```bash
python /vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0/tools/repair_scene_references_20260426.py --check
```

Final result on 2026-04-26:

```json
{}
```

Meaning:

- no remaining canonical `script.json` prefix mismatch under the repair rules
- no remaining `task1` / `task3` `scene_title` mismatch
- no remaining duplicated-prefix corruption in `task2` structured scene-reference fields covered by the checker

## Key Examples Confirmed

- `Chinese/ch232d9fa92b0f7586b39d9a6beb75dc2827489ed3/task_2_question_answering.csv`
  - `q23` fixed to `63、坟地，夜，外 | 65、坟地，夜，外`
  - loosened subjective “三个核心角色” wording remains preserved
- `Chinese/chad83aa5c1d76c9068cb7f2693704827931d91b36/script.json`
  - `序场A` / `序场B` both use prefix `0`
- `Chinese/ch6d1d9f66a85adc2119804cfbe50dea12dba17fc3`
  - repaired `17、17 .堕落街，夜，外` source title and linked task2 reference

## Caveats

- This repair intentionally preserves original screenplay body text such as `1场`, `95A`, `36A`, or spelled-out scene labels like `四十四、...` when they appear to be part of the underlying script body rather than accidental duplicated canonical prefixes.
- The checker focuses on structural scene-reference consistency, not semantic QA correctness.
- Future novelty cleanup, strict subset recomputation, and supplementation are not relevant to this repair and were not run.
