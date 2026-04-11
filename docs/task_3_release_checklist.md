# Task 3 Release Checklist

## Must Complete

- Freeze the canonical Task 3 metric set.
- Freeze the judge output schema for single-turn and multi-turn evaluation.
- Add one minimal prediction example for single-turn and multi-turn.
- Add one minimal evaluation example to the repository docs.
- Decide whether release reports should treat `violation_flags` as required or optional.

## Strongly Recommended

- Prepare a small human-validated subset for Task 3.
- Add a public judge prompt template.
- Add a prediction validation script.
- Add a compact evaluation script that reads prediction JSON and emits report summaries.
- Add a small worked example showing how `persona_only`, `persona_with_memory`, and `persona_with_memory_and_relations` differ.

## Optional For v0

- Release historical audit files.
- Release generation scripts.
- Release internal polish summaries.
- Release builder pipelines.

## Already Done

- Movie-level Task 3 packaging completed.
- Single-turn release files packaged.
- Multi-turn release files packaged.
- Task 3 schema documentation added.
- Task 3 data assets added to `STAGE_v0`.
- Chinese weak-tail role quality substantially improved.
