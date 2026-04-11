# Task 3: In-Script Character Role Play

## Overview

Task 3 evaluates whether a model or agent can answer as a screenplay character while remaining:

- faithful to script-grounded memory,
- consistent with persona constraints,
- stable across multi-turn interaction.

This task is intended to support two closely related use cases:

1. in-script role-play evaluation;
2. memory-aware role-play evaluation.

## Release Files

Each movie directory contains two Task 3 files:

- `task_3_in_script_character_role_play_single_turn.json`
- `task_3_in_script_character_role_play_multi_turn.json`

These are aggregated movie-level files built from all retained focal roles for that screenplay.

## Single-Turn Format

Top-level fields:

- `task`
- `task_version`
- `interaction_format`
- `movie_id`
- `language`
- `role_count`
- `roles`
- `instance_count`
- `instances`

Each item in `instances` contains:

- `task`
- `task_version`
- `instance_id`
- `movie_id`
- `language`
- `character`
- `role_dir`
- `interaction_format`
- `input`
- `reference`

`input` includes the model-facing prompt state, such as:

- `mode`
- `persona_card`
- `memory_context`
- `relation_context`
- `dialogue_history`
- `current_user_turn`

`reference` includes judge-facing supervision fields, such as:

- `question_id`
- `question_family`
- `memory_required`
- `source_memory_ids`
- `source_relation_ids`
- `supporting_facts`
- `contradicting_facts`
- `knowledge_boundary`
- `expected_stance`
- `expected_style`
- `persona_risk_type`
- `cross_turn_constraints`

## Multi-Turn Format

Top-level fields:

- `task`
- `task_version`
- `interaction_format`
- `movie_id`
- `language`
- `role_count`
- `roles`
- `episode_count`
- `episodes`

Each item in `episodes` contains:

- `task`
- `task_version`
- `instance_id`
- `episode_instance_id`
- `movie_id`
- `language`
- `character`
- `role_dir`
- `interaction_format`
- `episode_id`
- `mode`
- `episode_theme`
- `turn_count`
- `history_source`
- `shared_input`
- `turns`

`shared_input` currently contains the role-level `persona_card`.

Each turn contains:

- `turn_index`
- `question_id`
- `question_family`
- `input`
- `reference`

`input` includes the current turn prompt plus structured retrieval context:

- `mode`
- `memory_context`
- `relation_context`
- `dialogue_history_template`
- `current_user_turn`

`reference` contains the answerability and judging anchors for that turn.

## Recommended Uses

### Role-Play Evaluation

Use the released `input` fields as the actual model input.

Judge whether the response:

- sounds like the target character,
- stays within the script knowledge boundary,
- avoids fabricated memory,
- remains consistent with the provided persona card.

### Memory-Aware Evaluation

Task 3 can also be used to probe memory modules by comparing systems with and without structured memory injection.

Recommended comparisons:

- persona only
- persona + memory
- persona + memory + relations

Recommended focus:

- persona drift under multi-turn interaction
- memory retrieval usefulness
- consistency under follow-up pressure
- stability of character stance and style

## What Is Not Included In This Release

The release repository does not include:

- generation scripts,
- polish or second-pass audit files,
- builder pipelines,
- evaluation code.

Only task-facing benchmark assets are included here.

## Current Status

Task 3 data is close to release-ready as a benchmark asset.

The main remaining non-data work is protocol-side:

- final metric definition,
- judge interface definition,
- benchmark README examples,
- optional human-validated subset.
