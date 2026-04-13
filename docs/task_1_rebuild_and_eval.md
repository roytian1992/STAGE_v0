# Task 1 Rebuild And Eval

Task 1 now ships with a compact rebuild and evaluation entry point:

- `tools/task1/pipeline.py`

The code is organized so that users only need one command surface:

- single movie rebuild and optional evaluation
- batch rebuild and evaluation across English and/or Chinese movies

## Inputs

The rebuild pipeline assumes each movie directory contains:

- `script.json`
- focal-role source from one of:
  - `task_3_in_script_character_role_play_single_turn.json`
  - `task_3_in_script_character_role_play_multi_turn.json`
  - `gold_character_timelines_v1.json`
  - `task_1_character_timelines.json`

## Single Movie

```bash
python tools/task1/pipeline.py one \
  --movie-dir English/<movie_id> \
  --output-dir runs/task1_one/<movie_id> \
  --evaluate
```

This writes:

- `pred_task_1_character_timelines.json`
- `pred_task_1_cross_scene_arcs.json`
- `role_scene_cards.json`
- `character_segments.json`
- `character_milestones.json`
- `eval_v3.json` if `--evaluate` is enabled

## Batch

```bash
python tools/task1/pipeline.py batch \
  --benchmark-root . \
  --languages en,zh \
  --output-root runs/task1_batch \
  --report-path runs/task1_batch/report.json \
  --max-workers 4
```

Optional filters:

- `--movie-ids id1,id2,...`
- `--overwrite`

## Evaluation Metrics

`eval_v3.json` reports:

- `legacy_scene_grounding_precision`
- `legacy_scene_grounding_recall`
- `legacy_scene_grounding_f1`
- `node_grounding_precision`
- `node_grounding_recall`
- `node_grounding_f1`
- `development_correctness`
- `state_transition_correctness`
- `arc_narrative_aspect_correctness`
- `arc_progression_correctness`
- `overall`

The current Task 1 pipeline is recall-first and aims to preserve broad story coverage for the benchmark focal roles while keeping node text grounded in the original script.
