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

## Release Schema

The released benchmark gold files are intentionally simplified.

`task_1_character_timelines.json` keeps only:

- `movie_id`
- `language`
- `focal_character_timelines`
  - `character_name`
  - `aliases`
  - `timeline_summary`
  - `timeline_nodes`
    - `timeline_node_id`
    - `scene_id`
    - `scene_order`
    - `scene_title`
    - `importance`
    - `role_in_context`
    - `salient_development`
    - `goal_state`
    - `resulting_state`
    - `unresolved_issue`
    - `evidence_quotes`

`task_1_cross_scene_arcs.json` keeps only:

- `movie_id`
- `language`
- `cross_scene_arcs`
  - `arc_id`
  - `character_name`
  - `title`
  - `arc_focus`
  - `linked_timeline_node_ids`
  - `arc_summary`
  - `start_state`
  - `end_state`
  - `unresolved_issue`

Build provenance and audit-only fields are intentionally omitted from the released gold set.

## Evaluation Metrics

`eval_v3.json` reports:

- `gold_fact_recall`
- `pred_fact_precision`
- `fact_f1`
- `pred_transition_coherence`
- `important_pred_transition_coherence`
- `arc_narrative_aspect_correctness`
- `development_correctness`
- `state_transition_correctness`
- `arc_progression_correctness`
- `legacy_scene_grounding_precision`
- `legacy_scene_grounding_recall`
- `legacy_scene_grounding_f1`
- `node_grounding_precision`
- `node_grounding_recall`
- `node_grounding_f1`
- `overall`

Recommended primary Task 1 metrics:

- `gold_fact_recall`
- `pred_fact_precision`
- `fact_f1`
- `pred_transition_coherence`
- `arc_narrative_aspect_correctness`

Why the metric stack changed:

- strict `node_grounding_*` depends on 1-to-1 node alignment and under-credits merge/split differences
- macro fact coverage better reflects whether the predicted timeline covers the main narrative information
- predicted transition coherence evaluates the quality of the predicted trajectory itself rather than only matched gold/pred pairs
- arc narrative aspect correctness checks whether the long-range character storyline dimension is captured

Metric interpretation:

- `gold_fact_recall`
  - how much of the gold character trajectory is covered by the predicted timeline at the macro fact level
- `pred_fact_precision`
  - how much of the predicted macro content is supported by the gold timeline
- `fact_f1`
  - harmonic mean of macro recall and macro precision
- `pred_transition_coherence`
  - coherence of adjacent predicted node pairs, judged with node text and script scene evidence
- `important_pred_transition_coherence`
  - the same transition judgment but restricted to a small heuristic subset of salient adjacent pairs per character
- `arc_narrative_aspect_correctness`
  - whether the predicted arc captures the same long-range narrative aspect as the gold arc

Secondary and diagnostic metrics retained in `eval_v3.json`:

- `development_correctness`
- `state_transition_correctness`
- `arc_progression_correctness`
- `legacy_scene_grounding_*`
- `node_grounding_*`

Human validation remains part of the research process. These metrics are intended to be useful and inspectable, not perfect. In particular:

- macro fact metrics can vary with fact granularity and alias handling
- transition coherence can be harsh when adjacent predicted nodes span a coarse latent gap
- important-pair transition coherence is useful for audit, but may be less stable than the raw transition metric

The current Task 1 pipeline is recall-first and aims to preserve broad story coverage for the benchmark focal roles while keeping node text grounded in the original script.
