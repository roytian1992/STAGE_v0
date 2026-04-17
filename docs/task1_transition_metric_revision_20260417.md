# Task1 Transition Metric Revision 2026-04-17

This note records the 2026-04-17 revision of the Task1 transition metric and replaces the earlier interpretation for ongoing Task1 evaluation.

## Motivation

The earlier `pred_transition_coherence` setting was too harsh on otherwise reasonable predicted timelines.

Observed failure mode:

- several movies had very low transition scores
- but their macro Task1 quality remained strong
- in particular, `fact_f1` often stayed around `0.93-1.00`

This mismatch indicated that the issue was primarily in the transition metric design rather than in the underlying Task1 extraction quality.

## Main Problems In The Earlier Version

1. The judge prompt was too close to direct next-step progression.

- It under-credited cases where:
  - the character plausibly moved to another subthread
  - intermediate events were omitted
  - the later node was compatible rather than directly caused by the earlier node

2. Important-pair selection was too permissive about difficult pairs.

- It could select:
  - same-scene beat splits
  - large-gap pairs
  - early-to-late style pairs that are too hard for a local transition metric

3. Audit output was not rich enough.

- We needed richer pair-level logging to inspect:
  - `scene_order`
  - `beat_index`
  - node ids
  - compact node text

## What Changed

The metric name stays the same:

- `pred_transition_coherence`
- `important_pred_transition_coherence`

But the semantics are revised.

### 1. Transition Judge Prompt

The judge is now compatibility-based rather than direct-causality-based.

Current rule:

- output `TRUE` if the later node can plausibly follow the earlier node without contradiction
- do not require:
  - direct causality
  - a single uninterrupted thread
  - every intermediate event to be explicitly represented
- output `FALSE` only when:
  - the later node clearly contradicts the earlier node
  - the jump is implausible and unsupported by scene evidence
  - the node text misstates the scene-grounded character state

### 2. Important-Pair Selection

Important-pair selection is now more local and more anchored.

Current preference:

- local pairs with `1 <= scene_gap <= 6`
- pairs with explicit transition anchors such as:
  - `resulting_state`
  - `goal_state`
  - `unresolved_issue`

Current avoidance:

- same-scene pairs unless strongly anchored
- very large-gap pairs
- `early->late` style pairs with long span

### 3. Audit Logging

Pair-level audit output now includes:

- raw `scene_id`
- `scene_order`
- `beat_index`
- `timeline_node_id`
- compact node text fields
- flags such as:
  - `has_transition_anchor`
  - `is_local_pair`
  - `is_extreme_gap_pair`

## Code Changes

Files updated:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0/tools/task1/metrics.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/task1_v1_pilot/eval_task1_manifest_v8_20260417.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/task1_v1_pilot/eval_existing_task1_v8_20260417.py`

Also fixed:

- evaluation scripts had been overriding the working `8002` route and forcing `8001`
- current eval scripts now use `8002` as primary route

## 5-Movie Calibration

Calibration report:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/task1_v1_pilot/reports/task1_transition_calibration5_20260417_qwen8002.json`

The 5 movies were selected from the worst transition-score cases under the earlier metric.

### Before vs After

| movie_id | old_pred_transition | new_pred_transition | old_important_transition | new_important_transition |
|---|---:|---:|---:|---:|
| en1ae8880bb21245a39ad18f119e163fec | 0.1111 | 0.6667 | 0.0000 | 0.5000 |
| ch031fd9f8e8339ed61190f6e72d73dc8b05e8aef2 | 0.1224 | 0.3878 | 0.2000 | 0.5000 |
| en2e733a32750748b082f2ffb7f4608329 | 0.2174 | 0.7826 | 0.0000 | 0.7500 |
| ch060a57fa2cc7b375d4db567c3ec463f4c7f0dd75 | 0.2899 | 0.7101 | 0.1667 | 0.6667 |
| en0c08ce1c06774785b5d73d9effd69e6b | 0.3696 | 0.7609 | 0.2500 | 0.8750 |

Average over the 5 calibration movies:

- `pred_transition_coherence`: `0.2221 -> 0.6616`
- `important_pred_transition_coherence`: `0.1233 -> 0.6583`

## Interpretation

The calibration strongly suggests:

- the earlier low transition scores were mostly caused by metric harshness
- the revised metric better reflects whether a predicted timeline is narratively compatible
- this revision should replace the earlier interpretation for current Task1 evaluation

## Going Forward

For ongoing Task1 reporting, use the revised interpretation:

- `gold_fact_recall`
- `pred_fact_precision`
- `fact_f1`
- `pred_transition_coherence`
- `arc_narrative_aspect_correctness`

Use `important_pred_transition_coherence` as an audit-facing companion metric.
