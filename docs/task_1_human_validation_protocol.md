# Task 1 Human Validation Protocol

## Purpose

Task 1 metrics are designed to be useful, auditable, and aligned with the intended narrative construct. They are not assumed to be perfect automatic judges.

Human validation is used to:

- confirm that the automatic metrics are directionally reasonable
- surface recurring failure modes
- document where metric behavior diverges from human judgment

## Primary Metrics To Audit

- `gold_fact_recall`
- `pred_fact_precision`
- `fact_f1`
- `pred_transition_coherence`
- `arc_narrative_aspect_correctness`

## Suggested Sampling Strategy

For a pilot or release audit, sample a mixed set of movies that includes:

- English and Chinese scripts
- high-score and low-score cases
- at least one case with coarse timelines
- at least one case with known alias or naming variation

## What To Inspect Per Movie

1. Gold Task 1 timeline
2. Predicted Task 1 timeline
3. Gold Task 1 cross-scene arcs
4. Predicted Task 1 cross-scene arcs
5. `fact_coverage_details.json`
6. The scene text for a few transition pairs, especially ones judged `False`

## Audit Questions

### Macro Fact Coverage

- Does the predicted timeline cover the main durable developments in the gold timeline?
- Are missed facts genuinely important, or mostly granularity differences?
- Are predicted facts mostly grounded, or does the model add unsupported narrative claims?

### Transition Coherence

- Do adjacent predicted nodes form a plausible character trajectory?
- When the metric marks a pair as incoherent, is that because:
  - the pair is truly contradictory
  - the timeline is too coarse
  - the pair is important but under-specified
  - the scene evidence is insufficient

### Arc Narrative Aspect

- Does the predicted arc capture the same long-range storyline dimension?
- If it fails, is the model following the wrong thread or just naming it differently?

## Recommended Annotation Labels

For each audited movie, assign one label per primary metric:

- `reasonable`
- `slightly_off_but_acceptable`
- `clearly_misleading`

## Failure Mode Log

When a metric is not reasonable, record the failure type:

- `alias_or_name_variation`
- `fact_granularity_mismatch`
- `timeline_too_coarse`
- `transition_pair_too_harsh`
- `arc_match_too_strict`
- `judge_overly_permissive`
- `judge_overly_strict`
- `other`

Also record a short free-text note with one concrete example.

## Minimal Audit Template

Use the following structure for each movie:

```md
## <movie_id>

- gold_fact_recall:
  - label:
  - note:
- pred_fact_precision:
  - label:
  - note:
- fact_f1:
  - label:
  - note:
- pred_transition_coherence:
  - label:
  - note:
- arc_narrative_aspect_correctness:
  - label:
  - note:

Failure modes:
- ...

Representative evidence:
- ...
```

## How To Use Validation Results

Human validation is not meant to replace the automatic metrics. It is meant to:

- justify why the chosen metrics are acceptable for research use
- identify consistent weaknesses
- support explicit discussion of metric limitations in the paper or appendix

If a metric is useful in most audited cases but imperfect in some edge cases, that is acceptable as long as the edge cases are documented clearly.
