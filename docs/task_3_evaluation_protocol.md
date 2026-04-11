# Task 3 Evaluation Protocol

## Goal

This document defines the recommended evaluation contract for Task 3:
`In-Script Character Role Play`.

The protocol is designed to support two benchmark uses at once:

- character-faithful role-play evaluation;
- memory-aware role-play evaluation.

It is intentionally focused on the experiment contract rather than on a full
public evaluation runner.

## Canonical Evaluation Views

Task 3 should be reported under two interaction settings:

1. `single_turn`
2. `multi_turn`

Task 3 should also be reported under three prompt conditions:

1. `persona_only`
2. `persona_with_memory`
3. `persona_with_memory_and_relations`

The most useful comparison is not only overall score, but also the delta across
these three prompt conditions.

## Canonical Inputs

### Single-Turn

For each released instance, use:

- `input.persona_card`
- `input.memory_context`
- `input.relation_context`
- `input.dialogue_history`
- `input.current_user_turn`

The model should produce one in-character response.

### Multi-Turn

For each released episode, evaluate the full sequence of turns in order.

At each turn, use:

- `shared_input.persona_card`
- current turn `input.memory_context`
- current turn `input.relation_context`
- current turn `input.dialogue_history_template`
- current turn `input.current_user_turn`

The model should produce one in-character response per turn.

The evaluator should preserve prior model responses across turns exactly as they
were generated during the episode.

## Prediction Format

### Single-Turn Prediction

Recommended row format:

```json
{
  "instance_id": "...",
  "response": "..."
}
```

### Multi-Turn Prediction

Recommended row format:

```json
{
  "episode_instance_id": "...",
  "turn_responses": [
    {"turn_index": 1, "response": "..."},
    {"turn_index": 2, "response": "..."},
    {"turn_index": 3, "response": "..."}
  ]
}
```

## Primary Metrics

The canonical Task 3 report should include the following metrics.

### Single-Turn

- `character_fidelity`
- `memory_faithfulness`
- `boundary_compliance`
- `response_naturalness`
- `single_turn_score`

### Multi-Turn

- `character_fidelity`
- `memory_faithfulness`
- `boundary_compliance`
- `response_naturalness`
- `cross_turn_consistency`
- `multi_turn_score`

## Metric Definitions

### Character Fidelity

Does the response sound like the target character and remain compatible with:

- the persona card,
- expected stance,
- expected style,
- persona risk type.

This is the core persona-consistency metric.

### Memory Faithfulness

Does the response use the provided script-grounded memory correctly without:

- ignoring clearly relevant memory,
- fabricating unsupported memory,
- mixing up events, people, or motivations.

This is the core memory-module metric.

### Boundary Compliance

Does the response stay inside the provided knowledge boundary?

The judge should penalize:

- unsupported facts,
- implausible certainty beyond the evidence,
- contradiction of `forbidden` constraints,
- explicit conflict with `contradicting_facts`.

### Response Naturalness

Is the response readable, coherent, and plausible as character speech?

This is not a style-preference metric. It should focus on whether the answer is
usable as a role-play response rather than on literary quality.

### Cross-Turn Consistency

Multi-turn only.

Does the character remain stable across turns while responding to cumulative
follow-up pressure?

The judge should focus on:

- consistency of stance,
- consistency of self-knowledge,
- consistency of relation framing,
- consistency of remembered events,
- absence of persona drift.

## Judge Scale

Recommended judge scale for each dimension:

- integer score from `1` to `5`

Interpretation:

- `5`: clearly strong
- `4`: good with minor weakness
- `3`: mixed / partially acceptable
- `2`: weak
- `1`: clearly failing

## Aggregate Scores

### Single-Turn Score

```text
single_turn_score = mean(
  character_fidelity,
  memory_faithfulness,
  boundary_compliance,
  response_naturalness
)
```

### Multi-Turn Score

```text
multi_turn_score = mean(
  character_fidelity,
  memory_faithfulness,
  boundary_compliance,
  response_naturalness,
  cross_turn_consistency
)
```

## LLM-as-Judge Recommendation

Canonical setting:

- `LLM-as-judge = enabled`
- `judge_runs = 3`

Per-sample outputs should retain:

- per-dimension scores
- short rationale
- optional violation flags

Recommended report-level aggregates:

- mean by dimension
- mean total score
- standard deviation across judge runs

## Recommended Judge Output Schema

### Single-Turn Judge Row

```json
{
  "instance_id": "...",
  "character_fidelity": 4,
  "memory_faithfulness": 5,
  "boundary_compliance": 5,
  "response_naturalness": 4,
  "single_turn_score": 4.5,
  "violation_flags": [],
  "rationale": "..."
}
```

### Multi-Turn Judge Row

```json
{
  "episode_instance_id": "...",
  "character_fidelity": 4,
  "memory_faithfulness": 4,
  "boundary_compliance": 5,
  "response_naturalness": 4,
  "cross_turn_consistency": 4,
  "multi_turn_score": 4.2,
  "violation_flags": [],
  "rationale": "..."
}
```

## Recommended Violation Flags

Recommended optional binary flags:

- `unsupported_memory`
- `persona_drift`
- `relation_mismatch`
- `boundary_violation`
- `cross_turn_self_contradiction`

These should be treated as diagnostic outputs, not as the primary score.

## Canonical Reporting Slices

At minimum, report results by:

- language
- interaction format
- prompt condition

Recommended summary slices:

- overall single-turn
- overall multi-turn
- `persona_only`
- `persona_with_memory`
- `persona_with_memory_and_relations`
- English vs. Chinese

## Recommended Baseline Comparisons

The most meaningful Task 3 comparison is:

1. no explicit memory support
2. memory-supported role-play
3. memory + relation-supported role-play

If an external memory system is being evaluated, keep the answer model fixed and
change only the memory/retrieval mechanism when possible.

## Current Release Position

For `STAGE_v0`, the data assets are already packaged for Task 3.

The remaining work is mainly protocol-side:

- implement a public evaluation runner,
- finalize judge prompt templates,
- optionally prepare a human-validated subset.
