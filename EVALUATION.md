# STAGE Prediction and Evaluation

The `evaluation/` snapshot contains the runtime modules, bilingual prompts,
JSON schemas, and entry points used by the benchmark. The ordinary and
anonymous conditions must be run independently with the same frozen model
configuration. Prediction and judge models are separate configuration fields;
formal LLM-as-judge results require an independent judge model.

Full screenplay files are not included in this release. Obtain each screenplay
from the source URL recorded in the movie-info CSV files and prepare the local
`script.json` required by screenplay-dependent runners before execution.

## Task 1

Run one movie at a time. The runner extracts bounded entity-centric memory from
the screenplay, predicts every checkpoint under both previous-state settings,
and refuses screenplay or asset hash drift.

```bash
PYTHONPATH=evaluation/src python evaluation/scripts/run_task1_state_update_predictions.py \
  --reference-asset STAGE/English/<movie_id>/task_1_reference_state_update.json \
  --autoregressive-asset STAGE/English/<movie_id>/task_1_autoregressive_state_update.json \
  --script STAGE/English/<movie_id>/script.json \
  --config evaluation/configs/stage_release_evaluation.example.json \
  --output-root runs/task1/<movie_id> --workers 32 --dry-run
```

Remove `--dry-run` after preflight. Evaluation reports current-state coverage,
development coverage, evidence validity, and setting-level aggregates. Use
`run_task1_state_update_evaluation.py` with an independently configured judge.

## Task 2

Task 2 is closed-book question answering: the actor sees the question only.
Materialize a question manifest from the selected condition, run direct
predictions, and evaluate the answer against the hidden reference answer with
an independent judge. Evidence scenes and claims are evaluator-only fields and
must not be exposed to the actor.

```bash
PYTHONPATH=evaluation/src python evaluation/scripts/prepare_stage_task2_questions.py \
  --ordinary-root STAGE --anonymous-root STAGE_Anon \
  --output-root runs/task2/questions

PYTHONPATH=evaluation/src python evaluation/scripts/run_manifest_task2_direct_predictions.py \
  --questions-manifest runs/task2/questions/manifest.json \
  --config evaluation/configs/stage_release_task2.example.json \
  --output-root runs/task2/predictions --workers 64 --preflight-only
```

The formal score is answer correctness, aggregated first within a movie and
then equally across movies. Report ordinary and anonymous conditions
separately, with breakdowns by the six released `question_type` values. The
evolution construction family is not exposed as a replacement task type.

## Task 3

The actor receives the current user turn, local interaction context, and the
checkpoint-bounded role snapshot resolved from `task_3_role_assets.json`.
`evaluator_reference` is never included in the actor prompt. Formal evaluation
reports four 1--5 dimensions separately: character fidelity, memory
faithfulness, boundary compliance, and response naturalness. It also reports
future leakage, unknown-fact hallucination, and stance incompatibility rates.

All 151 screenplays support both checkpoint-bounded single-turn evaluation and
the retained legacy multi-turn evaluation.

```bash
PYTHONPATH=evaluation/src python evaluation/scripts/run_stage_task3_single_predictions.py \
  --release-root STAGE \
  --config evaluation/configs/stage_release_evaluation.example.json \
  --output-root runs/task3/ordinary --workers 64 --dry-run
```

The runner detects the reviewed core adapter and the modern checkpoint role
schema per movie. Remove `--dry-run` only after the complete 151-movie prompt
preflight passes.

Run the retained multi-turn task separately:

```bash
PYTHONPATH=evaluation/src python evaluation/scripts/run_task3_predictions.py \
  --release-root STAGE \
  --condition ordinary --mode multi \
  --config evaluation/configs/stage_release_evaluation.example.json \
  --output-root runs/task3/all-multi --workers 64 --dry-run
```

## Context Budget

The released protocol uses a 24,000-token context window. Each call reserves
its output budget and a safety margin before materialization. Task 1 processes
the screenplay as lossless scene-aligned chunks and maintains bounded shared
entity memory; Task 3 resolves only checkpoint-visible role state and memory.
Preflight is zero-call and fails when any fully materialized prompt exceeds the
configured budget.

## Aggregation

Do not pool every checkpoint, question, or dialogue as if they were independent
movies. Report item-level diagnostics, character-level summaries where
applicable, and equal-weight movie macro scores as the primary benchmark view.
Keep ordinary and anonymous conditions separate and report their paired gap.
