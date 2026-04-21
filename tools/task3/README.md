# Task3 Evaluation Tooling

This folder is intentionally slimmed down for release / push.
Only Task3 evaluation entrypoints and their runtime helpers are retained here.

## Retained Files

Single-turn evaluation:

- `run_manifest40_single_turn_eval_batch.py`
- `run_task3_single_turn_eval.py`
- `run_task3_role_eval_matrix.py`

Multi-turn evaluation:

- `run_task3_multi_turn_batch_eval.py`
- `run_task3_multi_turn_episode_eval.py`

Runtime helpers used by the evaluators:

- `task3_runtime_loader.py`
- `task3_llm_fallback.py`

## Not Retained Here

Asset construction, rebuilding, migration, repair, and other data-generation
scripts are intentionally excluded from this push-oriented copy of `STAGE_v0`.
