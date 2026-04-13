# STAGE v0

This repository stores the released STAGE data assets together with a lightweight Task 1 rebuild and evaluation pipeline.

Current release format:

- `English/<movie_id>/`
- `Chinese/<movie_id>/`

Each movie directory contains:

- `script.json`
- `task_1_character_timelines.json`
- `task_1_cross_scene_arcs.json`
- `task_2_question_answering.csv`
- `task_3_in_script_character_role_play_single_turn.json`
- `task_3_in_script_character_role_play_multi_turn.json`

Task 3 is packaged at the movie level and aggregates all selected focal roles for that screenplay.

Included code:

- `tools/task1/pipeline.py`
  One entry point for Task 1 single-movie rebuild, single-movie evaluation, and batch experiments.

Internal support modules for Task 1 are kept under `tools/task1/` as local implementation details.

Additional notes:

- Task 3 single-turn and multi-turn formats are documented in `docs/task_3_in_script_character_role_play.md`.
- A packaging summary for Task 3 is stored in `task_3_release_summary.json`.
- Task 1 rebuild and evaluation usage is documented in `docs/task_1_rebuild_and_eval.md`.
