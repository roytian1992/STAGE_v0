# STAGE v0

This repository stores the released STAGE data assets only.

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

This repository does not include builder code, evaluation code, or intermediate audit files.

Additional notes:

- Task 3 single-turn and multi-turn formats are documented in `docs/task_3_in_script_character_role_play.md`.
- A packaging summary for Task 3 is stored in `task_3_release_summary.json`.
