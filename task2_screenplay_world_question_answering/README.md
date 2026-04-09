# Task 2: Screenplay-World Question Answering

This folder contains the current release package for STAGE Task 2.

## Scope

Task 2 evaluates question answering over a screenplay-grounded story world. The released QA tables merge:

- broad-coverage scene-grounded QA
- broad-coverage event-chain QA
- supplementary cross-scene QA

## Current Release Stats

- Movies: `151`
- English movies: `109`
- Chinese movies: `42`
- Total QA rows: `7,528`

## Directory Layout

- `data/global/question_pairs_merged_all.csv`
  Global QA table across all movies.
- `data/per_movie/<language>/<movie_id>/question_pairs_merged.csv`
  Per-movie QA table.
- `manifests/movie_manifest.csv`
  One row per movie with language, row count, and relative file path.
- `schemas/question_pairs_merged.schema.json`
  Field-level schema for released QA tables.
- `scripts/stage_merge_qa_tables.py`
  Script used to merge broad-coverage QA and supplementary cross-scene QA into the released table format.

## Released Columns

- `id`
  Row id. In the global table this is prefixed by `movie_id`.
- `related_scenes`
  Scene title or a `|`-joined scene list for multi-scene questions.
- `question`
  Natural-language question.
- `answer`
  Ground-truth answer.
- `related_context`
  Supporting context kept in compact textual form.
- `question_type`
  Task-internal question category label stored in the current asset version.
- `question_source`
  QA source family. Current values include:
  - `broad_coverage_scene`
  - `broad_coverage_event_chain`
  - `supplementary_cross_scene`

## Notes

- This release package contains QA assets only. Evaluation scripts, baseline pipelines, and the final public task card are not yet included in this repository snapshot.
- The current `question_type` values are the labels stored in the released CSV assets. If a remapped taxonomy is finalized later, it should be published as a versioned update rather than silently overwriting this release.
