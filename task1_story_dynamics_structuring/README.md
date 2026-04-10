# Task 1: Story Dynamics Structuring

This folder contains the current release package for STAGE Task 1.

## Scope

Task 1 releases structured gold assets for screenplay-level story dynamics centered on focal characters. The current release includes:

- per-movie focal-character timelines
- per-movie cross-scene arcs grounded in those timelines

This package is intended as the released gold asset layer, not as the full builder audit dump.

## Current Release Stats

- Movies: `151`
- English movies: `109`
- Chinese movies: `42`
- Total focal-character timeline groups: `421`
- Total timeline nodes: `4,720`
- Total cross-scene arcs: `1,610`

## Directory Layout

- `data/per_movie/<language>/<movie_id>/gold_character_timelines_v1.json`
  Per-movie focal-character timeline asset.
- `data/per_movie/<language>/<movie_id>/gold_cross_scene_arcs_v1.json`
  Per-movie cross-scene arc asset.
- `manifests/movie_manifest.csv`
  One row per movie with counts and relative paths.
- `manifests/dataset_summary.json`
  Dataset-level summary counts.
- `schemas/gold_character_timelines_v1.schema.json`
  Field-level schema for timeline files.
- `schemas/gold_cross_scene_arcs_v1.schema.json`
  Field-level schema for arc files.
- `scripts/stage_build_gold_character_timelines.py`
  Builder script used to construct the released assets.
- `scripts/stage_batch_build_gold_character_timelines.py`
  Batch runner for the builder pipeline.

## Notes

- This release package contains final gold outputs only. Builder audit files are intentionally excluded from the release directory.
- The current assets are organized around focal characters aligned with the Task 3 character set where available.
- Evaluation scripts and solver baselines are not included in this repository snapshot yet.
