# STAGE: Evolving Stories

STAGE is a bilingual benchmark for evaluating how language models understand
narrative change across scenes and act consistently with an evolving story.
This release contains task assets for 151 movies in paired ordinary (`STAGE`)
and identity-anonymized (`STAGE_Anon`) conditions: 42 Chinese and 109 English.

## Screenplay policy

Full screenplay text is not distributed in this repository. No `script.json`
file is included. The two movie-info CSV files record screenplay source URLs
where available. Users are responsible for obtaining source text under the
applicable provider terms and preparing the local `script.json` required by
screenplay-dependent evaluation runners.

`convert_movie_csv_to_json.py` converts the movie-info CSV files into JSON
metadata arrays. It does not download or distribute screenplay text.

## Layout

```text
STAGE_v0/
  STAGE/                  ordinary task assets for 151 movies
  STAGE_Anon/             paired identity-anonymized task assets
  chinese_movie_info.csv  metadata for 42 Chinese movies
  english_movie_info.csv  metadata for 109 English movies
  convert_movie_csv_to_json.py
  evaluation/             prediction and evaluation code
  EVALUATION.md
  manifest.json
```

Each movie directory contains `info.json`, two Task I files, one Task II file,
and three Task III files. The public package includes no private identity
mapping.

## Counts

| Asset | Count |
|---|---:|
| Movies | 151 |
| Task I focal trajectories | 434 |
| Task I checkpoints | 2,925 |
| Task II questions | 5,010 |
| Task III role assets | 727 |
| Task III single-turn instances | 5,425 |
| Task III multi-turn episodes | 866 |
| Task III multi-turn turns | 2,598 |

Counts and SHA-256 checksums are recorded in `manifest.json`. Prediction and
evaluation entry points, bilingual prompts, schemas, example configs, and the
reference environment are provided under `evaluation/`; see `EVALUATION.md`.

## Rights

Screenplay rights remain with their respective holders. This repository does
not include full screenplay text.
