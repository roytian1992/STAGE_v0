#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_STAGE_ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Task3 asset character names to canonical Task1 character_name values."
    )
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--apply", action="store_true", help="Actually rewrite files. Default is dry-run.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("/vepfs-mlp2/c20250513/241404044/users/roytian/tmp_task3_character_name_normalization_report_20260422.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_name(text: Any) -> str:
    return " ".join(str(text or "").replace("_", " ").split()).strip().lower()


def canonical_map_for_movie(movie_dir: Path) -> Dict[str, str]:
    payload = load_json(movie_dir / "task_1_character_timelines.json")
    mapping: Dict[str, str] = {}
    for row in payload.get("focal_character_timelines", []) or []:
        canonical = str(row.get("character_name") or "").strip()
        if not canonical:
            continue
        mapping[normalize_name(canonical)] = canonical
        for alias in row.get("aliases", []) or []:
            alias_text = str(alias).strip()
            if alias_text:
                mapping[normalize_name(alias_text)] = canonical
    return mapping


def rewrite_scalar(
    *,
    item: Dict[str, Any],
    field: str,
    canonical_map: Dict[str, str],
) -> Tuple[int, Dict[str, str] | None]:
    current = str(item.get(field) or "").strip()
    if not current:
        return 0, None
    canonical = canonical_map.get(normalize_name(current))
    if not canonical or canonical == current:
        return 0, None
    item[field] = canonical
    return 1, {"from": current, "to": canonical}


def rewrite_entries(
    *,
    items: List[Dict[str, Any]],
    field: str,
    canonical_map: Dict[str, str],
) -> Tuple[int, List[Dict[str, str]]]:
    changed = 0
    examples: List[Dict[str, str]] = []
    for item in items:
        delta, example = rewrite_scalar(item=item, field=field, canonical_map=canonical_map)
        changed += delta
        if example and len(examples) < 5:
            examples.append(example)
    return changed, examples


def process_role_assets(movie_dir: Path, canonical_map: Dict[str, str], apply: bool) -> Dict[str, Any]:
    path = movie_dir / "task_3_role_assets.json"
    if not path.exists():
        return {"changes": 0, "examples": []}
    payload = load_json(path)
    changes = 0
    examples: List[Dict[str, str]] = []
    for role in payload.get("roles", []) or []:
        for field in ["character_name"]:
            delta, example = rewrite_scalar(item=role, field=field, canonical_map=canonical_map)
            changes += delta
            if example and len(examples) < 5:
                examples.append(example)
        alignment = role.get("task1_alignment")
        if isinstance(alignment, dict):
            delta, example = rewrite_scalar(item=alignment, field="task1_character_name", canonical_map=canonical_map)
            changes += delta
            if example and len(examples) < 5:
                examples.append(example)
        for relation in role.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            delta, example = rewrite_scalar(item=relation, field="target_character", canonical_map=canonical_map)
            changes += delta
            if example and len(examples) < 5:
                examples.append(example)
    if changes and apply:
        dump_json(path, payload)
    return {"changes": changes, "examples": examples}


def process_movie(movie_dir: Path, apply: bool) -> Dict[str, Any]:
    timeline_path = movie_dir / "task_1_character_timelines.json"
    role_assets_path = movie_dir / "task_3_role_assets.json"
    if not timeline_path.exists():
        return {"movie_dir": str(movie_dir), "skipped": "missing_task_1_character_timelines"}
    if not role_assets_path.exists():
        return {"movie_dir": str(movie_dir), "skipped": "missing_task_3_role_assets"}

    canonical_map = canonical_map_for_movie(movie_dir)
    report: Dict[str, Any] = {
        "movie_dir": str(movie_dir),
        "role_assets_changes": 0,
        "role_assets_examples": [],
        "single_turn_changes": 0,
        "multi_turn_changes": 0,
        "single_turn_examples": [],
        "multi_turn_examples": [],
    }

    role_asset_report = process_role_assets(movie_dir, canonical_map, apply)
    report["role_assets_changes"] = role_asset_report["changes"]
    report["role_assets_examples"] = role_asset_report["examples"]

    single_path = movie_dir / "task_3_in_script_character_role_play_single_turn.json"
    if single_path.exists():
        payload = load_json(single_path)
        changed, examples = rewrite_entries(
            items=payload.get("instances", []) or [],
            field="character",
            canonical_map=canonical_map,
        )
        report["single_turn_changes"] = changed
        report["single_turn_examples"] = examples
        if changed and apply:
            dump_json(single_path, payload)

    multi_path = movie_dir / "task_3_in_script_character_role_play_multi_turn.json"
    if multi_path.exists():
        payload = load_json(multi_path)
        changed, examples = rewrite_entries(
            items=payload.get("episodes", []) or [],
            field="character",
            canonical_map=canonical_map,
        )
        report["multi_turn_changes"] = changed
        report["multi_turn_examples"] = examples
        if changed and apply:
            dump_json(multi_path, payload)

    return report


def main() -> None:
    args = parse_args()
    reports: List[Dict[str, Any]] = []
    total_role_assets = 0
    total_single = 0
    total_multi = 0
    touched_movies = 0

    for language_dir in [args.stage_root / "Chinese", args.stage_root / "English"]:
        if not language_dir.exists():
            continue
        for movie_dir in sorted(path for path in language_dir.iterdir() if path.is_dir()):
            report = process_movie(movie_dir, apply=args.apply)
            reports.append(report)
            role_changes = int(report.get("role_assets_changes") or 0)
            single_changes = int(report.get("single_turn_changes") or 0)
            multi_changes = int(report.get("multi_turn_changes") or 0)
            total_role_assets += role_changes
            total_single += single_changes
            total_multi += multi_changes
            if role_changes or single_changes or multi_changes:
                touched_movies += 1

    summary = {
        "stage_root": str(args.stage_root),
        "apply": args.apply,
        "touched_movies": touched_movies,
        "total_role_assets_character_rewrites": total_role_assets,
        "total_single_turn_character_rewrites": total_single,
        "total_multi_turn_character_rewrites": total_multi,
        "movie_reports": reports,
    }
    args.report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "apply": args.apply,
                "touched_movies": touched_movies,
                "total_role_assets_character_rewrites": total_role_assets,
                "total_single_turn_character_rewrites": total_single,
                "total_multi_turn_character_rewrites": total_multi,
                "report_path": str(args.report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
