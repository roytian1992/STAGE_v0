#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LANG_DIRS = [ROOT / "Chinese", ROOT / "English"]
SCENE_GROUNDING = "Scene Grounding"

PREFIX_RE = re.compile(r"^\s*(-?\d+)\s*[、.．]\s*")
REPEATED_PREFIX_RE = re.compile(r"(?:-?\d+\s*[、.．]\s*){2,}")
SCENE_LABEL_RE = re.compile(
    r"^\s*(?:第?\s*\d+\s*场|\d+\s*场|场\s*\d+)\s*[，,、:：\-.\s/]*"
)
LEADING_SCENE_LABEL_RE = re.compile(r"^\s*(序场[A-Za-z]?|第?\s*-?\d+\s*场[A-Za-z]?)")
WHOLE_SCENE_LABEL_RE = re.compile(r"^\s*(序场[A-Za-z]?|第?\s*-?\d+\s*场[A-Za-z]?)\s*$")
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
PIPE_LIST_SPLIT_RE = re.compile(r"\s*\|\s*(?=(?:-?\d+\s*[、.．]|全片\b))")


@dataclass
class SceneInfo:
    scene_id: int
    raw_title: str
    canonical_title: str
    body_variants: list[str]
    label_variants: list[str]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(text: str) -> str:
    text = (text or "").replace("\ufeff", "").strip()
    text = (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("【", "[")
        .replace("】", "]")
        .replace("／", "/")
        .replace("，", ",")
        .replace("：", ":")
        .replace("；", ";")
        .replace("　", " ")
    )
    text = re.sub(r"\s+", "", text)
    return text.lower()


def unique_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        if not value:
            continue
        key = normalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def strip_first_prefix(text: str) -> str:
    return PREFIX_RE.sub("", (text or "").strip(), count=1).strip()


def strip_scene_label(text: str) -> str:
    cur = (text or "").strip()
    for _ in range(3):
        nxt = SCENE_LABEL_RE.sub("", cur, count=1).strip()
        if nxt == cur:
            break
        cur = nxt
    return cur


def strip_redundant_same_id_prefix(scene_id: int, body: str) -> str:
    cur = (body or "").strip()
    same_id_token = rf"{re.escape(str(scene_id))}"
    patterns = [
        re.compile(rf"^\s*{same_id_token}\s*[、.．]\s*"),
        re.compile(rf"^\s*{same_id_token}\s+(?!场\b)(?=[A-Za-z\u4e00-\u9fff])"),
    ]
    for pattern in patterns:
        nxt = pattern.sub("", cur, count=1).strip()
        if nxt != cur:
            cur = nxt
    return cur


def canonical_scene_title(scene_id: int, raw_title: str) -> str:
    body = strip_first_prefix(raw_title)
    body = strip_redundant_same_id_prefix(scene_id, body)
    prefix = 0 if body.startswith("序") else scene_id
    return f"{prefix}、{body}"


def split_prefix_chain(text: str) -> tuple[list[int], str]:
    prefixes: list[int] = []
    rest = text or ""
    while True:
        match = PREFIX_RE.match(rest)
        if not match:
            break
        prefixes.append(int(match.group(1)))
        rest = rest[match.end() :]
    return prefixes, rest.strip()


def text_to_flex_regex(text: str) -> str:
    parts = []
    for ch in text:
        if ch.isspace():
            parts.append(r"\s*")
        elif ch in "|/,:：;；，、()（）[]【】":
            parts.append(rf"\s*{re.escape(ch)}\s*")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def split_scene_reference_list(value: str, sep: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    if sep == ";":
        return [part.strip() for part in value.split(sep)]
    if sep == "|":
        if " | " in value:
            return [part.strip() for part in value.split(" | ")]
        if PIPE_LIST_SPLIT_RE.search(value):
            return [part.strip() for part in PIPE_LIST_SPLIT_RE.split(value)]
        return [value]
    return [part.strip() for part in value.split(sep)]


def extract_leading_scene_label(text: str) -> str | None:
    match = LEADING_SCENE_LABEL_RE.match((text or "").strip())
    if not match:
        return None
    return match.group(1).strip()


class ArticleRepairer:
    def __init__(self, article_dir: Path) -> None:
        self.article_dir = article_dir
        self.script_path = article_dir / "script.json"
        self.task1_path = article_dir / "task_1_character_timelines.json"
        self.task2_path = article_dir / "task_2_question_answering.csv"
        self.task3_path = article_dir / "task_3_role_assets.json"
        self.scene_infos: dict[int, SceneInfo] = {}
        self.title_by_id: dict[int, str] = {}
        self.full_norm_to_ids: dict[str, list[int]] = defaultdict(list)
        self.body_norm_to_ids: dict[str, list[int]] = defaultdict(list)

    def load_scene_index(self) -> dict[str, int]:
        script = read_json(self.script_path)
        changed = 0
        for scene in script:
            if not isinstance(scene, dict) or "_id" not in scene:
                continue
            scene_id = int(scene["_id"])
            raw_title = str(scene.get("title", ""))
            canonical = canonical_scene_title(scene_id, raw_title)
            if canonical != raw_title:
                scene["title"] = canonical
                changed += 1
            body_variants = unique_keep_order(
                [
                    strip_first_prefix(raw_title),
                    strip_first_prefix(canonical),
                    strip_scene_label(strip_first_prefix(raw_title)),
                    strip_scene_label(strip_first_prefix(canonical)),
                    strip_scene_label(strip_scene_label(strip_first_prefix(raw_title))),
                    strip_scene_label(strip_scene_label(strip_first_prefix(canonical))),
                ]
            )
            label_variants = unique_keep_order(
                [
                    label
                    for label in [
                        extract_leading_scene_label(strip_first_prefix(raw_title)),
                        extract_leading_scene_label(strip_first_prefix(canonical)),
                        extract_leading_scene_label(raw_title),
                        extract_leading_scene_label(canonical),
                        f"第{scene_id}场",
                        f"{scene_id}场",
                    ]
                    if label
                ]
            )
            self.scene_infos[scene_id] = SceneInfo(
                scene_id=scene_id,
                raw_title=raw_title,
                canonical_title=canonical,
                body_variants=body_variants,
                label_variants=label_variants,
            )
            self.title_by_id[scene_id] = canonical

        for scene_id, info in self.scene_infos.items():
            for variant in [info.raw_title, info.canonical_title]:
                key = normalize(variant)
                if scene_id not in self.full_norm_to_ids[key]:
                    self.full_norm_to_ids[key].append(scene_id)
            for variant in info.body_variants:
                key = normalize(variant)
                if scene_id not in self.body_norm_to_ids[key]:
                    self.body_norm_to_ids[key].append(scene_id)

        if changed:
            write_json(self.script_path, script)
        return {"script_title_updates": changed}

    def resolve_segment(self, segment: str) -> int | list[int] | None:
        segment = (segment or "").strip()
        if not segment or segment == "全片":
            return None
        exact = self.full_norm_to_ids.get(normalize(segment), [])
        if len(exact) == 1:
            return exact[0]
        bare_label = self.resolve_bare_scene_label(segment)
        if isinstance(bare_label, int):
            return bare_label
        prefixes, rest = split_prefix_chain(segment)
        search_keys = unique_keep_order(
            [
                segment,
                rest,
                rest.lstrip(" 、.．,，:：;；-/"),
                strip_scene_label(rest),
                strip_scene_label(strip_scene_label(rest)),
            ]
        )
        candidates: list[int] = []
        for key in search_keys:
            candidates.extend(self.body_norm_to_ids.get(normalize(key), []))
        candidates = sorted(set(candidates))
        if prefixes and rest:
            for prefix in reversed(prefixes):
                info = self.scene_infos.get(prefix)
                if not info:
                    continue
                if any(normalize(rest) == normalize(v) for v in info.body_variants):
                    return prefix
            pref_matches = [sid for sid in candidates if sid in prefixes]
            if len(pref_matches) == 1:
                return pref_matches[0]
        if len(candidates) == 1:
            return candidates[0]
        return candidates or None

    def resolve_bare_scene_label(self, segment: str) -> int | list[int] | None:
        match = WHOLE_SCENE_LABEL_RE.match((segment or "").strip())
        if not match:
            return None
        label = normalize(match.group(1))
        candidates = [
            scene_id
            for scene_id, info in self.scene_infos.items()
            if any(normalize(variant) == label for variant in info.label_variants)
        ]
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            return candidates[0]
        return candidates or None

    def resolve_scene_list(self, value: str, sep: str) -> tuple[str, int]:
        value = (value or "").strip()
        if not value or value == "全片":
            return value, 0
        parts = split_scene_reference_list(value, sep)
        if not parts:
            return value, 0
        ambiguous_keys = Counter(normalize(strip_scene_label(strip_first_prefix(part))) for part in parts)
        seen_ambiguous: Counter[str] = Counter()
        unresolved = 0
        resolved_parts: list[str] = []
        for part in parts:
            part = part.strip()
            result = self.resolve_segment(part)
            if isinstance(result, int):
                resolved_parts.append(self.title_by_id[result])
                continue
            if isinstance(result, list) and result:
                key = normalize(strip_scene_label(strip_first_prefix(part)))
                if ambiguous_keys[key] <= len(result):
                    resolved_parts.append(self.title_by_id[result[seen_ambiguous[key]]])
                    seen_ambiguous[key] += 1
                    continue
            unresolved += 1
            resolved_parts.append(part)
        return f" {sep} ".join(resolved_parts), unresolved

    def replace_candidate_mentions(self, text: str, scene_ids: list[int]) -> str:
        result = text

        def replace_bracket(match: re.Match[str]) -> str:
            inner = match.group(1)
            resolved = self.resolve_segment(inner)
            if isinstance(resolved, int):
                return f"[{self.title_by_id[resolved]}]"
            return match.group(0)

        result = BRACKET_RE.sub(replace_bracket, result)
        for scene_id in scene_ids:
            info = self.scene_infos.get(scene_id)
            if not info:
                continue
            for variant in unique_keep_order([info.raw_title, info.canonical_title]):
                if normalize(variant) == normalize(info.canonical_title):
                    continue
                pattern = re.compile(text_to_flex_regex(variant))
                result = pattern.sub(lambda _match, repl=info.canonical_title: repl, result)
            for variant in info.body_variants:
                if not variant:
                    continue
                pattern = re.compile(
                    rf"(?:-?\d+\s*[、.．]\s*)+{text_to_flex_regex(variant)}"
                )
                result = pattern.sub(lambda _match, repl=info.canonical_title: repl, result)
            for label in info.label_variants:
                pattern = re.compile(
                    rf"(?<![0-9A-Za-z第、，,.．;；]){text_to_flex_regex(label)}"
                    rf"(?=(?:中|里|场景|,|，|。|\]|）|\)|\s|$))"
                )
                result = pattern.sub(
                    lambda _match, repl=f"[{info.canonical_title}]": repl, result
                )
        return result

    def repair_task1_or_task3(self, path: Path) -> dict[str, int]:
        if not path.exists():
            return {"scene_title_updates": 0}
        data = read_json(path)
        updates = 0

        def walk(node: Any) -> None:
            nonlocal updates
            if isinstance(node, dict):
                if "scene_title" in node:
                    scene_key = node.get("scene_order")
                    if scene_key is None:
                        scene_key = node.get("scene_id")
                    try:
                        scene_id = int(scene_key)
                    except (TypeError, ValueError):
                        scene_id = None
                    canonical = self.title_by_id.get(scene_id) if scene_id is not None else None
                    if canonical and node.get("scene_title") != canonical:
                        node["scene_title"] = canonical
                        updates += 1
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(data)
        if updates:
            write_json(path, data)
        return {"scene_title_updates": updates}

    def repair_task2(self) -> dict[str, int]:
        if not self.task2_path.exists():
            return {
                "related_scenes_updates": 0,
                "scene_grounding_answer_updates": 0,
                "question_or_evidence_updates": 0,
                "unresolved_segments": 0,
            }
        rows, fieldnames = read_csv(self.task2_path)
        related_updates = 0
        answer_updates = 0
        text_updates = 0
        unresolved_total = 0
        for row in rows:
            related_before = row.get("related_scenes", "") or ""
            related_after, unresolved = self.resolve_scene_list(related_before, ";")
            unresolved_total += unresolved
            if related_after != related_before:
                row["related_scenes"] = related_after
                related_updates += 1

            candidate_scene_ids: list[int] = []
            if row.get("related_scenes"):
                for part in split_scene_reference_list(row["related_scenes"], ";"):
                    resolved = self.resolve_segment(part)
                    if isinstance(resolved, int):
                        candidate_scene_ids.append(resolved)

            answer_before = row.get("answer", "") or ""
            should_repair_answer = (
                (row.get("question_type") or "").strip() == SCENE_GROUNDING
                or "|" in answer_before
                or bool(REPEATED_PREFIX_RE.search(answer_before))
            )
            if should_repair_answer:
                answer_after, unresolved = self.resolve_scene_list(answer_before, "|")
                unresolved_total += unresolved
                if answer_after != answer_before:
                    row["answer"] = answer_after
                    answer_updates += 1
                for part in split_scene_reference_list(row.get("answer", ""), "|"):
                    resolved = self.resolve_segment(part)
                    if isinstance(resolved, int):
                        candidate_scene_ids.append(resolved)

            candidate_scene_ids = sorted(set(candidate_scene_ids))
            for field in ["question", "evidence_or_reason"]:
                before = row.get(field, "") or ""
                after = self.replace_candidate_mentions(before, candidate_scene_ids)
                if after != before:
                    row[field] = after
                    text_updates += 1

        if related_updates or answer_updates or text_updates:
            write_csv(self.task2_path, rows, fieldnames)
        return {
            "related_scenes_updates": related_updates,
            "scene_grounding_answer_updates": answer_updates,
            "question_or_evidence_updates": text_updates,
            "unresolved_segments": unresolved_total,
        }


def iter_articles() -> list[Path]:
    articles: list[Path] = []
    for lang_dir in LANG_DIRS:
        if not lang_dir.exists():
            continue
        for article_dir in sorted(path for path in lang_dir.iterdir() if path.is_dir()):
            if (article_dir / "script.json").exists():
                articles.append(article_dir)
    return articles


def run_check() -> dict[str, int]:
    stats = Counter()
    for article_dir in iter_articles():
        repairer = ArticleRepairer(article_dir)
        script = read_json(repairer.script_path)
        title_by_id = {
            int(scene["_id"]): canonical_scene_title(int(scene["_id"]), str(scene.get("title", "")))
            for scene in script
            if isinstance(scene, dict) and "_id" in scene
        }
        for scene in script:
            if not isinstance(scene, dict) or "_id" not in scene:
                continue
            expected = canonical_scene_title(int(scene["_id"]), str(scene.get("title", "")))
            if scene.get("title") != expected:
                stats["script_title_mismatches"] += 1
        for path in [repairer.task1_path, repairer.task3_path]:
            if not path.exists():
                continue
            data = read_json(path)

            def walk(node: Any) -> None:
                if isinstance(node, dict):
                    if "scene_title" in node:
                        scene_key = node.get("scene_order")
                        if scene_key is None:
                            scene_key = node.get("scene_id")
                        try:
                            scene_id = int(scene_key)
                        except (TypeError, ValueError):
                            scene_id = None
                        if scene_id in title_by_id and node.get("scene_title") != title_by_id[scene_id]:
                            stats["task_scene_title_mismatches"] += 1
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)

            walk(data)
        if repairer.task2_path.exists():
            rows, _ = read_csv(repairer.task2_path)
            for row in rows:
                for field in ["related_scenes", "question", "answer"]:
                    if REPEATED_PREFIX_RE.search((row.get(field, "") or "")):
                        stats["task2_repeated_prefix_hits"] += 1
    return dict(stats)


def run_apply() -> dict[str, int]:
    stats = Counter()
    for article_dir in iter_articles():
        repairer = ArticleRepairer(article_dir)
        for key, value in repairer.load_scene_index().items():
            stats[key] += value
        for key, value in repairer.repair_task1_or_task3(repairer.task1_path).items():
            stats[f"task1_{key}"] += value
        for key, value in repairer.repair_task1_or_task3(repairer.task3_path).items():
            stats[f"task3_{key}"] += value
        for key, value in repairer.repair_task2().items():
            stats[key] += value
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stats = run_check() if args.check else run_apply()
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
