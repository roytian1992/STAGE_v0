#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


DEFAULT_HTML = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/让子弹飞剧本 - 知乎.html")
DEFAULT_ARTICLE_DIR = Path(
    "/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0/Chinese/ch0a030ed73550489a6cf95bdd8d09a426c8f45a6a"
)
EXPECTED_SCENES = 184
TIME_RE = re.compile(r"^(日|夜|晨|黄昏)\s*([内外])\s*(.*)$")


def pick_source_text(html_path: Path) -> str:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    candidates = []
    selectors = [".RichText", "article"]
    for selector in selectors:
        candidates.clear()
        for node in soup.select(selector):
            text = node.get_text("\n", strip=True)
            if len(text) < 1000:
                continue
            if re.search(r"^1\s*[.．、：:]?\s*日\s*外", text, re.M) and re.search(
                r"^184\s*[.．、：:]?\s*日\s*外\s*青石岭", text, re.M
            ):
                candidates.append(text)
        if candidates:
            return max(candidates, key=len)
    if not candidates:
        raise RuntimeError(f"Could not find screenplay body in {html_path}")
    return max(candidates, key=len)


def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.replace("\xa0", " ").replace("　", " ").strip()
        if line:
            lines.append(line)
    return lines


def is_scene_start(line: str, scene_id: int) -> bool:
    return bool(re.match(rf"^{scene_id}(?=$|[\s.．、：:]|日|夜|晨|黄)", line))


def parse_heading(lines: list[str], idx: int, scene_id: int) -> tuple[str, int]:
    line = lines[idx]
    match = re.match(rf"^{scene_id}\s*[.．、：:]?\s*(.*)$", line)
    if not match:
        raise ValueError(f"Unexpected heading format for scene {scene_id}: {line!r}")
    rest = match.group(1).strip()

    title_time: str
    title_space: str
    title_loc: str
    consumed = 1

    if not rest:
        if idx + 2 >= len(lines):
            raise ValueError(f"Incomplete multiline heading for scene {scene_id}")
        match = TIME_RE.match(lines[idx + 1])
        if not match:
            raise ValueError(f"Missing time/int-ext line for scene {scene_id}: {lines[idx + 1]!r}")
        title_time, title_space, title_loc = match.groups()
        consumed = 2
        if not title_loc:
            title_loc = lines[idx + 2]
            consumed = 3
    else:
        match = TIME_RE.match(rest)
        if not match:
            raise ValueError(f"Unrecognized heading body for scene {scene_id}: {line!r}")
        title_time, title_space, title_loc = match.groups()
        if not title_loc:
            if idx + 1 >= len(lines):
                raise ValueError(f"Missing location line for scene {scene_id}")
            title_loc = lines[idx + 1]
            consumed = 2

    location = title_loc.replace("／", "/").replace(" ", "")
    title = f"{scene_id}、{title_time}.{title_space}.{location}"
    return title, idx + consumed


def parse_scenes(text: str) -> list[dict[str, str | int]]:
    lines = clean_lines(text)
    start_idx = next((i for i, line in enumerate(lines) if is_scene_start(line, 1)), None)
    if start_idx is None:
        raise RuntimeError("Could not find scene 1 in extracted text")

    heading_indices: dict[int, int] = {}
    cursor = start_idx
    for scene_id in range(1, EXPECTED_SCENES + 1):
        found = next((i for i in range(cursor, len(lines)) if is_scene_start(lines[i], scene_id)), None)
        if found is None:
            raise RuntimeError(f"Could not find scene {scene_id}")
        heading_indices[scene_id] = found
        cursor = found + 1

    scenes: list[dict[str, str | int]] = []
    for scene_id in range(1, EXPECTED_SCENES + 1):
        heading_idx = heading_indices[scene_id]
        title, content_start = parse_heading(lines, heading_idx, scene_id)
        next_heading = heading_indices.get(scene_id + 1, len(lines))
        content = "\n".join(lines[content_start:next_heading]).strip()
        scenes.append(
            {
                "_id": scene_id,
                "title": title,
                "subtitle": "",
                "content": content,
            }
        )
    return scenes


def write_script(article_dir: Path, scenes: list[dict[str, str | int]]) -> Path:
    output_path = article_dir / "script.json"
    output_path.write_text(
        json.dumps(scenes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--article-dir", type=Path, default=DEFAULT_ARTICLE_DIR)
    args = parser.parse_args()

    text = pick_source_text(args.html)
    scenes = parse_scenes(text)
    if len(scenes) != EXPECTED_SCENES:
        raise RuntimeError(f"Expected {EXPECTED_SCENES} scenes, got {len(scenes)}")
    output_path = write_script(args.article_dir, scenes)
    print(json.dumps({"script_path": str(output_path), "scene_count": len(scenes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
