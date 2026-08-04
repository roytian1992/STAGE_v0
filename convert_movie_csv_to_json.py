#!/usr/bin/env python3
"""Convert the movie metadata CSV files in this directory to JSON arrays."""

import csv
import json
from pathlib import Path


FILES = (
    ("chinese_movie_info.csv", "chinese_movie_info.json"),
    ("english_movie_info.csv", "english_movie_info.json"),
)


def convert_csv_to_json(csv_path: Path, json_path: Path) -> int:
    """Read a UTF-8 CSV with headers and write its rows as a JSON array."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")

    return len(rows)

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    for csv_name, json_name in FILES:
        count = convert_csv_to_json(base_dir / csv_name, base_dir / json_name)
        print(f"{csv_name} -> {json_name}: {count} records")


if __name__ == "__main__":
    main()
