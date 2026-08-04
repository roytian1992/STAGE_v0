from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .pipeline import CharacterTemporalPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build greenfield character-temporal assets, Task 1, and Task 3 single-turn "
            "from a completed narrative graph run."
        )
    )
    parser.add_argument("--script", type=Path, required=True, help="Path to script.json")
    parser.add_argument(
        "--graph-run-dir",
        type=Path,
        required=True,
        help="Completed and validated narrative graph run directory",
    )
    parser.add_argument("--config", type=Path, required=True, help="Temporal build JSON config")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("temporal_runs"),
        help="Versioned temporal run root (default: ./temporal_runs)",
    )
    parser.add_argument("--run-id", help="Explicit reproducible run ID")
    parser.add_argument("--resume", action="store_true", help="Resume an existing temporal run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = CharacterTemporalPipeline(
        script_path=args.script,
        graph_run_dir=args.graph_run_dir,
        config_path=args.config,
        output_root=args.output_root,
        run_id=args.run_id,
        resume=args.resume,
    )
    print(asyncio.run(pipeline.run()))


if __name__ == "__main__":
    main()
