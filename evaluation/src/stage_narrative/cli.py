from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .pipeline import NarrativeGraphPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an auditable entity KG and event-to-episode-to-storyline hierarchy."
    )
    parser.add_argument("--script", type=Path, required=True, help="Path to one script.json")
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON config")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Root for versioned run directories (default: ./runs)",
    )
    parser.add_argument("--run-id", help="Explicit reproducible run identifier")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume checkpoints in an existing movie/run-id directory",
    )
    parser.add_argument(
        "--kg-only",
        action="store_true",
        help="Stop after the semantic KG passes the KG quality gate",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = NarrativeGraphPipeline(
        script_path=args.script,
        config_path=args.config,
        output_root=args.output_root,
        run_id=args.run_id,
        resume=args.resume,
        kg_only=args.kg_only,
    )
    run_dir = asyncio.run(pipeline.run())
    print(run_dir)


if __name__ == "__main__":
    main()
