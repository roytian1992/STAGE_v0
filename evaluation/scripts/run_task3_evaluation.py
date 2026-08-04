#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage_narrative.evaluation.task3_runner import run_task3_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal Task 3 single-turn evaluation.")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--context-packs", type=Path, required=True)
    parser.add_argument("--gold-rubrics", type=Path, required=True)
    parser.add_argument("--pair-groups", type=Path, required=True)
    parser.add_argument("--pair-annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--evaluation-mode",
        choices=("formal_independent_evaluation", "self_judge_diagnostic"),
        default="formal_independent_evaluation",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(
        asyncio.run(
            run_task3_evaluation(
                instances_path=args.instances,
                context_packs_path=args.context_packs,
                gold_rubrics_path=args.gold_rubrics,
                pair_groups_path=args.pair_groups,
                pair_annotations_path=args.pair_annotations,
                prediction_path=args.predictions,
                config_path=args.config,
                output_dir=args.output_dir,
                workers=args.workers,
                evaluation_mode=args.evaluation_mode,
                resume=args.resume,
                preflight_only=args.preflight_only,
            )
        )
    )


if __name__ == "__main__":
    main()
