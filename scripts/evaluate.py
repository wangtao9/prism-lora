#!/usr/bin/env python3
"""Backward-compatible wrapper: delegates to the modular eval/ package.

The evaluation logic has been split into:
  - eval.judge_eval    (memory conflict detection)
  - eval.poet_eval     (poetry generation)
  - eval.cross_eval    (cross-domain evaluation)
  - eval.plot_results  (visualization)

This script preserves the original CLI interface for existing invocations
(e.g. run_all.sh, CLI args like --task, --port, --report).
"""

import argparse
import asyncio
import os
import sys

# Ensure the project root is on sys.path so `eval` package is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from eval.judge_eval import run_judge_comparison, load_test_data as load_judge_data
from eval.poet_eval import run_poet_comparison, load_test_data as load_poet_data
from eval.cross_eval import run_cross_eval
from eval.plot_results import generate_all_plots

RESULTS_DIR = os.path.join(BASE_DIR, "results")


async def run_full_evaluation(base_url: str) -> None:
    """Run all evaluations: judge, poet, cross-eval, then generate plots."""
    await run_judge_comparison(base_url, RESULTS_DIR)
    await run_poet_comparison(base_url, RESULTS_DIR)
    await run_cross_eval(base_url, RESULTS_DIR)
    generate_all_plots(RESULTS_DIR)


def main() -> None:
    """Entry point with argparse for backward compatibility."""
    parser = argparse.ArgumentParser(
        description="Prism-LoRA evaluation (backward-compatible wrapper)",
    )
    parser.add_argument(
        "--task",
        choices=["judge", "poet", "all"],
        default="all",
        help="Which task to evaluate (default: all)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Only generate plots from existing result JSONs (skip evaluation)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    args = parser.parse_args()

    if args.report:
        generate_all_plots(RESULTS_DIR)
        return

    base_url = f"http://localhost:{args.port}/v1"

    if args.task == "all":
        asyncio.run(run_full_evaluation(base_url))
    elif args.task == "judge":
        asyncio.run(run_judge_comparison(base_url, RESULTS_DIR))
    elif args.task == "poet":
        asyncio.run(run_poet_comparison(base_url, RESULTS_DIR))


if __name__ == "__main__":
    main()