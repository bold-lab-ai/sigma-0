#!/usr/bin/env python3
"""Validate or build a self-contained squashfs for a finished LOB bench run."""

from __future__ import annotations

import argparse

from base_model.config import read_key_value_config
from lobpipeline.pipeline.package_run import DEFAULT_OUTPUT_DIR, package_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="flat key:value package config")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--predictions-root", default=None)
    parser.add_argument("--scores-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="build the squashfs (default: validate the plan only)",
    )
    args = parser.parse_args()

    config = read_key_value_config(args.config) if args.config else {}
    run_name = args.run_name or config.get("run_name")
    predictions_root = args.predictions_root or config.get("predictions_root")
    scores_path = args.scores_path or config.get("scores_path")
    output_dir = args.output_dir or config.get("output_dir", DEFAULT_OUTPUT_DIR)
    if not (run_name and predictions_root and scores_path):
        parser.error("run_name, predictions_root and scores_path are required (via --config or flags)")

    plan = package_run(
        run_name=run_name,
        predictions_root=predictions_root,
        scores_path=scores_path,
        output_dir=output_dir,
        execute=args.execute,
    )
    state = "built" if args.execute else "planned"
    print(f"plan=package state={state} run_name={plan.run_name} predictions_root={plan.predictions_root}")
    print(f"scores={plan.scores_path} squashfs={plan.squashfs_path} index={plan.index_path}")


if __name__ == "__main__":
    main()
