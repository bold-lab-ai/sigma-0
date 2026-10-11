#!/usr/bin/env python3
"""Build a concrete base_model training launch plan."""

from __future__ import annotations

import argparse

from base_model.training import run_train
from base_model.models import available_architectures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--architecture",
        choices=available_architectures(),
        default=None,
        help="override the config architecture (default: mamba3)",
    )
    parser.add_argument("--execute", action="store_true", help="print the non-test sbatch command")
    args = parser.parse_args()
    result = run_train(
        config_path=args.config,
        dry_run=not args.execute,
        architecture=args.architecture,
    )
    print(f"plan=train name={result.name} config={result.config_path}")
    print(f"workdir={result.workdir}")
    print("command=" + result.shell_command(test_only=not args.execute))


if __name__ == "__main__":
    main()
