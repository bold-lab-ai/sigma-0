#!/usr/bin/env python3
"""Validate an base_model inference request."""

from __future__ import annotations

import argparse
import subprocess

from base_model.inference import run_inference
from base_model.models import available_architectures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--architecture", choices=available_architectures(), default=None)
    parser.add_argument("--execute", action="store_true", help="execute the in-repo inference runtime")
    args = parser.parse_args()
    result = run_inference(
        config_path=args.config,
        dry_run=not args.execute,
        architecture=args.architecture,
    )
    target = result.stock or result.universe
    print(
        f"plan=inference name={result.name} architecture={result.architecture} "
        f"target={target} period={result.period}"
    )
    print(f"output_root={result.output_root} config={result.config_path}")
    if args.execute:
        command = result.command()
        print("command=" + result.shell_command())
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
