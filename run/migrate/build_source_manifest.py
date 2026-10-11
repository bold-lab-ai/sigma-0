#!/usr/bin/env python3
"""Build a lightweight source manifest for known legacy roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def describe_path(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    manifest = [describe_path(Path(raw)) for raw in args.paths]
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

