#!/usr/bin/env python3
"""Resolve the newest complete smoke checkpoint for one architecture."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointMatch:
    manager: Path
    step: int
    modified_ns: int


def find_checkpoint(
    root: Path,
    architecture: str,
    *,
    d_model: int | None = None,
    n_layers: int | None = None,
) -> CheckpointMatch:
    """Find the newest, highest-step complete checkpoint matching metadata."""
    matches: list[CheckpointMatch] = []
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint scan root does not exist: {root}")

    for manager in root.iterdir():
        if not manager.is_dir():
            continue
        for step_dir in manager.iterdir():
            if not step_dir.is_dir() or not step_dir.name.isdigit():
                continue
            metadata_path = step_dir / "metadata" / "metadata"
            if not metadata_path.is_file() or not (step_dir / "state").is_dir():
                continue
            with metadata_path.open() as handle:
                config = json.load(handle).get("config", {})
            if config.get("architecture") != architecture:
                continue
            if config.get("token_mode") != "26tok":
                continue
            if d_model is not None and int(config.get("d_model", -1)) != d_model:
                continue
            if n_layers is not None and int(config.get("n_layers", -1)) != n_layers:
                continue
            matches.append(
                CheckpointMatch(
                    manager=manager.resolve(),
                    step=int(step_dir.name),
                    modified_ns=metadata_path.stat().st_mtime_ns,
                )
            )

    if not matches:
        raise ValueError(
            f"no complete 26-token {architecture} checkpoint under {root} "
            f"with d_model={d_model} n_layers={n_layers}"
        )
    return max(matches, key=lambda match: (match.modified_ns, match.step))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        required=True,
        choices=("mamba3", "s5", "gdn", "kda", "transformer", "nsa"),
    )
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-layers", type=int)
    args = parser.parse_args()
    match = find_checkpoint(
        args.root,
        args.architecture,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    print(f"{match.manager}\t{match.step}")


if __name__ == "__main__":
    main()
