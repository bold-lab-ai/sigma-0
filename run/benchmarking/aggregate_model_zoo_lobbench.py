#!/usr/bin/env python3
"""Aggregate the 15 fixed-30k paper summaries into the final seed table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ARCHITECTURES = ("mamba3", "s5", "gdn", "kda", "transformer")
SEEDS = (5, 42, 137)
METRICS = ("ws21", "ks21", "l1_21")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.artifact_root).expanduser().resolve()
    rows = []
    revisions = set()
    commits = set()
    sample_index_hashes = set()
    for architecture in ARCHITECTURES:
        per_seed = []
        for seed in SEEDS:
            evaluation = root / "evaluation_30k" / architecture / f"seed_{seed}"
            path = evaluation / "lobbench_summary.json"
            generation_path = evaluation / "generation_complete.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing production LOBBench summary: {path}")
            if not generation_path.is_file():
                raise FileNotFoundError(f"missing production generation manifest: {generation_path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            generation = json.loads(generation_path.read_text(encoding="utf-8"))
            if payload.get("feature_count") != 21 or payload.get("ws21") is None:
                raise ValueError(f"{path}: not a complete WS-21 result")
            if generation.get("checkpoint_step") != 30_000:
                raise ValueError(f"{generation_path}: expected checkpoint step 30000")
            if generation.get("architecture") != architecture or generation.get("training_seed") != seed:
                raise ValueError(f"{generation_path}: architecture/seed identity mismatch")
            if generation.get("sequences") != 3_136:
                raise ValueError(f"{generation_path}: expected 3136 generated sequences")
            if payload.get("benchmark_revision") != generation.get("lobbench_revision"):
                raise ValueError(f"{path}: benchmark revision differs from generation manifest")
            revisions.add(payload.get("benchmark_revision"))
            commits.add(generation.get("sigma0_commit"))
            sample_index_hashes.add(generation.get("sample_indices_sha256"))
            per_seed.append(payload)

        row: dict[str, object] = {"architecture": architecture, "n_seeds": len(SEEDS)}
        for metric in METRICS:
            values = [float(payload[metric]) for payload in per_seed]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values)
        rows.append(row)

    if len(revisions) != 1:
        raise ValueError(f"benchmark revision drift across runs: {sorted(revisions)}")
    if len(commits) != 1:
        raise ValueError(f"sigma-0 code drift across runs: {sorted(commits)}")
    if len(sample_index_hashes) != 1:
        raise ValueError(f"evaluation-window drift across runs: {sorted(sample_index_hashes)}")

    output = Path(args.output).expanduser().resolve() if args.output else root / "lobbench_30k_three_seed.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["architecture", "n_seeds"] + [
        f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
