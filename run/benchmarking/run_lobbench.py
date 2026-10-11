#!/usr/bin/env python3
"""Plan or execute authoritative LOBBench scoring for sigma-0 inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from lobpipeline.storage import sqfs
from lobpipeline.lob_bench.scoring import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DIVERGENCE_HORIZON,
    DEFAULT_SCORING_MODE,
    SCORING_MODES,
    build_lobbench_plan,
    execute_lobbench_plan,
    merge_lobbench_shards,
    score_run,
    summarize_plan_divergence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--inference-dir",
        default=None,
        help="directory containing data_gen/, data_real/, and data_cond/",
    )
    parser.add_argument("--output-dir", default="artifacts/lobbench")
    parser.add_argument("--stock", default="GOOG")
    parser.add_argument("--period", default="2026-01")
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--python", default=None)
    parser.add_argument("--n-workers", type=int, default=48)
    parser.add_argument("--run-id", default="sigma0")
    parser.add_argument(
        "--mode",
        default=DEFAULT_SCORING_MODE,
        choices=sorted(SCORING_MODES),
        help=(
            "which scoring mode of the driver to run; only the unconditional "
            "mode carries the WS-21 aggregate"
        ),
    )
    parser.add_argument(
        "--features",
        default=None,
        help="comma-separated subset for smoke tests; production defaults to WS-21",
    )
    parser.add_argument(
        "--divergence-horizon",
        type=int,
        default=DEFAULT_DIVERGENCE_HORIZON,
        help=(
            "messages per window of the divergence-over-time curve; each "
            "window of generated messages is scored against the full real "
            "distribution"
        ),
    )
    parser.add_argument(
        "--no-divergence",
        action="store_true",
        help=(
            "score the unconditional mode without the divergence curve it "
            "carries by default"
        ),
    )
    parser.add_argument(
        "--time-lag",
        type=int,
        default=None,
        help=(
            "override the time_lagged mode's lag (messages); required when "
            "cond+gen is at or below the config default of 500, where every "
            "sequence would be dropped"
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="score only this shard of the mode's scores; requires --shard-count",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="how many shards the scores are split across",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "merge the --shard-count shard files already written for this run "
            "into the unsharded file and summarize it; runs no scoring"
        ),
    )
    parser.add_argument("--summary", default=None)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    features = None
    if args.features is not None:
        features = tuple(part.strip() for part in args.features.split(",") if part.strip())

    if args.merge:
        if args.shard_count is None:
            parser.error("--merge requires --shard-count")
        summary = merge_lobbench_shards(
            run_name=args.run_name,
            output_dir=args.output_dir,
            shard_count=args.shard_count,
            stock=args.stock,
            period=args.period,
            benchmark_root=args.benchmark_root,
            run_id=args.run_id,
            features=features,
            mode=args.mode,
            summary_path=args.summary,
            divergence_horizon=args.divergence_horizon,
            with_divergence=not args.no_divergence,
        )
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return

    if args.inference_dir is None:
        result = score_run(args.run_name, manifest_path=args.manifest)
        print(f"plan=lobbench run_name={result.run_name} metrics={','.join(result.metrics)}")
        if result.rows is not None:
            print(f"manifest={result.manifest_path} rows={result.rows}")
        return

    # --inference-dir may name a directory or a SquashFS image built from one.
    # Resolving here rather than inside the scorer keeps the mount alive for the
    # whole run and leaves every downstream function taking a plain path, so the
    # vendored scorer under src/lobpipeline/scorer is untouched.
    with sqfs.resolve(args.inference_dir) as inference_dir:
        _run_plan(args, features, str(inference_dir))


def _run_plan(args, features, inference_dir: str) -> None:
    kwargs = {}
    if args.python is not None:
        kwargs["python"] = args.python
    plan = build_lobbench_plan(
        run_name=args.run_name,
        inference_dir=inference_dir,
        output_dir=args.output_dir,
        stock=args.stock,
        period=args.period,
        benchmark_root=args.benchmark_root,
        n_workers=args.n_workers,
        run_id=args.run_id,
        features=features,
        mode=args.mode,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        divergence_horizon=args.divergence_horizon,
        with_divergence=not args.no_divergence,
        time_lag=args.time_lag,
        **kwargs,
    )
    print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    if args.plan_only:
        return

    summary = execute_lobbench_plan(plan, summary_path=args.summary)
    # The divergence curve is the headline: distances per generation-step
    # window come first, and the pooled summary follows for the tables that
    # still want one number.
    if plan.mode != "divergence" and plan.divergence_score_path is not None:
        print(json.dumps(asdict(summarize_plan_divergence(plan)), indent=2, sort_keys=True))
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
