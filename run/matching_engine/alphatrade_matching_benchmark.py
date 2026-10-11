#!/usr/bin/env python3
"""Benchmark the current AlphaTrade in-memory matching step."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from alphatrade.matching import MatchingEngine


def git_commit() -> str:
    """Return the current short git commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--warmup-orders", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    order = {"side": "buy", "qty": 1, "price": 100.0}

    warmup = MatchingEngine()
    for _ in range(args.warmup_orders):
        warmup.step(order)
    print(f"warmup_orders={args.warmup_orders} warmup_events={len(warmup.events)}")

    per_order_us: list[float] = []
    elapsed_seconds: list[float] = []
    for repeat in range(args.repeats):
        engine = MatchingEngine()
        gc.collect()
        start_ns = time.perf_counter_ns()
        for _ in range(args.orders):
            engine.step(order)
        elapsed_ns = time.perf_counter_ns() - start_ns
        event_count = len(engine.events)
        if event_count != args.orders:
            raise RuntimeError(f"expected {args.orders} events, got {event_count}")
        elapsed_s = elapsed_ns / 1_000_000_000
        per_us = elapsed_ns / args.orders / 1000
        elapsed_seconds.append(elapsed_s)
        per_order_us.append(per_us)
        print(
            f"repeat={repeat} orders={args.orders} "
            f"elapsed_s={elapsed_s:.6f} per_order_us={per_us:.6f}"
        )

    summary = {
        "benchmark": "alphatrade.matching.MatchingEngine.step",
        "metric_scope": (
            "prebuilt order payload; EngineEvent construction plus append to "
            "the current in-memory accepted-event list"
        ),
        "orders_per_repeat": args.orders,
        "warmup_orders": args.warmup_orders,
        "repeats": args.repeats,
        "per_order_us": per_order_us,
        "elapsed_seconds": elapsed_seconds,
        "mean_us": statistics.fmean(per_order_us),
        "median_us": statistics.median(per_order_us),
        "min_us": min(per_order_us),
        "max_us": max(per_order_us),
        "stdev_us": statistics.stdev(per_order_us) if len(per_order_us) > 1 else 0.0,
        "commit": git_commit(),
        "hostname": platform.node(),
        "python": platform.python_version(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"RESULT_JSON={args.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
