#!/usr/bin/env python3
"""Maintain compact supervisor artifacts for one self-train LOBBench run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--action", required=True)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--alert", default=None)
    args = parser.parse_args()

    task_root = args.task_root.expanduser().resolve()
    task_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json(task_root / "manifest.json")
    previous = load_json(task_root / "state.json")
    now = utc_now()

    state = {
        "experiment": manifest.get("experiment", "selftrain_checkpoint_lobbench"),
        "state": args.state,
        "updated_at": now,
        "job_id": args.job_id,
        "action": args.action,
        "metric": args.metric,
        "alert": args.alert,
        "jobs": manifest.get("jobs", previous.get("jobs", {})),
        "checkpoint": manifest.get("checkpoint", previous.get("checkpoint", {})),
        "outputs": manifest.get("outputs", previous.get("outputs", {})),
    }
    (task_root / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    event = {
        "time": now,
        "job_id": args.job_id,
        "state": args.state,
        "metric": args.metric,
        "action": args.action,
        "alert": args.alert,
    }
    with (task_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")

    jobs = state["jobs"]
    checkpoint = state["checkpoint"]
    outputs = state["outputs"]
    lines = [
        "# Self-train checkpoint LOBBench status",
        "",
        f"- State: `{args.state}`",
        f"- Updated: `{now}`",
        f"- Action: `{args.action}`",
        f"- Active/last job: `{args.job_id or 'n/a'}`",
        (
            f"- Checkpoint: `{checkpoint.get('path', 'n/a')}` "
            f"step `{checkpoint.get('step', 'n/a')}`"
        ),
        (
            f"- Jobs: smoke `{jobs.get('smoke', 'n/a')}`, "
            f"generation `{jobs.get('generation', 'n/a')}`, "
            f"score `{jobs.get('score', 'n/a')}`, "
            f"finalizer `{jobs.get('finalizer', 'n/a')}`"
        ),
        f"- Metric: `{args.metric or 'pending'}`",
        f"- Alert: `{args.alert or 'none'}`",
        f"- Formal summary: `{outputs.get('lobbench_summary', 'pending')}`",
        "",
    ]
    (task_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
