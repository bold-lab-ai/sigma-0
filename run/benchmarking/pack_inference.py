#!/usr/bin/env python3
"""Pack an existing LOB-Bench inference tree into one SquashFS image.

For trees that were already written to Lustre before the generation launcher
learned to keep them node-local. It packs, verifies every file by sha256, and
then renames the source directory aside with a UTC stamp.

It does not delete anything, and that limit is deliberate: renaming does not
release an inode, so this command by itself does not lower the quota reading.
What it produces is the ability to release those inodes later without losing the
data, plus a tree that is still readable per file through a mount. The deletion
is a separate, human decision.

Usage::

    python run/benchmarking/pack_inference.py <evaluation-dir> [...]
    python run/benchmarking/pack_inference.py --dry-run <evaluation-dir>
    python run/benchmarking/pack_inference.py --keep-source <evaluation-dir>

``<evaluation-dir>`` is the directory that holds ``inference/`` (the one that
also holds ``lobbench_summary.json``), or the ``inference/`` directory itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lobpipeline.storage import sqfs  # noqa: E402

# Everything except the inference tree stays where it is. The scores in
# particular cannot be recomputed from an archived tree without a GPU, and the
# rank logs and manifests are what tie a score to the checkpoint that produced
# it.
KEEP_BESIDE_INFERENCE = (
    "lobbench",
    "lobbench_summary.json",
    "evaluation_complete.json",
    "generation_complete.json",
    "inference_inventory.json",
    "inference_index",
)


def locate_inference(target: Path) -> Path:
    """Accept either an evaluation directory or the inference tree itself."""
    if target.name == "inference":
        return target
    candidate = target / "inference"
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"no inference/ directory under {target}")


def pack_one(
    inference: Path,
    *,
    dry_run: bool,
    keep_source: bool,
    compression: str,
    processors: int | None,
) -> dict:
    image = sqfs.image_for(inference)
    record: dict = {
        "inference_dir": str(inference),
        "image": str(image),
        "sidecar": str(sqfs.sidecar_path(image)),
    }

    if image.is_file() and not dry_run:
        record["status"] = "already_packed"
        record["note"] = "an image already exists; re-verify it instead of rebuilding"
        return record

    rels = sqfs.list_rel_files(inference)
    record["n_files"] = len(rels)
    record["source_bytes"] = sum(os.path.getsize(inference / rel) for rel in rels)
    if dry_run:
        record["status"] = "dry_run"
        record["families"] = sqfs.family_counts(inference, rels)
        return record

    build = sqfs.build(
        inference,
        image,
        compression=compression,
        processors=processors,
        extra_metadata={
            "packed_by": "run/benchmarking/pack_inference.py",
            "evaluation_dir": str(inference.parent),
        },
    )
    record.update(
        {
            "image_bytes": build.image_bytes,
            "compression_ratio": round(build.compression_ratio, 3),
            "build_seconds": round(build.build_seconds, 2),
            "families": build.families,
            "tree_digest": build.tree_digest,
        }
    )

    verified = sqfs.verify(inference, image)
    record["verify_seconds"] = round(verified["verify_seconds"], 2)
    if verified["tree_digest"] != build.tree_digest:
        raise SystemExit(
            f"verify disagrees with build for {inference}: "
            f"{verified['tree_digest']} vs {build.tree_digest}"
        )
    record["verified"] = True

    if keep_source:
        record["status"] = "packed_source_kept"
        return record

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    aside = inference.with_name(f"inference_deprecated_{stamp}")
    inference.rename(aside)
    record["source_moved_to"] = str(aside)
    record["status"] = "packed"
    record["inodes_released"] = 0
    record["inodes_releasable_by_deleting_source"] = record["n_files"]
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="+", help="evaluation dir, or its inference/ dir")
    parser.add_argument("--dry-run", action="store_true", help="measure only, build nothing")
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="leave the source directory in place after a successful verify",
    )
    parser.add_argument("--compression", default=sqfs.DEFAULT_COMPRESSION)
    parser.add_argument("--processors", type=int, default=None)
    parser.add_argument("--report", default=None, help="write the JSON report here as well")
    args = parser.parse_args(argv)

    records = []
    failures = 0
    for raw in args.targets:
        inference = locate_inference(Path(raw).expanduser().resolve())
        try:
            record = pack_one(
                inference,
                dry_run=args.dry_run,
                keep_source=args.keep_source,
                compression=args.compression,
                processors=args.processors,
            )
        except (sqfs.SqfsError, OSError) as exc:
            failures += 1
            record = {"inference_dir": str(inference), "status": "failed", "error": str(exc)}
        records.append(record)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_targets": len(records),
        "n_failed": failures,
        "records": records,
    }
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
