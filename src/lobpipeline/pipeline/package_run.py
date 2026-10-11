"""Package a completed LOB bench run (predictions + scores) into one squashfs.

After inference and LOB bench scoring complete, the produced artifacts are bound
together as a single immutable, self-contained ``mksquashfs`` image so downstream
consumers mount one file instead of stat-ing thousands of small ``.npy`` files on
Lustre. The in-image layout is the recommended self-contained structure::

    <run_name>.squashfs
    |- manifest.json     run metadata (run_name, metric names, file inventory)
    |- scores.json       LOB bench scores for the run
    `- predictions/      per-ticker prediction artifacts (copied verbatim)

A sidecar ``index_<run_name>.json`` is written next to the squashfs, mirroring the
``index_<YYYY-MM>.json`` breadcrumb convention that ``node_wrapper.sh`` reads for
the training shards, so consumers can discover contents without mounting.

Only :func:`package_run` with ``execute=True`` shells out to ``mksquashfs``; the
plan-building and staging helpers are side-effect-light and importable anywhere.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..lob_bench.metrics import metric_names

DEFAULT_OUTPUT_DIR = "artifacts/packages"
DEFAULT_COMPRESSION = "zstd"
DEFAULT_PROCESSORS = 4


@dataclass(frozen=True)
class PackagePlan:
    """A validated request to package one run's predictions + scores."""

    run_name: str
    predictions_root: str
    scores_path: str
    output_dir: str
    squashfs_path: str
    index_path: str
    compression: str = DEFAULT_COMPRESSION


def build_package_plan(
    run_name: str,
    predictions_root: str | Path,
    scores_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    compression: str = DEFAULT_COMPRESSION,
) -> PackagePlan:
    """Validate packaging inputs and compute output paths (no side effects)."""
    if not run_name:
        raise ValueError("run_name must be non-empty")
    predictions = Path(predictions_root)
    if not predictions.is_dir():
        raise NotADirectoryError(f"predictions_root is not a directory: {predictions}")
    scores = Path(scores_path)
    if not scores.is_file():
        raise FileNotFoundError(f"scores file not found: {scores}")
    out = Path(output_dir)
    return PackagePlan(
        run_name=run_name,
        predictions_root=str(predictions),
        scores_path=str(scores),
        output_dir=str(out),
        squashfs_path=str(out / f"{run_name}.squashfs"),
        index_path=str(out / f"index_{run_name}.json"),
        compression=compression,
    )


def _inventory(staging: Path) -> list[str]:
    """Sorted POSIX-relative file list inside the staging tree (excl. manifest)."""
    return sorted(
        p.relative_to(staging).as_posix()
        for p in staging.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )


def stage_package(plan: PackagePlan, staging_dir: str | Path) -> Path:
    """Assemble the self-contained ``{manifest.json, scores.json, predictions/}`` tree.

    No squashfs is produced here, so this is safe to exercise in unit tests
    without ``mksquashfs``. Returns the populated staging directory.
    """
    staging = Path(staging_dir)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # predictions/ -- copied verbatim; packaging is content-agnostic.
    shutil.copytree(plan.predictions_root, staging / "predictions")

    # scores.json -- re-serialised from the run's scores file (validates JSON).
    scores_payload = json.loads(Path(plan.scores_path).read_text(encoding="utf-8"))
    (staging / "scores.json").write_text(
        json.dumps(scores_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # manifest.json -- run metadata + inventory of every other member.
    manifest = {
        "run_name": plan.run_name,
        "metrics": list(metric_names()),
        "contents": _inventory(staging),
        "source": {
            "predictions_root": plan.predictions_root,
            "scores_path": plan.scores_path,
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return staging


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file through sha256 (avoids loading large squashfs into memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def package_run(
    run_name: str,
    predictions_root: str | Path,
    scores_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    staging_dir: str | Path | None = None,
    compression: str = DEFAULT_COMPRESSION,
    processors: int = DEFAULT_PROCESSORS,
    execute: bool = False,
) -> PackagePlan:
    """Validate and, when ``execute`` is set, build the self-contained squashfs.

    With ``execute=False`` (the default) this only returns the validated plan and
    performs no ``mksquashfs`` call, mirroring the dry-run pattern of the inference
    and score entry points.
    """
    plan = build_package_plan(run_name, predictions_root, scores_path, output_dir, compression)
    if not execute:
        return plan

    out = Path(plan.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    staging = Path(staging_dir) if staging_dir is not None else out / f".stage_{run_name}"
    stage_package(plan, staging)

    squashfs = Path(plan.squashfs_path)
    if squashfs.exists():
        squashfs.unlink()
    subprocess.run(
        [
            "mksquashfs",
            str(staging),
            str(squashfs),
            "-comp",
            compression,
            "-b",
            "1M",
            "-noappend",
            "-no-progress",
            "-processors",
            str(processors),
        ],
        check=True,
    )

    index = {
        "run_name": run_name,
        "squashfs": squashfs.name,
        "bytes": squashfs.stat().st_size,
        "sha256": _sha256_file(squashfs),
        "compression": compression,
        "contents": _inventory(staging),
    }
    Path(plan.index_path).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shutil.rmtree(staging, ignore_errors=True)
    return plan
