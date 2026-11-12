"""The standard LOB-Bench evaluation: one protocol, one command, one record.

Why this module exists
----------------------
``run_lobbench.py`` is the scorer every number goes through, but it leaves the
protocol to whoever calls it: which windows, which scorer root, which horizon,
which checks first. Two runs that both print a WS-21 can therefore differ in any
of those, and nothing in the summary says so. This module fixes the protocol
and writes one record that states everything a reader needs to decide whether
two numbers are comparable.

What is fixed
-------------
- The windows are the frozen sample index
  (``run/benchmarking/model_zoo_GOOG_jan2026_3136_seq500_seed42.txt``). Its
  metadata names the stock, period, window count and message counts, so the
  protocol is read from it rather than restated here. The file's SHA-256 must
  equal the hash its metadata recorded when it was frozen.
- The scorer is the merged in-repo tree, never the Lustre default.
  ``merged_benchmark_root`` recomputes its hash from the files on disk and
  refuses a tree that differs from ``merge.json``.
- The mode is ``uncond`` with the divergence curve at the default horizon,
  unless the run asks to skip the curve (``divergence_horizon: null`` in the
  record). The headline does not depend on it. The
  curve is reported exactly as ``summarize_divergence_file`` lays it out,
  including its ``null`` windows. At the default horizon of 100 the four OFI
  scores have no windows (their 100-message rolling mean needs more than 100
  messages), so the panel covers 17 scores and says so in
  ``scores_without_windows`` and ``is_full_panel``.

What the record holds
---------------------
WS-21, KS-21 and L1-21; every feature's point estimate with its bootstrap
interval; the divergence curves; and the provenance: sigma-0 commit, scorer
hash, sample-index hash, a hash of each input directory, the scoring
interpreter's packages, worker count, Slurm job and the time each part took.
A run that cannot produce the three headline numbers writes no record.

Optional parts
--------------
Each is off unless asked for, and none of them changes the headline. A part
that was not run is ``null`` in the record.

- ``kl``: the Kullback-Leibler divergence KL(real || generated) of each of the
  21 features. It is computed here from the unconditional score pickle, which
  already holds every real and generated value with the bin the scorer's L1
  used, so it needs no second scoring pass and no change to the pinned scorer.
  Each bin gets one pseudo-count on each side, so a bin the generated data
  never reaches gives a large but finite value. KL is unbounded, so its mean
  over the features is dominated by the worst one; read the features.
- ``conditional``: the driver's ``cond`` mode, a second driver run. Three scores,
  each one feature compared within bins of another.
- ``context``: the driver's ``context`` mode, a second driver run. Its L1 is a
  structural zero (see docs/benchmarking.md), so the record keeps W1 and KS
  and writes L1 as ``null``.
- ``time_lagged``: the driver's ``time_lagged`` mode, a second driver run, at
  a lag of ``STANDARD_TIME_LAG`` messages. The driver's own lag of 500 needs
  sequences longer than 500 messages; ours are 250 + 250, so at 500 every
  window is dropped and the driver fails. At 250 the lagged value always comes
  from the conditioning messages and the evaluated value from the generated
  ones. The driver's score names still say ``t-lag=500``; the record states
  the lag that ran.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lobpipeline.lob_bench.scoring import (
    DEFAULT_DIVERGENCE_HORIZON,
    LOBBENCH_FEATURES,
    LobbenchDivergenceSummary,
    LobbenchSummary,
    metric_names,
)
from lobpipeline.scorer import (
    merged_benchmark_root,
    read_merge_record,
    source_files,
    tree_sha256,
)

STANDARD_EVAL_FORMAT = "sigma0.lobbench.standard_eval.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_INDEX_FILE = (
    REPO_ROOT / "run" / "benchmarking" / "model_zoo_GOOG_jan2026_3136_seq500_seed42.txt"
)
STANDARD_MODE = "uncond"
STANDARD_DIVERGENCE_HORIZON = DEFAULT_DIVERGENCE_HORIZON
# Messages between the lagged and the evaluated value in the time-lagged mode.
# Equal to the generated length, so the lag spans conditioning -> generation.
STANDARD_TIME_LAG = 250
# Driver modes the record can carry as optional sections, in run order.
OPTIONAL_MODES = ("cond", "context", "time_lagged")
OPTIONAL_SECTION = {"cond": "conditional", "context": "context", "time_lagged": "time_lagged"}
KL_METHOD = {
    "direction": "real || generated",
    "bins": "the scorer's L1 bins (the score frame's group column)",
    "smoothing": "one pseudo-count per bin on each side",
    "units": "nats",
}


@dataclass(frozen=True)
class StandardProtocol:
    """The window set every standard run is scored on, read from its index file."""

    index_file: str
    index_sha256: str
    canonical: bool
    stock: str
    period: str
    sequences: int
    conditioning_messages: int
    generated_messages: int
    dataset_length: int | None
    mode: str
    # ``None`` when the run skips the divergence curve. The headline numbers do
    # not depend on it, so records with and without the curve still compare on
    # the headline; only the curve itself is absent.
    divergence_horizon: int | None


def load_protocol(
    index_file: str | Path = CANONICAL_INDEX_FILE, *, divergence: bool = True
) -> StandardProtocol:
    """Read a frozen sample index and its metadata, refusing a modified file.

    A non-canonical index is allowed so the command can be smoke-tested on a
    small window set. The record says ``canonical: false`` for such a run, and
    its numbers are not standard results.
    """
    path = Path(index_file).expanduser().resolve()
    metadata_path = path.with_name(path.name + ".metadata.json")
    if not path.is_file():
        raise FileNotFoundError(f"sample index not found: {path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"sample index metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = metadata.get("sample_indices_sha256")
    if actual != recorded:
        raise ValueError(
            f"{path}: SHA-256 {actual} differs from the frozen value {recorded}"
        )
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sequences = int(metadata["n_samples"])
    if len(lines) != sequences:
        raise ValueError(f"{path}: {len(lines)} indices, metadata says {sequences}")
    if int(metadata["conditioning_messages"]) + int(metadata["generated_messages"]) != int(
        metadata["sequence_messages"]
    ):
        raise ValueError(f"{metadata_path}: message counts do not add up")
    return StandardProtocol(
        index_file=str(path),
        index_sha256=actual,
        canonical=path == CANONICAL_INDEX_FILE.resolve(),
        stock=str(metadata["ticker"]),
        period=str(metadata["period"]),
        sequences=sequences,
        conditioning_messages=int(metadata["conditioning_messages"]),
        generated_messages=int(metadata["generated_messages"]),
        dataset_length=metadata.get("dataset_length"),
        mode=STANDARD_MODE,
        divergence_horizon=STANDARD_DIVERGENCE_HORIZON if divergence else None,
    )


def standard_scorer(name: str = "lob_bench") -> tuple[Path, str]:
    """The merged scorer root and its tree hash, recomputed from disk.

    ``merged_benchmark_root`` re-hashes the tree and refuses a modified one, so
    the hash returned here is a property of the files that will run.
    """
    root = merged_benchmark_root(name)
    return root, read_merge_record(name)["tree_sha256"]


def directory_digest(directory: str | Path) -> dict[str, Any]:
    """Hash one input directory: file names, layout and bytes."""
    root = Path(directory).expanduser().resolve()
    files = source_files(root)
    return {
        "sha256": tree_sha256(root),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def _interval(value: Any) -> list[float] | None:
    """The bootstrap interval of one ``(point, ci, losses)`` score, or ``None``."""
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    try:
        pair = [float(bound) for bound in value[1]]
    except (TypeError, ValueError):
        return None
    if len(pair) != 2 or not all(math.isfinite(bound) for bound in pair):
        return None
    return pair


def pooled_intervals(
    score_path: str | Path, features: Sequence[str] = LOBBENCH_FEATURES
) -> dict[str, dict[str, list[float] | None]]:
    """Per feature and distance, the pooled bootstrap interval the driver wrote.

    ``summarize_score_file`` keeps only point estimates. The intervals are in
    the same trusted pickle, written by the pinned scorer in this run.
    """
    with gzip.open(Path(score_path), "rb") as handle:
        scores = pickle.load(handle)[0]
    out: dict[str, dict[str, list[float] | None]] = {}
    for feature in features:
        per_distance = scores[feature]
        out[feature] = {
            distance: _interval(per_distance.get(distance)) for distance in metric_names()
        }
    return out


def binned_kl(real_groups: Sequence[Any], generated_groups: Sequence[Any]) -> float | None:
    """KL(real || generated) over shared bins, with one pseudo-count per bin.

    ``None`` when either side has no observation, which is no measurement.
    """
    real_counts: dict[Any, int] = {}
    generated_counts: dict[Any, int] = {}
    for group in real_groups:
        real_counts[group] = real_counts.get(group, 0) + 1
    for group in generated_groups:
        generated_counts[group] = generated_counts.get(group, 0) + 1
    if not real_counts or not generated_counts:
        return None
    bins = set(real_counts) | set(generated_counts)
    real_total = sum(real_counts.values()) + len(bins)
    generated_total = sum(generated_counts.values()) + len(bins)
    divergence = 0.0
    for group in bins:
        p = (real_counts.get(group, 0) + 1) / real_total
        q = (generated_counts.get(group, 0) + 1) / generated_total
        divergence += p * math.log(p / q)
    return max(divergence, 0.0)


def _finite_groups(frame: Any, side: str) -> list[Any]:
    rows = frame.loc[frame["type"] == side]
    return [
        group
        for group, score in zip(rows["group"], rows["score"])
        if _finite(score)
    ]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pooled_kl(
    score_path: str | Path, features: Sequence[str] = LOBBENCH_FEATURES
) -> dict[str, Any]:
    """KL(real || generated) per feature, from the unconditional score pickle.

    The pickle's second element maps each feature to its score frame: one row
    per value, with ``score``, ``group`` (the L1 bin) and ``type``. The values
    and bins are the scorer's own, so nothing is rescored.
    """
    with gzip.open(Path(score_path), "rb") as handle:
        frames = pickle.load(handle)[1]
    per_feature: dict[str, float | None] = {}
    for feature in features:
        frame = frames.get(feature) if isinstance(frames, Mapping) else None
        if frame is None:
            raise ValueError(f"{score_path}: no score frame for {feature}")
        per_feature[feature] = binned_kl(
            _finite_groups(frame, "real"), _finite_groups(frame, "generated")
        )
    measured = [value for value in per_feature.values() if value is not None]
    return {
        **KL_METHOD,
        "mean": sum(measured) / len(measured) if len(measured) == len(features) else None,
        "features": per_feature,
    }


def optional_mode_section(
    summary: LobbenchSummary, *, scorer_tree_sha256: str, time_lag: int | None = None
) -> dict[str, Any]:
    """One optional driver mode as a record section.

    The context mode's L1 is a structural zero, so it is written as ``null``
    rather than as a number someone could read.
    """
    if summary.benchmark_revision != scorer_tree_sha256:
        raise ValueError(
            f"{summary.mode} summary names scorer {summary.benchmark_revision}, "
            f"verified tree is {scorer_tree_sha256}"
        )
    drop = {"l1"} if summary.mode == "context" else set()
    features = {
        name: {
            distance: (None if distance in drop else value)
            for distance, value in scores.items()
        }
        for name, scores in summary.feature_scores.items()
    }
    means = {
        distance: (None if distance in drop else value)
        for distance, value in summary.distances.items()
    }
    section: dict[str, Any] = {
        "mode": summary.mode,
        "score_path": summary.score_path,
        "means": means,
        "features": features,
    }
    if summary.mode == "time_lagged":
        section["time_lag"] = time_lag
    if drop:
        section["note"] = "L1 is a structural zero in this mode and is not reported."
    return section


def interpreter_packages(python: str) -> dict[str, str]:
    """Installed distributions of the scoring interpreter, name to version."""
    probe = (
        "import importlib.metadata as m, json, sys; "
        "d = {x.metadata['Name']: x.version for x in m.distributions()}; "
        "d['python'] = sys.version.split()[0]; print(json.dumps(d, sort_keys=True))"
    )
    result = subprocess.run(
        [python, "-c", probe], check=True, text=True, capture_output=True, timeout=120
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def repository_state(repo: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The sigma-0 commit the evaluation ran from, and whether it was modified."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    commit = git("rev-parse", "HEAD") or None
    # Untracked files count under the code roots: a new module the evaluation
    # imported is as much a change to what ran as an edited one.
    changes = git("status", "--porcelain", "--", "src", "run", "envs") if commit else ""
    return {
        "commit": commit,
        "modified": bool(changes) if commit else None,
        "changes": changes.splitlines()[:50],
    }


def file_digest(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    return {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def build_record(
    *,
    run_name: str,
    protocol: StandardProtocol,
    scorer_tree_sha256: str,
    pooled: LobbenchSummary,
    divergence: LobbenchDivergenceSummary | None,
    intervals: Mapping[str, Mapping[str, list[float] | None]],
    inputs: Mapping[str, Any],
    provenance: Mapping[str, Any],
    kl: Mapping[str, Any] | None = None,
    optional: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the standard record, refusing a run without its headline numbers.

    ``optional`` maps a driver mode (``cond``, ``context``, ``time_lagged``) to
    its section from ``optional_mode_section``. A mode left out is ``null``.
    """
    headline = {"ws21": pooled.ws21, "ks21": pooled.ks21, "l1_21": pooled.l1_21}
    missing = [name for name, value in headline.items() if value is None]
    if missing:
        raise ValueError(f"no standard record: {', '.join(missing)} not produced")
    if pooled.mode != protocol.mode:
        raise ValueError(f"scored mode {pooled.mode!r}, standard is {protocol.mode!r}")
    if tuple(pooled.feature_scores) != LOBBENCH_FEATURES:
        raise ValueError("pooled features differ from the pinned WS-21 list")
    if pooled.benchmark_revision != scorer_tree_sha256:
        raise ValueError(
            f"summary names scorer {pooled.benchmark_revision}, "
            f"verified tree is {scorer_tree_sha256}"
        )
    if (divergence is None) != (protocol.divergence_horizon is None):
        raise ValueError(
            "the divergence curve was "
            + ("skipped" if divergence is None else "scored")
            + " but the protocol says the opposite"
        )
    if divergence is not None and divergence.horizon_length != protocol.divergence_horizon:
        raise ValueError(
            f"divergence horizon {divergence.horizon_length}, "
            f"standard is {protocol.divergence_horizon}"
        )
    optional = dict(optional or {})
    unknown = sorted(set(optional) - set(OPTIONAL_MODES))
    if unknown:
        raise ValueError(f"unknown optional modes: {unknown}")
    for mode, section in optional.items():
        if section.get("mode") != mode:
            raise ValueError(f"optional section for {mode} holds mode {section.get('mode')!r}")
    if kl is not None and set(kl.get("features", {})) != set(LOBBENCH_FEATURES):
        raise ValueError("the KL section does not cover the 21 pinned features")
    features = {
        feature: {
            distance: {
                "point": pooled.feature_scores[feature][distance],
                "interval": intervals.get(feature, {}).get(distance),
            }
            for distance in metric_names()
        }
        for feature in LOBBENCH_FEATURES
    }
    divergence_fields = None
    if divergence is not None:
        divergence_fields = asdict(divergence)
        for name in ("run_name", "stock", "period", "mode", "score_path",
                     "benchmark_root", "benchmark_revision"):
            divergence_fields.pop(name, None)
    return {
        "format": STANDARD_EVAL_FORMAT,
        "run_name": run_name,
        "standard": protocol.canonical,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": asdict(protocol),
        "headline": headline,
        "features": features,
        "divergence": divergence_fields,
        "kl": dict(kl) if kl is not None else None,
        **{OPTIONAL_SECTION[mode]: (dict(optional[mode]) if mode in optional else None)
           for mode in OPTIONAL_MODES},
        "scorer": {
            "tree_sha256": scorer_tree_sha256,
            "benchmark_root": pooled.benchmark_root,
            "score_path": pooled.score_path,
            "divergence_score_path": divergence.score_path if divergence is not None else None,
        },
        "inputs": dict(inputs),
        "provenance": dict(provenance),
    }


def run_provenance(
    *,
    python: str,
    n_workers: int,
    generation_manifest: str | Path | None,
    seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Where and how the evaluation ran, and how long each part took."""
    provenance: dict[str, Any] = {
        "seconds": dict(seconds or {}),
        "sigma0": repository_state(),
        "scoring_python": python,
        "scoring_packages": interpreter_packages(python),
        "n_workers": n_workers,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
    }
    if generation_manifest is not None:
        manifest = Path(generation_manifest).expanduser().resolve()
        provenance["generation_manifest"] = {
            **file_digest(manifest),
            "content": json.loads(manifest.read_text(encoding="utf-8")),
        }
    else:
        provenance["generation_manifest"] = None
    return provenance


def write_record(record: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
