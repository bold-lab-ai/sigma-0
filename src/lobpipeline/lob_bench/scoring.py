"""Run and summarize the authoritative LOBBench scorer.

The first sigma-0 migration only retained a manifest-validation placeholder.
This module keeps that lightweight planning API, but also provides the real
inference-output -> LOBBench adapter used by the model-zoo experiments.  The
benchmark implementation remains the separately versioned FLAIROx
``lob_pipeline`` repository; sigma-0 records its Git revision in every summary
instead of silently carrying a second, drifting copy of the scorer.
"""

from __future__ import annotations

import ast
import gzip
import json
import math
import os
import pickle
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..scorer import recorded_revision
from .metrics import metric_names


DEFAULT_BENCHMARK_ROOT = Path(
    "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/lob_pipeline"
)

# The current scorer replaces the four reconstruction-sensitive ``*_levels``
# features with BBO-relative ``*_ticks`` features.  This is the paper's WS-21
# contract; the order is fixed so summaries and tables are deterministic.
LOBBENCH_FEATURES = (
    "spread",
    "orderbook_imbalance",
    "log_inter_arrival_time",
    "log_time_to_cancel",
    "ask_volume_touch",
    "bid_volume_touch",
    "ask_volume",
    "bid_volume",
    "limit_ask_order_depth",
    "limit_bid_order_depth",
    "ask_cancellation_depth",
    "bid_cancellation_depth",
    "limit_ask_order_ticks",
    "limit_bid_order_ticks",
    "ask_cancellation_ticks",
    "bid_cancellation_ticks",
    "vol_per_min",
    "ofi",
    "ofi_up",
    "ofi_stay",
    "ofi_down",
)


@dataclass(frozen=True)
class ScoringMode:
    """One scoring mode of the driver, and how to read the file it writes.

    ``config_name`` names the module-level dict in ``run_bench.py`` whose keys
    are the score names this mode produces.  ``pinned_features`` is set only
    where sigma-0 owns a published contract: the unconditional mode carries the
    paper's WS-21 feature list, and a driver that stops declaring exactly those
    21 names is a change to a published number, not a routine upgrade.  The
    other modes have no such contract yet, so their names are read from the
    driver rather than mirrored here.
    """

    name: str
    flag: str
    file_prefix: str
    config_name: str
    pinned_features: tuple[str, ...] | None


SCORING_MODES: dict[str, ScoringMode] = {
    "uncond": ScoringMode(
        name="uncond",
        flag="--uncond_only",
        file_prefix="scores_uncond",
        config_name="DEFAULT_SCORING_CONFIG",
        pinned_features=LOBBENCH_FEATURES,
    ),
    "cond": ScoringMode(
        name="cond",
        flag="--conditional",
        file_prefix="scores_cond",
        config_name="DEFAULT_SCORING_CONFIG_COND",
        pinned_features=None,
    ),
    "context": ScoringMode(
        name="context",
        flag="--context",
        file_prefix="scores_context",
        config_name="DEFAULT_SCORING_CONFIG_CONTEXT",
        pinned_features=None,
    ),
    "time_lagged": ScoringMode(
        name="time_lagged",
        flag="--time_lagged",
        file_prefix="scores_time_lagged",
        config_name="DEFAULT_SCORING_CONFIG_TIME_LAGGED",
        pinned_features=None,
    ),
    # Divergence-over-time: the generated stream is cut into consecutive
    # windows of ``divergence_horizon`` messages and each window is scored
    # against the full real distribution, so the result is one distance per
    # horizon step instead of one pooled number.  The driver reuses the
    # unconditional config for it (``run_bench.py`` filters ``--metrics``
    # against that namespace when either flag is set), so the same 21 names
    # are pinned here for the same reason they are pinned for ``uncond``.
    "divergence": ScoringMode(
        name="divergence",
        flag="--divergence",
        file_prefix="scores_div",
        config_name="DEFAULT_SCORING_CONFIG",
        pinned_features=LOBBENCH_FEATURES,
    ),
}

DEFAULT_SCORING_MODE = "uncond"

# Every window of this many generated messages becomes one point on the
# divergence curve.  100 matches the driver's own default.
DEFAULT_DIVERGENCE_HORIZON = 100


def resolve_mode(mode: str) -> ScoringMode:
    """Return the mode descriptor, listing the alternatives when it is unknown."""
    try:
        return SCORING_MODES[mode]
    except KeyError:
        raise ValueError(
            f"unknown scoring mode {mode!r}; known modes: "
            f"{', '.join(sorted(SCORING_MODES))}"
        ) from None


def driver_path(benchmark_root: str | Path) -> Path:
    """Return the authoritative driver, or fail before anything is staged."""
    driver = Path(benchmark_root).expanduser().resolve() / "lob_bench" / "run_bench.py"
    if not driver.is_file():
        raise FileNotFoundError(f"authoritative LOBBench driver not found: {driver}")
    return driver


def declared_features(benchmark_root: str | Path, mode: str) -> tuple[str, ...]:
    """Read, from the driver's source, the score names one mode will produce.

    The driver is parsed, never imported: importing it pulls in the whole
    scoring dependency set, which by design is not installed alongside sigma-0
    (see ``envs/lobbench_venv_setup.sh``).  Parsing also means a config entry
    that is commented out is correctly reported as absent, which is how the
    contextual config currently carries one name rather than two.
    """
    descriptor = resolve_mode(mode)
    driver = driver_path(benchmark_root)
    tree = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if descriptor.config_name not in targets:
            continue
        names: list[str] = []
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError(
                    f"{driver}: {descriptor.config_name} has a non-literal key; "
                    "sigma-0 cannot state what this mode will produce"
                )
            names.append(key.value)
        if not names:
            raise ValueError(f"{driver}: {descriptor.config_name} is empty")
        return tuple(names)
    raise ValueError(
        f"{driver}: no module-level {descriptor.config_name}; "
        f"this driver does not support mode {mode!r}"
    )


def mode_features(benchmark_root: str | Path, mode: str) -> tuple[str, ...]:
    """Return one mode's full score-name set, checked against any pinned list.

    For the unconditional mode this is where drift surfaces loudly: WS-21 is a
    mean over exactly the 21 pinned names, so a driver that declares a different
    set would otherwise keep publishing a number under a name it no longer has.
    """
    descriptor = resolve_mode(mode)
    features = declared_features(benchmark_root, mode)
    pinned = descriptor.pinned_features
    if pinned is not None and features != pinned:
        extra = sorted(set(features) - set(pinned))
        absent = sorted(set(pinned) - set(features))
        detail = (
            f"LOBBench {mode} contract drift at {benchmark_root}: "
            f"driver declares {extra} and not {absent}"
        )
        if any(name.endswith("_levels") for name in extra):
            detail += (
                "; this driver predates the *_ticks backport, and the four "
                "*_levels scores are the ones that drifted about sixteen times "
                "as much as the rest when compute-node compilation changed, so "
                "a mean over them is not the WS-21 in the paper"
            )
        raise ValueError(detail)
    return features


def shard_features(
    features: Sequence[str], shard_index: int, shard_count: int
) -> tuple[str, ...]:
    """Split score names across shards, deterministically and without overlap.

    Shards are strided rather than contiguous so that a run whose cost is
    dominated by a few names spreads that cost instead of piling it into one
    shard.  An empty shard is refused: it would produce a valid-looking pickle
    with no scores in it.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be positive; got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}); got {shard_index}"
        )
    if shard_count > len(features):
        raise ValueError(
            f"{shard_count} shards over {len(features)} score names would leave "
            "a shard empty"
        )
    return tuple(features[shard_index::shard_count])


@dataclass(frozen=True)
class ScorePlan:
    """Validated score request retained for migration-API compatibility."""

    run_name: str
    metrics: tuple[str, ...]
    manifest_path: str | None = None
    rows: int | None = None


@dataclass(frozen=True)
class InferenceInventory:
    """Counts proving that an inference run has the LOBBench file contract."""

    inference_dir: str
    generated_messages: int
    generated_books: int
    real_messages: int
    real_books: int
    conditional_messages: int
    conditional_books: int


@dataclass(frozen=True)
class LobbenchPlan:
    """Fully resolved command and paths for one scoring run of one mode."""

    run_name: str
    stock: str
    period: str
    mode: str
    inference_dir: str
    staging_root: str
    results_dir: str
    score_path: str
    benchmark_root: str
    benchmark_revision: str
    command: tuple[str, ...]
    expected_features: tuple[str, ...]
    shard_index: int | None
    shard_count: int | None
    inventory: InferenceInventory
    # Set whenever the run produces a divergence curve: either this plan's
    # mode is ``divergence``, or it is ``uncond`` and the driver was asked to
    # emit the curve alongside the pooled scores (the default).
    divergence_horizon: int | None = None
    divergence_score_path: str | None = None
    time_lag: int | None = None


@dataclass(frozen=True)
class LobbenchDivergenceSummary:
    """The divergence-over-time curves, and the panel means that head them.

    ``mean_curves`` averages each horizon step across the scored features, so
    it is the pooled panel's time axis: index ``k`` is the distance for
    generated messages ``(k*h, (k+1)*h]``.  A rising curve is error
    compounding over generation steps; the pooled scores cannot show it.
    ``final_over_first`` compresses each curve to one ratio for tables, and is
    ``None`` when its first point is unusable rather than pretending 1.0.
    """

    run_name: str
    stock: str
    period: str
    mode: str
    score_path: str
    benchmark_root: str
    benchmark_revision: str
    horizon_length: int
    n_horizons: int
    horizon_axis_messages: tuple[int, ...]
    feature_count: int
    expected_feature_count: int
    is_full_panel: bool
    # Scores the driver asked for that produced no windows at all.  A statistic
    # with an internal lookback longer than the horizon returns nothing for
    # every window -- ``eval.orderflow_imbalance`` is a 100-message rolling
    # mean, so the four OFI scores vanish at any horizon below 100.  The driver
    # does not say so; naming them here is the difference between a panel of 17
    # and a panel of 21 reported under the same label.
    scores_without_windows: tuple[str, ...]
    mean_curves: dict[str, list[float | None]]
    final_over_first: dict[str, float | None]
    feature_curves: dict[str, dict[str, list[float | None]]]
    # Per feature, per distance, per window: the bootstrap interval around that
    # window's point estimate, or ``None`` where the driver did not produce one.
    # A rising curve means nothing without them: the windows are a fifth the
    # size of the pooled sample, so their intervals are correspondingly wider.
    feature_curve_ci: dict[str, dict[str, list[list[float] | None]]]


@dataclass(frozen=True)
class LobbenchSummary:
    """Paper-facing aggregate plus the feature-level audit trail."""

    run_name: str
    stock: str
    period: str
    mode: str
    score_path: str
    benchmark_root: str
    benchmark_revision: str
    feature_count: int
    expected_feature_count: int
    ws21: float | None
    ks21: float | None
    l1_21: float | None
    distances: dict[str, float]
    feature_scores: dict[str, dict[str, float]]


def score_run(run_name: str, manifest_path: str | None = None) -> ScorePlan:
    """Validate an optional score manifest and return the metric plan."""
    if not run_name:
        raise ValueError("run_name must be non-empty")
    rows: int | None = None
    if manifest_path:
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = len(payload)
        elif isinstance(payload, dict):
            rows = len(payload.get("rows", []))
        else:
            raise ValueError(f"{path}: expected JSON object or list")
    return ScorePlan(run_name, metric_names(), manifest_path, rows)


def _glob_count(path: Path, pattern: str) -> int:
    return sum(1 for candidate in path.glob(pattern) if candidate.is_file())


def inventory_inference(inference_dir: str | Path) -> InferenceInventory:
    """Validate the three generated-data directories consumed by LOBBench."""
    root = Path(inference_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"inference directory not found: {root}")

    counts: dict[str, tuple[int, int]] = {}
    for name in ("data_gen", "data_real", "data_cond"):
        directory = root / name
        if not directory.is_dir():
            raise NotADirectoryError(f"missing LOBBench input directory: {directory}")
        messages = _glob_count(directory, "*message*.csv")
        books = _glob_count(directory, "*orderbook*.csv")
        counts[name] = (messages, books)

    for name in ("data_gen", "data_real"):
        messages, books = counts[name]
        if messages == 0 or books == 0:
            raise ValueError(
                f"{root / name}: expected message and orderbook CSVs; "
                f"found messages={messages}, books={books}"
            )

    if counts["data_gen"][0] != counts["data_real"][0]:
        raise ValueError(
            "generated/real message counts differ: "
            f"{counts['data_gen'][0]} != {counts['data_real'][0]}"
        )

    return InferenceInventory(
        inference_dir=str(root),
        generated_messages=counts["data_gen"][0],
        generated_books=counts["data_gen"][1],
        real_messages=counts["data_real"][0],
        real_books=counts["data_real"][1],
        conditional_messages=counts["data_cond"][0],
        conditional_books=counts["data_cond"][1],
    )


def _safe_name(value: str, field: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            f"{field} must start with an alphanumeric character and contain "
            f"only alphanumerics, '.', '_' or '-'; got {value!r}"
        )
    return value


def benchmark_revision(benchmark_root: str | Path) -> str:
    """Return the exact scorer revision, or ``unknown`` outside Git.

    A merged root answers with the hash of its own tree rather than a Git
    ``HEAD``.  The difference is whether the recorded revision resolves anywhere
    else: the Lustre checkout this repo points at by default sits on a branch
    that was never pushed, so the hash in those summaries names a scorer only to
    the machine that ran it, while a tree hash can be recomputed from the files
    it names.
    """
    root = Path(benchmark_root).expanduser().resolve()
    recorded = recorded_revision(root)
    if recorded is not None:
        return recorded
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _replace_directory_link(link: Path, target: Path) -> None:
    """Point ``link`` at ``target``, surviving concurrent shard staging.

    Every shard of one run stages the same three links, and an
    ``sbatch --array`` starts those processes together.  The old
    unlink-then-symlink pair had a window in which a sibling shard saw the
    link half-made and crashed on ``FileExistsError``, so the link is now
    built under a process-private name and moved into place with
    ``os.replace``, which is atomic within a directory.  Concurrent stagers
    of the *same* target all win; a link already pointing elsewhere is
    replaced; a non-symlink in the way is still refused.
    """
    try:
        if os.readlink(link) == str(target):
            return
    except OSError:
        # Either nothing is there or it is not a symlink.  One readlink
        # settles both without the window that `is_symlink()` followed by
        # `exists()` leaves open: `exists()` follows the link, so a sibling
        # that finished staging in between made this look like a real
        # directory in the way.
        pass
    if link.is_dir() and not link.is_symlink():
        raise FileExistsError(
            f"refusing to replace non-symlink LOBBench staging path: {link}"
        )
    staged = link.with_name(f".{link.name}.staging.{os.getpid()}")
    if staged.is_symlink() or staged.exists():
        staged.unlink()
    staged.symlink_to(target, target_is_directory=True)
    os.replace(staged, link)


def stage_inference(
    inference_dir: str | Path,
    staging_root: str | Path,
    run_name: str,
    stock: str,
    period: str,
) -> InferenceInventory:
    """Stage inference directories in LOBBench's model/stock/period layout."""
    run_name = _safe_name(run_name, "run_name")
    stock = _safe_name(stock, "stock")
    period = _safe_name(period, "period")
    inventory = inventory_inference(inference_dir)
    source = Path(inventory.inference_dir)
    destination = Path(staging_root).expanduser().resolve() / run_name / stock / period
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("data_gen", "data_real", "data_cond"):
        _replace_directory_link(destination / name, source / name)
    return inventory


def _normalize_features(
    features: Sequence[str] | None, universe: Sequence[str]
) -> tuple[str, ...]:
    """Reduce a requested subset to declared names, refusing anything else.

    The driver itself only warns when asked for a name it does not have and
    then scores the rest, so a typo there costs a silently short pickle and a
    zero exit code.  Refusing here is what turns that into a failure.
    """
    universe = tuple(universe)
    if features is None:
        return universe
    normalized = tuple(dict.fromkeys(feature.strip() for feature in features if feature.strip()))
    if not normalized:
        raise ValueError("at least one LOBBench feature is required")
    unknown = sorted(set(normalized) - set(universe))
    if unknown:
        raise ValueError(f"unknown LOBBench features: {', '.join(unknown)}")
    return normalized


def score_file_name(
    *,
    mode: str,
    stock: str,
    run_name: str,
    run_id: str,
    shard_index: int | None = None,
    divergence_horizon: int | None = None,
    baseline: bool = False,
) -> str:
    """Reproduce the name the driver will write for this mode and shard.

    The driver's per-mode names differ by more than a prefix, so this mirrors
    ``run_bench.py`` rather than templating one pattern.  sigma-0 never passes
    ``--model_version``, which is the one further difference between the
    unconditional name and the rest.  The divergence name carries the horizon
    between the shard suffix and the run id, so the same run scored at two
    horizons yields two files instead of one overwriting the other.
    """
    descriptor = resolve_mode(mode)
    shard_suffix = "" if shard_index is None else f"_shard{shard_index}"
    if mode == "divergence":
        if divergence_horizon is None:
            raise ValueError("divergence score files are named by their horizon")
        # The REAL baseline the driver writes alongside the curve replaces the
        # run name with the literal REAL; everything else about the name is the
        # same, so it is derived here rather than templated a second time.
        name = "REAL" if baseline else run_name
        return (
            f"{descriptor.file_prefix}_{stock}_{name}{shard_suffix}"
            f"_{divergence_horizon}_{run_id}.pkl"
        )
    return f"{descriptor.file_prefix}_{stock}_{run_name}{shard_suffix}_{run_id}.pkl"


def _layout(output_dir: str | Path) -> tuple[Path, Path]:
    """Return the staging and results directories sigma-0 writes under."""
    out = Path(output_dir).expanduser().resolve()
    return out / "bench_data", out / "results"


def build_lobbench_plan(
    *,
    run_name: str,
    inference_dir: str | Path,
    output_dir: str | Path,
    stock: str = "GOOG",
    period: str = "2026-01",
    benchmark_root: str | Path = DEFAULT_BENCHMARK_ROOT,
    python: str = sys.executable,
    n_workers: int = 48,
    run_id: str = "sigma0",
    features: Sequence[str] | None = None,
    mode: str = DEFAULT_SCORING_MODE,
    shard_index: int | None = None,
    shard_count: int | None = None,
    divergence_horizon: int = DEFAULT_DIVERGENCE_HORIZON,
    with_divergence: bool = True,
    time_lag: int | None = None,
) -> LobbenchPlan:
    """Validate, stage, and build the authoritative command for one mode.

    ``with_divergence`` makes the unconditional run also emit the
    divergence-over-time curve: the driver accepts both flags in one
    invocation and reuses the loaded data, so the curve costs one extra
    scoring pass and no second load.  It is on by default because a pooled
    WS-21 with no time axis cannot show error compounding over generation
    steps, which is the first thing to look at.  ``time_lag`` overrides the
    lag of the time-lagged mode's config (the driver's ``--time_lag``); it is
    refused for other modes rather than silently ignored.
    """
    descriptor = resolve_mode(mode)
    run_name = _safe_name(run_name, "run_name")
    stock = _safe_name(stock, "stock")
    period = _safe_name(period, "period")
    run_id = _safe_name(run_id, "run_id")
    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be given together")
    if divergence_horizon < 1:
        raise ValueError(f"divergence_horizon must be positive; got {divergence_horizon}")
    if time_lag is not None and mode != "time_lagged":
        raise ValueError(
            f"time_lag only applies to the time_lagged mode; got mode {mode!r}"
        )
    if time_lag is not None and time_lag < 1:
        raise ValueError(f"time_lag must be positive; got {time_lag}")

    root = Path(benchmark_root).expanduser().resolve()
    driver = driver_path(root)

    universe = mode_features(root, mode)
    selected_features = _normalize_features(features, universe)
    if shard_count is not None:
        selected_features = shard_features(
            selected_features, shard_index, shard_count
        )

    staging, results = _layout(output_dir)
    results.mkdir(parents=True, exist_ok=True)
    inventory = stage_inference(inference_dir, staging, run_name, stock, period)

    emits_divergence = mode == "divergence" or (mode == "uncond" and with_divergence)
    score_path = results / "scores" / score_file_name(
        mode=mode,
        stock=stock,
        run_name=run_name,
        run_id=run_id,
        shard_index=shard_index,
        divergence_horizon=divergence_horizon if mode == "divergence" else None,
    )
    divergence_score_path: Path | None = None
    if emits_divergence:
        divergence_score_path = results / "scores" / score_file_name(
            mode="divergence",
            stock=stock,
            run_name=run_name,
            run_id=run_id,
            shard_index=shard_index,
            divergence_horizon=divergence_horizon,
        )
    command = [
        python,
        "-u",
        str(driver),
        "--data_dir",
        str(staging),
        "--model_name",
        run_name,
        "--stock",
        stock,
        "--time_period",
        period,
        "--save_dir",
        str(results),
        descriptor.flag,
        "--run_id",
        run_id,
        "--n_workers",
        str(n_workers),
        "--progress_interval",
        "60",
    ]
    if mode == "uncond" and with_divergence:
        # One driver invocation, both flags: the loaded and materialized
        # sequences are reused, so the curve does not pay a second load.
        command.append(SCORING_MODES["divergence"].flag)
    if emits_divergence:
        command.extend(("--divergence_horizon", str(divergence_horizon)))
    if time_lag is not None:
        command.extend(("--time_lag", str(time_lag)))
    if shard_index is not None:
        command.extend(("--shard_id", str(shard_index)))
    if selected_features != universe:
        command.extend(("--metrics", ",".join(selected_features)))

    return LobbenchPlan(
        run_name=run_name,
        stock=stock,
        period=period,
        mode=mode,
        inference_dir=inventory.inference_dir,
        staging_root=str(staging),
        results_dir=str(results),
        score_path=str(score_path),
        benchmark_root=str(root),
        benchmark_revision=benchmark_revision(root),
        command=tuple(command),
        expected_features=selected_features,
        shard_index=shard_index,
        shard_count=shard_count,
        inventory=inventory,
        divergence_horizon=divergence_horizon if emits_divergence else None,
        divergence_score_path=(
            str(divergence_score_path) if divergence_score_path is not None else None
        ),
        time_lag=time_lag,
    )


def _point_estimate(value: Any, feature: str, distance: str) -> float:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(f"empty score payload for {feature}/{distance}")
        value = value[0]
    try:
        point = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score payload for {feature}/{distance}: {value!r}") from exc
    if not math.isfinite(point):
        raise ValueError(f"non-finite score for {feature}/{distance}: {point}")
    return point


def _load_score_payload(score_path: str | Path) -> tuple[Path, Mapping[str, Any], Any]:
    """Open one score pickle and return its path, scores and score frames."""
    path = Path(score_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LOBBench score file not found: {path}")
    # Score files are generated locally by the pinned benchmark process.  Never
    # use this helper on an untrusted pickle.
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, tuple) or len(payload) < 1:
        raise ValueError(f"{path}: expected (scores, score_dfs) tuple")
    scores = payload[0]
    if not isinstance(scores, Mapping):
        raise ValueError(f"{path}: scores payload is not a mapping")
    return path, scores, payload[1] if len(payload) > 1 else None


def merge_shard_scores(
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    expected_features: Sequence[str],
) -> tuple[str, ...]:
    """Combine per-shard score files, refusing overlap or a missing name.

    Merging shards is a dictionary update, so every way it can go wrong is a
    way it can go wrong quietly: a shard that scored nothing contributes no
    keys, and two shards that scored the same name overwrite each other with
    no trace of which survived.  Both are refused here, and the merged set is
    required to be exactly the set that was asked for.
    """
    expected = tuple(expected_features)
    if not expected:
        raise ValueError("expected_features must be non-empty")
    if not shard_paths:
        raise ValueError("at least one shard file is required")

    merged_scores: dict[str, Any] = {}
    merged_frames: dict[str, Any] = {}
    owners: dict[str, Path] = {}
    for shard_path in shard_paths:
        path, scores, frames = _load_score_payload(shard_path)
        overlap = sorted(set(scores) & set(merged_scores))
        if overlap:
            raise ValueError(
                f"{path}: scores already produced by "
                f"{owners[overlap[0]]}: {', '.join(overlap)}"
            )
        owners.update({name: path for name in scores})
        merged_scores.update(scores)
        if isinstance(frames, Mapping):
            merged_frames.update(frames)

    missing = [feature for feature in expected if feature not in merged_scores]
    unexpected = sorted(set(merged_scores) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"merged shards do not cover the requested scores: "
            f"missing={missing}, unexpected={unexpected}"
        )

    ordered = {feature: merged_scores[feature] for feature in expected}
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wb") as handle:
        pickle.dump((ordered, merged_frames), handle)
    return expected


def summarize_score_file(
    score_path: str | Path,
    *,
    run_name: str,
    stock: str,
    period: str,
    benchmark_root: str | Path,
    benchmark_revision_value: str | None = None,
    expected_features: Sequence[str] | None = None,
    mode: str = DEFAULT_SCORING_MODE,
) -> LobbenchSummary:
    """Load one trusted LOBBench pickle and calculate WS/KS/L1 means."""
    descriptor = resolve_mode(mode)
    path, scores, _ = _load_score_payload(score_path)

    if descriptor.pinned_features is not None:
        # Drift against the driver is checked when the plan is built, which is
        # before any scoring happens.  Repeating it here would only make this
        # function unusable on an archived pickle whose scorer has moved.
        universe: tuple[str, ...] = descriptor.pinned_features
    elif expected_features is None:
        # Nothing was asked for, so the mode's whole declared set is required.
        universe = mode_features(benchmark_root, mode)
    else:
        universe = tuple(expected_features)
    expected = _normalize_features(expected_features, universe)
    actual = set(scores)
    missing = [feature for feature in expected if feature not in actual]
    unexpected = sorted(actual - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"LOBBench feature mismatch: missing={missing}, unexpected={unexpected}"
        )

    feature_scores: dict[str, dict[str, float]] = {}
    values: dict[str, list[float]] = {name: [] for name in metric_names()}
    for feature in expected:
        raw_distances = scores[feature]
        if not isinstance(raw_distances, Mapping):
            raise ValueError(f"{feature}: expected distance mapping")
        feature_scores[feature] = {}
        for distance in metric_names():
            if distance not in raw_distances:
                raise ValueError(f"{feature}: missing {distance} score")
            point = _point_estimate(raw_distances[distance], feature, distance)
            feature_scores[feature][distance] = point
            values[distance].append(point)

    means = {
        distance: sum(points) / len(points)
        for distance, points in values.items()
    }
    # WS-21 names a mean over the 21 unconditional features.  Any other mode
    # averages different quantities over different names, so it leaves these
    # three fields empty rather than publishing its mean under a taken name.
    is_ws21 = mode == "uncond" and expected == LOBBENCH_FEATURES
    full_ws21 = means["wasserstein"] if is_ws21 else None
    full_ks21 = means["ks"] if is_ws21 else None
    full_l1_21 = means["l1"] if is_ws21 else None
    root = Path(benchmark_root).expanduser().resolve()
    return LobbenchSummary(
        run_name=run_name,
        stock=stock,
        period=period,
        mode=mode,
        score_path=str(path),
        benchmark_root=str(root),
        benchmark_revision=(
            benchmark_revision_value
            if benchmark_revision_value is not None
            else benchmark_revision(root)
        ),
        feature_count=len(expected),
        expected_feature_count=len(expected),
        ws21=full_ws21,
        ks21=full_ks21,
        l1_21=full_l1_21,
        distances=means,
        feature_scores=feature_scores,
    )


def _curve_point(value: Any, feature: str, distance: str, index: int) -> float | None:
    """One horizon step of a curve: a finite float, or ``None`` kept honest.

    Unlike the pooled path this does not refuse non-finite values: a late
    horizon window can legitimately have too little data to score, and
    failing the whole summary over it would hide the earlier, valid part of
    the curve.  ``None`` survives JSON; ``NaN`` does not.
    """
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(
                f"empty horizon payload for {feature}/{distance}[{index}]"
            )
        value = value[0]
    try:
        point = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid horizon payload for {feature}/{distance}[{index}]: {value!r}"
        ) from exc
    return point if math.isfinite(point) else None


def _curve_interval(value: Any) -> list[float] | None:
    """The bootstrap interval of one horizon window, when the driver gave one.

    Each window is stored as ``(point, ci, losses)``.  Anything else -- a bare
    float, a pair that will not convert -- yields ``None`` rather than an
    invented interval.
    """
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    bounds = value[1]
    try:
        pair = [float(bound) for bound in bounds]
    except (TypeError, ValueError):
        return None
    if len(pair) != 2 or not all(math.isfinite(bound) for bound in pair):
        return None
    return pair


def summarize_divergence_file(
    score_path: str | Path,
    *,
    run_name: str,
    stock: str,
    period: str,
    benchmark_root: str | Path,
    horizon_length: int,
    benchmark_revision_value: str | None = None,
    expected_features: Sequence[str] | None = None,
) -> LobbenchDivergenceSummary:
    """Load one trusted divergence pickle and lay its curves out per horizon.

    The divergence payload differs from the pooled one by exactly one level:
    ``scores[feature][distance]`` is a *list* of ``(point, ci, losses)``
    tuples, one per horizon window, instead of a single tuple.  Every curve in
    the file must have the same length -- the windows are cut from the same
    sequences by the same rule -- so a length mismatch is a corrupted file,
    not a summarizable one.
    """
    if horizon_length < 1:
        raise ValueError(f"horizon_length must be positive; got {horizon_length}")
    path, scores, _ = _load_score_payload(score_path)

    universe: tuple[str, ...] = SCORING_MODES["divergence"].pinned_features
    expected = _normalize_features(expected_features, universe)
    actual = set(scores)
    missing = [feature for feature in expected if feature not in actual]
    unexpected = sorted(actual - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"LOBBench divergence feature mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    n_horizons: int | None = None
    feature_curves: dict[str, dict[str, list[float | None]]] = {}
    feature_intervals: dict[str, dict[str, list[list[float] | None]]] = {}
    empty: list[str] = []
    for feature in expected:
        raw_curves = scores[feature]
        if not isinstance(raw_curves, Mapping):
            raise ValueError(f"{feature}: expected per-distance curve mapping")
        if all(
            isinstance(raw_curves.get(distance), (list, tuple))
            and not raw_curves[distance]
            for distance in metric_names()
        ):
            # Every distance is an empty list: this score produced no window at
            # this horizon.  Recorded and excluded, never averaged as a zero.
            empty.append(feature)
            continue
        feature_curves[feature] = {}
        for distance in metric_names():
            if distance not in raw_curves:
                raise ValueError(f"{feature}: missing {distance} curve")
            horizon_points = raw_curves[distance]
            if not isinstance(horizon_points, (list, tuple)) or not horizon_points:
                raise ValueError(
                    f"{feature}/{distance}: expected a non-empty list of "
                    f"per-horizon scores; got {type(horizon_points).__name__}"
                )
            curve = [
                _curve_point(value, feature, distance, index)
                for index, value in enumerate(horizon_points)
            ]
            feature_intervals.setdefault(feature, {})[distance] = [
                _curve_interval(value) for value in horizon_points
            ]
            if n_horizons is None:
                n_horizons = len(curve)
            elif len(curve) != n_horizons:
                raise ValueError(
                    f"{feature}/{distance}: {len(curve)} horizons where "
                    f"{n_horizons} were expected; the file is inconsistent"
                )
            feature_curves[feature][distance] = curve

    scored = tuple(f for f in expected if f in feature_curves)
    if n_horizons is None:
        raise ValueError(
            f"{path}: no score produced a window at horizon {horizon_length}; "
            "every requested statistic has a lookback longer than the horizon"
        )
    mean_curves: dict[str, list[float | None]] = {}
    final_over_first: dict[str, float | None] = {}
    for distance in metric_names():
        means: list[float | None] = []
        for index in range(n_horizons):
            points = [feature_curves[f][distance][index] for f in scored]
            finite = [p for p in points if p is not None]
            # A mean over a quietly shrinking feature set would not be the
            # same quantity from one horizon to the next.
            means.append(
                sum(finite) / len(finite) if len(finite) == len(points) else None
            )
        mean_curves[distance] = means
        first, last = means[0], means[-1]
        usable = (
            first is not None and last is not None and first > 0.0
        )
        final_over_first[distance] = (last / first) if usable else None

    root = Path(benchmark_root).expanduser().resolve()
    return LobbenchDivergenceSummary(
        run_name=run_name,
        stock=stock,
        period=period,
        mode="divergence",
        score_path=str(path),
        benchmark_root=str(root),
        benchmark_revision=(
            benchmark_revision_value
            if benchmark_revision_value is not None
            else benchmark_revision(root)
        ),
        horizon_length=horizon_length,
        n_horizons=n_horizons,
        horizon_axis_messages=tuple(
            (index + 1) * horizon_length for index in range(n_horizons)
        ),
        feature_count=len(scored),
        expected_feature_count=len(expected),
        is_full_panel=scored == LOBBENCH_FEATURES,
        scores_without_windows=tuple(empty),
        mean_curves=mean_curves,
        final_over_first=final_over_first,
        feature_curves=feature_curves,
        feature_curve_ci=feature_intervals,
    )


def summarize_divergence_floor(
    score_path: str | Path,
    *,
    expected_features: Sequence[str] | None = None,
) -> dict[str, dict[str, list[list[float] | None]]]:
    """Read the REAL-vs-REAL baseline the driver writes beside the curve.

    Read what this is before using it, because its name invites the wrong
    reading -- one this repository made and then falsified.

    ``--div_error_bounds`` keeps only the rows of a horizon frame whose type is
    ``real``, copies them, and labels the copy ``generated``. The point estimate
    is therefore zero by construction (measured: fifteen of fifteen exactly
    0.0). The interval is the bootstrap spread of the distance between two
    copies of the **entire** real sample.

    That is NOT the floor of the comparison the curve reports.
    ``score_prediction_horizons`` scores all the real data once and reuses it
    for every window; only the generated side is cut. So the observed distance
    compares a full real sample against a window holding a fraction of the
    generated stream, and the binding sample size is the generated window --
    which never enters this baseline at all.

    Measured, which is how the mistake surfaced: halving the horizon from 50 to
    25 halves the generated sample per window and should raise a genuine floor
    by about sqrt(2). Across 17 scores the ratio came out at a median 0.993 --
    flat, because the quantity does not depend on the window.

    Useful for: the noise level of the real side at full sample size. Not
    useful for: dividing an observed per-window distance by, which would be a
    ratio against the wrong reference.
    """
    _, scores, _ = _load_score_payload(score_path)
    wanted = tuple(expected_features) if expected_features else tuple(scores)
    missing = [f for f in wanted if f not in scores]
    if missing:
        raise ValueError(f"floor file is missing scores: {missing}")
    out: dict[str, dict[str, list[list[float] | None]]] = {}
    for feature in wanted:
        per_distance = scores[feature]
        if not isinstance(per_distance, Mapping):
            # The driver's baseline covers one distance for some configs; a
            # bare list is that shape rather than a corrupt file.
            per_distance = {"l1": per_distance}
        out[feature] = {
            distance: [_curve_interval(window) for window in curve]
            for distance, curve in per_distance.items()
        }
    return out


def _benchmark_environment(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    additions = [
        root / "pip_packages_extra",
        root / "pip_packages",
    ]
    visible = [str(path) for path in additions if path.is_dir()]
    if visible:
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(visible + ([current] if current else []))
    return env


def summarize_plan_divergence(plan: LobbenchPlan) -> LobbenchDivergenceSummary:
    """Summarize the divergence curve a finished plan wrote, from the plan."""
    if plan.divergence_score_path is None or plan.divergence_horizon is None:
        raise ValueError("this plan does not produce a divergence curve")
    return summarize_divergence_file(
        plan.divergence_score_path,
        run_name=plan.run_name,
        stock=plan.stock,
        period=plan.period,
        benchmark_root=plan.benchmark_root,
        horizon_length=plan.divergence_horizon,
        benchmark_revision_value=plan.benchmark_revision,
        expected_features=plan.expected_features,
    )


def execute_lobbench_plan(
    plan: LobbenchPlan,
    *,
    summary_path: str | Path | None = None,
) -> LobbenchSummary | LobbenchDivergenceSummary:
    """Execute a plan, require its exact output, and write its JSON summaries.

    A plan that produces a divergence curve gets that curve summarized and
    written no matter which mode carried it: for ``mode="divergence"`` the
    curve is the primary summary and the return value; for an unconditional
    run with the curve alongside, the pooled summary stays the return value
    and keeps its file name (nothing downstream of ``lobbench_summary.json``
    changes shape), while the curve lands next to it in
    ``lobbench_divergence_summary*.json``.
    """
    root = Path(plan.benchmark_root)
    subprocess.run(
        list(plan.command),
        cwd=root / "lob_bench",
        env=_benchmark_environment(root),
        check=True,
    )
    shard_suffix = "" if plan.shard_index is None else f"_shard{plan.shard_index}"

    divergence_summary: LobbenchDivergenceSummary | None = None
    if plan.divergence_score_path is not None:
        divergence_summary = summarize_plan_divergence(plan)

    if plan.mode == "divergence":
        assert divergence_summary is not None
        destination = (
            Path(summary_path).expanduser().resolve()
            if summary_path is not None
            else Path(plan.results_dir)
            / f"lobbench_summary_divergence{shard_suffix}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(divergence_summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return divergence_summary

    summary = summarize_score_file(
        plan.score_path,
        run_name=plan.run_name,
        stock=plan.stock,
        period=plan.period,
        benchmark_root=plan.benchmark_root,
        benchmark_revision_value=plan.benchmark_revision,
        expected_features=plan.expected_features,
        mode=plan.mode,
    )
    default_summary_name = (
        "lobbench_summary.json"
        if plan.mode == DEFAULT_SCORING_MODE and plan.shard_index is None
        else f"lobbench_summary_{plan.mode}{shard_suffix}.json"
    )
    destination = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else Path(plan.results_dir) / default_summary_name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if divergence_summary is not None:
        curve_destination = (
            Path(plan.results_dir) / f"lobbench_divergence_summary{shard_suffix}.json"
        )
        curve_destination.write_text(
            json.dumps(asdict(divergence_summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def merge_lobbench_shards(
    *,
    run_name: str,
    output_dir: str | Path,
    shard_count: int,
    stock: str = "GOOG",
    period: str = "2026-01",
    benchmark_root: str | Path = DEFAULT_BENCHMARK_ROOT,
    run_id: str = "sigma0",
    features: Sequence[str] | None = None,
    mode: str = DEFAULT_SCORING_MODE,
    summary_path: str | Path | None = None,
    divergence_horizon: int = DEFAULT_DIVERGENCE_HORIZON,
    with_divergence: bool = True,
) -> LobbenchSummary | LobbenchDivergenceSummary:
    """Combine one run's shards into the unsharded file and summarize it.

    Shard paths are derived from the same naming function that built them, so
    a shard that never ran is a missing file rather than a glob that quietly
    matches fewer entries than it should.  When the shards were scored by
    unconditional plans that carried the divergence curve (the default), the
    curve's shards are merged and summarized here too; shards produced before
    the curve existed are merged with ``with_divergence=False`` rather than
    inventing an empty curve for them.
    """
    resolve_mode(mode)
    run_name = _safe_name(run_name, "run_name")
    stock = _safe_name(stock, "stock")
    period = _safe_name(period, "period")
    run_id = _safe_name(run_id, "run_id")

    root = Path(benchmark_root).expanduser().resolve()
    universe = mode_features(root, mode)
    expected = _normalize_features(features, universe)
    # Reject a shard layout that could not have produced these scores before
    # touching the filesystem.
    shard_features(expected, 0, shard_count)

    _, results = _layout(output_dir)
    scores_dir = results / "scores"

    def _merge_one_mode(
        merge_mode: str, horizon: int | None
    ) -> Path:
        shard_paths = [
            scores_dir
            / score_file_name(
                mode=merge_mode,
                stock=stock,
                run_name=run_name,
                run_id=run_id,
                shard_index=index,
                divergence_horizon=horizon,
            )
            for index in range(shard_count)
        ]
        merged = scores_dir / score_file_name(
            mode=merge_mode,
            stock=stock,
            run_name=run_name,
            run_id=run_id,
            divergence_horizon=horizon,
        )
        merge_shard_scores(shard_paths, merged, expected_features=expected)
        return merged

    if mode == "divergence":
        merged_path = _merge_one_mode("divergence", divergence_horizon)
        divergence_summary = summarize_divergence_file(
            merged_path,
            run_name=run_name,
            stock=stock,
            period=period,
            benchmark_root=root,
            horizon_length=divergence_horizon,
            expected_features=expected,
        )
        destination = (
            Path(summary_path).expanduser().resolve()
            if summary_path is not None
            else results / "lobbench_summary_divergence.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(divergence_summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return divergence_summary

    merged_path = _merge_one_mode(mode, None)
    summary = summarize_score_file(
        merged_path,
        run_name=run_name,
        stock=stock,
        period=period,
        benchmark_root=root,
        expected_features=expected,
        mode=mode,
    )
    destination = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else results / (
            "lobbench_summary.json"
            if mode == DEFAULT_SCORING_MODE
            else f"lobbench_summary_{mode}.json"
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mode == "uncond" and with_divergence:
        merged_curve_path = _merge_one_mode("divergence", divergence_horizon)
        divergence_summary = summarize_divergence_file(
            merged_curve_path,
            run_name=run_name,
            stock=stock,
            period=period,
            benchmark_root=root,
            horizon_length=divergence_horizon,
            expected_features=expected,
        )
        (results / "lobbench_divergence_summary.json").write_text(
            json.dumps(asdict(divergence_summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary
