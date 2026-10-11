from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pytest

from lobpipeline.lob_bench.scoring import (
    LOBBENCH_FEATURES,
    SCORING_MODES,
    build_lobbench_plan,
    declared_features,
    inventory_inference,
    merge_lobbench_shards,
    merge_shard_scores,
    mode_features,
    resolve_mode,
    score_file_name,
    shard_features,
    summarize_divergence_file,
    summarize_score_file,
)

# What the driver declares for the modes that have no pinned contract.  These
# are fixtures, not a second copy of the contract: the tests that care read
# them back out of the stub driver rather than asserting them from here.
STUB_COND = ("ask_volume | spread", "spread | time", "spread | volatility")
STUB_CONTEXT = ("ask_volume | spread",)
STUB_TIME_LAGGED = (
    "ask_volume | spread (t-lag=500)",
    "ask_volume_touch | ofi (t-lag=500)",
)


def _write_inference_tree(root: Path, n: int = 2) -> None:
    for directory in ("data_gen", "data_real", "data_cond"):
        target = root / directory
        target.mkdir(parents=True)
        for index in range(n):
            (target / f"GOOG_2026-01-02_message_real_id_{index}.csv").write_text(
                "1,2,3\n", encoding="utf-8"
            )
            (target / f"GOOG_2026-01-02_orderbook_real_id_{index}.csv").write_text(
                "1,2,3\n", encoding="utf-8"
            )


def _write_driver(root: Path, **overrides: tuple[str, ...]) -> Path:
    """Write a driver stub declaring the score names sigma-0 parses out of it.

    sigma-0 reads these names from the driver's source rather than mirroring
    them, so the stub has to be a parseable module with real dict literals.
    """
    declared: dict[str, tuple[str, ...]] = {
        "DEFAULT_SCORING_CONFIG": LOBBENCH_FEATURES,
        "DEFAULT_SCORING_CONFIG_COND": STUB_COND,
        "DEFAULT_SCORING_CONFIG_CONTEXT": STUB_CONTEXT,
        "DEFAULT_SCORING_CONFIG_TIME_LAGGED": STUB_TIME_LAGGED,
    }
    declared.update(overrides)
    driver = root / "lob_bench" / "run_bench.py"
    driver.parent.mkdir(parents=True, exist_ok=True)
    body = ['"""Driver stub."""\n']
    for name, keys in declared.items():
        entries = "".join(f"    {key!r}: None,\n" for key in keys)
        body.append(f"\n{name} = {{\n{entries}}}\n")
    driver.write_text("".join(body), encoding="utf-8")
    return driver


def _write_scores(path: Path, features: tuple[str, ...], offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump((_score_payload(features, offset), {}), handle)


def _score_payload(features: tuple[str, ...], offset: int = 0) -> dict:
    return {
        feature: {
            "l1": (float(index + 1 + offset), [0.0, 1.0], []),
            "wasserstein": (float(2 * (index + 1 + offset)), [0.0, 1.0], []),
            "ks": (float(3 * (index + 1 + offset)), [0.0, 1.0], []),
        }
        for index, feature in enumerate(features)
    }


def _load_scores(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)[0]


def _divergence_payload(
    features: tuple[str, ...], n_horizons: int = 3
) -> dict:
    """A curve pickle payload: per feature, per distance, one tuple per horizon.

    The l1 point for feature ``i`` at horizon ``h`` is ``(i+1)*(h+1)``, so the
    across-feature mean at horizon ``h`` is ``(h+1) * mean(i+1)`` and the
    final/first ratio is exactly ``n_horizons`` -- both checkable in closed
    form by the tests.
    """
    factors = {"l1": 1.0, "wasserstein": 2.0, "ks": 3.0}
    return {
        feature: {
            distance: [
                (factor * (index + 1) * (h + 1), [0.0, 1.0], [])
                for h in range(n_horizons)
            ]
            for distance, factor in factors.items()
        }
        for index, feature in enumerate(features)
    }


def _write_divergence_scores(
    path: Path, features: tuple[str, ...], n_horizons: int = 3
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump((_divergence_payload(features, n_horizons), {}), handle)


def test_inventory_and_plan_stage_real_inference_contract(tmp_path: Path) -> None:
    inference = tmp_path / "inference"
    benchmark = tmp_path / "benchmark"
    output = tmp_path / "output"
    _write_inference_tree(inference)
    _write_driver(benchmark)

    inventory = inventory_inference(inference)
    assert inventory.generated_messages == 2
    assert inventory.real_books == 2

    plan = build_lobbench_plan(
        run_name="mamba3-s5",
        inference_dir=inference,
        output_dir=output,
        benchmark_root=benchmark,
        python="python3",
        run_id="smoke",
        features=("spread", "orderbook_imbalance"),
    )
    staged = output / "bench_data" / "mamba3-s5" / "GOOG" / "2026-01"
    assert (staged / "data_gen").resolve() == (inference / "data_gen").resolve()
    assert plan.expected_features == ("spread", "orderbook_imbalance")
    assert plan.command[-2:] == ("--metrics", "spread,orderbook_imbalance")
    assert plan.score_path.endswith(
        "scores/scores_uncond_GOOG_mamba3-s5_smoke.pkl"
    )
    assert plan.mode == "uncond"
    assert plan.shard_index is None


def test_ws21_summary_requires_and_averages_all_features(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.pkl"
    _write_scores(score_path, LOBBENCH_FEATURES)
    summary = summarize_score_file(
        score_path,
        run_name="mamba3-s5",
        stock="GOOG",
        period="2026-01",
        benchmark_root=tmp_path,
        benchmark_revision_value="abc123",
    )
    mean_index = (1 + len(LOBBENCH_FEATURES)) / 2
    assert summary.l1_21 == mean_index
    assert summary.ws21 == 2 * mean_index
    assert summary.ks21 == 3 * mean_index
    assert summary.feature_count == 21
    assert summary.benchmark_revision == "abc123"
    assert summary.mode == "uncond"


def test_partial_smoke_summary_is_not_mislabeled_ws21(tmp_path: Path) -> None:
    features = ("spread", "orderbook_imbalance")
    score_path = tmp_path / "scores.pkl"
    _write_scores(score_path, features)
    summary = summarize_score_file(
        score_path,
        run_name="smoke",
        stock="GOOG",
        period="2026-01",
        benchmark_root=tmp_path,
        benchmark_revision_value="abc123",
        expected_features=features,
    )
    assert summary.ws21 is None
    assert summary.distances["wasserstein"] == 3.0


def test_summary_rejects_incomplete_production_scores(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.pkl"
    _write_scores(score_path, LOBBENCH_FEATURES[:-1])
    with pytest.raises(ValueError, match="feature mismatch"):
        summarize_score_file(
            score_path,
            run_name="broken",
            stock="GOOG",
            period="2026-01",
            benchmark_root=tmp_path,
            benchmark_revision_value="abc123",
        )


# --- scoring modes ---------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "flag", "prefix"),
    [
        ("uncond", "--uncond_only", "scores_uncond"),
        ("cond", "--conditional", "scores_cond"),
        ("context", "--context", "scores_context"),
        ("time_lagged", "--time_lagged", "scores_time_lagged"),
    ],
)
def test_each_mode_carries_its_own_flag_and_file_name(
    tmp_path: Path, mode: str, flag: str, prefix: str
) -> None:
    inference = tmp_path / "inference"
    benchmark = tmp_path / "benchmark"
    _write_inference_tree(inference)
    _write_driver(benchmark)

    plan = build_lobbench_plan(
        run_name="mamba3-s5",
        inference_dir=inference,
        output_dir=tmp_path / "output",
        benchmark_root=benchmark,
        python="python3",
        run_id="smoke",
        mode=mode,
    )
    assert flag in plan.command
    assert Path(plan.score_path).name == f"{prefix}_GOOG_mamba3-s5_smoke.pkl"
    # Nothing was subset out, so the driver is not asked to filter anything.
    assert "--metrics" not in plan.command
    assert plan.expected_features == mode_features(benchmark, mode)


def test_contextual_scores_come_from_the_driver_not_a_local_copy(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    _write_driver(
        benchmark,
        DEFAULT_SCORING_CONFIG_CONTEXT=("ask_volume | spread", "bid_volume | spread"),
    )
    # A second entry enabled upstream is picked up here without a sigma-0 edit,
    # which is the whole reason these names are not mirrored in this repo.
    assert declared_features(benchmark, "context") == (
        "ask_volume | spread",
        "bid_volume | spread",
    )


def test_unconditional_contract_drift_is_refused(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    _write_driver(benchmark, DEFAULT_SCORING_CONFIG=LOBBENCH_FEATURES[:-1])
    with pytest.raises(ValueError, match="contract drift"):
        mode_features(benchmark, "uncond")


def test_a_driver_without_the_mode_is_named_in_the_error(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    driver = benchmark / "lob_bench" / "run_bench.py"
    driver.parent.mkdir(parents=True)
    driver.write_text("DEFAULT_SCORING_CONFIG = {'spread': None}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not support mode 'context'"):
        declared_features(benchmark, "context")


def test_unknown_mode_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown scoring mode"):
        resolve_mode("regime")


def test_only_the_unconditional_namespace_pins_its_scores() -> None:
    # The divergence mode scores the unconditional config per horizon window,
    # so it carries the same 21-name contract; the conditioned modes have no
    # published contract and read their names from the driver.
    assert SCORING_MODES["uncond"].pinned_features == LOBBENCH_FEATURES
    assert SCORING_MODES["divergence"].pinned_features == LOBBENCH_FEATURES
    for name, descriptor in SCORING_MODES.items():
        if name not in ("uncond", "divergence"):
            assert descriptor.pinned_features is None


def test_non_unconditional_summary_leaves_the_ws21_fields_empty(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    _write_driver(benchmark)
    score_path = tmp_path / "scores_context.pkl"
    _write_scores(score_path, STUB_CONTEXT)
    summary = summarize_score_file(
        score_path,
        run_name="mamba3-s5",
        stock="GOOG",
        period="2026-01",
        benchmark_root=benchmark,
        benchmark_revision_value="abc123",
        mode="context",
    )
    assert summary.mode == "context"
    assert summary.ws21 is None and summary.ks21 is None and summary.l1_21 is None
    assert summary.feature_count == len(STUB_CONTEXT)
    assert summary.distances["wasserstein"] == 2.0


# --- sharding --------------------------------------------------------------


def test_shards_partition_the_scores(tmp_path: Path) -> None:
    shards = [shard_features(LOBBENCH_FEATURES, index, 4) for index in range(4)]
    flat = [feature for shard in shards for feature in shard]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(LOBBENCH_FEATURES)
    assert shards == [shard_features(LOBBENCH_FEATURES, i, 4) for i in range(4)]


def test_a_shard_layout_that_would_leave_a_shard_empty_is_refused() -> None:
    with pytest.raises(ValueError, match="would leave"):
        shard_features(("spread", "ofi"), 0, 3)
    with pytest.raises(ValueError, match="shard_index must be in"):
        shard_features(LOBBENCH_FEATURES, 3, 3)


def test_shard_plan_names_its_shard_and_asks_for_only_its_scores(
    tmp_path: Path,
) -> None:
    inference = tmp_path / "inference"
    benchmark = tmp_path / "benchmark"
    _write_inference_tree(inference)
    _write_driver(benchmark)

    plan = build_lobbench_plan(
        run_name="mamba3-s5",
        inference_dir=inference,
        output_dir=tmp_path / "output",
        benchmark_root=benchmark,
        python="python3",
        run_id="smoke",
        shard_index=1,
        shard_count=3,
    )
    assert plan.expected_features == shard_features(LOBBENCH_FEATURES, 1, 3)
    assert ("--shard_id", "1") == plan.command[
        plan.command.index("--shard_id") : plan.command.index("--shard_id") + 2
    ]
    assert plan.command[-2:] == ("--metrics", ",".join(plan.expected_features))
    assert Path(plan.score_path).name == "scores_uncond_GOOG_mamba3-s5_shard1_smoke.pkl"


def test_half_a_shard_layout_is_refused(tmp_path: Path) -> None:
    inference = tmp_path / "inference"
    benchmark = tmp_path / "benchmark"
    _write_inference_tree(inference)
    _write_driver(benchmark)
    with pytest.raises(ValueError, match="must be given together"):
        build_lobbench_plan(
            run_name="mamba3-s5",
            inference_dir=inference,
            output_dir=tmp_path / "output",
            benchmark_root=benchmark,
            python="python3",
            shard_index=0,
        )


def test_merge_rebuilds_the_whole_run_scores_exactly(tmp_path: Path) -> None:
    whole = _score_payload(LOBBENCH_FEATURES)
    paths = []
    for index in range(3):
        subset = shard_features(LOBBENCH_FEATURES, index, 3)
        path = tmp_path / f"shard{index}.pkl"
        with gzip.open(path, "wb") as handle:
            pickle.dump(({name: whole[name] for name in subset}, {}), handle)
        paths.append(path)

    merged_path = tmp_path / "merged.pkl"
    merge_shard_scores(paths, merged_path, expected_features=LOBBENCH_FEATURES)
    merged = _load_scores(merged_path)
    assert tuple(merged) == LOBBENCH_FEATURES
    assert merged == whole


def test_merge_refuses_a_shard_that_scored_nothing(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        subset = shard_features(LOBBENCH_FEATURES, index, 3)
        if index == 2:
            subset = subset[:-1]
        path = tmp_path / f"shard{index}.pkl"
        _write_scores(path, subset)
        paths.append(path)
    with pytest.raises(ValueError, match="missing="):
        merge_shard_scores(
            paths, tmp_path / "merged.pkl", expected_features=LOBBENCH_FEATURES
        )


def test_merge_refuses_two_shards_that_scored_the_same_name(tmp_path: Path) -> None:
    first = tmp_path / "shard0.pkl"
    second = tmp_path / "shard1.pkl"
    _write_scores(first, LOBBENCH_FEATURES[:11])
    _write_scores(second, LOBBENCH_FEATURES[10:])
    with pytest.raises(ValueError, match="already produced by"):
        merge_shard_scores(
            [first, second], tmp_path / "merged.pkl", expected_features=LOBBENCH_FEATURES
        )


def test_merge_entry_point_derives_shard_paths_and_summarizes(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    _write_driver(benchmark)
    output = tmp_path / "output"
    scores_dir = output / "results" / "scores"
    whole = _score_payload(LOBBENCH_FEATURES)
    whole_curves = _divergence_payload(LOBBENCH_FEATURES)
    for index in range(3):
        subset = shard_features(LOBBENCH_FEATURES, index, 3)
        path = scores_dir / f"scores_uncond_GOOG_mamba3-s5_shard{index}_smoke.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle:
            pickle.dump(({name: whole[name] for name in subset}, {}), handle)
        # The default unconditional plan writes the divergence curve
        # alongside the pooled scores, so the shards to merge carry both.
        curve_path = (
            scores_dir / f"scores_div_GOOG_mamba3-s5_shard{index}_100_smoke.pkl"
        )
        with gzip.open(curve_path, "wb") as handle:
            pickle.dump(({name: whole_curves[name] for name in subset}, {}), handle)

    summary = merge_lobbench_shards(
        run_name="mamba3-s5",
        output_dir=output,
        shard_count=3,
        benchmark_root=benchmark,
        run_id="smoke",
    )
    # The sharded scores aggregate to the same WS-21 as the unsharded ones:
    # only the bootstrap intervals depend on how the work was split, and the
    # summary is built from point estimates.
    mean_index = (1 + len(LOBBENCH_FEATURES)) / 2
    assert summary.ws21 == 2 * mean_index
    assert summary.feature_count == 21
    assert (output / "results" / "lobbench_summary.json").is_file()
    merged = _load_scores(
        scores_dir / "scores_uncond_GOOG_mamba3-s5_smoke.pkl"
    )
    assert tuple(merged) == LOBBENCH_FEATURES
    # The curve shards were merged and summarized alongside the pooled ones.
    assert (output / "results" / "lobbench_divergence_summary.json").is_file()
    merged_curves = _load_scores(
        scores_dir / "scores_div_GOOG_mamba3-s5_100_smoke.pkl"
    )
    assert tuple(merged_curves) == LOBBENCH_FEATURES


def test_merge_without_divergence_serves_pre_curve_shards(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    _write_driver(benchmark)
    output = tmp_path / "output"
    scores_dir = output / "results" / "scores"
    for index in range(3):
        subset = shard_features(LOBBENCH_FEATURES, index, 3)
        _write_scores(
            scores_dir / f"scores_uncond_GOOG_mamba3-s5_shard{index}_smoke.pkl", subset
        )
    # Shards scored before the curve existed have no scores_div files; the
    # explicit opt-out merges them instead of demanding a curve they never had.
    summary = merge_lobbench_shards(
        run_name="mamba3-s5",
        output_dir=output,
        shard_count=3,
        benchmark_root=benchmark,
        run_id="smoke",
        with_divergence=False,
    )
    assert summary.feature_count == 21
    assert not (output / "results" / "lobbench_divergence_summary.json").exists()


def test_merge_reports_the_shard_that_never_ran(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    _write_driver(benchmark)
    output = tmp_path / "output"
    scores_dir = output / "results" / "scores"
    for index in range(2):
        subset = shard_features(LOBBENCH_FEATURES, index, 3)
        _write_scores(
            scores_dir / f"scores_uncond_GOOG_mamba3-s5_shard{index}_smoke.pkl", subset
        )
    with pytest.raises(FileNotFoundError, match="shard2"):
        merge_lobbench_shards(
            run_name="mamba3-s5",
            output_dir=output,
            shard_count=3,
            benchmark_root=benchmark,
            run_id="smoke",
        )

# --- divergence over time --------------------------------------------------


def test_unconditional_plan_carries_the_divergence_curve_by_default(
    tmp_path: Path,
) -> None:
    inference = tmp_path / "inference"
    benchmark = tmp_path / "benchmark"
    _write_inference_tree(inference)
    _write_driver(benchmark)
    plan = build_lobbench_plan(
        run_name="mamba3-s5",
        inference_dir=inference,
        output_dir=tmp_path / "output",
        benchmark_root=benchmark,
        python="python3",
        run_id="smoke",
    )
    assert "--divergence" in plan.command
    assert ("--divergence_horizon", "100") == tuple(
        plan.command[plan.command.index("--divergence_horizon") :][:2]
    )
    assert plan.divergence_horizon == 100
    assert Path(plan.divergence_score_path).name == (
        "scores_div_GOOG_mamba3-s5_100_smoke.pkl"
    )
    # The pooled score file itself is unchanged by the companion curve.
    assert Path(plan.score_path).name == "scores_uncond_GOOG_mamba3-s5_smoke.pkl"


def test_the_curve_can_be_declined_and_other_modes_never_carry_it(
    tmp_path: Path,
) -> None:
    inference = tmp_path / "inference"
    benchmark = tmp_path / "benchmark"
    _write_inference_tree(inference)
    _write_driver(benchmark)
    declined = build_lobbench_plan(
        run_name="mamba3-s5",
        inference_dir=inference,
        output_dir=tmp_path / "output",
        benchmark_root=benchmark,
        python="python3",
        run_id="smoke",
        with_divergence=False,
    )
    assert "--divergence" not in declined.command
    assert declined.divergence_score_path is None
    for mode in ("cond", "context", "time_lagged"):
        plan = build_lobbench_plan(
            run_name="mamba3-s5",
            inference_dir=inference,
            output_dir=tmp_path / "output",
            benchmark_root=benchmark,
            python="python3",
            run_id="smoke",
            mode=mode,
        )
        assert "--divergence" not in plan.command
        assert plan.divergence_score_path is None


def test_divergence_mode_names_its_file_by_horizon(tmp_path: Path) -> None:
    inference = tmp_path / "inference"
    benchmark = tmp_path / "benchmark"
    _write_inference_tree(inference)
    _write_driver(benchmark)
    plan = build_lobbench_plan(
        run_name="mamba3-s5",
        inference_dir=inference,
        output_dir=tmp_path / "output",
        benchmark_root=benchmark,
        python="python3",
        run_id="smoke",
        mode="divergence",
        divergence_horizon=50,
    )
    assert "--divergence" in plan.command
    assert Path(plan.score_path).name == "scores_div_GOOG_mamba3-s5_50_smoke.pkl"
    assert plan.score_path == plan.divergence_score_path
    assert plan.expected_features == LOBBENCH_FEATURES
    with pytest.raises(ValueError, match="named by their horizon"):
        score_file_name(
            mode="divergence", stock="GOOG", run_name="m", run_id="smoke"
        )


def test_time_lag_reaches_the_driver_and_only_for_time_lagged(
    tmp_path: Path,
) -> None:
    inference = tmp_path / "inference"
    benchmark = tmp_path / "benchmark"
    _write_inference_tree(inference)
    _write_driver(benchmark)
    plan = build_lobbench_plan(
        run_name="mamba3-s5",
        inference_dir=inference,
        output_dir=tmp_path / "output",
        benchmark_root=benchmark,
        python="python3",
        run_id="smoke",
        mode="time_lagged",
        time_lag=200,
    )
    assert ("--time_lag", "200") == tuple(
        plan.command[plan.command.index("--time_lag") :][:2]
    )
    with pytest.raises(ValueError, match="only applies to the time_lagged"):
        build_lobbench_plan(
            run_name="mamba3-s5",
            inference_dir=inference,
            output_dir=tmp_path / "output",
            benchmark_root=benchmark,
            python="python3",
            run_id="smoke",
            mode="uncond",
            time_lag=200,
        )


def test_divergence_summary_lays_out_curves_and_panel_means(
    tmp_path: Path,
) -> None:
    score_path = tmp_path / "scores_div.pkl"
    _write_divergence_scores(score_path, LOBBENCH_FEATURES, n_horizons=3)
    summary = summarize_divergence_file(
        score_path,
        run_name="mamba3-s5",
        stock="GOOG",
        period="2026-01",
        benchmark_root=tmp_path,
        horizon_length=100,
        benchmark_revision_value="abc123",
    )
    assert summary.n_horizons == 3
    assert summary.horizon_axis_messages == (100, 200, 300)
    assert summary.is_full_panel is True
    mean_index = (1 + len(LOBBENCH_FEATURES)) / 2
    assert summary.mean_curves["l1"] == [
        mean_index,
        2 * mean_index,
        3 * mean_index,
    ]
    assert summary.mean_curves["wasserstein"][0] == 2 * mean_index
    # The l1 point triples from first to last horizon by construction.
    assert summary.final_over_first["l1"] == pytest.approx(3.0)
    assert summary.feature_curves["spread"]["ks"] == [3.0, 6.0, 9.0]


def test_divergence_summary_keeps_a_nan_window_honest(tmp_path: Path) -> None:
    payload = _divergence_payload(("spread", "ofi"), n_horizons=2)
    payload["spread"]["l1"][1] = (float("nan"), [0.0, 1.0], [])
    score_path = tmp_path / "scores_div.pkl"
    with gzip.open(score_path, "wb") as handle:
        pickle.dump((payload, {}), handle)
    summary = summarize_divergence_file(
        score_path,
        run_name="m",
        stock="GOOG",
        period="2026-01",
        benchmark_root=tmp_path,
        horizon_length=100,
        benchmark_revision_value="abc123",
        expected_features=("spread", "ofi"),
    )
    # The unusable window is None in the curve, and the panel mean at that
    # horizon refuses to average over a quietly smaller feature set.
    assert summary.feature_curves["spread"]["l1"] == [1.0, None]
    assert summary.mean_curves["l1"][1] is None
    assert summary.final_over_first["l1"] is None


def test_divergence_summary_rejects_ragged_curves(tmp_path: Path) -> None:
    payload = _divergence_payload(("spread", "ofi"), n_horizons=2)
    payload["ofi"]["ks"] = payload["ofi"]["ks"][:1]
    score_path = tmp_path / "scores_div.pkl"
    with gzip.open(score_path, "wb") as handle:
        pickle.dump((payload, {}), handle)
    with pytest.raises(ValueError, match="horizons"):
        summarize_divergence_file(
            score_path,
            run_name="m",
            stock="GOOG",
            period="2026-01",
            benchmark_root=tmp_path,
            horizon_length=100,
            benchmark_revision_value="abc123",
            expected_features=("spread", "ofi"),
        )

# --- staging under concurrency ---------------------------------------------


def _stage_once(args: tuple[str, str]) -> str | None:
    inference, staging = args
    from lobpipeline.lob_bench.scoring import stage_inference

    try:
        stage_inference(inference, staging, "mamba3-s5", "GOOG", "2026-01")
        return None
    except Exception as exc:  # noqa: BLE001 - the failure mode is the assert
        return f"{type(exc).__name__}: {exc}"


def test_concurrent_shards_can_stage_the_same_run(tmp_path: Path) -> None:
    """Array shards of one run stage the same links at the same moment.

    The unlink-then-symlink implementation lost this race often enough that
    six concurrent stagers failed more runs than they completed; staging is
    now atomic per link, so every shard must succeed.
    """
    import multiprocessing

    inference = tmp_path / "inference"
    _write_inference_tree(inference)
    staging = tmp_path / "staging"
    jobs = [(str(inference), str(staging))] * 24
    with multiprocessing.Pool(6) as pool:
        failures = [error for error in pool.map(_stage_once, jobs) if error]
    assert failures == []
    for name in ("data_gen", "data_real", "data_cond"):
        link = staging / "mamba3-s5" / "GOOG" / "2026-01" / name
        assert link.is_symlink()
        assert link.resolve() == (inference / name).resolve()


def test_staging_replaces_a_stale_link_and_refuses_a_real_directory(
    tmp_path: Path,
) -> None:
    from lobpipeline.lob_bench.scoring import stage_inference

    old = tmp_path / "old_inference"
    new = tmp_path / "new_inference"
    _write_inference_tree(old)
    _write_inference_tree(new)
    staging = tmp_path / "staging"
    stage_inference(old, staging, "run", "GOOG", "2026-01")
    stage_inference(new, staging, "run", "GOOG", "2026-01")
    link = staging / "run" / "GOOG" / "2026-01" / "data_gen"
    assert link.resolve() == (new / "data_gen").resolve()

    blocked = staging / "run" / "GOOG" / "2026-02"
    blocked.mkdir(parents=True)
    (blocked / "data_gen").mkdir()
    with pytest.raises(FileExistsError, match="non-symlink"):
        stage_inference(new, staging, "run", "GOOG", "2026-02")


def test_divergence_summary_carries_each_window_interval(tmp_path: Path) -> None:
    """A rising curve means nothing without the intervals around its points."""
    payload = _divergence_payload(("spread", "ofi"), n_horizons=2)
    payload["spread"]["l1"][0] = (1.0, [0.9, 1.1], [])
    payload["spread"]["l1"][1] = (2.0, "not an interval", [])
    score_path = tmp_path / "scores_div.pkl"
    with gzip.open(score_path, "wb") as handle:
        pickle.dump((payload, {}), handle)
    summary = summarize_divergence_file(
        score_path,
        run_name="m",
        stock="GOOG",
        period="2026-01",
        benchmark_root=tmp_path,
        horizon_length=100,
        benchmark_revision_value="abc123",
        expected_features=("spread", "ofi"),
    )
    assert summary.feature_curve_ci["spread"]["l1"][0] == [0.9, 1.1]
    # An unusable interval is absent, not invented.
    assert summary.feature_curve_ci["spread"]["l1"][1] is None
    assert summary.feature_curve_ci["ofi"]["wasserstein"][0] == [0.0, 1.0]
