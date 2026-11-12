from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import pandas as pd
import pytest

from lobpipeline.lob_bench.scoring import LOBBENCH_FEATURES, LobbenchSummary
from lobpipeline.lob_bench.standard_eval import (
    STANDARD_TIME_LAG,
    binned_kl,
    build_record,
    load_protocol,
    optional_mode_section,
    pooled_kl,
)
from lobpipeline.scorer import read_merge_record
DISTANCES = ("l1", "wasserstein", "ks")


def _index(tmp_path: Path, indices: list[int]) -> Path:
    path = tmp_path / "idx.txt"
    text = "".join(f"{index}\n" for index in indices)
    path.write_text(text, encoding="utf-8")
    metadata = {
        "conditioning_messages": 250,
        "generated_messages": 250,
        "sequence_messages": 500,
        "n_samples": len(indices),
        "dataset_length": 1000,
        "period": "2026-01",
        "ticker": "GOOG",
        "sample_indices_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    path.with_name(path.name + ".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return path


def _pooled(revision: str) -> LobbenchSummary:
    return LobbenchSummary(
        run_name="r",
        stock="GOOG",
        period="2026-01",
        mode="uncond",
        score_path="/x/scores.pkl",
        benchmark_root="/x",
        benchmark_revision=revision,
        feature_count=21,
        expected_feature_count=21,
        ws21=0.5,
        ks21=0.2,
        l1_21=0.3,
        distances={"l1": 0.3, "wasserstein": 0.5, "ks": 0.2},
        feature_scores={f: {d: 0.1 for d in DISTANCES} for f in LOBBENCH_FEATURES},
    )


def _mode_summary(mode: str, revision: str) -> LobbenchSummary:
    return LobbenchSummary(
        run_name="r",
        stock="GOOG",
        period="2026-01",
        mode=mode,
        score_path=f"/x/scores_{mode}.pkl",
        benchmark_root="/x",
        benchmark_revision=revision,
        feature_count=1,
        expected_feature_count=1,
        ws21=None,
        ks21=None,
        l1_21=None,
        distances={"l1": 0.0, "wasserstein": 0.4, "ks": 0.2},
        feature_scores={"ask_volume | spread": {"l1": 0.0, "wasserstein": 0.4, "ks": 0.2}},
    )


def test_identical_bins_give_zero_kl() -> None:
    assert binned_kl([1, 1, 2, 3], [1, 1, 2, 3]) == 0.0


def test_a_bin_the_generated_data_never_reaches_is_large_but_finite() -> None:
    value = binned_kl([1] * 100, [2] * 100)
    assert value is not None and math.isfinite(value) and value > 3.0


def test_no_observations_is_no_kl() -> None:
    assert binned_kl([], [1, 2]) is None
    assert binned_kl([1, 2], []) is None


def test_pooled_kl_reads_the_scorer_bins_from_the_pickle(tmp_path: Path) -> None:
    frames = {
        feature: pd.DataFrame(
            {
                "score": [1.0, 2.0, float("nan"), 1.0, 1.0],
                "group": [1, 2, 3, 1, 1],
                "type": ["real", "real", "generated", "generated", "generated"],
            }
        )
        for feature in LOBBENCH_FEATURES
    }
    path = tmp_path / "scores.pkl"
    with gzip.open(path, "wb") as handle:
        pickle.dump(({}, frames), handle)
    kl = pooled_kl(path)
    # The NaN row is not an observation, so bin 3 is not a bin.
    expected = binned_kl([1, 2], [1, 1])
    assert kl["features"]["spread"] == pytest.approx(expected)
    assert kl["mean"] == pytest.approx(expected)
    assert kl["direction"] == "real || generated"


def test_context_l1_is_not_reported() -> None:
    revision = read_merge_record()["tree_sha256"]
    section = optional_mode_section(_mode_summary("context", revision), scorer_tree_sha256=revision)
    assert section["features"]["ask_volume | spread"]["l1"] is None
    assert section["means"]["l1"] is None
    assert section["means"]["wasserstein"] == 0.4
    assert "structural zero" in section["note"]


def test_time_lagged_section_states_the_lag_that_ran() -> None:
    revision = read_merge_record()["tree_sha256"]
    section = optional_mode_section(
        _mode_summary("time_lagged", revision),
        scorer_tree_sha256=revision,
        time_lag=STANDARD_TIME_LAG,
    )
    assert section["time_lag"] == 250
    assert section["means"]["l1"] == 0.0


def test_an_optional_mode_scored_elsewhere_is_refused() -> None:
    revision = read_merge_record()["tree_sha256"]
    with pytest.raises(ValueError, match="summary names scorer"):
        optional_mode_section(_mode_summary("cond", "0" * 64), scorer_tree_sha256=revision)


def _record(tmp_path: Path, **overrides):
    revision = read_merge_record()["tree_sha256"]
    arguments = dict(
        run_name="r",
        protocol=load_protocol(_index(tmp_path, [3, 7]), divergence=False),
        scorer_tree_sha256=revision,
        pooled=_pooled(revision),
        divergence=None,
        intervals={},
        inputs={},
        provenance={},
    )
    arguments.update(overrides)
    return build_record(**arguments)


def test_extras_are_null_unless_run(tmp_path: Path) -> None:
    record = _record(tmp_path)
    for name in ("kl", "conditional", "context", "time_lagged"):
        assert record[name] is None
    assert record["headline"]["ws21"] == 0.5


def test_extras_land_in_their_sections_and_leave_the_headline(tmp_path: Path) -> None:
    revision = read_merge_record()["tree_sha256"]
    kl = {"mean": 0.1, "features": {feature: 0.1 for feature in LOBBENCH_FEATURES}}
    cond = optional_mode_section(_mode_summary("cond", revision), scorer_tree_sha256=revision)
    record = _record(tmp_path, kl=kl, optional={"cond": cond})
    assert record["kl"]["mean"] == 0.1
    assert record["conditional"]["mode"] == "cond"
    assert record["context"] is None and record["time_lagged"] is None
    assert record["headline"] == {"ws21": 0.5, "ks21": 0.2, "l1_21": 0.3}


def test_a_kl_section_missing_features_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="21 pinned features"):
        _record(tmp_path, kl={"mean": 0.1, "features": {"spread": 0.1}})


def test_an_unknown_optional_mode_is_refused(tmp_path: Path) -> None:
    revision = read_merge_record()["tree_sha256"]
    section = optional_mode_section(_mode_summary("cond", revision), scorer_tree_sha256=revision)
    with pytest.raises(ValueError, match="unknown optional modes"):
        _record(tmp_path, optional={"divergence": section})
    with pytest.raises(ValueError, match="holds mode"):
        _record(tmp_path, optional={"context": section})
