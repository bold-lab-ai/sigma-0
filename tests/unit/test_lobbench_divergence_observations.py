"""A divergence window with no generated observation is not a measurement.

The driver writes a distance for such a window anyway: NaN for W1 and KS and
0.0 for L1. These tests pin that ``summarize_divergence_file`` reads the score
frames to tell that 0.0 from a real one.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pandas as pd

from lobpipeline.lob_bench.scoring import summarize_divergence_file

FACTORS = {"l1": 1.0, "wasserstein": 2.0, "ks": 3.0}


def _payload(features: tuple[str, ...], n_horizons: int) -> dict:
    return {
        feature: {
            distance: [
                (factor * (index + 1) * (h + 1), [0.0, 1.0], []) for h in range(n_horizons)
            ]
            for distance, factor in FACTORS.items()
        }
        for index, feature in enumerate(features)
    }


def _frame(finite: int, nan: int = 0) -> pd.DataFrame:
    scores = [0.5] * 3 + [0.1] * finite + [float("nan")] * nan
    types = ["real"] * 3 + ["generated"] * (finite + nan)
    return pd.DataFrame({"score": scores, "group": [0] * len(scores), "type": types})


def _summarize(tmp_path: Path, payload: dict, frames: dict, features: tuple[str, ...]):
    path = tmp_path / "scores_div.pkl"
    with gzip.open(path, "wb") as handle:
        pickle.dump((payload, frames), handle)
    return summarize_divergence_file(
        path,
        run_name="m",
        stock="GOOG",
        period="2026-01",
        benchmark_root=tmp_path,
        horizon_length=100,
        benchmark_revision_value="abc123",
        expected_features=features,
    )


def test_a_score_with_no_generated_observation_has_no_windows(tmp_path: Path) -> None:
    # What the driver writes for OFI inside a 100-message window.
    payload = _payload(("spread", "ofi"), n_horizons=2)
    payload["ofi"] = {
        "l1": [(0.0, [0.0, 0.0], [])] * 2,
        "wasserstein": [(float("nan"), [0.0, 0.0], [])] * 2,
        "ks": [(float("nan"), [0.0, 0.0], [])] * 2,
    }
    frames = {
        "spread": [_frame(5), _frame(5)],
        "ofi": [_frame(0, nan=4), _frame(0, nan=4)],
    }
    summary = _summarize(tmp_path, payload, frames, ("spread", "ofi"))
    assert summary.scores_without_windows == ("ofi",)
    assert "ofi" not in summary.feature_curves
    assert not summary.is_full_panel
    # The panel is spread alone: OFI's 0.0 does not dilute L1, and its NaN
    # does not blank W1.
    assert summary.mean_curves["l1"] == [1.0, 2.0]
    assert summary.mean_curves["wasserstein"] == [2.0, 4.0]


def test_one_unobserved_window_is_none_for_every_distance(tmp_path: Path) -> None:
    payload = _payload(("spread",), n_horizons=2)
    payload["spread"]["l1"][1] = (0.0, [0.0, 0.0], [])
    frames = {"spread": [_frame(5), _frame(0)]}
    summary = _summarize(tmp_path, payload, frames, ("spread",))
    assert summary.feature_curves["spread"]["l1"] == [1.0, None]
    assert summary.feature_curves["spread"]["wasserstein"] == [2.0, None]
    assert summary.mean_curves["l1"] == [1.0, None]


def test_a_pickle_without_frames_is_summarized_as_before(tmp_path: Path) -> None:
    payload = _payload(("spread",), n_horizons=2)
    payload["spread"]["l1"][1] = (0.0, [0.0, 0.0], [])
    summary = _summarize(tmp_path, payload, {}, ("spread",))
    assert summary.feature_curves["spread"]["l1"] == [1.0, 0.0]
