"""The impact score must be eq. 7 of LOB-Bench: a mean over lags, not a sum.

Nagy et al. (2025), "LOB-Bench", section 4.1, define the per-event dissimilarity

    Delta R_pi = (1/L) * sum_{l=1..L} |R_pi^real(l) - R_pi^gen(l)|

and the reported score as the mean of Delta R_pi over the six event types.
`impact_compare` summed over the lag grid instead, so every per-event value and
`Mean_Diffs` came out L times the paper's quantity, with L the grid length.

The tests replace the per-lag estimator with a stub that returns known means,
so they check the aggregation only. The stub values differ by lag, so a sum
and a mean cannot agree by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

SCORER = Path(__file__).resolve().parents[2] / "src" / "lobpipeline" / "scorer" / "lob_bench"
EVENTS = ("MO_0", "MO_1", "LO_0", "LO_1", "CA_0", "CA_1")


@pytest.fixture(scope="module")
def impact():
    """Import the scorer module by path; it uses bare module names internally."""
    sys.path.insert(0, str(SCORER))
    try:
        import impact as module
    except ImportError as exc:  # the scoring dependency set is not installed here
        pytest.skip(f"scorer not importable: {exc}")
    return module


def _stub(table, seen):
    """Stand in for _apply_to_Sequences: look up (real, gen) means by event and lag."""
    ci = (0.0, 0.0)

    def fake(fun, CI, max_seq, args, loader):
        event, lag, _ticksize = args
        seen.append((event, lag))
        real, gen = table[event][lag]
        return (real, ci), (gen, ci)

    return fake


def _run(impact, monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    seen = []
    table = {}
    grid = sorted({int(v) for v in (10 ** np.arange(0.3, 2.4, step=0.1)).astype(int)})
    for event in EVENTS:
        table[event] = {lag: (float(rng.normal()), float(rng.normal())) for lag in grid}
    monkeypatch.setattr(impact, "_apply_to_Sequences", _stub(table, seen))
    res = impact.impact_compare(loader=None, ticker="T", model="M", save_dir=str(tmp_path))
    return res, table, seen


def test_per_event_score_is_the_mean_over_the_evaluated_lags(impact, monkeypatch, tmp_path):
    res, _table, _seen = _run(impact, monkeypatch, tmp_path)
    x, ys_real, ys_gen, _ci_r, _ci_g, abs_diffs = res["Saved_Tuple"]
    for k, _event in enumerate(EVENTS):
        per_lag = np.abs(np.asarray(ys_real[k]) - np.asarray(ys_gen[k]))
        assert len(per_lag) == len(x)
        assert abs_diffs[k] == pytest.approx(per_lag.mean())
        # The pre-fix value was the sum, len(x) times larger.
        assert abs_diffs[k] != pytest.approx(per_lag.sum())


def test_mean_diffs_is_the_mean_over_event_types(impact, monkeypatch, tmp_path):
    res, _table, _seen = _run(impact, monkeypatch, tmp_path)
    abs_diffs = res["Saved_Tuple"][-1]
    assert len(abs_diffs) == len(EVENTS)
    assert res["Mean_Diffs"] == pytest.approx(np.mean(abs_diffs))


def test_a_constant_gap_scores_that_gap_whatever_the_grid_length(impact, monkeypatch, tmp_path):
    """If gen sits c ticks from real at every lag, Delta R_pi is c, not L*c."""
    seen = []
    c = 0.25
    grid = sorted({int(v) for v in (10 ** np.arange(0.3, 2.4, step=0.1)).astype(int)})
    table = {e: {lag: (1.0, 1.0 + c) for lag in grid} for e in EVENTS}
    monkeypatch.setattr(impact, "_apply_to_Sequences", _stub(table, seen))
    res = impact.impact_compare(loader=None, ticker="T", model="M", save_dir=str(tmp_path))
    assert res["Mean_Diffs"] == pytest.approx(c)
