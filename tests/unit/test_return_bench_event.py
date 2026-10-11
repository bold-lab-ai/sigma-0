"""Unit tests for the event-anchored return bench.

Everything here runs on synthetic arrays with closed-form answers. The three
`searchsorted` boundaries this module depends on -- `side='right'`, the `-1`, and the
`j < 0` case -- each correspond to a real mistake (reading the future, skipping a
message that shares the target timestamp, mistaking an untouched book for missing
data), and every one of them produces plausible numbers on real data without raising.
Synthetic inputs are the only thing that pins them down.
"""

from __future__ import annotations

import numpy as np
import pytest

from lobpipeline.return_bench.event_anchored import (
    CSV_RESTING_ASK,
    CSV_RESTING_BID,
    PROC_RESTING_ASK,
    PROC_RESTING_BID,
    compute_metrics,
    distribution_comparison,
    price_at_horizon,
    sample_returns,
    shuffle_null,
)

MS = 1e-3


def _seq(ms_offsets, prices, t0=100.0):
    """Times at t0 + given millisecond offsets, paired with mid prices."""
    return t0 + np.asarray(ms_offsets, dtype=np.float64) * MS, np.asarray(prices, dtype=np.float64)


# --------------------------------------------------------------------------------
# T1 -- the coverage check, and proof that this test can actually fail
# --------------------------------------------------------------------------------

def _naive_price(times, mid, t0, p0, horizon_ms):
    """The implementation without a coverage check, kept only to be tested against.

    This is what the bench looked like before measurement showed that 42% of 500 ms
    samples end early. It never raises and never returns NaN; it just answers a
    different question than the one asked.
    """
    j = int(np.searchsorted(times, t0 + horizon_ms * MS, side="right")) - 1
    return mid[j]


def test_T1_uncovered_horizon_is_reported_not_silently_truncated():
    # Sequence ends 3 ms after the anchor; a 10 ms horizon has no data.
    times, mid = _seq([0, 1, 2, 3], [100.0, 101.0, 102.0, 103.0])
    t0, p0 = times[0], 100.0

    covered, price = price_at_horizon(times, mid, t0, p0, horizon_ms=10.0)
    assert covered is False
    assert np.isnan(price)

    # The naive version returns the last row and looks perfectly healthy. If this
    # assertion ever fails the test above has stopped proving anything.
    assert _naive_price(times, mid, t0, p0, horizon_ms=10.0) == 103.0


def test_T1b_coverage_boundary_is_inclusive():
    times, mid = _seq([0, 10], [100.0, 105.0])
    t0 = times[0]
    # Last message lands exactly on t0 + 10 ms: covered.
    assert price_at_horizon(times, mid, t0, 100.0, 10.0) == (True, 105.0)
    # One microsecond further out and there is no data.
    assert price_at_horizon(times, mid, t0, 100.0, 10.001)[0] is False


# --------------------------------------------------------------------------------
# T2 / T3 -- which row gets selected
# --------------------------------------------------------------------------------

def test_T2_selects_message_landing_exactly_on_the_target():
    times, mid = _seq([0, 5, 10, 15], [100.0, 101.0, 102.0, 103.0])
    covered, price = price_at_horizon(times, mid, times[0], 100.0, 10.0)
    assert covered is True
    # side='left' would return 101.0 (the t=5 row), one message short.
    assert price == 102.0


def test_T3_does_not_read_from_the_future():
    times, mid = _seq([0, 3, 12], [100.0, 101.0, 999.0])
    covered, price = price_at_horizon(times, mid, times[0], 100.0, 10.0)
    assert covered is True
    # Dropping the -1 would return 999.0, which happens 2 ms after the horizon ends.
    assert price == 101.0


def test_T2b_shared_timestamps_take_the_last_one():
    # Real data routinely has several messages inside the same microsecond.
    times, mid = _seq([0, 10, 10, 10, 20], [100.0, 101.0, 102.0, 103.0, 104.0])
    _, price = price_at_horizon(times, mid, times[0], 100.0, 10.0)
    assert price == 103.0


# --------------------------------------------------------------------------------
# T4 -- an untouched book is data, not a coverage failure
# --------------------------------------------------------------------------------

def test_T4_no_message_before_target_returns_p0_and_counts_as_covered():
    times, mid = _seq([5, 20], [101.0, 102.0])
    covered, price = price_at_horizon(times, mid, t0=100.0, p0=100.0, horizon_ms=1.0)
    assert covered is True          # the sequence does extend past t0 + 1 ms
    assert price == 100.0           # but nothing traded yet, so the mid is still p0


# --------------------------------------------------------------------------------
# T5 -- log return arithmetic
# --------------------------------------------------------------------------------

def test_T5_log_return_value():
    gen_t, gen_m = _seq([10], [101.0])
    real_t, real_m = _seq([10], [99.0])
    out = sample_returns([10.0], t0=100.0, p0=100.0,
                         gen_times=gen_t, gen_mid=gen_m,
                         real_times=real_t, real_mid=real_m)
    pred, real = out[10.0]
    assert pred == pytest.approx(0.0099503308531681, rel=1e-12)
    assert real == pytest.approx(np.log(0.99), rel=1e-12)


def test_T5b_horizon_absent_when_either_side_is_uncovered():
    gen_t, gen_m = _seq([1], [101.0])        # generated stream stops at 1 ms
    real_t, real_m = _seq([50], [99.0])      # real stream reaches 50 ms
    out = sample_returns([10.0], 100.0, 100.0, gen_t, gen_m, real_t, real_m)
    assert out == {}                          # paired stats must use the same samples

    # Non-positive p0 cannot produce a return either.
    assert sample_returns([10.0], 100.0, 0.0, real_t, real_m, real_t, real_m) == {}


# --------------------------------------------------------------------------------
# T6 / T7 -- metrics
# --------------------------------------------------------------------------------

def test_T6_ic_reaches_both_extremes():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_metrics(x, x)["IC"] == pytest.approx(1.0)
    assert compute_metrics(x, x)["ranked_IC"] == pytest.approx(1.0)
    assert compute_metrics(x[::-1], x)["IC"] == pytest.approx(-1.0)
    assert compute_metrics(x[::-1], x)["ranked_IC"] == pytest.approx(-1.0)


def test_T7_direction_accuracy_and_zero_filter():
    pred = np.array([1.0, -1.0, 1.0, -1.0])
    real = np.array([1.0, 1.0, -1.0, -1.0])
    m = compute_metrics(pred, real)
    assert m["direction_acc"] == pytest.approx(0.5)
    assert m["N_eff"] == 4
    assert m["N_total"] == 4

    # A zero realized return carries no direction and must not dilute the accuracy.
    pred2 = np.append(pred, 1.0)
    real2 = np.append(real, 0.0)
    m2 = compute_metrics(pred2, real2)
    assert m2["N_total"] == 5
    assert m2["N_eff"] == 4
    assert m2["direction_acc"] == pytest.approx(0.5)


def test_T7b_too_few_samples_gives_nan_not_an_exception():
    m = compute_metrics(np.array([1.0]), np.array([1.0]))
    assert np.isnan(m["IC"]) and m["N_total"] == 1


# --------------------------------------------------------------------------------
# T9 -- the two direction conventions must stay distinct
# --------------------------------------------------------------------------------

def test_T9_direction_conventions_do_not_collide():
    """*_proc.npy uses {0,1}; inference output csv uses {-1,+1}.

    Mixing them silently yields an empty selection rather than an error, which reads
    as "the data contains no buyer-initiated trades" instead of "wrong constant".
    """
    assert {PROC_RESTING_BID, PROC_RESTING_ASK} == {1, 0}
    assert {CSV_RESTING_BID, CSV_RESTING_ASK} == {1, -1}
    # The bid side happens to be 1 in both layers; the ask side is what differs, and
    # that is exactly the value a filter would get wrong.
    assert PROC_RESTING_ASK != CSV_RESTING_ASK


# --------------------------------------------------------------------------------
# Diagnostics behave sanely at their fixed points
# --------------------------------------------------------------------------------

def test_distribution_comparison_is_exact_under_identity():
    x = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.05])
    d = distribution_comparison(x, x)
    assert d["std_ratio"] == pytest.approx(1.0)
    assert d["ks_stat"] == pytest.approx(0.0)
    assert d["abs_median_gen_bp"] == pytest.approx(d["abs_median_real_bp"])


def test_shuffle_null_centres_on_zero():
    rng = np.random.default_rng(7)
    real = rng.normal(size=200)
    pred = real + rng.normal(scale=0.5, size=200)
    assert compute_metrics(pred, real)["IC"] > 0.7      # paired: strong signal
    null = shuffle_null(pred, real, n_draws=200, seed=1)
    assert abs(null["IC_mean"]) < 3 * null["IC_sd"]     # unpaired: none left
    assert null["IC_sd"] > 0
