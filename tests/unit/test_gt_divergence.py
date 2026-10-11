"""Sim-vs-real book divergence, ported from LOBelia.

These assertions pin the L2 column layout that `compute_gt_divergence` assumes:
(ask_px, ask_vol, bid_px, bid_vol) repeated per level. That layout is shared
with return_bench's `mid = (book[:, 0] + book[:, 2]) / 2`, so if it ever drifts
both the divergence metric and every mid-price series silently change meaning.
"""

from __future__ import annotations

import numpy as np

from base_model.inference.generate import compute_gt_divergence


TICK_SIZE = 100


def _two_level_books():
    """One step, two levels: [ask_px, ask_vol, bid_px, bid_vol] x 2."""
    gen = np.array(
        [[10100, 5, 10000, 7, 10200, 3, 9900, 4]],
        dtype=np.float32,
    )
    real = np.array(
        [[10200, 2, 10000, 7, 10300, 3, 9900, 1]],
        dtype=np.float32,
    )
    return gen, real


def test_mid_divergence_is_measured_in_ticks() -> None:
    gen, real = _two_level_books()
    out = compute_gt_divergence(gen, real, TICK_SIZE)
    # gen mid = (10100 + 10000) / 2 = 10050
    # real mid = (10200 + 10000) / 2 = 10100
    # |10050 - 10100| / 100 tick = 0.5 ticks
    assert float(out["mid_divergence"][0]) == 0.5


def test_spread_divergence_is_measured_in_ticks() -> None:
    gen, real = _two_level_books()
    out = compute_gt_divergence(gen, real, TICK_SIZE)
    # gen spread = 100, real spread = 200 -> 1 tick apart
    assert float(out["spread_divergence"][0]) == 1.0


def test_volume_l1_sums_each_side_separately() -> None:
    gen, real = _two_level_books()
    out = compute_gt_divergence(gen, real, TICK_SIZE)
    # ask volumes: |5-2| + |3-3| = 3
    # bid volumes: |7-7| + |4-1| = 3
    assert float(out["vol_l1_ask"][0]) == 3.0
    assert float(out["vol_l1_bid"][0]) == 3.0


def test_identical_books_have_zero_divergence() -> None:
    gen, _ = _two_level_books()
    out = compute_gt_divergence(gen, gen, TICK_SIZE)
    for key in ("mid_divergence", "spread_divergence", "vol_l1_ask", "vol_l1_bid"):
        assert float(out[key][0]) == 0.0


def test_divergence_is_per_step_not_aggregated() -> None:
    gen, real = _two_level_books()
    gen3 = np.repeat(gen, 3, axis=0)
    real3 = np.repeat(real, 3, axis=0)
    out = compute_gt_divergence(gen3, real3, TICK_SIZE)
    for key in ("mid_divergence", "spread_divergence", "vol_l1_ask", "vol_l1_bid"):
        assert out[key].shape == (3,)
