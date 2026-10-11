"""Pins for D-I3 (faithful initialisation) and D-O2 (the full-depth criterion).

D-I3's fix is not "use a wider snapshot". It is that the replay must reproduce
the generator's initialisation *procedure*: reset on the wide book from before
the conditioning window, then replay the conditioning messages, so that orders
created during that window exist individually rather than being collapsed into
one synthetic order per price level. The test below is built so that the two
modes must disagree, because a test that passes in both would not be testing
the thing that changed.

Engine message layout is [type, side, qty, price, oid, tid, time_s, time_ns]
with side -1 for asks and +1 for bids, matching init_msgs_from_l2's
even-row-ask, odd-row-bid convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from post_training.heuristic_learning.fidelity import compare_paths, replay_stream


ASK, BID = -1, 1
NEW, DELETE = 1, 3


def _book(levels):
    """levels: [(ask_p, ask_q, bid_p, bid_q), ...]"""
    return np.asarray([v for lv in levels for v in lv], dtype=np.int64)


def _m(type_, side, qty, price, oid, t=0):
    return [type_, side, qty, price, oid, 0, 34200 + t, 0]


INIT = _book([(11000, 5, 10000, 7), (12000, 3, 9000, 2)])


# ------------------------------------------------------------------- D-I3 --
# A partial cancel, deliberately. Creating an order in the warmup and then
# deleting all of it leaves both modes at the same book, so such a fixture
# would pass without testing anything: the first draft of this file did exactly
# that and had to be rewritten.
WARMUP = np.asarray([_m(NEW, BID, 50, 9500, 777, 1)], dtype=np.int64)
PARTIAL = np.asarray([_m(2, BID, 20, 9500, 777, 2)], dtype=np.int64)


def _bid_ladder(row):
    return [(int(row[4 * i + 2]), int(row[4 * i + 3])) for i in range(len(row) // 4)]


def test_faithful_mode_can_cancel_an_order_the_warmup_created():
    """The whole point. An order created during the conditioning window is a
    real order with a real id in faithful mode, so a later cancel resolves by
    id exactly as it did for the generator."""
    l2, _ = replay_stream(PARTIAL, INIT, n_levels=3, warmup_msgs=WARMUP,
                          init_time=(34200, 0))
    ladder = _bid_ladder(l2[0])
    assert (9500, 30) in ladder, f"expected 9500@30 after a partial cancel, got {ladder}"
    assert (10000, 7) in ladder, "the untouched init level must survive"


def test_degraded_mode_never_had_the_order_at_all():
    """Same input, no warmup. The order was never created, so there is nothing
    to cancel and the price is simply absent."""
    l2, _ = replay_stream(PARTIAL, INIT, n_levels=3)
    assert 9500 not in [p for p, _ in _bid_ladder(l2[0])]


def test_the_two_modes_disagree_on_this_fixture():
    """If this ever passes trivially, the fixture has stopped exercising the
    difference and the other two tests mean nothing."""
    faithful, _ = replay_stream(PARTIAL, INIT, 3, warmup_msgs=WARMUP,
                                init_time=(34200, 0))
    degraded, _ = replay_stream(PARTIAL, INIT, 3)
    assert not np.array_equal(faithful, degraded)


def test_engine_capacity_follows_the_init_width_in_faithful_mode():
    """A 500-level init book creates 1000 synthetic orders. Left at the default
    capacity the engine would silently drop most of them, so book_depth and
    nOrders are derived from the input width rather than defaulted."""
    wide = _book([(11000 + 10 * i, 5, 10000 - 10 * i, 7) for i in range(60)])
    stream = np.asarray([_m(DELETE, BID, 7, 10000, -3, 1)], dtype=np.int64)
    l2, _ = replay_stream(stream, wide, n_levels=3, warmup_msgs=np.zeros((0, 8), dtype=np.int64),
                          init_time=(34200, 0))
    assert l2.shape[1] == 12, "output width must follow n_levels, not the init width"
    assert int(l2[0][2]) != 10000 or int(l2[0][3]) != 7, (
        "the cancel of init order -3 should have changed the top bid level"
    )


def test_output_width_is_independent_of_init_width():
    """Faithful mode initialises wide and compares narrow. Conflating the two
    is what the previous code did."""
    wide = _book([(11000 + 10 * i, 5, 10000 - 10 * i, 7) for i in range(20)])
    stream = np.asarray([_m(NEW, BID, 1, 9990, 900, 1)], dtype=np.int64)
    for out_levels in (1, 3, 10):
        l2, _ = replay_stream(stream, wide, n_levels=out_levels,
                              warmup_msgs=np.zeros((0, 8), dtype=np.int64),
                              init_time=(34200, 0))
        assert l2.shape[1] == 4 * out_levels


# ------------------------------------------------------------------- D-O2 --
def test_full_depth_criterion_is_strictly_stronger_than_mid():
    """Mid moves only when the touch moves, so a book can be wrong at every
    level below it and still score 1.0 on mid. Anything the full criterion
    calls equal, the mid criterion must too."""
    a = np.asarray([[11000, 5, 10000, 7, 12000, 3, 9000, 2]], dtype=np.int64)
    b = a.copy()
    b[0, 5] = 99                      # a depth-only difference, mid unchanged
    r = compare_paths(a, b)
    assert r["exact_mid_frac"] == 1.0, "mid should not notice a depth-only change"
    assert r["exact_full_frac"] == 0.0, "the full criterion must notice it"
    assert r["first_divergence_step"] == -1
    assert r["first_full_divergence_step"] == 0


def test_identical_paths_are_exact_on_both_criteria():
    a = np.asarray([[11000, 5, 10000, 7], [11000, 4, 10000, 7]], dtype=np.int64)
    r = compare_paths(a, a.copy())
    assert r["exact_mid_frac"] == 1.0 and r["exact_full_frac"] == 1.0
    assert r["first_full_divergence_step"] == -1


@pytest.mark.parametrize("shift", [1, 100])
def test_a_touch_change_is_caught_by_both(shift):
    a = np.asarray([[11000, 5, 10000, 7]], dtype=np.int64)
    b = a.copy()
    b[0, 0] += shift
    r = compare_paths(a, b)
    assert r["exact_mid_frac"] == 0.0 and r["exact_full_frac"] == 0.0


# ------------------------------------------------------------------- D-T1 --
def test_declared_theta_is_read_and_an_override_wins():
    """The replay used to construct JAXLOB_Configuration() with defaults and
    hope. Reading what the run declared makes the dependency explicit; an
    explicit argument still wins, because that is an experiment and the
    manifest is a record of what ran."""
    import json
    import tempfile
    from pathlib import Path

    from post_training.heuristic_learning.fidelity import load_theta

    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "generated" / "background_seed1"
        run.mkdir(parents=True)
        assert load_theta(run) == {}, "no manifest must not raise, and must not guess"

        (Path(d) / "manifest.json").write_text(json.dumps(
            {"engine": {"tick_size": 25, "init_id": -2, "irrelevant": 1}}))
        assert load_theta(run) == {"tick_size": 25, "init_id": -2}, \
            "only declared engine keys, and only the ones a replay must match"


def test_a_corrupt_manifest_does_not_take_the_replay_down():
    """A record that cannot be parsed is a reason to fall back to defaults and
    say so elsewhere, not a reason for every fidelity run to crash."""
    import tempfile
    from pathlib import Path

    from post_training.heuristic_learning.fidelity import load_theta

    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "generated" / "background_seed1"
        run.mkdir(parents=True)
        (Path(d) / "manifest.json").write_text("{not json")
        assert load_theta(run) == {}
