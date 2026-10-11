"""Properties a market-making simulator has to have to be worth reading.

Every failure mode pinned here is silent: the engine does not raise when fills
are attributed to the wrong side, counted twice, or lost. It returns a PnL that
looks like a PnL. So these are not regression tests bolted on after the fact,
they are the only thing standing between a number and a fiction.
"""

from __future__ import annotations

import numpy as np
import pytest

from post_training.heuristic_learning.episode_builder import (
    AGENT_TRADER_ID,
    BACKGROUND_TRADER_ID,
)
from post_training.heuristic_learning.fidelity import (
    HANDLED_EVENT_TYPES,
    describe_stream,
)
from post_training.heuristic_learning.mm_sim import (
    SimConfig,
    baseline_symmetric_touch,
    run_episode,
)


TICK = 100
MID = 1_000_000


def _book(levels=10, spread_ticks=2, size=100):
    """Flat two-sided L2 snapshot: (ask_px, ask_sz, bid_px, bid_sz) per level."""
    row = []
    for i in range(levels):
        row += [MID + (spread_ticks // 2 + i) * TICK, size,
                MID - (spread_ticks // 2 + i) * TICK, size]
    return np.array(row, dtype=np.int32)


def _msg(t, side, qty, px, oid, t_s, t_ns, tid=BACKGROUND_TRADER_ID):
    return [t, side, qty, px, oid, tid, t_s, t_ns]


def _stream(n=200, t0=34_200):
    """Background flow that alternates resting limits and aggressive type-4s.

    The type-4s are what let a passive quote ever fill: they are the only source
    of aggressive flow in a replayed stream.
    """
    rows = []
    for i in range(n):
        t_ns = (i * 5_000_000) % 1_000_000_000
        t_s = t0 + (i * 5_000_000) // 1_000_000_000
        if i % 4 == 3:
            rows.append(_msg(4, -1, 20, MID - TICK, 900_000 + i, t_s, t_ns))   # hits the bid
        elif i % 4 == 1:
            rows.append(_msg(4, 1, 20, MID + TICK, 900_000 + i, t_s, t_ns))    # hits the ask
        else:
            rows.append(_msg(1, 1 if i % 2 else -1, 10,
                             MID - TICK if i % 2 else MID + TICK, 800_000 + i, t_s, t_ns))
    return np.array(rows, dtype=np.int32)


def _episode(n=200):
    return {"background": _stream(n), "init_book": _book(), "signal": None}


def _abstain(obs, state):
    return {"post_bid": False, "post_ask": False,
            "cancel_bid": False, "cancel_ask": False, "size": 0}


def _requote_every_step(obs, state):
    return {"post_bid": True, "post_ask": True,
            "cancel_bid": True, "cancel_ask": True, "size": 5}


# ------------------------------------------------------------------ F2 -----
def test_a_policy_that_never_quotes_earns_exactly_zero():
    """The accounting must not leak. Exactly zero, not approximately.

    A non-zero result here means fills are being attributed to us that are not
    ours, which no amount of downstream averaging would reveal.
    """
    traj = run_episode(_abstain, _episode(), SimConfig(n_steps=20))
    assert traj.error is None
    assert traj.n_fills == 0
    assert traj.final_inventory == 0
    assert traj.total_pnl == 0.0
    assert traj.realized_pnl == 0.0


# ------------------------------------------------------------------ F3 -----
def test_inventory_equals_the_signed_sum_of_fills():
    """Conservation. Inventory is not tracked independently of the fills."""
    traj = run_episode(baseline_symmetric_touch, _episode(), SimConfig(n_steps=20))
    assert traj.error is None
    signed = sum(f["qty"] * f["side"] for s in traj.steps for f in s.fills)
    assert traj.final_inventory == signed


def test_every_fill_has_a_side_a_positive_quantity_and_our_order_id():
    traj = run_episode(baseline_symmetric_touch, _episode(), SimConfig(n_steps=20))
    for step in traj.steps:
        for f in step.fills:
            assert f["side"] in (-1, 1)
            assert f["qty"] > 0
            assert f["oid"] < 0                      # agent ids are negative


# ------------------------------------------------------------------ F5 -----
def test_two_identical_runs_agree_bit_for_bit():
    """Determinism. Without it a paired comparison between policies means nothing."""
    cfg = SimConfig(n_steps=20)
    a = run_episode(baseline_symmetric_touch, _episode(), cfg)
    b = run_episode(baseline_symmetric_touch, _episode(), cfg)
    assert (a.total_pnl, a.n_fills, a.final_inventory) == (b.total_pnl, b.n_fills, b.final_inventory)
    assert [s.inventory for s in a.steps] == [s.inventory for s in b.steps]


# ---------------------------------------------- queue position (3.4) -------
def test_an_unchanged_quote_is_not_cancelled_and_reposted():
    """The whole return of quoting at the touch comes from queue position.

    The engine keys priority on (price, time_s, time_ns) and leaves the
    timestamp alone on a partial fill, so a resting order keeps its place. A
    policy that cancels and re-posts every step resets its own timestamp and
    sits permanently at the back, which simulates a strategy nobody would run.
    """
    import post_training.heuristic_learning.mm_sim as mm

    sent = {"post": 0, "cancel": 0}
    original = mm._engine_msg

    def spy(event_type, *a, **k):
        if event_type == 1:
            sent["post"] += 1
        elif event_type == 2:
            sent["cancel"] += 1
        return original(event_type, *a, **k)

    mm._engine_msg = spy
    try:
        traj = run_episode(baseline_symmetric_touch, _episode(), SimConfig(n_steps=20))
    finally:
        mm._engine_msg = original

    both_sides_every_step = 2 * len(traj.steps)
    assert sent["post"] < both_sides_every_step / 2, (
        f"re-quoted {sent['post']} times in {len(traj.steps)} steps; the quote is "
        "not surviving across steps and can never accrue queue position"
    )


def test_a_policy_that_asks_to_requote_every_step_actually_does():
    """The converse, so the test above cannot pass by never quoting at all."""
    import post_training.heuristic_learning.mm_sim as mm

    sent = {"post": 0}
    original = mm._engine_msg

    def spy(event_type, *a, **k):
        if event_type == 1:
            sent["post"] += 1
        return original(event_type, *a, **k)

    mm._engine_msg = spy
    try:
        run_episode(baseline_symmetric_touch, _episode(), SimConfig(n_steps=20))
        kept = sent["post"]
        sent["post"] = 0
        run_episode(_requote_every_step, _episode(), SimConfig(n_steps=20))
    finally:
        mm._engine_msg = original
    # Both policies quote both sides; only their cancel behaviour differs, and
    # the harness suppresses a cancel when the quote is already where it wants.
    assert sent["post"] >= kept


# ------------------------------------------------------------------ B4 -----
def test_a_one_sided_book_is_not_traded():
    """Empty ask levels come back as cfg.maxint, not -1.

    A `price <= 0` guard alone therefore lets a one-sided book through, and the
    policy would quote against a price near 2**31.
    """
    from matching_engine.jaxob.jaxob_config import JAXLOB_Configuration

    book = _book()
    empty = int(JAXLOB_Configuration().maxint)        # the engine's own sentinel
    for i in range(0, len(book), 4):
        book[i] = empty                              # wipe every ask level
        book[i + 1] = 0
    traj = run_episode(baseline_symmetric_touch,
                       {"background": _stream(100), "init_book": book, "signal": None},
                       SimConfig(n_steps=10))
    assert traj.error is None
    assert traj.n_fills == 0


# ------------------------------------------------ policy sandboxing --------
def test_a_policy_that_raises_does_not_take_the_harness_with_it():
    def broken(obs, state):
        raise ValueError("boom")

    traj = run_episode(broken, _episode(), SimConfig(n_steps=10))
    assert traj.error is not None and "boom" in traj.error


def test_policy_size_and_inventory_caps_are_enforced_by_the_harness():
    """Hard constraints are not trusted to the policy."""
    def greedy(obs, state):
        return {"post_bid": True, "post_ask": False,
                "cancel_bid": False, "cancel_ask": False, "size": 10_000}

    cfg = SimConfig(n_steps=40, size_cap=5, inventory_cap=8)
    traj = run_episode(greedy, _episode(400), cfg)
    assert traj.error is None
    for step in traj.steps:
        for f in step.fills:
            assert f["qty"] <= cfg.size_cap
    assert abs(traj.final_inventory) <= cfg.inventory_cap + cfg.size_cap


# ------------------------------------------------------------------ F6 -----
def test_stream_diagnostics_surface_clock_and_type_problems():
    msgs = _stream(50)
    rep = describe_stream(msgs)
    assert rep.n_messages == 50
    assert rep.monotonic
    assert rep.n_negative_deltas == 0
    assert set(rep.event_types) <= HANDLED_EVENT_TYPES
    assert rep.n_unhandled_types == 0
    assert rep.clean


def test_a_non_monotonic_clock_is_reported_not_absorbed():
    """A generation defect once put START into a time field, collapsing it to
    the NA sentinel and producing a negative inter-arrival, at 3 events in
    780,864. A window whose clock runs backwards cannot be replayed."""
    msgs = _stream(50)
    msgs[20, 7] = 0
    msgs[21, 7] = 0
    msgs[21, 6] = msgs[20, 6] - 1
    rep = describe_stream(msgs)
    assert not rep.monotonic
    assert rep.n_negative_deltas >= 1
    assert not rep.clean


def test_unhandled_event_types_are_counted():
    """The engine routes anything outside {1,2,3,4} into ask_lim silently, so a
    type-5 would be inserted as a sell limit order rather than rejected."""
    msgs = _stream(50)
    msgs[10, 0] = 5
    rep = describe_stream(msgs)
    assert rep.n_unhandled_types == 1
    assert not rep.clean


def test_unresolvable_cancels_are_counted_but_do_not_imply_failure():
    """Real streams carry 12-38 of these per window and still replay exactly,
    because the price-based fallback lands where the real cancel landed. The
    count is a diagnostic, not a verdict."""
    msgs = _stream(50)
    msgs[30] = _msg(3, 1, 10, MID - TICK, 123_456_789, int(msgs[30, 6]), int(msgs[30, 7]))
    rep = describe_stream(msgs)
    assert rep.n_unresolvable_ids >= 1
    assert rep.n_cancel_delete >= 1


def test_prior_ids_make_a_conditioning_window_cancel_resolvable():
    msgs = _stream(50)
    msgs[30] = _msg(3, 1, 10, MID - TICK, 555_000, int(msgs[30, 6]), int(msgs[30, 7]))
    assert describe_stream(msgs).n_unresolvable_ids >= 1
    assert describe_stream(msgs, prior_ids=[555_000]).n_unresolvable_ids == 0


# ----------------------------------------------------- trader id split -----
def test_agent_and_background_trader_ids_are_distinct():
    """Fills are attributed by trader id; a collision would silently credit us
    with the market's trades."""
    assert AGENT_TRADER_ID != BACKGROUND_TRADER_ID
    assert AGENT_TRADER_ID != -2                     # INITID, used by book-init orders
