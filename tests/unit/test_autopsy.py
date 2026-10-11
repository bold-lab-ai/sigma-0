"""Pins for the first-divergence autopsy.

The autopsy is what moved the gen arm's onset from "class R or class O" to a
named defect with a measurement, so its two design decisions are load-bearing
and are tested rather than trusted:

  - books are compared as price -> quantity maps, not positional arrays. A
    positional comparison reports a wall of differences for a single level
    removal, because everything below shifts up.
  - the deepest level is excluded, because activity below the snapshot pushes
    new levels into view at the bottom edge and that is truncation, not
    disagreement.

Get either wrong and every attribution downstream is wrong in the same
direction.
"""

from __future__ import annotations

import numpy as np

from post_training.heuristic_learning.autopsy import book_map, delta, diagnose
from post_training.heuristic_learning.fidelity import LOOKUP_MISS


# [ask_p, ask_q, bid_p, bid_q] per level
def _book(levels):
    return np.asarray([v for lv in levels for v in lv], dtype=np.int64)


def _msg(type_=3, side=1, qty=10, price=100, oid=555):
    return np.asarray([type_, side, qty, price, oid, 0, 34200, 0], dtype=np.int64)


# ------------------------------------------------------------- price maps --
def test_book_map_keys_on_price_not_position():
    b = _book([(110, 5, 100, 7), (120, 3, 90, 2), (130, 1, 80, 1)])
    m = book_map(b, drop_deepest=1)          # deepest level dropped
    assert m == {("ask", 110): 5, ("bid", 100): 7, ("ask", 120): 3, ("bid", 90): 2}


def test_removing_a_level_is_a_two_entry_change_not_a_cascade():
    """The point of keying on price, and the honest limit of it.

    Positionally, deleting the top ask changes every cell in the book. By
    price it changes two: the level that went (real) and one edge entry
    arriving from below (artefact). Dropping the deepest level cannot remove
    that second one -- the kept set slides from {p0..p_{L-2}} to
    {p1..p_{L-1}} -- which is why a single autopsy is evidence and not proof.
    """
    before = book_map(_book([(110, 5, 100, 7), (120, 3, 90, 2), (130, 1, 80, 1)]), 1)
    after = book_map(_book([(120, 3, 100, 7), (130, 1, 90, 2), (140, 9, 80, 1)]), 1)
    d = delta(before, after)
    assert d == {("ask", 110): -5, ("ask", 130): 1}, d
    real = {k: v for k, v in d.items() if k[1] in dict.fromkeys(p for _, p in before)}
    assert real == {("ask", 110): -5}, "the genuine change is still a single entry"


def test_the_deepest_level_is_excluded():
    """Activity below the snapshot pushes a level into view at the bottom.
    Counting that edge as a disagreement manufactures divergences."""
    a = _book([(110, 5, 100, 7), (120, 3, 90, 2)])
    b = _book([(110, 5, 100, 7), (999, 1, 1, 1)])
    assert book_map(a, drop_deepest=1) == book_map(b, drop_deepest=1)
    assert book_map(a, drop_deepest=0) != book_map(b, drop_deepest=0)


def test_empty_level_sentinels_are_not_treated_as_prices():
    """get_L2_state fills empty asks with maxint and empty bids with -1."""
    m = book_map(_book([(110, 5, 100, 7), (2 ** 31 - 1, 0, -1, 0), (1, 1, 1, 1)]), 1)
    assert m == {("ask", 110): 5, ("bid", 100): 7}


# ------------------------------------------------------------- signatures --
def test_recorded_but_not_applied_is_distinguished_from_the_reverse():
    empty: dict = {}
    moved = {("bid", 100): -13}
    assert "RECORDED_NOT_APPLIED" in diagnose(_msg(), empty, moved, 10)
    assert "APPLIED_NOT_RECORDED" in diagnose(_msg(), moved, empty, 10)


def test_different_price_needs_disjoint_prices():
    """The measured gen-arm cause: the recorded price is not where the
    generator removed quantity."""
    sigs = diagnose(_msg(), {("ask", 9286100): -10}, {("ask", 9286200): -10}, 10)
    assert "DIFFERENT_PRICE" in sigs
    assert "DIFFERENT_AMOUNT" not in sigs


def test_same_price_different_amount_is_its_own_signature():
    sigs = diagnose(_msg(), {("ask", 100): -10}, {("ask", 100): -4}, 10)
    assert "DIFFERENT_AMOUNT" in sigs
    assert "DIFFERENT_PRICE" not in sigs


def test_identical_deltas_produce_no_book_signature():
    same = {("ask", 100): -10}
    sigs = diagnose(_msg(type_=3), same, dict(same), 10)
    assert sigs == ["UNCLASSIFIED"], sigs


def test_message_level_signatures_are_independent_of_the_books():
    same = {("ask", 100): -10}
    assert "AGGRESSIVE_EXECUTION" in diagnose(_msg(type_=4), same, dict(same), 10)
    assert "MISS_SENTINEL" in diagnose(_msg(oid=LOOKUP_MISS), same, dict(same), 10)
    assert "OUT_OF_DEPTH" in diagnose(_msg(oid=-86), same, dict(same), 10)
    assert "OUT_OF_DEPTH" not in diagnose(_msg(oid=-8), same, dict(same), 10)


def test_the_sentinel_is_not_also_reported_as_out_of_depth():
    """-99 is below a 10-level init span, but calling it OUT_OF_DEPTH would
    attribute a D-R2 instance to D-I1."""
    same = {("ask", 100): -10}
    sigs = diagnose(_msg(oid=LOOKUP_MISS), same, dict(same), 10)
    assert "MISS_SENTINEL" in sigs
    assert "OUT_OF_DEPTH" not in sigs
