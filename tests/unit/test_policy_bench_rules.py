"""Legality of background messages, and the guarantee that protects agent orders.

The rule that matters most is the asymmetric one: agent orders count in queue
numbering and get filled by background executions, but a background cancel or
delete can never name one. If that broke, the generator could remove our
quotes, and every passive-fill number would be wrong without any error.
"""

from __future__ import annotations

import pytest

from lobpipeline.policy_bench import rules
from lobpipeline.policy_bench.book import AGENT, ASK, BG, BID, Book
from lobpipeline.policy_bench.codec import SLOT_NA, Message

BUY, SELL = 1, 0


def _msg(evt, d, levels, qslot=-1):
    return Message(evt, d, 0, 1, 34_200, 0, tuple(levels), qslot)


def _book():
    bk = Book()
    bk.add(BID, 10_000, 1, 200)
    bk.add(BID, 10_000, 2, 100)
    bk.add(BID, 10_000, 3, 100, AGENT)
    bk.add(BID, 10_000, 4, 300)
    bk.add(BID, 9_900, 5, 50)
    bk.add(ASK, 10_100, 6, 70)
    bk.add(ASK, 10_200, 7, 30)
    return bk


@pytest.mark.parametrize("d,price,reason", [
    (BUY, 9_950, None),
    (BUY, 10_100, "new_crosses"),
    (SELL, 10_050, None),
    (SELL, 10_000, "new_crosses"),
])
def test_new_orders_rest_outside_the_spread(d, price, reason):
    assert rules.check(_msg(1, d, [(price, 10)]), _book()) == reason


def test_new_order_of_size_zero_is_illegal():
    assert rules.check(_msg(1, BUY, [(9_000, 0)]), _book()) == "new_size_zero"


@pytest.mark.parametrize("evt,size,slot,reason", [
    (3, 100, 1, None),
    (3, 100, 2, "slot_agent"),            # same size as order 2, but slot 2 is ours
    (3, 99, 1, "delete_size"),
    (2, 99, 1, None),
    (2, 100, 1, "cancel_size"),           # a cancel of everything is a delete
    (3, 300, 3, None),                    # the agent order still takes up slot 2
    (3, 300, 4, "slot_out_of_range"),
    (3, 300, SLOT_NA, "slot_na"),
])
def test_cancels_and_deletes_name_one_background_order(evt, size, slot, reason):
    assert rules.check(_msg(evt, BUY, [(10_000, size)], slot), _book()) == reason


def test_cancel_at_an_empty_price_is_illegal():
    assert rules.check(_msg(3, BUY, [(9_800, 1)], 0), _book()) == "no_level"


def _deep_book(agent_at=()):
    bk = Book()
    for s in range(105):
        bk.add(BID, 10_000, s, 10, AGENT if s in agent_at else BG)
    return bk


def test_qslot_99_means_any_qualifying_order_from_slot_99_on():
    m = _msg(3, BUY, [(10_000, 10)], 99)
    assert rules.slot_candidates(m, _deep_book()) == list(range(99, 105))
    assert rules.slot_candidates(m, _deep_book(agent_at={100, 102})) == [99, 101, 103, 104]


def test_qslot_99_is_illegal_when_only_agent_orders_are_left_there():
    m = _msg(3, BUY, [(10_000, 10)], 99)
    assert rules.check(m, _deep_book(agent_at=set(range(99, 105)))) == "no_candidate"


@pytest.mark.parametrize("levels,reason", [
    ([(10_000, 250)], None),
    ([(10_000, 700), (9_900, 10)], None),
    ([(10_000, 700), (9_900, 50)], None),
    ([(9_900, 10)], "exec_not_best"),
    ([(10_000, 699), (9_900, 10)], "exec_partial_inner"),
    ([(10_000, 700), (9_800, 10)], "exec_skips_level"),
    ([(10_000, 701)], "exec_over_volume"),
    ([(10_000, 0)], "exec_size_zero"),
])
def test_executions_sweep_from_the_best_price(levels, reason):
    assert rules.check(_msg(4, BUY, levels), _book()) == reason


def test_background_execution_fills_the_agent_in_queue_order():
    bk = _book()
    fills = rules.apply(_msg(4, BUY, [(10_000, 350)]), bk, oid=-1)
    assert [(o.oid, q) for o, q in fills] == [(1, 200), (2, 100), (3, 50)]
    assert bk.queue(BID, 10_000)[0].owner == AGENT


def test_apply_resolves_qslot_99_with_the_given_slot():
    bk = _deep_book()
    rules.apply(_msg(3, BUY, [(10_000, 10)], 99), bk, oid=-1, slot=103)
    assert not bk.has(103) and bk.has(99)


def test_every_reason_is_declared():
    assert len(set(rules.REASONS)) == len(rules.REASONS)


def test_time_consistency():
    m = Message(1, BUY, 1, 500, 34_201, 700, ((9_000, 1),))
    assert rules.time_consistent(m, 34_200 * 10**9 + 200)
    assert not rules.time_consistent(m, 34_200 * 10**9 + 201)
