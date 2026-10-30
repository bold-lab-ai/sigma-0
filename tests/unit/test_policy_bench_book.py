"""The policy bench's L3 book and the real-data replay that builds it.

The book is the state everything else reads, so its failures are the silent
kind: a fill attributed to the wrong order or a level volume that drifts still
produces a book that looks like a book. These tests pin the FIFO and slot
semantics on hand-built cases; G1 (run/post_training/policy_bench/) checks the
same code against the real queue files.
"""

from __future__ import annotations

import numpy as np
import pytest

from post_training.policy_bench.book import (
    AGENT, ASK, BG, BID, EMPTY_ASK_PRICE, EMPTY_BID_PRICE, SLOT_NEW, SLOT_UNKNOWN, Book,
)
from post_training.policy_bench.daydata import DayFiles, Snapshots


def _row(oid, evt, d, price, size):
    r = np.zeros(14, np.int64)
    r[[0, 1, 2, 3, 5]] = oid, evt, d, price, size
    return r


def _book():
    bk = Book()
    bk.add(BID, 10_000, 1, 200)
    bk.add(BID, 10_000, 2, 100)
    bk.add(BID, 10_000, 3, 100, AGENT)
    bk.add(BID, 10_000, 4, 300)
    bk.add(BID, 9_900, 5, 50)
    bk.add(ASK, 10_100, 6, 70)
    return bk


def test_execute_fills_front_of_queue_first():
    bk = _book()
    fills = bk.execute(BID, 10_000, 250)
    assert [(o.oid, q) for o, q in fills] == [(1, 200), (2, 50)]
    assert bk.volume(BID, 10_000) == 450
    assert bk.slot_of(3) == 1


def test_execute_reaches_agent_only_after_the_orders_ahead():
    bk = _book()
    fills = bk.execute(BID, 10_000, 400)
    assert [(o.oid, q, o.owner) for o, q in fills] == [(1, 200, BG), (2, 100, BG), (3, 100, AGENT)]
    assert bk.slot_of(4) == 0


def test_execute_refuses_more_than_the_level_holds():
    with pytest.raises(ValueError):
        _book().execute(ASK, 10_100, 71)


def test_level_disappears_when_its_last_order_leaves():
    bk = _book()
    bk.remove_at(ASK, 10_100, 0)
    assert bk.best(ASK) is None
    assert not bk.has(6)
    assert bk.l10(2)[[0, 1]].tolist() == [EMPTY_ASK_PRICE, 0]


def test_reduce_keeps_the_order_until_it_is_empty():
    bk = _book()
    bk.reduce_at(BID, 9_900, 0, 20)
    assert bk.queue(BID, 9_900)[0].size == 30
    bk.reduce_at(BID, 9_900, 0, 30)
    assert bk.prices(BID) == [10_000]


def test_l10_layout_and_best_first_order():
    bk = _book()
    bk.add(ASK, 10_300, 7, 5)
    bk.add(ASK, 10_200, 8, 9)
    out = bk.l10(3)
    assert out.tolist() == [10_100, 70, 10_000, 700,
                            10_200, 9, 9_900, 50,
                            10_300, 5, EMPTY_BID_PRICE, 0]


def test_apply_row_reports_the_slot_before_applying():
    bk = _book()
    assert bk.apply_row(_row(9, 1, 1, 10_000, 10)) == SLOT_NEW
    assert bk.slot_of(9) == 4
    assert bk.apply_row(_row(2, 3, 1, 10_000, 100)) == 1
    assert bk.apply_row(_row(4, 4, 1, 10_000, 100)) == 2
    assert bk.queue(BID, 10_000)[2].size == 200
    assert bk.apply_row(_row(99, 3, 1, 10_000, 1)) == SLOT_UNKNOWN
    assert bk.apply_row(_row(5, 3, 1, 10_000, 50)) == SLOT_UNKNOWN   # rests at 9_900, not here


def test_cross_drops_both_sides_through_the_price():
    bk = _book()
    bk.cross(9_950)
    assert bk.prices(BID) == [9_900]
    assert not bk.has(1) and not bk.has(3)
    assert bk.prices(ASK) == [10_100]


def _write_day(tmp_path):
    """A four-row day: snapshots before rows 0 and 2, one extra before row 2."""
    msgs = np.stack([
        _row(10, 1, 1, 10_000, 100),
        _row(11, 1, 0, 10_100, 9999),            # true size 20000
        _row(10, 4, 1, 10_000, 60),
        _row(11, 2, 0, 10_100, 9999),            # true size 15000
    ])
    month = tmp_path / "shard" / "2026-01"
    month.mkdir(parents=True)
    np.save(month / "X_message_10_proc.npy", msgs)
    np.save(month / "X_orderbook_10_proc.npy", np.zeros((5, 43), np.int64))
    fb = tmp_path / "fb"
    fb.mkdir()
    np.save(fb / "X_message_10_proc_queue.npy", np.zeros((4, 2), np.int32))
    # snapshot 0: one ask at 10_200; snapshot 1 (before row 2): after rows 0, 1 and nothing else
    np.savez(fb / "X_message_10_proc_book_full.npz",
             snap_row=np.array([0, 2]), grid=np.array(2), n_msgs=np.array(4),
             lvl_ptr=np.array([0, 1, 4]), lvl_side=np.array([ASK, ASK, ASK, BID], np.int8),
             lvl_price=np.array([10_200, 10_100, 10_200, 10_000]),
             ord_ptr=np.array([0, 1, 2, 3, 4]), ord_size=np.array([5, 20_000, 5, 100]),
             ord_id=np.array([1, 11, 1, 10]))
    np.savez(fb / "X_message_10_proc_extra.npz", before_row=np.array([3]), order_id=np.array([12]),
             event_type=np.array([1]), direction=np.array([1]), price_abs=np.array([9_900]),
             size=np.array([7]), queue_slot=np.array([-1]))
    np.savez(fb / "X_message_10_proc_bigsize.npz", row=np.array([3, 1]), size=np.array([15_000, 20_000]))
    entry = dict(month="2026-01", message="X_message_10_proc.npy", orderbook="X_orderbook_10_proc.npy",
                 fullbook_dir=str(fb))
    return DayFiles.open(str(tmp_path / "shard"), entry)


def test_replay_undoes_the_size_clip_and_inserts_extras(tmp_path):
    day = _write_day(tmp_path)
    bk = day.book_before(4)
    assert bk.levels(ASK) == {10_100: [(11, 5_000, BG)], 10_200: [(1, 5, BG)]}
    assert bk.levels(BID) == {10_000: [(10, 40, BG)], 9_900: [(12, 7, BG)]}


def test_replay_from_snapshot_matches_the_next_snapshot(tmp_path):
    day = _write_day(tmp_path)
    bk = Book.from_snapshot(day.snapshots, 0)
    slots = [s for _, s in day.replay(bk, 0, 2)]
    assert slots == [SLOT_NEW, SLOT_NEW]
    ref = Book.from_snapshot(day.snapshots, 1)
    assert bk.levels(ASK) == ref.levels(ASK) and bk.levels(BID) == ref.levels(BID)


def test_book_before_an_extra_row_includes_that_extra(tmp_path):
    day = _write_day(tmp_path)
    assert 9_900 in day.book_before(3).prices(BID)
    assert 9_900 not in day.book_before(2).prices(BID)


@pytest.mark.parametrize("save", [np.savez, np.savez_compressed])
def test_one_snapshot_equals_the_whole_archive(tmp_path, save):
    z = dict(snap_row=np.array([0, 2, 5]), grid=np.array(2), lvl_ptr=np.array([0, 1, 4, 6]),
             lvl_side=np.array([ASK, ASK, ASK, BID, ASK, BID], np.int8),
             lvl_price=np.array([10_200, 10_100, 10_200, 10_000, 10_300, 9_800]),
             ord_ptr=np.array([0, 1, 2, 4, 5, 6, 8]), ord_size=np.array([5, 7, 5, 3, 100, 9, 1, 2]),
             ord_id=np.array([1, 11, 1, 13, 10, 20, 21, 22]))
    save(tmp_path / "b.npz", **z)
    snaps = Snapshots(str(tmp_path / "b.npz"))
    for s in range(3):
        a, b = Book.from_snapshot(z, s), Book.from_snapshot(snaps.one(s), 0)
        assert a.levels(ASK) == b.levels(ASK) and a.levels(BID) == b.levels(BID)
    assert snaps["ord_id"].tolist() == z["ord_id"].tolist() and int(snaps["grid"]) == 2
