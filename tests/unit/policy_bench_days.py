"""A six-row synthetic stock-day in the v3 layout, shared by the policy bench tests."""

from __future__ import annotations

import numpy as np

from post_training.policy_bench.book import ASK, BID
from post_training.policy_bench.daydata import build_add_index


def l10_body(ask, aq, bid, bq):
    body = np.zeros(40, np.int64)
    body[0::4], body[2::4] = 9_999_999_999, -9_999_999_999
    body[:4] = ask, aq, bid, bq
    return body


def _row(oid, evt, d, price, size, t=(34_200, 0)):
    r = np.zeros(14, np.int64)
    r[[0, 1, 2, 3, 5, 8, 9]] = oid, evt, d, price, size, *t
    r[10:] = -9999
    return r


STEM = "X_2026-01-05_34200000_57600000_message_10_proc"


def write_day(tmp_path, ns=(0,) * 6):
    """A six-row day. Snapshot 0 (before row 0): ask 10,100 x 5, bid 10,000 x 5.
    Row i is at 34,200 s + ns[i] nanoseconds (all at 34,200 s by default)."""
    rows = [(20, 1, 1, 9_900, 1), (21, 1, 0, 10_300, 1), (22, 1, 1, 9_800, 1),
            (10, 4, 0, 10_100, 2), (10, 4, 0, 10_100, 3), (23, 1, 0, 10_400, 1)]
    msgs = np.stack([_row(*r, t=(34_200, n)) for r, n in zip(rows, ns)])
    month = tmp_path / "shard" / "2026-01"
    month.mkdir(parents=True)
    np.save(month / f"{STEM}.npy", msgs)
    l10 = np.zeros((7, 43), np.int64)
    l10[0, 3:] = l10_body(10_100, 5, 10_000, 5)
    np.save(month / "X_orderbook_10_proc.npy", l10)
    fb = tmp_path / "fb"
    fb.mkdir()
    np.save(fb / f"{STEM}_queue.npy", np.zeros((6, 2), np.int32))
    np.savez(fb / f"{STEM}_book_full.npz", snap_row=np.array([0]), grid=np.array(100), n_msgs=np.array(6),
             lvl_ptr=np.array([0, 2]), lvl_side=np.array([ASK, BID], np.int8), lvl_price=np.array([10_100, 10_000]),
             ord_ptr=np.array([0, 1, 2]), ord_size=np.array([5, 5]), ord_id=np.array([10, 11]))
    (tmp_path / "adds" / "2026-01").mkdir(parents=True)
    np.save(tmp_path / "adds" / "2026-01" / f"{STEM}_adds.npy", build_add_index(msgs))
    return dict(month="2026-01", message=f"{STEM}.npy", orderbook="X_orderbook_10_proc.npy",
                fullbook_dir=str(fb), date="2026-01-05")
