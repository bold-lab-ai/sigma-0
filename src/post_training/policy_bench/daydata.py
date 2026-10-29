"""One stock-day of the v3 dataset, and the real-data replay that rebuilds its book.

Files per stock-day. The manifest entry names the first two inside a monthly
squashfs shard mounted at `<shard_root>/<month>/`, and `fullbook_dir` holds the
rest:

    entry["message"]       processed messages, (n, 14) int64; size clipped at 9999
    entry["orderbook"]     processed L10 rows, (n + 1, 43); row R = book BEFORE row R
    <stem>_queue.npy       (n, 2) int32: [queue_slot, queue_len] BEFORE row R
    <stem>_book_full.npz   full-book snapshots BEFORE rows 0, grid, 2 grid, ...
    <stem>_extra.npz       raw messages the processed file dropped (crosses, ...)
    <stem>_bigsize.npz     true sizes of the rows whose size was clipped

"The book before row R" always includes every extra with before_row <= R.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from post_training.policy_bench.book import Book

EXTRA_FIELDS = ("order_id", "event_type", "direction", "price_abs", "size")


@dataclass
class DayFiles:
    entry: dict
    messages: np.ndarray
    l10: np.ndarray
    queue: np.ndarray
    snapshots: dict              # <stem>_book_full.npz, read once (NpzFile re-reads on every access)
    extra: dict | None
    bigsize: tuple[np.ndarray, np.ndarray] | None

    @classmethod
    def open(cls, shard_root: str, entry: dict) -> "DayFiles":
        base = os.path.join(shard_root, entry["month"])
        stem = os.path.splitext(os.path.basename(entry["message"]))[0]
        fb = os.path.join(entry["fullbook_dir"], stem)
        return cls(
            entry=entry,
            messages=np.load(os.path.join(base, entry["message"]), mmap_mode="r"),
            l10=np.load(os.path.join(base, entry["orderbook"]), mmap_mode="r"),
            queue=np.load(fb + "_queue.npy", mmap_mode="r"),
            snapshots=dict(np.load(fb + "_book_full.npz")),
            extra=_load_extra(fb + "_extra.npz"),
            bigsize=_load_bigsize(fb + "_bigsize.npz"),
        )

    @property
    def n_msgs(self) -> int:
        return int(self.messages.shape[0])

    @property
    def grid(self) -> int:
        return int(self.snapshots["grid"])

    def snapshot_rows(self) -> np.ndarray:
        return np.asarray(self.snapshots["snap_row"])

    def rows(self, lo: int, hi: int) -> np.ndarray:
        """Messages [lo, hi) with the 9999 size clip undone."""
        msg = np.array(self.messages[lo:hi])
        if self.bigsize is not None and len(self.bigsize[0]):
            a, b = np.searchsorted(self.bigsize[0], lo), np.searchsorted(self.bigsize[0], hi)
            msg[self.bigsize[0][a:b] - lo, 5] = self.bigsize[1][a:b]
        return msg

    def extras_before(self, lo: int, hi: int) -> Iterator[tuple[int, tuple[int, ...]]]:
        """(before_row, row) for extras with lo <= before_row < hi, in file order.
        `row` is shaped like a processed message for Book.apply_row."""
        if self.extra is None:
            return
        br = self.extra["before_row"]
        a, b = np.searchsorted(br, lo, "left"), np.searchsorted(br, hi, "left")
        for j in range(a, b):
            oid, evt, d, price, size = (int(self.extra[c][j]) for c in EXTRA_FIELDS)
            yield int(br[j]), (oid, evt, d, price, 0, size)

    def book_before(self, row: int) -> Book:
        """Full book before message `row`: nearest snapshot at or before it, then replay."""
        snap = self.snapshot_rows()
        s = int(np.searchsorted(snap, row, "right")) - 1
        bk = Book.from_snapshot(self.snapshots, s)
        for _ in self.replay(bk, int(snap[s]), row):
            pass
        return bk

    def replay(self, book: Book, lo: int, hi: int) -> Iterator[tuple[int, int]]:
        """Advance `book` from before row lo to before row hi.

        Yields (r, slot) once row r and the extras that precede row r + 1 are
        applied, so the book is then the book before row r + 1. `slot` is
        row r's queue slot measured before it was applied."""
        msg = self.rows(lo, hi)
        pending = list(self.extras_before(lo + 1, hi + 1))
        j = 0
        for k in range(hi - lo):
            r = lo + k
            slot = book.apply_row(msg[k])
            while j < len(pending) and pending[j][0] == r + 1:
                book.apply_row(pending[j][1])
                j += 1
            yield r, slot


def _load_extra(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    x = np.load(path)
    out = {k: np.asarray(x[k]) for k in ("before_row",) + EXTRA_FIELDS}
    o = np.argsort(out["before_row"], kind="stable")
    return {k: v[o] for k, v in out.items()}


def _load_bigsize(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(path):
        return None
    x = np.load(path)
    r, q = np.asarray(x["row"]), np.asarray(x["size"])
    o = np.argsort(r, kind="stable")
    return r[o], q[o]
