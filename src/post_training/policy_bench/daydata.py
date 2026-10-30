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

`build_add_index` gives the first event-1 row of every order id of a day
(stored as <stem>_adds.npy by run/post_training/policy_bench/build_add_index.py).

"The book before row R" always includes every extra with before_row <= R.
"""

from __future__ import annotations

import os
import zipfile
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
    snapshots: "Snapshots"
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
            snapshots=Snapshots(fb + "_book_full.npz"),
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
        bk = Book.from_snapshot(self.snapshots.one(s), 0)
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


class Snapshots:
    """<stem>_book_full.npz, read lazily.

    The order arrays of a big day are compressed members of several GB once
    inflated (GOOGL 2026-01-21: 3.1 GB). `one(s)` streams out snapshot s alone;
    indexing by key inflates a whole member once and keeps it."""

    SMALL = ("snap_row", "lvl_ptr", "grid")

    def __init__(self, path: str) -> None:
        self.path = path
        with np.load(path) as z:
            self._arrays = {k: np.asarray(z[k]) for k in self.SMALL}

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            with np.load(self.path) as z:
                self._arrays[key] = np.asarray(z[key])
        return self._arrays[key]

    def one(self, s: int) -> dict:
        """Snapshot s as a one-snapshot mapping for Book.from_snapshot(..., 0)."""
        lp = self._arrays["lvl_ptr"]
        l0, l1 = int(lp[s]), int(lp[s + 1])
        with zipfile.ZipFile(self.path) as zf:
            side = _read_range(zf, "lvl_side", l0, l1)
            price = _read_range(zf, "lvl_price", l0, l1)
            ptr = _read_range(zf, "ord_ptr", l0, l1 + 1)
            a, b = int(ptr[0]), int(ptr[-1])
            size = _read_range(zf, "ord_size", a, b)
            oid = _read_range(zf, "ord_id", a, b)
        return dict(lvl_ptr=np.array([0, l1 - l0]), lvl_side=side, lvl_price=price,
                    ord_ptr=ptr - a, ord_size=size, ord_id=oid)


def _read_range(zf: zipfile.ZipFile, name: str, lo: int, hi: int) -> np.ndarray:
    """Elements [lo, hi) of a 1-D array member, without inflating the rest of it."""
    with zf.open(name + ".npy") as f:
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        _, _, dtype = read(f)
        f.seek(f.tell() + lo * dtype.itemsize)
        return np.frombuffer(f.read((hi - lo) * dtype.itemsize), dtype).copy()


def build_add_index(msg: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """The first event-1 row of every order id in processed rows `msg` (a whole
    day; a memory map is read in chunks), as a (6, n) int64 array sorted by id:
    oid, row, price_rel, size, t_s, t_ns. Rows are contiguous, so
    np.searchsorted on row 0 of a memory-mapped index reads a few pages."""
    cols = [0, 4, 5, 8, 9]        # oid, price_rel, size, t_s, t_ns
    parts = []
    for lo in range(0, len(msg), chunk):
        m = np.asarray(msg[lo:lo + chunk])
        r = np.nonzero(m[:, 1] == 1)[0]
        parts.append(np.column_stack([m[r][:, cols[:1]], r + lo, m[r][:, cols[1:]]]))
    adds = np.concatenate(parts) if parts else np.zeros((0, 6), np.int64)
    _, first = np.unique(adds[:, 0], return_index=True)
    return np.ascontiguousarray(adds[first].T.astype(np.int64))


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
