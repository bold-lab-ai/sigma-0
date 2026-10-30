"""The message log both models read: processed rows and the L10 row before each.

`RowLog` holds 14-column processed rows (signal.processed_row) and 43-column
L10 rows (book.L10Writer), with one more L10 row than processed rows: the last
one is the book now. `start_state` builds the log, the book and the add
registry of an episode from a real day, the path gate G4 part B checks
against the files (job 7842553: 2,000 of 2,000 decisions equal).
"""

from __future__ import annotations

import os

import numpy as np

from post_training.policy_bench.book import Book, L10Writer
from post_training.policy_bench.daydata import DayFiles
from post_training.policy_bench.signal import AddRegistry, price_rel_clip, processed_row

MSG_COLS = [0, 1, 2, 3, 5, 6, 7, 8, 9]    # the processed columns a raw message supplies


class RowLog:
    """Processed rows and L10 rows, grown one row at a time."""

    def __init__(self, rows: np.ndarray, l10: np.ndarray, capacity: int) -> None:
        if len(l10) != len(rows) + 1:
            raise ValueError(f"{len(rows)} rows need {len(rows) + 1} L10 rows, got {len(l10)}")
        n = len(rows)
        self._rows = np.zeros((n + capacity, 14), np.int64)
        self._l10 = np.zeros((n + capacity + 1, 43), np.int64)
        self._rows[:n], self._l10[:n + 1] = rows, l10
        self.n = n

    def append(self, row: np.ndarray, l10_after: np.ndarray) -> None:
        if self.n == len(self._rows):
            raise ValueError("row log is full")
        self._rows[self.n] = row
        self._l10[self.n + 1] = l10_after
        self.n += 1

    def rows(self, last: int) -> np.ndarray:
        return self._rows[self.n - last:self.n]

    def l10(self, last: int) -> np.ndarray:
        """The L10 rows before the last `last - 1` rows and the book now."""
        return self._l10[self.n + 1 - last:self.n + 1]

    @property
    def now(self) -> np.ndarray:
        return self._l10[self.n]


def open_day(shard_root: str, add_dir: str, entry: dict) -> tuple[DayFiles, np.ndarray]:
    """A stock-day and its add index (build_add_index.py output)."""
    stem = os.path.splitext(os.path.basename(entry["message"]))[0]
    index = np.load(os.path.join(add_dir, entry["month"], stem + "_adds.npy"), mmap_mode="r")
    return DayFiles.open(shard_root, entry), index


def start_state(day: DayFiles, index: np.ndarray, start: int, history: int, capacity: int,
                book: Book | None = None, at: int = 0):
    """Book, row log, add registry and L10 writer before row `start`, with the
    `history` rows before it rebuilt from our book. Returns (book, log, reg, writer).

    Given `book`, the real book before row `at` of the same day (at <= start -
    history), it is replayed forward instead of rebuilt from a snapshot."""
    s0 = start - history
    if s0 < 0:
        raise ValueError(f"start {start} has fewer than {history} rows before it")
    if book is None:
        book = day.book_before(s0)
    elif at > s0:
        raise ValueError(f"cannot replay back from row {at} to {s0}")
    else:
        for _ in day.replay(book, at, s0):
            pass
    reg = AddRegistry(index, s0)
    clip = price_rel_clip(day.entry["message"])
    raw = np.asarray(day.messages[s0:start])
    rows = np.zeros((history, 14), np.int64)
    l10 = np.zeros((history + 1, 43), np.int64)
    l10[0] = day.l10[s0]
    writer = L10Writer(l10[0])
    for r, _ in day.replay(book, s0, start):
        k = r - s0
        f = raw[k]
        rows[k] = processed_row(*(int(x) for x in f[MSG_COLS]), l10[k], reg, clip)
        reg.observe(rows[k])
        l10[k + 1] = writer.next_row(book, int(f[8]), int(f[9]))
    return book, RowLog(rows, l10, capacity), reg, writer
