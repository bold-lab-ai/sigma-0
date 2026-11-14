"""Markets: where the background flow comes from and how agent orders fill.

`ReplayMarket` replays a real day row by row. Agent orders are virtual: they
fill against the book but never change it, so the rows that follow stay the
real ones. It takes market orders only; a resting order's fate depends on how
the market reacts to it, which a replay cannot know, so a limit order raises
instead of getting a guessed fill. The fill model is a choice:

    MidFill    the whole order at the mid (nano-sigma's PnL bench, variant A)
    BookFill   the opposite side from the best price, level by level (taker)

Orders become active `latency` background messages after they are submitted,
or, with `latency_ns`, before the first real row whose time is at least the
decision's time (the last row's) plus latency_ns.
Episodes of one day in increasing start order reuse the day's book: the next
start state is replayed forward from where the last episode ended, not rebuilt
from a snapshot. The book is the real one either way (agent orders never touch
it), so this changes the cost, not the result.
An order that would take the position past +-inventory_cap, or that meets an
empty side, is refused and counted.
"""

from __future__ import annotations

import numpy as np

from post_training.policy_bench.book import ASK, BID, Book
from post_training.policy_bench.codec import NS
from post_training.policy_bench.history import MSG_COLS, open_day, start_state
from post_training.policy_bench.signal import price_rel_clip, processed_row
from post_training.policy_bench.trace import Trace
from post_training.policy_bench.types import BUY, Episode, Fill, MarketOrder, Obs, mid_of


class MidFill:
    """A market order fills whole at the mid."""

    def fill(self, order: MarketOrder, book: Book, k: int) -> list[Fill]:
        mid = mid_of(book.l10(1))
        if mid is None:
            return []
        return [Fill(k, order.side, mid, order.size, False, mid)]


class BookFill:
    """A market order takes the opposite side from the best price. Refused
    (no fill) if the side holds less than the order."""

    def fill(self, order: MarketOrder, book: Book, k: int) -> list[Fill]:
        mid = mid_of(book.l10(1))
        side = ASK if order.side == BUY else BID
        if mid is None or sum(book.volume(side, p) for p in book.prices(side)) < order.size:
            return []
        out, left = [], order.size
        for p in book.prices(side):
            take = min(left, book.volume(side, p))
            out.append(Fill(k, order.side, float(p), take, False, mid))
            left -= take
            if left == 0:
                break
        return out


class ReplayMarket:
    """One episode at a time (batch 1); the row index of every method is 0."""
    supports = (MarketOrder,)
    batch = 1

    def __init__(self, shard_root: str, add_dir: str, fill, history: int = 3968,
                 n_messages: int = 1024, latency: int = 1, inventory_cap: int = 1000,
                 latency_ns: int | None = None) -> None:
        self.shard_root, self.add_dir, self.fill_model = shard_root, add_dir, fill
        self.history, self.n_messages, self.latency, self.cap = history, n_messages, latency, inventory_cap
        self.latency_ns = latency_ns           # None: `latency` messages; else nanoseconds after the decision
        if latency_ns is not None and latency_ns < 0:
            raise ValueError(f"latency_ns {latency_ns} < 0")

    def reset(self, eps: list[Episode]) -> None:
        if len(eps) != 1:
            raise ValueError(f"ReplayMarket runs one episode at a time, got {len(eps)}")
        self.load(0, eps[0])

    def unload(self, i: int = 0) -> None:
        pass                                   # the day stays open for the next episode

    def load(self, i: int, ep: Episode) -> None:
        last = getattr(self, "ep", None)
        if last is not None and last.entry["message"] == ep.entry["message"] \
                and last.start + self.k <= ep.start - self.history:
            day, index, book, at = self.day, self.index, self.book, last.start + self.k
        else:
            (day, index), book, at = open_day(self.shard_root, self.add_dir, ep.entry), None, 0
        if ep.start + self.n_messages > day.n_msgs:
            raise ValueError(f"{ep.entry['message']}: {ep.start} + {self.n_messages} rows past the day")
        self.ep, self.day, self.index = ep, day, index
        self.book, self.rowlog, self.reg, self.writer, _ = start_state(day, index, ep.start, self.history,
                                                                    self.n_messages, book, at)
        self.clip = price_rel_clip(ep.entry["message"])
        self._rows = day.replay(self.book, ep.start, ep.start + self.n_messages)
        self._raw = np.asarray(day.messages[ep.start:ep.start + self.n_messages])
        self.k, self.position, self.cash, self.refused = 0, 0, 0.0, 0
        self._pending: list[tuple[int | None, int | None, MarketOrder]] = []   # due at k, due at ns
        self._fills: list[Fill] = []
        self._new: list[Fill] = []
        self._mid = [mid_of(self.rowlog.now[3:])]
        self._decisions: list[int] = []
        self._signals: list[np.ndarray] = []
        self._decided = -1

    def log(self, i: int = 0):
        return self.rowlog

    def observe(self, i: int = 0) -> Obs:
        t = self.rowlog.rows(1)[0]
        obs = Obs(self.k, int(t[8]), int(t[9]), self.rowlog.now[3:].copy(), self.position, self.cash,
                  [], self._new, pending=[o for _, _, o in self._pending])
        self._new = []
        return obs

    def record_decision(self, i: int, signal: np.ndarray | None) -> None:
        self._decided = self.k
        self._decisions.append(self.k)
        if signal is not None:
            self._signals.append(np.asarray(signal, np.float64))

    def submit(self, i: int, orders) -> None:
        for o in orders:
            if not isinstance(o, self.supports):
                raise NotImplementedError(f"ReplayMarket takes market orders only, got {type(o).__name__}")
            if self.latency_ns is None:
                self._pending.append((self.k + self.latency, None, o))
            else:
                t = self.rowlog.rows(1)[0]
                self._pending.append((None, int(t[8]) * NS + int(t[9]) + self.latency_ns, o))
        self._activate()                       # latency 0: before the next message

    def advance(self) -> list[Fill]:
        """Activate due orders, then apply the next real row. Returns the fills."""
        fills = self._activate()
        r, _ = next(self._rows)
        f = self._raw[r - self.ep.start]
        row = processed_row(*(int(x) for x in f[MSG_COLS]), self.rowlog.now, self.reg, self.clip)
        self.reg.observe(row)
        self.rowlog.append(row, self.writer.next_row(self.book, int(f[8]), int(f[9])))
        self.k += 1
        self._mid.append(mid_of(self.rowlog.now[3:]))
        return fills

    def done(self) -> bool:
        return self.k >= self.n_messages

    def finished(self, i: int = 0) -> bool:
        return self.done()

    def run(self, every: int) -> list[int]:
        """Advance to the next decision (k a multiple of `every`, not yet decided) or the end."""
        while not (self.done() or (self.k % every == 0 and self._decided != self.k)):
            self.advance()
        return [0]

    def trace(self, i: int = 0) -> Trace:
        return Trace.build(self.ep, self._mid, self.rowlog.now[3:], self._fills, self._decisions,
                           self._signals, self.refused, book=self.book)

    def _activate(self) -> list[Fill]:
        out: list[Fill] = []
        t_next = None                          # time of the next real row
        if self.k < self.n_messages:
            f = self._raw[self.k]
            t_next = int(f[8]) * NS + int(f[9])

        def is_due(at, ns):
            return at <= self.k if ns is None else t_next is not None and ns <= t_next
        due = [o for at, ns, o in self._pending if is_due(at, ns)]
        self._pending = [(at, ns, o) for at, ns, o in self._pending if not is_due(at, ns)]
        for o in due:
            if abs(self.position + o.side * o.size) > self.cap:
                self.refused += 1
                continue
            fills = self.fill_model.fill(o, self.book, self.k)
            if not fills:
                self.refused += 1
                continue
            for x in fills:
                self.position += x.side * x.size
                self.cash -= x.side * x.size * x.price
            self._fills += fills
            self._new += fills
            out += fills
        return out
