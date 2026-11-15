"""L3 order book with an owner on every resting order.

Every price level is a FIFO queue. The book is the state the bench runs on:
the simulator's QSLOT and L10 row, and later the signal's input, are read from
it, never the other way round.

There are two ways to change it, because the two kinds of input name an order
differently:

    apply_row(row)        one processed LOBSTER row. Real data names the order
                          by id. Used to replay real days and to build the
                          start book of an episode.
    add / remove_at /     generated messages carry a queue slot but no order
    reduce_at / execute   id, and agent orders are handled by the bench. Both
                          go through these, by (side, price, slot).

`execute` fills a level front to back. That is the only place an agent order
can be filled by background flow, so the generator decides how much trades at
a price and the queue decides whose order it was.

Conventions follow nano-sigma's fullbook.py: ASK = +1, BID = -1, processed
direction 1 is the bid side, prices are LOBSTER integers (TICK = 100 is one
cent).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np


ASK, BID = 1, -1
BG, AGENT = 0, 1

# What a processed L10 row holds for a level that does not exist (LOBSTER).
EMPTY_ASK_PRICE = 9_999_999_999
EMPTY_BID_PRICE = -9_999_999_999

# Queue slot codes, as in <stem>_queue.npy column 0.
SLOT_NEW = -1        # event 1 references no resting order
SLOT_UNKNOWN = -2    # the referenced order is not in the book


def side_of_direction(d: int) -> int:
    """Processed direction (1 buy, 0 sell) to book side."""
    return BID if int(d) == 1 else ASK


@dataclass(slots=True)
class RestingOrder:
    oid: int
    size: int
    owner: int = BG


class Book:
    def __init__(self) -> None:
        self._q: dict[int, dict[int, list[RestingOrder]]] = {ASK: {}, BID: {}}
        self._vol: dict[int, dict[int, int]] = {ASK: {}, BID: {}}
        self._px: dict[int, list[int]] = {ASK: [], BID: []}   # ascending
        self._loc: dict[int, tuple[int, int]] = {}            # oid -> (side, price)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_snapshot(cls, z, s: int) -> "Book":
        """Book stored as snapshot `s` of a <stem>_book_full.npz (all orders BG)."""
        bk = cls()
        l0, l1 = int(z["lvl_ptr"][s]), int(z["lvl_ptr"][s + 1])
        sides, prices = z["lvl_side"][l0:l1], z["lvl_price"][l0:l1]
        op, sizes, ids = z["ord_ptr"], z["ord_size"], z["ord_id"]
        for k in range(l1 - l0):
            a, b = int(op[l0 + k]), int(op[l0 + k + 1])
            for oid, q in zip(ids[a:b], sizes[a:b]):
                bk.add(int(sides[k]), int(prices[k]), int(oid), int(q))
        return bk

    # ------------------------------------------------------------ reads
    def best(self, side: int) -> int | None:
        px = self._px[side]
        if not px:
            return None
        return px[0] if side == ASK else px[-1]

    def prices(self, side: int) -> list[int]:
        """Prices of one side, best first."""
        px = self._px[side]
        return list(px) if side == ASK else px[::-1]

    def queue(self, side: int, price: int) -> list[RestingOrder]:
        """The FIFO queue at a price (empty list if none). Do not mutate."""
        return self._q[side].get(price, [])

    def volume(self, side: int, price: int) -> int:
        return self._vol[side].get(price, 0)

    def has(self, oid: int) -> bool:
        return oid in self._loc

    def location(self, oid: int) -> tuple[int, int] | None:
        """(side, price) of a resting order, or None."""
        return self._loc.get(oid)

    def slot_of(self, oid: int, side: int | None = None, price: int | None = None) -> int:
        """Queue position of an order, or SLOT_UNKNOWN. With side and price
        given, the order must rest exactly there."""
        loc = self._loc.get(oid)
        if loc is None or (side is not None and loc != (side, price)):
            return SLOT_UNKNOWN
        for i, o in enumerate(self._q[loc[0]][loc[1]]):
            if o.oid == oid:
                return i
        raise AssertionError(f"order {oid} indexed but not queued")

    def l10(self, n: int = 10) -> np.ndarray:
        """(4 n,) int64 in L10 file order: ask_p, ask_q, bid_p, bid_q per level."""
        out = np.zeros(4 * n, dtype=np.int64)
        a, b = self._px[ASK][:n], self._px[BID][:-n - 1:-1]    # best n of each side
        for i in range(n):
            out[4 * i] = a[i] if i < len(a) else EMPTY_ASK_PRICE
            out[4 * i + 1] = self._vol[ASK][a[i]] if i < len(a) else 0
            out[4 * i + 2] = b[i] if i < len(b) else EMPTY_BID_PRICE
            out[4 * i + 3] = self._vol[BID][b[i]] if i < len(b) else 0
        return out

    def levels(self, side: int) -> dict[int, list[tuple[int, int, int]]]:
        """price -> [(oid, size, owner), ...] for one side, for comparisons."""
        return {p: [(o.oid, o.size, o.owner) for o in q] for p, q in self._q[side].items()}

    # ------------------------------------------------------------ writes
    def add(self, side: int, price: int, oid: int, size: int, owner: int = BG) -> int:
        """Append an order at the back of its queue. Returns its slot."""
        if oid in self._loc:
            raise ValueError(f"order {oid} already rests in the book")
        q = self._q[side].get(price)
        if q is None:
            q = self._q[side][price] = []
            self._vol[side][price] = 0
            bisect.insort(self._px[side], price)
        q.append(RestingOrder(oid, size, owner))
        self._vol[side][price] += size
        self._loc[oid] = (side, price)
        return len(q) - 1

    def remove_at(self, side: int, price: int, slot: int) -> RestingOrder:
        """Delete the whole order at a queue slot."""
        q = self._q[side][price]
        o = q.pop(slot)
        self._vol[side][price] -= o.size
        del self._loc[o.oid]
        if not q:
            self._drop_level(side, price)
        return o

    def reduce_at(self, side: int, price: int, slot: int, qty: int) -> RestingOrder:
        """Take `qty` off the order at a slot; the order leaves at size <= 0."""
        q = self._q[side][price]
        o = q[slot]
        take = min(qty, o.size)
        o.size -= qty
        self._vol[side][price] -= take
        if o.size <= 0:
            q.pop(slot)
            del self._loc[o.oid]
            if not q:
                self._drop_level(side, price)
        return o

    def execute(self, side: int, price: int, qty: int) -> list[tuple[RestingOrder, int]]:
        """Fill `qty` against one level, front of the queue first.
        Returns [(order, filled), ...]. Raises if the level holds less."""
        if qty > self.volume(side, price):
            raise ValueError(f"execute {qty} at {price} but the level holds {self.volume(side, price)}")
        fills = []
        q = self._q[side][price]
        while qty > 0:
            o = q[0]
            take = min(qty, o.size)
            fills.append((o, take))
            self.reduce_at(side, price, 0, take)
            qty -= take
        return fills

    def cross(self, price: int) -> None:
        """Opening or closing cross at `price`: drop asks below it and bids above it."""
        for p in [p for p in self._px[ASK] if p < price]:
            self._clear_level(ASK, p)
        for p in [p for p in self._px[BID] if p > price]:
            self._clear_level(BID, p)

    def apply_row(self, row) -> int:
        """Apply one processed row (events 1-4 by order id; 6 is a cross; the
        rest leave the visible book alone). Returns the row's queue slot before
        it was applied, with the codes of <stem>_queue.npy."""
        oid, evt, d, price, size = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        side = side_of_direction(d)
        if evt == 1:
            self.add(side, price, oid, size)
            return SLOT_NEW
        if evt == 6:
            self.cross(price)
            return SLOT_NEW
        if evt not in (2, 3, 4):
            return SLOT_NEW
        slot = self.slot_of(oid, side, price)
        if slot == SLOT_UNKNOWN:
            return slot
        if evt == 3:
            self.remove_at(side, price, slot)
        else:
            self.reduce_at(side, price, slot, size)
        return slot

    # ------------------------------------------------------------ internals
    def _drop_level(self, side: int, price: int) -> None:
        del self._q[side][price]
        del self._vol[side][price]
        px = self._px[side]
        px.pop(bisect.bisect_left(px, price))

    def _clear_level(self, side: int, price: int) -> None:
        for o in self._q[side][price]:
            del self._loc[o.oid]
        self._drop_level(side, price)


class L10Writer:
    """Full 43-column L10 rows, as the data files hold them (job 7832710, 100%):

        col 0      change of the mid since the previous row, in ticks. The mid is
                   (ask1 + bid1) / 2 rounded half to even to a tick; an empty side
                   carries its last price forward
        cols 1, 2  time (s, ns) of the previous row's message
        cols 3..   Book.l10()

    One call per row: `next_row` after a row has been applied gives the row of
    the book before the next message. A side that has been empty since `first`
    gives col 0 = 0 (the file's value there is unknown).
    """

    def __init__(self, first: np.ndarray) -> None:
        self._ask = self._bid = None
        self._mid = self._observe(int(first[3]), int(first[5]))

    def next_row(self, book: Book, t_s: int, t_ns: int) -> np.ndarray:
        body = book.l10()
        mid = self._observe(int(body[0]), int(body[2]))
        d = 0 if mid is None or self._mid is None else int(mid - self._mid)
        self._mid = mid
        return np.concatenate([np.array([d, t_s, t_ns], dtype=np.int64), body])

    def _observe(self, ask: int, bid: int) -> float | None:
        if ask < EMPTY_ASK_PRICE:
            self._ask = ask
        if bid > EMPTY_BID_PRICE:
            self._bid = bid
        if self._ask is None or self._bid is None:
            return None
        # the expression the files were verified against, float rounding included
        return float(np.round((self._ask + self._bid) / 2, -2)) / 100
