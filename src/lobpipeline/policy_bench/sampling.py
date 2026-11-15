"""How a background message is sampled from the simulator's next-token distribution.

A newtok message is (positions):

    0 event, 1 direction, 2-5 dt (s, ms, us, ns), 6-10 t (ks, s, ms, us, ns)
    then a level (sign, two dP digits, two size digits) and, for events 2 and 3,
    a QSLOT; an execution repeats levels and ends with EOE.

The absolute time is never sampled: once dt is known, positions 6-10 are forced
to t_prev + dt (`MessageState.forced`). The sampler draws a whole message and
`finish` accepts it if it decodes, is the canonical encoding of what it decodes
to, ends no later than the file's last time, passes rules.check in the
current book and leaves both sides of the book non-empty. (dP is counted from
the mid, and a one-sided book has a placeholder price on the empty side: no
message could be written after it.) A rejected message is drawn again; after 32 rejections
`legal_tokens` masks every position to the tokens that can still end in a
legal message, so the draw cannot fail.
"""

from __future__ import annotations

import numpy as np

from lobpipeline.policy_bench import rules
from lobpipeline.policy_bench.book import BG, Book, side_of_direction
from lobpipeline.policy_bench.codec import MAX_DP, MAX_SIZE, NS, DecodeError, Message, decode, encode
from lobpipeline.policy_bench.nanosigma.data import DIR0, EVT0, PRICE0, SIGN0, SIZE0, TICK, TIME0
from lobpipeline.policy_bench.nanosigma.newtok import EOE, N_QSLOT, QSLOT0

MAX_REJECTIONS = 32
PREFIX = 11
LAST_SLOT = N_QSLOT - 1
MAX_EXEC_LEVELS = 20      # a longer execution is cut and rejected


def file_end_ns(message_file: str) -> int:
    """The last time a stock-day's file covers, from `<T>_<date>_<start ms>_<end ms>_message...`."""
    parts = message_file.rsplit("/", 1)[-1].split("_")
    return int(parts[3]) * 10**6


class MessageState:
    """The tokens of one message as they are chosen."""

    def __init__(self, t_prev: int) -> None:
        self.t_prev = t_prev
        self.toks: list[int] = []

    @property
    def evt(self) -> int:
        return self.toks[0] - EVT0 + 1

    def dt(self) -> int:
        v = [t - TIME0 for t in self.toks[2:6]]
        return v[0] * NS + v[1] * 10**6 + v[2] * 10**3 + v[3]

    def forced(self) -> int | None:
        """The token the next position must be, if it is not sampled."""
        k = len(self.toks)
        if not PREFIX - 5 <= k < PREFIX:
            return None
        t = self.t_prev + self.dt()
        ts, tn = t // NS, t % NS
        vals = [ts // 1000, ts % 1000, tn // 10**6, tn // 10**3 % 1000, tn % 1000]
        return TIME0 + vals[k - (PREFIX - 5)]

    def push(self, tok: int) -> None:
        self.toks.append(int(tok))

    def kind(self) -> str:
        """What the next token is: evt, dir, dt_s, dt_digit, time (forced), sign, price1,
        price2, size1, size2, qslot, sign_or_eoe, or done."""
        k, t = len(self.toks), self.toks
        if k < PREFIX:
            return ("evt", "dir", "dt_s", "dt_digit", "dt_digit", "dt_digit")[k] if k < 6 else "time"
        if not EVT0 <= t[0] < EVT0 + 4:
            return "done"                  # a bad event token: let finish reject it
        evt, j = self.evt, k - PREFIX
        if evt in (1, 2, 3):
            if j < 5:
                return ("sign", "price1", "price2", "size1", "size2")[j]
            return "qslot" if evt in (2, 3) and j == 5 else "done"
        if (t[-1] == EOE and j > 0) or j > 5 * MAX_EXEC_LEVELS:
            return "done"
        r = j % 5
        if r == 0:
            return "sign" if j == 0 else "sign_or_eoe"
        return ("sign", "price1", "price2", "size1", "size2")[r]

    def done(self) -> bool:
        return len(self.toks) >= PREFIX and self.kind() == "done"


def finish(state: MessageState, book: Book, mid: int, end: int) -> tuple[Message | None, str | None]:
    """(message, None) if the tokens are a legal message now, else (None, reason)."""
    try:
        m = decode(state.toks, mid)
    except DecodeError:
        return None, "decode"
    if list(encode(m, mid)) != state.toks:
        return None, "noncanonical"
    if m.time > end:
        return None, "after_close"
    if any(p <= 0 for p, _ in m.levels):
        return None, "price_not_positive"
    reason = rules.check(m, book)
    if reason is None and empties_side(m, book):
        reason = "empties_side"
    return (m, None) if reason is None else (None, reason)


def side_volume(book: Book, side: int) -> int:
    return sum(book.volume(side, p) for p in book.prices(side))


def empties_side(m: Message, book: Book) -> bool:
    """Would this legal message take the last share off its side of the book?"""
    side = side_of_direction(m.d)
    if m.evt == 3:
        return len(book.prices(side)) == 1 and len(book.queue(side, m.levels[0][0])) == 1
    if m.evt == 4:
        return sum(q for _, q in m.levels) >= side_volume(book, side)
    return False


# ---------------------------------------------------------------- masking
# Every rule is a set of allowed values, kept as inclusive ranges [(lo, hi), ...].
# A multi-token field (dP, size) is masked digit by digit through its ranges.
def _values(ranges) -> np.ndarray:
    if not ranges:
        return np.zeros(0, np.int64)
    return np.unique(np.concatenate([np.arange(a, b + 1) for a, b in ranges]))


def _lead(ranges, base: int):
    """Ranges of value // base."""
    return [(a // base, b // base) for a, b in ranges]


def _tail(ranges, base: int, lead: int):
    """Ranges of value % base among values with value // base == lead."""
    out = []
    for a, b in ranges:
        lo, hi = max(a, lead * base), min(b, lead * base + base - 1)
        if lo <= hi:
            out.append((lo - lead * base, hi - lead * base))
    return out


def _on_grid(price: int, mid: int) -> bool:
    """A price the dP tokens can name: a whole number of ticks from the mid, within MAX_DP."""
    return (price - mid) % TICK == 0 and abs(price - mid) // TICK <= MAX_DP


def _size_ranges(evt: int, rem: int):
    """Sizes a cancel (2) or delete (3) may take from an order with `rem` shares left."""
    if evt == 3:
        q = min(rem, MAX_SIZE)
        return [(q, q)]
    return [(1, min(rem - 1, MAX_SIZE))] if rem >= 2 else []


def _last_order(book: Book, side: int) -> bool:
    ps = book.prices(side)
    return len(ps) == 1 and len(book.queue(side, ps[0])) == 1


def _bg_sizes(book: Book, side: int, price: int, evt: int):
    if evt == 3 and _last_order(book, side):
        return []
    out = []
    for o in book.queue(side, price):
        if o.owner == BG:
            out += _size_ranges(evt, o.size)
    return out


def _cancel_prices(book: Book, side: int, evt: int, mid: int) -> list[int]:
    return [p for p in book.prices(side) if _on_grid(p, mid) and _bg_sizes(book, side, p, evt)]


def _new_dp(book: Book, side: int, mid: int):
    """dP ranges of a new order that does not cross and has a positive price."""
    lo, hi = max(-MAX_DP, 1 - mid // TICK), MAX_DP
    opp = book.best(-side)
    if opp is not None:
        if side > 0:                          # an ask rests above the best bid
            lo = max(lo, (opp - mid) // TICK + 1)
        else:                                 # a bid rests below the best ask
            hi = min(hi, -((mid - opp) // TICK) - 1)
    return [(lo, hi)] if lo <= hi else []


def _dp_ranges(evt: int, side: int, book: Book, mid: int, level: int):
    if evt == 1:
        return _new_dp(book, side, mid)
    if evt in (2, 3):
        return [((p - mid) // TICK,) * 2 for p in _cancel_prices(book, side, evt, mid)]
    prices = book.prices(side)
    if level < len(prices) and _on_grid(prices[level], mid) and _exec_max(book, side, level) >= 1:
        return [((prices[level] - mid) // TICK,) * 2]
    return []


def _exec_max(book: Book, side: int, level: int) -> int:
    """Most shares an execution may take at a level: never the whole side."""
    prices = book.prices(side)
    vol = book.volume(side, prices[level])
    return min(vol - 1 if level == len(prices) - 1 else vol, MAX_SIZE)


def _side_ok(evt: int, side: int, book: Book, mid: int) -> bool:
    return bool(_dp_ranges(evt, side, book, mid, 0))


def _abs_ranges(dp_ranges, positive: bool):
    """|dP| ranges under one sign token: SIGN0 + 1 is dP > 0, SIGN0 is dP <= 0."""
    out = []
    for a, b in dp_ranges:
        if positive and b >= 1:
            out.append((max(a, 1), b))
        if not positive and a <= 0:
            out.append((-min(b, 0), -a))
    return out


def _level(state: MessageState, mid: int, k: int):
    """(tokens, price, size) of level k so far; price and size None until complete."""
    cur = state.toks[PREFIX + 5 * k:PREFIX + 5 * k + 5]
    price = size = None
    if len(cur) >= 3:
        a = (cur[1] - PRICE0) * 1000 + (cur[2] - PRICE0)
        price = mid + (a if cur[0] == SIGN0 + 1 else -a) * TICK
    if len(cur) == 5:
        size = (cur[3] - SIZE0) * 100 + (cur[4] - SIZE0)
    return cur, price, size


def _exec_may_continue(state: MessageState, book: Book, mid: int, side: int, k: int) -> bool:
    """Level k of an execution was taken whole and level k + 1 exists."""
    _, price, size = _level(state, mid, k)
    return size == min(book.volume(side, price), MAX_SIZE) and bool(_dp_ranges(4, side, book, mid, k + 1))



def legal_tokens(state: MessageState, book: Book, mid: int, end: int) -> np.ndarray:
    """Token ids the next position may take so that the message can still end legal."""
    kind = state.kind()
    if kind == "time":
        return np.array([state.forced()])
    if kind == "evt":
        evts = [e for e in (1, 2, 3, 4) if any(_side_ok(e, s, book, mid) for s in (1, -1))]
        return EVT0 + np.array(evts, np.int64) - 1
    evt = state.evt
    if kind == "dir":
        return np.array([DIR0 + d for d in (0, 1) if _side_ok(evt, side_of_direction(d), book, mid)], np.int64)
    if kind in ("dt_s", "dt_digit"):
        room = end - state.t_prev
        k = len(state.toks) - 2                          # 0 for dt_s, 1..3 for the digits
        scale = (NS, 10**6, 10**3, 1)[k]
        sofar = sum((t - TIME0) * w for t, w in zip(state.toks[2:], (NS, 10**6, 10**3)))
        top = min(999, (room - sofar) // scale)
        return TIME0 + np.arange(0, top + 1) if top >= 0 else np.zeros(0, np.int64)
    side = side_of_direction(state.toks[1] - DIR0)
    k = (len(state.toks) - PREFIX) // 5 if evt == 4 else 0
    cur, price, _ = _level(state, mid, k)
    if kind in ("sign", "sign_or_eoe"):
        dps = _dp_ranges(evt, side, book, mid, k)
        signs = [SIGN0 + int(pos) for pos in (False, True) if _abs_ranges(dps, pos)]
        if kind == "sign_or_eoe":
            return np.array((signs if _exec_may_continue(state, book, mid, side, k - 1) else []) + [EOE], np.int64)
        return np.array(signs, np.int64)
    if kind in ("price1", "price2"):
        a = _abs_ranges(_dp_ranges(evt, side, book, mid, k), cur[0] == SIGN0 + 1)
        if kind == "price1":
            return PRICE0 + _values(_lead(a, 1000))
        return PRICE0 + _values(_tail(a, 1000, cur[1] - PRICE0))
    if kind in ("size1", "size2"):
        if evt == 1:
            sizes = [(1, MAX_SIZE)]
        elif evt in (2, 3):
            sizes = _bg_sizes(book, side, price, evt)
        else:
            sizes = [(1, _exec_max(book, side, k))]
        if kind == "size1":
            return SIZE0 + _values(_lead(sizes, 100))
        return SIZE0 + _values(_tail(sizes, 100, cur[3] - SIZE0))
    if kind == "qslot":
        _, price, size = _level(state, mid, 0)
        q = book.queue(side, price)
        ok = [s for s in range(len(q)) if q[s].owner == BG and rules._size_ok(evt, size, q[s].size)
              and not (evt == 3 and _last_order(book, side))]
        return QSLOT0 + np.array(sorted({min(s, LAST_SLOT) for s in ok}), np.int64)
    raise ValueError(f"no token follows a finished message ({kind})")
