"""newtok messages, the unit the simulator reads and writes.

A Message is one newtok message: one processed row, or, for executions, one
burst of rows (consecutive executions with the same timestamp and side),
grouped by price in the order the rows hit them. `encode` gives exactly the
tokens nano-sigma's `encode_window` gives for that message, given the mid
before it; `decode` is the strict inverse, applied to generated tokens.

Prices inside a Message are absolute. Only the tokens are relative to the mid,
so a Message means the same thing whatever book it is later applied to.

encode is lossy in five places, so a round trip returns these clipped:
size to 0..9999, dt_s to 0..999, |dP| to 999999 ticks, dP floored to the
tick, QSLOT >= 99 to 99.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lobpipeline.policy_bench.nanosigma.data import (
    C_DIR, C_DTNS, C_DTS, C_EVT, C_SIZE, C_TNS, C_TS, DIR0, EVT0, NA, PRICE0, SIGN0, SIZE0, TICK, TIME0,
)
from lobpipeline.policy_bench.nanosigma.newtok import C_PRICE_ABS, EOE, N_QSLOT, QSLOT0

SLOT_NONE = -1   # events 1 and 4 carry no QSLOT
SLOT_NA = -2     # the NA token: the referenced order was not in the book

MAX_SIZE = 9999
MAX_DP = 999_999
NS = 10**9


class DecodeError(ValueError):
    pass


@dataclass(frozen=True)
class Message:
    evt: int                               # 1 new, 2 cancel, 3 delete, 4 execution
    d: int                                 # processed direction: 1 bid side, 0 ask side
    dt_s: int
    dt_ns: int
    t_s: int
    t_ns: int
    levels: tuple[tuple[int, int], ...]    # (price_abs, size); one entry unless evt == 4
    qslot: int = SLOT_NONE                 # 0..99 for events 2 and 3, else SLOT_NONE / SLOT_NA

    @property
    def time(self) -> int:
        return self.t_s * NS + self.t_ns

    @property
    def dt(self) -> int:
        return self.dt_s * NS + self.dt_ns


def mid_floor(ask1: int, bid1: int) -> int:
    """The newtok reference price: mid before the message, floored to the tick."""
    return (int(ask1) + int(bid1)) // 2 // TICK * TICK


def burst_bounds(msg: np.ndarray) -> list[tuple[int, int]]:
    """[(i, j), ...]: rows i..j-1 form one message (the encode_window rule)."""
    msg = np.asarray(msg)
    n = msg.shape[0]
    evt = np.clip(msg[:, C_EVT], 1, 4)
    d = np.clip(msg[:, C_DIR], 0, 1)
    same = np.zeros(n, bool)
    if n > 1:
        same[1:] = ((evt[1:] == 4) & (evt[:-1] == 4) & (msg[1:, C_TS] == msg[:-1, C_TS])
                    & (msg[1:, C_TNS] == msg[:-1, C_TNS]) & (d[1:] == d[:-1]))
    starts = np.flatnonzero(~same)
    ends = np.append(starts[1:], n)
    return list(zip(starts.tolist(), ends.tolist()))


def message_from_rows(rows: np.ndarray, qslot: int = SLOT_NONE) -> Message:
    """One Message from the processed rows of one message (unclipped values).
    `qslot` is the first row's queue slot (<stem>_queue.npy codes)."""
    r0 = rows[0]
    evt = int(np.clip(r0[C_EVT], 1, 4))
    if evt == 4:
        # Group by tick, as encode_window does (it groups by floored dP); a level
        # keeps the price of its first row.
        agg: dict[int, list[int]] = {}
        for r in rows:
            p = int(r[C_PRICE_ABS])
            agg.setdefault(p // TICK, [p, 0])[1] += int(r[C_SIZE])
        levels = tuple((p, q) for p, q in agg.values())
    else:
        levels = ((int(r0[C_PRICE_ABS]), int(r0[C_SIZE])),)
    if evt in (2, 3):
        slot = int(qslot) if qslot >= 0 else SLOT_NA
    else:
        slot = SLOT_NONE
    return Message(evt, int(np.clip(r0[C_DIR], 0, 1)), int(r0[C_DTS]), int(r0[C_DTNS]),
                   int(r0[C_TS]), int(r0[C_TNS]), levels, slot)


# ---------------------------------------------------------------- encode
def _time_tokens(m: Message) -> list[int]:
    dts = min(max(m.dt_s, 0), 999)
    dtn = min(max(m.dt_ns, 0), NS - 1)
    ts = min(max(m.t_s, 0), 86399)
    tn = min(max(m.t_ns, 0), NS - 1)
    vals = [dts, dtn // 10**6, dtn // 10**3 % 1000, dtn % 1000,
            ts // 1000, ts % 1000, tn // 10**6, tn // 10**3 % 1000, tn % 1000]
    return [TIME0 + v for v in vals]


def _level_tokens(price: int, size: int, mid: int) -> list[int]:
    dp = (price - mid) // TICK
    a = min(abs(dp), MAX_DP)
    q = min(max(size, 0), MAX_SIZE)
    return [SIGN0 + 1 if dp > 0 else SIGN0, PRICE0 + a // 1000 % 1000, PRICE0 + a % 1000,
            SIZE0 + q // 100, SIZE0 + q % 100]


def encode(m: Message, mid: int) -> np.ndarray:
    toks = [EVT0 + m.evt - 1, DIR0 + m.d] + _time_tokens(m)
    for p, q in m.levels:
        toks += _level_tokens(p, q, mid)
    if m.evt == 4:
        toks.append(EOE)
    elif m.evt in (2, 3):
        toks.append(QSLOT0 + min(m.qslot, N_QSLOT - 1) if m.qslot >= 0 else NA)
    return np.asarray(toks, dtype=np.int64)


def encode_stream(msg: np.ndarray, l10: np.ndarray, qslot: np.ndarray):
    """Same contract as nano-sigma's encode_window (msg rows, 43-col L10 rows
    before each row, queue slots): (tokens, msg_of_token, first_row)."""
    msg = np.asarray(msg, np.int64)
    toks, mot, first = [], [], []
    for k, (i, j) in enumerate(burst_bounds(msg)):
        m = message_from_rows(msg[i:j], int(qslot[i]))
        t = encode(m, mid_floor(l10[i, 3], l10[i, 5]))
        toks.append(t)
        mot.append(np.full(len(t), k, np.int64))
        first.append(i)
    return np.concatenate(toks), np.concatenate(mot), np.asarray(first, np.int64)


# ---------------------------------------------------------------- decode
def _take(toks, i: int, lo: int, n: int, what: str) -> int:
    if i >= len(toks):
        raise DecodeError(f"message ends before {what}")
    v = int(toks[i]) - lo
    if not 0 <= v < n:
        raise DecodeError(f"token {int(toks[i])} at {i} is not a {what}")
    return v


def decode(toks, mid: int) -> Message:
    """Strict inverse of encode. Checks structure only; whether the message
    can happen in a given book is rules.check's job."""
    evt = _take(toks, 0, EVT0, 4, "event") + 1
    d = _take(toks, 1, DIR0, 2, "direction")
    v = [_take(toks, 2 + k, TIME0, 1000, "time digit") for k in range(9)]
    dt_s, dt_ns = v[0], v[1] * 10**6 + v[2] * 10**3 + v[3]
    t_s, t_ns = v[4] * 1000 + v[5], v[6] * 10**6 + v[7] * 10**3 + v[8]
    i = 11

    def level(i):
        s = _take(toks, i, SIGN0, 2, "price sign")
        a = _take(toks, i + 1, PRICE0, 1000, "price digit") * 1000 + _take(toks, i + 2, PRICE0, 1000, "price digit")
        q = _take(toks, i + 3, SIZE0, 100, "size digit") * 100 + _take(toks, i + 4, SIZE0, 100, "size digit")
        return (mid + (a if s else -a) * TICK, q), i + 5

    levels = []
    qslot = SLOT_NONE
    if evt == 4:
        while i < len(toks) and int(toks[i]) != EOE:
            lv, i = level(i)
            levels.append(lv)
        if i >= len(toks):
            raise DecodeError("execution without EOE")
        if not levels:
            raise DecodeError("execution with no level")
        i += 1
    else:
        lv, i = level(i)
        levels.append(lv)
        if evt in (2, 3):
            if i >= len(toks):
                raise DecodeError("message ends before QSLOT")
            qslot = SLOT_NA if int(toks[i]) == NA else _take(toks, i, QSLOT0, N_QSLOT, "QSLOT")
            i += 1
    if i != len(toks):
        raise DecodeError(f"{len(toks) - i} tokens after the end of the message")
    return Message(evt, d, dt_s, dt_ns, t_s, t_ns, tuple(levels), qslot)
