"""Plain data passed between the policy, the market, rollout and score.

Prices are LOBSTER integers (TICK = 100 is one cent, 10,000 is one dollar).
Sides are those of book.py: BUY takes from the asks and rests on the bid
(book side BID); SELL the other way round.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BUY, SELL = 1, -1


@dataclass(frozen=True)
class MarketOrder:
    side: int        # BUY or SELL
    size: int


@dataclass(frozen=True)
class LimitOrder:
    side: int
    price: int
    size: int


@dataclass(frozen=True)
class Cancel:
    oid: int         # an agent order id from Obs.open_orders


Order = MarketOrder | LimitOrder | Cancel


@dataclass(frozen=True)
class Fill:
    k: int           # background messages advanced before the fill
    side: int        # BUY or SELL, from the agent's side
    price: float     # LOBSTER units; MidFill can give half a unit
    size: int
    maker: bool
    mid: float       # mid of the book when the fill happened


@dataclass(frozen=True)
class Episode:
    entry: dict      # a manifest entry (ticker, date, month, message, orderbook, fullbook_dir, ...)
    start: int       # first row after the history: the decision at k = 0 sees rows < start
    seed: int = 0


@dataclass
class Obs:
    k: int
    t_s: int
    t_ns: int
    l10: np.ndarray                  # (40,) book at k: ask_p, ask_q, bid_p, bid_q per level
    position: int                    # shares
    cash: float                      # LOBSTER units x shares
    open_orders: list = field(default_factory=list)
    fills: list = field(default_factory=list)       # since the previous observation
    signal: np.ndarray | None = None                # predicted returns per horizon, bps


def mid_of(l10: np.ndarray) -> float | None:
    """Mid of an L10 body (40 columns), or None unless 0 < bid < ask (both sides present)."""
    ask, bid = int(l10[0]), int(l10[2])
    if not 0 < bid < ask < 9_000_000_000:
        return None
    return 0.5 * (ask + bid)
