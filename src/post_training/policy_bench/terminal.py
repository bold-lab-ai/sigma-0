"""What an open position is worth at the end of an episode.

Each rule returns (value, taker_shares): the value in LOBSTER units x shares,
and the shares the rule takes from the book (fees are charged on them). nan
when the end book cannot price the position.
"""

from __future__ import annotations

import numpy as np

from post_training.policy_bench.trace import Trace


class MarkToMid:
    def __call__(self, tr: Trace, position: int) -> tuple[float, float]:
        return (0.0 if position == 0 else position * float(tr.mid[-1])), 0.0


class MarkToMicro:
    """The size-weighted mid of the best levels: (ask x bid_q + bid x ask_q) / (ask_q + bid_q)."""

    def __call__(self, tr: Trace, position: int) -> tuple[float, float]:
        if position == 0:
            return 0.0, 0.0
        ask, aq, bid, bq = (float(x) for x in tr.l10_end[:4])
        if not (0 < bid < ask < 9e9) or aq + bq <= 0:
            return float("nan"), 0.0
        return position * (ask * bq + bid * aq) / (aq + bq), 0.0


class FlattenTaker:
    """Close the position against the end book, level by level (10 levels)."""

    def __call__(self, tr: Trace, position: int) -> tuple[float, float]:
        if position == 0:
            return 0.0, 0.0
        body = np.asarray(tr.l10_end, np.float64).reshape(10, 4)
        px, qty = (body[:, 2], body[:, 3]) if position > 0 else (body[:, 0], body[:, 1])
        left, value = abs(position), 0.0
        for p, q in zip(px, qty):
            if left == 0 or q <= 0 or not 0 < p < 9e9:
                break
            take = min(left, q)
            value += take * p
            left -= take
        if left > 0:
            return float("nan"), 0.0
        return float(np.sign(position) * value), float(abs(position))
