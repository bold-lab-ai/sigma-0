"""Quote both best prices, leaning with the signal and the inventory.

At each decision: a bid at the best bid and an ask at the best ask, size
`size`. A side is not quoted when the predicted return at `index` argues
against it by more than `threshold` bps (no ask when it predicts a rise, no
bid when it predicts a fall), or when a fill there would take the position
past +-cap. An order already resting at the price it should have is kept, so
it keeps its place in the queue; any other open order is cancelled.
"""

from __future__ import annotations

from post_training.policy_bench.types import BUY, SELL, Cancel, LimitOrder


class SkewQuoter:
    def __init__(self, index: int, size: int = 100, threshold: float = 0.5, cap: int = 1000) -> None:
        self.index, self.size, self.threshold, self.cap = index, size, threshold, cap

    def reset(self, episode) -> None:
        pass

    def step(self, obs) -> list:
        ask, bid = int(obs.l10[0]), int(obs.l10[2])
        s = float(obs.signal[self.index]) if obs.signal is not None else 0.0
        want = {}
        if s > -self.threshold and obs.position + self.size <= self.cap:
            want[BUY] = bid
        if s < self.threshold and obs.position - self.size >= -self.cap:
            want[SELL] = ask
        out, kept = [], set()
        for oid, side, price, _ in obs.open_orders:
            if want.get(side) == price and side not in kept:
                kept.add(side)
            else:
                out.append(Cancel(oid))
        out += [LimitOrder(side, price, self.size) for side, price in want.items() if side not in kept]
        return out
