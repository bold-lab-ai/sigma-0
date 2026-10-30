"""nano-sigma's PnL-bench rule as a policy: hold sign(predicted return) x size.

At every decision the target position is sign(signal[index]) x size, reached
with one market order. With a decision only at k = 0 and the episode as long as
the horizon, this is the PnL bench's bet: one bet per decision point, held h
messages.
"""

from __future__ import annotations

import numpy as np

from post_training.policy_bench.types import BUY, SELL, MarketOrder


class SignRule:
    def __init__(self, index: int, size: int = 1) -> None:
        self.index, self.size = index, size

    def reset(self, episode) -> None:
        pass

    def step(self, obs) -> list:
        target = int(np.sign(obs.signal[self.index])) * self.size
        d = target - obs.position
        if d == 0:
            return []
        return [MarketOrder(BUY if d > 0 else SELL, abs(d))]
