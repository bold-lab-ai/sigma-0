"""rollout(policy, signal, market, episodes, every) -> one Trace per episode.

    for each episode:
        market.reset(ep); policy.reset(ep)
        until market.done():
            every `every` background messages: observe, add the signal, let the
            policy act, submit its orders (active after the market's latency)
            advance one background message

The open position is not closed here; score.py values it with a terminal rule.
"""

from __future__ import annotations

import numpy as np

from post_training.policy_bench.history import RowLog
from post_training.policy_bench.trace import Trace


def predict(signal, log: RowLog) -> np.ndarray:
    """The signal's predicted returns (bps, one per horizon) on the last rows of the log."""
    P = signal.prefill
    tok, book = signal.inputs(log.rows(P), log.l10(P + 1))
    return signal.predict(tok[None], book[None])[0]


def rollout(policy, signal, market, episodes, every: int = 64) -> list[Trace]:
    out = []
    for ep in episodes:
        market.reset(ep)
        policy.reset(ep)
        k = 0
        while not market.done():
            if k % every == 0:
                obs = market.observe()
                obs.signal = None if signal is None else predict(signal, market.log)
                market.record_decision(obs.signal)
                market.submit(policy.step(obs))
            market.advance()
            k += 1
        out.append(market.trace())
    return out
