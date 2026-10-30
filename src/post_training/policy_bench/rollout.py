"""rollout(policy, signal, market, episodes, every) -> one Trace per episode.

A market runs `market.batch` episodes at once (ReplayMarket one, SimMarket
several); each gets its own copy of the policy.

    for each wave of episodes:
        market.reset(wave); each policy copy .reset(episode)
        until market.done():
            every `every` background messages: observe every row, add the
            signal (one batch over the rows), let each policy act, submit its
            orders (active after the market's latency)
            advance every row by one background message

The open position is not closed here; score.py values it with a terminal rule.
"""

from __future__ import annotations

import copy

import numpy as np

from post_training.policy_bench.history import RowLog
from post_training.policy_bench.trace import Trace


def predict(signal, logs: list[RowLog], micro: int = 4) -> np.ndarray:
    """The signal's predicted returns (rows, horizons) in bps on the last rows of each log,
    `micro` rows per forward (nano-sigma's return_features.py micro batch)."""
    P = signal.prefill
    out = []
    for lo in range(0, len(logs), micro):
        x = [signal.inputs(log.rows(P), log.l10(P + 1)) for log in logs[lo:lo + micro]]
        out.append(signal.predict(np.stack([t for t, _ in x]), np.stack([b for _, b in x])))
    return np.concatenate(out)


def rollout(policy, signal, market, episodes, every: int = 64) -> list[Trace]:
    out: list[Trace] = []
    B = getattr(market, "batch", 1)
    for w in range(0, len(episodes), B):
        wave = list(episodes[w:w + B])
        market.reset(wave)
        policies = [copy.deepcopy(policy) for _ in wave]
        for p, ep in zip(policies, wave):
            p.reset(ep)
        k = 0
        while not market.done():
            if k % every == 0:
                obs = [market.observe(i) for i in range(len(wave))]
                sig = None if signal is None else predict(signal, [market.log(i) for i in range(len(wave))])
                for i, o in enumerate(obs):
                    o.signal = None if sig is None else sig[i]
                    market.record_decision(i, o.signal)
                    market.submit(i, policies[i].step(o))
            market.advance()
            k += 1
        out += [market.trace(i) for i in range(len(wave))]
    return out
