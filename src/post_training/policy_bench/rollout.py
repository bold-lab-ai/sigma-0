"""rollout(policy, signal, market, episodes, every) -> one Trace per episode.

A market has `market.batch` slots (ReplayMarket one, SimMarket several). Each
slot runs one episode with its own copy of the policy; when its episode ends,
the slot takes the next one (continuous batching), so no slot waits for a
slower one.

    load the first `batch` episodes
    until every episode has ended:
        market.run(every): rows move on until one waits, at a decision
        (every `every` background messages) or at its end
        an ended slot: keep its trace, load the next episode
        a waiting slot: observe it, add the signal, let its policy act,
        submit the orders (active after the market's latency)

The signal reads one row per forward, so a prediction does not depend on
which rows happen to decide together. The open position is not closed here;
score.py values it with a terminal rule.
"""

from __future__ import annotations

import copy

import numpy as np

from post_training.policy_bench.history import RowLog
from post_training.policy_bench.trace import Trace


def predict(signal, logs: list[RowLog], micro: int = 1) -> np.ndarray:
    """The signal's predicted returns (rows, horizons) in bps on the last rows of each log,
    `micro` rows per forward."""
    P = signal.prefill
    out = []
    for lo in range(0, len(logs), micro):
        x = [signal.inputs(log.rows(P), log.l10(P + 1)) for log in logs[lo:lo + micro]]
        out.append(signal.predict(np.stack([t for t, _ in x]), np.stack([b for _, b in x])))
    return np.concatenate(out)


def rollout(policy, signal, market, episodes, every: int = 64) -> list[Trace]:
    out: list[Trace | None] = [None] * len(episodes)
    todo = list(range(len(episodes)))[::-1]
    slots: dict[int, tuple[int, object]] = {}          # slot -> (episode index, policy copy)

    def start(i: int) -> None:
        if not todo:
            slots.pop(i, None)
            market.unload(i)
            return
        j = todo.pop()
        market.load(i, episodes[j])
        p = copy.deepcopy(policy)
        p.reset(episodes[j])
        slots[i] = (j, p)

    for i in range(min(getattr(market, "batch", 1), len(episodes))):
        start(i)
    while slots:
        waiting = market.run(every)
        for i in waiting:
            if market.finished(i):
                out[slots[i][0]] = market.trace(i)
                start(i)
        decide = [i for i in waiting if i in slots and not market.finished(i)]
        if not decide:
            continue
        sig = None if signal is None else predict(signal, [market.log(i) for i in decide])
        for n, i in enumerate(decide):
            o = market.observe(i)
            o.signal = None if sig is None else sig[n]
            market.record_decision(i, o.signal)
            market.submit(i, slots[i][1].step(o))
    return out
