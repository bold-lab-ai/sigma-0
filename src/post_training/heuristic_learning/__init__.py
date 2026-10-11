"""Closed-loop market making driven by an LLM-as-optimizer.

Four parts form the loop, and keeping their responsibilities apart is what
makes the resulting PnL mean anything:

    policy       rule-based at-touch quoting code -- the thing being optimized
    world model  frozen checkpoints -- background order flow and a mid-price
                 curve; stochastic, learned
    simulator    the JAX matching engine -- price-time priority; deterministic
    optimizer    a headless LLM call that rewrites the policy from PnL plus
                 trajectory diagnostics; gradient-free, not RL, not evolution

The load-bearing separation is between the last two. The world model learns an
observational distribution p(market), but market making is an intervention,
p(market | do(my quotes)). Answering an interventional question with an
observational model is optimistic in a specific way: the generator was never
trained to punish you, so adverse selection is under-counted. The generator
therefore must never produce the fills of your own orders -- the deterministic
matching engine settles those -- and a real-data backtest stays the final judge.
"""

from post_training.heuristic_learning.checkpoints import (
    CheckpointDrift,
    CheckpointPin,
    PINS,
    resolve,
    verify,
)

__all__ = [
    "CheckpointDrift",
    "CheckpointPin",
    "PINS",
    "resolve",
    "verify",
]
