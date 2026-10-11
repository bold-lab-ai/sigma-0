"""Rewards for post-training the state model.

The text-to-video analogy that motivated this package ("a cup is pushed off the
table and does not shatter") maps onto the order book as three reward families:

    prompt adherence -> conditioning consistency (the future follows its context)
    physics          -> book validity (positive spread, monotone ladder, sizes > 0)
    aesthetics       -> statistical sanity (spread and return scales look like data)

Every reward here scores a *raw-coordinate* book frame sequence (4 columns per
level: ask_p, ask_s, bid_p, bid_s).  When the model works in the structured
chart representation, decode first — the chart's inverse map is differentiable
(exp / cumsum), so ReFL gradients pass through the decoder into the model.

Note on validity under the chart representation: the chart is valid *by
construction*, so ``BookValidityReward`` saturates at 1.0 there and carries no
gradient.  It has teeth for raw-coordinate models, and it is the correctness
anchor for rule-based scoring in GRPO / AWM / RAM when the representation does
not already enforce the constraint.  Statistical rewards do the work otherwise.

Two scoring modes per rule-based reward:

    hard (default)  non-differentiable rates in [0, 1]; use with GRPO/AWM/RAM,
                    which only need a scalar.
    soft            softplus hinges, differentiable; use with ReFL, which
                    backpropagates through the reward.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import nn

RewardFn = Callable[[torch.Tensor], torch.Tensor]
"""``reward(frames) -> (B,)`` with frames (B, L, C_raw) in raw coordinates."""


def split_book_columns(frames: torch.Tensor):
    """(B, L, 4*levels) -> ask_p, ask_s, bid_p, bid_s each (B, L, levels)."""
    if frames.shape[-1] % 4 != 0:
        raise ValueError(f"raw book width must be divisible by 4, got {frames.shape[-1]}")
    x = frames.view(*frames.shape[:-1], -1, 4)
    return x[..., 0], x[..., 1], x[..., 2], x[..., 3]


class BookValidityReward(nn.Module):
    """Structural validity: no crossed book, monotone ladders, positive sizes.

    Empty levels (size == 0 carries price 0 in LOBSTER-style books) are masked
    out of the ladder tests — a naive test misreads an empty level as a
    violation (the all-rows vs populated-rows distinction BookDiT documents).

    hard:  r = 1 - mean violation rate over frames         (flat regions, no grad)
    soft:  r = exp(-mean softplus hinge / temperature)     (differentiable)
    """

    def __init__(self, soft: bool = False, temperature: float = 1.0, price_scale: float = 100.0):
        super().__init__()
        self.soft = soft
        self.temperature = temperature
        self.price_scale = price_scale  # hinge slope normaliser, in price ticks

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        ask_p, ask_s, bid_p, bid_s = split_book_columns(frames)
        pop_a = ask_s > 0
        pop_b = bid_s > 0

        # Crossed / locked book at the touch (both sides populated).
        touch = pop_a[..., 0] & pop_b[..., 0]
        spread = ask_p[..., 0] - bid_p[..., 0]
        # Ladder steps between consecutive populated levels.
        step_a = ask_p[..., 1:] - ask_p[..., :-1]  # should be > 0
        step_b = bid_p[..., :-1] - bid_p[..., 1:]  # should be > 0
        mask_a = pop_a[..., 1:] & pop_a[..., :-1]
        mask_b = pop_b[..., 1:] & pop_b[..., :-1]

        if not self.soft:
            crossed = ((spread <= 0) & touch).float().sum(dim=-1)
            n_touch = touch.float().sum(dim=-1).clamp_min(1.0)
            bad_a = ((step_a <= 0) & mask_a).float().sum(dim=(-2, -1))
            bad_b = ((step_b <= 0) & mask_b).float().sum(dim=(-2, -1))
            n_steps = (mask_a.float().sum(dim=(-2, -1)) + mask_b.float().sum(dim=(-2, -1))).clamp_min(1.0)
            neg_size = ((ask_s < 0).float() + (bid_s < 0).float()).sum(dim=(-2, -1))
            n_cells = 2.0 * ask_s.shape[-1] * ask_s.shape[-2]
            rate = crossed / n_touch + (bad_a + bad_b) / n_steps + neg_size / n_cells
            return (1.0 - rate / 3.0).clamp(0.0, 1.0)

        sp = nn.functional.softplus
        h_cross = (sp(-spread / self.price_scale) * touch.float()).sum(dim=-1) / touch.float().sum(dim=-1).clamp_min(1.0)
        h_ladd = (
            (sp(-step_a / self.price_scale) * mask_a.float()).sum(dim=(-2, -1))
            + (sp(-step_b / self.price_scale) * mask_b.float()).sum(dim=(-2, -1))
        ) / (mask_a.float().sum(dim=(-2, -1)) + mask_b.float().sum(dim=(-2, -1))).clamp_min(1.0)
        h_size = (sp(-ask_s) + sp(-bid_s)).mean(dim=(-2, -1))
        return torch.exp(-(h_cross + h_ladd + h_size) / self.temperature)


class ReturnSanityReward(nn.Module):
    """Per-frame mid moves must stay inside a plausible band for the clock.

    On a 1 ms frame clock a mid log-return above ``max_abs_return`` in one frame
    is the cup shattering: not impossible in data, but the tail the model should
    not manufacture.  Differentiable by construction (soft hinge only).
    """

    def __init__(self, max_abs_return: float = 0.005, temperature: float = 1.0):
        super().__init__()
        self.max_abs_return = max_abs_return
        self.temperature = temperature

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        ask_p, _, bid_p, _ = split_book_columns(frames)
        mid = 0.5 * (ask_p[..., 0] + bid_p[..., 0]).clamp_min(1e-6)
        ret = torch.log(mid[:, 1:]) - torch.log(mid[:, :-1])
        excess = nn.functional.softplus(ret.abs() - self.max_abs_return)
        return torch.exp(-excess.mean(dim=-1) / self.temperature)


class SpreadBandReward(nn.Module):
    """Spread should live inside a data-calibrated band [lo, hi] (price units)."""

    def __init__(self, lo: float, hi: float, temperature: float = 1.0):
        super().__init__()
        if not lo < hi:
            raise ValueError("need lo < hi")
        self.lo, self.hi = lo, hi
        self.temperature = temperature

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        ask_p, _, bid_p, _ = split_book_columns(frames)
        spread = ask_p[..., 0] - bid_p[..., 0]
        scale = self.hi - self.lo
        below = nn.functional.softplus((self.lo - spread) / scale)
        above = nn.functional.softplus((spread - self.hi) / scale)
        return torch.exp(-(below + above).mean(dim=-1) / self.temperature)


class CompositeReward(nn.Module):
    """Weighted geometric mean of component rewards (each already in (0, 1])."""

    def __init__(self, components: Sequence[nn.Module], weights: Optional[Sequence[float]] = None):
        super().__init__()
        self.components = nn.ModuleList(components)
        n = len(components)
        if n == 0:
            raise ValueError("need at least one reward component")
        self.weights = list(weights) if weights is not None else [1.0] * n
        if len(self.weights) != n:
            raise ValueError("weights and components disagree in length")

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(frames.shape[0], device=frames.device, dtype=frames.dtype)
        wsum = sum(self.weights)
        for w, comp in zip(self.weights, self.components):
            r = comp(frames).clamp_min(1e-8)
            total = total + w * torch.log(r)
        return torch.exp(total / max(wsum, 1e-8))


def default_rule_reward(soft: bool = False) -> CompositeReward:
    """The 'cup survives' bundle: validity + return sanity, equally weighted.

    Calibrate ``SpreadBandReward`` from data before adding it; its band is
    corpus-specific, so it is not part of the default.
    """
    return CompositeReward([BookValidityReward(soft=soft), ReturnSanityReward()])
