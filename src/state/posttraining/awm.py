"""AWM — advantage-weighted matching: GRPO's data, pretraining's loss.

Same inputs as Flow-GRPO (a group of G sampled sequences, one scalar reward
each, group-normalised advantages), different objective:

    1. Do NOT differentiate the per-step transition likelihood.
    2. Take each sampled clean sequence x0_i, noise it exactly the way
       pretraining does, and compute the plain flow-matching regression
       ||v_theta - (eps - x0_i)||^2 at a handful of noise levels.
    3. Multiply by A_i.  Positive-advantage sequences pull the velocity field
       toward their trajectories; negative-advantage sequences flip the sign
       and push the field away from theirs.

Compared with GRPO this removes the two expensive commitments: no SDE rollout
is required (the group can be sampled with the fast deterministic ODE), and the
loss is the pretraining objective itself, so it does not fight the flow-matching
geometry.  The reward is still "shattered or not" — it just enters through the
weight instead of through a per-step likelihood.

K (``n_noise_draws``) is small by design: training touches each sequence at a
few noise levels only, while sampling still runs the full ODE schedule.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..flow import ModelFn, PinFn, RectifiedFlow, fm_loss, interpolate, velocity_target


class AWMTrainer:
    """One optimisation step = G ODE rollouts + advantage-weighted FM loss.

    adv_clip bounds |A_i|: an unlucky near-zero group std otherwise turns one
    trajectory into the whole gradient.  frame_mask limits both noising and
    loss to generated frames when rollouts are context-conditioned.

    fm_anchor_weight mixes in the plain pretraining loss on REAL windows
    (pass ``data_batch`` to ``step``).  Measured necessity, not decoration:
    on a full-day GOOG run the unanchored version peaked at step ~50 and then
    collapsed — group reward std fell 0.0046 -> 0.0005 (all G rollouts nearly
    identical) and the reward gave back its gain; with anchor 0.25 the same
    300 steps rose monotonically with group std intact.  Reward-only
    optimisation walks off the data manifold; the anchor holds the model on
    it while the advantages shift probability mass within it.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        flow: RectifiedFlow,
        reward: Callable[[torch.Tensor], torch.Tensor],
        decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        group_size: int = 8,
        n_noise_draws: int = 4,
        adv_clip: float = 3.0,
        adv_eps: float = 1e-4,
        fm_anchor_weight: float = 0.0,
    ):
        if group_size < 2:
            raise ValueError("group_size must be >= 2 to form an advantage")
        if n_noise_draws < 1:
            raise ValueError("n_noise_draws must be >= 1")
        self.model_fn = model_fn
        self.flow = flow
        self.reward = reward
        self.decode = decode if decode is not None else (lambda x: x)
        self.group_size = group_size
        self.n_noise_draws = n_noise_draws
        self.adv_clip = adv_clip
        self.adv_eps = adv_eps
        self.fm_anchor_weight = fm_anchor_weight

    def step(
        self,
        frame_shape: tuple,
        device: torch.device,
        pin: Optional[PinFn] = None,
        frame_mask: Optional[torch.Tensor] = None,
        data_batch: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        # 1. sample the group with the cheap deterministic sampler
        shape = (self.group_size, *frame_shape)
        x0 = self.flow.sample_ode(self.model_fn, shape, device, generator=generator, pin=pin)

        with torch.no_grad():
            r = self.reward(self.decode(x0))  # (G,)
        adv = (r - r.mean()) / (r.std(unbiased=False) + self.adv_eps)
        adv = adv.clamp(-self.adv_clip, self.adv_clip)

        mask = frame_mask
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(self.group_size, -1)

        # 2-3. pretraining objective on the sampled sequences, weighted by A_i
        losses = []
        for _ in range(self.n_noise_draws):
            t = torch.rand(self.group_size, device=device)
            eps = torch.randn(x0.shape, device=device, generator=generator)
            x_t = interpolate(x0, eps, t)
            v_pred = self.model_fn(x_t, t)
            losses.append(
                fm_loss(v_pred, velocity_target(x0, eps), weight=adv, frame_mask=mask)
            )
        loss = torch.stack(losses).mean()

        anchor = torch.zeros((), device=device)
        if self.fm_anchor_weight > 0.0 and data_batch is not None:
            t_a = torch.rand(data_batch.shape[0], device=device)
            eps_a = torch.randn(data_batch.shape, device=device, generator=generator)
            x_a = interpolate(data_batch, eps_a, t_a)
            anchor = fm_loss(self.model_fn(x_a, t_a), velocity_target(data_batch, eps_a))
            loss = loss + self.fm_anchor_weight * anchor

        return {
            "loss": loss,
            "reward_mean": r.mean(),
            "reward_std": r.std(unbiased=False),
            "advantage_abs": adv.abs().mean(),
            "fm_anchor": anchor.detach(),
        }
