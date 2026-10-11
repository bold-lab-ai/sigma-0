"""Flow-GRPO / DanceGRPO: group-relative policy optimisation on the denoising chain.

The move (Seedance-2.0-lineage):

    1. For ONE conditioning context, sample a GROUP of G trajectories.  The ODE
       is first turned into a marginal-preserving SDE (``RectifiedFlow`` with
       noise_scale > 0) — without injected noise the G rollouts collapse onto
       nearly one path and the group advantage has no variance to learn from.
    2. Score each finished sequence with a scalar reward.  The reward may be a
       black box: a rule ("the final frame still holds a valid book" = 1/0),
       a human-preference model, anything — no gradient flows through it.
    3. Group-normalise:  A_i = (r_i - mean_G) / (std_G + eps).
    4. For every kept transition of every trajectory, push  A_i * log
       p_theta(x_{t-dt} | x_t)  through a PPO-style clipped ratio against the
       behaviour policy that produced the rollout.

Credit assignment is sequence-level: the reward is one scalar for the whole
clip, and every step of a good trajectory is reinforced equally — which frame
"started shattering" is not attributed (that refinement is the stepwise
variant, out of scope here).  ``timestep_fraction`` subsamples the chain to cut
the G * num_steps likelihood bill, the standard Flow-GRPO economy.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from ..flow import ModelFn, PinFn, RectifiedFlow, SdeStepRecord


class FlowGRPOTrainer:
    """One optimisation step = G stochastic rollouts + clipped policy gradient.

    model_fn           trainable policy (conditioning bound by the caller)
    flow               must carry noise_scale > 0 (exploration SDE)
    reward             black-box ``reward(frames) -> (B,)``, higher better
    decode             optional map from model space to reward space
    group_size         G rollouts per context
    clip_range         PPO ratio clip epsilon
    timestep_fraction  fraction of transitions that enter the loss
    frame_mask         (L,) or (B, L) 1 = generated frame; REQUIRED when a pin
                       conditions the rollout (pinned frames are not policy
                       output and must not enter the likelihood)
    """

    def __init__(
        self,
        model_fn: ModelFn,
        flow: RectifiedFlow,
        reward: Callable[[torch.Tensor], torch.Tensor],
        decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        group_size: int = 8,
        clip_range: float = 0.2,
        timestep_fraction: float = 0.25,
        adv_eps: float = 1e-4,
    ):
        if flow.noise_scale <= 0.0:
            raise ValueError(
                "Flow-GRPO needs a stochastic sampler: construct RectifiedFlow "
                "with noise_scale > 0, otherwise the group has no variance"
            )
        if group_size < 2:
            raise ValueError("group_size must be >= 2 to form an advantage")
        if not 0.0 < timestep_fraction <= 1.0:
            raise ValueError("timestep_fraction must be in (0, 1]")
        self.model_fn = model_fn
        self.flow = flow
        self.reward = reward
        self.decode = decode if decode is not None else (lambda x: x)
        self.group_size = group_size
        self.clip_range = clip_range
        self.timestep_fraction = timestep_fraction
        self.adv_eps = adv_eps

    @torch.no_grad()
    def rollout_group(
        self,
        frame_shape: tuple,
        device: torch.device,
        pin: Optional[PinFn] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, List[SdeStepRecord]]:
        """G trajectories under the CURRENT policy (the behaviour policy).

        frame_shape is (L, C); the group dimension becomes the batch:
        returns x0 (G, L, C) plus the per-step records of the shared rollout.
        """
        shape = (self.group_size, *frame_shape)
        return self.flow.sample_sde_trajectory(
            self.model_fn, shape, device, generator=generator, pin=pin
        )

    def step(
        self,
        frame_shape: tuple,
        device: torch.device,
        pin: Optional[PinFn] = None,
        frame_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        if pin is not None and frame_mask is None:
            raise ValueError(
                "a pinned rollout needs frame_mask so pinned frames stay out of "
                "the likelihood"
            )
        x0, records = self.rollout_group(frame_shape, device, pin=pin, generator=generator)

        # 2. black-box scalar per trajectory
        with torch.no_grad():
            r = self.reward(self.decode(x0))  # (G,)

        # 3. group-relative advantage
        adv = (r - r.mean()) / (r.std(unbiased=False) + self.adv_eps)

        # 4. clipped surrogate over a subsample of transitions
        n_keep = max(1, int(round(len(records) * self.timestep_fraction)))
        keep = torch.randperm(len(records), generator=generator)[:n_keep].tolist()

        mask = frame_mask
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(self.group_size, -1)

        losses, ratios = [], []
        for idx in keep:
            rec = records[idx]
            logp_new = self.flow.step_log_prob(self.model_fn, rec, frame_mask=mask)
            with torch.no_grad():
                logp_old = self.flow.step_log_prob(self.model_fn, rec, frame_mask=mask)
            # Behaviour policy == current policy on the first pass, so the raw
            # ratio starts at 1; the clip becomes active on reused rollouts.
            ratio = torch.exp(logp_new - logp_old)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
            losses.append(-torch.minimum(unclipped, clipped).mean())
            ratios.append(ratio.detach().mean())

        loss = torch.stack(losses).mean()
        return {
            "loss": loss,
            "reward_mean": r.mean(),
            "reward_std": r.std(unbiased=False),
            "advantage_abs": adv.abs().mean(),
            "ratio_mean": torch.stack(ratios).mean(),
            "kept_transitions": n_keep,
        }
