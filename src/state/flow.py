"""Rectified-flow core shared by state-model pretraining and every post-training method.

One convention, used everywhere in ``src/state`` (matching sigma-flow's FlowDiT and
BookDiT's trainer, so numbers and checkpoints translate):

    t = 0 is clean data, t = 1 is pure noise
    x_t = (1 - t) * x0 + t * eps,        eps ~ N(0, I)
    v*  = eps - x0                       (the velocity target; dx_t/dt = v*)
    sampling integrates from t = 1 down to t = 0 with  x <- x - dt * v

Three conversion identities follow from the interpolation.  With
a = E[x0 | x_t], b = E[eps | x_t] and v = b - a (what the network predicts):

    x_t = (1 - t) a + t b        (definition of the interpolant)
    =>  a = x_t - t * v          (predicted clean state,  ``x0_from_v``)
    =>  b = x_t + (1 - t) * v    (predicted noise,        ``eps_from_v``)
    score(x_t) = -b / t          (Tweedie for sigma_t = t, ``score_from_v``)

Every post-training file uses one or more of these:

    ReFL       needs ``x0_from_v``      (one-jump prediction to score with a reward)
    Flow-GRPO  needs ``sde_step``       (a Gaussian transition with a tractable log-prob)
    AWM        needs ``fm_loss``        (per-sample weights on the plain FM objective)
    RAM        needs ``fm_loss``        (a reward-adjusted regression target)
    DMD        needs ``score_from_v``   (teacher/fake score difference)

The SDE used for exploration (Flow-GRPO) keeps the same marginals as the ODE.
For the rectified-flow path the marginal-preserving reverse SDE is

    dx = [ v_theta(x, t) - (sigma_t^2 / 2) * score(x, t) ] dt + sigma_t dW,   t: 1 -> 0

and substituting ``score_from_v`` gives the drift used in ``sde_step``.  As
sigma_t -> 0 the update reduces exactly to the Euler ODE step; a unit test pins
that degeneration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

# The score conversion divides by t; below this floor the SDE falls back to the
# deterministic update instead of amplifying float noise by 1/t.
_T_FLOOR = 1e-4

ModelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``model_fn(x_t, t) -> v`` with x_t (B, L, C) and t (B,) in [0, 1].

Conditioning (history frames, prompts) is bound inside the callable by the
caller, so the flow core stays architecture-agnostic.
"""


def interpolate(x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) x0 + t eps, with t broadcast from (B,) to x0's shape."""
    t = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1.0 - t) * x0 + t * eps


def velocity_target(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """The rectified-flow regression target v* = eps - x0."""
    return eps - x0


def x0_from_v(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """One-jump clean-state prediction: x0_hat = x_t - t * v."""
    t = t.view(-1, *([1] * (x_t.dim() - 1)))
    return x_t - t * v


def eps_from_v(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Predicted noise: eps_hat = x_t + (1 - t) * v."""
    t = t.view(-1, *([1] * (x_t.dim() - 1)))
    return x_t + (1.0 - t) * v


def score_from_v(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Marginal score via Tweedie: score = -eps_hat / t (t floored for stability)."""
    t = t.view(-1, *([1] * (x_t.dim() - 1))).clamp_min(_T_FLOOR)
    return -(x_t + (1.0 - t) * v) / t


def fm_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flow-matching MSE with optional per-sample weights and per-frame mask.

    weight      (B,)    per-sample scalar (AWM advantages; may be negative).
    frame_mask  (B, L)  1 where the loss counts (horizon frames), 0 elsewhere
                        (context frames are conditioning, not targets).
    """
    err = (v_pred - v_target).pow(2)  # (B, L, C)
    if frame_mask is not None:
        mask = frame_mask.unsqueeze(-1).to(err.dtype)
        per_sample = (err * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
    else:
        per_sample = err.mean(dim=tuple(range(1, err.dim())))
    if weight is not None:
        per_sample = per_sample * weight.to(per_sample.dtype)
    return per_sample.mean()


@dataclass
class SdeStepRecord:
    """Everything needed to recompute one transition's log-prob with gradients."""

    x_t: torch.Tensor  # state before the step (B, L, C)
    x_s: torch.Tensor  # state after the step  (B, L, C)
    t: torch.Tensor  # scalar time before the step, shape (B,)
    dt: float  # positive step size (t decreases by dt)
    noise_std: float  # sigma_t * sqrt(dt) actually used (0 => deterministic)


class RectifiedFlow:
    """Sampling and log-prob machinery around a velocity model.

    ``noise_scale`` a controls the exploration schedule sigma_t = a * sqrt(t / (1 - t))
    (clipped near t = 1), the schedule used by Flow-GRPO.  a = 0 recovers the ODE.
    """

    def __init__(self, num_steps: int = 50, noise_scale: float = 0.0):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        self.num_steps = num_steps
        self.noise_scale = noise_scale

    # -- schedules -----------------------------------------------------------

    def timesteps(self, device: torch.device) -> torch.Tensor:
        """Uniform grid from 1 to 0 inclusive, num_steps + 1 points."""
        return torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

    def sigma(self, t: float) -> float:
        """Exploration noise level sigma_t; finite at t -> 1 by clipping."""
        if self.noise_scale == 0.0:
            return 0.0
        t = min(max(t, 0.0), 1.0 - 1e-3)
        return self.noise_scale * math.sqrt(t / (1.0 - t))

    # -- one transition ------------------------------------------------------

    def drift(self, model_fn: ModelFn, x_t: torch.Tensor, t: torch.Tensor, sigma_t: float) -> torch.Tensor:
        """Reverse-time drift v - (sigma^2/2) * score; equals v when sigma = 0."""
        v = model_fn(x_t, t)
        if sigma_t == 0.0:
            return v
        score = score_from_v(x_t, v, t)
        return v - 0.5 * (sigma_t**2) * score

    def sde_step(
        self,
        model_fn: ModelFn,
        x_t: torch.Tensor,
        t_val: float,
        dt: float,
        generator: Optional[torch.Generator] = None,
    ) -> SdeStepRecord:
        """One Euler-Maruyama step from t to t - dt (both in [0, 1], dt > 0).

        The transition is Gaussian:
            x_s ~ N( x_t - dt * drift(x_t, t),  sigma_t^2 * dt * I )
        which is what makes the Flow-GRPO likelihood ratio tractable.
        """
        B = x_t.shape[0]
        t = torch.full((B,), t_val, device=x_t.device, dtype=x_t.dtype)
        sigma_t = self.sigma(t_val) if t_val > _T_FLOOR else 0.0
        with torch.no_grad():
            mean = x_t - dt * self.drift(model_fn, x_t, t, sigma_t)
        noise_std = sigma_t * math.sqrt(dt)
        if noise_std > 0.0:
            noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
            x_s = mean + noise_std * noise
        else:
            x_s = mean
        return SdeStepRecord(x_t=x_t, x_s=x_s, t=t, dt=dt, noise_std=noise_std)

    def step_log_prob(
        self,
        model_fn: ModelFn,
        record: SdeStepRecord,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """log p_theta(x_s | x_t) for a recorded transition, differentiable in theta.

        Recomputes the mean through the current parameters (the rollout used the
        behaviour policy; this is the trainable side of the GRPO ratio).  Returns
        a (B,) sum of per-dimension Gaussian log-densities over unmasked frames.
        """
        if record.noise_std <= 0.0:
            raise ValueError(
                "step_log_prob needs a stochastic transition; rollout with noise_scale > 0"
            )
        t_val = float(record.t[0].item())
        sigma_t = self.sigma(t_val)
        mean = record.x_t - record.dt * self.drift(model_fn, record.x_t, record.t, sigma_t)
        var = record.noise_std**2
        logp = -0.5 * ((record.x_s - mean).pow(2) / var + math.log(2.0 * math.pi * var))
        if frame_mask is not None:
            logp = logp * frame_mask.unsqueeze(-1).to(logp.dtype)
        return logp.sum(dim=tuple(range(1, logp.dim())))

    # -- full trajectories ---------------------------------------------------

    @torch.no_grad()
    def sample_ode(
        self,
        model_fn: ModelFn,
        shape: tuple,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        pin: Optional["PinFn"] = None,
    ) -> torch.Tensor:
        """Plain Euler integration from noise to data; the production sampler."""
        x = torch.randn(shape, device=device, generator=generator)
        ts = self.timesteps(device)
        for i in range(self.num_steps):
            t_val = float(ts[i].item())
            dt = float((ts[i] - ts[i + 1]).item())
            t = torch.full((shape[0],), t_val, device=device, dtype=x.dtype)
            x = x - dt * model_fn(x, t)
            if pin is not None:
                x = pin(x, float(ts[i + 1].item()))
        return x

    def sample_sde_trajectory(
        self,
        model_fn: ModelFn,
        shape: tuple,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        pin: Optional["PinFn"] = None,
    ) -> tuple[torch.Tensor, List[SdeStepRecord]]:
        """Stochastic rollout that keeps every transition for GRPO log-probs.

        When ``pin`` re-imposes conditioning frames after each transition, the
        pinned coordinates are no longer draws from the Gaussian transition, so
        any later ``step_log_prob`` on these records must mask them out (pass
        the complementary frame mask).
        """
        x = torch.randn(shape, device=device, generator=generator)
        ts = self.timesteps(device)
        records: List[SdeStepRecord] = []
        for i in range(self.num_steps):
            t_val = float(ts[i].item())
            dt = float((ts[i] - ts[i + 1]).item())
            rec = self.sde_step(model_fn, x, t_val, dt, generator=generator)
            x = rec.x_s
            if pin is not None:
                x = pin(x, float(ts[i + 1].item()))
                rec = SdeStepRecord(
                    x_t=rec.x_t, x_s=x, t=rec.t, dt=rec.dt, noise_std=rec.noise_std
                )
            records.append(rec)
        return x, records


PinFn = Callable[[torch.Tensor, float], torch.Tensor]
"""``pin(x, t_next) -> x`` re-imposes known frames after a sampler step.

The RePaint-style conditioner: replace observed frames by their forward-noised
true values at level t_next, leaving generated frames untouched."""


def make_repaint_pin(
    x_obs: torch.Tensor,
    obs_mask: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> PinFn:
    """Standard inpainting conditioner over frames.

    x_obs     (B, L, C) true values wherever obs_mask is 1 (rest ignored)
    obs_mask  (B, L) or (B, L, 1); 1 = conditioning frame, 0 = generated frame
    """
    if obs_mask.dim() == 2:
        obs_mask = obs_mask.unsqueeze(-1)
    obs_mask = obs_mask.to(x_obs.dtype)

    def pin(x: torch.Tensor, t_next: float) -> torch.Tensor:
        noise = torch.randn(x_obs.shape, device=x_obs.device, dtype=x_obs.dtype, generator=generator)
        x_obs_t = (1.0 - t_next) * x_obs + t_next * noise
        return obs_mask * x_obs_t + (1.0 - obs_mask) * x

    return pin
