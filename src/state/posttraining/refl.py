"""ReFL: backpropagate a differentiable reward through the one-jump prediction.

The move (Seedance-1.0-style reward feedback learning, ReFL):

    1. Roll the sampler from pure noise part of the way, WITHOUT gradients,
       to a late time t_mid (late in generation = small t in this package's
       convention, where t = 0 is data).
    2. Take ONE velocity evaluation with gradients and jump to the clean
       prediction:  x0_hat = x_t - t * v_theta(x_t, t).
    3. Score x0_hat with a differentiable reward and descend on -r.  The
       gradient enters through that single jump — the standard ReFL memory
       trade: the trajectory is treated as data, only the last hop learns.
    4. Mix in a little of the pretraining flow-matching loss so the model
       cannot collapse onto reward-pleasing garbage.

What the reward must contain, it gets: "the cup must not shatter" lives inside
``reward`` (validity, return sanity, or a learned scorer) — nothing here knows
about books.  If the reward only sees decoded frames and only late timesteps,
dynamics stay untrained; scoring the chart-space latent at arbitrary t (the
PRFL variant) is the same code with ``decode=None`` and a wider ``t_range``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..flow import ModelFn, RectifiedFlow, fm_loss, interpolate, velocity_target, x0_from_v


class ReFLTrainer:
    """One optimisation step = one reward rollout + optional FM anchor.

    model_fn   velocity model with conditioning already bound (see flow.ModelFn)
    flow       RectifiedFlow used for the no-grad prefix rollout
    reward     differentiable ``reward(frames) -> (B,)``; higher is better
    decode     optional map from model space to reward space (e.g. chart -> raw
               book).  Must be differentiable; identity when None.
    fm_weight  weight of the pretraining anchor computed on ``data_batch``
    t_range    where the graded jump happens; late in generation means small t
    """

    def __init__(
        self,
        model_fn: ModelFn,
        flow: RectifiedFlow,
        reward: Callable[[torch.Tensor], torch.Tensor],
        decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fm_weight: float = 0.1,
        t_range: tuple[float, float] = (0.05, 0.4),
    ):
        if not 0.0 <= t_range[0] < t_range[1] <= 1.0:
            raise ValueError(f"bad t_range {t_range}")
        self.model_fn = model_fn
        self.flow = flow
        self.reward = reward
        self.decode = decode if decode is not None else (lambda x: x)
        self.fm_weight = fm_weight
        self.t_range = t_range

    def _rollout_prefix(self, shape, device, t_stop: float, generator=None) -> torch.Tensor:
        """Euler from t = 1 down to t_stop, no gradients (trajectory as data)."""
        x = torch.randn(shape, device=device, generator=generator)
        ts = self.flow.timesteps(device)
        with torch.no_grad():
            for i in range(self.flow.num_steps):
                t_val = float(ts[i].item())
                if t_val <= t_stop:
                    break
                dt = float((ts[i] - ts[i + 1]).item())
                t = torch.full((shape[0],), t_val, device=device, dtype=x.dtype)
                x = x - dt * self.model_fn(x, t)
        return x

    def step(
        self,
        shape: tuple,
        device: torch.device,
        data_batch: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        lo, hi = self.t_range
        t_mid = float(torch.empty(1).uniform_(lo, hi, generator=None).item())
        x_t = self._rollout_prefix(shape, device, t_stop=t_mid, generator=generator)

        t = torch.full((shape[0],), t_mid, device=device, dtype=x_t.dtype)
        v = self.model_fn(x_t, t)  # the one graded evaluation
        x0_hat = x0_from_v(x_t, v, t)
        r = self.reward(self.decode(x0_hat))
        loss = -r.mean()

        anchor = torch.zeros((), device=device)
        if self.fm_weight > 0.0 and data_batch is not None:
            eps = torch.randn_like(data_batch)
            t_fm = torch.rand(data_batch.shape[0], device=device)
            x_fm = interpolate(data_batch, eps, t_fm)
            anchor = fm_loss(self.model_fn(x_fm, t_fm), velocity_target(data_batch, eps))
            loss = loss + self.fm_weight * anchor

        return {
            "loss": loss,
            "reward_mean": r.detach().mean(),
            "fm_anchor": anchor.detach(),
            "t_mid": t_mid,
        }
