"""RAM — reward-adjusted matching: change the regression TARGET, not the weight.

The pipeline stays exactly the pretraining pipeline — sample, noise, regress —
which is what makes it scale; the reward never needs a gradient and there is no
per-step SDE likelihood.  The one change is where the regression points:

    1. The current model generates sequences x0_i (fast ODE, no gradients).
    2. A black-box reward scores each one; scores are min-max normalised
       inside the group to r_tilde in [0, 1].
    3. Each sequence is noised exactly as in pretraining: x_t = (1-t) x0 + t eps.
    4. The regression target is built from an ADJUSTED endpoint

           x0_adj_i = (1 - w_i) * x0_i + w_i * x0_plus,
           w_i      = beta * (1 - r_tilde_i),
           x0_plus  = softmax(r / temperature)-weighted group mean,

       so a shattered sequence no longer anchors the field at its own endpoint
       — its target is dragged toward where the good sequences ended — while a
       clean sequence keeps (almost) the plain pretraining target.
    5. Loss:  || v_theta(x_t, t) - (eps - x0_adj) ||^2,  target built no-grad.

Contrast with AWM (same rollout data, same rewards): AWM keeps the target and
multiplies the loss by a signed advantage; RAM keeps the plain unweighted loss
and rewrites the target's endpoint.  AWM can push away with a negative weight;
RAM only redirects — every target remains a convex combination of sampled
endpoints, so the regression never chases a point outside the data hull.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..flow import ModelFn, PinFn, RectifiedFlow, fm_loss, interpolate, velocity_target


class RAMTrainer:
    """One optimisation step = G ODE rollouts + reward-adjusted FM regression.

    beta in [0, 1] scales how far a zero-reward endpoint moves toward the
    good-group centroid (1 = all the way).  temperature controls how sharply
    the centroid concentrates on the best sequences.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        flow: RectifiedFlow,
        reward: Callable[[torch.Tensor], torch.Tensor],
        decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        group_size: int = 8,
        n_noise_draws: int = 4,
        beta: float = 1.0,
        temperature: float = 0.5,
    ):
        if group_size < 2:
            raise ValueError("group_size must be >= 2: the adjusted endpoint needs a group")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.model_fn = model_fn
        self.flow = flow
        self.reward = reward
        self.decode = decode if decode is not None else (lambda x: x)
        self.group_size = group_size
        self.n_noise_draws = n_noise_draws
        self.beta = beta
        self.temperature = temperature

    @torch.no_grad()
    def adjusted_endpoints(self, x0: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build x0_adj from raw endpoints and raw rewards (steps 2 and 4)."""
        span = (r.max() - r.min()).clamp_min(1e-8)
        r_tilde = (r - r.min()) / span  # (G,) in [0, 1]
        w = (self.beta * (1.0 - r_tilde)).view(-1, *([1] * (x0.dim() - 1)))
        good = torch.softmax(r / self.temperature, dim=0).view(-1, *([1] * (x0.dim() - 1)))
        x0_plus = (good * x0).sum(dim=0, keepdim=True)  # reward-weighted centroid
        return (1.0 - w) * x0 + w * x0_plus, r_tilde

    def step(
        self,
        frame_shape: tuple,
        device: torch.device,
        pin: Optional[PinFn] = None,
        frame_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        # 1. generate with the cheap deterministic sampler
        shape = (self.group_size, *frame_shape)
        x0 = self.flow.sample_ode(self.model_fn, shape, device, generator=generator, pin=pin)

        with torch.no_grad():
            r = self.reward(self.decode(x0))  # (G,) black box
        x0_adj, r_tilde = self.adjusted_endpoints(x0, r)

        mask = frame_mask
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(self.group_size, -1)

        # 3 + 5. pretraining-shaped regression toward the adjusted endpoints.
        # Noise the RAW endpoints (that is the state the sampler will actually
        # visit); only the target endpoint is adjusted.
        losses = []
        for _ in range(self.n_noise_draws):
            t = torch.rand(self.group_size, device=device)
            eps = torch.randn(x0.shape, device=device, generator=generator)
            x_t = interpolate(x0, eps, t)
            v_pred = self.model_fn(x_t, t)
            losses.append(
                fm_loss(v_pred, velocity_target(x0_adj, eps), frame_mask=mask)
            )
        loss = torch.stack(losses).mean()

        return {
            "loss": loss,
            "reward_mean": r.mean(),
            "endpoint_shift": (x0_adj - x0).abs().mean(),
            "r_tilde_mean": r_tilde.mean(),
        }
