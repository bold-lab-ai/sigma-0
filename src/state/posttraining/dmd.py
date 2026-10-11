"""DMD: distribution matching distillation — make the student look like the teacher.

No reward anywhere in this file.  The move:

    1. The STUDENT rolls out its own sequence causally (Self-Forcing: block by
       block, each block conditioned on its own previous output, so training
       visits the student's own mistakes, not teacher-fed prefixes).
    2. The whole rollout is noised to a random level t.
    3. Two score estimates at (x_t, t):  the frozen TEACHER says how the data
       distribution denoises; the FAKE critic — trained online by plain flow
       matching on the student's own outputs — says how the student
       distribution denoises.
    4. The student descends along (s_fake - s_real): mass moves from where the
       student over-produces toward where the teacher says real sequences live.
    5. The fake critic is refreshed on the fresh (detached) rollouts.

If the teacher itself pushes books through the ladder, the student will learn
the same violation — distribution matching cannot create a constraint the
teacher does not carry.  For "must not shatter" on top of "look like the
teacher", combine with a reward method: ``DMDRTrainer`` below adds the ReFL
term on the same rollout (matching keeps the student on the teacher manifold,
the reward keeps it out of the shattered corner).

Scores come from velocities via the conversion in ``flow.score_from_v``; the
score difference is applied through a stop-gradient pseudo-loss whose gradient
w.r.t. the rollout is exactly (s_fake - s_real), with the DMD2-style
per-sample normaliser  1 / mean|x_gen - x0_teacher|  to keep the scale sane.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..flow import (
    ModelFn,
    RectifiedFlow,
    fm_loss,
    interpolate,
    score_from_v,
    velocity_target,
    x0_from_v,
)


class DMDTrainer:
    """Distillation step for a few-step student against a frozen teacher.

    student_fn      trainable few-step generator's velocity model
    teacher_fn      frozen pretrained velocity model (the real-score oracle)
    fake_fn         trainable critic velocity model tracking the student's
                    output distribution (updated here, by FM on rollouts)
    flow            sampler settings for the student rollout
    student_steps   how few steps the student uses to generate (distillation
                    target regime, e.g. 4)
    n_blocks        Self-Forcing: blocks rolled out sequentially, each
                    conditioned on the previous block's output via ``rebind``
    t_range         noise levels where matching is enforced
    """

    def __init__(
        self,
        student_fn: ModelFn,
        teacher_fn: ModelFn,
        fake_fn: ModelFn,
        flow: RectifiedFlow,
        student_steps: int = 4,
        n_blocks: int = 1,
        t_range: tuple[float, float] = (0.02, 0.98),
        rebind: Optional[Callable[[torch.Tensor], ModelFn]] = None,
    ):
        if n_blocks > 1 and rebind is None:
            raise ValueError(
                "multi-block Self-Forcing needs rebind(prev_block) -> ModelFn "
                "to condition each block on the student's own previous output"
            )
        self.student_fn = student_fn
        self.teacher_fn = teacher_fn
        self.fake_fn = fake_fn
        self.flow = flow
        self.student_steps = student_steps
        self.n_blocks = n_blocks
        self.t_range = t_range
        self.rebind = rebind

    def _student_rollout(self, fn: ModelFn, shape, device, generator=None) -> torch.Tensor:
        """Few-step Euler WITH gradients — the student is the thing training."""
        x = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(1.0, 0.0, self.student_steps + 1, device=device)
        for i in range(self.student_steps):
            t_val = float(ts[i].item())
            dt = float((ts[i] - ts[i + 1]).item())
            t = torch.full((shape[0],), t_val, device=device, dtype=x.dtype)
            x = x - dt * fn(x, t)
        return x

    def rollout(self, shape, device, generator=None) -> torch.Tensor:
        """Causal Self-Forcing rollout: n_blocks blocks, each fed its own past."""
        blocks = []
        fn = self.student_fn
        for _ in range(self.n_blocks):
            block = self._student_rollout(fn, shape, device, generator=generator)
            blocks.append(block)
            if self.rebind is not None:
                # The next block conditions on generated (not real) history;
                # gradients stay within each block, the exposure is what matters.
                fn = self.rebind(block.detach())
        return torch.cat(blocks, dim=1) if len(blocks) > 1 else blocks[0]

    def step(
        self,
        frame_shape: tuple,
        device: torch.device,
        batch_size: int = 8,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        shape = (batch_size, *frame_shape)
        x_gen = self.rollout(shape, device, generator=generator)

        lo, hi = self.t_range
        t = lo + (hi - lo) * torch.rand(batch_size, device=device)
        eps = torch.randn(x_gen.shape, device=device, generator=generator)
        x_t = interpolate(x_gen, eps, t)

        # 3. two denoisers look at the same corrupted rollout
        with torch.no_grad():
            v_real = self.teacher_fn(x_t, t)
            v_fake = self.fake_fn(x_t, t)
            s_real = score_from_v(x_t, v_real, t)
            s_fake = score_from_v(x_t, v_fake, t)
            grad = s_fake - s_real  # direction that moves student mass toward real
            # DMD2 normaliser: scale by the teacher's own reconstruction error
            denom = (x_gen - x0_from_v(x_t, v_real, t)).abs().mean(
                dim=tuple(range(1, x_gen.dim())), keepdim=True
            ).clamp_min(1e-4)

        # 4. pseudo-loss whose x_gen-gradient is (s_fake - s_real) / denom
        loss_student = (x_gen * (grad / denom)).mean()

        # 5. critic refresh: plain FM on the detached student outputs
        x_gen_d = x_gen.detach()
        t_c = torch.rand(batch_size, device=device)
        eps_c = torch.randn(x_gen_d.shape, device=device, generator=generator)
        x_tc = interpolate(x_gen_d, eps_c, t_c)
        loss_fake = fm_loss(self.fake_fn(x_tc, t_c), velocity_target(x_gen_d, eps_c))

        return {
            "loss_student": loss_student,
            "loss_fake": loss_fake,
            "score_gap": grad.abs().mean(),
            "x_gen": x_gen,
        }


class DMDRTrainer(DMDTrainer):
    """DMD + reward: matching makes it look like the teacher, reward keeps the
    cup unbroken.  The reward term is the ReFL form — differentiable reward on
    the student's own rollout — sharing one rollout with the matching term.
    """

    def __init__(
        self,
        *args,
        reward: Callable[[torch.Tensor], torch.Tensor],
        decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reward_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reward = reward
        self.decode = decode if decode is not None else (lambda x: x)
        self.reward_weight = reward_weight

    def step(self, frame_shape, device, batch_size: int = 8, generator=None) -> dict:
        out = super().step(frame_shape, device, batch_size=batch_size, generator=generator)
        r = self.reward(self.decode(out["x_gen"]))
        out["reward_mean"] = r.detach().mean()
        out["loss_student"] = out["loss_student"] - self.reward_weight * r.mean()
        return out
