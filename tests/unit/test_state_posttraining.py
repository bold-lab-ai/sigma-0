"""The four post-training methods: one smoke step each, plus their contracts."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from state.dit import FrameDiT, FrameDiTConfig
from state.flow import RectifiedFlow
from state.posttraining import (
    AWMTrainer,
    DMDRTrainer,
    DMDTrainer,
    FlowGRPOTrainer,
    RAMTrainer,
    ReFLTrainer,
)
from state.rewards import BookValidityReward, ReturnSanityReward, default_rule_reward

LEVELS = 2
C = 4 * LEVELS  # tiny chart width
SHAPE = (6, C)  # (frames, channels)
DEV = torch.device("cpu")


def tiny_model(seed: int = 0) -> FrameDiT:
    torch.manual_seed(seed)
    cfg = FrameDiTConfig(channels=C, hidden=32, depth=1, heads=2, max_seq_len=16, conditional=False)
    model = FrameDiT(cfg)
    # Perturb the zero-init head so rollouts are not the identity on noise.
    with torch.no_grad():
        model.velocity_head.weight.add_(0.01 * torch.randn_like(model.velocity_head.weight))
    return model


def grad_flows(model: torch.nn.Module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def spread_reward(frames: torch.Tensor) -> torch.Tensor:
    """A smooth differentiable stand-in reward on chart-space tensors."""
    sig = torch.sigmoid(frames[..., 0])
    return sig.mean(dim=tuple(range(1, sig.dim())))


# --------------------------------------------------------------------- rewards


def _book_tensor(crossed: bool) -> torch.Tensor:
    book = torch.zeros(1, 4, 4 * LEVELS)
    ask0, bid0 = (99.0, 101.0) if crossed else (101.0, 99.0)
    for k in range(LEVELS):
        book[..., 4 * k] = ask0 + 2 * k
        book[..., 4 * k + 1] = 10.0
        book[..., 4 * k + 2] = bid0 - 2 * k
        book[..., 4 * k + 3] = 10.0
    return book


def test_validity_reward_separates_valid_from_crossed():
    r = BookValidityReward()
    assert r(_book_tensor(crossed=False)).item() == pytest.approx(1.0)
    assert r(_book_tensor(crossed=True)).item() < 1.0


def test_soft_validity_is_differentiable():
    book = _book_tensor(crossed=True).requires_grad_(True)
    r = BookValidityReward(soft=True)(book)
    r.sum().backward()
    assert book.grad is not None and torch.isfinite(book.grad).all()


def test_return_sanity_penalises_jumps():
    calm = _book_tensor(crossed=False)
    wild = calm.clone()
    wild[:, 2:, 0] += 500.0  # the mid teleports mid-sequence
    wild[:, 2:, 2] += 500.0
    r = ReturnSanityReward()
    assert r(wild).item() < r(calm).item()


def test_default_bundle_composes():
    assert 0.0 < default_rule_reward()(_book_tensor(crossed=False)).item() <= 1.0


# ------------------------------------------------------------------------ ReFL


def test_refl_step_backpropagates_reward():
    model = tiny_model()
    trainer = ReFLTrainer(model, RectifiedFlow(num_steps=6), spread_reward, fm_weight=0.1)
    out = trainer.step(shape=(4, *SHAPE), device=DEV, data_batch=torch.randn(4, *SHAPE))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert grad_flows(model)


# ------------------------------------------------------------------- Flow-GRPO


def test_grpo_requires_a_stochastic_sampler():
    model = tiny_model()
    with pytest.raises(ValueError):
        FlowGRPOTrainer(model, RectifiedFlow(num_steps=6, noise_scale=0.0), spread_reward)


def test_grpo_step_learns_and_zero_variance_reward_is_neutral():
    model = tiny_model()
    flow = RectifiedFlow(num_steps=6, noise_scale=0.3)
    trainer = FlowGRPOTrainer(model, flow, spread_reward, group_size=4, timestep_fraction=0.5)
    out = trainer.step(frame_shape=SHAPE, device=DEV)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert grad_flows(model)

    # A constant reward gives zero advantage: nothing to reinforce.
    flat = FlowGRPOTrainer(
        model, flow, lambda f: torch.ones(f.shape[0]), group_size=4, timestep_fraction=0.5
    )
    out = flat.step(frame_shape=SHAPE, device=DEV)
    assert out["loss"].abs().item() == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------------- AWM


def test_awm_step_learns_without_sde():
    model = tiny_model()
    trainer = AWMTrainer(model, RectifiedFlow(num_steps=6), spread_reward, group_size=4, n_noise_draws=2)
    out = trainer.step(frame_shape=SHAPE, device=DEV)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert grad_flows(model)


def test_awm_fm_anchor_enters_the_loss():
    model = tiny_model()
    flow = RectifiedFlow(num_steps=6)
    data = torch.randn(4, *SHAPE)
    flat_reward = lambda f: torch.ones(f.shape[0])  # zero advantage isolates the anchor
    plain = AWMTrainer(model, flow, flat_reward, group_size=4)
    anchored = AWMTrainer(model, flow, flat_reward, group_size=4, fm_anchor_weight=0.5)
    out_plain = plain.step(frame_shape=SHAPE, device=DEV, data_batch=data)
    out_anch = anchored.step(frame_shape=SHAPE, device=DEV, data_batch=data)
    # With zero advantage the weighted term vanishes; only the anchor remains.
    assert out_plain["loss"].abs().item() == pytest.approx(0.0, abs=1e-6)
    assert out_anch["fm_anchor"].item() > 0.0
    assert out_anch["loss"].item() == pytest.approx(
        0.5 * out_anch["fm_anchor"].item(), rel=1e-4
    )


def test_awm_constant_reward_is_neutral():
    model = tiny_model()
    trainer = AWMTrainer(
        model, RectifiedFlow(num_steps=6), lambda f: torch.ones(f.shape[0]), group_size=4
    )
    out = trainer.step(frame_shape=SHAPE, device=DEV)
    assert out["loss"].abs().item() == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------------- RAM


def test_ram_adjusts_only_bad_endpoints():
    model = tiny_model()
    trainer = RAMTrainer(model, RectifiedFlow(num_steps=6), spread_reward, group_size=4, beta=1.0)
    x0 = torch.randn(4, *SHAPE)
    r = torch.tensor([0.1, 0.5, 0.9, 1.0])
    adj, r_tilde = trainer.adjusted_endpoints(x0, r)
    # The best sequence keeps its own endpoint exactly (w = 0).
    assert torch.allclose(adj[3], x0[3], atol=1e-6)
    # The worst sequence moves the most.
    shifts = (adj - x0).abs().mean(dim=(1, 2))
    assert shifts[0] > shifts[1] > shifts[2]

    zero_beta = RAMTrainer(model, RectifiedFlow(num_steps=6), spread_reward, group_size=4, beta=0.0)
    adj0, _ = zero_beta.adjusted_endpoints(x0, r)
    assert torch.allclose(adj0, x0, atol=1e-6)


def test_ram_step_learns():
    model = tiny_model()
    trainer = RAMTrainer(model, RectifiedFlow(num_steps=6), spread_reward, group_size=4, n_noise_draws=2)
    out = trainer.step(frame_shape=SHAPE, device=DEV)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert grad_flows(model)


# ------------------------------------------------------------------------- DMD


def test_dmd_moves_student_not_teacher():
    student, teacher, fake = tiny_model(0), tiny_model(1), tiny_model(2)
    teacher.requires_grad_(False)
    trainer = DMDTrainer(
        student_fn=student, teacher_fn=teacher, fake_fn=fake,
        flow=RectifiedFlow(num_steps=6), student_steps=2,
    )
    out = trainer.step(frame_shape=SHAPE, device=DEV, batch_size=3)
    assert torch.isfinite(out["loss_student"]) and torch.isfinite(out["loss_fake"])
    (out["loss_student"] + out["loss_fake"]).backward()
    assert grad_flows(student) and grad_flows(fake)
    assert all(p.grad is None for p in teacher.parameters())


def test_dmd_multiblock_needs_rebind():
    student, teacher, fake = tiny_model(0), tiny_model(1), tiny_model(2)
    with pytest.raises(ValueError):
        DMDTrainer(student, teacher, fake, RectifiedFlow(num_steps=4), n_blocks=2)


def test_dmdr_adds_the_reward_term():
    student, teacher, fake = tiny_model(0), tiny_model(1), tiny_model(2)
    base = DMDTrainer(student, teacher, fake, RectifiedFlow(num_steps=4), student_steps=2)
    torch.manual_seed(42)
    g = torch.Generator().manual_seed(9)
    out_base = base.step(frame_shape=SHAPE, device=DEV, batch_size=3, generator=g)
    with_r = DMDRTrainer(
        student, teacher, fake, RectifiedFlow(num_steps=4), student_steps=2,
        reward=spread_reward, reward_weight=1.0,
    )
    torch.manual_seed(42)
    g = torch.Generator().manual_seed(9)
    out_r = with_r.step(frame_shape=SHAPE, device=DEV, batch_size=3, generator=g)
    assert "reward_mean" in out_r
    assert not torch.allclose(out_base["loss_student"], out_r["loss_student"])
