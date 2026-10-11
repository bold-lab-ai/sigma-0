"""State-model core: frame clock, chart validity, flow identities, backbone."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from state.dit import FrameDiT, FrameDiTConfig, bind_condition, bind_unconditional
from state.flow import (
    RectifiedFlow,
    interpolate,
    velocity_target,
    x0_from_v,
    eps_from_v,
    score_from_v,
)
from state.frames import resample_events
from state.representation import decode_frames, decode_frames_torch, encode_frames
from state.train import pretrain_step, sample_context_masks


def _valid_book(T: int, levels: int, seed: int = 0) -> np.ndarray:
    """A random but structurally valid raw book (T, 4*levels)."""
    rng = np.random.default_rng(seed)
    mid = 10000.0 + np.cumsum(rng.integers(-5, 6, size=T)) * 1.0
    spread = rng.integers(2, 20, size=T).astype(float)
    book = np.zeros((T, 4 * levels))
    ask = mid + spread / 2
    bid = mid - spread / 2
    for k in range(levels):
        if k > 0:
            ask = ask + rng.integers(1, 10, size=T)
            bid = bid - rng.integers(1, 10, size=T)
        book[:, 4 * k] = ask
        book[:, 4 * k + 1] = rng.integers(1, 500, size=T)
        book[:, 4 * k + 2] = bid
        book[:, 4 * k + 3] = rng.integers(1, 500, size=T)
    return book


# ---------------------------------------------------------------- frame clock


def test_resample_forward_fills_and_counts_events():
    times = np.array([100, 1_200_000, 1_300_000, 5_600_000], dtype=np.int64)
    states = np.arange(4, dtype=np.float64)[:, None]
    fs = resample_events(times, states, interval_ns=1_000_000)
    # Grid: 1ms, 2ms, 3ms, 4ms, 5ms, and the last event lands after 5ms.
    assert fs.times_ns[0] == 1_000_000 and fs.interval_ns == 1_000_000
    # 1ms frame: last event at or before 1ms is the first (t=100).
    assert fs.frames[0, 0] == 0
    # 2ms frame folds the 1.2ms and 1.3ms events; state is the later one.
    assert fs.frames[1, 0] == 2
    assert fs.events_per_frame[1] == 2
    # Quiet frames repeat the previous state (forward fill), folding nothing.
    assert fs.frames[2, 0] == 2 and fs.events_per_frame[2] == 0
    # Every event is folded into exactly one frame, except any after the grid end.
    assert fs.events_per_frame.sum() == np.searchsorted(times, fs.times_ns[-1], side="right")


def test_resample_horizon_is_wall_time():
    times = np.arange(0, 10_000_000, 37_000, dtype=np.int64)  # bursty-ish
    states = np.zeros((len(times), 1))
    fs = resample_events(times, states, interval_ns=1_000_000)
    assert fs.horizon_ns == len(fs.frames) * 1_000_000


def test_resample_rejects_disorder():
    with pytest.raises(ValueError):
        resample_events(np.array([5, 3], dtype=np.int64), np.zeros((2, 1)))


# ------------------------------------------------------------- representation


def test_chart_round_trip_is_exact_on_prices_and_sizes():
    book = _valid_book(T=64, levels=10)
    chart, anchor = encode_frames(book, min_gap=1.0)
    back = decode_frames(chart, anchor)
    np.testing.assert_allclose(back, book, rtol=0, atol=1e-3)


def test_chart_decode_of_noise_is_a_valid_book():
    rng = np.random.default_rng(1)
    noise_chart = rng.normal(size=(32, 40)).astype(np.float32)
    book = decode_frames(noise_chart, anchor_mid=10000.0)
    ask_p = book[:, [4 * k for k in range(10)]]
    bid_p = book[:, [4 * k + 2 for k in range(10)]]
    assert (ask_p[:, 0] > bid_p[:, 0]).all()  # never crossed
    assert (np.diff(ask_p, axis=1) > 0).all()  # ask ladder strictly up
    assert (np.diff(bid_p, axis=1) < 0).all()  # bid ladder strictly down


def test_torch_decode_matches_numpy_and_is_differentiable():
    book = _valid_book(T=16, levels=4, seed=2)
    chart, anchor = encode_frames(book)
    ct = torch.tensor(chart).unsqueeze(0).requires_grad_(True)
    out = decode_frames_torch(ct, torch.tensor([anchor]))
    np.testing.assert_allclose(
        out.detach().numpy()[0], decode_frames(chart, anchor), rtol=0, atol=1e-2
    )
    out.sum().backward()
    assert ct.grad is not None and torch.isfinite(ct.grad).all()


def test_activity_channel_appends():
    book = _valid_book(T=8, levels=2)
    chart, _ = encode_frames(book, events_per_frame=np.arange(8))
    assert chart.shape[1] == 9
    np.testing.assert_allclose(chart[:, -1], np.log1p(np.arange(8)), atol=1e-6)


# ------------------------------------------------------------------ flow core


def test_conversion_identities():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 8, 5, generator=g)
    eps = torch.randn(4, 8, 5, generator=g)
    t = torch.rand(4, generator=g) * 0.8 + 0.1
    x_t = interpolate(x0, eps, t)
    v = velocity_target(x0, eps)
    assert torch.allclose(x0_from_v(x_t, v, t), x0, atol=1e-5)
    assert torch.allclose(eps_from_v(x_t, v, t), eps, atol=1e-5)
    tb = t.view(-1, 1, 1)
    assert torch.allclose(score_from_v(x_t, v, t), -eps / tb, atol=1e-4)


def test_sde_with_zero_noise_degenerates_to_ode():
    model = lambda x, t: 0.3 * x  # a linear velocity field
    ode = RectifiedFlow(num_steps=8, noise_scale=0.0)
    g1 = torch.Generator().manual_seed(7)
    x_ode = ode.sample_ode(model, (2, 4, 3), torch.device("cpu"), generator=g1)
    g2 = torch.Generator().manual_seed(7)
    x_sde, records = ode.sample_sde_trajectory(model, (2, 4, 3), torch.device("cpu"), generator=g2)
    assert torch.allclose(x_ode, x_sde, atol=1e-6)
    assert all(r.noise_std == 0.0 for r in records)


def test_step_log_prob_requires_noise_and_backpropagates():
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 3)
    model = lambda x, t: lin(x)
    stoch = RectifiedFlow(num_steps=6, noise_scale=0.3)
    x, records = stoch.sample_sde_trajectory(model, (2, 4, 3), torch.device("cpu"))
    noisy = [r for r in records if r.noise_std > 0]
    assert noisy, "the exploration schedule produced no stochastic transition"
    logp = stoch.step_log_prob(model, noisy[0])
    assert torch.isfinite(logp).all()
    logp.sum().backward()
    assert lin.weight.grad is not None

    det = RectifiedFlow(num_steps=6, noise_scale=0.0)
    x, records = det.sample_sde_trajectory(model, (1, 4, 3), torch.device("cpu"))
    with pytest.raises(ValueError):
        det.step_log_prob(model, records[0])


# ------------------------------------------------------------------- backbone


def test_framedit_parameter_account_matches_bookdit_champion():
    # BookDiT's published account for width 128 / depth 4 / heads 4 / 41
    # channels is exactly 953,769; the backbone is a port, so it must agree.
    cfg = FrameDiTConfig(channels=41, hidden=128, depth=4, heads=4, conditional=False)
    model = FrameDiT(cfg)
    assert sum(p.numel() for p in model.parameters()) == 953_769


def test_conditional_columns_start_dead():
    torch.manual_seed(0)
    cfg_u = FrameDiTConfig(channels=8, hidden=32, depth=1, heads=2, conditional=False)
    cfg_c = FrameDiTConfig(channels=8, hidden=32, depth=1, heads=2, conditional=True)
    uncond, cond = FrameDiT(cfg_u), FrameDiT(cfg_c)
    # Copy shared weights; the conditional model's extra input columns are zero.
    with torch.no_grad():
        cond.input_proj.weight[:, :8] = uncond.input_proj.weight
        cond.input_proj.bias.copy_(uncond.input_proj.bias)
        for pu, pc in zip(uncond.parameters(), cond.parameters()):
            if pu.shape == pc.shape:
                pc.copy_(pu)
    x = torch.randn(2, 6, 8)
    t = torch.rand(2)
    got = bind_unconditional(cond)(x, t)
    want = uncond(x, t)
    assert torch.allclose(got, want, atol=1e-5)


def test_zero_init_head_outputs_zero():
    cfg = FrameDiTConfig(channels=6, hidden=32, depth=2, heads=2, conditional=False)
    model = FrameDiT(cfg)
    out = model(torch.randn(2, 5, 6), torch.rand(2))
    assert torch.allclose(out, torch.zeros_like(out))


def test_rope_rejects_overflow_and_uses_fp32_cache():
    cfg = FrameDiTConfig(channels=4, hidden=16, depth=1, heads=2, max_seq_len=8, conditional=False)
    model = FrameDiT(cfg)
    rope = model.blocks[0].attn.rope
    assert rope.cos.dtype == torch.float32  # the sigma-flow bf16 collapse fix
    with pytest.raises(ValueError):
        model(torch.randn(1, 9, 4), torch.rand(1))


# ----------------------------------------------------------------- pretraining


def test_pretrain_step_masks_context_and_learns():
    torch.manual_seed(0)
    cfg = FrameDiTConfig(channels=8, hidden=32, depth=1, heads=2, conditional=True)
    model = FrameDiT(cfg)
    x0 = torch.randn(4, 12, 8)
    out = pretrain_step(model, x0, max_ctx=6)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_context_masks_are_prefixes_and_per_sample():
    g = torch.Generator().manual_seed(3)
    obs = sample_context_masks(64, 10, 8, torch.device("cpu"), generator=g)
    assert obs.shape == (64, 10)
    diffs = obs[:, 1:] - obs[:, :-1]
    assert (diffs <= 0).all()  # once generation starts it never reverts
    assert obs.sum(dim=1).unique().numel() > 1  # lengths genuinely vary
