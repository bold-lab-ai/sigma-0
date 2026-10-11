"""State-model pretraining: conditional flow matching on frame windows.

The loop fixes the three defects the sigma-flow autopsy identified as the
reasons that codebase never trained properly:

    1. context length is drawn PER SAMPLE, not one scalar per batch;
    2. the FM loss is masked to generated frames — a context frame's velocity
       is analytically recoverable from its own clean value, so leaving it in
       the loss fills half the objective with trivially solvable terms;
    3. normalisation is not optional: the trainer fits it on the training days
       and stores it inside every checkpoint.

Checkpoints: saved every ``save_every`` steps and at the end, with a
``latest_checkpoint.json`` breadcrumb written atomically next to them; resume
reads the breadcrumb (never a directory listing).  Keep ``--out`` on
node-local storage during cluster runs and copy back afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Optional

import torch

from .dataset import EncodedDay, FrameWindowDataset, encode_series, fit_normalizer, load_wide_book_day
from .dit import FrameDiT, FrameDiTConfig, bind_condition
from .flow import fm_loss, interpolate, velocity_target
from .frames import DEFAULT_INTERVAL_NS
from .representation import ChartNormalizer


def sample_context_masks(
    batch: int, seq_len: int, max_ctx: int, device: torch.device, generator=None
) -> torch.Tensor:
    """Per-sample prefix lengths ctx_i ~ U{0..max_ctx} -> obs_mask (B, L)."""
    if not 0 <= max_ctx < seq_len:
        raise ValueError("need 0 <= max_ctx < seq_len")
    ctx = torch.randint(0, max_ctx + 1, (batch,), device=device, generator=generator)
    pos = torch.arange(seq_len, device=device).unsqueeze(0)
    return (pos < ctx.unsqueeze(1)).float()


def pretrain_step(
    model: FrameDiT,
    x0: torch.Tensor,
    max_ctx: int,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """One conditional-FM step on clean windows x0 (B, L, C)."""
    B, L, _ = x0.shape
    obs = sample_context_masks(B, L, max_ctx, x0.device, generator=generator)
    model_fn = bind_condition(model, x0, obs)

    t = torch.rand(B, device=x0.device, generator=generator)
    eps = torch.randn(x0.shape, device=x0.device, generator=generator)
    x_t = interpolate(x0, eps, t)
    v_pred = model_fn(x_t, t)
    loss = fm_loss(v_pred, velocity_target(x0, eps), frame_mask=1.0 - obs)
    return {"loss": loss, "ctx_mean": obs.sum(dim=1).mean()}


def save_checkpoint(
    out_dir: str,
    step: int,
    model: FrameDiT,
    opt: torch.optim.Optimizer,
    normalizer: Optional[ChartNormalizer],
    cfg: FrameDiTConfig,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{step:08d}.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "normalizer": normalizer.state_dict() if normalizer is not None else None,
            "config": vars(cfg),
        },
        path,
    )
    # Atomic breadcrumb; resume reads this file, never a directory listing.
    crumb = {"path": path, "step": step}
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(crumb, f)
    os.replace(tmp, os.path.join(out_dir, "latest_checkpoint.json"))
    return path


def load_checkpoint(out_dir: str, model: FrameDiT, opt: Optional[torch.optim.Optimizer]) -> int:
    crumb_path = os.path.join(out_dir, "latest_checkpoint.json")
    if not os.path.exists(crumb_path):
        return 0
    with open(crumb_path) as f:
        crumb = json.load(f)
    state = torch.load(crumb["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if opt is not None and state.get("optimizer") is not None:
        opt.load_state_dict(state["optimizer"])
    return int(state["step"])


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True, help="wide-book npy files (43-col contract)")
    p.add_argument("--out", required=True, help="checkpoint directory")
    p.add_argument("--interval-ns", type=int, default=DEFAULT_INTERVAL_NS,
                   help="frame clock; default 1 ms per frame")
    p.add_argument("--seq-len", type=int, default=512, help="frames per window")
    p.add_argument("--max-ctx", type=int, default=256, help="max context prefix, frames")
    p.add_argument("--min-gap", type=float, default=1.0,
                   help="chart log floor; MUST match the data's price unit")
    p.add_argument("--activity-channel", action="store_true",
                   help="append log1p(events per frame) as a channel")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    series = [load_wide_book_day(f, interval_ns=args.interval_ns) for f in args.data]
    normalizer = fit_normalizer(series, min_gap=args.min_gap, with_activity=args.activity_channel)
    days = [
        encode_series(s, normalizer, min_gap=args.min_gap, with_activity=args.activity_channel)
        for s in series
    ]
    dataset = FrameWindowDataset(days, seq_len=args.seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    channels = days[0].chart.shape[1]
    cfg = FrameDiTConfig(
        channels=channels, hidden=args.hidden, depth=args.depth, heads=args.heads,
        max_seq_len=max(args.seq_len, 512), conditional=True,
    )
    model = FrameDiT(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_step = load_checkpoint(args.out, model, opt)
    if start_step:
        print(f"[resume] continuing from step {start_step}")

    step = start_step
    while step < args.steps:
        for window, _anchor in loader:
            if step >= args.steps:
                break
            x0 = window.to(device)
            out = pretrain_step(model, x0, max_ctx=args.max_ctx)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}  loss {out['loss'].item():.4f}  ctx {out['ctx_mean'].item():.0f}")
            if step % args.save_every == 0:
                save_checkpoint(args.out, step, model, opt, normalizer, cfg)
    save_checkpoint(args.out, step, model, opt, normalizer, cfg)
    print(f"[done] {step} steps; checkpoints in {args.out}")


if __name__ == "__main__":
    main()
