"""FrameDiT: the state-model backbone — WanDiT block, mask-concat conditioning.

A deliberate merge of the two reference implementations, one decision each:

    backbone      BookDiT's WanDiT: per-block modulation table + shared 6-way
                  time projection, qk-RMSNorm, bidirectional attention,
                  zero-init velocity head.  Chosen because its parameter
                  account is published (953,769 at width 128 / depth 4 /
                  41 channels) and asserted in this package's tests.
    conditioning  sigma-flow's V2 mask-concat: the input is
                  cat([x_t, cond, mask]) with cond = clean context frames and
                  mask 0 = given / 1 = generate.  Chosen because it is the one
                  path in either reference with zero train/inference gap — no
                  RePaint pinning needed at sampling time, and the conditioning
                  columns of the input projection start at zero (the
                  SDXL-inpaint trick) so pretraining from an unconditional
                  checkpoint is a no-op at step 0.

Fixes carried from the sigma-flow autopsy (things that made it not work):

    * RoPE cache is fp32, non-persistent, sized from max_seq_len, indexed with
      an integer arange — under bf16 the original collapsed 8192 positions
      onto 769 distinct values and silently destroyed long-context attention;
    * timesteps enter in the model dtype but the sinusoid is computed in fp32.

Sequence conditioning enters ONLY through the concat channels; the flow time t
conditions through AdaLN.  There is no class token and no cross-attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FrameDiTConfig:
    channels: int = 41  # chart channels per frame
    hidden: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 4.0
    max_seq_len: int = 4096
    freq_dim: int = 256
    conditional: bool = True  # mask-concat input (2C+1) vs plain (C)

    @property
    def head_dim(self) -> int:
        if self.hidden % self.heads != 0:
            raise ValueError("hidden must divide by heads")
        return self.hidden // self.heads

    @property
    def input_dim(self) -> int:
        return 2 * self.channels + 1 if self.conditional else self.channels


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t*1000, cos-first (the convention both
    references share), computed in fp32."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.to(torch.float32).mul(1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)


class RotaryEmbedding(nn.Module):
    """1-D RoPE with an fp32 cache sized up front (the sigma-flow fix)."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        angles = pos[:, None] * inv_freq[None]  # (S, head_dim/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.max_seq_len = max_seq_len

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, heads, L, head_dim), interleaved rotation, fp32 math."""
        L = x.shape[-2]
        if L > self.max_seq_len:
            raise ValueError(f"sequence length {L} exceeds RoPE cache {self.max_seq_len}")
        cos = self.cos[:L].to(torch.float32)
        sin = self.sin[:L].to(torch.float32)
        xf = x.to(torch.float32)
        x_even, x_odd = xf[..., 0::2], xf[..., 1::2]
        out = torch.stack(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        ).flatten(-2)
        return out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class _Attention(nn.Module):
    def __init__(self, cfg: FrameDiTConfig, rope: RotaryEmbedding):
        super().__init__()
        H, hd = cfg.hidden, cfg.head_dim
        self.q = nn.Linear(H, H)
        self.k = nn.Linear(H, H)
        self.v = nn.Linear(H, H)
        self.o = nn.Linear(H, H)
        self.q_norm = RMSNorm(hd)
        self.k_norm = RMSNorm(hd)
        self.rope = rope
        self.heads = cfg.heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        hd = H // self.heads

        def heads_view(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.heads, hd).transpose(1, 2)

        q = self.rope.rotate(self.q_norm(heads_view(self.q(x))))
        k = self.rope.rotate(self.k_norm(heads_view(self.k(x))))
        v = heads_view(self.v(x))
        out = F.scaled_dot_product_attention(q, k, v)  # bidirectional
        return self.o(out.transpose(1, 2).reshape(B, L, H))


class _WanBlock(nn.Module):
    """Pre-norm block with a learned 6-way modulation table added to the
    shared time embedding — BookDiT's conditioning at 768 params per block
    instead of a per-block Linear(H, 6H)."""

    def __init__(self, cfg: FrameDiTConfig, rope: RotaryEmbedding):
        super().__init__()
        H = cfg.hidden
        self.norm1 = nn.LayerNorm(H, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(H, elementwise_affine=False)
        self.attn = _Attention(cfg, rope)
        ffn_dim = int(H * cfg.mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(H, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, H)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, H) / H**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        m = (self.modulation + e).chunk(6, dim=1)  # six (B, 1, H)
        x = x + m[2] * self.attn(self.norm1(x) * (1 + m[1]) + m[0])
        x = x + m[5] * self.ffn(self.norm2(x) * (1 + m[4]) + m[3])
        return x


class FrameDiT(nn.Module):
    """Velocity model over frame sequences: forward(x, t) -> v.

    x  (B, L, input_dim)  — plain chart frames, or cat([x_t, cond, mask])
    t  (B,)               — flow time in [0, 1]
    returns (B, L, channels)
    """

    def __init__(self, cfg: FrameDiTConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        self.input_proj = nn.Linear(cfg.input_dim, H)
        self.time_embedding = nn.Sequential(
            nn.Linear(cfg.freq_dim, H), nn.SiLU(), nn.Linear(H, H)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(H, 6 * H))
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        self.blocks = nn.ModuleList(_WanBlock(cfg, rope) for _ in range(cfg.depth))
        self.final_norm = nn.LayerNorm(H)
        self.velocity_head = nn.Linear(H, cfg.channels)

        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)
        if cfg.conditional:
            # The conditioning columns start dead so an unconditional
            # checkpoint's behaviour is reproduced exactly at load time.
            with torch.no_grad():
                self.input_proj.weight[:, cfg.channels :].zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tok = self.input_proj(x)
        e = self.time_embedding(timestep_embedding(t, self.cfg.freq_dim))
        e = self.time_projection(e).view(-1, 6, self.cfg.hidden)
        for blk in self.blocks:
            tok = blk(tok, e)
        return self.velocity_head(self.final_norm(tok))


def bind_condition(model: FrameDiT, context: torch.Tensor, obs_mask: torch.Tensor):
    """Close a FrameDiT over its conditioning -> a flow.ModelFn.

    context   (B, L, C) clean frames; only positions with obs_mask = 1 are read
    obs_mask  (B, L) 1 = given context, 0 = to generate
    The flow-side mask channel is the complement (1 = generate), matching the
    sigma-flow convention.
    """
    if not model.cfg.conditional:
        raise ValueError("model was built with conditional=False")
    obs = obs_mask.unsqueeze(-1).to(context.dtype)
    cond = context * obs
    gen_mask = 1.0 - obs

    def fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(torch.cat([x_t, cond, gen_mask], dim=-1), t)

    return fn


def bind_unconditional(model: FrameDiT):
    """A ModelFn with empty conditioning (mask = generate everywhere)."""
    if not model.cfg.conditional:
        return model.__call__

    def fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = torch.zeros_like(x_t)
        mask = torch.ones(*x_t.shape[:2], 1, device=x_t.device, dtype=x_t.dtype)
        return model(torch.cat([x_t, cond, mask], dim=-1), t)

    return fn
