"""The newtok simulator run one chunk at a time, with a KV cache.

`NanoSigmaNewTok.forward(inp, msg_idx, book)` reads a whole window at once. A
rollout feeds tokens as they are decided (forced prefill and agent messages,
then one sampled token at a time) and throws away a rejected message, so the
same weights are run here with a cache that can be cut back to any length.

Inputs per position k, as in nano-sigma's training (train_newtok.shift):

    tokens[k]  the token before target k (START at k = 0)
    book[k]    book_features of the book before the message that target k is in

`KVSimulator.feed` on a whole window equals the model's forward on it; that is
gate G3 (run/post_training/policy_bench/check_simulator.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from post_training.policy_bench.nanosigma.model import apply_rope, rope_cache
from post_training.policy_bench.nanosigma.model_newtok import NanoSigmaNewTok

# cuDNN attention builds a graph per key length, which costs ~25 ms on every
# decoding step (job 7835093); the other kernels take any length.
_KERNELS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def load_newtok(path: str, device: torch.device | str = "cpu") -> NanoSigmaNewTok:
    """newtok_a_r1-style checkpoint -> model in eval mode."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = NanoSigmaNewTok(d=a["d"], n_heads=a["heads"], n_fused=a["n_fused"], use_book=not a["no_book"])
    m.load_state_dict(ck["model"], strict=True)
    return m.to(device).eval()


class KVSimulator:
    """One sequence. `feed` appends positions and returns their logits;
    `truncate` drops positions from the end."""

    def __init__(self, model: NanoSigmaNewTok, max_len: int, dtype: torch.dtype = torch.bfloat16) -> None:
        if not model.use_book:
            raise ValueError("the bench simulator has a book branch")
        self.m, self.max_len, self.dtype = model, max_len, dtype
        dev = model.embed.weight.device
        self.blocks = [*model.msg_blocks, *model.book_blocks, *model.blocks]
        h, dh = model.n_heads, model.d // model.n_heads
        self.k = torch.zeros(len(self.blocks), 1, h, max_len, dh, device=dev, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.cos, self.sin = rope_cache(max_len, dh, dev)
        self.n = 0

    def truncate(self, n: int) -> None:
        if not 0 <= n <= self.n:
            raise ValueError(f"truncate to {n} with {self.n} positions")
        self.n = n

    @torch.no_grad()
    def feed(self, tokens: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        """tokens (L,) ids, book (L, 503) -> logits (L, vocab), float32."""
        L = tokens.shape[0]
        if self.n + L > self.max_len:
            raise ValueError(f"{self.n} + {L} positions exceed max_len {self.max_len}")
        m, lo = self.m, self.n
        cos, sin = self.cos[lo:lo + L], self.sin[lo:lo + L]
        with torch.autocast(tokens.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32), \
                sdpa_kernel(_KERNELS):
            i = 0
            x = m.embed(tokens[None])
            for blk in m.msg_blocks:
                x = self._block(i, blk, x, cos, sin); i += 1
            b = book[None].to(x.dtype)
            b = m.book_proj(b + m.book_pre(m.book_pre_norm(b)))
            for blk in m.book_blocks:
                b = self._block(i, blk, b, cos, sin); i += 1
            x = m.fuse(torch.cat([x, b], dim=-1))
            for blk in m.blocks:
                x = self._block(i, blk, x, cos, sin); i += 1
            logits = m.head(m.norm(x))
        self.n = lo + L
        return logits[0].float()

    def _block(self, i, blk, x, cos, sin):
        x = x + self._attn(i, blk.attn, blk.n1(x), cos, sin)
        return x + blk.mlp(blk.n2(x))

    def _attn(self, i, a, x, cos, sin):
        B, L, D = x.shape
        q, k, v = a.qkv(x).view(B, L, 3, a.h, a.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        lo, hi = self.n, self.n + L
        self.k[i, :, :, lo:hi] = k.to(self.dtype)
        self.v[i, :, :, lo:hi] = v.to(self.dtype)
        kk, vv = self.k[i, :, :, :hi].to(q.dtype), self.v[i, :, :, :hi].to(q.dtype)
        if lo == 0:
            y = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        elif L == 1:
            y = F.scaled_dot_product_attention(q, kk, vv)
        else:   # new position j sees the cache and new positions <= j
            mask = torch.arange(hi, device=x.device)[None] <= torch.arange(lo, hi, device=x.device)[:, None]
            y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        return a.proj(y.transpose(1, 2).reshape(B, L, D))
