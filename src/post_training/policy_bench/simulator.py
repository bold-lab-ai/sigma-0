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

`BatchKVSimulator` holds B sequences of their own lengths. `prefill` feeds a
chunk to one row; `step` feeds one token to every active row at that row's own
position, which is how B episodes are decoded together. `KVSimulator` is the
one-row case. A step attends over the longest row's length, masked per row.
A row's logits do not change with that length (bit-equal, job 7848159:
fully masked key blocks add exactly zero). `full_keys` fixes the length at
max_len for a kernel where that would not hold; it costs 45 ms per step at
batch 48 against 30 to 41 ms (job 7848120).
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


class BatchKVSimulator:
    """B sequences, each with its own length. Rows are cut back one at a time."""

    def __init__(self, model: NanoSigmaNewTok, batch: int, max_len: int,
                 dtype: torch.dtype = torch.bfloat16, full_keys: bool = False) -> None:
        if not model.use_book:
            raise ValueError("the bench simulator has a book branch")
        self.m, self.batch, self.max_len, self.dtype = model, batch, max_len, dtype
        self.full_keys = full_keys
        self.dev = model.embed.weight.device
        self.blocks = [*model.msg_blocks, *model.book_blocks, *model.blocks]
        h, dh = model.n_heads, model.d // model.n_heads
        self.k = torch.zeros(len(self.blocks), batch, h, max_len, dh, device=self.dev, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.cos, self.sin = rope_cache(max_len, dh, self.dev)
        self.n = [0] * batch

    def truncate(self, row: int, n: int) -> None:
        if not 0 <= n <= self.n[row]:
            raise ValueError(f"truncate row {row} to {n} with {self.n[row]} positions")
        self.n[row] = n

    @torch.no_grad()
    def prefill(self, row: int, tokens: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        """Append tokens (L,) with book (L, 503) to one row -> logits (L, vocab), float32."""
        L, lo = tokens.shape[0], self.n[row]
        if lo + L > self.max_len:
            raise ValueError(f"{lo} + {L} positions exceed max_len {self.max_len}")
        hi = lo + L
        cos, sin = self.cos[lo:hi], self.sin[lo:hi]
        r = slice(row, row + 1)

        def attend(i, q, k, v):
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            self.k[i, r, :, lo:hi] = k.to(self.dtype)
            self.v[i, r, :, lo:hi] = v.to(self.dtype)
            kk, vv = self.k[i, r, :, :hi].to(q.dtype), self.v[i, r, :, :hi].to(q.dtype)
            if lo == 0:
                return F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
            if L == 1:
                return F.scaled_dot_product_attention(q, kk, vv)
            # new position j sees the cache and new positions <= j
            mask = torch.arange(hi, device=q.device)[None] <= torch.arange(lo, hi, device=q.device)[:, None]
            return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)

        logits = self._forward(tokens[None], book[None], attend)
        self.n[row] = hi
        return logits[0]

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, book: torch.Tensor, active: list[bool] | None = None) -> torch.Tensor:
        """One token per row: tokens (B,), book (B, 503) -> logits (B, vocab), float32.
        Inactive rows are computed but keep their length and cache."""
        active = [True] * self.batch if active is None else list(active)
        if len(tokens) != self.batch or len(active) != self.batch:
            raise ValueError(f"step takes all {self.batch} rows")
        for b in range(self.batch):
            if active[b] and self.n[b] >= self.max_len:
                raise ValueError(f"row {b} is full at max_len {self.max_len}")
        # an inactive row writes one slot past its end (or its last slot), which nothing reads
        pos = torch.tensor([min(n, self.max_len - 1) for n in self.n], device=self.dev)
        hi = self.max_len if self.full_keys else int(pos.max()) + 1
        rows = torch.arange(self.batch, device=self.dev)
        cos, sin = self.cos[pos][:, None, None], self.sin[pos][:, None, None]     # (B, 1, 1, dh / 2)
        mask = (torch.arange(hi, device=self.dev)[None] <= pos[:, None])[:, None, None]

        def attend(i, q, k, v):
            q, k = _rope_rows(q, cos, sin), _rope_rows(k, cos, sin)
            self.k[i, rows, :, pos] = k[:, :, 0].to(self.dtype)
            self.v[i, rows, :, pos] = v[:, :, 0].to(self.dtype)
            kk, vv = self.k[i, :, :, :hi].to(q.dtype), self.v[i, :, :, :hi].to(q.dtype)
            return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)

        logits = self._forward(tokens[:, None], book[:, None], attend)
        for b in range(self.batch):
            self.n[b] += int(active[b])
        return logits[:, 0]

    def _forward(self, tokens, book, attend):
        """The newtok forward (NanoSigmaNewTok.hidden + head) with attention by `attend`."""
        m = self.m
        with torch.autocast(tokens.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32), \
                sdpa_kernel(_KERNELS):
            i = 0
            x = m.embed(tokens)
            for blk in m.msg_blocks:
                x = _block(i, blk, x, attend); i += 1
            b = book.to(x.dtype)
            b = m.book_proj(b + m.book_pre(m.book_pre_norm(b)))
            for blk in m.book_blocks:
                b = _block(i, blk, b, attend); i += 1
            x = m.fuse(torch.cat([x, b], dim=-1))
            for blk in m.blocks:
                x = _block(i, blk, x, attend); i += 1
            return m.head(m.norm(x)).float()


class KVSimulator:
    """One sequence. `feed` appends positions and returns their logits;
    `truncate` drops positions from the end."""

    def __init__(self, model: NanoSigmaNewTok, max_len: int, dtype: torch.dtype = torch.bfloat16) -> None:
        self._b = BatchKVSimulator(model, 1, max_len, dtype)
        self.max_len = max_len

    @property
    def n(self) -> int:
        return self._b.n[0]

    def truncate(self, n: int) -> None:
        self._b.truncate(0, n)

    def feed(self, tokens: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        """tokens (L,) ids, book (L, 503) -> logits (L, vocab), float32."""
        return self._b.prefill(0, tokens, book)


def _block(i, blk, x, attend):
    x = x + _attention(i, blk.attn, blk.n1(x), attend)
    return x + blk.mlp(blk.n2(x))


def _attention(i, a, x, attend):
    B, L, D = x.shape
    q, k, v = a.qkv(x).view(B, L, 3, a.h, a.dh).permute(2, 0, 3, 1, 4)
    return a.proj(attend(i, q, k, v).transpose(1, 2).reshape(B, L, D))


def _rope_rows(x, cos, sin):
    """apply_rope with one position per row: x (B, H, 1, dh), cos and sin (B, 1, 1, dh / 2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)
