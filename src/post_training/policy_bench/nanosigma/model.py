"""The nanoSigma baseline: a plain decoder-only Transformer laid out like
sigma-0's PaddedLobPredModel.

    message tokens --> Embedding --> 1 block --------------------\
                                                                  concat -> Linear(2d->d) -> n_fused blocks -> norm -> head
    book (503-d per message) --> MLP@503 -> Linear(503->d) -> repeat x26 -> 1 block /

The book vector of message i is repeated over that message's 26 token positions,
as sigma-0's repeat_book does. One output head predicts the next token.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import D_BOOK, MSG_TOKENS, START, VOCAB


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).type_as(x)


def rope_cache(n, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(n, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc = nn.Linear(d, hidden, bias=False)
        self.out = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.out(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.n2 = RMSNorm(d)
        self.mlp = MLP(d, 4 * d)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class NanoSigma(nn.Module):
    def __init__(self, d=512, n_heads=8, n_fused=6, n_msg_layers=1, n_book_layers=1,
                 book_hidden=1024, use_book=True, checkpoint=False):
        super().__init__()
        self.d, self.n_heads, self.use_book, self.checkpoint = d, n_heads, use_book, checkpoint
        self.embed = nn.Embedding(VOCAB, d)
        self.msg_blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_msg_layers)])
        if use_book:
            self.book_pre_norm = RMSNorm(D_BOOK)
            self.book_pre = MLP(D_BOOK, book_hidden)
            self.book_proj = nn.Linear(D_BOOK, d, bias=False)
            self.book_blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_book_layers)])
            self.fuse = nn.Linear(2 * d, d, bias=False)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_fused)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") and "book" not in n or n.endswith("mlp.out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * (n_fused + n_msg_layers + n_book_layers)))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _run(self, blocks, x, cos, sin):
        for blk in blocks:
            if self.checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return x

    def hidden(self, tokens, book=None):
        """tokens: (B, T) ids, T = n_msgs * 26 (input, already shifted).
        book: (B, n_msgs, 503) float. Returns (B, T, d)."""
        B, T = tokens.shape
        cos, sin = rope_cache(T, self.d // self.n_heads, tokens.device)
        x = self._run(self.msg_blocks, self.embed(tokens), cos, sin)
        if self.use_book:
            b = book.to(x.dtype)
            b = b + self.book_pre(self.book_pre_norm(b))
            b = self.book_proj(b).repeat_interleave(MSG_TOKENS, dim=1)
            b = self._run(self.book_blocks, b, cos, sin)
            x = self.fuse(torch.cat([x, b], dim=-1))
        x = self._run(self.blocks, x, cos, sin)
        return self.norm(x)

    def forward(self, tokens, book=None):
        return self.head(self.hidden(tokens, book))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- message-level variant
# Per-token embedding widths (26 positions of the 26-token scheme). Each token gets its own table and
# width; the book is one more part of the same concatenation, so every token stands at the same level
# as the order book. Layout: evt, dir, price(3), size(2), dt_s, dt_ns(3), t_s(2), t_ns(3),
# price_ref(3), size_ref(2), t_s_ref(2), t_ns_ref(3).
TOKEN_WIDTHS = ([32, 16] + [64, 64, 64] + [48, 48] + [32] + [32, 32, 32] + [24, 24] + [24, 24, 24]
                + [32, 32, 32] + [24, 24] + [16, 16] + [16, 16, 16])
assert len(TOKEN_WIDTHS) == MSG_TOKENS


class NanoSigmaMsg(nn.Module):
    """One trunk position per MESSAGE instead of per token.

    message i-1 tokens --> 26 embeddings (own width each) --\\
                                                              concat -> Linear -> n_trunk blocks over messages -> h_i
    book before message i --> RMSNorm+MLP@503 -> Linear(503->d_book) /

    h_i + embedding of the previous token + field position --> n_depth causal blocks over the 26 tokens
    of message i --> head. Same autoregressive factorisation and loss as the token model."""

    def __init__(self, d=512, n_heads=8, n_trunk=6, n_depth=2, d_book=256, book_hidden=1024,
                 widths=TOKEN_WIDTHS, use_book=True, checkpoint=False):
        super().__init__()
        self.d, self.n_heads, self.use_book, self.checkpoint = d, n_heads, use_book, checkpoint
        self.tok_emb = nn.ModuleList([nn.Embedding(VOCAB, w) for w in widths])
        width = sum(widths)
        if use_book:
            self.book_pre_norm = RMSNorm(D_BOOK)
            self.book_pre = MLP(D_BOOK, book_hidden)
            self.book_proj = nn.Linear(D_BOOK, d_book, bias=False)
            width += d_book
        self.in_norm = RMSNorm(width)
        self.inp = nn.Linear(width, d, bias=False)
        self.trunk = nn.ModuleList([Block(d, n_heads) for _ in range(n_trunk)])
        self.trunk_norm = RMSNorm(d)
        self.dec_emb = nn.Embedding(VOCAB, d)
        self.dec_pos = nn.Parameter(torch.zeros(MSG_TOKENS, d))
        self.depth = nn.ModuleList([Block(d, n_heads) for _ in range(n_depth)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.apply(NanoSigma._init)
        nn.init.normal_(self.dec_pos, std=0.02)
        for n, p in self.named_parameters():
            if (n.endswith("attn.proj.weight") or n.endswith("mlp.out.weight")) and "book" not in n:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * (n_trunk + n_depth)))

    _run = NanoSigma._run

    def forward(self, tokens, book=None):
        """tokens: (B, N*26) ids, already shifted by one (START first), as for NanoSigma.
        book: (B, N, 503). Returns logits (B, N*26, VOCAB)."""
        return self.head(self.hidden(tokens, book))

    def hidden(self, tokens, book=None):
        """Same contract as NanoSigma.hidden: (B, N*26) shifted ids -> (B, N*26, d). Position 26i reads
        messages < i and the book before message i, and nothing of message i or later."""
        B, T = tokens.shape
        N = T // MSG_TOKENS
        dec_in = tokens.view(B, N, MSG_TOKENS)                       # token (i, j) - 1, START at (0, 0)
        start = torch.full((B, 1, MSG_TOKENS), tokens[0, 0].item(), dtype=tokens.dtype, device=tokens.device)
        prev = torch.cat([start, tokens[:, 1:(N - 1) * MSG_TOKENS + 1].reshape(B, N - 1, MSG_TOKENS)], 1)
        parts = [emb(prev[..., k]) for k, emb in enumerate(self.tok_emb)]
        if self.use_book:
            b = book.to(parts[0].dtype)
            b = b + self.book_pre(self.book_pre_norm(b))
            parts.append(self.book_proj(b))
        x = self.inp(self.in_norm(torch.cat(parts, -1)))
        cos, sin = rope_cache(N, self.d // self.n_heads, tokens.device)
        h = self.trunk_norm(self._run(self.trunk, x, cos, sin))       # (B, N, d)
        y = h.unsqueeze(2) + self.dec_emb(dec_in) + self.dec_pos.to(h.dtype)
        y = y.reshape(B * N, MSG_TOKENS, self.d)
        c2, s2 = rope_cache(MSG_TOKENS, self.d // self.n_heads, tokens.device)
        return self.norm(self._run(self.depth, y, c2, s2)).view(B, T, self.d)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- field-embed variant
# Field of each of the 26 token positions, and the global id range of each field in the shared vocab.
from .data import DIR0, EVT0, PRICE0, SIGN0, SIZE0, TIME0  # noqa: E402

FIELD_OF_POS = (["evt", "dir", "sign", "price", "price", "size", "size"] + ["time"] * 9
                + ["sign", "price", "price", "size", "size"] + ["time"] * 5)
FIELD_RANGE = {"time": (TIME0, 1000), "evt": (EVT0, 4), "size": (SIZE0, 100), "price": (PRICE0, 1000),
               "sign": (SIGN0, 2), "dir": (DIR0, 2)}
N_SPECIAL = 4  # MASK, HIDDEN, NA, START keep local ids 0..3
assert len(FIELD_OF_POS) == MSG_TOKENS


class NanoSigmaField(nn.Module):
    """Token-level trunk with the book as its own token: 27 positions per message, [book, t1 .. t26].

    Each of the 26 token positions has its own input table and output head over that field's local
    vocabulary (field values + the 4 special ids). The book vector (book BEFORE the message) goes
    through RMSNorm + MLP@503 + Linear(503->d) and becomes the first token of the message. The loss is
    on the 26 message tokens only: position 27i+k-1 predicts token k of message i."""

    is_field = True

    def __init__(self, d=512, n_heads=8, n_fused=6, book_hidden=1024, use_book=True, checkpoint=False):
        super().__init__()
        self.d, self.n_heads, self.use_book, self.checkpoint = d, n_heads, use_book, checkpoint
        base = [FIELD_RANGE[f][0] for f in FIELD_OF_POS]
        size = [FIELD_RANGE[f][1] for f in FIELD_OF_POS]
        self.register_buffer("base", torch.tensor(base), persistent=False)
        self.register_buffer("size", torch.tensor(size), persistent=False)
        self.emb = nn.ModuleList([nn.Embedding(n + N_SPECIAL, d) for n in size])
        self.heads = nn.ModuleList([nn.Linear(d, n + N_SPECIAL, bias=False) for n in size])
        if use_book:
            self.book_pre_norm = RMSNorm(D_BOOK)
            self.book_pre = MLP(D_BOOK, book_hidden)
            self.book_proj = nn.Linear(D_BOOK, d, bias=False)
        else:
            self.book_const = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_fused)])
        self.norm = RMSNorm(d)
        self.apply(NanoSigma._init)
        for n, p in self.named_parameters():
            if n.endswith("attn.proj.weight") or n.endswith("mlp.out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_fused))

    _run = NanoSigma._run

    def to_local(self, rows):
        """(..., 26) global ids -> per-field local ids (specials stay 0..3)."""
        loc = rows - self.base + N_SPECIAL
        loc = torch.where(rows < N_SPECIAL, rows, loc)
        return loc.clamp(min=0).minimum(self.size + N_SPECIAL - 1)

    def nll(self, tokens, book=None):
        """tokens: (B, N*26) global ids, NOT shifted. book: (B, N, 503).
        Returns per-token negative log-likelihood (B, N*26), float32."""
        B, T = tokens.shape
        N = T // MSG_TOKENS
        loc = self.to_local(tokens.view(B, N, MSG_TOKENS))
        tok = torch.stack([self.emb[k](loc[..., k]) for k in range(MSG_TOKENS)], 2)   # (B, N, 26, d)
        if self.use_book:
            b = book.to(tok.dtype)
            b = self.book_proj(b + self.book_pre(self.book_pre_norm(b)))
        else:
            b = self.book_const.to(tok.dtype).expand(B, N, self.d)
        x = torch.cat([b.unsqueeze(2), tok], 2).reshape(B, N * (MSG_TOKENS + 1), self.d)
        cos, sin = rope_cache(x.shape[1], self.d // self.n_heads, tokens.device)
        h = self.norm(self._run(self.blocks, x, cos, sin)).view(B, N, MSG_TOKENS + 1, self.d)
        out = []
        for k in range(MSG_TOKENS):                      # position k (book or token k-1) predicts token k
            logits = self.heads[k](h[:, :, k]).float()
            out.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), loc[..., k].reshape(-1),
                                       reduction="none").view(B, N))
        return torch.stack(out, 2).reshape(B, T)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())



# ---------------------------------------------------------------- LOBERT-style variant
# Coarse message token (LOBERT, Linna et al. 2025): event x direction x signed price bin x size bin x round lot,
# computed from the 26 fine tokens. Price bins |p| (ticks): 0,1,2,3,4-5,6-10,11-50,51-200,>200, times the sign;
# size bins: <50, 50-99, 100-199, >=200; round lot: size % 100 == 0. Coarse id 0 = no message (masked/special).
PRICE_EDGES = (1, 2, 3, 4, 6, 11, 51, 201)
SIZE_EDGES = (50, 100, 200)
N_PRICE_BINS = 2 * (len(PRICE_EDGES) + 1)
N_COARSE = 1 + 4 * 2 * N_PRICE_BINS * (len(SIZE_EDGES) + 1) * 2


def coarse_tokens(tok):
    """(B, N, 26) fine ids -> (B, N) coarse ids in [0, N_COARSE)."""
    evt = tok[..., 0] - EVT0
    dirn = tok[..., 1] - DIR0
    pos = (tok[..., 2] - SIGN0).clamp(0, 1)                       # 1: price > 0 (zero counts as negative)
    p = (tok[..., 3] - PRICE0) * 1000 + (tok[..., 4] - PRICE0)
    s = (tok[..., 5] - SIZE0) * 100 + (tok[..., 6] - SIZE0)
    pb = torch.bucketize(p, torch.tensor(PRICE_EDGES, device=tok.device), right=True) * 2 + pos
    sb = torch.bucketize(s, torch.tensor(SIZE_EDGES, device=tok.device), right=True)
    rnd = ((s % 100) == 0).long()
    c = (((evt * 2 + dirn) * N_PRICE_BINS + pb) * (len(SIZE_EDGES) + 1) + sb) * 2 + rnd
    ok = (evt >= 0) & (evt < 4) & (dirn >= 0) & (dirn < 2) & (tok[..., 2] >= SIGN0)
    return torch.where(ok, c + 1, torch.zeros_like(c))


class NanoSigmaLobert(nn.Module):
    """One trunk position per message, LOBERT-style input and target.

    e_i = E_c[coarse(msg i-1)] + f2(gelu(f1([B[x_{i-1,1}], ..., B[x_{i-1,26}]]))) + book(before msg i)
    B: one shared (VOCAB x d_fine) table for the 26 fine ids; f2 starts at zero, so at init the input is the
    coarse token (+ book) only. n_trunk blocks over messages; the head predicts the COARSE token of message i
    (one loss per message, the fine values are input only)."""

    is_coarse = True

    def __init__(self, d=512, n_heads=8, n_trunk=6, d_fine=32, book_hidden=1024, use_book=True, checkpoint=False):
        super().__init__()
        self.d, self.n_heads, self.use_book, self.checkpoint = d, n_heads, use_book, checkpoint
        self.coarse_emb = nn.Embedding(N_COARSE, d)
        self.fine_emb = nn.Embedding(VOCAB, d_fine)
        self.f1 = nn.Linear(MSG_TOKENS * d_fine, d)
        self.f2 = nn.Linear(d, d)
        if use_book:
            self.book_pre_norm = RMSNorm(D_BOOK)
            self.book_pre = MLP(D_BOOK, book_hidden)
            self.book_proj = nn.Linear(D_BOOK, d, bias=False)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_trunk)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, N_COARSE, bias=False)
        self.apply(NanoSigma._init)
        nn.init.zeros_(self.f2.weight)
        nn.init.zeros_(self.f2.bias)
        for n, p in self.named_parameters():
            if (n.endswith("attn.proj.weight") or n.endswith("mlp.out.weight")) and "book" not in n:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_trunk))

    _run = NanoSigma._run

    def trunk(self, msgs, book):
        """msgs: (B, N, 26) fine ids of messages 0..N-1; book: (B, N, 503), row i = book before message i.
        Returns (B, N, d); position i has read messages < i and book row i."""
        B, N, _ = msgs.shape
        prev = torch.cat([torch.zeros_like(msgs[:, :1]), msgs[:, :-1]], 1)   # START slot: coarse 0, fine MASK
        x = self.coarse_emb(coarse_tokens(prev))
        x = x + self.f2(F.gelu(self.f1(self.fine_emb(prev).flatten(2))))
        if self.use_book:
            b = book.to(x.dtype)
            b = b + self.book_pre(self.book_pre_norm(b))
            x = x + self.book_proj(b)
        cos, sin = rope_cache(N, self.d // self.n_heads, msgs.device)
        return self.norm(self._run(self.blocks, x, cos, sin))

    def nll(self, tok, book=None):
        """tok: (B, N*26) UNshifted fine ids. Coarse NLL of each message, spread evenly over its 26 token
        slots, so (B, N*26) like the other models: mean per token x 26 = coarse nats per message."""
        B, T = tok.shape
        msgs = tok.view(B, T // MSG_TOKENS, MSG_TOKENS)
        logits = self.head(self.trunk(msgs, book))
        nll = F.cross_entropy(logits.float().flatten(0, 1), coarse_tokens(msgs).flatten(), reduction="none")
        return (nll / MSG_TOKENS).view(B, -1, 1).expand(-1, -1, MSG_TOKENS).reshape(B, T)

    def hidden(self, tokens, book=None):
        """Same contract as NanoSigma.hidden ((B, N*26) SHIFTED ids -> (B, N*26, d)): the message-level state
        repeated over the 26 slots of each message, so position 26i is the state before message i."""
        B, T = tokens.shape
        unshifted = torch.cat([tokens[:, 1:], torch.zeros_like(tokens[:, :1])], 1)
        h = self.trunk(unshifted.view(B, T // MSG_TOKENS, MSG_TOKENS), book)
        return h.repeat_interleave(MSG_TOKENS, dim=1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

class NanoSigmaTokCoarse(NanoSigma):
    """The 26-token model (26 positions per message, same embedding, book and blocks), but with the coarse
    target of NanoSigmaLobert: the head predicts the COARSE token of message i, read only at position 26i
    (input = last token of message i-1, book row i), so one loss per message. Isolates the coarse target from
    packing: same input and cost as the 26-token model, same target as the packed model."""

    is_coarse = True

    def __init__(self, **kw):
        super().__init__(**kw)
        self.head = nn.Linear(self.d, N_COARSE, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def nll(self, tok, book=None):
        """tok: (B, N*26) UNshifted fine ids -> (B, N*26) coarse NLL per message spread over its 26 slots."""
        B, T = tok.shape
        inp = torch.cat([torch.full_like(tok[:, :1], START), tok[:, :-1]], 1)
        logits = self.head(self.hidden(inp, book)[:, ::MSG_TOKENS])
        msgs = tok.view(B, T // MSG_TOKENS, MSG_TOKENS)
        nll = F.cross_entropy(logits.float().flatten(0, 1), coarse_tokens(msgs).flatten(), reduction="none")
        return (nll / MSG_TOKENS).view(B, -1, 1).expand(-1, -1, MSG_TOKENS).reshape(B, T)


def build_model(args, checkpoint=False):
    """Model of a train.py run from its saved args (ckpt["args"] or config.json)."""
    common = dict(d=args["d"], n_heads=args["heads"], use_book=not args["no_book"], checkpoint=checkpoint)
    arch = args.get("arch", "token")
    if arch == "lobert":
        return NanoSigmaLobert(n_trunk=args["n_fused"], **common)
    if arch == "token_coarse":
        return NanoSigmaTokCoarse(n_fused=args["n_fused"], **common)
    if arch == "message":
        return NanoSigmaMsg(n_trunk=args["n_fused"], n_depth=args["n_depth"], **common)
    if arch == "field":
        return NanoSigmaField(n_fused=args["n_fused"], **common)
    return NanoSigma(n_fused=args["n_fused"], **common)
