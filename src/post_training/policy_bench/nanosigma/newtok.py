"""New tokenization, scenario (a) of sigma-0 PR #1: variable-length messages.

Token layout (same ids as the reference tokenizer /projects/public/u6oz/quant/book_tokenizer_E):
    26-token vocab ids 0..2111 unchanged, plus EOL 2112, EOB 2113, EOE 2114, BOOK_START 2115,
    QSLOT q = 2116 + q (q = 0..99). VOCAB_NEW = 2216.
    prefix (11): Event, Dir, dt_s, dt_ms, dt_us, dt_ns, t_ks, t_s, t_ms, t_us, t_ns
    new order (16):    prefix + dP(3) + Size(2)
    cancellation (17): prefix + dP(3) + Size(2) + QSLOT     (LOBSTER type 2)
    deletion (17):     prefix + dP(3) + Size(2) + QSLOT     (LOBSTER type 3)
    execution (12+5k): prefix + k x (dP(3) + Size(2)) + EOE (consecutive type-4 rows with the same
                       timestamp and side form one burst; one group per price, sizes summed)
dP is in ticks relative to the mid just before the message, floored to the tick (user decision).
QSLOT comes from the full-book replay (<stem>_queue.npy col 0); a missing slot (-2, order not in
the book) is the NA token.

A window is a run of L processed message rows. Several rows of a burst become one message, so a
window of L rows has <= L messages and <= 17 L tokens. Per-message cross-entropy is reported per
processed row (total nats / L), the same unit as the 26-token baseline.
"""
import os
from collections import OrderedDict

import numpy as np

from .data import (NA, TIME0, EVT0, SIZE0, PRICE0, SIGN0, DIR0, TICK, C_EVT, C_DIR, C_SIZE,
                   C_DTS, C_DTNS, C_TS, C_TNS, book_features)

C_PRICE_ABS = 3
EOL, EOB, EOE, BOOK_START, QSLOT0 = 2112, 2113, 2114, 2115, 2116
N_QSLOT = 100
VOCAB_NEW = QSLOT0 + N_QSLOT
MAX_TOK_PER_ROW = 17
PAD = -100


def _d3(x):
    return np.stack([x // 1000 % 1000, x % 1000], 1)


def _time_prefix(m):
    """(n, 9) time tokens: dt_s dt_ms dt_us dt_ns t_ks t_s t_ms t_us t_ns."""
    dts = np.clip(m[:, C_DTS], 0, 999); dtn = np.clip(m[:, C_DTNS], 0, 10**9 - 1)
    ts = np.clip(m[:, C_TS], 0, 86399); tn = np.clip(m[:, C_TNS], 0, 10**9 - 1)
    cols = [dts, dtn // 10**6, dtn // 10**3 % 1000, dtn % 1000,
            ts // 1000, ts % 1000, tn // 10**6, tn // 10**3 % 1000, tn % 1000]
    return TIME0 + np.stack(cols, 1)


def _dp(dp_ticks):
    s = np.where(dp_ticks > 0, SIGN0 + 1, SIGN0)
    a = np.minimum(np.abs(dp_ticks), 999_999)
    return np.concatenate([s[:, None], PRICE0 + _d3(a)], 1)


def _size(q):
    q = np.clip(q, 0, 9999)
    return SIZE0 + np.stack([q // 100, q % 100], 1)


def encode_window(msg, book, qslot):
    """msg (L, 14) int64 rows, book (L, 43) book BEFORE each row, qslot (L,) int.
    Returns tokens (T,) int64, msg_of_token (T,) int64 index into the message list,
    book_rows (M,) int64 = window row of each message's first row (for its book vector)."""
    msg = np.asarray(msg, np.int64); book = np.asarray(book, np.int64); L = msg.shape[0]
    evt = np.clip(msg[:, C_EVT], 1, 4); d = np.clip(msg[:, C_DIR], 0, 1)
    mid = (book[:, 3] + book[:, 5]) // 2 // TICK * TICK
    dp = (msg[:, C_PRICE_ABS] - mid) // TICK
    pre = np.concatenate([(EVT0 + evt - 1)[:, None], (DIR0 + d)[:, None], _time_prefix(msg)], 1)
    dpt = _dp(dp); szt = _size(msg[:, C_SIZE])
    qs = np.where((qslot >= 0), QSLOT0 + np.minimum(qslot, N_QSLOT - 1), NA)
    # burst grouping: an execution row continues the previous message if that is an execution
    # with the same timestamp and side
    same = np.zeros(L, bool)
    if L > 1:
        same[1:] = ((evt[1:] == 4) & (evt[:-1] == 4) & (msg[1:, C_TS] == msg[:-1, C_TS])
                    & (msg[1:, C_TNS] == msg[:-1, C_TNS]) & (d[1:] == d[:-1]))
    toks, mot, first = [], [], []
    i = 0; m_i = 0
    while i < L:
        j = i + 1
        while j < L and same[j]: j += 1
        first.append(i)
        if evt[i] == 4:
            seen = OrderedDict()
            for r in range(i, j):   # every level of a burst is relative to the mid before the burst's first row
                p = int((msg[r, C_PRICE_ABS] - mid[i]) // TICK); seen[p] = seen.get(p, 0) + int(msg[r, C_SIZE])
            ps = np.fromiter(seen.keys(), np.int64); qq = np.fromiter(seen.values(), np.int64)
            body = np.concatenate([_dp(ps), _size(qq)], 1).reshape(-1)
            t = np.concatenate([pre[i], body, [EOE]])
        elif evt[i] == 1:
            t = np.concatenate([pre[i], dpt[i], szt[i]])
        else:
            t = np.concatenate([pre[i], dpt[i], szt[i], [qs[i]]])
        toks.append(t); mot.append(np.full(len(t), m_i)); m_i += 1; i = j
    return np.concatenate(toks), np.concatenate(mot), np.asarray(first, np.int64)


class NewTokCache:
    """Memory-maps message, book and queue-slot files of a stock-day (bounded LRU)."""

    def __init__(self, shard_root, max_open=64):
        self.root, self.max_open, self.cache = shard_root, max_open, OrderedDict()

    def get(self, e):
        key = (e["month"], e["message"])
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        base = os.path.join(self.root, e["month"])
        m = np.load(os.path.join(base, e["message"]), mmap_mode="r")
        b = np.load(os.path.join(base, e["orderbook"]), mmap_mode="r")
        q = np.load(os.path.join(e["fullbook_dir"], os.path.basename(e["message"])[:-4] + "_queue.npy"), mmap_mode="r")
        self.cache[key] = (m, b, q)
        if len(self.cache) > self.max_open: self.cache.popitem(last=False)
        return m, b, q


def read_window_newtok(cache, e, start, length):
    """tokens (T,), msg_of_token (T,), book vectors (M, 503) float32, n_rows."""
    m, b, q = cache.get(e)
    msg = np.asarray(m[start:start + length]); book = np.asarray(b[start:start + length])
    qs = np.asarray(q[start:start + length, 0])
    t, mot, first = encode_window(msg, book, qs)
    return t, mot, book_features(book[first]), length


def collate(items, max_tokens):
    """Pad a list of read_window_newtok outputs to max_tokens. Returns numpy arrays:
    tokens (B, T) int32 (PAD where empty), msg_idx (B, T) int32, book (B, Mmax, 503) float32, rows (B,)."""
    B = len(items); Mmax = max(x[2].shape[0] for x in items)
    tok = np.full((B, max_tokens), PAD, np.int32); mi = np.zeros((B, max_tokens), np.int32)
    bk = np.zeros((B, Mmax, items[0][2].shape[1]), np.float32); rows = np.zeros(B, np.int64)
    for k, (t, m, b, n) in enumerate(items):
        assert len(t) <= max_tokens, (len(t), max_tokens)
        tok[k, :len(t)] = t; mi[k, :len(t)] = m; mi[k, len(t):] = m[-1]; bk[k, :b.shape[0]] = b; rows[k] = n
    return tok, mi, bk, rows
