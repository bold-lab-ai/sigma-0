"""Data for nanoSigma: manifests, the 26-token message encoding, the 503-d book
vector, and window samplers. numpy only.

Inputs are the paired LOBSTER L10 files (`*_message_10_proc.npy`, 14 int64
columns, and `*_orderbook_10_proc.npy`, 43 int64 columns, one more row than the
message file; book row R is the book BEFORE message R). A manifest entry names
one stock-day inside a monthly squashfs shard mounted at `<shard_root>/<month>/`.
"""

import json
import os
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------- vocabulary
# Same layout and vocabulary size as sigma-0's 26-token scheme (2112 ids).
MASK, HIDDEN, NA, START = 0, 1, 2, 3
TIME0 = 4          # 0..999   -> 4..1003
EVT0 = 1004        # 1..4     -> 1004..1007
SIZE0 = 1008       # 0..99    -> 1008..1107
PRICE0 = 1108      # 0..999   -> 1108..2107
SIGN0 = 2108       # -1, +1   -> 2108, 2109
DIR0 = 2110        # 0, 1     -> 2110, 2111
VOCAB = 2112
MSG_TOKENS = 26
NA_VAL = -9999

# message columns
C_EVT, C_DIR, C_PRICE, C_SIZE, C_DTS, C_DTNS, C_TS, C_TNS = 1, 2, 4, 5, 6, 7, 8, 9
C_PREF, C_SREF, C_TSREF, C_TNSREF = 10, 11, 12, 13


def _digits(x, n, base=1000):
    """(rows,) int -> (rows, n) base-`base` digits, most significant first."""
    out = np.empty((x.shape[0], n), dtype=np.int64)
    for i in range(n):
        out[:, n - 1 - i] = x % base
        x = x // base
    return out


def _price_toks(p):
    sign = np.where(p > 0, SIGN0 + 1, SIGN0)            # zero counts as negative, as in sigma-0
    a = np.minimum(np.abs(p), 999_999)
    return np.concatenate([sign[:, None], PRICE0 + _digits(a, 2)], axis=1)


def _size_toks(s):
    s = np.clip(s, 0, 9999)
    return SIZE0 + np.stack([s // 100, s % 100], axis=1)


def _time_toks(x, n):
    x = np.clip(x, 0, 1000 ** n - 1)
    return TIME0 + _digits(x, n)


def _with_na(toks, raw):
    toks[raw == NA_VAL] = NA
    return toks


def encode_messages(m):
    """(n, 14) int64 message rows -> (n, 26) int64 token ids.

    Layout: evt, dir, price(sign, hi, lo), size(hi, lo), dt_s, dt_ns(3),
    time_s(2), time_ns(3), price_ref(3), size_ref(2), time_s_ref(2), time_ns_ref(3).
    Reference fields are NA for orders that reference nothing.
    """
    m = np.asarray(m, dtype=np.int64)
    out = [
        (EVT0 + np.clip(m[:, C_EVT], 1, 4) - 1)[:, None],
        (DIR0 + np.clip(m[:, C_DIR], 0, 1))[:, None],
        _price_toks(m[:, C_PRICE]),
        _size_toks(m[:, C_SIZE]),
        (TIME0 + np.clip(m[:, C_DTS], 0, 999))[:, None],
        _time_toks(m[:, C_DTNS], 3),
        _time_toks(m[:, C_TS], 2),
        _time_toks(m[:, C_TNS], 3),
        _with_na(_price_toks(m[:, C_PREF]), m[:, C_PREF]),
        _with_na(_size_toks(m[:, C_SREF]), m[:, C_SREF]),
        _with_na(_time_toks(m[:, C_TSREF], 2), m[:, C_TSREF]),
        _with_na(_time_toks(m[:, C_TNSREF], 3), m[:, C_TNSREF]),
    ]
    return np.concatenate(out, axis=1)


# ---------------------------------------------------------------- book vector
BOOK_LEVELS = 500   # price ticks around mid in the volume image
D_BOOK = BOOK_LEVELS + 3
TICK = 100


def book_features(b):
    """(n, 43) int64 L10 book rows -> (n, 503) float32, as sigma-0's
    transform_L2_state_numpy(book, 500, 100): [delta mid, time of day, ns
    fraction, 500-tick volume image centred on mid, ask side negative, /1000]."""
    b = np.asarray(b, dtype=np.int64)
    n = b.shape[0]
    out = np.zeros((n, D_BOOK), dtype=np.float32)
    # delta mid; clipped because days with an empty book side carry placeholder prices (|dmid| up to ~5e5)
    out[:, 0] = np.clip(b[:, 0], -10000, 10000)
    out[:, 1] = (b[:, 1] - 34200) / 23400.0
    out[:, 2] = b[:, 2] / 1e9
    prices = b[:, 3::2]          # ask1, bid1, ask2, bid2, ...
    sizes = b[:, 4::2]
    mid = np.ceil((prices[:, 0] + prices[:, 1]) / (2 * TICK)).astype(np.int64) * TICK
    idx = (prices - mid[:, None]) // TICK + BOOK_LEVELS // 2
    ok = (idx >= 0) & (idx < BOOK_LEVELS)
    rows = np.broadcast_to(np.arange(n)[:, None], idx.shape)
    img = np.zeros((n, BOOK_LEVELS), dtype=np.float32)
    img[rows[ok], idx[ok]] = sizes[ok]
    img[:, BOOK_LEVELS // 2:] *= -1
    out[:, 3:] = img / 1000.0
    return out


# ---------------------------------------------------------------- manifests
def load_manifest(path):
    with open(path) as f:
        return json.load(f)


class FileCache:
    """Keeps a bounded number of memory-mapped day files open."""

    def __init__(self, shard_root, max_open=64):
        self.root = shard_root
        self.max_open = max_open
        self.cache = OrderedDict()

    def get(self, entry):
        key = (entry["month"], entry["message"])
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        base = os.path.join(self.root, entry["month"])
        m = np.load(os.path.join(base, entry["message"]), mmap_mode="r")
        b = np.load(os.path.join(base, entry["orderbook"]), mmap_mode="r")
        self.cache[key] = (m, b)
        if len(self.cache) > self.max_open:
            self.cache.popitem(last=False)
        return m, b


def read_window(cache, entry, start, length):
    """Tokens (length, 26) int64 and book (length, 503) float32 for messages
    [start, start + length) of one stock-day."""
    m, b = cache.get(entry)
    msg = np.asarray(m[start:start + length])
    book = np.asarray(b[start:start + length])
    return encode_messages(msg), book_features(book)


def _overlaps(start, length, excl):
    end = start + length
    return any(start < e and s < end for s, e in excl)


class TrainSampler:
    """Uniform random windows over all train messages: a stock-day is drawn with
    probability proportional to its number of valid start positions, then a start
    is drawn uniformly. Windows overlapping a val window (`exclude`) are redrawn."""

    def __init__(self, entries, length, seed):
        self.entries = [e for e in entries if e["n_msgs"] >= length]
        self.length = length
        w = np.array([e["n_msgs"] - length + 1 for e in self.entries], dtype=np.float64)
        self.cum = np.cumsum(w / w.sum())
        self.rng = np.random.default_rng(seed)

    def set_length(self, length):
        if length != self.length:
            self.__init__(self.entries, length, int(self.rng.integers(1 << 62)))

    def sample(self):
        while True:
            i = int(np.searchsorted(self.cum, self.rng.random(), side="right"))
            i = min(i, len(self.entries) - 1)
            e = self.entries[i]
            s = int(self.rng.integers(0, e["n_msgs"] - self.length + 1))
            if not _overlaps(s, self.length, e.get("exclude", ())):
                return e, s


def val_windows(entries):
    """The frozen val windows: list of (entry, start, length)."""
    out = []
    for e in entries:
        for s, t in e["windows"]:
            out.append((e, int(s), int(t - s)))
    return out


def test_subset(entries, n_windows, length, seed):
    """A fixed random subset of 4096-aligned test windows, drawn uniformly over
    all aligned windows (same design as val). Deterministic for a given seed."""
    counts = np.array([e["n_msgs"] // length for e in entries], dtype=np.int64)
    total = int(counts.sum())
    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(total, size=min(n_windows, total), replace=False))
    cum = np.cumsum(counts)
    out = []
    for p in picks:
        i = int(np.searchsorted(cum, p, side="right"))
        k = int(p - (cum[i - 1] if i else 0))
        out.append((entries[i], k * length, length))
    return out
