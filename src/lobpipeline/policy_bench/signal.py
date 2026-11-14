"""The return signal: nano-sigma `returnft_lobert_pass1` on the bench's state.

The signal reads processed rows, one per LOBSTER message (an execution burst
is one row per filled order), and the L10 row before each. Two columns of a
processed row depend on more than the message itself (job 7836700, 100%):

    col 4       price_rel = (price_abs - mid floored to a tick) / TICK, the
                mid of the book before the row, truncated toward zero (a
                sub-penny price below the mid: -8399.95 -> -8399, job 7842392)
                and clipped to +-price_rel_clip(file) ticks. Two preprocessing runs made the files: regular-session
                files (`*_34200000_57600000_*`) clip at 999, the others at
                999,999 (jobs 7837738, 7837813, 7837914)
    cols 10-13  for events 2, 3, 4: price_rel, size, t_s, t_ns of the first
                event-1 row of the same order id earlier in the day's file;
                NA (-9999) when there is none (the order came from the
                opening snapshot or a dropped raw message). NA for events 1

`AddRegistry` and `processed_row` rebuild them for generated rows. A day's
prefix is read once into its add index (`daydata.build_add_index`, stored by
run/benchmarking/policy_bench/build_add_index.py), not per decision.
`Signal.predict` is tools/return_features.py's decision geometry: P rows of
context, the book at t, the feature at the first position of row P + 1, and
the fine-tuned linear head scaled back to bps.
"""

from __future__ import annotations

import numpy as np
import torch

from lobpipeline.policy_bench.codec import mid_floor
from lobpipeline.policy_bench.daydata import build_add_index
from lobpipeline.policy_bench.nanosigma.data import MSG_TOKENS, NA_VAL, START, TICK, book_features, encode_messages
from lobpipeline.policy_bench.nanosigma.model import build_model

C_OID, C_EVT, C_DIR, C_PRICE_ABS, C_PRICE_REL, C_SIZE = 0, 1, 2, 3, 4, 5
C_DTS, C_DTNS, C_TS, C_TNS = 6, 7, 8, 9
MAX_SIZE = 9999


def price_rel_clip(message_file: str) -> int:
    """The price_rel clip of a stock-day, from its message file name."""
    return 999 if "_34200000_57600000_" in message_file else 999_999


class AddRegistry:
    """Order id -> (price_rel, size, t_s, t_ns) of its first event-1 row.

    Orders added before row `before` of the day come from the day's add index;
    rows observed after that (real or generated) are kept in a dict."""

    def __init__(self, index: np.ndarray | None = None, before: int = 0) -> None:
        self._index, self._before = index, before
        self._ref: dict[int, tuple[int, int, int, int]] = {}

    @classmethod
    def from_rows(cls, msg: np.ndarray) -> "AddRegistry":
        """The registry after the processed rows `msg` (e.g. rows 0..s of a day)."""
        return cls(build_add_index(msg), len(msg))

    def ref(self, oid: int) -> tuple[int, int, int, int]:
        r = self._lookup(oid)
        return (NA_VAL,) * 4 if r is None else r

    def observe(self, row: np.ndarray) -> None:
        oid = int(row[C_OID])
        if int(row[C_EVT]) == 1 and self._lookup(oid) is None:
            self._ref[oid] = tuple(int(row[c]) for c in (C_PRICE_REL, C_SIZE, C_TS, C_TNS))

    def _lookup(self, oid: int) -> tuple[int, int, int, int] | None:
        r = self._ref.get(oid)
        ix = self._index
        if r is None and ix is not None and ix.shape[1]:
            i = int(np.searchsorted(ix[0], oid))
            if i < ix.shape[1] and int(ix[0, i]) == oid and int(ix[1, i]) < self._before:
                r = tuple(int(x) for x in ix[2:, i])
        return r


def processed_row(oid: int, evt: int, d: int, price: int, size: int, dt_s: int, dt_ns: int,
                  t_s: int, t_ns: int, l10_before: np.ndarray, reg: AddRegistry, clip: int) -> np.ndarray:
    """One 14-column processed row, given the 43-column L10 row of the book before it
    and the day's price_rel clip."""
    row = np.empty(14, np.int64)
    off = price - mid_floor(int(l10_before[3]), int(l10_before[5]))
    rel = off // TICK if off >= 0 else -(-off // TICK)
    row[:10] = (oid, evt, d, price, max(-clip, min(clip, rel)),
                min(size, MAX_SIZE), dt_s, dt_ns, t_s, t_ns)
    row[10:] = reg.ref(oid) if evt in (2, 3, 4) else (NA_VAL,) * 4
    return row


class Signal:
    """The fine-tuned backbone and its return head."""

    def __init__(self, path: str, device: torch.device | str = "cpu") -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = build_model(ck["args"])
        self.model.load_state_dict(ck["model"], strict=True)
        self.model.to(device).eval()
        self.horizons = list(ck["horizons"])
        self.head = torch.nn.Linear(self.model.d, len(self.horizons)).to(device)
        self.head.load_state_dict(ck["return_head"])
        self.scale = torch.tensor(ck["scale_bps"], dtype=torch.float32, device=device)
        self.prefill = int(ck["finetune"]["prefill"])
        self.device = torch.device(device)

    def inputs(self, msg: np.ndarray, l10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """P processed rows and the P + 1 L10 rows before them (the last one is
        the book at t) -> flat tokens ((P + 1) * 26,) and book vectors (P + 1, 503)."""
        P = len(msg)
        if len(l10) != P + 1:
            raise ValueError(f"{P} rows need {P + 1} L10 rows, got {len(l10)}")
        tok = np.zeros((P + 1, MSG_TOKENS), np.int64)    # row P + 1 stays masked
        tok[:P] = encode_messages(msg)
        return tok.reshape(-1), book_features(l10)

    @torch.no_grad()
    def predict(self, tok: np.ndarray, book: np.ndarray) -> np.ndarray:
        """Batched `inputs` (B, T), (B, P + 1, 503) -> predicted returns (B, horizons) in bps."""
        tok = torch.as_tensor(tok, device=self.device)
        inp = torch.cat([torch.full_like(tok[:, :1], START), tok[:, :-1]], dim=1)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            h = self.model.hidden(inp, torch.as_tensor(book, device=self.device))
        pos = (book.shape[1] - 1) * MSG_TOKENS
        return (self.head(h[:, pos].float()) * self.scale).cpu().numpy()
