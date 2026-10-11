"""Structured chart representation: frames the model cannot make invalid.

Adopted from BookDiT's ``midspread`` chart, adapted to this package's fixed
wall-clock frame grid.  A raw 10-level frame (4 columns per level: ask_p,
ask_s, bid_p, bid_s) becomes, level-major:

    channel 0        log(ask_1 - bid_1)                      log spread
    channel 1        ask size 1                              unchanged
    channel 2        mid_t - mid_{t-1}   (first frame: 0)    touch coordinate
    channel 3        bid size 1                              unchanged
    channel 4k       log(ask_{k+1} - ask_k)                  ask ladder gap
    channel 4k+1     ask size k+1                            unchanged
    channel 4k+2     log(bid_k - bid_{k+1})                  bid ladder gap
    channel 4k+3     bid size k+1                            unchanged
    [channel 4L]     log1p(events folded into the frame)     optional activity

The decode maps spread and gaps through exp (range strictly positive) and
rebuilds the ladder cumulatively, so ANY real-valued chart tensor — including
pure noise — decodes to a book with positive spread and strictly monotone
ladders.  Validity is an identity of the coordinate system, not something the
network must learn; BookDiT measures 0.000 crossed/locked/non-monotone rates
in this chart against 54.4% crossed when the same model trains in raw
coordinates.

Differences from BookDiT, both due to the wall-clock grid of ``frames.py``:

    * no delta-time channel — on a fixed grid the inter-frame gap is the
      constant ``interval_ns``, so the channel would carry zero information;
    * an optional activity channel replaces it — log1p(events per frame) is
      exactly the information the fixed clock folds away.

``min_gap`` floors spread/gap arguments before the log.  Its scale is the
data's price unit: 1.0 for raw LOBSTER integer prices (real gaps >= 100), and
it MUST be re-chosen for rescaled data — a floor above the typical gap silently
flattens the whole ladder to log(min_gap).

The torch decode is differentiable (exp / cumsum), which is what lets ReFL and
DMDR push reward gradients from raw-space rewards back through the chart.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def chart_channels(levels: int, with_activity: bool = False) -> int:
    return 4 * levels + (1 if with_activity else 0)


def encode_frames(
    book: np.ndarray,
    events_per_frame: Optional[np.ndarray] = None,
    min_gap: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Raw book frames (T, 4L) -> chart (T, 4L [+1]), plus the mid anchor.

    The anchor (first-frame mid) is what the decoder needs to place the
    absolute price level; everything else in the chart is relative.
    """
    book = np.asarray(book, dtype=np.float64)
    if book.ndim != 2 or book.shape[1] % 4 != 0:
        raise ValueError(f"expected (T, 4*levels) raw frames, got {book.shape}")
    L = book.shape[1] // 4
    ask = book[:, [4 * l for l in range(L)]]
    bid = book[:, [4 * l + 2 for l in range(L)]]

    chart = book.copy()
    mid = 0.5 * (ask[:, 0] + bid[:, 0])
    d_mid = np.zeros_like(mid)
    d_mid[1:] = mid[1:] - mid[:-1]

    chart[:, 0] = np.log(np.maximum(ask[:, 0] - bid[:, 0], min_gap))
    chart[:, 2] = d_mid
    for k in range(1, L):
        chart[:, 4 * k] = np.log(np.maximum(ask[:, k] - ask[:, k - 1], min_gap))
        chart[:, 4 * k + 2] = np.log(np.maximum(bid[:, k - 1] - bid[:, k], min_gap))

    if events_per_frame is not None:
        act = np.log1p(np.asarray(events_per_frame, dtype=np.float64))
        chart = np.concatenate([chart, act[:, None]], axis=1)

    return chart.astype(np.float32), float(mid[0])


def decode_frames(chart: np.ndarray, anchor_mid: float) -> np.ndarray:
    """Inverse of ``encode_frames`` (numpy, exact); drops the activity channel."""
    chart = np.asarray(chart, dtype=np.float64)
    nb = chart.shape[1] // 4 * 4
    L = nb // 4
    out = chart[:, :nb].copy()

    mid = np.cumsum(chart[:, 2]) + anchor_mid
    half = 0.5 * np.exp(chart[:, 0])
    ask = mid + half
    bid = mid - half
    out[:, 0], out[:, 2] = ask, bid
    for k in range(1, L):
        ask = ask + np.exp(chart[:, 4 * k])
        bid = bid - np.exp(chart[:, 4 * k + 2])
        out[:, 4 * k], out[:, 4 * k + 2] = ask, bid
    return out.astype(np.float32)


def decode_frames_torch(chart: torch.Tensor, anchor_mid: torch.Tensor) -> torch.Tensor:
    """Differentiable decode, batched: (B, T, 4L [+1]) -> (B, T, 4L).

    anchor_mid is (B,) — per-sequence first-frame mid.  Gradients flow through
    exp and cumsum, so a raw-space reward trains a chart-space model.
    """
    if chart.dim() != 3:
        raise ValueError(f"expected (B, T, C), got {tuple(chart.shape)}")
    nb = chart.shape[-1] // 4 * 4
    L = nb // 4
    body = chart[..., :nb]

    mid = torch.cumsum(body[..., 2], dim=1) + anchor_mid.view(-1, 1)
    half = 0.5 * torch.exp(body[..., 0])
    ask = mid + half
    bid = mid - half

    cols = list(torch.unbind(body, dim=-1))
    cols[0], cols[2] = ask, bid
    for k in range(1, L):
        ask = ask + torch.exp(body[..., 4 * k])
        bid = bid - torch.exp(body[..., 4 * k + 2])
        cols[4 * k], cols[4 * k + 2] = ask, bid
    return torch.stack(cols, dim=-1)


class ChartNormalizer:
    """Global per-channel z-score, the BookDiT normalisation: one mean and one
    std per channel over the whole training split, shipped with the checkpoint
    so no side files are needed."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, std_floor: float = 1e-6):
        mean = np.asarray(mean, dtype=np.float64)
        std = np.maximum(np.asarray(std, dtype=np.float64), std_floor)
        if mean.shape != std.shape or mean.ndim != 1:
            raise ValueError("mean/std must be matching 1-D vectors")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, chart: np.ndarray) -> "ChartNormalizer":
        return cls(chart.mean(axis=0), chart.std(axis=0))

    def normalize(self, chart: np.ndarray) -> np.ndarray:
        return ((chart - self.mean) / self.std).astype(np.float32)

    def denormalize(self, chart: np.ndarray) -> np.ndarray:
        return (chart * self.std + self.mean).astype(np.float32)

    def denormalize_torch(self, chart: torch.Tensor) -> torch.Tensor:
        std = torch.as_tensor(self.std, device=chart.device, dtype=chart.dtype)
        mean = torch.as_tensor(self.mean, device=chart.device, dtype=chart.dtype)
        return chart * std + mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "std": self.std.copy()}
