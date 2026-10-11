"""Windowed frame datasets over resampled book states.

The pipeline this feeds:

    wide-book npy (event level, 43 cols)
        --frames.resample_wide_book-->   FrameSeries (1 ms grid by default)
        --representation.encode_frames-->  chart (T, C) + anchor_mid
        --ChartNormalizer-->             model space
        --FrameWindowDataset-->          (seq_len, C) windows

A window is ``seq_len`` consecutive frames; on the default clock that is
``seq_len`` milliseconds of market time — the model's horizon is defined in
wall time, not in event count.  Context/horizon splitting happens in the
training loop (per-sample random context length), not here, so one dataset
serves pretraining and every post-training method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .frames import DEFAULT_INTERVAL_NS, FrameSeries, resample_wide_book
from .representation import ChartNormalizer, encode_frames


@dataclass
class EncodedDay:
    chart: np.ndarray  # (T, C) normalised chart frames
    anchor_mid: float
    interval_ns: int


def encode_series(
    series: FrameSeries,
    normalizer: Optional[ChartNormalizer],
    min_gap: float = 1.0,
    with_activity: bool = False,
) -> EncodedDay:
    chart, anchor = encode_frames(
        series.frames,
        events_per_frame=series.events_per_frame if with_activity else None,
        min_gap=min_gap,
    )
    if normalizer is not None:
        chart = normalizer.normalize(chart)
    return EncodedDay(chart=chart, anchor_mid=anchor, interval_ns=series.interval_ns)


def load_wide_book_day(
    path: str,
    interval_ns: int = DEFAULT_INTERVAL_NS,
) -> FrameSeries:
    """One event-level npy file (43-column contract) -> frame series."""
    wide = np.load(path, mmap_mode="r")
    return resample_wide_book(np.asarray(wide), interval_ns=interval_ns)


class FrameWindowDataset(Dataset):
    """Non-overlapping-stride windows over one or more encoded days.

    Returns (window, anchor_mid) with window a float32 tensor (seq_len, C).
    The anchor is the window's own first-frame mid so decoded books are
    absolutely placed per window.
    """

    def __init__(self, days: Sequence[EncodedDay], seq_len: int, stride: Optional[int] = None):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.days = list(days)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        self.index: List[tuple[int, int]] = []
        for d_idx, day in enumerate(self.days):
            T = len(day.chart)
            for start in range(0, T - seq_len + 1, self.stride):
                self.index.append((d_idx, start))
        if not self.index:
            raise ValueError(
                f"no window of {seq_len} frames fits any provided day "
                f"(day lengths: {[len(d.chart) for d in self.days]})"
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        d_idx, start = self.index[i]
        day = self.days[d_idx]
        window = day.chart[start : start + self.seq_len]
        # Re-anchor: d_mid at the window's first frame becomes 0 so every
        # window is translation-free; the absolute level rides in the anchor.
        window = window.copy()
        return torch.from_numpy(window), float(day.anchor_mid)


def fit_normalizer(
    series_list: Sequence[FrameSeries],
    min_gap: float = 1.0,
    with_activity: bool = False,
) -> ChartNormalizer:
    """Global per-channel statistics over the (training) days, BookDiT-style."""
    charts = [
        encode_frames(
            s.frames,
            events_per_frame=s.events_per_frame if with_activity else None,
            min_gap=min_gap,
        )[0]
        for s in series_list
    ]
    return ChartNormalizer.fit(np.concatenate(charts, axis=0))
