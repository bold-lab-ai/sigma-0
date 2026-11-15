"""The bench's data splits, by date.

Two layers of data meet here. The frozen models (simulator newtok_a_r1, signal
returnft_lobert_pass1) were trained on dataset_q4_2025_jan2026_v3: train
2025-10 to 2025-12 (validation windows inside those months), test 2026-01.
On their training months the signal is in-sample and too good, so no policy
data may come from before MODEL_DATA_END.

The policy splits all lie after it, in time order, split by whole days
(episodes of one date share the market, so a date belongs to one split only):

  train  2026-01-02 .. 2026-01-15  fit a policy's parameters
  dev    2026-01-16 .. 2026-01-30  compare and select policies; never fit on it
  test   2026-02-02 .. 2026-02-13  score a frozen policy, once

train and dev are the models' test month; test was built for the bench and no
model has seen it.
"""

from __future__ import annotations

MODEL_DATA_END = "2025-12-31"

SPLITS = {
    "train": ("2026-01-02", "2026-01-15"),
    "dev": ("2026-01-16", "2026-01-30"),
    "test": ("2026-02-02", "2026-02-13"),
}


def split_of(date: str) -> str | None:
    """The split a date (YYYY-MM-DD) belongs to, or None."""
    for name, (lo, hi) in SPLITS.items():
        if lo <= date <= hi:
            return name
    return None


def in_split(entries: list[dict], split: str) -> list[dict]:
    """The manifest entries whose date is in `split`."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; one of {sorted(SPLITS)}")
    return [e for e in entries if split_of(e["date"]) == split]


def check_episodes(eps, split: str) -> None:
    """Raise unless every episode's date is in `split`."""
    bad = sorted({ep.entry["date"] for ep in eps if split_of(ep.entry["date"]) != split})
    if bad:
        raise ValueError(f"{len(bad)} date(s) outside the {split} split: {bad[:5]}")
