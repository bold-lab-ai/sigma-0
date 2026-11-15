"""Which (stock-day, row) pairs the bench runs.

`return_bench_decisions` is nano-sigma's test decision set
(tools/return_features.py, build_windows): every `stride` rows from the open
of each stock-day, keeping the span prefill + 1 + max(horizons) clear of the
entry's bad_rows and exclude ranges. On the January test days that is 70,183
decisions (gate G4).

`sample_episodes` draws the bench's episode sets: a stock-day uniformly, then a
start row uniformly, kept if the rows from start - history to start + n_messages
are clear of bad rows, the book at the start has both sides, and the start
time is inside [t_lo, t_hi] (by default 9:35 to 15:30, away from the opening
auction and with room before the close).
The same seed gives the same set; which dates a set may use is fixed by splits.py.
"""

from __future__ import annotations

import os

import numpy as np

STRIDE = 50_000       # tools/return_features.py --test_stride
T_LO, T_HI = 34_500, 55_800


def return_bench_decisions(entries: list[dict], prefill: int, horizons: list[int],
                           stride: int = STRIDE) -> list[tuple[dict, int]]:
    """[(entry, s), ...]: the window starts s; the decision is at row s + prefill."""
    span = prefill + 1 + max(horizons)
    out = []
    for e in entries:
        bad = list(e.get("bad_rows", [])) + list(e.get("exclude", []))
        for s in range(0, e["n_msgs"] - span + 1, stride):
            if not any(s < t and b < s + span for b, t in bad):
                out.append((e, s))
    return out


def _clear(e: dict, lo: int, hi: int) -> bool:
    bad = list(e.get("bad_rows", [])) + list(e.get("exclude", []))
    return not any(lo < t and b < hi for b, t in bad)


def sample_episodes(entries: list[dict], shard_root: str, n: int, seed: int, history: int = 3968,
                    n_messages: int = 1024, t_lo: int = T_LO, t_hi: int = T_HI,
                    max_tries: int = 100) -> list[dict]:
    """[{message, month, start, seed}, ...]: n episodes, each a stock-day and a start row."""
    rng = np.random.default_rng(seed)
    days = [e for e in entries if e["n_msgs"] > history + n_messages]
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > max_tries * n:
            raise RuntimeError(f"{len(out)} of {n} episodes after {tries} draws")
        e = days[int(rng.integers(len(days)))]
        start = int(rng.integers(history, e["n_msgs"] - n_messages))
        if not _clear(e, start - history, start + n_messages):
            continue
        base = os.path.join(shard_root, e["month"])
        t = int(np.load(os.path.join(base, e["message"]), mmap_mode="r")[start, 8])
        book = np.load(os.path.join(base, e["orderbook"]), mmap_mode="r")[start]
        if t_lo <= t < t_hi and 0 < int(book[5]) < int(book[3]) < 9_000_000_000:
            out.append(dict(message=e["message"], month=e["month"], start=start, seed=len(out)))
    return out


def load_episodes(path: str, entries: list[dict]):
    """The Episodes of a sample_episodes file, with their manifest entries."""
    import json

    from lobpipeline.policy_bench.types import Episode
    by_key = {(e["month"], e["message"]): e for e in entries}
    return [Episode(by_key[(x["month"], x["message"])], int(x["start"]), int(x["seed"]))
            for x in json.load(open(path))["episodes"]]
