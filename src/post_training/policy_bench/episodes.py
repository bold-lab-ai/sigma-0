"""Which (stock-day, row) pairs the bench runs.

`return_bench_decisions` is nano-sigma's test decision set
(tools/return_features.py, build_windows): every `stride` rows from the open
of each stock-day, keeping the span prefill + 1 + max(horizons) clear of the
entry's bad_rows and exclude ranges. On the January test days that is 70,183
decisions (gate G4).
"""

from __future__ import annotations

STRIDE = 50_000       # tools/return_features.py --test_stride


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
