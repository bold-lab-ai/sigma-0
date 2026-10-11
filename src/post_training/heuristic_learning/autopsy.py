"""What exactly went wrong at the step where a replay first disagreed.

This is a description tool. It changes nothing, fixes nothing, and proposes no
repair. It exists because docs/fidelity_defect_taxonomy.md narrowed the gen
arm's onset to "class R or class O" and then stopped, and a defect register
whose central entry says "somewhere in these two classes" has not finished
describing the problem.

Two design decisions carry the whole thing.

**Books are compared as price -> quantity maps, not as positional arrays.**
Deleting a level shifts every level below it up by one, so a positional
comparison reports a wall of differences for a single removal, and the real
disagreement is buried. Comparing by price makes a one-order change look like
a one-order change.

**The deepest level is excluded, and that only helps partially.** An L2
snapshot is truncated, so activity below the window pushes new levels into view
at the bottom edge, which is an artefact rather than a disagreement. Dropping
the deepest level suppresses the case where only that slot moves. It cannot
suppress the general case: remove the top level of an L-level book keeping
L-1, and the kept set goes from {p0..p_{L-2}} to {p1..p_{L-1}}, so one edge
entry appears no matter how many levels are dropped.

Do not read a single autopsy as proof of a mechanism for that reason. The
claim that depth truncation drives the real arm rests on the narrowing sweep
(first divergence moves later, monotonically, as the comparison window
shrinks) and on the divergence price lying outside the initial visible range.
Both are robust to this edge effect; a single step's signature is not.

Signatures are reported together rather than resolved into one verdict,
because more than one can hold at a step and picking one would be a guess:

    RECORDED_NOT_APPLIED   ours moved, the generator's book did not. The
                           generator wrote a message it never applied: D-R1.
    APPLIED_NOT_RECORDED   the generator's book moved, ours did not.
    DIFFERENT_PRICE        both moved, at disjoint prices. The message's price
                           and its order id name different orders: D-R3.
    DIFFERENT_AMOUNT       both moved at the same price, by different amounts.
    AGGRESSIVE_EXECUTION   a type 4. Which resting orders it consumes depends
                           on within-level queue position, which an L2 snapshot
                           does not carry: D-O1, a bound rather than a bug.
    MISS_SENTINEL          the id is the lookup-miss sentinel, which at
                           book_depth >= 49 is also a valid init id: D-R2, D-R6.
    OUT_OF_DEPTH           the id names an init order this replay cannot have
                           created: D-I1.

`python -m post_training.heuristic_learning.autopsy --episodes DIR --stock GS`
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from post_training.heuristic_learning.episode_builder import (
    fixture_paths,
    read_book_csv,
    read_message_csv,
)
from post_training.heuristic_learning.fidelity import (
    LOOKUP_MISS,
    discover_windows,
    init_id_span,
    replay_stream,
)

EVENT_NAMES = {1: "NEW", 2: "PARTIAL_CANCEL", 3: "DELETE", 4: "EXECUTION"}


# ------------------------------------------------------------------- books --
def book_map(row: np.ndarray, drop_deepest: int = 1) -> Dict[Tuple[str, int], int]:
    """(side, price) -> quantity, with the truncation edge dropped.

    Layout is [ask_p, ask_q, bid_p, bid_q] per level. Empty levels carry a
    sentinel price (maxint on asks, -1 on bids) and are skipped rather than
    treated as a price of their own.
    """
    n = len(row) // 4 - drop_deepest
    out: Dict[Tuple[str, int], int] = {}
    for i in range(max(n, 0)):
        ap, aq, bp, bq = (int(row[4 * i]), int(row[4 * i + 1]),
                          int(row[4 * i + 2]), int(row[4 * i + 3]))
        if 0 < ap < 2 ** 31 - 1:
            out[("ask", ap)] = out.get(("ask", ap), 0) + aq
        if bp > 0:
            out[("bid", bp)] = out.get(("bid", bp), 0) + bq
    return out


def delta(before: Dict, after: Dict) -> Dict[Tuple[str, int], int]:
    """Signed quantity change per (side, price). Absent means zero."""
    keys = set(before) | set(after)
    d = {k: after.get(k, 0) - before.get(k, 0) for k in keys}
    return {k: v for k, v in d.items() if v != 0}


# --------------------------------------------------------------- diagnosis --
def diagnose(msg: np.ndarray, ref_d: Dict, our_d: Dict, n_levels: int) -> List[str]:
    sigs: List[str] = []
    etype, oid = int(msg[0]), int(msg[4])
    lo, _ = init_id_span(n_levels)

    if our_d and not ref_d:
        sigs.append("RECORDED_NOT_APPLIED")
    if ref_d and not our_d:
        sigs.append("APPLIED_NOT_RECORDED")
    if ref_d and our_d:
        shared = set(ref_d) & set(our_d)
        if not shared:
            sigs.append("DIFFERENT_PRICE")
        elif any(ref_d[k] != our_d[k] for k in shared) or set(ref_d) != set(our_d):
            sigs.append("DIFFERENT_AMOUNT")

    if etype == 4:
        sigs.append("AGGRESSIVE_EXECUTION")
    if oid == LOOKUP_MISS:
        sigs.append("MISS_SENTINEL")
    elif oid < lo:
        sigs.append("OUT_OF_DEPTH")
    if not sigs:
        sigs.append("UNCLASSIFIED")
    return sigs


def autopsy_window(run_dir: Path, stock: str, date: str, index: int,
                   arm: str = "gen") -> Optional[Dict[str, Any]]:
    p = fixture_paths(run_dir, stock, date, index)
    mk, bk = ("real_msg", "real_book") if arm == "real" else ("gen_msg", "gen_book")
    if not (p[mk].is_file() and p[bk].is_file()):
        return None

    cond_book = read_book_csv(p["cond_book"])
    n_levels = cond_book.shape[1] // 4
    init_book = cond_book[-1]
    cond_msgs = read_message_csv(p["cond_msg"])
    prior_ids = cond_msgs[cond_msgs[:, 0] == 1, 4].tolist()

    msgs = read_message_csv(p[mk])
    ref = read_book_csv(p[bk])
    ours, _ = replay_stream(msgs, init_book, n_levels, prior_ids)

    n = min(len(ours), len(ref))
    ref_maps = [book_map(ref[i]) for i in range(n)]
    our_maps = [book_map(ours[i]) for i in range(n)]

    k = next((i for i in range(n) if ref_maps[i] != our_maps[i]), -1)
    if k < 0:
        return {"window": f"{date} #{index}", "arm": arm, "first_step": -1,
                "signatures": [], "n_levels": n_levels}

    init_map = book_map(np.asarray(init_book))
    ref_prev = ref_maps[k - 1] if k > 0 else init_map
    our_prev = our_maps[k - 1] if k > 0 else init_map
    ref_d, our_d = delta(ref_prev, ref_maps[k]), delta(our_prev, our_maps[k])
    msg = msgs[k]

    disagree = delta(our_maps[k], ref_maps[k])
    return {
        "window": f"{date} #{index}",
        "arm": arm,
        "n_levels": n_levels,
        "first_step": k,
        "message": {
            "type": int(msg[0]), "type_name": EVENT_NAMES.get(int(msg[0]), "?"),
            "side": int(msg[1]), "qty": int(msg[2]), "price": int(msg[3]),
            "oid": int(msg[4]),
        },
        "ref_delta": dict(sorted(ref_d.items())),
        "our_delta": dict(sorted(our_d.items())),
        "disagreement": dict(sorted(disagree.items())),
        "signatures": diagnose(msg, ref_d, our_d, n_levels),
    }


# ----------------------------------------------------------------- reports --
def _fmt(d: Dict) -> str:
    if not d:
        return "(nothing moved)"
    return "  ".join(f"{s}@{p}{v:+d}" for (s, p), v in list(d.items())[:5])


def _print_one(a: Dict[str, Any]) -> None:
    if a["first_step"] < 0:
        print(f"{a['window']:22s} {a['arm']:5s}  no divergence")
        return
    m = a["message"]
    print(f"\n{a['window']}  arm={a['arm']}  step {a['first_step']}")
    print(f"  message   type={m['type']}({m['type_name']}) side={m['side']:+d} "
          f"qty={m['qty']} price={m['price']} oid={m['oid']}")
    print(f"  generator {_fmt(a['ref_delta'])}")
    print(f"  replay    {_fmt(a['our_delta'])}")
    print(f"  disagree  {_fmt(a['disagreement'])}")
    print(f"  -> {', '.join(a['signatures'])}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="first-divergence autopsy")
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--stock", default="GS")
    ap.add_argument("--arm", default="gen", choices=("gen", "real", "both"))
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args(argv)

    run_dir = Path(args.episodes)
    windows = discover_windows(run_dir, args.stock, args.n)
    if not windows:
        print(f"no windows under {run_dir / 'data_gen'}")
        return 1

    arms = ("real", "gen") if args.arm == "both" else (args.arm,)
    tally: Counter = Counter()
    steps: List[int] = []
    for date, idx in windows:
        for arm in arms:
            a = autopsy_window(run_dir, args.stock, date, idx, arm)
            if a is None:
                continue
            _print_one(a)
            tally.update(a["signatures"])
            if a["first_step"] >= 0:
                steps.append(a["first_step"])

    print("\nsignature tally (a step can carry several):")
    for s, c in tally.most_common():
        print(f"  {c:3d}  {s}")
    if steps:
        print(f"\nfirst divergence step: min {min(steps)}  median "
              f"{int(np.median(steps))}  max {max(steps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
