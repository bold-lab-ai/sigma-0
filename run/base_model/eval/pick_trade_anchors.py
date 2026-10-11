"""Pick conditioning windows that end on a trade, for event-anchored evaluation.

Writes a JSON file of explicit (file_idx, seq_start) window positions, to be handed
to inference via --window_starts_file.

Why this exists: LOBSTER_Dataset places windows on a fixed non-overlapping grid, so
what a conditioning window ends on is decided by the grid rather than chosen. Over
2000 GOOG windows only 1.95% end on a trade (49.4% end on a submission, 48.5% on a
cancellation), so selecting dataset indices cannot produce an event-anchored sample
and choosing window starts directly is the only way.

Pure CPU: it reads .npy headers and two columns per file. Safe on a login node.

    python run/base_model/eval/pick_trade_anchors.py \
        --data_dir <preprocessed dir> --n_samples 1024 --out anchors.json

Column layout of *_proc.npy (encoding_26tok.py:490-491):

    0 order_id | 1 event_type | 2 direction | 3 price_abs | 4 price | 5 size
    6 delta_t_s | 7 delta_t_ns | 8 time_s | 9 time_ns | 10-13 *_ref

Note direction is {0, 1} at this layer, where 1 means the resting order sat on the
bid. Inference *output* csv uses {-1, +1} for the same thing; do not share a constant
between the two.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from base_model.data.data_files import resolve_data_files  # noqa: E402

EVENT_TYPE_COL = 1
DIRECTION_COL = 2
EVENT_TYPE_TRADE = 4

# Resting side at the preprocessed layer. encoding_26tok.py:487 maps LOBSTER's -1/+1
# through (d+1)/2, so 1 is the bid.
PROC_RESTING_BID = 1
PROC_RESTING_ASK = 0
SIDE_NAME = {PROC_RESTING_BID: "hit_bid_seller_initiated",
             PROC_RESTING_ASK: "lifted_ask_buyer_initiated"}


def one_file_per_day(msg_files):
    """Indices of the files to draw anchors from, at most one per trading day.

    The GOOG store holds some days twice, under names that differ only in the
    LOBSTER time range: GOOG_2026-02-06_24900000_57900000 and
    GOOG_2026-02-06_34200000_57600000. They are not different sessions. Both were
    truncated to 09:30-16:00 by preprocessing, both have 8,386,647 rows, and they
    agree on order_id, event_type, direction, size and every timestamp; only the
    relative `price` and `price_ref` columns differ, so they are two encodings of
    one day.

    Drawing from both makes the same execution appear as two independent anchors.
    On this data that affected 45% of a 1024-anchor draw, which inflates the
    sample count an IC's significance is judged against while looking like an
    ordinary larger sample.

    Note this filters which files are *drawn from*, never the file list itself:
    file_idx is a position in the list inference resolves, so removing entries
    here would point every window at the wrong day.
    """
    first = OrderedDict()
    for i, f in enumerate(msg_files):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        key = m.group(1) if m else os.path.basename(f)
        first.setdefault(key, i)
    return set(first.values()), first


def candidate_anchors(msg_file, book_file, n_cond_msgs, n_messages):
    """(seq_start, direction) for every trade that can anchor a full window.

    A window is usable when it fits inside both the message file and the book file.
    The dataset slices book as [seq_start : seq_end + 1] under inference=True, hence
    the extra row.
    """
    msgs = np.load(msg_file, mmap_mode="r")
    n_msg_rows = msgs.shape[0]
    n_book_rows = np.load(book_file, mmap_mode="r").shape[0]
    max_end = min(n_msg_rows, n_book_rows - 1)

    etype = np.asarray(msgs[:, EVENT_TYPE_COL])
    trade_rows = np.flatnonzero(etype == EVENT_TYPE_TRADE)
    if trade_rows.size == 0:
        return np.empty((0,), np.int64), np.empty((0,), np.int64), 0

    # Put the trade at the final conditioning position, so t0 is the moment it executes
    # and both sides of the comparison start from the resulting book.
    starts = trade_rows - n_cond_msgs + 1
    fits = (starts >= 0) & (starts + n_messages <= max_end)
    dirs = np.asarray(msgs[:, DIRECTION_COL])[trade_rows]
    return starts[fits].astype(np.int64), dirs[fits].astype(np.int64), int(trade_rows.size)


def stratified_take(rows, n_samples, rng):
    """Take n_samples with the two trade directions as close to balanced as possible.

    Balance matters because the whole point of splitting by direction is comparing
    buyer- against seller-initiated impact, and a 24/15 split (what an unstratified
    draw gave on the first 39 anchors) leaves the smaller group too thin to say
    anything about.
    """
    by_side = {}
    for r in rows:
        by_side.setdefault(r[2], []).append(r)

    picked = []
    sides = sorted(by_side)
    per_side = {s: len(by_side[s]) for s in sides}
    target = n_samples // max(1, len(sides))

    leftovers = []
    for s in sides:
        pool = by_side[s]
        rng.shuffle(pool)
        take = min(target, len(pool))
        picked.extend(pool[:take])
        leftovers.extend(pool[take:])

    # Any shortfall in one direction is made up from whatever remains, rather than
    # silently returning fewer samples than asked for.
    if len(picked) < n_samples and leftovers:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: n_samples - len(picked)])

    rng.shuffle(picked)
    return picked[:n_samples], per_side


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pick conditioning windows ending on a trade")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--n_cond_msgs", type=int, default=250)
    ap.add_argument("--n_gen_msgs", type=int, default=250)
    ap.add_argument("--n_samples", type=int, default=1024)
    ap.add_argument("--test_split", type=float, default=0.1,
                    help="must match the value inference uses, or file_idx points elsewhere")
    ap.add_argument("--day_indeces", type=str, default=None, help="comma-separated, optional")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-duplicate-days", action="store_true",
                    help="draw from every file, including days present twice under "
                         "different LOBSTER time ranges (see one_file_per_day)")
    args = ap.parse_args(argv)

    n_messages = args.n_cond_msgs + args.n_gen_msgs
    day_indeces = [int(x) for x in args.day_indeces.split(",")] if args.day_indeces else None

    msg_files, book_files, _ = resolve_data_files(
        args.data_dir, day_indeces=day_indeces, test_split=args.test_split)
    if not msg_files:
        print(f"ERROR: no data files under {args.data_dir}", file=sys.stderr)
        return 1
    print(f"Files          : {len(msg_files)} (test_split={args.test_split})")

    draw_from, first_by_day = one_file_per_day(msg_files)
    n_dup = len(msg_files) - len(draw_from)
    if args.allow_duplicate_days:
        draw_from = set(range(len(msg_files)))
        if n_dup:
            print(f"Duplicate days : {n_dup} extra file(s) INCLUDED (--allow-duplicate-days); "
                  f"the same execution can be drawn more than once")
    elif n_dup:
        print(f"Duplicate days : {n_dup} extra file(s) skipped for drawing "
              f"({len(first_by_day)} distinct trading days)")

    # Say the number out loud. Anchors inside one trading day share a regime and can
    # overlap the same order flow, so a day is the resampling unit for any error bar
    # on these results. When only a handful of days survive test_split, that estimate
    # silently stops working: one day gives a zero-width interval, a few give one that
    # is far too narrow. Both failures make a result look *stronger*, which is the
    # direction least likely to be questioned -- and both have already happened here.
    n_days = len(draw_from)
    if n_days < 5:
        print(f"WARNING        : only {n_days} trading day(s) will be drawn from. A "
              f"day-block bootstrap needs more;")
        print(f"                 with {n_days} it returns "
              f"{'a zero-width interval' if n_days == 1 else 'an interval that is far too narrow'}. "
              f"Raise --test_split or pass more months.")
        print(f"                 int() truncates: e.g. 17 files x 0.1 -> 1 file, not 2.")

    rows, total_trades, per_file = [], 0, []
    for file_idx, (mf, bf) in enumerate(zip(msg_files, book_files)):
        if file_idx not in draw_from:
            per_file.append((os.path.basename(mf), 0, 0))
            continue
        starts, dirs, n_trades = candidate_anchors(mf, bf, args.n_cond_msgs, n_messages)
        total_trades += n_trades
        per_file.append((os.path.basename(mf), n_trades, int(starts.size)))
        rows.extend((file_idx, int(s), int(d)) for s, d in zip(starts, dirs))

    print(f"Trades found   : {total_trades}")
    print(f"Usable anchors : {len(rows)}  (dropped {total_trades - len(rows)} that "
          f"cannot fit a {n_messages}-message window)")
    for name, n_t, n_u in per_file:
        print(f"    {name}: {n_t} trades -> {n_u} usable")

    if not rows:
        print("ERROR: no usable anchors", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    picked, per_side = stratified_take(rows, args.n_samples, rng)
    got = Counter(r[2] for r in picked)

    print(f"\nSelected       : {len(picked)} / {args.n_samples} requested")
    for side in sorted(per_side):
        print(f"    direction={side} ({SIDE_NAME.get(side, '?')}): "
              f"{got.get(side, 0)} selected of {per_side[side]} available")
    if len(picked) < args.n_samples:
        print(f"    NOTE: {args.n_samples - len(picked)} short; not enough anchors exist "
              f"in this data. Widen --test_split or use more days.")

    payload = {
        "data_dir": os.path.abspath(args.data_dir),
        "n_cond_msgs": args.n_cond_msgs,
        "n_gen_msgs": args.n_gen_msgs,
        "n_messages": n_messages,
        "test_split": args.test_split,
        "day_indeces": day_indeces,
        "seed": args.seed,
        "distinct_days_drawn_from": len(first_by_day),
        "allow_duplicate_days": bool(args.allow_duplicate_days),
        # Recorded so inference can assert it resolved the same list. file_idx is a
        # position in this list and means nothing without it.
        "message_files": [os.path.abspath(f) for f in msg_files],
        "windows": [[r[0], r[1]] for r in picked],
        "anchor_direction": [r[2] for r in picked],
        "direction_convention": "proc layer: 1=resting bid (seller-initiated), 0=resting ask (buyer-initiated)",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
