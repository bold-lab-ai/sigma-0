#!/usr/bin/env python3
"""G1 and G2 of the policy bench on real stock-days (CPU only).

G1, the book. Replay a whole day by order id from the first full-book
snapshot and compare, at every row:
    our queue slot with <stem>_queue.npy column 0
    our top 10 levels with the processed L10 row
and, at every later snapshot, every order (id, size) with the stored snapshot.

G2, the codec and the rules. On the same day:
    encode_stream gives exactly nano-sigma's encode_window tokens
    decode(encode(m)) returns m up to the documented clipping
    t = t_prev + dt holds
    rules.check accepts real messages (reasons counted per event type)
    a second book driven the way generated messages drive it (new orders by
      price, cancels and deletes by queue slot, never by order id) stays
      equal to the first

Executions are the exception. The bench fills them front of queue first, but
about 4% of real executions skip an older order at the level (job 7832642).
So the second book takes the real fills, and the share of executions whose
real fills equal the front-first prediction is reported as g2_exec_fifo_ok.

Usage, with the month shards mounted (see check_book_codec.sbatch):

    python run/post_training/policy_bench/check_book_codec.py \
        --shard_root <mnt> --manifest <v3>/test.json --out <file.json> --n_days 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool

import numpy as np

from post_training.policy_bench import rules
from post_training.policy_bench.book import ASK, BID, Book, side_of_direction
from post_training.policy_bench.codec import (
    DecodeError, Message, burst_bounds, decode, encode, encode_stream, message_from_rows, mid_floor,
)
from post_training.policy_bench.daydata import DayFiles
from post_training.policy_bench.nanosigma.newtok import encode_window

SHARD_ROOT = ""


def _same_book(a: Book, b: Book) -> bool:
    return a.levels(ASK) == b.levels(ASK) and a.levels(BID) == b.levels(BID)


def _roundtrip_reason(m: Message, back: Message, mid: int) -> str:
    """Why decode(encode(m)) differs from m: one of the documented clips, or a bug."""
    if back == m:
        return "exact"
    if (back.evt, back.d, back.t_ns, back.dt_ns, len(back.levels)) != (m.evt, m.d, m.t_ns, m.dt_ns, len(m.levels)):
        return "bug"
    if back.t_s != m.t_s:
        return "bug"
    tags = []
    if back.dt_s != m.dt_s:
        tags.append("dt_s_clip" if m.dt_s > 999 and back.dt_s == 999 else "bug")
    for (p, q), (p2, q2) in zip(m.levels, back.levels):
        if q2 != q:
            tags.append("size_clip" if q > 9999 and q2 == 9999 else "bug")
        if p2 != p:
            dp = (p - mid) // 100
            tags.append("dp_clip" if abs(dp) > 999_999 else "price_off_tick" if p % 100 else "bug")
    if back.qslot != m.qslot:
        tags.append("qslot_clip" if m.qslot > 99 and back.qslot == 99 else "bug")
    return "bug" if "bug" in tags else "+".join(sorted(set(tags)))


def _front_first(book: Book, m: Message) -> Counter:
    """Fills (order id -> qty) if each level is taken front of queue first."""
    out = Counter()
    side = side_of_direction(m.d)
    for price, qty in m.levels:
        for o in book.queue(side, price):
            if qty <= 0:
                break
            take = min(qty, o.size)
            out[o.oid] += take
            qty -= take
    return out


def _real_fills(rows: np.ndarray) -> Counter:
    out = Counter()
    for r in rows:
        out[int(r[0])] += int(r[5])
    return out


def _l10_hypotheses(msg: np.ndarray, l10: np.ndarray) -> dict:
    """Which message do columns 0..2 of an L10 row describe? Counted on the file."""
    n = msg.shape[0]
    t_prev = np.all(l10[1:n, 1:3] == msg[:n - 1, 8:10], axis=1)
    t_cur = np.all(l10[1:n, 1:3] == msg[1:n, 8:10], axis=1)
    ask, bid = l10[:n, 3].astype(np.float64), l10[:n, 5].astype(np.float64)
    ask[ask >= 9_999_999_999] = np.nan
    bid[bid <= -9_999_999_999] = np.nan
    mid = np.round((_ffill(ask) + _ffill(bid)) / 2, -2) / 100
    dmid = np.diff(mid)
    ok = ~np.isnan(dmid)
    return dict(rows=int(n - 1), time_is_prev_msg=int(t_prev.sum()), time_is_this_msg=int(t_cur.sum()),
                dmid_rows=int(ok.sum()), dmid_is_rounded_mid_change=int((l10[1:n, 0][ok] == dmid[ok]).sum()))


def _ffill(x: np.ndarray) -> np.ndarray:
    idx = np.where(np.isnan(x), 0, np.arange(len(x)))
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def check_day(entry: dict) -> dict:
    t0 = time.time()
    day = DayFiles.open(SHARD_ROOT, entry)
    n = day.n_msgs
    snap = day.snapshot_rows()
    if int(snap[0]) != 0:
        return dict(entry=entry["message"], skipped="first snapshot is not before row 0")
    msg = np.asarray(day.messages[:n])                 # clipped, as the model saw it
    true = day.rows(0, n)                              # sizes unclipped, for the books
    l10 = np.asarray(day.l10[:n + 1])
    queue = np.asarray(day.queue[:n])

    res = Counter()
    res["rows"] = n
    snap_index = {int(r): s for s, r in enumerate(snap)}

    # ---------------- G2 parity, whole day in one call
    ref = encode_window(msg, l10[:n], queue[:, 0])
    got = encode_stream(msg, l10[:n], queue[:, 0])
    res["g2_parity_ok"] = int(all(np.array_equal(a, b) for a, b in zip(ref, got)))
    res["g2_tokens"] = int(len(ref[0]))

    bk1 = Book.from_snapshot(day.snapshots, 0)        # G1: by order id
    bk2 = Book.from_snapshot(day.snapshots, 0)        # G2: by slot, as generated messages are
    gen = day.replay(bk1, 0, n)
    extras = list(day.extras_before(1, n + 1))
    xj = 0
    legality = Counter()
    roundtrip = Counter()
    first_bug = None

    for i, j in burst_bounds(msg):
        # bk2 gets every extra that precedes row i
        while xj < len(extras) and extras[xj][0] <= i:
            bk2.apply_row(extras[xj][1])
            xj += 1

        # ---- G2 at the message boundary: both books are the book before row i
        res["g2_messages"] += 1
        if not np.array_equal(bk2.l10(), l10[i, 3:43]):
            res["g2_book2_l10_bad"] += 1
        s = snap_index.get(i)
        if s is not None and i > 0:
            res["g2_snapshots"] += 1
            res["g2_snapshot_bad"] += int(not _same_book(bk2, Book.from_snapshot(day.snapshots, s)))
        m = message_from_rows(msg[i:j], int(queue[i, 0]))
        mid = mid_floor(l10[i, 3], l10[i, 5])
        try:
            back = decode(encode(m, mid), mid)
            reason = _roundtrip_reason(m, back, mid)
        except DecodeError as e:
            back, reason = None, "bug"
            first_bug = first_bug or f"row {i}: {e}"
        roundtrip[reason] += 1
        if reason == "bug" and first_bug is None:
            first_bug = f"row {i}: {m} -> {back}"
        if i > 0:
            res["g2_time_checked"] += 1
            res["g2_time_ok"] += int(rules.time_consistent(m, int(msg[i - 1, 8]) * 10**9 + int(msg[i - 1, 9])))
        if back is not None:
            legality[f"evt{m.evt}:{rules.check(back, bk1) or 'legal'}"] += 1

        # ---- drive bk2 with the true message: by slot, executions by their real fills
        tm = message_from_rows(true[i:j], int(queue[i, 0]))
        try:
            if m.evt == 1:
                rules.apply(tm, bk2, oid=int(true[i, 0]))
            elif m.evt in (2, 3):
                if queue[i, 0] >= 0:
                    rules.apply(tm, bk2, oid=-1, slot=int(queue[i, 0]))
            else:
                res["g2_exec_msgs"] += 1
                res["g2_exec_multi_level"] += int(len(tm.levels) > 1)
                res["g2_exec_fifo_ok"] += int(_front_first(bk2, tm) == _real_fills(true[i:j]))
                for r in range(i, j):
                    bk2.apply_row(true[r])
        except (ValueError, IndexError, KeyError):
            res[f"g2_apply_failed_evt{m.evt}"] += 1

        # ---- G1, row by row through the message
        for r in range(i, j):
            if r > i:
                while xj < len(extras) and extras[xj][0] <= r:
                    res["g2_extra_inside_burst"] += 1
                    bk2.apply_row(extras[xj][1])
                    xj += 1
            rr, slot = next(gen)
            assert rr == r
            res["g1_slot_checked"] += 1
            res["g1_slot_bad"] += int(slot != int(queue[r, 0]))
            res["g1_l10_checked"] += 1
            res["g1_l10_bad"] += int(not np.array_equal(bk1.l10(), l10[r + 1, 3:43]))
            s = snap_index.get(r + 1)
            if s is not None:
                ref_bk = Book.from_snapshot(day.snapshots, s)
                res["g1_snapshots"] += 1
                res["g1_snapshot_bad"] += int(not _same_book(bk1, ref_bk))

    out = dict(entry=entry["message"], ticker=entry["ticker"], date=entry["date"], seconds=round(time.time() - t0, 1),
               counts=dict(res), legality=dict(legality), roundtrip=dict(roundtrip),
               l10_columns=_l10_hypotheses(msg, l10), first_bug=first_bug)
    return out


def _safe(entry):
    try:
        return check_day(entry)
    except Exception:
        return dict(entry=entry["message"], error=traceback.format_exc())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shard_root", required=True)
    p.add_argument("--manifest", required=True, help="a v3 manifest split, e.g. test.json")
    p.add_argument("--out", required=True)
    p.add_argument("--n_days", type=int, default=40)
    p.add_argument("--max_day_msgs", type=int, default=2_000_000, help="skip larger days (memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args(argv)

    global SHARD_ROOT
    SHARD_ROOT = a.shard_root
    entries = [e for e in json.load(open(a.manifest)) if e["n_msgs"] <= a.max_day_msgs]
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(entries), size=min(a.n_days, len(entries)), replace=False).tolist())
    days = [entries[k] for k in pick]
    print(f"{len(days)} stock-days from {a.manifest}", flush=True)

    with Pool(min(a.workers, len(days))) as pool:
        per_day = []
        for r in pool.imap_unordered(_safe, days):
            per_day.append(r)
            print(json.dumps({k: r.get(k) for k in ("entry", "seconds", "error", "skipped")}), flush=True)

    total = Counter()
    legality = Counter()
    roundtrip = Counter()
    l10c = Counter()
    for r in per_day:
        total.update(r.get("counts", {}))
        legality.update(r.get("legality", {}))
        roundtrip.update(r.get("roundtrip", {}))
        l10c.update(r.get("l10_columns", {}))
    total["days_ok"] = sum("counts" in r for r in per_day)
    total["days_failed"] = sum("error" in r for r in per_day)
    summary = dict(args=vars(a), total=dict(total), legality=dict(sorted(legality.items())),
                   roundtrip=dict(roundtrip), l10_columns=dict(l10c), per_day=per_day)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("total", "legality", "roundtrip", "l10_columns")}, indent=1))
    return 0 if total["days_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
