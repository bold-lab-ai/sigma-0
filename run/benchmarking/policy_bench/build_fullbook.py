#!/usr/bin/env python3
"""Build the full-book files of stock-days from the raw L0 messages (CPU only).

The bench reads, per stock-day, four files next to the processed messages
(daydata.py): <stem>_queue.npy, _book_full.npz, _extra.npz, _bigsize.npz.
They exist for 2025-10 to 2026-01 (lob_fullbook_q4_2025_jan2026, built by a
script that is not readable to us). This rebuilds them from the raw L0 day:

1. Raw rows: the L0 CSV of quant/lobster/<YYYY>/<YYYYMMDD>.squashfs, read as
   regen.py reads it (time split into s and ns, LOBSTER direction 1 buy, -1 sell).
2. Processed rows, by regen.py's rule: regular-hours (34,200 <= t < 57,600 s)
   events of type 1-4, minus far orders (price < R / 100 or > 100 R, R the mean
   price of those events), minus the first one. The rule is checked against the
   processed file: order id (int32), event, direction, price, size (<= 9,999)
   and time must be equal row by row. On a day whose raw messages begin after
   the open the file drops more at the start; the start is then taken from
   the file's first row (late_start).
3. One replay of the whole day (pre-market included) with book.Book, which
   keeps every order id in its FIFO queue: event 1 adds at the back, 2 takes
   shares, 3 deletes, 4 executes, 5 and 7 leave the book alone. Event 6 (a
   cross) leaves it alone too, except with --cross_cleanup, which drops the
   asks below and bids above a cross with a price. The existing queue and
   book_full files were made with that cleanup (AAPL 2026-01-28, INTC
   2026-01-15: asks below the opening cross that arrived in the same instant
   are marked unknown, jobs 8362586, 8362625); the processed L10 files
   without it (their top 10 keep those asks). A cross at price 0 (a listing
   on another exchange) never cleans: it would drop every bid (job 8358944).
4. Outputs, by processed row R:
   queue      (n, 2) int32 [slot, orders in the level] before row R, codes of
              book.apply_row (-1 for event 1; -2 in both columns for an order
              not in the book)
   book_full  every order of the book before rows 0, grid, 2 grid, ...: asks
              by ascending price, then bids by descending price, FIFO in a level
   extra      regular-hours events after the first processed row that the
              processed file dropped: type 1-4 far orders and type 6 crosses
              with a price (slot -1, the book unchanged; a listing on another
              exchange has its cross at price 0, job 8359428), with the row
              they precede and slot
   bigsize    rows whose raw size was clipped to 9,999, and the raw size
   replay     counts, and the share of rows whose top 10 levels before the row
              equal the processed L10 file

--compare_dir checks each built day against existing files of the same stem,
array by array; --compare_dir manifest reads them from each entry's fullbook_dir.

    python run/benchmarking/policy_bench/build_fullbook.py --shard_root <mount> --lobster <quant>/lobster \
        --manifest <file.json> --month 2026-01 --out_dir <dir> [--compare_dir <dir>] [--limit 20]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess as sp
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from lobpipeline.policy_bench.book import ASK, BID, Book

GRID = 4096
CLIP = 9999
RTH = (34200 * 10**9, 57600 * 10**9)
SEVENZ = "/projects/public/u6oz/quant/lob_preproc_sp500_squashfs_bookfix20261012/code/7zz"


def stem(entry: dict) -> str:
    return os.path.splitext(os.path.basename(entry["message"]))[0]


def raw_day(lobster: str, ticker: str, date: str, tmp: str) -> dict:
    """The raw L0 messages of one stock-day as arrays (regen.py's parsing)."""
    img = os.path.join(lobster, date[:4], date.replace("-", "") + ".squashfs")
    arc = os.path.join(tmp, f"{ticker}.7z")
    with open(arc, "wb") as f:
        sp.run(["unsquashfs", "-cat", img, f"{ticker}.7z"], stdout=f, check=True, stderr=sp.DEVNULL)
    names = [l.split()[-1] for l in sp.run([SEVENZ, "l", "-ba", arc], stdout=sp.PIPE, check=True,
                                           text=True).stdout.splitlines()]
    names = [n for n in names if n.endswith(".csv") and "_message_" in n]
    if len(names) != 1:
        raise RuntimeError(f"{arc}: message members {names}")
    data = sp.run([SEVENZ, "e", "-so", arc, names[0]], stdout=sp.PIPE, stderr=sp.DEVNULL, check=True).stdout
    os.remove(arc)
    df = pd.read_csv(io.BytesIO(data), header=None, usecols=range(6),
                     dtype={0: str, 1: np.int64, 2: np.int64, 3: np.int64, 4: np.int64, 5: np.int64})
    ts = df[0].str.split(".", expand=True)
    sec = ts[0].astype(np.int64).values
    ns = ts[1].str.ljust(9, "0").str[:9].astype(np.int64).values
    return dict(member=names[0], sec=sec, ns=ns, t=sec * 10**9 + ns, typ=df[1].values, oid=df[2].values,
                size=df[3].values, price=df[4].values, dirn=df[5].values)


def processed_index(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """(raw index of every processed row, raw index of every regular-hours far order of type 1-4 or cross)."""
    t, typ, price = raw["t"], raw["typ"], raw["price"]
    idx = np.flatnonzero((t >= RTH[0]) & (t < RTH[1]) & (typ >= 1) & (typ <= 4))
    p = price[idx]
    R = p.mean()
    keep = ~((p < R / 100) | (p > R * 100))
    cross = np.flatnonzero((t >= RTH[0]) & (t < RTH[1]) & (typ == 6) & (price > 0))   # price 0: a cross elsewhere
    return idx[keep][1:], np.sort(np.concatenate([idx[~keep], cross]))


def check_alignment(raw: dict, proc: np.ndarray, msg: np.ndarray) -> int:
    """Rows where the raw rows taken as processed differ from the processed file (0 expected)."""
    if len(proc) != len(msg):
        return max(len(proc), len(msg))
    ours = np.column_stack([raw["oid"][proc].astype(np.int64).astype(np.int32), raw["typ"][proc],
                            (raw["dirn"][proc] + 1) // 2, raw["price"][proc], np.minimum(raw["size"][proc], CLIP),
                            raw["sec"][proc], raw["ns"][proc]])
    theirs = np.asarray(msg[:, [0, 1, 2, 3, 5, 8, 9]], np.int64)
    return int((ours != theirs).any(1).sum())


def late_start(raw: dict, proc: np.ndarray, msg: np.ndarray, look: int = 1000) -> int:
    """Leading rows of `proc` before the processed file's first row (0 on almost every day).

    A day whose raw messages begin after the open has more rows dropped at the
    start (AMCR 2026-01-15: first raw event at 34,284 s, 10 more rows, job
    8363129). That file came from the original pipeline, not regen.py, whose
    rule there is unknown, so the start is taken from the file: the first of
    our rows with the time, order id and event of its row 0."""
    if len(proc) <= len(msg) or not len(msg):
        return 0
    m0 = (int(msg[0, 8]), int(msg[0, 9]), int(msg[0, 0]), int(msg[0, 1]))
    for s in range(min(look, len(proc) - len(msg) + 1)):
        i = proc[s]
        if (int(raw["sec"][i]), int(raw["ns"][i]), int(np.int32(np.int64(raw["oid"][i]))), int(raw["typ"][i])) == m0:
            return s
    return 0


def snapshot(bk: Book) -> tuple[list, list, list, np.ndarray, np.ndarray]:
    """(level sides, level prices, orders per level, order ids, order sizes) of the whole book."""
    sides, prices, counts, ids, sizes = [], [], [], [], []
    for side in (ASK, BID):
        for p in bk.prices(side) if side == BID else sorted(bk.prices(side)):
            q = bk.queue(side, p)
            sides.append(side)
            prices.append(p)
            counts.append(len(q))
            ids += [o.oid for o in q]
            sizes += [o.size for o in q]
    return sides, prices, counts, np.asarray(ids, np.int64), np.asarray(sizes, np.int32)


def build(raw: dict, msg: np.ndarray, l10: np.ndarray, grid: int = GRID, cross_cleanup: bool = False) -> dict:
    proc, far = processed_index(raw)
    lead = late_start(raw, proc, msg)
    proc = proc[lead:]
    n = len(proc)
    bad_align = check_alignment(raw, proc, msg)
    if len(proc) != len(msg):
        raise ValueError(f"the processed rule gives {len(proc)} rows, the processed file has {len(msg)}")
    role = np.zeros(len(raw["t"]), np.int8)                 # 0 book only, 1 processed, 2 extra
    role[proc] = 1
    far = far[far > proc[0]] if n else far[:0]
    role[far] = 2
    row_of = np.full(len(raw["t"]), -1, np.int64)
    row_of[proc] = np.arange(n)
    before_row = np.searchsorted(proc, far)                  # processed rows before each extra

    oid32 = raw["oid"].astype(np.int64).astype(np.int32).astype(np.int64)
    d01 = (raw["dirn"] + 1) // 2
    queue = np.full((n, 2), -1, np.int32)
    ex_slot = np.full(len(far), -1, np.int64)
    snap_row, lvl_side, lvl_price, lvl_n, ords = [], [], [], [], []
    top_bad = unknown = 0
    bk = Book()
    k_ex = 0
    cols = zip(oid32.tolist(), raw["typ"].tolist(), d01.tolist(), raw["price"].tolist(), raw["size"].tolist(),
               row_of.tolist(), role.tolist())
    for oid, typ, d, price, size, r, rl in cols:
        if typ == 6 and not (cross_cleanup and price > 0):
            k_ex += int(rl == 2)                              # an extra, slot -1, book unchanged
            continue
        row = (oid, typ, d, price, 0, size)
        if r >= 0:
            if r % grid == 0:
                s, p, c, o, z = snapshot(bk)
                snap_row.append(r)
                lvl_side.append(s); lvl_price.append(p); lvl_n.append(c); ords.append((o, z))
            if not np.array_equal(bk.l10(), np.asarray(l10[r, 3:], np.int64)):
                top_bad += 1
            if typ in (2, 3, 4):
                queue[r, 1] = len(bk.queue(BID if d == 1 else ASK, price))
        slot = bk.apply_row(row)
        if r >= 0:
            queue[r, 0] = slot
            if slot == -2:                                    # the files write -2 for both columns
                queue[r, 1] = -2
                unknown += 1
        elif rl == 2:
            ex_slot[k_ex] = slot
            k_ex += 1

    lp = np.cumsum([0] + [len(s) for s in lvl_side])
    counts = np.concatenate([np.asarray(c, np.int64) for c in lvl_n]) if lvl_n else np.zeros(0, np.int64)
    n_ord = sum(len(o) for o, _ in ords)
    big = proc[raw["size"][proc] > CLIP]
    return dict(
        queue=queue,
        book_full=dict(snap_row=np.asarray(snap_row, np.int64), lvl_ptr=lp.astype(np.int64),
                       lvl_side=np.concatenate([np.asarray(s, np.int8) for s in lvl_side]) if lvl_side else np.zeros(0, np.int8),
                       lvl_price=np.concatenate([np.asarray(p, np.int64) for p in lvl_price]) if lvl_price else np.zeros(0, np.int64),
                       ord_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                       ord_size=np.concatenate([z for _, z in ords]) if ords else np.zeros(0, np.int32),
                       ord_id=np.concatenate([o for o, _ in ords]) if ords else np.zeros(0, np.int64),
                       grid=np.int64(grid), n_msgs=np.int64(n)),
        extra=dict(before_row=before_row.astype(np.int64), order_id=oid32[far], event_type=raw["typ"][far].astype(np.int64),
                   direction=d01[far].astype(np.int64), price_abs=raw["price"][far].astype(np.int64),
                   size=raw["size"][far].astype(np.int64), queue_slot=ex_slot),
        bigsize=dict(row=row_of[big], size=raw["size"][big].astype(np.int64)),
        replay=dict(proc_msgs=n, proc_book_rows=int(len(l10)), raw_member=raw["member"], raw_msgs=int(len(raw["t"])),
                    raw_types=np.bincount(raw["typ"], minlength=8)[1:8].tolist(), align_bad_rows=bad_align,
                    late_start_rows=lead, unknown_refs=unknown, top10_match_rate=1 - top_bad / max(n, 1), n_extra=int(len(far)),
                    n_snapshots=len(snap_row), n_snapshot_levels=int(lp[-1]), n_snapshot_orders=int(n_ord)),
    )


def write(out: dict, base: str) -> None:
    for name, kind in (("queue", "npy"), ("book_full", "npz"), ("extra", "npz"), ("bigsize", "npz"), ("replay", "json")):
        f = f"{base}_{name}.{kind}"
        tmp = f + ".tmp"
        with open(tmp, "wb") as fh:
            if kind == "npy":
                np.save(fh, out[name])
            elif kind == "npz":
                np.savez_compressed(fh, **out[name])
            else:
                fh.write(json.dumps(out[name], indent=1).encode())
        os.chmod(tmp, 0o664)
        os.replace(tmp, f)


def compare(out: dict, base: str) -> dict:
    """Arrays that differ from existing files at `base` (an empty dict: identical)."""
    diff = {}
    q = np.load(base + "_queue.npy")
    for c, name in ((0, "queue_slot"), (1, "queue_len")):
        if q.shape != out["queue"].shape or (q[:, c] != out["queue"][:, c]).any():
            bad = np.flatnonzero(q[:, c] != out["queue"][:, c]) if q.shape == out["queue"].shape else []
            diff[name] = dict(rows=int(len(bad)) if len(q) == len(out["queue"]) else "shape",
                              first=[(int(r), q[r].tolist(), out["queue"][r].tolist()) for r in bad[:3]])
    for name in ("book_full", "extra", "bigsize"):
        if not os.path.exists(f"{base}_{name}.npz"):
            diff[name] = "missing"
            continue
        z = np.load(f"{base}_{name}.npz")
        for k in z.files:
            a, b = np.asarray(z[k]), np.asarray(out[name][k])
            if a.shape != b.shape or not np.array_equal(a, b):
                diff[f"{name}.{k}"] = dict(theirs=list(a.shape), ours=list(b.shape),
                                           first=int(np.flatnonzero(a.ravel()[:min(a.size, b.size)] != b.ravel()[:min(a.size, b.size)])[0])
                                           if a.size and b.size and (a.ravel()[:min(a.size, b.size)] != b.ravel()[:min(a.size, b.size)]).any() else None)
    return diff


def _item(arg):
    entry, a = arg
    t0 = time.time()
    root = os.path.join(a.shard_root, entry["month"])
    msg = np.load(os.path.join(root, entry["message"]), mmap_mode="r")
    l10 = np.load(os.path.join(root, entry["orderbook"]), mmap_mode="r")
    with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
        raw = raw_day(a.lobster, entry["ticker"], entry["date"], tmp)
    try:
        out = build(raw, msg, l10, cross_cleanup=a.cross_cleanup)
    except Exception as err:                                  # noqa: BLE001 (reported per day; the run goes on)
        return dict(ticker=entry["ticker"], date=entry["date"], error=f"{type(err).__name__}: {err}"[:500],
                    seconds=round(time.time() - t0, 1))
    res = dict(ticker=entry["ticker"], date=entry["date"], **{k: v for k, v in out["replay"].items() if k != "raw_member"})
    if a.out_dir:
        d = os.path.join(a.out_dir, entry["month"], entry["ticker"])
        os.makedirs(d, exist_ok=True)
        write(out, os.path.join(d, stem(entry)))
    if a.compare_dir == "manifest":
        res["diff"] = compare(out, os.path.join(entry["fullbook_dir"], stem(entry)))
    elif a.compare_dir:
        res["diff"] = compare(out, os.path.join(a.compare_dir, entry["month"], entry["ticker"], stem(entry)))
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_root", required=True, help="<root>/<month>/ holds the processed files")
    p.add_argument("--lobster", required=True, help="quant/lobster: <YYYY>/<YYYYMMDD>.squashfs")
    p.add_argument("--manifest", required=True)
    p.add_argument("--month", required=True)
    p.add_argument("--out_dir", default="", help="write the files under <out_dir>/<month>/<ticker>/")
    p.add_argument("--compare_dir", default="", help="compare with <compare_dir>/<month>/<ticker>/<stem>_*, or 'manifest': each "
                   "entry's fullbook_dir (8 META days of Jan 2026 point to the _v3changed tree)")
    p.add_argument("--tickers", default="", help="comma-separated; default all")
    p.add_argument("--limit", type=int, default=0, help="first N days after the seed shuffle (0: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cross_cleanup", action="store_true", help="drop asks below and bids above a cross (event 6) with a price; "
                   "the rule of the existing queue and book_full files")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    p.add_argument("--report", required=True, help="one JSON line per day")
    a = p.parse_args()

    entries = [e for e in json.load(open(a.manifest)) if e["month"] == a.month]
    if a.tickers:
        keep = set(a.tickers.split(","))
        entries = [e for e in entries if e["ticker"] in keep]
    order = np.random.default_rng(a.seed).permutation(len(entries))
    entries = [entries[int(i)] for i in order][:a.limit or None]
    done = set()
    if os.path.exists(a.report):
        done = {(r["ticker"], r["date"]) for r in map(json.loads, open(a.report))}
    todo = [(e, a) for e in sorted(entries, key=lambda e: -e["n_msgs"]) if (e["ticker"], e["date"]) not in done]
    print(f"{len(entries)} days, {len(todo)} to do", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, open(a.report, "a") as rep:
        for k, res in enumerate(pool.imap_unordered(_item, todo)):
            rep.write(json.dumps(res) + "\n")
            rep.flush()
            if "error" in res:
                print(f"{k + 1}/{len(todo)} {res['ticker']} {res['date']} ERROR {res['error']}", flush=True)
                continue
            ok = not res.get("diff") and res["align_bad_rows"] == 0 and res["top10_match_rate"] == 1.0
            print(f"{k + 1}/{len(todo)} {res['ticker']} {res['date']} {res['seconds']} s "
                  f"{'ok' if ok else 'DIFF ' + json.dumps({x: res[x] for x in ('align_bad_rows', 'top10_match_rate')}) + ' ' + json.dumps(res.get('diff'))[:600]}",
                  flush=True)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
