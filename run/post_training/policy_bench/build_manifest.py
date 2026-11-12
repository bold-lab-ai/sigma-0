#!/usr/bin/env python3
"""Manifest of one month's stock-days for the bench, by the rules of dataset v3 (CPU only).

The Jan 2026 manifest (quant/dataset_q4_2025_jan2026_v3/test.json) was made by
build_v3.py there. This makes the same entries for a month it does not cover:

    ticker, date, month, message, orderbook, n_msgs, fullbook_dir, [bad_rows]

- Stock-days: the tickers of --tickers (v3's tickers.json) with a message
  file <T>/<T>_<date>_<start>_<end>_message_10_proc.npy in the shard's
  index.json. Most files are named for the raw window 24900000_57900000; the
  megacaps have 34200000_57600000. Where a day has both (10 META days of
  February 2026; the other copy holds 11 damaged AMZN and META files), the
  34200000_57600000 file is used, as in the January manifest.
- check_pair, as in build_v3: book rows = message rows + 1, and the book's
  time columns 1:3 from row 1 equal the message time columns 8:10. A day that
  fails is left out and listed under problems.
- bad_rows: runs [start, end) of book rows whose best ask or bid is a
  placeholder (>= 9e9 or <= 0), as build_v3.placeholder_runs.
- fullbook_dir: <--fullbook_root>/<month>/<ticker> (build_fullbook.py --out_dir).
- With --lobster, a stock-day is kept only if the raw L0 image of its date
  (<lobster>/<YYYY>/<YYYYMMDD>.squashfs) holds <ticker>.7z, as v2's
  requires_raw_L0 (the full-book files are built from it). In Jan 2026 this
  drops the same 70 stock-days that test.json lacks. The raw tickers per date
  are reported, so that a date with partial raw coverage (v2 dropped four
  such dates in Oct and Nov 2025) can be seen; none is dropped here.
- Every entry also carries last_time, the time (s) of its last message, so
  that days whose data stop early can be seen. None is left out for it here.

    python run/post_training/policy_bench/build_manifest.py --shard_root <mount> --month 2026-02 \
        --tickers <v3>/tickers.json --lobster <quant>/lobster --fullbook_root <dir> --out <manifest.json>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess as sp
from multiprocessing import Pool

import numpy as np

NAME = re.compile(r"^([^/]+)/\1_(\d{4}-\d{2}-\d{2})_(\d+)_(\d+)_message_10_proc\.npy$")
PREFERRED = ("34200000", "57600000")
CHUNK = 1 << 22


def placeholder_runs(book: np.ndarray, n: int) -> list[list[int]]:
    bad = []
    for s in range(0, n, CHUNK):
        ask = np.asarray(book[s:min(n, s + CHUNK), 3])
        bid = np.asarray(book[s:min(n, s + CHUNK), 5])
        bad.extend((np.flatnonzero((ask >= 9e9) | (ask <= 0) | (bid >= 9e9) | (bid <= 0)) + s).tolist())
    runs: list[list[int]] = []
    for r in bad:
        if runs and runs[-1][1] == r:
            runs[-1][1] = r + 1
        else:
            runs.append([r, r + 1])
    return runs


def _entry(arg):
    root, month, fb_root, ticker, date, message = arg
    orderbook = message.replace("_message_10_", "_orderbook_10_")
    try:
        m = np.load(os.path.join(root, message), mmap_mode="r")
        b = np.load(os.path.join(root, orderbook), mmap_mode="r")
    except Exception as err:                                  # noqa: BLE001 (a damaged file is reported)
        return None, dict(message=message, problem=[f"load: {type(err).__name__}: {err}"[:300]])
    probs = []
    if b.shape[0] != m.shape[0] + 1:
        probs.append(f"book rows {b.shape[0]} != msg rows + 1")
    elif not np.array_equal(np.asarray(b[1:, 1:3]), np.asarray(m[:, 8:10])):
        probs.append("book/msg times differ")
    if probs:
        return None, dict(message=message, problem=probs)
    e = dict(ticker=ticker, date=date, month=month, message=message, orderbook=orderbook, n_msgs=int(m.shape[0]),
             fullbook_dir=os.path.join(fb_root, month, ticker), last_time=int(m[-1, 8]))
    runs = placeholder_runs(b, e["n_msgs"])
    if runs:
        e["bad_rows"] = runs
    return e, None


def raw_tickers(lobster: str, date: str) -> set | None:
    """Tickers with a <ticker>.7z in the raw L0 image of `date` (None: no image)."""
    img = os.path.join(lobster, date[:4], date.replace("-", "") + ".squashfs")
    if not os.path.exists(img):
        return None
    out = sp.run(["unsquashfs", "-l", img], stdout=sp.PIPE, stderr=sp.DEVNULL, text=True, check=True).stdout
    return {os.path.basename(l)[:-3] for l in out.split() if l.endswith(".7z")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_root", required=True, help="<root>/<month>/ is the mounted shard")
    p.add_argument("--month", required=True)
    p.add_argument("--tickers", required=True, help="JSON list of tickers (v3 tickers.json)")
    p.add_argument("--fullbook_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lobster", default="", help="quant/lobster: keep stock-days with raw L0 only")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()

    root = os.path.join(a.shard_root, a.month)
    shapes = json.load(open(os.path.join(root, "index.json")))["shapes"]
    tickers = set(json.load(open(a.tickers)))
    files: dict = {}
    for k in sorted(shapes):
        g = NAME.match(k)
        if g and g.group(1) in tickers and g.group(2).startswith(a.month):
            files.setdefault((g.group(1), g.group(2)), []).append((g.group(3, 4) != PREFERRED, k))
    raw, no_raw = {}, []
    if a.lobster:
        raw = {d: raw_tickers(a.lobster, d) for d in sorted({d for _, d in files})}
        no_raw = sorted(f"{t} {d}" for t, d in files if raw[d] is None or t not in raw[d])
        files = {k: v for k, v in files.items() if f"{k[0]} {k[1]}" not in no_raw}
    todo = [(root, a.month, a.fullbook_root, t, d, min(v)[1]) for (t, d), v in sorted(files.items())]
    doubles = [min(v)[1] for v in files.values() if len(v) > 1]
    print(f"{len(todo)} stock-days of {len(tickers)} tickers; {len(doubles)} stored twice, "
          f"the {'_'.join(PREFERRED)} file used; {len(no_raw)} without raw L0 left out", flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(_entry, todo, chunksize=4)
    entries = sorted((e for e, _ in res if e), key=lambda e: (e["ticker"], e["date"]))
    problems = [q for _, q in res if q]
    missing = sorted(tickers - {e["ticker"] for e in entries})
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(entries, fh)
    os.chmod(tmp, 0o664)
    os.replace(tmp, a.out)
    days = sorted({e["date"] for e in entries})
    per_day = {d: sum(e["date"] == d for e in entries) for d in days}
    early = [(e["ticker"], e["date"], e["last_time"]) for e in entries if e["last_time"] < 57_000]
    summary = dict(month=a.month, stock_days=len(entries), tickers=len({e["ticker"] for e in entries}),
                   days=len(days), per_day=per_day, messages=sum(e["n_msgs"] for e in entries),
                   with_bad_rows=sum("bad_rows" in e for e in entries), stored_twice=doubles, problems=problems,
                   tickers_missing=missing, no_raw_L0=no_raw,
                   raw_tickers_per_date={d: None if v is None else len(v) for d, v in raw.items()},
                   last_message_before_57000=early)
    with open(a.out.replace(".json", "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    os.chmod(a.out.replace(".json", "_summary.json"), 0o664)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_day", "last_message_before_57000", "tickers_missing")}, indent=1))
    print("days", per_day)
    print(f"{len(early)} stock-days end before 57,000 s:", early[:40])


if __name__ == "__main__":
    main()
