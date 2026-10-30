#!/usr/bin/env python3
"""G4 of the policy bench: the return signal (one GPU).

A. Reference. The vendored model and returnft_lobert_pass1's head on the
   return bench's January test decisions (every 50,000 rows from the open of
   each test stock-day, off placeholder rows; 70,183 decisions), read from the
   bookfix shards the signal was trained and scored on. The fine-tuned head's
   pooled Pearson at 128 messages must equal the board's 0.2979.
B. Our rows. For --n_ours decisions on the simulator's data (v2_20261013 shards
   with the v3 full-book files), rebuild every processed row of the context
   from our book (price_rel with the day's clip and the reference columns via
   the day's add index, AddRegistry and processed_row, the L10 rows via
   L10Writer) and compare with the files:
   all 14 message columns, all 43 L10 columns, the tokens and book vectors.

    python run/post_training/policy_bench/check_signal.py --signal_root <bookfix mnt> \
        --signal_manifest <v2>/test.json --sim_root <v2_20261013 mnt> \
        --sim_manifest <v3>/test.json --add_dir <build_add_index.py out_dir> \
        --ckpt <returnft_lobert_pass1>/ckpt.pt --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch

from post_training.policy_bench.book import L10Writer
from post_training.policy_bench.daydata import DayFiles
from post_training.policy_bench.nanosigma.data import book_features, encode_messages
from post_training.policy_bench.signal import AddRegistry, Signal, price_rel_clip, processed_row

STRIDE = 50_000       # tools/return_features.py --test_stride
ASK1, BID1 = 3, 5
G = {}


def decisions(entries, P, H):
    span = P + 1 + max(H)
    out = []
    for e in entries:
        bad = list(e.get("bad_rows", [])) + list(e.get("exclude", []))
        for s in range(0, e["n_msgs"] - span + 1, STRIDE):
            if not any(s < t and b < s + span for b, t in bad):
                out.append((e, s))
    return out


def _valid(row):
    return 0 < row[BID1] < row[ASK1] < 9e9


def _ref_item(arg):
    e, s = arg
    root, P, H = G["root"], G["P"], G["H"]
    base = f"{root}/{e['month']}/"
    m = np.load(base + e["message"], mmap_mode="r")
    b = np.load(base + e["orderbook"], mmap_mode="r")
    tok = np.zeros((P + 1, 26), np.int64)
    tok[:P] = encode_messages(np.asarray(m[s:s + P]))
    bt = np.asarray(b[s + P])
    mid = 0.5 * (bt[ASK1] + bt[BID1])
    r = np.array([(0.5 * (f[ASK1] + f[BID1]) - mid) / mid * 1e4 if _valid(bt) and _valid(f) else np.nan
                  for f in (np.asarray(b[s + P + h]) for h in H)])
    spread = (bt[ASK1] - bt[BID1]) / mid * 1e4 if _valid(bt) else np.nan
    return tok.reshape(-1), book_features(np.asarray(b[s:s + P + 1])), r, spread, e["date"]


def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def part_a(sig, entries, root, workers, micro=4, limit=0):
    ds = decisions(entries, sig.prefill, sig.horizons)
    if limit:
        ds = ds[:limit]
    G.update(root=root, P=sig.prefill, H=sig.horizons)
    preds, rs, spreads = [], [], []
    t0 = time.time()
    with Pool(workers) as pool:
        batch = []
        for k, item in enumerate(pool.imap(_ref_item, ds, chunksize=4)):
            batch.append(item)
            if len(batch) == micro or k == len(ds) - 1:
                preds.append(sig.predict(np.stack([x[0] for x in batch]), np.stack([x[1] for x in batch])))
                rs += [x[2] for x in batch]
                spreads += [x[3] for x in batch]
                batch = []
            if k % 5000 == 0:
                print(f"A {k}/{len(ds)} {time.time() - t0:.0f} s", flush=True)
    pred, r, spread = np.concatenate(preds), np.array(rs), np.array(spreads)
    out = dict(decisions=len(ds), seconds=time.time() - t0, horizons={})
    for j, h in enumerate(sig.horizons):
        ok = np.isfinite(r[:, j]) & np.isfinite(spread)
        p, y = pred[ok, j], r[ok, j]
        nz = (p != 0) & (y != 0)
        out["horizons"][str(h)] = dict(n=int(ok.sum()), head_pearson=_pearson(p, y),
                                       head_direction_acc=float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))))
    return out


def _ours_item(arg):
    e, s = arg
    P = G["P"]
    day = DayFiles.open(G["root"], e)
    stem = os.path.splitext(os.path.basename(e["message"]))[0]
    reg = AddRegistry(np.load(f"{G['add_dir']}/{e['month']}/{stem}_adds.npy", mmap_mode="r"), s)
    win = np.asarray(day.messages[s:s + P])
    bk = day.book_before(s)
    l10 = np.zeros((P + 1, 43), np.int64)
    l10[0] = day.l10[s]
    w = L10Writer(l10[0])
    rows = np.zeros((P, 14), np.int64)
    clip = price_rel_clip(e["message"])
    c = Counter()
    for r, _ in day.replay(bk, s, s + P):
        k = r - s
        f = win[k]
        rows[k] = processed_row(*(int(x) for x in f[[0, 1, 2, 3, 5, 6, 7, 8, 9]]), l10[k], reg, clip)
        reg.observe(rows[k])
        l10[k + 1] = w.next_row(bk, int(f[8]), int(f[9]))
    bad = rows != win
    for col in np.nonzero(bad.any(0))[0]:
        c[f"msg_col{col}_bad"] += int(bad[:, col].sum())
    lbad = l10 != np.asarray(day.l10[s:s + P + 1])
    for col in np.nonzero(lbad.any(0))[0]:
        c[f"l10_col{col}_bad"] += int(lbad[:, col].sum())
    t1, b1 = encode_messages(win), book_features(np.asarray(day.l10[s:s + P + 1]))
    t2, b2 = encode_messages(rows), book_features(l10)
    c["rows"] += P
    c["tokens_equal"] += int(np.array_equal(t1, t2))
    c["book_equal"] += int(np.array_equal(b1, b2))
    first_bad = int(np.nonzero(bad.any(1))[0][0]) + s if bad.any() else None
    return dict(message=e["message"], start=s, first_bad_row=first_bad, counts=dict(c))


def part_b(entries, root, add_dir, P, H, n, workers, exclude):
    ds = [d for d in decisions(entries, P, H) if d[0]["message"] not in exclude]
    rng = np.random.default_rng(0)
    ds = [ds[i] for i in sorted(rng.choice(len(ds), min(n, len(ds)), replace=False))]
    G.update(root=root, add_dir=add_dir, P=P)
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_ours_item, ds, chunksize=1)
    tot = Counter()
    for x in res:
        tot.update(x["counts"])
    return dict(decisions=len(ds), seconds=time.time() - t0, total=dict(tot),
                bad=[x for x in res if x["first_bad_row"] is not None][:20])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signal_root", required=True, help="bookfix shards, <root>/<month>/")
    p.add_argument("--signal_manifest", required=True, help="v2 test.json")
    p.add_argument("--sim_root", required=True, help="v2_20261013 shards")
    p.add_argument("--sim_manifest", required=True, help="v3 test.json (fullbook_dir)")
    p.add_argument("--add_dir", required=True, help="build_add_index.py out_dir, for the sim shards")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_ours", type=int, default=2000)
    p.add_argument("--limit_a", type=int, default=0, help="smoke only")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--skip_a", action="store_true")
    a = p.parse_args()

    sig = Signal(a.ckpt, "cuda")
    sm = json.load(open(a.signal_manifest))
    vm = json.load(open(a.sim_manifest))
    n_sig = {(e["month"], e["message"]): e["n_msgs"] for e in sm}
    differ = {e["message"] for e in vm if n_sig.get((e["month"], e["message"])) != e["n_msgs"]}
    res = dict(torch=torch.__version__, horizons=sig.horizons, prefill=sig.prefill,
               days_differ_between_shards=sorted(differ),
               reference=dict(head_pearson_128=0.2979, decisions=70183, source="nano-sigma board, return bench 128"))
    res["b"] = part_b(vm, a.sim_root, a.add_dir, sig.prefill, sig.horizons, a.n_ours, a.workers, differ)
    print("B", json.dumps(res["b"]["total"]), flush=True)
    if not a.skip_a:
        res["a"] = part_a(sig, sm, a.signal_root, a.workers, limit=a.limit_a)
        print("A", json.dumps(res["a"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
