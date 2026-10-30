#!/usr/bin/env python3
"""Write the add index of every stock-day of one month (CPU only).

For each manifest entry of --month, reads the processed message file once and
writes <out_dir>/<month>/<stem>_adds.npy (daydata.build_add_index). Existing
outputs are kept, so a timed-out job can be resubmitted. --roots takes several
mounts of the same shard: squashfuse decompresses on one thread per mount.

    python run/post_training/policy_bench/build_add_index.py --roots <mnt1>,<mnt2> \
        --manifest <v3>/test.json --month 2026-01 --out_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from post_training.policy_bench.daydata import build_add_index


def stem(entry: dict) -> str:
    return os.path.splitext(os.path.basename(entry["message"]))[0]


def _item(arg):
    root, entry, out = arg
    msg = np.load(os.path.join(root, entry["month"], entry["message"]), mmap_mode="r")
    ix = build_add_index(msg)
    tmp = out + ".tmp.npy"
    np.save(tmp, ix)
    os.chmod(tmp, 0o664)
    os.replace(tmp, out)
    return int(msg.shape[0]), int(ix.shape[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", required=True, help="comma-separated mounts, each <root>/<month>/")
    p.add_argument("--manifest", required=True)
    p.add_argument("--month", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()

    roots = a.roots.split(",")
    out = os.path.join(a.out_dir, a.month)
    os.makedirs(out, exist_ok=True)
    todo = []
    entries = sorted((e for e in json.load(open(a.manifest)) if e["month"] == a.month), key=lambda e: -e["n_msgs"])
    for e in entries:
        f = os.path.join(out, stem(e) + "_adds.npy")
        if not os.path.exists(f):
            todo.append((roots[len(todo) % len(roots)], e, f))
    print(f"{len(entries)} days, {len(todo)} to do", flush=True)
    t0, rows, orders = time.time(), 0, 0
    with Pool(a.workers) as pool:
        for k, (n, m) in enumerate(pool.imap_unordered(_item, todo)):
            rows, orders = rows + n, orders + m
            if k % 500 == 0 or k == len(todo) - 1:
                print(f"{k + 1}/{len(todo)} days, {rows / 1e9:.2f}G rows, {orders / 1e9:.2f}G orders, "
                      f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
