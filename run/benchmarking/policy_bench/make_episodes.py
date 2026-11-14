#!/usr/bin/env python3
"""Draw an episode set of the bench (episodes.sample_episodes) and write it as JSON.

    python run/benchmarking/policy_bench/make_episodes.py --shard_root <v2_20261013 mnt> \
        --manifest <v3>/test.json --n 200 --seed 20261030 --out <dir>/dev_episodes.json
"""

from __future__ import annotations

import argparse
import json
import os

from lobpipeline.policy_bench.episodes import T_HI, T_LO, sample_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--month", default="2026-01")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--history", type=int, default=3968)
    p.add_argument("--n_messages", type=int, default=1024)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    entries = [e for e in json.load(open(a.manifest)) if e["month"] == a.month]
    eps = sample_episodes(entries, a.shard_root, a.n, a.seed, a.history, a.n_messages)
    meta = dict(manifest=a.manifest, month=a.month, n=a.n, seed=a.seed, history=a.history,
                n_messages=a.n_messages, t_lo=T_LO, t_hi=T_HI, stock_days=len(entries),
                distinct_stock_days=len({x["message"] for x in eps}))
    json.dump(dict(meta=meta, episodes=eps), open(a.out, "w"), indent=1)
    os.chmod(a.out, 0o664)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
