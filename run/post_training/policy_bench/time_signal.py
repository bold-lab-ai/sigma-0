#!/usr/bin/env python3
"""How long one decision's signal takes (one GPU): the wall time a policy of the
bench waits for its prediction, to set latency_ns.

For the first --n episodes of the config's set, the start state is rebuilt as
the bench does it (ReplayMarket.load), and the signal is computed for that one
decision, batch 1, --repeats times after --warmup untimed calls. Two parts are
timed: `inputs` (CPU: the rows to tokens and book vectors) and `predict` (the
GPU forward over the whole context and the head, synchronised).

    python run/post_training/policy_bench/time_signal.py --config configs/policy_bench/v1.yaml \
        --shard_root <mount> --out <file.json> [--n 16] [--repeats 20]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import yaml

from post_training.policy_bench.episodes import load_episodes
from post_training.policy_bench.market import BookFill, ReplayMarket
from post_training.policy_bench.nanosigma.data import MSG_TOKENS
from post_training.policy_bench.signal import Signal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shard_root", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    eps = load_episodes(cfg["episodes"], entries)[:a.n]
    signal = Signal(cfg["signal"], torch.device("cuda"))
    P = signal.prefill
    market = ReplayMarket(a.shard_root, cfg["add_dir"], BookFill(), history=cfg["history"],
                          n_messages=cfg["n_messages"])
    t_in, t_pred = [], []
    for ep in eps:
        market.load(0, ep)
        log = market.log(0)
        for j in range(a.warmup + a.repeats):
            t0 = time.perf_counter()
            tok, book = signal.inputs(log.rows(P), log.l10(P + 1))
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            signal.predict(tok[None], book[None])
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if j >= a.warmup:
                t_in.append(t1 - t0)
                t_pred.append(t3 - t2)

    def q(x):
        x = np.asarray(x) * 1e3
        return dict(median_ms=float(np.median(x)), p95_ms=float(np.percentile(x, 95)),
                    max_ms=float(x.max()), mean_ms=float(x.mean()))
    total = np.asarray(t_in) + np.asarray(t_pred)
    res = dict(config=a.config, episodes=len(eps), repeats=a.repeats, prefill_rows=P,
               tokens=int((P + 1) * MSG_TOKENS), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               inputs=q(t_in), predict=q(t_pred), total=q(total))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
