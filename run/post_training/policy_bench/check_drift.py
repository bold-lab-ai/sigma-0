#!/usr/bin/env python3
"""Book drift: generated flow against the real continuation of the same episodes (one GPU).

For the first --n_episodes of the config's episode set, NoTrade runs in
SimMarket and, after every accepted background message, the book is
measured: half spread (bps of the mid), shares and non-empty levels in the top
10 of each side, the event, and how many draws were rejected first. The real
rows after the same start are grouped into messages (codec.burst_bounds) and
measured the same way from the L10 file. Both are pooled by message index in
blocks of --block messages.

Why: in the dev evaluation (jobs 7848223, 7848225) the half spread at the
end of an episode had a median of 0.79 bps generated against 2.66 bps real.

    python run/post_training/policy_bench/check_drift.py --config configs/policy_bench/v1.yaml \
        --shard_root <mount> --out <file.json> [--n_episodes 96] [--block 128]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

from post_training.policy_bench.codec import burst_bounds
from post_training.policy_bench.daydata import DayFiles
from post_training.policy_bench.episodes import load_episodes
from post_training.policy_bench.policies import NoTrade
from post_training.policy_bench.rollout import rollout
from post_training.policy_bench.sim_market import SimMarket
from post_training.policy_bench.simulator import load_newtok

PLACEHOLDER = 9_000_000_000


def book_stats(l10: np.ndarray) -> tuple[float, int, int, int, int]:
    """(half spread bps, bid shares, ask shares, bid levels, ask levels) of a 43-column L10 row."""
    b = np.asarray(l10[3:], np.int64).reshape(10, 4)            # ask_p, ask_q, bid_p, bid_q
    ask_ok = (b[:, 1] > 0) & (np.abs(b[:, 0]) < PLACEHOLDER)
    bid_ok = (b[:, 3] > 0) & (np.abs(b[:, 2]) < PLACEHOLDER)
    a, d = int(b[0, 0]), int(b[0, 2])
    hs = (a - d) / (a + d) * 1e4 if ask_ok[0] and bid_ok[0] else np.nan
    return hs, int(b[bid_ok, 3].sum()), int(b[ask_ok, 1].sum()), int(bid_ok.sum()), int(ask_ok.sum())


class _Recording(SimMarket):
    """SimMarket that measures the book after every background message."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.stats: dict[tuple, list] = {}

    def load(self, i, ep):
        super().load(i, ep)
        self.stats[(ep.entry["message"], ep.start)] = []

    def _apply_background(self, r, m):
        super()._apply_background(r, m)
        self.stats[(r.ep.entry["message"], r.ep.start)].append((m.evt, r.attempt, *book_stats(r.log.now)))


def real_stats(shard_root: str, ep, n_rows: int) -> list:
    day = DayFiles.open(shard_root, ep.entry)
    msg = np.asarray(day.messages[ep.start:ep.start + n_rows])
    out = []
    for i, j in burst_bounds(msg):
        out.append((int(msg[i, 1]), 0, *book_stats(day.l10[ep.start + j])))
    return out


def pooled(per_ep: list[list], block: int, n: int) -> list[dict]:
    out = []
    for lo in range(0, n, block):
        x = [s for st in per_ep for s in st[lo:lo + block]]
        if not x:
            break
        a = np.array(x, np.float64)
        evt = a[:, 0].astype(int)
        out.append(dict(
            messages=f"{lo}-{lo + block - 1}", n=len(a),
            half_spread_bps_median=float(np.nanmedian(a[:, 2])), half_spread_bps_mean=float(np.nanmean(a[:, 2])),
            bid_shares_top10=float(a[:, 3].mean()), ask_shares_top10=float(a[:, 4].mean()),
            bid_levels=float(a[:, 5].mean()), ask_levels=float(a[:, 6].mean()),
            event_share={str(e): float((evt == e).mean()) for e in sorted(set(evt.tolist()))},
            first_draw_rejected=float((a[:, 1] > 0).mean()),
        ))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shard_root", required=True)
    p.add_argument("--n_episodes", type=int, default=96)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    eps = load_episodes(cfg["episodes"], entries)[:a.n_episodes]
    n = cfg["n_messages"]
    real = [real_stats(a.shard_root, ep, n) for ep in eps]

    dev = torch.device("cuda")
    market = _Recording(load_newtok(cfg["simulator"], dev), a.shard_root, cfg["add_dir"], cfg["batch"],
                        history=cfg["history"], n_messages=n, latency=cfg["latency"],
                        inventory_cap=cfg["inventory_cap"])
    t0 = time.time()
    rollout(NoTrade(), None, market, eps, every=n)
    sim = [market.stats[(ep.entry["message"], ep.start)] for ep in eps]

    end_real = np.array([st[-1][2] for st in real])
    end_sim = np.array([st[-1][2] for st in sim])
    res = dict(config=a.config, episodes=len(eps), n_messages=n, block=a.block, seconds=time.time() - t0,
               torch=torch.__version__,
               start=pooled([st[:1] for st in real], 1, 1)[0],
               sim=pooled(sim, a.block, n), real=pooled(real, a.block, n),
               end_half_spread_bps=dict(sim_median=float(np.nanmedian(end_sim)), real_median=float(np.nanmedian(end_real)),
                                        sim_tighter_share=float(np.mean(end_sim < end_real))))
    json.dump(res, open(a.out, "w"), indent=1)
    for k in ("sim", "real"):
        for b in res[k]:
            print(k, b["messages"], f"hs med {b['half_spread_bps_median']:.2f}", f"bid {b['bid_shares_top10']:.0f}",
                  f"ask {b['ask_shares_top10']:.0f}", f"lv {b['bid_levels']:.1f}/{b['ask_levels']:.1f}",
                  b["event_share"], f"rej {b['first_draw_rejected']:.3f}", flush=True)
    print(json.dumps(res["end_half_spread_bps"]))


if __name__ == "__main__":
    main()
