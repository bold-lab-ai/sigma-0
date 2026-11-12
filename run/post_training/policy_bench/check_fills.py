#!/usr/bin/env python3
"""A one-share order at the back of the best queue: simulator against the real market (one GPU).

What a passive (maker) policy earns depends on the queue at the best price:
how fast the orders ahead go, how often the price moves away, and where the
mid goes after a fill. For --n_episodes starts drawn like the bench's
(episodes.sample_episodes, seed --seed), alternately on the bid and the ask:

    sim   SimMarket from the real book; at k = 0 the agent rests one share at
          the best price (latency 0, at the back of the queue), then the
          simulator generates
    real  the same order put into the real messages as a counterfactual. One
          share changes nobody's behaviour, so price-time priority alone says
          when it fills: a bid at p fills at the first execution of the bid
          side below p, or at p of an order that came after it (the orders
          ahead are the ones queued at p before the start). Asks the other way.

Reported per side: the share filled within --window messages and at 16, 64,
256; the median fill time; the markout, the mid h messages after the fill
against the fill price in bps, signed so that a gain to the agent is
positive; and the shares queued ahead at the start (sim and real must agree).

    python run/post_training/policy_bench/check_fills.py --config configs/policy_bench/v1.yaml \
        --shard_root <mount> --out <file.json> [--n_episodes 500] [--window 512]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import yaml

from post_training.policy_bench.book import ASK, BID
from post_training.policy_bench.codec import burst_bounds
from post_training.policy_bench.daydata import DayFiles
from post_training.policy_bench.episodes import sample_episodes
from post_training.policy_bench.rollout import rollout
from post_training.policy_bench.sim_market import SimMarket
from post_training.policy_bench.simulator import load_newtok
from post_training.policy_bench.types import BUY, SELL, Episode, LimitOrder, mid_of

OID, EVT, DIR, PRICE = 0, 1, 2, 3
WITHIN = [16, 64, 256]
MARKOUTS = [1, 16, 64, 128]


class QuoteOnce:
    """Rests one share at the best price of the episode's side at k = 0, then nothing."""

    def __init__(self, sides: dict) -> None:
        self.sides = sides

    def reset(self, episode) -> None:
        self.side = self.sides[(episode.entry["message"], episode.start)]

    def step(self, obs) -> list:
        if obs.k:
            return []
        return [LimitOrder(self.side, int(obs.l10[2] if self.side == BUY else obs.l10[0]), 1)]


class _Recording(SimMarket):
    """SimMarket that keeps the order ids queued at the quoted best price at the start."""

    sides: dict = {}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ahead: dict = {}

    def load(self, i, ep):
        super().load(i, ep)
        r = self.rows[i]
        side = BID if self.sides[(ep.entry["message"], ep.start)] == BUY else ASK
        p = r.book.best(side)
        self.ahead[(ep.entry["message"], ep.start)] = [(o.oid, o.size) for o in r.book.queue(side, p)]


def real_outcome(day: DayFiles, start: int, side: int, ahead: set, n: int):
    """(fill message index or None, mids after messages 0 .. n - 1) of the counterfactual order."""
    rows = day.rows(start, start + 8 * n)
    bounds = burst_bounds(rows)[:n]
    rows = rows[:bounds[-1][1]]
    k_of_row = np.repeat(np.arange(len(bounds)), [j - i for i, j in bounds])
    b0 = day.l10[start]
    price = int(b0[5]) if side == BUY else int(b0[3])
    ex = (rows[:, EVT] == 4) & (rows[:, DIR] == (1 if side == BUY else 0))   # direction 1: bids executed
    through = rows[:, PRICE] < price if side == BUY else rows[:, PRICE] > price
    behind = (rows[:, PRICE] == price) & ~np.isin(rows[:, OID], list(ahead))
    hit = np.flatnonzero(ex & (through | behind))
    mids = [mid_of(day.l10[start + j][3:]) for _, j in bounds]
    return (int(k_of_row[hit[0]]) if len(hit) else None), np.array([np.nan if m is None else m for m in mids]), price


def outcome_stats(fills: list, window: int) -> dict:
    """fills: [(k or None, price, side, mids after messages 0..), ...]."""
    ks = np.array([np.inf if k is None else k for k, *_ in fills], np.float64)
    out = dict(n=len(fills), filled=float((ks < window).mean()),
               **{f"filled_{w}": float((ks < w).mean()) for w in WITHIN},
               fill_k_median=float(np.median(ks[ks < window])) if (ks < window).any() else None)
    for h in MARKOUTS:
        v = []
        for k, price, side, mids in fills:
            if k is not None and k < window and k + h < len(mids) and np.isfinite(mids[k + h]):
                v.append(side * (mids[k + h] - price) / price * 1e4)
        v = np.array(v)
        out[f"markout_{h}"] = dict(n=len(v), mean=float(v.mean()) if len(v) else None,
                                   se=float(v.std() / np.sqrt(len(v))) if len(v) > 1 else None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shard_root", required=True)
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--max_rejections", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    by_key = {(e["month"], e["message"]): e for e in entries}
    n = a.window + max(MARKOUTS) + 1
    hist = sample_episodes(entries, a.shard_root, a.n_episodes, a.seed, history=cfg["history"], n_messages=8 * n)
    eps = [Episode(by_key[(h["month"], h["message"])], h["start"], h["seed"]) for h in hist]
    sides = {(ep.entry["message"], ep.start): BUY if i % 2 == 0 else SELL for i, ep in enumerate(eps)}

    t0 = time.time()
    dev = torch.device("cuda")
    market = _Recording(load_newtok(cfg["simulator"], dev), a.shard_root, cfg["add_dir"], cfg["batch"],
                        history=cfg["history"], n_messages=n, latency=0, inventory_cap=10**9,
                        max_rejections=a.max_rejections)
    market.sides = sides
    traces = rollout(QuoteOnce(sides), None, market, eps, every=n)
    seconds = time.time() - t0

    res = dict(config=a.config, n_episodes=len(eps), window=a.window, seed=a.seed, max_rejections=a.max_rejections,
               seconds=seconds, torch=torch.__version__, sides={})
    acc = {s: {"sim": [], "real": [], "ahead_sim": [], "ahead_real": [], "same_queue": 0, "refused": 0}
           for s in ("bid", "ask")}
    for ep, tr in zip(eps, traces):
        key = (ep.entry["message"], ep.start)
        side = sides[key]
        v = acc["bid" if side == BUY else "ask"]
        if tr.refused:
            v["refused"] += 1
            continue
        day = DayFiles.open(a.shard_root, ep.entry)
        bk = day.book_before(ep.start)
        bside = BID if side == BUY else ASK
        real_q = [(o.oid, o.size) for o in bk.queue(bside, bk.best(bside))]
        v["same_queue"] += int(real_q == market.ahead[key])
        k_real, mids_real, price = real_outcome(day, ep.start, side, {o for o, _ in real_q}, n)
        maker = tr.fills[tr.fills[:, 4] == 1]
        k_sim = int(maker[0, 0]) if len(maker) else None
        v["sim"].append((k_sim, price, side, tr.mid[1:]))           # mid[m + 1]: after message m
        v["real"].append((k_real, price, side, mids_real))
        v["ahead_sim"].append(sum(q for _, q in market.ahead[key]))
        v["ahead_real"].append(sum(q for _, q in real_q))
    for s, v in acc.items():
        m = len(v["sim"])
        res["sides"][s] = dict(episodes=m, refused=v["refused"], same_queue=v["same_queue"] / max(m, 1),
                               ahead_shares_median=float(np.median(v["ahead_real"])) if m else None,
                               sim=outcome_stats(v["sim"], a.window), real=outcome_stats(v["real"], a.window))
    json.dump(res, open(a.out, "w"), indent=1)
    for s, v in res["sides"].items():
        print(f"== {s}: {v['episodes']} episodes, refused {v['refused']}, same queue {v['same_queue']:.3f}, "
              f"ahead median {v['ahead_shares_median']}")
        for k in ("real", "sim"):
            x = v[k]
            print(f"  {k:4s} filled<{a.window} {x['filled']:.3f} "
                  + " ".join(f"<{w} {x[f'filled_{w}']:.3f}" for w in WITHIN)
                  + f" median k {x['fill_k_median']}  markout "
                  + " ".join(f"h{h} {x[f'markout_{h}']['mean'] if x[f'markout_{h}']['mean'] is None else round(x[f'markout_{h}']['mean'], 3)}"
                             for h in MARKOUTS))
    print(f"{seconds:.0f} s")


if __name__ == "__main__":
    main()
