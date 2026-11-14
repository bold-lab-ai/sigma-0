#!/usr/bin/env python3
"""How the simulator answers a real sweep, against what the real market did next (one GPU).

A sweep is a real execution message that takes two or more price levels. For
--per_bucket sweeps of each size (2, 3-4, 5+ levels) drawn from the config's
month, three continuations of the same moment are compared:

    real   the real messages after the sweep
    sweep  SimMarket from the real book before the sweep; the same sweep is fed
           as an agent market order of the same side and size (latency 0), then
           the simulator generates. The order comes 1 ns after the previous
           message, or, with --real_dt, at the real sweep's own time (the gap
           before it as in the market)
    none   the same start and seed without the sweep (the simulator's own course)

sweep and none share their random numbers, so their difference is the sweep's
effect in the simulator. After m = 0 .. --n_messages messages (m = 0 is right
after the sweep), measured on the book:

    mid_bps        mid move since the book before the sweep, signed so that the
                   sweep's direction is positive (a buy sweep pushes it up)
    half_spread    half spread, bps of the mid
    side_depth     swept side's top-10 shares over their value before the sweep
    side_back      share of sweeps whose swept side is back at its best price
                   before the sweep or better

Before generating, the book right after the fed sweep is compared with the
real one (`book_equal`): they differ only if the real sweep took shares the
book did not show.

`by_gap` splits mid_bps over all sweeps by the real gap between the sweep and
the message before it (GAPS), so that the market's answer to a sweep after a
pause can be compared with the simulator's.

    python run/benchmarking/policy_bench/check_sweeps.py --config configs/bench/policy_bench_v1.yaml \
        --shard_root <mount> --out <file.json> [--per_bucket 150] [--n_messages 256]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import yaml

from lobpipeline.policy_bench.codec import MAX_SIZE, burst_bounds
from lobpipeline.policy_bench.daydata import DayFiles
from lobpipeline.policy_bench.episodes import T_HI, T_LO, _clear
from lobpipeline.policy_bench.policies import NoTrade
from lobpipeline.policy_bench.rollout import rollout
from lobpipeline.policy_bench.sim_market import SimMarket
from lobpipeline.policy_bench.simulator import load_newtok
from lobpipeline.policy_bench.types import BUY, SELL, Episode, MarketOrder

BUCKETS = {"2": (2, 2), "3-4": (3, 4), "5+": (5, 10**9)}
HORIZONS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
BIG = 9_000_000_000
EVT, DIR, PRICE, SIZE, TS, TNS = 1, 2, 3, 5, 8, 9
NS = 10**9
GAPS = {"<1us": (0, 10**3), "1us-1ms": (10**3, 10**6), "1-10ms": (10**6, 10**7), ">=10ms": (10**7, 10**18)}


def book(l10_row: np.ndarray) -> tuple:
    """(mid, half spread bps, best ask, best bid, ask top-10 shares, bid top-10 shares) of a 43-column row."""
    b = np.asarray(l10_row[3:], np.int64).reshape(10, 4)
    a, d = int(b[0, 0]), int(b[0, 2])
    ok = 0 < d < a < BIG
    mid = (a + d) / 2 if ok else np.nan
    hs = (a - d) / (a + d) * 1e4 if ok else np.nan
    aq = int(b[(b[:, 1] > 0) & (np.abs(b[:, 0]) < BIG), 1].sum())
    bq = int(b[(b[:, 3] > 0) & (np.abs(b[:, 2]) < BIG), 3].sum())
    return mid, hs, a, d, aq, bq


def find_sweeps(day: DayFiles, e: dict, history: int, n_rows: int) -> list[dict]:
    """Real execution messages over two or more levels, at starts the bench could use."""
    msg = np.asarray(day.messages)
    st, en = np.array(burst_bounds(msg), np.int64).T
    keep = (msg[st, EVT] == 4) & (en - st >= 2)
    out = []
    for i, j in zip(st[keep].tolist(), en[keep].tolist()):
        prices = msg[i:j, PRICE]
        levels = 1 + int((prices[1:] != prices[:-1]).sum())
        if levels < 2 or i < history or i + n_rows >= len(msg) or not T_LO <= int(msg[i, TS]) < T_HI:
            continue
        if not _clear(e, i - history, i + n_rows):
            continue
        b0 = day.l10[i]
        if not 0 < int(b0[5]) < int(b0[3]) < BIG:
            continue
        rows = day.rows(i, j)
        per_level = {}
        for p, q in zip(rows[:, PRICE], rows[:, SIZE]):
            per_level[int(p)] = per_level.get(int(p), 0) + int(q)
        if max(per_level.values()) > MAX_SIZE:
            continue
        t = int(msg[i, TS]) * NS + int(msg[i, TNS])
        gap = t - (int(msg[i - 1, TS]) * NS + int(msg[i - 1, TNS]))
        out.append(dict(start=i, end=j, levels=levels, size=int(rows[:, SIZE].sum()), t=t, gap=gap,
                        side=BUY if int(msg[i, DIR]) == 0 else SELL))   # direction 0: asks executed
    return out


class SweepOnce:
    """Sends the episode's real sweep as one market order at k = 0, then nothing."""

    def __init__(self, sweeps: dict) -> None:
        self.sweeps = sweeps

    def reset(self, episode) -> None:
        self.s = self.sweeps[(episode.entry["message"], episode.start)]

    def step(self, obs) -> list:
        return [MarketOrder(self.s["side"], self.s["size"])] if obs.k == 0 else []


class _Recording(SimMarket):
    """SimMarket that keeps the book before the start, after the agent's sweep and after every
    background message."""

    variant = ""
    times: dict | None = None             # (message, start) -> the real sweep's time, under --real_dt

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.books: dict = {}

    def _agent_message(self, r, o, t=None):
        if self.times is not None and t is None:
            t = self.times[(r.ep.entry["message"], r.ep.start)]
        return super()._agent_message(r, o, t)

    def _key(self, r):
        return (r.ep.entry["message"], r.ep.start, self.variant)

    def load(self, i, ep):
        super().load(i, ep)
        r = self.rows[i]
        self.books[self._key(r)] = dict(pre=r.log.now.copy(), after=None, path=[])

    def _activate(self, r):
        n = r.gen["agent_messages"]
        super()._activate(r)
        if r.gen["agent_messages"] > n:
            self.books[self._key(r)]["after"] = r.log.now.copy()

    def _apply_background(self, r, m):
        super()._apply_background(r, m)
        self.books[self._key(r)]["path"].append(r.log.now.copy())


def measures(pre: np.ndarray, path: list, side: int, n: int) -> np.ndarray:
    """(n + 1, 4): mid_bps, half_spread, side_depth, side_back after each book of `path`
    (NaN past its end)."""
    m0, _, a0, d0, aq0, bq0 = book(pre)
    out = []
    for row in path[:n + 1]:
        mid, hs, a, d, aq, bq = book(row)
        if side == BUY:                       # took the asks
            depth, back = aq / max(aq0, 1), float(a <= a0)
        else:
            depth, back = bq / max(bq0, 1), float(d >= d0)
        out.append((side * (mid - m0) / m0 * 1e4, hs, depth, back))
    out += [(np.nan,) * 4] * (n + 1 - len(out))
    return np.array(out, np.float64)


def real_path(day: DayFiles, s: dict, n: int) -> list:
    raw = np.asarray(day.messages[s["end"]:s["end"] + 8 * n])
    bounds = burst_bounds(raw)[:n]
    return [day.l10[s["end"]]] + [day.l10[s["end"] + j] for _, j in bounds]


def summary(x: list[np.ndarray], n: int) -> dict:
    """Mean and standard error per horizon of each measure, over sweeps."""
    hz = [h for h in HORIZONS if h <= n]
    a = np.stack([v[hz] for v in x])                        # (sweeps, horizons, 4)
    out = {}
    for k, name in enumerate(("mid_bps", "half_spread", "side_depth", "side_back")):
        v = a[:, :, k]
        cnt = np.isfinite(v).sum(0)
        out[name] = dict(horizons=hz, mean=np.nanmean(v, 0).round(4).tolist(),
                         se=(np.nanstd(v, 0) / np.sqrt(np.maximum(cnt, 1))).round(4).tolist())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shard_root", required=True)
    p.add_argument("--per_bucket", type=int, default=150)
    p.add_argument("--per_day", type=int, default=3, help="most sweeps of one bucket from one stock-day")
    p.add_argument("--n_messages", type=int, default=256)
    p.add_argument("--max_rejections", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--real_dt", action="store_true", help="feed each sweep at its real time, not 1 ns after "
                   "the previous message")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    rng = np.random.default_rng(a.seed)
    n_rows = 8 * a.n_messages
    chosen = {b: [] for b in BUCKETS}
    t0 = time.time()
    for e in rng.permutation(len(entries)):
        e = entries[int(e)]
        if e["n_msgs"] <= cfg["history"] + n_rows:
            continue
        if all(len(v) >= a.per_bucket for v in chosen.values()):
            break
        day = DayFiles.open(a.shard_root, e)
        found = find_sweeps(day, e, cfg["history"], n_rows)
        for b, (lo, hi) in BUCKETS.items():
            pool = [s for s in found if lo <= s["levels"] <= hi]
            take = min(a.per_day, a.per_bucket - len(chosen[b]), len(pool))
            for k in rng.choice(len(pool), take, replace=False) if take > 0 else []:
                chosen[b].append(dict(pool[int(k)], entry=e, bucket=b))
    print({b: len(v) for b, v in chosen.items()}, f"{time.time() - t0:.0f} s to find", flush=True)

    sweeps = [s for v in chosen.values() for s in v]
    eps = [Episode(s["entry"], s["start"], i) for i, s in enumerate(sweeps)]
    by_key = {(s["entry"]["message"], s["start"]): s for s in sweeps}
    dev = torch.device("cuda")
    market = _Recording(load_newtok(cfg["simulator"], dev), a.shard_root, cfg["add_dir"], cfg["batch"],
                        history=cfg["history"], n_messages=a.n_messages, latency=0, inventory_cap=10**9,
                        max_rejections=a.max_rejections)
    market.variant = "sweep"
    if a.real_dt:
        market.times = {(s["entry"]["message"], s["start"]): s["t"] for s in sweeps}
    tr_sweep = rollout(SweepOnce(by_key), None, market, eps, every=a.n_messages)
    market.variant, market.times = "none", None
    rollout(NoTrade(), None, market, eps, every=a.n_messages)
    seconds = time.time() - t0

    res = dict(config=a.config, n_messages=a.n_messages, per_bucket=a.per_bucket, seed=a.seed,
               max_rejections=a.max_rejections, real_dt=a.real_dt, seconds=seconds, torch=torch.__version__,
               buckets={})
    raw = {}
    gaps = {"real": [], "sweep": [], "none": [], "gap": []}
    for b in BUCKETS:
        acc = {"real": [], "sweep": [], "none": []}
        equal = refused = 0
        sizes, levels = [], []
        for s, tr in zip(sweeps, tr_sweep):
            if s["bucket"] != b:
                continue
            if tr.refused:
                refused += 1
                continue
            key = (s["entry"]["message"], s["start"])
            A, B = market.books[key + ("sweep",)], market.books[key + ("none",)]
            day = DayFiles.open(a.shard_root, s["entry"])
            real = real_path(day, s, a.n_messages)
            equal += int(np.array_equal(np.asarray(A["after"][3:], np.int64), np.asarray(real[0][3:], np.int64)))
            n = a.n_messages
            acc["real"].append(measures(A["pre"], real, s["side"], n))
            acc["sweep"].append(measures(A["pre"], [A["after"]] + A["path"], s["side"], n))
            acc["none"].append(measures(B["pre"], [B["pre"]] + B["path"], s["side"], n))
            for k in ("real", "sweep", "none"):
                gaps[k].append(acc[k][-1])
            gaps["gap"].append(s["gap"])
            sizes.append(s["size"])
            levels.append(s["levels"])
        n_ok = len(acc["real"])
        res["buckets"][b] = dict(sweeps=n_ok, refused=refused, book_equal=equal / max(n_ok, 1),
                                 size_median=float(np.median(sizes)) if sizes else None,
                                 levels_mean=float(np.mean(levels)) if levels else None,
                                 **{k: summary(v, a.n_messages) for k, v in acc.items() if v})
        for k, v in acc.items():
            if v:
                raw[f"{b}/{k}"] = np.stack(v)
    g = np.array(gaps["gap"], np.int64)
    raw["gap"] = g
    res["by_gap"] = {}
    for name, (lo, hi) in GAPS.items():
        sel = (g >= lo) & (g < hi)
        if sel.any():
            res["by_gap"][name] = dict(sweeps=int(sel.sum()), **{
                k: summary([v for v, x in zip(gaps[k], sel) if x], a.n_messages)["mid_bps"]
                for k in ("real", "sweep", "none")})
    json.dump(res, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out.replace(".json", "_paths.npz"), **raw)
    for b, v in res["buckets"].items():
        print(f"== {b} levels: {v['sweeps']} sweeps, refused {v['refused']}, book equal {v['book_equal']:.3f}, "
              f"size median {v['size_median']}")
        for name in ("mid_bps", "half_spread", "side_depth", "side_back"):
            for k in ("real", "sweep", "none"):
                if k in v:
                    print(f"  {name:11s} {k:5s}", " ".join(f"{x:8.3f}" for x in v[k][name]["mean"]))
    for name, v in res["by_gap"].items():
        print(f"== gap {name}: {v['sweeps']} sweeps, mid_bps at {v['real']['horizons']}")
        for k in ("real", "sweep", "none"):
            print(f"  {k:5s}", " ".join(f"{x:8.3f}" for x in v[k]["mean"]))
    print(f"{seconds:.0f} s")


if __name__ == "__main__":
    main()
