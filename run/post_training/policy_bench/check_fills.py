#!/usr/bin/env python3
"""A one-share order at the back of the best queue: simulator against the real market (one GPU).

What a passive (maker) policy earns depends on the queue at the best price:
how fast the orders ahead go, how often the price moves away, and where the
mid goes after a fill. For --n_episodes starts drawn like the bench's
(episodes.sample_episodes, seed --seed), alternately on the bid and the ask,
three paths of the same start are compared:

    real   the real messages. The order is put in as a counterfactual: a bid
           at p fills at the first bid-side execution below p, or at p of an
           order that was not queued ahead of it at the start (asks the other
           way). This assumes one share changes nobody's behaviour, which
           fails while the share would be alone at the best price: the real
           book then has no order at p (`alone_s` measures how long).
    agent  SimMarket from the real book; at k = 0 the agent rests one share at
           the best price (latency 0, at the back of the queue). Its fill is
           the simulator's own match.
    none   the same start and seeds without the order, with the real path's
           counterfactual rule. agent against none is the rule's error inside
           the simulator; real against none compares the market with the
           simulator by the same method, so that error cancels.

--sampler picks the simulator's rule for invalid draws: mask (max_rejections
0), reject (32 whole-message draws, then masks), keep (nano-sigma's rule: an
invalid first draw stays in the token stream, no row is written). Under keep
a step without a row is not a message: every path is measured in messages
that wrote rows (and in seconds), and the simulator runs longer to have
--window of them.

Per path and side: the share filled within --window messages and at 16, 64,
256; fill time in messages and seconds; markouts, the mid h messages after the
fill against the fill price in bps, signed so that a gain to the agent is
positive. How it filled, before the fill (or the window's end): the shares
ahead removed by executions and by cancels, whether the best price moved
away from p, the size of the execution message that filled, and whether the
level at p is still there after it (`level_stays`; markouts are also split by
it). And the mid's move over the window (std, bps), the overall volatility,
and the seconds the window's messages take (the simulator's clock).

    python run/post_training/policy_bench/check_fills.py --config configs/policy_bench/v1.yaml \
        --shard_root <mount> --out <file.json> [--sampler mask] [--n_episodes 500] [--window 512]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import yaml

from post_training.policy_bench.book import ASK, BID
from post_training.policy_bench.codec import NS, burst_bounds
from post_training.policy_bench.daydata import DayFiles
from post_training.policy_bench.episodes import sample_episodes
from post_training.policy_bench.policies import NoTrade
from post_training.policy_bench.rollout import rollout
from post_training.policy_bench.sampling import MAX_REJECTIONS
from post_training.policy_bench.sim_market import AGENT_OID0, SimMarket
from post_training.policy_bench.simulator import load_newtok
from post_training.policy_bench.types import BUY, SELL, Episode, LimitOrder

OID, EVT, DIR, PRICE, SIZE, TS, TNS = 0, 1, 2, 3, 5, 8, 9
WITHIN = [16, 64, 256]
MARKOUTS = [1, 16, 64, 128]
SAMPLERS = {"mask": dict(max_rejections=0), "reject": dict(max_rejections=MAX_REJECTIONS),
            "keep": dict(max_rejections=MAX_REJECTIONS, keep_invalid=True)}
KEEP_STRETCH = 2.0          # keep writes rows on about 75% of steps (lobbench/README.md); 1.6 was short (8363489)
# A path: rows (m, 8) int64 [message, oid, evt, d, price, size, t_ns, best price of the quoted side
# after the row], mids (n,) after each message, t0 (ns of the last history row).


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
    """SimMarket that keeps, per episode and variant, the queue at the quoted best at the start
    and every row written, with the best price of the quoted side after it."""

    sides: dict = {}
    variant = ""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ahead: dict = {}
        self.paths: dict = {}

    def load(self, i, ep):
        super().load(i, ep)
        r = self.rows[i]
        key = (ep.entry["message"], ep.start, self.variant)
        side = BID if self.sides[key[:2]] == BUY else ASK
        self.ahead[key] = [(o.oid, o.size) for o in r.book.queue(side, r.book.best(side))]
        self.paths[key] = dict(t0=self._t_prev(r), rows=[], col=5 if side == BID else 3)

    def _row(self, r, oid, evt, d, price, size, dt, t):
        super()._row(r, oid, evt, d, price, size, dt, t)
        p = self.paths[(r.ep.entry["message"], r.ep.start, self.variant)]
        p["rows"].append((r.k, oid, evt, d, price, size, t, int(r.log.now[p["col"]])))


def mid_or_nan(l10_row) -> float:
    a, b = int(l10_row[3]), int(l10_row[5])
    return (a + b) / 2 if 0 < b < a < 9_000_000_000 else np.nan


def real_path(day: DayFiles, start: int, side: int, n: int) -> dict:
    rows = np.asarray(day.rows(start, start + 8 * n), np.int64)
    bounds = burst_bounds(rows)[:n]
    rows = rows[:bounds[-1][1]]
    k = np.repeat(np.arange(len(bounds)), [j - i for i, j in bounds])
    col = 5 if side == BUY else 3
    best = np.asarray(day.l10[start + 1:start + 1 + len(rows), col], np.int64)     # l10[x]: book before row x
    t = rows[:, TS] * NS + rows[:, TNS]
    mids = np.array([mid_or_nan(day.l10[start + j]) for _, j in bounds])
    t0 = int(day.l10[start, 1]) * NS + int(day.l10[start, 2])
    out = np.column_stack([k, rows[:, OID], rows[:, EVT], rows[:, DIR], rows[:, PRICE], rows[:, SIZE], t, best])
    return dict(rows=out, mids=mids, t0=t0, real=True)


def sim_path(market: _Recording, key: tuple, mids: np.ndarray) -> dict:
    """The recorded path on the axis of steps that wrote a row of a background message
    (the agent's own messages do not count; its fills come from background messages)."""
    p = market.paths[key]
    rows = np.array(p["rows"], np.int64).reshape(-1, 8)
    own = (rows[:, 1] >= AGENT_OID0) & (rows[:, 2] != 4)
    valid = np.unique(rows[~own, 0])
    rank = np.full(int(rows[:, 0].max()) + 2 if len(rows) else 1, -1, np.int64)
    rank[valid] = np.arange(len(valid))
    rows = rows[rank[rows[:, 0]] >= 0]
    rows[:, 0] = rank[rows[:, 0]]
    return dict(rows=rows, mids=mids[1:][valid], t0=p["t0"])   # mids[m + 1]: after step m


def counterfactual(rows: np.ndarray, ahead: set, side: int, price: int) -> int | None:
    ours = 1 if side == BUY else 0
    ex = (rows[:, 2] == 4) & (rows[:, 3] == ours) & (rows[:, 1] < AGENT_OID0)
    through = rows[:, 4] < price if side == BUY else rows[:, 4] > price
    behind = (rows[:, 4] == price) & ~np.isin(rows[:, 1], list(ahead))
    hit = np.flatnonzero(ex & (through | behind))
    return int(rows[hit[0], 0]) if len(hit) else None


def describe(path: dict, k: int | None, ahead: list, side: int, price: int, window: int) -> dict:
    """How the episode went for the quote, with fill message k (None: not filled)."""
    rows, mids = path["rows"], path["mids"]
    ours = 1 if side == BUY else 0
    ids = list({o for o, _ in ahead})
    k = k if k is not None and k < window else None
    end = window if k is None else k
    pre = rows[(rows[:, 0] < end) & (rows[:, 1] < AGENT_OID0)]
    mine = np.isin(pre[:, 1], ids) & (pre[:, 3] == ours) & (pre[:, 4] == price)
    away = (pre[:, 7] > price) if side == BUY else (pre[:, 7] < price)
    last = min(window, len(mids)) - 1
    ok = last >= 0 and np.isfinite(mids[0]) and np.isfinite(mids[last])
    d = dict(k=k, ahead=sum(q for _, q in ahead),
             ahead_exec=int(pre[mine & (pre[:, 2] == 4), 5].sum()),
             ahead_cancel=int(pre[mine & np.isin(pre[:, 2], [2, 3]), 5].sum()),
             moved_away=bool(away.any()),
             window_s=float((rows[rows[:, 0] < window, 6].max() - path["t0"]) / NS) if (rows[:, 0] < window).any()
             else float("nan"),
             move_bps=float(side * (mids[last] - mids[0]) / mids[0] * 1e4) if ok else float("nan"))
    if k is not None:
        at = rows[rows[:, 0] == k]
        exe = at[(at[:, 2] == 4) & (at[:, 3] == ours)]
        d["fill_s"] = float((at[0, 6] - path["t0"]) / NS)
        d["fill_msg_size"] = int(exe[:, 5].sum())
        d["level_stays"] = bool(at[-1, 7] == price)
        d["markouts"] = {h: float(side * (mids[k + h] - price) / price * 1e4)
                         if k + h < len(mids) and np.isfinite(mids[k + h]) else None for h in MARKOUTS}
        if path.get("real"):
            # how long the share would have been alone at the best price before the fill
            gone = (rows[:, 0] < k) & ((rows[:, 7] < price) if side == BUY else (rows[:, 7] > price))
            i = np.flatnonzero(gone)
            d["alone_s"] = float((at[0, 6] - rows[i[0], 6]) / NS) if len(i) else 0.0
            d["through"] = bool(((exe[:, 4] < price) if side == BUY else (exe[:, 4] > price)).any())
    return d


def _mean_se(v) -> dict:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], np.float64)
    return dict(n=len(v), mean=float(v.mean()) if len(v) else None,
                se=float(v.std() / np.sqrt(len(v))) if len(v) > 1 else None)


def stats(ds: list[dict], window: int) -> dict:
    ks = np.array([np.inf if d["k"] is None else d["k"] for d in ds], np.float64)
    f = [d for d in ds if d["k"] is not None]
    out = dict(n=len(ds), filled=float((ks < window).mean()),
               **{f"filled_{w}": float((ks < w).mean()) for w in WITHIN},
               fill_k_median=float(np.median(ks[ks < window])) if f else None,
               fill_s_median=float(np.median([d["fill_s"] for d in f])) if f else None,
               ahead_exec_share=_mean_se([d["ahead_exec"] / max(d["ahead"], 1) for d in ds]),
               ahead_cancel_share=_mean_se([d["ahead_cancel"] / max(d["ahead"], 1) for d in ds]),
               moved_away=float(np.mean([d["moved_away"] for d in ds])),
               move_std_bps=float(np.nanstd([d["move_bps"] for d in ds])),
               move_abs_bps=float(np.nanmean(np.abs([d["move_bps"] for d in ds]))),
               window_s_median=float(np.nanmedian([d["window_s"] for d in ds])),
               fill_msg_size_mean=float(np.mean([d["fill_msg_size"] for d in f])) if f else None,
               fill_msg_size_one=float(np.mean([d["fill_msg_size"] == 1 for d in f])) if f else None,
               fill_msg_size_median=float(np.median([d["fill_msg_size"] for d in f])) if f else None,
               level_stays=float(np.mean([d["level_stays"] for d in f])) if f else None)
    for h in MARKOUTS:
        out[f"markout_{h}"] = _mean_se([d["markouts"][h] for d in f])
        for stays in (True, False):
            out[f"markout_{h}_level_{'stays' if stays else 'gone'}"] = _mean_se(
                [d["markouts"][h] for d in f if d["level_stays"] == stays])
    if f and "through" in f[0]:
        out["through"] = float(np.mean([d["through"] for d in f]))
        out["alone_share"] = float(np.mean([d["alone_s"] > 0 for d in f]))
        out["alone_s_median"] = float(np.median([d["alone_s"] for d in f]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shard_root", required=True)
    p.add_argument("--sampler", choices=sorted(SAMPLERS), default="mask")
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    by_key = {(e["month"], e["message"]): e for e in entries}
    n = a.window + max(MARKOUTS) + 1
    n_sim = int(np.ceil(n * KEEP_STRETCH)) if a.sampler == "keep" else n
    hist = sample_episodes(entries, a.shard_root, a.n_episodes, a.seed, history=cfg["history"], n_messages=8 * n)
    eps = [Episode(by_key[(h["month"], h["message"])], h["start"], h["seed"]) for h in hist]
    sides = {(ep.entry["message"], ep.start): BUY if i % 2 == 0 else SELL for i, ep in enumerate(eps)}

    t0 = time.time()
    market = _Recording(load_newtok(cfg["simulator"], torch.device("cuda")), a.shard_root, cfg["add_dir"],
                        cfg["batch"], history=cfg["history"], n_messages=n_sim, latency=0, inventory_cap=10**9,
                        **SAMPLERS[a.sampler])
    market.sides = sides
    market.variant = "agent"
    tr_agent = rollout(QuoteOnce(sides), None, market, eps, every=n_sim)
    market.variant = "none"
    tr_none = rollout(NoTrade(), None, market, eps, every=n_sim)
    seconds = time.time() - t0

    res = dict(config=a.config, sampler=a.sampler, n_episodes=len(eps), window=a.window, seed=a.seed,
               n_sim_steps=n_sim, seconds=seconds, torch=torch.__version__, sides={})
    acc = {s: dict(real=[], agent=[], none=[], same_queue=0, refused=0, short=0, gen={}) for s in ("bid", "ask")}
    for ep, ta, tn in zip(eps, tr_agent, tr_none):
        key = (ep.entry["message"], ep.start)
        side = sides[key]
        v = acc["bid" if side == BUY else "ask"]
        for g, c in ta.gen.items():
            v["gen"][g] = v["gen"].get(g, 0) + int(c)
        if ta.refused:
            v["refused"] += 1
            continue
        day = DayFiles.open(a.shard_root, ep.entry)
        bk = day.book_before(ep.start)
        bside = BID if side == BUY else ASK
        price = bk.best(bside)
        real_q = [(o.oid, o.size) for o in bk.queue(bside, price)]
        v["same_queue"] += int(real_q == market.ahead[key + ("agent",)] == market.ahead[key + ("none",)])
        ahead = {o for o, _ in real_q}
        rp = real_path(day, ep.start, side, n)
        pa = sim_path(market, key + ("agent",), ta.mid)
        pn = sim_path(market, key + ("none",), tn.mid)
        if min(len(pa["mids"]), len(pn["mids"])) < n:
            v["short"] += 1                                  # keep: fewer rows than the window needs
        hit = pa["rows"][(pa["rows"][:, 1] >= AGENT_OID0) & (pa["rows"][:, 2] == 4)]
        k_agent = int(hit[0, 0]) if len(hit) else None
        v["real"].append(describe(rp, counterfactual(rp["rows"], ahead, side, price), real_q, side, price, a.window))
        v["agent"].append(describe(pa, k_agent, real_q, side, price, a.window))
        v["none"].append(describe(pn, counterfactual(pn["rows"], ahead, side, price), real_q, side, price, a.window))
    for s, v in acc.items():
        m = len(v["real"])
        res["sides"][s] = dict(episodes=m, refused=v["refused"], short=v["short"],
                               same_queue=v["same_queue"] / max(m, 1),
                               ahead_shares_median=float(np.median([d["ahead"] for d in v["real"]])) if m else None,
                               gen=v["gen"], **{k: stats(v[k], a.window) for k in ("real", "agent", "none")})
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump({s: {k: acc[s][k] for k in ("real", "agent", "none")} for s in acc},
              open(a.out.replace(".json", "_episodes.json"), "w"))

    def r2(x):
        return None if x is None else round(x, 3)
    for s, v in res["sides"].items():
        print(f"== {a.sampler} {s}: {v['episodes']} episodes, refused {v['refused']}, short {v['short']}, "
              f"same queue {v['same_queue']:.3f}, ahead median {v['ahead_shares_median']}")
        for k in ("real", "none", "agent"):
            x = v[k]
            print(f"  {k:5s} filled {x['filled']:.3f} " + " ".join(f"<{w} {x[f'filled_{w}']:.3f}" for w in WITHIN)
                  + f" k_med {x['fill_k_median']} s_med {r2(x['fill_s_median'])}"
                  + f" exec {r2(x['ahead_exec_share']['mean'])} cancel {r2(x['ahead_cancel_share']['mean'])}"
                  + f" away {x['moved_away']:.3f} stays {r2(x['level_stays'])} size {x['fill_msg_size_median']}"
                  + f" move_std {x['move_std_bps']:.2f} window_s {x['window_s_median']:.1f} mo "
                  + " ".join(f"h{h} {r2(x[f'markout_{h}']['mean'])}" for h in MARKOUTS))
    print(f"{seconds:.0f} s")


if __name__ == "__main__":
    main()
