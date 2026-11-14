#!/usr/bin/env python3
"""A resting order at the back of the best queue: simulator against the real market (one GPU).

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

--order_size sets the order's shares (default 1). With more than one share the
order can fill in parts: the counterfactual gives it, in order, the shares of
every execution that would have reached it (through its price, or at its price
behind the queue ahead), up to its size. Per path: the first fill (all the
measures above), the full fill, the share of the order filled within the
window, and markouts weighted by the shares of each fill.

Level answers (real and none paths, all starts): every background execution in
the window, with the shares at the best price of its side just before it, is
classed as taking part of that level, exactly that level, or more (a deeper
price or more shares than shown). With the mid's move after it in the
aggressor's direction, the half spread before it, and the markouts of the resting
side at the level's price (a maker that was really filled, no counterfactual).
This asks how a thin best level is taken, and what its last shares earn, in the
market and in the simulator without the order.

Order fates (real and none paths): every background order that joins the queue
at the best price of its side (an add at the best price) in the first
--window - FATE_HORIZON messages, followed for FATE_HORIZON messages: whether
and when it is first executed, cancelled before any execution, and the markouts
of its first execution. Real orders are cancelled by their owners, so this is
not the agent's never-cancelled order, but it needs no counterfactual: it
compares the simulator's makers with the market's. --skip_agent runs the
simulator without the order only.

    python run/benchmarking/policy_bench/check_fills.py --config configs/bench/policy_bench_v1.yaml \
        --shard_root <mount> --out <file.json> [--sampler mask] [--order_size 1] [--n_episodes 500] [--window 512]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import yaml

from lobpipeline.policy_bench.book import ASK, BID
from lobpipeline.policy_bench.codec import NS, burst_bounds
from lobpipeline.policy_bench.daydata import DayFiles
from lobpipeline.policy_bench.episodes import sample_episodes
from lobpipeline.policy_bench.policies import NoTrade
from lobpipeline.policy_bench.rollout import rollout
from lobpipeline.policy_bench.sampling import MAX_REJECTIONS
from lobpipeline.policy_bench.sim_market import AGENT_OID0, SimMarket
from lobpipeline.policy_bench.simulator import load_newtok
from lobpipeline.policy_bench.types import BUY, SELL, Episode, LimitOrder

OID, EVT, DIR, PRICE, SIZE, TS, TNS = 0, 1, 2, 3, 5, 8, 9
WITHIN = [16, 64, 256]
MARKOUTS = [1, 16, 64, 128]
SAMPLERS = {"mask": dict(max_rejections=0), "reject": dict(max_rejections=MAX_REJECTIONS),
            "keep": dict(max_rejections=MAX_REJECTIONS, keep_invalid=True)}
KEEP_STRETCH = 2.0          # keep writes rows on about 75% of steps (lobbench/README.md); 1.6 was short (8363489)
LEVEL_BUCKETS = [(1, 1), (2, 9), (10, 99), (100, 10**12)]     # shares at the best before an execution
LEVEL_MOVES = [1, 16, 64]
OUTCOMES = ("part", "exact", "beyond")
FATE_HORIZON = 256
FATE_WITHIN = [16, 64, 256]
FATE_SIZES = [(1, 99), (100, 100), (101, 10**12)]
# A path: rows (m, 12) int64 [message, oid, evt, d, price, size, t_ns, best price of the quoted side
# after the row, ask price, ask shares, bid price, bid shares after the row], mids (n,) after each
# message, t0 (ns of the last history row), top0 (ask price, ask shares, bid price, bid shares before
# message 0).


class QuoteOnce:
    """Rests `size` shares at the best price of the episode's side at k = 0, then nothing."""

    def __init__(self, sides: dict, size: int = 1) -> None:
        self.sides = sides
        self.size = size

    def reset(self, episode) -> None:
        self.side = self.sides[(episode.entry["message"], episode.start)]

    def step(self, obs) -> list:
        if obs.k:
            return []
        return [LimitOrder(self.side, int(obs.l10[2] if self.side == BUY else obs.l10[0]), self.size)]


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
        self.paths[key] = dict(t0=self._t_prev(r), rows=[], col=5 if side == BID else 3,
                               top0=np.asarray(r.log.now[3:7], np.int64).copy())

    def _row(self, r, oid, evt, d, price, size, dt, t):
        super()._row(r, oid, evt, d, price, size, dt, t)
        p = self.paths[(r.ep.entry["message"], r.ep.start, self.variant)]
        now = r.log.now
        p["rows"].append((r.k, oid, evt, d, price, size, t, int(now[p["col"]]),
                          int(now[3]), int(now[4]), int(now[5]), int(now[6])))


def mid_or_nan(l10_row) -> float:
    a, b = int(l10_row[3]), int(l10_row[5])
    return (a + b) / 2 if 0 < b < a < 9_000_000_000 else np.nan


def real_path(day: DayFiles, start: int, side: int, n: int) -> dict:
    rows = np.asarray(day.rows(start, start + 8 * n), np.int64)
    bounds = burst_bounds(rows)[:n]
    rows = rows[:bounds[-1][1]]
    k = np.repeat(np.arange(len(bounds)), [j - i for i, j in bounds])
    col = 5 if side == BUY else 3
    after = np.asarray(day.l10[start + 1:start + 1 + len(rows)], np.int64)     # l10[x]: book before row x
    t = rows[:, TS] * NS + rows[:, TNS]
    mids = np.array([mid_or_nan(day.l10[start + j]) for _, j in bounds])
    t0 = int(day.l10[start, 1]) * NS + int(day.l10[start, 2])
    out = np.column_stack([k, rows[:, OID], rows[:, EVT], rows[:, DIR], rows[:, PRICE], rows[:, SIZE], t,
                           after[:, col], after[:, 3:7]])
    return dict(rows=out, mids=mids, t0=t0, top0=np.asarray(day.l10[start, 3:7], np.int64), real=True)


def sim_path(market: _Recording, key: tuple, mids: np.ndarray) -> dict:
    """The recorded path on the axis of steps that wrote a row of a background message
    (the agent's own messages do not count; its fills come from background messages)."""
    p = market.paths[key]
    rows = np.array(p["rows"], np.int64).reshape(-1, 12)
    own = (rows[:, 1] >= AGENT_OID0) & (rows[:, 2] != 4)
    valid = np.unique(rows[~own, 0])
    rank = np.full(int(rows[:, 0].max()) + 2 if len(rows) else 1, -1, np.int64)
    rank[valid] = np.arange(len(valid))
    rows = rows[rank[rows[:, 0]] >= 0]
    rows[:, 0] = rank[rows[:, 0]]
    return dict(rows=rows, mids=mids[1:][valid], t0=p["t0"], top0=p["top0"])   # mids[m + 1]: after step m


def counterfactual(rows: np.ndarray, ahead: set, side: int, price: int, size: int) -> list[tuple[int, int]]:
    """Fills (message, shares) of an order of `size` at `price` behind `ahead`: in order, the shares of
    every execution on its side through its price, or at its price of an order not queued ahead."""
    ours = 1 if side == BUY else 0
    ex = (rows[:, 2] == 4) & (rows[:, 3] == ours) & (rows[:, 1] < AGENT_OID0)
    through = rows[:, 4] < price if side == BUY else rows[:, 4] > price
    behind = (rows[:, 4] == price) & ~np.isin(rows[:, 1], list(ahead))
    out, left = [], size
    for i in np.flatnonzero(ex & (through | behind)):
        take = min(left, int(rows[i, 5]))
        out.append((int(rows[i, 0]), take))
        left -= take
        if not left:
            break
    return out


def agent_fills(rows: np.ndarray) -> list[tuple[int, int]]:
    hit = rows[(rows[:, 1] >= AGENT_OID0) & (rows[:, 2] == 4)]
    return [(int(k), int(q)) for k, q in hit[:, [0, 5]]]


def level_answers(path: dict, window: int) -> list[tuple]:
    """(shares at the best of its side before it, outcome, move at LEVEL_MOVES, half spread, maker
    markout at MARKOUTS) per background execution message in the window. Outcome 0 part of the level,
    1 exactly the level, 2 beyond it. The move is the mid h messages after against the mid before, in
    the aggressor's direction; the maker markout is the mid h messages after against the best price
    before, signed for the resting side; all in bps. For paths without agent rows (real, none)."""
    rows, mids = path["rows"], path["mids"]
    b = np.searchsorted(rows[:, 0], np.arange(window + 1))
    prev, out = np.asarray(path["top0"], np.int64), []
    for m in range(window):
        r = rows[b[m]:b[m + 1]]
        if not len(r):
            continue
        ex = r[(r[:, 2] == 4) & (r[:, 1] < AGENT_OID0)]
        if len(ex):
            bid = int(ex[0, 3]) == 1                     # direction 1: the bid side is hit
            p0, v0 = (prev[2], prev[3]) if bid else (prev[0], prev[1])
            s = int(ex[:, 5].sum())
            outcome = 2 if (ex[:, 4] != p0).any() or s > v0 else int(s == v0)
            ok = 0 < prev[2] < prev[0] < 9_000_000_000
            mid0 = (prev[0] + prev[2]) / 2
            sign = -1 if bid else 1
            moves = [sign * (mids[m + h] - mid0) / mid0 * 1e4 if ok and m + h < len(mids) else np.nan
                     for h in LEVEL_MOVES]
            maker = [-sign * (mids[m + h] - p0) / p0 * 1e4 if ok and m + h < len(mids) else np.nan
                     for h in MARKOUTS]
            half = (prev[0] - prev[2]) / 2 / mid0 * 1e4 if ok else np.nan
            out.append((int(v0), outcome, *moves, half, *maker))
        prev = r[-1, 8:12]
    return out


def order_fates(path: dict, window: int) -> list[tuple]:
    """(size, first execution after (messages, -1 none), cancelled before any execution (0/1), markout
    of the first execution at MARKOUTS) per background order joining the best queue in messages
    [0, window - FATE_HORIZON), followed FATE_HORIZON messages. For paths without agent rows."""
    rows, mids = path["rows"], path["mids"]
    b = np.searchsorted(rows[:, 0], np.arange(window + 1))
    prev, live, out = np.asarray(path["top0"], np.int64), {}, []
    for m in range(window):
        r = rows[b[m]:b[m + 1]]
        for o in [o for o, f in live.items() if m - f["k0"] >= FATE_HORIZON]:
            out.append(live.pop(o)["rec"])
        for row in r:
            oid, evt, d, price, size = (int(x) for x in row[1:6])
            if evt == 1:
                best = prev[2] if d == 1 else prev[0]
                if m < window - FATE_HORIZON and price == best:
                    live[oid] = dict(k0=m, rec=[size, -1, 0] + [np.nan] * len(MARKOUTS))
            elif oid in live:
                f = live[oid]
                if evt == 4 and f["rec"][1] < 0:
                    f["rec"][1] = m - f["k0"]
                    sgn = 1 if d == 1 else -1                 # a bid that is executed bought
                    f["rec"][3:] = [sgn * (mids[m + h] - price) / price * 1e4 if m + h < len(mids) else np.nan
                                    for h in MARKOUTS]
                elif evt == 3 and f["rec"][1] < 0:
                    f["rec"][2] = 1
                    out.append(live.pop(oid)["rec"])
        if len(r):
            prev = r[-1, 8:12]
    return out + [f["rec"] for f in live.values()]


def fate_stats(fates: list) -> dict:
    a = np.array(fates, np.float64).reshape(-1, 3 + len(MARKOUTS))
    out = {}
    for lo, hi in [(1, 10**12)] + FATE_SIZES:
        x = a[(a[:, 0] >= lo) & (a[:, 0] <= hi)]
        hit = x[x[:, 1] >= 0]
        d = dict(n=len(x), **{f"executed_{w}": float(((x[:, 1] >= 0) & (x[:, 1] < w)).mean()) if len(x) else None
                              for w in FATE_WITHIN},
                 deleted_first=float(x[:, 2].mean()) if len(x) else None,
                 first_exec_median=float(np.median(hit[:, 1])) if len(hit) else None,
                 **{f"markout_{h}": _mean_se(hit[:, 3 + j]) for j, h in enumerate(MARKOUTS)})
        out["all" if hi == 10**12 and lo == 1 else (f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 10**12 else f"{lo}+")] = d
    return out


def level_stats(answers: list[tuple]) -> dict:
    nm = len(LEVEL_MOVES)
    a = np.array(answers, np.float64).reshape(-1, 3 + nm + len(MARKOUTS))
    out = {}
    for lo, hi in LEVEL_BUCKETS:
        x = a[(a[:, 0] >= lo) & (a[:, 0] <= hi)]
        d = dict(n=len(x), **{o: float((x[:, 1] == i).mean()) if len(x) else None for i, o in enumerate(OUTCOMES)})
        for j, h in enumerate(LEVEL_MOVES):
            d[f"move_{h}"] = _mean_se(x[:, 2 + j])
            for i, o in enumerate(OUTCOMES):
                d[f"move_{h}_{o}"] = _mean_se(x[x[:, 1] == i, 2 + j])
        d["half_spread"] = _mean_se(x[:, 2 + nm])
        for j, h in enumerate(MARKOUTS):
            d[f"maker_{h}"] = _mean_se(x[:, 3 + nm + j])
            d[f"maker_{h}_exact"] = _mean_se(x[x[:, 1] == 1, 3 + nm + j])
        out[f"{lo}-{hi}" if hi < 10**12 else f"{lo}+"] = d
    return out


def describe(path: dict, fills: list, ahead: list, side: int, price: int, size: int, window: int) -> dict:
    """How the episode went for the quote, with its fills [(message, shares)] (first fill k)."""
    rows, mids = path["rows"], path["mids"]
    ours = 1 if side == BUY else 0
    ids = list({o for o, _ in ahead})
    fills = [(k, q) for k, q in fills if k < window]
    k = fills[0][0] if fills else None
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
    cum = np.cumsum([q for _, q in fills]) if fills else np.zeros(0)
    d["filled_frac"] = float(cum[-1] / size) if fills else 0.0
    d["k_full"] = int(fills[int(np.searchsorted(cum, size))][0]) if fills and cum[-1] >= size else None
    w = {}
    for h in MARKOUTS:
        v = [(q, side * (mids[j + h] - price) / price * 1e4) for j, q in fills
             if j + h < len(mids) and np.isfinite(mids[j + h])]
        w[h] = float(sum(q * x for q, x in v) / sum(q for q, _ in v)) if v else None
    d["markouts_w"] = w
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
    kf = np.array([np.inf if d["k_full"] is None else d["k_full"] for d in ds], np.float64)
    out.update(filled_full=float((kf < window).mean()),
               **{f"filled_full_{w}": float((kf < w).mean()) for w in WITHIN},
               fill_full_k_median=float(np.median(kf[kf < window])) if (kf < window).any() else None,
               filled_frac=float(np.mean([d["filled_frac"] for d in ds])),
               **{f"markout_w_{h}": _mean_se([d["markouts_w"][h] for d in f]) for h in MARKOUTS})
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
    p.add_argument("--order_size", type=int, default=1)
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--out", required=True)
    p.add_argument("--real_only", action="store_true", help="the real path and its level answers only (CPU)")
    p.add_argument("--skip_agent", action="store_true", help="the simulator without the order only")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    by_key = {(e["month"], e["message"]): e for e in entries}
    n = a.window + max(MARKOUTS) + 1
    n_sim = int(np.ceil(n * KEEP_STRETCH)) if a.sampler == "keep" else n
    hist = sample_episodes(entries, a.shard_root, a.n_episodes, a.seed, history=cfg["history"], n_messages=8 * n)
    eps = [Episode(by_key[(h["month"], h["message"])], h["start"], h["seed"]) for h in hist]
    sides = {(ep.entry["message"], ep.start): BUY if i % 2 == 0 else SELL for i, ep in enumerate(eps)}

    paths = ("real",) if a.real_only else ("real", "none") if a.skip_agent else ("real", "agent", "none")
    t0 = time.time()
    if a.real_only:
        tr_agent = tr_none = [None] * len(eps)
    else:
        market = _Recording(load_newtok(cfg["simulator"], torch.device("cuda")), a.shard_root, cfg["add_dir"],
                            cfg["batch"], history=cfg["history"], n_messages=n_sim, latency=0,
                            inventory_cap=10**9, **SAMPLERS[a.sampler])
        market.sides = sides
        market.variant = "agent"
        tr_agent = [None] * len(eps) if a.skip_agent else rollout(QuoteOnce(sides, a.order_size), None, market,
                                                                   eps, every=n_sim)
        market.variant = "none"
        tr_none = rollout(NoTrade(), None, market, eps, every=n_sim)
    seconds = time.time() - t0

    res = dict(config=a.config, sampler=a.sampler, order_size=a.order_size, n_episodes=len(eps),
               window=a.window, seed=a.seed, n_sim_steps=n_sim, seconds=seconds, torch=torch.__version__,
               real_only=a.real_only, skip_agent=a.skip_agent, sides={})
    acc = {s: dict(real=[], agent=[], none=[], same_queue=0, refused=0, short=0, gen={}) for s in ("bid", "ask")}
    levels, fates = {"real": [], "none": []}, {"real": [], "none": []}
    for ep, ta, tn in zip(eps, tr_agent, tr_none):
        key = (ep.entry["message"], ep.start)
        side = sides[key]
        v = acc["bid" if side == BUY else "ask"]
        tg = ta if ta is not None else tn
        if tg is not None:
            for g, c in tg.gen.items():
                v["gen"][g] = v["gen"].get(g, 0) + int(c)
            if tg.refused:
                v["refused"] += 1
                continue
        day = DayFiles.open(a.shard_root, ep.entry)
        bk = day.book_before(ep.start)
        bside = BID if side == BUY else ASK
        price = bk.best(bside)
        real_q = [(o.oid, o.size) for o in bk.queue(bside, price)]
        ahead = {o for o, _ in real_q}
        rp = real_path(day, ep.start, side, n)
        v["real"].append(describe(rp, counterfactual(rp["rows"], ahead, side, price, a.order_size), real_q, side,
                                  price, a.order_size, a.window))
        levels["real"] += level_answers(rp, a.window)
        fates["real"] += order_fates(rp, a.window)
        if tn is None:
            continue
        pn = sim_path(market, key + ("none",), tn.mid)
        v["same_queue"] += int(real_q == market.ahead[key + ("none",)]
                               and (ta is None or real_q == market.ahead[key + ("agent",)]))
        if len(pn["mids"]) < n:
            v["short"] += 1                                  # keep: fewer rows than the window needs
        if ta is not None:
            pa = sim_path(market, key + ("agent",), ta.mid)
            v["short"] += int(len(pa["mids"]) < n <= len(pn["mids"]))
            v["agent"].append(describe(pa, agent_fills(pa["rows"]), real_q, side, price, a.order_size, a.window))
        v["none"].append(describe(pn, counterfactual(pn["rows"], ahead, side, price, a.order_size), real_q, side,
                                  price, a.order_size, a.window))
        levels["none"] += level_answers(pn, a.window)
        fates["none"] += order_fates(pn, a.window)
    for s, v in acc.items():
        m = len(v["real"])
        res["sides"][s] = dict(episodes=m, refused=v["refused"], short=v["short"],
                               same_queue=v["same_queue"] / max(m, 1),
                               ahead_shares_median=float(np.median([d["ahead"] for d in v["real"]])) if m else None,
                               gen=v["gen"], **{k: stats(v[k], a.window) for k in paths})
    res["level_answers"] = {k: level_stats(levels[k]) for k in levels if levels[k]}
    res["order_fates"] = {k: fate_stats(fates[k]) for k in fates if fates[k]}
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump(dict(**{s: {k: acc[s][k] for k in paths} for s in acc}, levels=levels, fates=fates),
              open(a.out.replace(".json", "_episodes.json"), "w"))

    def r2(x):
        return None if x is None else round(x, 3)
    for s, v in res["sides"].items():
        print(f"== {a.sampler} {s}: {v['episodes']} episodes, refused {v['refused']}, short {v['short']}, "
              f"same queue {v['same_queue']:.3f}, ahead median {v['ahead_shares_median']}")
        for k in [k for k in ("real", "none", "agent") if k in paths]:
            x = v[k]
            print(f"  {k:5s} filled {x['filled']:.3f} " + " ".join(f"<{w} {x[f'filled_{w}']:.3f}" for w in WITHIN)
                  + f" k_med {x['fill_k_median']} s_med {r2(x['fill_s_median'])}"
                  + f" exec {r2(x['ahead_exec_share']['mean'])} cancel {r2(x['ahead_cancel_share']['mean'])}"
                  + f" away {x['moved_away']:.3f} stays {r2(x['level_stays'])} size {x['fill_msg_size_median']}"
                  + f" move_std {x['move_std_bps']:.2f} window_s {x['window_s_median']:.1f} mo "
                  + " ".join(f"h{h} {r2(x[f'markout_{h}']['mean'])}" for h in MARKOUTS))
            print(f"        full {x['filled_full']:.3f} <64 {x['filled_full_64']:.3f} k_full_med "
                  f"{x['fill_full_k_median']} frac {x['filled_frac']:.3f} mo_w "
                  + " ".join(f"h{h} {r2(x[f'markout_w_{h}']['mean'])}" for h in MARKOUTS))
    for k, lv in res["level_answers"].items():
        for b, d in lv.items():
            print(f"levels {k:4s} {b:>6s}: n {d['n']} " + " ".join(f"{o} {r2(d[o])}" for o in OUTCOMES)
                  + " move " + " ".join(f"h{h} {r2(d[f'move_{h}']['mean'])}" for h in LEVEL_MOVES)
                  + " | exact h16 " + str(r2(d["move_16_exact"]["mean"])) + " beyond h16 "
                  + str(r2(d["move_16_beyond"]["mean"])) + f" | half {r2(d['half_spread']['mean'])} maker exact "
                  + " ".join(f"h{h} {r2(d[f'maker_{h}_exact']['mean'])}" for h in MARKOUTS))
    for k, fs in res["order_fates"].items():
        for b, d in fs.items():
            print(f"fates {k:4s} {b:>5s}: n {d['n']} " + " ".join(f"exec<{w} {r2(d[f'executed_{w}'])}" for w in FATE_WITHIN)
                  + f" deleted_first {r2(d['deleted_first'])} first_exec_med {d['first_exec_median']} mo "
                  + " ".join(f"h{h} {r2(d[f'markout_{h}']['mean'])}({r2(d[f'markout_{h}']['se'])})" for h in MARKOUTS))
    print(f"{seconds:.0f} s")


if __name__ == "__main__":
    main()
