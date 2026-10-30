#!/usr/bin/env python3
"""Generate LOB-Bench inputs with SimMarket under one of three samplers (one GPU).

Histories: --n_histories starts drawn by episodes.sample_episodes (seed
--seed) over the config's month. Each history gets --n_gen background
messages from the simulator (NoTrade) and is written in LOBSTER form:

    data_cond/<T>_<date>_message_real_id_<i>.csv     the --n_cond real rows before the start
    data_cond/<T>_<date>_orderbook_real_id_<i>.csv   the L10 book before each of them and after the last
    data_real/<T>_<date>_{message,orderbook}_real_id_<i>.csv          the real rows of the next --n_gen messages
    data_gen/<T>_<date>_{message,orderbook}_real_id_<i>_gen_id_0.csv  the generated rows

Message columns time (s, 9 decimals), type, order id, size, price, direction
(1 buy, -1 sell); book rows are the 40 L10 columns after each row. Order ids
are renumbered from 1 per history (the scorer reads them as Int32).

Samplers: mask (every draw masked to sampling.legal_tokens), reject (32
whole-message draws, then masks), keep (a rejected draw stays in the token
stream and the book is unchanged, nano-sigma's rule; it writes no row).
Two negative controls draw uniformly among the legal tokens instead of from
the model: uniform at every position, uniform_fields only at the price, size,
queue-slot and end-of-execution positions (event, direction and dt still
come from the model, under masks). If uniform_fields scored like mask, the
masks and not the model would make the score.

    python run/post_training/policy_bench/lobbench_gen.py --config configs/policy_bench/v1.yaml \
        --shard_root <mount> --sampler mask --out <dir> [--n_histories 864 --shard 0 --n_shards 3]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from post_training.policy_bench.codec import burst_bounds
from post_training.policy_bench.daydata import DayFiles
from post_training.policy_bench.episodes import sample_episodes
from post_training.policy_bench.policies import NoTrade
from post_training.policy_bench.rollout import rollout
from post_training.policy_bench.sampling import MAX_REJECTIONS, legal_tokens
from post_training.policy_bench.sim_market import SimMarket
from post_training.policy_bench.simulator import load_newtok
from post_training.policy_bench.types import Episode

SAMPLERS = {"mask": dict(max_rejections=0), "reject": dict(max_rejections=MAX_REJECTIONS),
            "keep": dict(max_rejections=MAX_REJECTIONS, keep_invalid=True),
            "uniform": dict(max_rejections=0), "uniform_fields": dict(max_rejections=0)}
FIELDS = frozenset({"sign", "sign_or_eoe", "price1", "price2", "size1", "size2", "qslot"})
UNIFORM = {"uniform": None, "uniform_fields": FIELDS}        # positions drawn uniformly (None: all)
RAW = dict(oid=0, evt=1, d=2, price=3, size=5, t_s=8, t_ns=9)


class _Recording(SimMarket):
    """SimMarket that keeps every row it writes: (t_ns, evt, oid, size, price, d) and the L10 after."""

    def load(self, i, ep):
        super().load(i, ep)
        self.rows[i].rec = []

    def _row(self, r, oid, evt, d, price, size, dt, t):
        super()._row(r, oid, evt, d, price, size, dt, t)
        r.rec.append(((t, evt, oid, size, price, d), r.log.now[3:].copy()))

    def trace(self, i):
        self.out[(self.rows[i].ep.entry["message"], self.rows[i].ep.start)] = self.rows[i].rec
        return super().trace(i)


class _Uniform(_Recording):
    """A negative control: at the positions in `kinds` (None: all) a legal token drawn uniformly."""

    kinds: frozenset | None = None

    def _draw(self, r, logits):
        st = r.state
        if st.forced() is None and (self.kinds is None or st.kind() in self.kinds):
            ok = legal_tokens(st, r.book, self._mid(r), r.end)
            return int(ok[r.rng.integers(len(ok))])
        return super()._draw(r, logits)


def lobster_rows(msgs, ids):
    out = []
    for t, evt, oid, size, price, d in msgs:
        k = ids.setdefault(int(oid), len(ids) + 1)
        out.append(f"{int(t) // 10**9}.{int(t) % 10**9:09d},{int(evt)},{k},{int(size)},{int(price)},"
                   f"{1 if int(d) == 1 else -1}")
    return "\n".join(out) + "\n"


def book_rows(books):
    return "\n".join(",".join(str(int(x)) for x in b) for b in books) + "\n"


def raw_msgs(day, lo, hi):
    m = day.rows(lo, hi)
    return [(int(x[RAW["t_s"]]) * 10**9 + int(x[RAW["t_ns"]]), x[RAW["evt"]], x[RAW["oid"]], x[RAW["size"]],
             x[RAW["price"]], x[RAW["d"]]) for x in m]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shard_root", required=True)
    p.add_argument("--sampler", choices=sorted(SAMPLERS), required=True)
    p.add_argument("--n_histories", type=int, default=864)
    p.add_argument("--n_gen", type=int, default=500)
    p.add_argument("--n_cond", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    by_key = {(e["month"], e["message"]): e for e in entries}
    hist = sample_episodes(entries, a.shard_root, a.n_histories, a.seed, history=cfg["history"],
                           n_messages=4 * a.n_gen)
    ids = list(range(a.n_histories))[a.shard::a.n_shards]
    eps = [Episode(by_key[(hist[i]["month"], hist[i]["message"])], hist[i]["start"], hist[i]["seed"]) for i in ids]

    dev = torch.device("cuda")
    cls = _Uniform if a.sampler in UNIFORM else _Recording
    market = cls(load_newtok(cfg["simulator"], dev), a.shard_root, cfg["add_dir"], cfg["batch"],
                 history=cfg["history"], n_messages=a.n_gen, latency=cfg["latency"],
                 inventory_cap=cfg["inventory_cap"], **SAMPLERS[a.sampler])
    market.out = {}
    market.kinds = UNIFORM.get(a.sampler)
    t0 = time.time()
    traces = rollout(NoTrade(), None, market, eps, every=a.n_gen)
    seconds = time.time() - t0

    for sub in ("data_cond", "data_real", "data_gen"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    gen_stats = {}
    for i, ep in zip(ids, eps):
        day = DayFiles.open(a.shard_root, ep.entry)
        stem = os.path.basename(ep.entry["message"]).split("_")
        tag = f"{stem[0]}_{stem[1]}"
        rid = {}
        s = ep.start
        cond = raw_msgs(day, s - a.n_cond, s)
        cond_books = np.asarray(day.l10[s - a.n_cond:s + 1])[:, 3:]
        raw = np.asarray(day.messages[s:s + 4 * a.n_gen])
        bounds = burst_bounds(raw)
        r1 = bounds[a.n_gen][0] if len(bounds) > a.n_gen else len(raw)
        real = raw_msgs(day, s, s + r1)
        real_books = np.asarray(day.l10[s + 1:s + r1 + 1])[:, 3:]
        rec = market.out[(ep.entry["message"], ep.start)]
        gen = [m for m, _ in rec]
        gen_books = np.stack([b for _, b in rec]) if rec else np.zeros((0, 40), np.int64)
        base = os.path.join(a.out, "{}", f"{tag}_{{}}_real_id_{i}")
        with open(base.format("data_cond", "message") + ".csv", "w") as f:
            f.write(lobster_rows(cond, rid))
        with open(base.format("data_cond", "orderbook") + ".csv", "w") as f:
            f.write(book_rows(cond_books))
        with open(base.format("data_real", "message") + ".csv", "w") as f:
            f.write(lobster_rows(real, dict(rid)))
        with open(base.format("data_real", "orderbook") + ".csv", "w") as f:
            f.write(book_rows(real_books))
        with open(base.format("data_gen", "message") + "_gen_id_0.csv", "w") as f:
            f.write(lobster_rows(gen, dict(rid)))
        with open(base.format("data_gen", "orderbook") + "_gen_id_0.csv", "w") as f:
            f.write(book_rows(gen_books))
    for t in traces:
        for k, v in t.gen.items():
            gen_stats[k] = gen_stats.get(k, 0) + v
    meta = dict(config=a.config, sampler=a.sampler, histories=len(eps), ids=ids, n_gen=a.n_gen, n_cond=a.n_cond,
                seed=a.seed, seconds=seconds, torch=torch.__version__, gen=gen_stats)
    json.dump(meta, open(os.path.join(a.out, f"meta_{a.sampler}_shard{a.shard}.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("sampler", "histories", "seconds")}), json.dumps(gen_stats))


if __name__ == "__main__":
    main()
