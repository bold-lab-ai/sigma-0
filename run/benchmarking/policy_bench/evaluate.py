#!/usr/bin/env python3
"""Evaluate one policy on the bench: one policy in, one result out (one GPU).

Everything but the policy comes from the config (configs/bench/policy_bench_v1.yaml):
simulator, signal, data, episodes, decision interval, latency, inventory cap
and fees. The policy is a class path and keyword arguments. Writes the traces
and, for every terminal rule of the config, score() with the config's fees.

    python run/benchmarking/policy_bench/evaluate.py --config configs/bench/policy_bench_v1.yaml \
        --shard_root <mount of the config's shards> \
        --policy lobpipeline.policy_bench.policies:SignRule --kwargs '{"index": 2, "size": 100}' \
        --out <dir>/<name>.json [--market replay] [--limit N]
"""

from __future__ import annotations

import argparse
import importlib
import json
import time

import torch
import yaml

from lobpipeline.policy_bench import trace as trace_io
from lobpipeline.policy_bench.costs import MakerTaker
from lobpipeline.policy_bench.episodes import load_episodes
from lobpipeline.policy_bench.market import BookFill, ReplayMarket
from lobpipeline.policy_bench.rollout import rollout
from lobpipeline.policy_bench.sampling import MAX_REJECTIONS
from lobpipeline.policy_bench.score import score
from lobpipeline.policy_bench.signal import Signal
from lobpipeline.policy_bench.sim_market import SimMarket
from lobpipeline.policy_bench.simulator import load_newtok
from lobpipeline.policy_bench.terminal import FlattenTaker, MarkToMicro, MarkToMid

TERMINALS = {"mark_to_mid": MarkToMid, "mark_to_micro": MarkToMicro, "flatten_taker": FlattenTaker}


def load_policy(path: str, kwargs: dict):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)(**kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shard_root", required=True, help="a mount of the config's shards, <root>/<month>/")
    p.add_argument("--policy", required=True, help="module:Class")
    p.add_argument("--kwargs", default="{}", help="JSON keyword arguments of the policy")
    p.add_argument("--market", choices=("sim", "replay"), default="sim")
    p.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke)")
    p.add_argument("--max_rejections", type=int, default=None,
                   help="SimMarket draws before the masks (default: the config's, else 32; 0: every draw masked)")
    p.add_argument("--latency_ns", type=int, default=None,
                   help="orders reach the market this many ns after the decision (default: the config's "
                        "latency_ns; if neither is set, the config's latency in messages)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    eps = load_episodes(cfg["episodes"], entries)
    if a.limit:
        eps = eps[:a.limit]
    policy = load_policy(a.policy, json.loads(a.kwargs))
    dev = torch.device("cuda")
    signal = Signal(cfg["signal"], dev)
    latency_ns = a.latency_ns if a.latency_ns is not None else cfg.get("latency_ns")
    common = dict(history=cfg["history"], n_messages=cfg["n_messages"], latency=cfg["latency"],
                  inventory_cap=cfg["inventory_cap"], latency_ns=latency_ns)
    max_rej = a.max_rejections if a.max_rejections is not None else cfg.get("max_rejections", MAX_REJECTIONS)
    if a.market == "sim":
        market = SimMarket(load_newtok(cfg["simulator"], dev), a.shard_root, cfg["add_dir"], cfg["batch"],
                           max_rejections=max_rej, **common)
    else:
        market = ReplayMarket(a.shard_root, cfg["add_dir"], BookFill(), **common)
    t0 = time.time()
    traces = rollout(policy, signal, market, eps, every=cfg["every"])
    cost = MakerTaker(rebate=cfg["maker_rebate"], fee=cfg["taker_fee"])
    res = dict(config=cfg, policy=a.policy, kwargs=json.loads(a.kwargs), market=a.market,
               max_rejections=max_rej if a.market == "sim" else None, latency_ns=latency_ns, episodes=len(eps),
               seconds=time.time() - t0, torch=torch.__version__,
               scores={name: score(traces, TERMINALS[name](), cost) for name in cfg["terminals"]})
    trace_io.save(a.out.replace(".json", "_traces.npz"), traces)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["scores"], indent=1))


if __name__ == "__main__":
    main()
