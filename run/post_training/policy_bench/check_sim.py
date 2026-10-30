#!/usr/bin/env python3
"""G6, G7 and the signal half of G8: SimMarket with newtok_a_r1 (one GPU).

Episodes start at nano-sigma's test decision points (episodes.return_bench_decisions:
clear of bad rows), the first --n_episodes of a fixed shuffle.

G6a  NoTrade twice on the same episodes and seeds: identical traces.
G6b  Two markets on the same episode and seed, one with a single agent limit
     order one tick under the best bid, submitted and applied at k = 1. Right after
     it is applied the two books differ by exactly that order. Then both run
     on; the first message after which the mids differ is reported (the order
     itself does not move the mid: any difference is the flow reacting).
G7   Over NoTrade episodes of --n_messages: the share of background messages
     whose first draw is rejected, by reason, and the fallbacks to the masks.
     nano-sigma's LOB-Bench reports 11.3% invalid generated messages for
     new tokens (a), kept with the book unchanged: a different count of a
     different rule set, so the two are compared, not equated.
G8s  The signal on generated flow: at every decision, its prediction against
     the mid change over the next h generated messages (h <= what is left),
     pooled Pearson per horizon. On real data the head gives 0.386 / 0.343 /
     0.297 at 16 / 64 / 128 (G4).

    python run/post_training/policy_bench/check_sim.py --shard_root <v2_20261013 mnt> \
        --manifest <v3>/test.json --add_dir <add index> --sim_ckpt <newtok_a_r1>/ckpt.pt \
        --signal_ckpt <returnft_lobert_pass1>/ckpt.pt --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from post_training.policy_bench import trace as trace_io
from post_training.policy_bench.book import AGENT, ASK, BID
from post_training.policy_bench.episodes import return_bench_decisions
from post_training.policy_bench.policies import NoTrade
from post_training.policy_bench.rollout import rollout
from post_training.policy_bench.sampling import MAX_REJECTIONS
from post_training.policy_bench.signal import Signal
from post_training.policy_bench.sim_market import SimMarket
from post_training.policy_bench.simulator import load_newtok
from post_training.policy_bench.types import BUY, Episode, LimitOrder

PREFILL, HORIZONS = 3968, [16, 64, 128, 512, 1024]


def episodes(manifest, n, seed=0):
    ds = return_bench_decisions(json.load(open(manifest)), PREFILL, HORIZONS)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(ds), n, replace=False))
    return [Episode(ds[i][0], ds[i][1] + PREFILL, seed=int(i)) for i in pick]


def same(a, b):
    return all(np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True) for f in ("mid", "l10_end")) \
        and a.gen == b.gen


def g6b(model, a, ep, n_steps):
    """Two markets, one with an agent bid one tick under the best bid."""
    mk = [SimMarket(model, a.shard_root, a.add_dir, 1, n_messages=n_steps, max_rejections=a.max_rejections) for _ in range(2)]
    for m in mk:
        m.reset([ep])
        m.advance()                                   # k = 0, the same draw in both
    r0, r1 = mk[0].rows[0], mk[1].rows[0]
    out = dict(books_equal_before=r0.book.levels(BID) == r1.book.levels(BID)
               and r0.book.levels(ASK) == r1.book.levels(ASK))
    price = r1.book.best(BID) - 100
    mk[1].latency = 0                                 # due now, at k = 1
    mk[1].submit(0, [LimitOrder(BUY, price, 100)])
    mk[1]._activate(r1)                               # what advance() does first at k = 1
    extra = {p: [o for o in q if o not in r0.book.levels(BID).get(p, [])] for p, q in r1.book.levels(BID).items()}
    extra = {p: q for p, q in extra.items() if q}
    out.update(agent_price=price, agent_messages=r1.gen["agent_messages"],
               only_difference=(extra == {price: [(r1.next_agent - 1, 100, AGENT)]}
                                and r0.book.levels(ASK) == r1.book.levels(ASK)
                                and {p: q for p, q in r1.book.levels(BID).items() if p != price}
                                == {p: q for p, q in r0.book.levels(BID).items() if p != price}))
    while not mk[0].done():
        mk[0].advance()
        mk[1].advance()
    diff = [k for k, (x, y) in enumerate(zip(r0.mids, r1.mids)) if x != y]
    out["first_differing_mid"] = diff[0] if diff else None
    return out


def g8s(traces, horizons):
    xs = {h: ([], []) for h in horizons}
    for t in traces:
        for k, s in zip(t.decisions, t.signal):
            for j, h in enumerate(horizons):
                if k + h < len(t.mid) and np.isfinite(t.mid[k]) and np.isfinite(t.mid[k + h]):
                    xs[h][0].append(s[j])
                    xs[h][1].append((t.mid[k + h] - t.mid[k]) / t.mid[k] * 1e4)
    return {str(h): dict(n=len(p), pearson=float(np.corrcoef(p, y)[0, 1]) if len(p) > 2 else None,
                         realised_std_bps=float(np.std(y)) if y else None)
            for h, (p, y) in xs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_root", required=True)
    p.add_argument("--manifest", required=True, help="v3 test.json")
    p.add_argument("--add_dir", required=True)
    p.add_argument("--sim_ckpt", required=True)
    p.add_argument("--signal_ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_episodes", type=int, default=48)
    p.add_argument("--n_messages", type=int, default=1024)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--repeat_episodes", type=int, default=8)
    p.add_argument("--repeat_messages", type=int, default=128)
    p.add_argument("--max_rejections", type=int, default=MAX_REJECTIONS, help="0: every draw masked")
    a = p.parse_args()

    dev = torch.device("cuda")
    model = load_newtok(a.sim_ckpt, dev)
    sig = Signal(a.signal_ckpt, dev)
    eps = episodes(a.manifest, a.n_episodes)
    res = dict(episodes=len(eps), n_messages=a.n_messages, batch=a.batch, max_rejections=a.max_rejections,
               torch=torch.__version__)

    # G6a
    small = SimMarket(model, a.shard_root, a.add_dir, a.repeat_episodes, n_messages=a.repeat_messages,
                      max_rejections=a.max_rejections)
    t1 = rollout(NoTrade(), None, small, eps[:a.repeat_episodes])
    t2 = rollout(NoTrade(), None, small, eps[:a.repeat_episodes])
    res["g6a"] = dict(episodes=len(t1), messages=a.repeat_messages, identical=sum(same(x, y) for x, y in zip(t1, t2)))
    print("G6a", json.dumps(res["g6a"]), flush=True)
    del small
    torch.cuda.empty_cache()

    # G6b
    res["g6b"] = g6b(model, a, eps[0], a.repeat_messages)
    print("G6b", json.dumps(res["g6b"]), flush=True)
    torch.cuda.empty_cache()

    # G7 and G8s
    market = SimMarket(model, a.shard_root, a.add_dir, a.batch, n_messages=a.n_messages,
                       max_rejections=a.max_rejections)
    t0 = time.time()
    traces = rollout(NoTrade(), sig, market, eps, every=64)
    secs = time.time() - t0
    gen = Counter()
    for t in traces:
        gen.update(t.gen)
    msgs = gen["messages"]
    res["g7"] = dict(messages=msgs, seconds=secs, messages_per_s=msgs / secs,
                     first_draw_rejected_share=gen["first_rejected"] / msgs, fallbacks=gen["fallback"],
                     rejections_per_message=sum(v for k, v in gen.items() if k.startswith("rejected_")) / msgs,
                     reasons={k[9:]: v for k, v in sorted(gen.items()) if k.startswith("rejected_")},
                     reference_lobbench_invalid=0.113)
    print("G7", json.dumps(res["g7"]), flush=True)
    res["g8_signal_on_generated_flow"] = g8s(traces, HORIZONS)
    res["g8_reference_real_head_pearson"] = {"16": 0.3856, "64": 0.3430, "128": 0.2968}
    print("G8s", json.dumps(res["g8_signal_on_generated_flow"]), flush=True)
    trace_io.save(a.out.replace(".json", "_traces.npz"), traces)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
