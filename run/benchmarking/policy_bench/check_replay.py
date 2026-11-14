#!/usr/bin/env python3
"""G5 of the policy bench: ReplayMarket + the sign rule against nano-sigma's PnL bench (one GPU).

The PnL bench (tools/return_bench.py, pnl()) bets sign(prediction) x $1 at each
of the 70,183 January test decisions, at the mid at t, and values it at the
mid 128 rows later: variant A. Variant B also pays half the spread at t. Both
average the bets equally.

Here each decision is one episode of the bench itself: the 3,968 rows before t
rebuilt from our book, the signal on them, SignRule at k = 0 (one share,
latency 0), 128 replayed rows, MarkToMid, no fees. MidFill gives A; BookFill
fills at the best opposite price, which is B. With nano-sigma's own head on
the same decisions (G4, job 7842392) the reference is A 0.627753 and B -1.863064
bps per bet, hit rate 0.506946 (the board's 0.612 / -1.879 come from a ridge
readout, not the head).

With --reference_npz (the G4 probe's per-decision head predictions) the check
splits in two: the bench's A and B must equal the PnL bench formula on our own
signs exactly (the accounting, mids and fills), and the signs that differ from
nano-sigma's are counted with their size (the signal runs at batch 1 here, at
batch 4 there, so bf16 rounding can flip a prediction near zero).

    python run/benchmarking/policy_bench/check_replay.py --shard_root <v2_20261013 mnt> \
        --signal_manifest <v2>/test.json --sim_manifest <v3>/test.json --add_dir <add index> \
        --ckpt <returnft_lobert_pass1>/ckpt.pt --out <file.json> [--reference_npz <G4 probe preds>]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from lobpipeline.policy_bench import trace as trace_io
from lobpipeline.policy_bench.costs import NO_COST
from lobpipeline.policy_bench.episodes import return_bench_decisions
from lobpipeline.policy_bench.market import BookFill, MidFill, ReplayMarket
from lobpipeline.policy_bench.policies import SignRule
from lobpipeline.policy_bench.rollout import rollout
from lobpipeline.policy_bench.score import score
from lobpipeline.policy_bench.terminal import MarkToMid
from lobpipeline.policy_bench.types import Episode

H_BET = 128
REFERENCE = dict(a_bps_per_bet=0.627753, b_bps_per_bet=-1.863064, hit_rate=0.506946, decisions=70183,
                 source="nano-sigma return_features.py head on the same decisions (job 7842392), "
                        "tools/return_bench.py pnl(), c_exch_bps 0")
W = {}


def _init(ckpt, shard_root, add_dir, history):
    import torch
    from lobpipeline.policy_bench.signal import Signal
    torch.set_num_threads(1)
    W["signal"] = Signal(ckpt, "cuda")
    W["markets"] = {name: ReplayMarket(shard_root, add_dir, fill, history=history, n_messages=H_BET, latency=0)
                    for name, fill in (("mid", MidFill()), ("book", BookFill()))}
    W["policy"] = SignRule(W["signal"].horizons.index(H_BET), size=1)


def _day(arg):
    i, eps = arg
    return i, {name: rollout(W["policy"], W["signal"], m, eps, every=H_BET) for name, m in W["markets"].items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_root", required=True, help="v2_20261013 shards, <root>/<month>/")
    p.add_argument("--signal_manifest", required=True, help="v2 test.json: the decision set")
    p.add_argument("--sim_manifest", required=True, help="v3 test.json (fullbook_dir)")
    p.add_argument("--add_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference_npz", default="", help="G4 probe preds: per-decision comparison")
    p.add_argument("--prefill", type=int, default=3968)
    p.add_argument("--horizons", default="16,64,128,512,1024")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="smoke only")
    a = p.parse_args()

    H = [int(h) for h in a.horizons.split(",")]
    v3 = {(e["month"], e["message"]): e for e in json.load(open(a.sim_manifest))}
    ds = return_bench_decisions(json.load(open(a.signal_manifest)), a.prefill, H)
    if a.limit:
        ds = ds[:a.limit]
    eps = [Episode(v3[(e["month"], e["message"])], s + a.prefill) for e, s in ds]
    groups, cur = [], []
    for ep in eps:                                    # decisions are grouped by day, starts increasing
        if cur and cur[-1].entry["message"] != ep.entry["message"]:
            groups.append(cur)
            cur = []
        cur.append(ep)
    groups += [cur] if cur else []
    order = sorted(range(len(groups)), key=lambda i: -groups[i][-1].start)    # long days first
    t0 = time.time()
    res = {"mid": [None] * len(groups), "book": [None] * len(groups)}
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.ckpt, a.shard_root, a.add_dir, a.prefill)) as pool:
        for k, (i, out) in enumerate(pool.imap_unordered(_day, [(i, groups[i]) for i in order])):
            for name in res:
                res[name][i] = out[name]
            if k % 500 == 0:
                print(f"{k}/{len(groups)} days, {time.time() - t0:.0f} s", flush=True)
    traces = {name: [t for g in v for t in g] for name, v in res.items()}
    out = dict(decisions=len(eps), days=len(groups), seconds=time.time() - t0, reference=REFERENCE,
               a=score(traces["mid"], MarkToMid(), NO_COST), b=score(traces["book"], MarkToMid(), NO_COST))
    sig = np.stack([t.signal[0] for t in traces["mid"]])
    out["signal_equal_between_markets"] = bool(np.array_equal(sig, np.stack([t.signal[0] for t in traces["book"]])))
    if a.reference_npz:
        z = np.load(a.reference_npz)
        j = H.index(H_BET)
        ref = z["kang"][:len(eps), j]
        mine = sig[:, j]
        r = z["r"][:len(eps), j]
        ours_r = np.array([(t.mid[-1] - t.mid[0]) / t.mid[0] * 1e4 for t in traces["mid"]])
        sp = z["spread"][:len(eps)]
        flip = np.sign(mine) != np.sign(ref)

        def pnl_bench(bet):            # tools/return_bench.py pnl(), c_exch_bps 0
            g = np.abs(bet).sum()
            return float((bet * r).sum() / g), float((bet * r - np.abs(bet) * 0.5 * sp).sum() / g)

        a_ours, b_ours = pnl_bench(np.sign(mine))
        out["per_decision"] = dict(
            pred_bit_equal=int((mine == ref).sum()), pred_max_abs_diff=float(np.abs(mine - ref).max()),
            sign_differs=int(flip.sum()),
            sign_differs_max_abs_pred=float(np.abs(np.r_[mine[flip], ref[flip]]).max()) if flip.any() else 0.0,
            return_max_abs_diff=float(np.nanmax(np.abs(ours_r - r))),
            pnl_bench_reference_signs=pnl_bench(np.sign(ref)), pnl_bench_our_signs=(a_ours, b_ours),
            bench_equals_pnl_bench_on_our_signs=bool(out["a"]["bps_per_bet"] == a_ours
                                                     and out["b"]["bps_per_bet"] == b_ours))
    base = os.path.splitext(a.out)[0]
    for name in traces:
        trace_io.save(f"{base}_{name}_traces.npz", traces[name])
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
