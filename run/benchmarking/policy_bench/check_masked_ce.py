#!/usr/bin/env python3
"""How much of the simulator's probability lies on illegal tokens, on real data (one GPU).

For the first --n_episodes of the config's episode set: the simulator reads
the SimMarket context (the last CONTEXT_MESSAGES - n_messages - AGENT_MARGIN
real messages before the start) and then the next --n_messages real
messages, teacher forced. At every token of those messages, with our book
before the message:

    ce          -log p(real token)
    masked ce   -log p(real token) / p(legal tokens), the masks of sampling.legal_tokens
    legal mass  p(legal tokens), the share of probability the masks keep

The real token must always be legal (a mask that refuses a real message
would be too strict); the count of exceptions is reported. Results by event
and field. CE per message is the sum over a message's tokens (Kang reports
10.95 nats per message for newtok_a_r1 on whole test windows).

    python run/benchmarking/policy_bench/check_masked_ce.py --config configs/bench/policy_bench_v1.yaml \
        --shard_root <mount> --out <file.json> [--n_episodes 96] [--n_messages 512]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

from lobpipeline.policy_bench.codec import burst_bounds, encode_stream, mid_floor
from lobpipeline.policy_bench.episodes import load_episodes
from lobpipeline.policy_bench.history import open_day, start_state
from lobpipeline.policy_bench.nanosigma.data import EVT0, START, book_features
from lobpipeline.policy_bench.sampling import MessageState, file_end_ns, legal_tokens
from lobpipeline.policy_bench.sim_market import AGENT_MARGIN, CONTEXT_MESSAGES, MAX_TOKENS_PER_MESSAGE
from lobpipeline.policy_bench.simulator import KVSimulator, load_newtok

EVENTS = {1: "new", 2: "cancel", 3: "delete", 4: "execution"}


def episode(sim, dev, shard_root, add_dir, ep, context, n_msg, acc, per_msg):
    day, index = open_day(shard_root, add_dir, ep.entry)
    end = file_end_ns(ep.entry["message"])
    # rows: `need` real rows after the start, enough for n_msg messages
    need = min(2 * n_msg, day.n_msgs - ep.start - 1)
    H = 3968
    _, log, _, _, slots = start_state(day, index, ep.start + need, H + need, 0)
    rows, l10 = log.rows(H + need), log.l10(H + need + 1)
    bounds = burst_bounds(rows)
    m0 = next(k for k, (i, _) in enumerate(bounds) if i >= H)          # first message after the start
    m1 = min(m0 + n_msg, len(bounds) - 1)                                # whole messages only
    c0 = max(0, m0 - context)
    r0, r1 = bounds[c0][0], bounds[m1][0]
    tok, mot, first = encode_stream(rows[r0:r1], l10[r0:r1], slots[r0:r1])
    bvec = book_features(l10[r0:r1][first])[mot]
    inp = np.concatenate([[START], tok[:-1]])
    sim.truncate(0)
    logits = sim.feed(torch.as_tensor(inp, device=dev), torch.as_tensor(bvec, device=dev))
    k0 = int(np.searchsorted(mot, m0 - c0))
    logp = torch.log_softmax(logits[k0:].double(), -1).cpu().numpy()   # positions of the real messages
    # our book before each real message: replay from the start
    book = day.book_before(ep.start)
    it = day.replay(book, ep.start, ep.start + (r1 - H))
    for _ in range(bounds[m0][0] - H):                                   # a burst cut by the start
        next(it)
    for m in range(m0, m1):
        i, j = bounds[m]
        lo = int(np.searchsorted(mot, m - c0)) - k0
        hi = int(np.searchsorted(mot, m - c0, "right")) - k0
        st = MessageState(int(rows[i - 1, 8]) * 10**9 + int(rows[i - 1, 9]))
        mid = mid_floor(int(l10[i, 3]), int(l10[i, 5]))
        ev = EVENTS.get(int(tok[k0 + lo]) - EVT0 + 1, "bad")
        ce_msg = mce_msg = 0.0
        for pos in range(lo, hi):
            t = int(tok[k0 + pos])
            if st.done():                                                 # longer than the grammar allows
                acc[(ev, "beyond_grammar")]["n"] += 1
                break
            kind = st.kind()
            ok = legal_tokens(st, book, mid, end)
            lp = logp[pos]
            legal = float(np.exp(lp[ok]).sum()) if len(ok) else 0.0
            v = acc[(ev, kind)]
            v["n"] += 1
            v["ce"] += -lp[t]
            v["legal_mass"] += legal
            if t in set(ok.tolist()):
                v["masked_ce"] += -lp[t] + np.log(legal)
                mce_msg += -lp[t] + np.log(legal)
            else:
                v["real_illegal"] += 1
                mce_msg += -lp[t]
            ce_msg += -lp[t]
            st.push(t)
        per_msg[ev].append((ce_msg, mce_msg))
        for _ in range(i, j):                                             # the message's rows
            next(it)
    return m1 - m0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shard_root", required=True)
    p.add_argument("--n_episodes", type=int, default=96)
    p.add_argument("--n_messages", type=int, default=512)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    entries = [e for e in json.load(open(cfg["manifest"])) if e["month"] == cfg["month"]]
    eps = load_episodes(cfg["episodes"], entries)[:a.n_episodes]
    dev = torch.device("cuda")
    sim = KVSimulator(load_newtok(cfg["simulator"], dev), CONTEXT_MESSAGES * MAX_TOKENS_PER_MESSAGE + 8192)
    context = CONTEXT_MESSAGES - cfg["n_messages"] - AGENT_MARGIN
    acc = defaultdict(lambda: defaultdict(float))
    per_msg = defaultdict(list)
    t0, n = time.time(), 0
    with torch.no_grad():
        for e, ep in enumerate(eps):
            n += episode(sim, dev, a.shard_root, cfg["add_dir"], ep, context, a.n_messages, acc, per_msg)
            print(f"{e + 1}/{len(eps)} episodes, {n} messages, {time.time() - t0:.0f} s", flush=True)

    fields = {}
    for (ev, kind), v in sorted(acc.items()):
        n_ok = v["n"] - v["real_illegal"]
        fields[f"{ev}/{kind}"] = dict(n=int(v["n"]), ce=v["ce"] / v["n"],
                                      masked_ce=v["masked_ce"] / n_ok if n_ok else None,
                                      legal_mass=v["legal_mass"] / v["n"], real_illegal=int(v["real_illegal"]))
    events = {}
    allm = []
    for ev, x in per_msg.items():
        x = np.array(x)
        allm.append(x)
        events[ev] = dict(messages=len(x), share=0.0, ce_per_msg=float(x[:, 0].mean()),
                          masked_ce_per_msg=float(x[:, 1].mean()))
    allm = np.concatenate(allm)
    for ev in events:
        events[ev]["share"] = events[ev]["messages"] / len(allm)
    res = dict(config=a.config, episodes=len(eps), messages=int(len(allm)), seconds=time.time() - t0,
               torch=torch.__version__, ce_per_msg=float(allm[:, 0].mean()), masked_ce_per_msg=float(allm[:, 1].mean()),
               real_illegal=int(sum(f["real_illegal"] for f in fields.values())), events=events, fields=fields)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("messages", "ce_per_msg", "masked_ce_per_msg", "real_illegal")}))
    for ev, v in sorted(events.items()):
        print(ev, json.dumps(v))
    for k, v in fields.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
