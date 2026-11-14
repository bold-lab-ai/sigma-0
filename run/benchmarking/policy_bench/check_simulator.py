#!/usr/bin/env python3
"""G3 of the policy bench: the newtok simulator (one GPU).

A. Reference. nano-sigma's own input pipeline (read_window_newtok, collate,
   shift) and the vendored model on newtok_a_r1's 939 test windows, batched
   as its final eval was (8 ranks, 2 windows per batch). The test CE must
   equal the one in the run's metrics.jsonl.
B. Our inputs. For the first --n_ours windows, the bench's own path (book
   from the nearest snapshot, replay, L10Writer rows, our queue slots,
   encode_stream) must give byte-identical tokens, message index and book
   vectors.
C. KV cache. For the first --n_kv windows, KVSimulator fed in chunks and one
   token at a time against the full forward: logit difference, CE of both,
   argmax agreement, and the time of one decoding step.
D. Batched decoding. --n_batch windows in one BatchKVSimulator, prefilled to
   different lengths, then stepped together one token per row: each row's
   logits against a one-row KVSimulator fed the same tokens (both
   precisions; C checks that one against the full forward), and the time of one
   step for batch sizes --batch_sizes at a full-window context.

    python run/benchmarking/policy_bench/check_simulator.py --shard_root <mnt> \
        --manifest <v3>/test.json --ckpt <newtok_a_r1>/ckpt.pt \
        --test_subset <run dir>/test_subset.json --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from lobpipeline.policy_bench.book import L10Writer
from lobpipeline.policy_bench.codec import encode_stream
from lobpipeline.policy_bench.daydata import DayFiles
from lobpipeline.policy_bench.nanosigma.data import MASK, START, book_features
from lobpipeline.policy_bench.nanosigma.newtok import (
    MAX_TOK_PER_ROW, PAD, NewTokCache, collate, read_window_newtok,
)
from lobpipeline.policy_bench.simulator import BatchKVSimulator, KVSimulator, load_newtok

KANG_WORLD, KANG_MICRO = 8, 2      # newtok_a_r1 config.json: world 8, eval_micro 2


def shift(tokens: torch.Tensor) -> torch.Tensor:
    """train_newtok.shift: START first, PAD inputs become MASK."""
    x = torch.cat([torch.full_like(tokens[:, :1], START), tokens[:, :-1]], dim=1)
    return torch.where(x < 0, torch.full_like(x, MASK), x)


@torch.no_grad()
def part_a(model, windows, root, dev):
    cache = NewTokCache(root, max_open=256)
    tot = np.zeros(3)
    t0 = time.time()
    for rank in range(KANG_WORLD):
        mine = windows[rank::KANG_WORLD]
        for i in range(0, len(mine), KANG_MICRO):
            chunk = [read_window_newtok(cache, e, s, n) for e, s, n in mine[i:i + KANG_MICRO]]
            tok, mi, bk, rows = collate(chunk, MAX_TOK_PER_ROW * chunk[0][3])
            tok = torch.from_numpy(tok).long().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(shift(tok), torch.from_numpy(mi).to(dev), torch.from_numpy(bk).to(dev))
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tok.view(-1),
                                   ignore_index=PAD, reduction="sum")
            tot += [loss.double().item(), (tok != PAD).sum().item(), float(rows.sum())]
        print(f"A rank {rank} done {time.time() - t0:.0f} s", flush=True)
    return dict(test_ce=tot[0] / tot[1], test_ce_per_msg=tot[0] / tot[2], tokens=int(tot[1]),
                seconds=time.time() - t0)


def our_inputs(day: DayFiles, s: int, n: int):
    """Tokens, message index and book vectors of messages [s, s + n), from our book."""
    bk = day.book_before(s)
    l10 = np.zeros((n, 43), np.int64)
    l10[0] = day.l10[s]
    w = L10Writer(l10[0])
    slots = np.zeros(n, np.int64)
    msg = np.asarray(day.messages[s:s + n])
    for r, slot in day.replay(bk, s, s + n):
        k = r - s
        slots[k] = slot
        if k + 1 < n:
            l10[k + 1] = w.next_row(bk, int(msg[k, 8]), int(msg[k, 9]))
    tok, mot, first = encode_stream(msg, l10, slots)
    return tok, mot, book_features(l10[first]), l10


def part_b(windows, root, n):
    cache = NewTokCache(root, max_open=64)
    out = []
    for e, s, L in windows[:n]:
        t, m, b, _ = read_window_newtok(cache, e, s, L)
        day = DayFiles.open(root, e)
        t2, m2, b2, l10 = our_inputs(day, s, L)
        out.append(dict(message=e["message"], start=s, tokens=int(len(t)),
                        tokens_equal=bool(np.array_equal(t, t2)), msg_idx_equal=bool(np.array_equal(m, m2)),
                        book_equal=bool(np.array_equal(b, b2)),
                        l10_rows_bad=int((np.asarray(day.l10[s:s + L]) != l10).any(1).sum())))
        print("B", json.dumps(out[-1]), flush=True)
    return out


@torch.no_grad()
def part_c(model, windows, root, n, dev, n_steps=256):
    """bf16 (the eval's precision) and float32, each against the full forward in the same precision."""
    cache = NewTokCache(root, max_open=16)
    out = []
    for e, s, L in windows[:n]:
        t, m, b, _ = read_window_newtok(cache, e, s, L)
        tok = torch.from_numpy(t).long().to(dev)
        mi = torch.from_numpy(m).long().to(dev)
        book = torch.from_numpy(b).to(dev)
        inp = shift(tok[None])[0]
        T, h = len(t), len(t) // 2
        for dtype in (torch.bfloat16, torch.float32):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                full = model(inp[None], mi[None], book[None])[0].float()
            sim = KVSimulator(model, max_len=T, dtype=dtype)
            parts = [sim.feed(inp[:h], book[mi[:h]])]
            torch.cuda.synchronize()
            t0 = time.time()
            for k in range(h, h + n_steps):
                parts.append(sim.feed(inp[k:k + 1], book[mi[k:k + 1]]))
            torch.cuda.synchronize()
            step_ms = (time.time() - t0) / n_steps * 1e3
            for k in range(h + n_steps, T, 4096):
                parts.append(sim.feed(inp[k:k + 4096], book[mi[k:k + 4096]]))
            kv = torch.cat(parts)
            d = (kv - full).abs()
            out.append(dict(message=e["message"], start=s, tokens=T, dtype=str(dtype).split(".")[-1],
                            max_abs_logit_diff=d.max().item(), mean_abs_logit_diff=d.mean().item(),
                            ce_full=F.cross_entropy(full, tok).item(), ce_kv=F.cross_entropy(kv, tok).item(),
                            argmax_agree=(kv.argmax(-1) == full.argmax(-1)).float().mean().item(),
                            step_ms=step_ms))
            print("C", json.dumps(out[-1]), flush=True)
            del full, kv, parts, sim
    return out


def part_d(model, windows, root, n, sizes, dev, n_steps=128):
    cache = NewTokCache(root, max_open=16)
    rows = []
    for e, s, L in windows[:n]:
        t, m, b, _ = read_window_newtok(cache, e, s, L)
        tok = torch.from_numpy(t).long().to(dev)
        rows.append((tok, shift(tok[None])[0], torch.from_numpy(b).to(dev)[torch.from_numpy(m).long().to(dev)]))
    T = max(len(r[0]) for r in rows)
    out = dict(equal=[], timing=[])
    for dtype in (torch.bfloat16, torch.float32):
        starts = [len(r[0]) // 2 + 97 * k for k, r in enumerate(rows)]       # rows at different lengths
        sim = BatchKVSimulator(model, len(rows), T, dtype)
        for k, (_, inp, bk) in enumerate(rows):
            sim.prefill(k, inp[:starts[k]], bk[:starts[k]])
        got = []
        for j in range(n_steps):
            idx = [st + j for st in starts]
            got.append(sim.step(torch.stack([r[1][i] for r, i in zip(rows, idx)]),
                                torch.stack([r[2][i] for r, i in zip(rows, idx)])))
        got = torch.stack(got, 1)                                            # (rows, n_steps, V)
        del sim
        for k, (tok, inp, bk) in enumerate(rows):
            ref = KVSimulator(model, max_len=T, dtype=dtype)
            ref.feed(inp[:starts[k]], bk[:starts[k]])
            one = torch.cat([ref.feed(inp[i:i + 1], bk[i:i + 1]) for i in range(starts[k], starts[k] + n_steps)])
            del ref
            y = tok[starts[k]:starts[k] + n_steps]
            d = (got[k] - one).abs()
            out["equal"].append(dict(row=k, start=starts[k], dtype=str(dtype).split(".")[-1],
                                     max_abs_logit_diff=d.max().item(),
                                     ce_batch=F.cross_entropy(got[k], y).item(), ce_one=F.cross_entropy(one, y).item(),
                                     argmax_agree=(got[k].argmax(-1) == one.argmax(-1)).float().mean().item()))
            print("D", json.dumps(out["equal"][-1]), flush=True)
    _, inp, bk = rows[0]
    ctx = len(inp) - 64
    for B in sizes:
        sim = None
        try:
            sim = BatchKVSimulator(model, B, len(inp), torch.bfloat16)
            for k in range(B):
                sim.prefill(k, inp[:ctx], bk[:ctx])
            tok, book = inp[ctx:ctx + 1].expand(B), bk[ctx:ctx + 1].expand(B, -1)
            sim.step(tok, book)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(32):
                sim.step(tok, book)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / 32 * 1e3
            out["timing"].append(dict(batch=B, context_tokens=ctx, step_ms=ms, tokens_per_s=B / ms * 1e3,
                                      max_mem_gb=torch.cuda.max_memory_allocated() / 1e9))
        except torch.cuda.OutOfMemoryError:
            out["timing"].append(dict(batch=B, context_tokens=ctx, out_of_memory=True))
        print("D", json.dumps(out["timing"][-1]), flush=True)
        del sim
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_root", required=True)
    p.add_argument("--manifest", required=True, help="v3 test.json (has fullbook_dir)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_subset", required=True, help="the run's test_subset.json")
    p.add_argument("--metrics", default="", help="the run's metrics.jsonl, for the reference CE")
    p.add_argument("--out", required=True)
    p.add_argument("--n_ours", type=int, default=64)
    p.add_argument("--n_kv", type=int, default=8)
    p.add_argument("--n_batch", type=int, default=4)
    p.add_argument("--batch_sizes", default="1,8,16,32,48")
    p.add_argument("--skip_a", action="store_true")
    p.add_argument("--only_d", action="store_true")
    a = p.parse_args()

    by_key = {(e["month"], e["message"]): e for e in json.load(open(a.manifest))}
    windows = [(by_key[(mo, msg)], int(s), int(n)) for mo, msg, s, n in json.load(open(a.test_subset))]
    dev = torch.device("cuda")
    model = load_newtok(a.ckpt, dev)
    res = dict(windows=len(windows), torch=torch.__version__)
    if a.metrics:
        final = [json.loads(x) for x in open(a.metrics) if '"final"' in x][-1]
        res["reference"] = dict(test_ce=final["test_ce"], test_ce_per_msg=final["test_ce_per_msg"])
    if a.only_d:
        res["d"] = part_d(model, windows, a.shard_root, a.n_batch, [int(x) for x in a.batch_sizes.split(",")], dev)
        json.dump(res, open(a.out, "w"), indent=1)
        return
    res["b"] = part_b(windows, a.shard_root, a.n_ours)
    res["c"] = part_c(model, windows, a.shard_root, a.n_kv, dev)
    res["d"] = part_d(model, windows, a.shard_root, a.n_batch, [int(x) for x in a.batch_sizes.split(",")], dev)
    if not a.skip_a:
        res["a"] = part_a(model, windows, a.shard_root, dev)
    res["summary"] = dict(
        a_test_ce=res.get("a", {}).get("test_ce"), ref_test_ce=res.get("reference", {}).get("test_ce"),
        b_windows=len(res["b"]), b_all_equal=all(x["tokens_equal"] and x["msg_idx_equal"] and x["book_equal"]
                                                for x in res["b"]),
        b_l10_rows_bad=sum(x["l10_rows_bad"] for x in res["b"]),
        c={dt: dict(max_abs_logit_diff=max(x["max_abs_logit_diff"] for x in cs),
                    max_ce_diff=max(abs(x["ce_kv"] - x["ce_full"]) for x in cs),
                    min_argmax_agree=min(x["argmax_agree"] for x in cs),
                    step_ms=float(np.median([x["step_ms"] for x in cs])))
           for dt in ("bfloat16", "float32") for cs in [[x for x in res["c"] if x["dtype"] == dt]]})
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
