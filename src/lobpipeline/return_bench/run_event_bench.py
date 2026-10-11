"""CLI for the event-anchored return bench.

Scores generated LOB sequences on the mid-price return a fixed number of milliseconds
after a real trade, rather than a fixed number of messages after a window boundary.

    python -m lobpipeline.return_bench.run_event_bench \
        --infer_dir <dir with data_cond/ data_gen/ data_real/> \
        --horizons 1,5,10,25,50

Two self-checks are built in rather than left to ad-hoc scripts, because they are the
evidence that the numbers mean anything:

``--identity-check``  feeds the real stream in as if it were generated. IC, Ranked IC
                      and Direction Accuracy must come back at exactly 1.0 / 1.0 / 100%.
                      Anything less means files are mispaired, timestamps are off by a
                      row, or p0 was read before the anchoring trade instead of after.

``--shuffle-null``    re-pairs samples at random to show what the metrics look like
                      when there is provably no signal. Without it there is no way to
                      say whether an observed IC is large.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
import pandas as pd

from .event_anchored import (
    CSV_RESTING_ASK,
    CSV_RESTING_BID,
    EVENT_TYPE_TRADE,
    LOBSTER_COLS,
    clock_diagnostics,
    compute_metrics,
    distribution_comparison,
    load_mid,
    load_times,
    sample_returns,
    magnitude_skill,
    shuffle_null,
    trade_sign_baseline,
)

DEFAULT_HORIZONS_MS = "1,5,10,25,50"

# How exact --identity-check demands the answer be. Named, and asserted by CI, because
# loosening it is the one edit that disables the check while leaving every test green:
# 1.0 satisfies a 0.01 tolerance exactly as well as a 1e-9 one, so nothing goes red and
# the check keeps printing PASSED while no longer telling a correct implementation apart
# from one that is 1% wrong.
IDENTITY_TOL = 1e-9


def find_paired_files(infer_dir: str) -> list[dict]:
    """Pair cond/gen/real by real_id.

    data_cond/ supplies the anchor: its final message is the trade, and its final book
    row is the resulting order book. Note data_cond's book has one more row than its
    message file (inference writes ``b_seq_pv[:, :n_cond+1]``), so ``mid[-1]`` is the
    state *after* the anchoring trade, which is what p0 must be.
    """
    gen_dir = os.path.join(infer_dir, "data_gen")
    real_dir = os.path.join(infer_dir, "data_real")
    cond_dir = os.path.join(infer_dir, "data_cond")

    gen_books = sorted(glob(f"{gen_dir}/*_orderbook_*_gen_id_*.npy"))
    ext = ".npy"
    if not gen_books:
        gen_books = sorted(glob(f"{gen_dir}/*_orderbook_*_gen_id_*.csv"))
        ext = ".csv"

    pairs = []
    for gen_book in gen_books:
        base = os.path.basename(gen_book)
        m = re.search(r"real_id_(\d+)", base)
        if not m:
            continue
        rid = m.group(1)
        m_date = re.search(r"(\d{4}-\d{2}-\d{2})", base)
        date = m_date.group(1) if m_date else ""

        gen_msg = gen_book.replace("_orderbook_", "_message_")
        real_book = glob(f"{real_dir}/*{date}*_orderbook_real_id_{rid}{ext}")
        real_msg = glob(f"{real_dir}/*{date}*_message_real_id_{rid}{ext}")
        cond_book = glob(f"{cond_dir}/*{date}*_orderbook_real_id_{rid}{ext}")
        cond_msg = glob(f"{cond_dir}/*{date}*_message_real_id_{rid}{ext}")

        if not (os.path.exists(gen_msg) and real_book and real_msg and cond_book and cond_msg):
            continue
        pairs.append({
            "real_id": rid, "date": date,
            "gen_book": gen_book, "gen_msg": gen_msg,
            "real_book": real_book[0], "real_msg": real_msg[0],
            "cond_book": cond_book[0], "cond_msg": cond_msg[0],
        })
    return pairs


def _read_anchor(cond_msg: str) -> tuple[float, int, int]:
    """(t0, event_type, direction) from the last conditioning message."""
    if cond_msg.endswith(".npy"):
        msg = np.load(cond_msg)
        # 14-col layout: 1 event_type, 2 direction, 8/9 time. Note this layout carries
        # direction as {0,1}; the csv path below carries {-1,+1}.
        last = msg[-1]
        t0 = float(last[8]) + float(last[9]) * 1e-9
        return t0, int(last[1]), int(last[2])
    df = pd.read_csv(cond_msg, header=None, names=LOBSTER_COLS)
    last = df.iloc[-1]
    return float(last["time"]), int(last["event_type"]), int(last["direction"])


def collect(pairs, horizons, workers=32, identity=False, require_trade=True):
    """Per-horizon paired returns plus anchor bookkeeping."""
    rec = {h: {"pred": [], "real": [], "dir": [], "n_gen": [], "n_real": []} for h in horizons}
    stats_ = {"n_pairs": len(pairs), "n_used": 0, "n_not_trade": 0, "n_bad": 0}
    per_sample_cov = []

    def one(p):
        try:
            t0, etype, direction = _read_anchor(p["cond_msg"])
            if require_trade and etype != EVENT_TYPE_TRADE:
                return ("not_trade", None)
            p0 = load_mid(p["cond_book"])[-1]
            real_t, real_m = load_times(p["real_msg"]), load_mid(p["real_book"])
            if identity:
                gen_t, gen_m = real_t, real_m
            else:
                gen_t, gen_m = load_times(p["gen_msg"]), load_mid(p["gen_book"])
            rets = sample_returns(horizons, t0, p0, gen_t, gen_m, real_t, real_m)
            clocks = {h: clock_diagnostics(gen_t, real_t, t0, h) for h in rets}
            return ("ok", (rets, clocks, direction))
        except Exception:
            return ("bad", None)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tag, payload in pool.map(one, pairs):
            if tag == "not_trade":
                stats_["n_not_trade"] += 1
                continue
            if tag == "bad" or payload is None:
                stats_["n_bad"] += 1
                continue
            rets, clocks, direction = payload
            if not rets:
                stats_["n_bad"] += 1
                continue
            stats_["n_used"] += 1
            per_sample_cov.append(set(rets.keys()))
            for h, (pg, pr) in rets.items():
                rec[h]["pred"].append(pg)
                rec[h]["real"].append(pr)
                rec[h]["dir"].append(direction)
                rec[h]["n_gen"].append(clocks[h]["n_msg_gen"])
                rec[h]["n_real"].append(clocks[h]["n_msg_real"])

    for h in horizons:
        for k in rec[h]:
            rec[h][k] = np.asarray(rec[h][k], dtype=np.float64)
    return rec, stats_, per_sample_cov


def build_table(rec, horizons, n_used) -> pd.DataFrame:
    rows = []
    for h in horizons:
        m = compute_metrics(rec[h]["pred"], rec[h]["real"])
        d = distribution_comparison(rec[h]["pred"], rec[h]["real"])
        n_cov = int(rec[h]["pred"].size)
        base = compute_metrics(trade_sign_baseline(rec[h]["dir"]), rec[h]["real"])
        rows.append({
            "horizon_ms": h,
            **m,
            "baseline_ranked_IC": base["ranked_IC"],
            "baseline_direction_acc": base["direction_acc"],
            **magnitude_skill(rec[h]["pred"], rec[h]["real"]),
            "N_cov": n_cov,
            "cov_pct": 100.0 * n_cov / n_used if n_used else np.nan,
            "gen_msg": float(np.mean(rec[h]["n_gen"])) if n_cov else np.nan,
            "real_msg": float(np.mean(rec[h]["n_real"])) if n_cov else np.nan,
            **d,
        })
    return pd.DataFrame(rows)


def print_table(df, title):
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)
    print(f"{'Horizon':>8} | {'IC':>8} | {'Rank IC':>8} | {'Dir Acc':>8} | {'N_total':>7} | "
          f"{'N_eff':>6} | {'N_cov':>6} | {'cov%':>6} | {'gen#msg':>8} | {'real#msg':>8}")
    print("-" * 108)
    for _, r in df.iterrows():
        acc = f"{r['direction_acc']*100:>7.2f}%" if np.isfinite(r["direction_acc"]) else f"{'n/a':>8}"
        print(f"{int(r['horizon_ms']):>6d}ms | {r['IC']:>8.4f} | {r['ranked_IC']:>8.4f} | {acc} | "
              f"{int(r['N_total']):>7d} | {int(r['N_eff']):>6d} | {int(r['N_cov']):>6d} | "
              f"{r['cov_pct']:>5.1f}% | {r['gen_msg']:>8.1f} | {r['real_msg']:>8.1f}")
    print("=" * 108)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Event-anchored return bench (trade -> N ms mid return)")
    ap.add_argument("--infer_dir", required=True, help="dir containing data_cond/ data_gen/ data_real/")
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS_MS, help="comma-separated milliseconds")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--name", default="")
    ap.add_argument("--identity-check", action="store_true",
                    help="feed real in as generated; IC must be exactly 1.0")
    ap.add_argument("--shuffle-null", action="store_true",
                    help="report metrics under randomly re-paired samples")
    ap.add_argument("--allow-non-trade-anchor", action="store_true",
                    help="do not require the last conditioning message to be a trade")
    args = ap.parse_args(argv)

    horizons = [float(x) for x in args.horizons.split(",")]
    pairs = find_paired_files(args.infer_dir)
    print(f"Inference dir : {args.infer_dir}")
    print(f"Horizons (ms) : {horizons}")
    print(f"Paired seqs   : {len(pairs)}")
    if not pairs:
        print("ERROR: no paired cond/gen/real files found", file=sys.stderr)
        return 1

    rec, st, cov_sets = collect(pairs, horizons, args.workers,
                                identity=args.identity_check,
                                require_trade=not args.allow_non_trade_anchor)
    print(f"Anchored      : {st['n_used']}  (rejected: not-a-trade {st['n_not_trade']}, unreadable {st['n_bad']})")
    if st["n_used"] == 0:
        print("ERROR: no usable anchored samples", file=sys.stderr)
        return 1

    df = build_table(rec, horizons, st["n_used"])
    title = "EVENT-ANCHORED RETURN BENCH"
    if args.identity_check:
        title += "  [IDENTITY CHECK: IC/RankIC must be 1.0, DirAcc 100%]"
    print_table(df, title)

    if args.identity_check:
        bad = [(int(r["horizon_ms"]), r["IC"], r["ranked_IC"], r["direction_acc"])
               for _, r in df.iterrows()
               if not (abs(r["IC"] - 1.0) < IDENTITY_TOL
                       and abs(r["ranked_IC"] - 1.0) < IDENTITY_TOL
                       and abs(r["direction_acc"] - 1.0) < IDENTITY_TOL)]
        if bad:
            print("\nIDENTITY CHECK FAILED at horizons:", bad, file=sys.stderr)
            return 2
        print("\nIDENTITY CHECK PASSED — pairing, time alignment and metrics are self-consistent.")

    # Common subset: only samples covered at every horizon. Curve shapes are comparable
    # across horizons only here; the main table's per-horizon sets differ in membership.
    if len(horizons) > 1 and cov_sets:
        common = set(horizons)
        for s in cov_sets:
            common &= s
        n_common = sum(1 for s in cov_sets if set(horizons) <= s)
        print(f"\nSamples covered at every horizon: {n_common} / {st['n_used']}")

    # The level a model must clear to have contributed anything. Printed right under
    # the main table because an IC read without it is easy to over-interpret: on the
    # SP500 ladder this baseline beats every model, which no confidence interval or
    # noise floor would have shown.
    print("\n--- versus the trade-sign baseline (no model output; the anchoring trade's own side) ---")
    print(f"{'Horizon':>8} | {'model RankIC':>12} | {'baseline':>9} | {'model DirAcc':>12} | "
          f"{'baseline':>9} | {'model clears it?':>16}")
    for _, r in df.iterrows():
        b_ric, b_acc = r["baseline_ranked_IC"], r["baseline_direction_acc"]
        clears = "yes" if (r["ranked_IC"] > b_ric and r["direction_acc"] > b_acc) else "NO"
        acc = f"{r['direction_acc']*100:>11.2f}%" if np.isfinite(r["direction_acc"]) else f"{'n/a':>12}"
        bacc = f"{b_acc*100:>8.2f}%" if np.isfinite(b_acc) else f"{'n/a':>9}"
        print(f"{int(r['horizon_ms']):>6d}ms | {r['ranked_IC']:>12.4f} | {b_ric:>9.4f} | {acc} | "
              f"{bacc} | {clears:>16}")
    print("The baseline predicts sign from the anchoring trade's own side, which is in the")
    print("conditioning window. It emits one magnitude for every sample, so it cannot rank")
    print("magnitude at all -- that column is where a generative model has room:")
    print(f"{'Horizon':>8} | {'magnitude rank corr':>19} | {'n correctly signed':>18}")
    for _, r in df.iterrows():
        mr = f"{r['magnitude_ric']:>19.4f}" if np.isfinite(r["magnitude_ric"]) else f"{'n/a':>19}"
        print(f"{int(r['horizon_ms']):>6d}ms | {mr} | {int(r['n_signed_correct']):>18}")

    print("\n--- distribution: does the generator move the price far enough? ---")
    print(f"{'Horizon':>8} | {'std_gen bp':>10} | {'std_real bp':>11} | {'ratio':>6} | "
          f"{'|med| gen':>9} | {'|med| real':>10} | {'KS':>6} | {'KS p':>7}")
    for _, r in df.iterrows():
        print(f"{int(r['horizon_ms']):>6d}ms | {r['std_gen_bp']:>10.4f} | {r['std_real_bp']:>11.4f} | "
              f"{r['std_ratio']:>6.3f} | {r['abs_median_gen_bp']:>9.4f} | "
              f"{r['abs_median_real_bp']:>10.4f} | {r['ks_stat']:>6.3f} | {r['ks_pvalue']:>7.4f}")

    print("\n--- by trade direction (real side; +1 = trade hit the bid = seller-initiated) ---")
    print(f"{'Horizon':>8} | {'dir':>4} | {'N':>4} | {'mean bp':>9} | {'down%':>7} | {'IC':>8} | {'DirAcc':>8}")
    for h in horizons:
        for dv, lab in ((CSV_RESTING_BID, "+1"), (CSV_RESTING_ASK, "-1")):
            m = rec[h]["dir"] == dv
            if m.sum() < 3:
                continue
            rr, pp = rec[h]["real"][m], rec[h]["pred"][m]
            nz = np.abs(rr) > 1e-8
            down = 100.0 * np.mean(rr[nz] < 0) if nz.sum() else np.nan
            mm = compute_metrics(pp, rr)
            acc = f"{mm['direction_acc']*100:>7.2f}%" if np.isfinite(mm["direction_acc"]) else f"{'n/a':>8}"
            print(f"{int(h):>6d}ms | {lab:>4} | {int(m.sum()):>4d} | {rr.mean()*1e4:>+9.3f} | "
                  f"{down:>6.1f}% | {mm['IC']:>8.4f} | {acc}")

    if args.shuffle_null:
        print("\n--- shuffle null (what the metrics look like with no signal) ---")
        print(f"{'Horizon':>8} | {'IC mean':>8} | {'IC sd':>7} | {'DirAcc mean':>11} | {'2sd band':>10}")
        for h in horizons:
            nl = shuffle_null(rec[h]["pred"], rec[h]["real"])
            if not np.isfinite(nl["IC_sd"]):
                continue
            print(f"{int(h):>6d}ms | {nl['IC_mean']:>+8.4f} | {nl['IC_sd']:>7.4f} | "
                  f"{nl['dir_acc_mean']*100:>10.2f}% | +/-{2*nl['IC_sd']:>7.4f}")
        print("Direction Accuracy under the null is not 50%: realized returns are not "
              "sign-balanced, so compare against the column above, not against 0.5.")

    out_dir = args.output_dir or os.path.join(args.infer_dir, "return_bench_event")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{args.name}" if args.name else ""
    if args.identity_check:
        suffix += "_identity"
    csv_path = os.path.join(out_dir, f"return_bench_event{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
