#!/usr/bin/env python3
"""Build wide-book (L500) window-start snapshot npz files for simulator init.

PROVENANCE
    Genericized (2026-07-15) from the EGGROLL-GAN builder
    /home/u6gb/alexbismuth.u6gb/AlphaTrade/experiments/exp_0626_EGGROLL_GAN/
    tools/data_build/make_snapshots.py. ALL data logic (alignment keys, window
    matching, offset verification, empty-level normalization, gates, npz
    schema) is copied verbatim — only the experiment-specific path wiring was
    replaced by CLI arguments plus an optional multi-day batch mode, and the
    era-aware price-align floor from the unseen100 sbatch driver was folded in
    (see effective_min_align).

NPZ SCHEMA (exact schema of the historical lob_preproc_linf/GOOG files,
consumed by LOBSTER_Dataset via wide_book_files — lobster_dataloader.py):
    snapshots           int32  (n_win, 4*levels)   L-level book state BEFORE
                                                   the window-start message;
                                                   row layout = LOBSTER L2
                                                   [ask_p, ask_v, bid_p, bid_v]
                                                   per level, empty level (0,0)
    local_seq_indices   int64  (n_win,)            window index idx (win // n_messages)
    target_market_rows  int64  (n_win,)            idx * n_messages
    n_levels            int64  scalar
    date, ticker        str    scalars
File name: <TICKER>_<date>_L<levels>_linf_snapshots_<n_win>.npz
(the 'linf' token is kept for loader-glob compatibility).

INPUTS (per day)
    --canon      canonical preprocessed day message npy (the file the eval
                 dataloader reads, post-DataCleaner + tok.preproc), .npy[.zst]
    --canon_book canonical L10 orderbook npy[.zst] (verify gate / align check)
    --m0         raw m0 message csv.zst (recon lN_engine stage-1 replay)
    --ob         cpp-engine L500 orderbook csv.zst (same stage-1 replay)

ALIGNMENT
    Canonical row R is matched to its raw m0 row r by exact content key
    (time_s, time_ns, order_id, event_type, size, price_abs) — the canonical
    npy keeps all six (cols 8,9,0,1,5,3) with deterministic int casts
    replicated here. time_ns is matched nearest within +/-2 ns (the canonical
    preproc's float64 s->ns conversion is off by +1 ns on ~half the rows);
    a unique winner is required, ambiguous windows are dropped. The snapshot
    for window start R is ob500 csv row (r + offset); offset is NOT assumed:
    `verify` mode gates both candidate offsets {-1, 0} against the canonical
    L10 book over ALL windows of a day and requires a clear winner.

GATES (generate mode — a failing day writes nothing and exits nonzero)
    coverage  >= --min_coverage (default 0.90): fraction of windows matched in m0
    crossed   <= --max_crossed  (default 0.02): fraction of books with best
                ask <= best bid (structural validity; catches broken offsets)
    price-align >= effective min_align: best ask/bid PRICE agreement vs the
                cleaned canonical L10 book. This is a tripwire for offset /
                stream misalignment, NOT a quality bar (raw vs cleaned volumes
                legitimately differ). Era-aware default — see effective_min_align.

WINDOWING
    Windows are non-overlapping: window length == stride == --n_messages
    (default 1000). That is the only windowing the source builder supports.

BATCH MODE (generate only)
    Pass --dates d1,d2,... or --days-json FILE (JSON list of YYYY-MM-DD, or an
    object with a "days" key) together with --canonical-npy-dir and
    --l500-csv-dir instead of --canon/--canon_book/--m0/--ob. Per-day files are
    resolved with ONE os.scandir per directory (cached) and substring matching
    — no recursive globs, no per-file stat storm (Lustre-safe). --offset must
    come from a prior `verify` run (typically -1 for the GOOG-era replays).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ZSTD = '/usr/bin/zstd'
SENTINEL = 9_999_999_999
# time_ns is deliberately NOT part of the exact key: the canonical preproc's
# float64 seconds->ns conversion is off by +1 ns vs a fresh pandas parse on
# ~half the rows. Exact 5-key match + nearest-time_ns within +/-2 ns, unique
# winner required.
KEY = ['time_s', 'time_ns', 'order_id', 'event_type', 'size', 'price_abs']
KEY5 = ['time_s', 'order_id', 'event_type', 'size', 'price_abs']
NS_TOL = 2
# canonical msg npy column order (encoding.py Message_Tokenizer.preproc):
# 0 order_id, 1 event_type, 2 direction, 3 price_abs, 4 price, 5 size,
# 6 delta_t_s, 7 delta_t_ns, 8 time_s, 9 time_ns, then *_ref features.
CANON_KEY_COLS = [8, 9, 0, 1, 5, 3]

DEFAULT_MIN_ALIGN = 0.10


def log(*a):
    print('[make_snapshots]', *a, flush=True)


def load_canon(path, scratch):
    if path.endswith('.zst'):
        out = os.path.join(scratch, os.path.basename(path)[:-len('.zst')])
        subprocess.run([ZSTD, '-d', '-q', '-f', path, '-o', out], check=True)
        path = out
    return np.load(path, mmap_mode='r')


def parse_m0(m0_zst, scratch):
    """Raw m0 csv -> DataFrame with the same int casts as preproc_patched/encoding."""
    csv = os.path.join(scratch, os.path.basename(m0_zst)[:-len('.zst')])
    subprocess.run([ZSTD, '-d', '-q', '-f', m0_zst, '-o', csv], check=True)
    # cpp m0 csv has 7 fields per row: the LOBSTER 6 + a trailing 'null'
    # column. Name all 7 so there is no column-count ambiguity / ParserWarning.
    m = pd.read_csv(csv, header=None, index_col=False,
                    names=['time', 'event_type', 'order_id', 'size', 'price',
                           'direction', 'aux_null'])
    os.remove(csv)
    df = pd.DataFrame({
        'time_s': m.time.astype(int),
        'time_ns': ((m.time % 1) * 1_000_000_000).astype(int),
        # uint32 space: the canonical preproc wraps IDs >= 2^31 to negative
        # int32 (high-volume days, end-of-day); raw IDs are correct positive.
        'order_id': m.order_id.astype(np.int64) & 0xFFFFFFFF,
        'event_type': m.event_type.astype(np.int64),
        'size': m['size'].clip(upper=9999).astype(np.int64),
        'price_abs': m.price.astype(np.int64),
        'raw_row': np.arange(len(m), dtype=np.int64),
    })
    return df


def match_windows(canon, win_starts, raw_df):
    """Return (win_starts_kept, raw_rows): exact 5-key match, nearest time_ns
    within +/-NS_TOL ns, unique winner required (ambiguous windows dropped)."""
    c = np.asarray(canon[win_starts][:, CANON_KEY_COLS], dtype=np.float64)
    cdf = pd.DataFrame(c.astype(np.int64), columns=KEY)
    # un-wrap int32-overflowed canonical order_ids into the same uint32 space
    # as the raw side (see parse_m0) so the 5-key match holds end-of-day.
    cdf['order_id'] = cdf['order_id'].to_numpy() & 0xFFFFFFFF
    cdf['win'] = np.asarray(win_starts, dtype=np.int64)
    cand = cdf.merge(raw_df.rename(columns={'time_ns': 'time_ns_raw'}),
                     on=KEY5, how='inner')
    cand['dtns'] = (cand.time_ns - cand.time_ns_raw).abs()
    cand = cand.loc[cand.dtns <= NS_TOL]
    # ambiguous = several distinct raw rows tied at the minimal ns distance
    min_d = cand.groupby('win').dtns.transform('min')
    at_min = cand.loc[cand.dtns == min_d]
    n_raw = at_min.groupby('win').raw_row.nunique()
    ambiguous = set(n_raw[n_raw > 1].index)
    best = (at_min.loc[~at_min.win.isin(ambiguous)]
            .drop_duplicates('win', keep='first')
            .sort_values('win'))
    n_miss = len(win_starts) - len(best)
    if n_miss:
        log(f'  WARN {n_miss}/{len(win_starts)} windows dropped '
            f'(no 5-key+/-{NS_TOL}ns match: '
            f'{len(win_starts) - len(best) - len(ambiguous)}, '
            f'ambiguous: {len(ambiguous)}) — loader will L10-fallback')
    return best['win'].to_numpy(np.int64), best['raw_row'].to_numpy(np.int64)


def collect_ob_rows(ob_zst, rows, levels):
    """Single sequential pass over the zstd stream; grab the requested rows."""
    width = 4 * levels
    want = sorted({int(r) for r in rows if r >= 0})
    res = {}
    proc = subprocess.Popen([ZSTD, '-dc', '-q', ob_zst],
                            stdout=subprocess.PIPE, bufsize=8 * 1024 * 1024)
    it = iter(want)
    nxt = next(it, None)
    for i, line in enumerate(proc.stdout):
        if nxt is None:
            break
        if i == nxt:
            a = np.array(line.rstrip(b'\n').split(b','), dtype=np.int64)
            if a.shape[0] < width:
                raise RuntimeError(f'ob row {i}: {a.shape[0]} cols < {width}')
            res[i] = a[:width]
            nxt = next(it, None)
    proc.stdout.close()
    proc.kill()
    proc.wait()
    if nxt is not None:
        raise RuntimeError(f'ob stream ended before row {nxt}')
    return res


def normalize(a):
    """LOBSTER empty-level sentinels (+/-9999999999, 0) -> (0, 0); int32."""
    pv = a.reshape(-1, 2).copy()
    pv[np.abs(pv[:, 0]) >= SENTINEL] = 0
    out = pv.reshape(-1)
    assert np.all(np.abs(out) < 2**31), 'value overflows int32 after normalization'
    return out.astype(np.int32)


def snapshots_for(ob_rows_map, raw_rows, offset, levels):
    n = len(raw_rows)
    snaps = np.zeros((n, 4 * levels), dtype=np.int32)
    ok = np.ones(n, dtype=bool)
    for j, r in enumerate(raw_rows):
        rr = int(r) + offset
        if rr < 0 or rr not in ob_rows_map:
            ok[j] = False
            continue
        snaps[j] = normalize(ob_rows_map[rr])
    return snaps, ok


CANON_COLS = ['order_id', 'event_type', 'direction', 'price_abs', 'price_rel',
              'size', 'dt_s', 'dt_ns', 'time_s', 'time_ns',
              'price_ref', 'size_ref', 'time_s_ref', 'time_ns_ref']


def diag_unmatched(canon, win_starts, matched_wins, raw_df, k=8):
    """Dump unmatched window-start rows next to their nearest raw candidates."""
    matched = set(int(w) for w in matched_wins)
    bad = [int(w) for w in win_starts if int(w) not in matched][:k]
    for w in bad:
        row = np.asarray(canon[w], dtype=np.float64)
        ncol = min(len(CANON_COLS), row.shape[0])
        log(f'--- unmatched canon row {w}:')
        log('    ' + '  '.join(f'{CANON_COLS[i]}={row[i]:.0f}' for i in range(ncol)))
        oid, et = int(row[0]), int(row[1])
        cand = raw_df[(raw_df.order_id == oid) & (raw_df.event_type == et)]
        for _, r in cand.head(3).iterrows():
            log(f'    raw[oid+etype] row {int(r.raw_row)}: time_s={r.time_s} '
                f'time_ns={r.time_ns} size={r["size"]} price_abs={r.price_abs}')
        ts, tns = int(row[8]), int(row[9])
        cand2 = raw_df[(raw_df.time_s == ts) & ((raw_df.time_ns - tns).abs() <= 2)]
        for _, r in cand2.head(4).iterrows():
            log(f'    raw[time±2ns] row {int(r.raw_row)}: oid={r.order_id} '
                f'etype={r.event_type} size={r["size"]} price_abs={r.price_abs} '
                f'time_ns={r.time_ns}')
    if len(matched):
        w = int(sorted(matched)[0])
        row = np.asarray(canon[w], dtype=np.float64)
        ncol = min(len(CANON_COLS), row.shape[0])
        log(f'--- (for contrast) MATCHED canon row {w}:')
        log('    ' + '  '.join(f'{CANON_COLS[i]}={row[i]:.0f}' for i in range(ncol)))


def book_gate(snaps, ok, canon_book, win_starts, levels):
    """Fraction of windows whose top-10 levels equal the canonical L10 book
    row (the recon chain) at the same window start — an independent
    dual-engine ground truth, row-aligned by construction (book[R] = state
    before message R, same convention as the snapshot)."""
    w = min(40, 4 * levels)
    n_eq = n_tot = 0
    for j, R in enumerate(win_starts):
        if not ok[j] or R >= canon_book.shape[0]:
            continue
        ref_row = np.asarray(canon_book[int(R), 3:3 + w], dtype=np.int64)
        ref_row = normalize(ref_row).astype(np.int64)
        n_tot += 1
        if np.array_equal(snaps[j][:w].astype(np.int64), ref_row):
            n_eq += 1
    return (n_eq / n_tot if n_tot else 0.0), n_eq, n_tot


def crossed_fraction(snaps, ok):
    """Intrinsic structural check (no reference needed): fraction of kept
    snapshots whose best ask price <= best bid price — a crossed/invalid book.
    Real cpp L2 books are essentially never crossed; a high rate means a broken
    offset or a corrupt source. (The official linf failed exactly this.)"""
    n_bad = n_tot = 0
    for j in range(len(snaps)):
        if not ok[j]:
            continue
        a, b = int(snaps[j][0]), int(snaps[j][2])   # best ask price, best bid price
        if a == 0 or b == 0:
            continue
        n_tot += 1
        if a <= b:
            n_bad += 1
    return (n_bad / n_tot if n_tot else 0.0), n_tot


def book_align(snaps, ok, canon_book, win_starts, levels):
    """Alignment vs the canonical cleaned L10 book. This is NOT a quality gate:
    raw-stream and cleaned-stream books legitimately differ in *volume*, so we
    only check that best ask/bid PRICES agree — which confirms the ob-row offset
    is correct. Returns (price_frac, exact_frac, n)."""
    w = min(40, 4 * levels)
    n_px = n_exact = n_tot = 0
    for j, R in enumerate(win_starts):
        if not ok[j] or R >= canon_book.shape[0]:
            continue
        ref = normalize(np.asarray(canon_book[int(R), 3:3 + w], dtype=np.int64)).astype(np.int64)
        snap = snaps[j][:w].astype(np.int64)
        n_tot += 1
        if snap[0] == ref[0] and snap[2] == ref[2]:   # best ask & bid PRICES only
            n_px += 1
        if np.array_equal(snap, ref):
            n_exact += 1
    return (n_px / n_tot if n_tot else 0.0), (n_exact / n_tot if n_tot else 0.0), n_tot


def effective_min_align(args):
    """Resolve the price-align floor for the current --date.

    Era-aware price-align floor — preserved from the EGGROLL unseen100 driver
    (tools/data_build/unseen100/_run_unseen100_snapshots.sbatch), original
    comment verbatim:

      # Era-aware price-align floor (probe j5488941, 2026-07-03): the exact-touch-price
      # tripwire is regime-dependent — pre-split GOOG (~$2,700, spreads of tens of ticks)
      # measures 8-22%, jumping to 52-87% right after the 2022-07-15 20:1 split. Offset -1
      # beat offset 0 on all 11 probe days; structural gates (coverage/crossed) unchanged.

    An explicit --min_align always wins. Otherwise days strictly before
    --era-split-date (default 2022-07-18, the first post-split GOOG session)
    use --era-min-align (default 0.05); later days use 0.10. The defaults are
    the GOOG values; other tickers with a regime change should override the
    two era knobs rather than lowering --min_align globally.
    """
    if args.min_align is not None:
        return args.min_align
    if args.era_split_date and args.date and args.date < args.era_split_date:
        return args.era_min_align
    return DEFAULT_MIN_ALIGN


def cmd_verify(args, scratch):
    """Pick the ob-row offset by gating the L500 books against the canonical
    L10 book npy over ALL windows of the day. (The March linf npz is NOT used
    as ground truth: an audit showed 24% structurally impossible rows.)"""
    canon = load_canon(args.canon, scratch)
    canon_book = load_canon(args.canon_book, scratch)
    n = canon.shape[0]
    k = n // args.n_messages
    win_starts = np.arange(k, dtype=np.int64) * args.n_messages
    log(f'canon rows={n} -> {k} windows; book rows={canon_book.shape[0]}')
    raw_df = parse_m0(args.m0, scratch)
    wins, raws = match_windows(canon, win_starts, raw_df)
    if len(wins) < 0.999 * k:
        diag_unmatched(canon, win_starts, wins, raw_df)
        log(f'FATAL: only {len(wins)}/{k} windows matched in m0')
        return 4
    offsets = (-1, 0)
    need = {int(r) + o for r in raws for o in offsets}
    ob_map = collect_ob_rows(args.ob, need, args.levels)
    results = {}
    for o in offsets:
        snaps, ok = snapshots_for(ob_map, raws, o, args.levels)
        frac, n_eq, n_tot = book_gate(snaps, ok, canon_book, wins, args.levels)
        results[o] = frac
        log(f'  offset={o:+d}: L10 book-gate {n_eq}/{n_tot} ({frac:.4%})')
    winner = max(results, key=results.get)
    others = [v for k_, v in results.items() if k_ != winner]
    if results[winner] < args.gate_min or results[winner] - max(others) < 0.05:
        log(f'FATAL: no offset passes gate>={args.gate_min} with clear margin '
            f'— do NOT generate')
        return 3
    log(f'VERIFIED offset={winner:+d} (gate {results[winner]:.4%})')
    if args.offset_out:
        with open(args.offset_out, 'w') as f:
            f.write(str(winner))
    return 0


def cmd_generate(args, scratch):
    canon = load_canon(args.canon, scratch)
    n_canon = canon.shape[0]
    k = n_canon // args.n_messages
    win_starts = np.arange(k, dtype=np.int64) * args.n_messages
    log(f'{args.date}: canonical rows={n_canon} -> {k} windows')
    raw_df = parse_m0(args.m0, scratch)
    wins, raws = match_windows(canon, win_starts, raw_df)
    need = {int(r) + args.offset for r in raws}
    ob_map = collect_ob_rows(args.ob, need, args.levels)
    snaps, ok = snapshots_for(ob_map, raws, args.offset, args.levels)
    # Clean-build gates. We do NOT require an exact match to the cleaned book
    # (raw vs cleaned volumes legitimately differ — and the official linf, which
    # this replaces, is itself crossed/corrupt). We require only that the books
    # are STRUCTURALLY valid (not crossed), well-COVERED, and price-ALIGNED
    # enough to confirm the offset is right.
    cov = int(ok.sum()) / k if k else 0.0
    crossed, n_cross = crossed_fraction(snaps, ok)
    min_align = effective_min_align(args)
    info = ''
    if args.canon_book:
        canon_book = load_canon(args.canon_book, scratch)
        pf, ef, nt = book_align(snaps, ok, canon_book, wins, args.levels)
        info = f', price-align {pf:.2%}, exact {ef:.2%} (n={nt})'
    log(f'{args.date}: coverage {int(ok.sum())}/{k} ({cov:.2%}), '
        f'crossed {crossed:.3%} (n={n_cross}){info}')
    if cov < args.min_coverage:
        log(f'FATAL: {args.date} coverage {cov:.2%} < {args.min_coverage:.0%} — not writing')
        return 6
    if crossed > args.max_crossed:
        log(f'FATAL: {args.date} crossed {crossed:.2%} > {args.max_crossed:.0%} '
            f'(broken offset / corrupt source) — not writing')
        return 7
    if args.canon_book and pf < min_align:
        log(f'FATAL: {args.date} price-align {pf:.2%} < {min_align:.0%} '
            f'(offset/stream misaligned) — not writing')
        return 8
    wins, snaps = wins[ok], snaps[ok]
    idx = wins // args.n_messages
    out = os.path.join(args.out_dir,
                       f'{args.ticker}_{args.date}_L{args.levels}_linf_snapshots_{len(idx)}.npz')
    np.savez_compressed(out,
             snapshots=snaps,
             local_seq_indices=idx.astype(np.int64),
             target_market_rows=(idx * args.n_messages).astype(np.int64),
             n_levels=np.int64(args.levels),
             date=args.date,
             ticker=args.ticker)
    log(f'{args.date}: wrote {out} ({len(idx)}/{k} windows)')
    return 0


def cmd_audit(args, scratch):
    """Measure what the existing npz actually aligns to: per reference window,
    compare the npz book's parity-implied mid vs the canonical message's
    p_ref-implied mid at idx*n_messages (current loader behavior) and at
    idx*500 (alternative segmentation hypothesis). Also classify parity
    crossings — a 'crossed' book under the even=ask/odd=bid reading means the
    npz layout is NOT parity levels."""
    ref = np.load(args.ref)
    idxs = ref['local_seq_indices'].astype(np.int64)
    snaps = ref['snapshots']
    canon = load_canon(args.canon, scratch)
    n = canon.shape[0]
    log(f'audit: canon rows={n}, ref windows={len(idxs)}, '
        f'snap width={snaps.shape[1]}')

    def canon_mid(row):
        if row >= n:
            return None
        r = np.asarray(canon[row], dtype=np.float64)
        return float(r[3] - 100.0 * r[4])      # price_abs - 100*price_rel = p_ref

    def ref_book_stats(s):
        pv = s.astype(np.int64).reshape(-1, 2)
        nz = pv[:, 0] != 0
        ev = pv[::2][nz[::2]]                  # parity-even entries ("asks")
        od = pv[1::2][nz[1::2]]                # parity-odd entries ("bids")
        if not len(ev) or not len(od):
            return None
        best_ask, best_bid = ev[0, 0], od[0, 0]   # ladder heads
        return best_ask, best_bid, (best_ask < best_bid)

    rows = []
    for i, idx in enumerate(idxs):
        st = ref_book_stats(snaps[i])
        if st is None:
            continue
        ba, bb, crossed = st
        ref_mid = (ba + bb) / 2.0
        m1 = canon_mid(int(idx) * args.n_messages)
        m2 = canon_mid(int(idx) * 500)
        rows.append((int(idx), ref_mid, m1, m2, crossed))
        if i < 12:
            d1 = (ref_mid - m1) / 100 if m1 else float('nan')
            d2 = (ref_mid - m2) / 100 if m2 else float('nan')
            log(f'  idx={idx:5d} ref_mid={ref_mid:9.0f} ba={ba} bb={bb} '
                f'crossed={crossed}  d1k={d1:+8.1f}tk  d500={d2:+8.1f}tk')
    d1 = np.array([(r[1] - r[2]) / 100 for r in rows if r[2] is not None])
    d2 = np.array([(r[1] - r[3]) / 100 for r in rows if r[3] is not None])
    xfrac = np.mean([r[4] for r in rows])
    log(f'SUMMARY crossed_fraction={xfrac:.3f}')
    log(f'SUMMARY |mid diff| @idx*{args.n_messages}: median={np.median(np.abs(d1)):.1f}tk '
        f'mean={np.mean(np.abs(d1)):.1f}tk  signed median={np.median(d1):+.1f}tk')
    log(f'SUMMARY |mid diff| @idx*500 : median={np.median(np.abs(d2)):.1f}tk '
        f'mean={np.mean(np.abs(d2)):.1f}tk  signed median={np.median(d2):+.1f}tk')
    s0 = snaps[0].astype(np.int64)
    log('ref row0 first 24 vals: ' + str(s0[:24].tolist()))
    return 0


# --------------------------------------------------------------------------
# Batch-mode path resolution (the genericized replacement for the per-
# experiment sbatch drivers' `ls "$FARM"/*"$D"*message*` lookups).
# ONE os.scandir per directory, cached — no recursive globs, no per-file
# stat storm (Lustre-safe).
# --------------------------------------------------------------------------
_DIR_CACHE = {}


def _listdir_cached(d):
    if d not in _DIR_CACHE:
        with os.scandir(d) as it:
            _DIR_CACHE[d] = sorted(e.name for e in it)
    return _DIR_CACHE[d]


def _find_one(d, substrings, suffixes, what, required=True):
    hits = [n for n in _listdir_cached(d)
            if all(s in n for s in substrings) and n.endswith(tuple(suffixes))]
    if not hits:
        if required:
            raise FileNotFoundError(
                f'no {what} matching {substrings} + {suffixes} in {d}')
        return None
    if len(hits) > 1:
        log(f'  WARN: {len(hits)} {what} candidates in {d}, using first: {hits[0]}')
    return os.path.join(d, hits[0])


def _load_days(args):
    if args.dates:
        return [d.strip() for d in args.dates.split(',') if d.strip()]
    with open(args.days_json) as f:
        obj = json.load(f)
    days = obj['days'] if isinstance(obj, dict) else obj
    return [str(d) for d in days]


def cmd_batch_generate(args, scratch):
    """Multi-day generate: resolve per-day inputs from --canonical-npy-dir
    (canonical message + L10 book npy[.zst]) and --l500-csv-dir (stage-1 m0
    csv.zst + L500 orderbook csv.zst), then run the unchanged single-day
    cmd_generate. Skips days whose npz already exists (re-run friendly, same
    idempotency the sbatch drivers had)."""
    days = _load_days(args)
    log(f'batch generate: {len(days)} days, ticker={args.ticker}, '
        f'offset={args.offset:+d}')
    existing = set(_listdir_cached(args.out_dir))
    rc_all = 0
    for date in days:
        if any(args.ticker in n and date in n and n.endswith('.npz')
               for n in existing):
            log(f'== {date} already built, skip ==')
            continue
        try:
            args.date = date
            args.canon = _find_one(
                args.canonical_npy_dir, [f'{args.ticker}_{date}', 'message'],
                ('.npy', '.npy.zst'), 'canonical message npy')
            args.canon_book = _find_one(
                args.canonical_npy_dir, [f'{args.ticker}_{date}', 'orderbook'],
                ('.npy', '.npy.zst'), 'canonical L10 book npy',
                required=not args.no_canon_book)
            m0 = _find_one(args.l500_csv_dir, [f'{args.ticker}_{date}_'],
                           ('_message_0.csv.zst',), 'stage-1 m0 csv')
            args.m0 = m0
            # same derivation as the sbatch drivers:
            # OB="${M0%_message_0.csv.zst}_orderbook_500.csv.zst"
            args.ob = (m0[:-len('_message_0.csv.zst')]
                       + f'_orderbook_{args.levels}.csv.zst')
            if not os.path.exists(args.ob):
                raise FileNotFoundError(f'no stage-1 ob{args.levels} for {date}: {args.ob}')
            rc = cmd_generate(args, scratch)
        except FileNotFoundError as e:
            log(f'FATAL: {date}: {e}')
            rc = 2
        if rc:
            rc_all = rc
            if not args.keep_going:
                return rc
    return rc_all


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['verify', 'generate', 'audit'])
    # ---- single-day explicit inputs (original interface, unchanged) ----
    p.add_argument('--canon', help='canonical *_message_10_proc.npy[.zst]')
    p.add_argument('--canon_book', help='canonical L10 *orderbook*proc.npy[.zst] (gate)')
    p.add_argument('--m0', help='raw *_message_0.csv.zst')
    p.add_argument('--ob', help='cpp *_orderbook_500.csv.zst')
    p.add_argument('--gate_min', type=float, default=0.99)
    # clean-build gates (generate mode)
    p.add_argument('--min_coverage', type=float, default=0.90,
                   help='[generate] min fraction of windows matched in m0')
    p.add_argument('--max_crossed', type=float, default=0.02,
                   help='[generate] max fraction of crossed (ask<=bid) books')
    p.add_argument('--min_align', type=float, default=None,
                   help='[generate] min best-ask/bid price agreement vs cleaned '
                        'book (catches a broken offset; NOT a quality bar). '
                        'Default: era-aware — see effective_min_align '
                        f'({DEFAULT_MIN_ALIGN} post-split, --era-min-align before '
                        '--era-split-date)')
    p.add_argument('--era-split-date', default='2022-07-18',
                   help='[generate] first post-regime-change session (GOOG 20:1 '
                        'split default); empty string disables the era floor')
    p.add_argument('--era-min-align', type=float, default=0.05,
                   help='[generate] price-align floor for days before --era-split-date')
    p.add_argument('--levels', type=int, default=500)
    p.add_argument('--n_messages', type=int, default=1000,
                   help='window length AND stride (non-overlapping windows)')
    p.add_argument('--scratch', required=True)
    p.add_argument('--ref', help='[verify] existing linf npz to reproduce')
    p.add_argument('--offset_out', help='[verify] write winning offset here')
    p.add_argument('--out_dir', '--out-dir', dest='out_dir',
                   help='[generate] output directory')
    p.add_argument('--offset', type=int,
                   help='[generate] verified ob-row offset (from a verify run)')
    p.add_argument('--ticker', default='GOOG')
    p.add_argument('--date', help='[generate, single-day] YYYY-MM-DD')
    # ---- batch mode (generate only): day list + directory resolution ----
    p.add_argument('--dates', help='[generate, batch] comma-separated YYYY-MM-DD list')
    p.add_argument('--days-json', dest='days_json',
                   help='[generate, batch] JSON file: list of days, or {"days": [...]}')
    p.add_argument('--canonical-npy-dir', dest='canonical_npy_dir',
                   help='[generate, batch] dir with canonical message + L10 book '
                        'npy[.zst] (e.g. a squashfuse farm)')
    p.add_argument('--l500-csv-dir', dest='l500_csv_dir',
                   help='[generate, batch] dir with stage-1 *_message_0.csv.zst '
                        'and *_orderbook_<levels>.csv.zst replay outputs')
    p.add_argument('--no-canon-book', dest='no_canon_book', action='store_true',
                   help='[generate, batch] skip the price-align check when the '
                        'canonical dir has no L10 book files')
    p.add_argument('--keep-going', dest='keep_going', action='store_true',
                   help='[generate, batch] do not stop at the first failing day '
                        '(exit code = last failure)')
    args = p.parse_args()
    os.makedirs(args.scratch, exist_ok=True)
    if args.mode == 'audit':
        assert args.ref and args.canon, '--ref/--canon required'
        return cmd_audit(args, args.scratch)
    if args.mode == 'verify':
        assert args.canon and args.canon_book and args.m0 and args.ob, \
            '--canon/--canon_book/--m0/--ob required'
        return cmd_verify(args, args.scratch)
    # generate
    assert args.out_dir and args.offset is not None, '--out_dir/--offset required'
    os.makedirs(args.out_dir, exist_ok=True)
    batch = bool(args.dates or args.days_json)
    if batch:
        assert args.canonical_npy_dir and args.l500_csv_dir, \
            'batch mode: --canonical-npy-dir/--l500-csv-dir required'
        return cmd_batch_generate(args, args.scratch)
    assert args.canon and args.m0 and args.ob and args.date, \
        'single-day mode: --canon/--m0/--ob/--date required'
    return cmd_generate(args, args.scratch)


if __name__ == '__main__':
    sys.exit(main())
