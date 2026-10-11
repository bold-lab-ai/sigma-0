#!/usr/bin/env python3
# Provenance: verbatim from Jess Carr's FLAIR rotation repo (lob-evals),
# evals/lob_evals.py, integrated 2026-07-15. Archive:
# /lus/lfs1aip2/projects/public/u6gb/lob_evals_jcarr/ (formerly jess_repo/).
# Observational metrics suite over pipeline score outputs. Repoint paths per
# fork. Dependencies: numpy and pandas at import; scipy.stats (and
# scipy.optimize) imported lazily by several metrics, so they are required in
# practice; scikit-learn is genuinely optional (the GBM row is skipped without
# it).
"""
lob_evals.py - unified, provenance-stamped LOB evaluation driver.

DESIGN PRINCIPLES (read before editing):
  1. ANALYZE-ONLY. Never launches SLURM / never generates data. Consumes the
     output of a *validated* clean-tree inference run. Generation is separate
     (see gen_launch.sh).
  2. NO L10/L500 MIXING. Each EXPERIMENT has exactly one book depth. Every
     result is tagged with its experiment (-> depth + ticker set). L10 and L500
     numbers never commingle.
  3. PROVENANCE ON EVERY NUMBER. Each result carries `origin` = {experiment,
     ticker, depth, checkpoint, checkpoint_step, run_dir, job_id}. A
     _provenance.json is written next to every result.
  4. SANITY GATE. Before any model-dependent number, generated output is checked
     for the corruption signature (spreads far outside the real range). On
     failure the eval RAISES rather than emit a number. Guard against the
     encoding-swap bug silently producing tables.

USAGE (run with the pandas-2.x venv):
  $WORK/pd_alt/bin/python lob_evals.py --experiment goog_l500 --evals all
  $WORK/pd_alt/bin/python lob_evals.py --experiment multi_l10 --tickers GOOG MSFT --evals tail regime
  $WORK/pd_alt/bin/python lob_evals.py --experiment goog_l500 --dry-run   # check paths only

EVALS: tail, regime, joint, otm, stylized, repro, tstr, baselines, position.
  - Set CMSG/GMSG in CONFIG to match the run that produced the data (they drive the
    dense-feature reshape and the position-in-window tau bins). Default 250/250.
  - tstr's GBM (nonlinear) row needs scikit-learn in the venv; the linear ratio (the
    headline number) does NOT. To enable GBM:  $WORK/pd_alt/bin/pip install scikit-learn
  - position computes the single-c drift curve for one run; assemble the alpha(c)/c*
    surface across cells with fit_cg_surface() once the c-sweep cells have all run.
  - baselines gives the iid-resampler gap-closure exactly; the (omega,d) bigram's
    book-feature gap-closure needs the pipeline decoder (a gen-side task), noted in output.
"""
import argparse, glob, gzip, json, os, pickle, re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import numpy as np, pandas as pd

# =============================================================================
# CONFIG - EDIT HERE. Every path/parameter explicit. One place to audit.
# =============================================================================
PUBROOT = "/lus/lfs1aip2/projects/public/s5e/quant_team"
WORK    = "/lus/lfs1aip2/projects/u6gb/jesszcarr/lob_pipeline"
CONTROL = f"{WORK}/../lob_pipeline_control"

CHECKPOINT      = f"{PUBROOT}/quant/AlphaTrade/experiments/exp_R1_Mamba3/checkpoints/j3417629_pw8u0edj_3417629"
CHECKPOINT_STEP = 46050

# Two experiment classes, NEVER mixed. Fill run_dirs/scores_dirs per ticker
# after launching+validating each generation run (gen_launch.sh).
#   run_dir    = inference_results/<name>_<TICKER>_<jobid>  (has data_gen/ data_real/ data_cond/)
#   scores_dir = results_<name>/scores
EXPERIMENTS = {
    "goog_l500": {
        "depth": 500,
        "tickers": ["GOOG"],
        # Smoke rehearsal (N=256): point these at the 290M smoke once it is LOB-Bench scored:
        #   run_dir   : inference_results/jcarr-290m-smoke_GOOG_4685074
        #   scores_dir: results_jcarr-290m-smoke*/scores   (glob it; suffix may be -c250g250)
        # Then swap to the full N=3136 run (4685521 inference) for deck-final numbers.
        "run_dirs":    {"GOOG": f"{CONTROL}/inference_results/REPLACE_goog_l500_GOOG_JOBID"},
        "scores_dirs": {"GOOG": f"{CONTROL}/results_REPLACE_goog_l500/scores"},
        # NOTE: this baseline is the 78M s46050 model. CHECKPOINT is now ALSO the 78M
        # s46050 (j3417629_pw8u0edj, step 46050 — Aramis's confirmed canonical model, 22 May),
        # so the "repro" eval is a genuine same-model reproducibility check again (expect WS
        # within bootstrap CI). The earlier 290M j4559297 switch was reverted.
        "baseline_pickle": f"{PUBROOT}/lob_pipeline/results_ext-mamba3-s46050-c250g250/scores/scores_uncond_GOOG_ext-mamba3-s46050-c250g250_integrated_3423070.pkl",
    },
    "multi_l10": {
        "depth": 10,
        "tickers": ["GOOG"],   # extend: ["GOOG","MSFT","NVDA",...]
        "run_dirs":    {},     # {"GOOG": f"{CONTROL}/inference_results/jcarr-goog-l10-..._GOOG_JOBID"}
        "scores_dirs": {},     # {"GOOG": f"{CONTROL}/results_jcarr-goog-l10-.../scores"}
        "baseline_pickle": None,
    },
}

N_SEQUENCES = int(os.environ.get("NSEQ", 3136))
# c/g are derived per-run from the run-dir name in main() (the reliable source of truth:
# every cell dir is ...-c<NN>g<NN>...). Env CMSG/GMSG, if set, OVERRIDE the parsed value.
_CMSG_ENV = os.environ.get("CMSG"); _GMSG_ENV = os.environ.get("GMSG")
CMSG = int(_CMSG_ENV) if _CMSG_ENV is not None else 250  # default replaced per-run; see derive_cg_from_rundir
GMSG = int(_GMSG_ENV) if _GMSG_ENV is not None else 250  # drives dense reshape + position bins

_CG_RE = re.compile(r"-c(\d+)g(\d+)")
def derive_cg_from_rundir(run_dir):
    """Parse (c,g) from a run-dir basename like jcarr-...-c250g3750_GOOG_4908397.
    Returns (c,g) or (None,None). Env vars, if explicitly set, take precedence (handled in main)."""
    if not run_dir: return (None, None)
    m = _CG_RE.search(os.path.basename(run_dir.rstrip("/")))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)
SANITY_MAX_FRAC_OUT_OF_RANGE = 0.05      # gen spreads above real p99.9 allowed before "corrupt"
N_BOOT = 1000
RNG_SEED = 0

# TSTR / predictive-utility config
TSTR_H = 50                              # mid-direction horizon (messages)
TSTR_OFI_W = 50                          # OFI averaging window (Cont-Kukanov-Stoikov)
TSTR_TEST_FRAC = 0.5                     # fraction of REAL windows held out as the shared test set
# Position-in-window config
TAU_BINS = 5                             # number of within-window position bins
TAU_BIN_WIDTH = int(os.environ.get("TAU_BIN_WIDTH", 50))  # fixed bin width in MESSAGES; n_bins = g // width

# event_type codes (encoding_26tok.py: range(1,5), allowed [1,2,3,4]):
EVENT_LIMIT, EVENT_CANCEL, EVENT_DELETE, EVENT_EXEC = 1, 2, 3, 4
MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]
# orderbook CSV: col0=best ask px, col1=best ask sz, col2=best bid px, col3=best bid sz, ...

OUT_ROOT = f"{WORK}/eval_results"   # writable on-cluster; was /mnt/user-data (container-only)

# =============================================================================
# PROVENANCE
# =============================================================================
@dataclass
class Origin:
    experiment: str; ticker: str; depth: int
    checkpoint: str; checkpoint_step: int
    run_dir: str; job_id: str; n_sequences_expected: int
    ckpt_source: str = "config-fallback"   # where checkpoint/step came from: 'log:...' (trustworthy) or 'config-fallback' (unverified)
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

def _job_id(run_dir): 
    b = os.path.basename(run_dir.rstrip("/")); return b.split("_")[-1] if "_" in b else "unknown"

def resolve_checkpoint(run_dir):
    """Return (ckpt_path, step, source) for the data in run_dir, read from the
    ACTUAL inference invocation rather than the hardcoded CONFIG constants.

    This is the fix for the silent-mislabel failure mode: build_origin used to
    stamp the CONFIG CHECKPOINT into every result regardless of what truly
    generated the data, so a 290M run could be labelled '78M s46050'. We now
    scrape the real --ckpt_path / --checkpoint_step from the run's logs.

    Looks, in order, at:
      1. the pipeline .out log (the 'Command: ... --ckpt_path X --checkpoint_step N' line)
      2. the per-run log_rank0.txt inside run_dir
      3. falls back to CONFIG CHECKPOINT/CHECKPOINT_STEP (source='config-fallback')
    """
    name = os.path.basename(run_dir.rstrip("/"))
    candidates = []
    for root in (CONTROL, WORK):                       # pipeline logs live in <tree>/logs/
        candidates += glob.glob(f"{root}/logs/integrated_{name}.out")
        candidates += glob.glob(f"{root}/logs/*{name}*.out")
    candidates += glob.glob(f"{run_dir}/log_rank0.txt")           # per-run rank logs
    candidates += glob.glob(f"{run_dir}/**/log_rank0.txt", recursive=True)

    ck_re   = re.compile(r"--ckpt_path\s+\"?([^\s\"]+)")
    step_re = re.compile(r"--checkpoint_step\s+(\d+)")
    for fp in candidates:
        try:
            txt = open(fp, errors="ignore").read()
        except OSError:
            continue
        cm = ck_re.search(txt)
        if cm:
            sm = step_re.search(txt)
            step = int(sm.group(1)) if sm else CHECKPOINT_STEP
            return cm.group(1), step, f"log:{os.path.basename(fp)}"
    return CHECKPOINT, CHECKPOINT_STEP, "config-fallback"

def build_origin(experiment, ticker, run_dir_override=None):
    cfg = EXPERIMENTS[experiment]; rd = run_dir_override or cfg["run_dirs"].get(ticker, "")
    ckpt, step, src = resolve_checkpoint(rd) if rd else (CHECKPOINT, CHECKPOINT_STEP, "config-fallback")
    if src == "config-fallback" and rd:
        print(f"  !! WARNING: no inference log found for {os.path.basename(rd)} -- "
              f"provenance checkpoint is the CONFIG default and may NOT match the data.")
    else:
        print(f"  checkpoint (from {src}): {os.path.basename(ckpt)} step {step}")
    return Origin(experiment, ticker, cfg["depth"], ckpt, step,
                  rd, _job_id(rd), N_SEQUENCES, ckpt_source=src)

def stamp(result, origin):
    """Attach origin to every result dict so no number is path-ambiguous."""
    result["origin"] = asdict(origin); return result

def write_provenance(out_dir, origin, extra=None):
    """Write a _provenance.json next to results: origin + any extra diagnostics (e.g. sanity)."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {"origin": asdict(origin)}
    if extra: payload.update(extra)
    with open(os.path.join(out_dir, "_provenance.json"), "w") as f:
        json.dump(payload, f, indent=2)

# =============================================================================
# LOADERS
# =============================================================================
def load_scores(pkl):
    try:
        with open(pkl, "rb") as fh: return pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, OSError):
        with gzip.open(pkl, "rb") as fh: return pickle.load(fh)

def _read_msg(fp):
    df = pd.read_csv(fp, header=None, names=MSG_COLS)
    for c in MSG_COLS: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_messages(data_dir, kind):
    pat = "*message*gen*" if kind == "gen" else "*message*"
    out = {}
    for fp in sorted(glob.glob(os.path.join(data_dir, pat))):
        if kind == "real" and "gen" in os.path.basename(fp): continue
        out[os.path.basename(fp)] = _read_msg(fp)
    return out

def load_books(data_dir, kind):
    pat = "*orderbook*gen*" if kind == "gen" else "*orderbook*"
    out = {}
    for fp in sorted(glob.glob(os.path.join(data_dir, pat))):
        if kind == "real" and "gen" in os.path.basename(fp): continue
        # Every book-using eval only touches the best level (cols 0-3: ask_px, ask_sz,
        # bid_px, bid_sz). Reading all ~2000 L500 columns holds tens of GB and swap-thrashes.
        out[os.path.basename(fp)] = pd.read_csv(fp, header=None, usecols=[0, 1, 2, 3])
    return out

# =============================================================================
# SANITY GATE
# =============================================================================
class GenerationCorrupt(RuntimeError): pass

def _all_spreads(books):
    v = []
    for df in books.values():
        if df.shape[1] >= 3: v.append((df.iloc[:,0] - df.iloc[:,2]).values)
    return np.concatenate(v) if v else np.array([])

def assert_generation_sane(books_real, books_gen, origin):
    sr, sg = _all_spreads(books_real), _all_spreads(books_gen)
    if sr.size == 0 or sg.size == 0:
        raise GenerationCorrupt(f"[{origin.experiment}/{origin.ticker}] no spreads "
                                f"(real={sr.size}, gen={sg.size})")
    real_max = float(np.percentile(sr[sr > 0], 99.9))
    frac = float((sg > real_max).mean())
    diag = {"real_spread_p999": real_max, "gen_frac_above": frac,
            "gen_spread_med": float(np.median(sg)), "real_spread_med": float(np.median(sr))}
    if frac > SANITY_MAX_FRAC_OUT_OF_RANGE:
        raise GenerationCorrupt(
            f"[{origin.experiment}/{origin.ticker}] GENERATION LOOKS CORRUPT: "
            f"{frac:.1%} of gen spreads > real p99.9 ({real_max:.0f}); "
            f"gen median {diag['gen_spread_med']:.0f} vs real {diag['real_spread_med']:.0f}. "
            f"Refusing to score. (encoding-swap bug signature.)")
    return diag

# =============================================================================
# HELPERS
# =============================================================================
def _boot_ci(stat_fn, *arrays, n=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed); m = len(arrays[0]); out = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        out.append(stat_fn(*[a[idx] for a in arrays]))
    lo, hi = np.percentile(out, [2.5, 97.5]); return float(lo), float(hi)

def _w1(a, b, n_grid=4000):
    """W1 via a fixed quantile grid on already-sorted arrays (np.interp, no re-sort).
    n_grid caps the grid (784k-point grids were the tail/regime/baselines bottleneck);
    W1 is unchanged to ~4 sig figs vs the full grid."""
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0: return float("nan")
    m = min(a.size, b.size, n_grid)
    qs = (np.arange(m) + 0.5) / m
    qa = np.interp(qs * (a.size - 1), np.arange(a.size), a)
    qb = np.interp(qs * (b.size - 1), np.arange(b.size), b)
    return float(np.mean(np.abs(qa - qb)))

def _mid_from_book(df):
    return (df.iloc[:,0].values + df.iloc[:,2].values) / 2.0

def _sliced_w2(X, Y, n_dirs=50, seed=RNG_SEED):
    """Sliced-Wasserstein-2 between two 2D point clouds (Cramer-Wold projections).
    X,Y assumed already on a common scale (standardize before calling)."""
    rng = np.random.default_rng(seed); d = X.shape[1]; vals = []
    for _ in range(n_dirs):
        th = rng.normal(size=d); th /= (np.linalg.norm(th) + 1e-12)
        xp, yp = np.sort(X @ th), np.sort(Y @ th)
        n = min(len(xp), len(yp))
        if n == 0: continue
        qs = (np.arange(n) + 0.5) / n
        vals.append(np.mean((np.quantile(xp, qs) - np.quantile(yp, qs)) ** 2))
    return float(np.sqrt(np.mean(vals))) if vals else float("nan")

def _std_jointly(A, B):
    """Standardize two (n,d) clouds by the pooled mean/std so projections are comparable."""
    P = np.vstack([A, B]); mu = P.mean(0); sd = P.std(0) + 1e-12
    return (A - mu) / sd, (B - mu) / sd

def _auc(scores, y):
    """Rank-AUC (Mann-Whitney) with tie handling via average ranks."""
    from scipy.stats import rankdata
    y = np.asarray(y); n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0: return float("nan")
    r = rankdata(scores)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def _fit_logistic(X, y, l2=1e-3, iters=50):
    """Ridge-regularized IRLS logistic regression. X already standardized, no intercept col.
    Returns (w, b) for p = sigmoid(X@w + b). Small feature count -> tiny normal equations."""
    n, d = X.shape; Xb = np.hstack([X, np.ones((n, 1))]); w = np.zeros(d + 1)
    R = l2 * np.eye(d + 1); R[-1, -1] = 0.0  # don't penalize intercept
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        Wd = np.clip(p * (1 - p), 1e-6, None)
        H = Xb.T @ (Wd[:, None] * Xb) + R
        g = Xb.T @ (y - p) - R @ w
        try: w = w + np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
    return w[:-1], w[-1]

def _predict_logistic(X, w, b):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))

def _ofi_series(bdf):
    """Cont-Kukanov-Stoikov order-flow imbalance from L1 book updates.
    OFI_n = e^b_n - e^a_n, with e^b = q^b 1[P^b>=P^b_-] - q^b_- 1[P^b<=P^b_-], symmetric for ask."""
    aP = bdf.iloc[:, 0].values.astype(float); aS = bdf.iloc[:, 1].values.astype(float)
    bP = bdf.iloc[:, 2].values.astype(float); bS = bdf.iloc[:, 3].values.astype(float)
    db = np.where(bP[1:] >= bP[:-1], bS[1:], 0.0) - np.where(bP[1:] <= bP[:-1], bS[:-1], 0.0)
    da = np.where(aP[1:] <= aP[:-1], aS[1:], 0.0) - np.where(aP[1:] >= aP[:-1], aS[:-1], 0.0)
    return np.concatenate([[0.0], db - da])

def _tstr_features(books, H=TSTR_H, W=TSTR_OFI_W):
    """Build [windowed OFI, book imbalance, spread] features + sign(mid_{t+H}-mid_t) labels.
    Returns (X, y, seq_id) where seq_id indexes the source window (for leakage-free splits)."""
    Xs, ys, sid = [], [], []
    for wi, (k, bdf) in enumerate(sorted(books.items())):
        if bdf.shape[1] < 4 or len(bdf) <= W + H: continue
        aS = bdf.iloc[:, 1].values.astype(float); bS = bdf.iloc[:, 3].values.astype(float)
        spread = (bdf.iloc[:, 0].values - bdf.iloc[:, 2].values).astype(float)
        imb = (bS - aS) / (bS + aS + 1e-9)
        ofi = _ofi_series(bdf); csum = np.cumsum(ofi)
        ofi_w = csum.copy(); ofi_w[W:] = csum[W:] - csum[:-W]   # trailing W-sum
        mid = _mid_from_book(bdf)
        for t in range(W, len(bdf) - H):
            Xs.append([ofi_w[t], imb[t], spread[t]])
            ys.append(1 if mid[t + H] > mid[t] else 0); sid.append(wi)
    if not Xs: return np.empty((0, 3)), np.empty(0), np.empty(0, int)
    return np.asarray(Xs, float), np.asarray(ys, float), np.asarray(sid, int)

def _reshape_dense(dfs, feat, kind, win_len):
    """Reshape a dense feature's long-form scores to (n_windows, win_len) in file order.
    Returns None if the feature isn't dense/divisible (guards the 3136x250 assumption)."""
    if feat not in dfs: return None
    d = dfs[feat]; v = d[d.type == ("real" if kind == "real" else "generated")]["score"].values
    if v.size == 0 or v.size % win_len != 0: return None
    return v.reshape(-1, win_len)

# =============================================================================
# EVALS  (each: pure fn -> dict, stamped with origin)
# =============================================================================
def eval_tail_conditioned(scores, dfs, origin, quantiles=(0.90, 0.99)):
    """WS_q^tail = W1 over real+gen mass above real's q-quantile, per feature.

    ws_bulk is the RAW full-distribution W1 (_w1(r,g)), so it is on the SAME scale as the
    tails and the bulk->tail ratio is meaningful (this is the deck's "WS_bulk" column and
    the .90->.99 ratio). The normalized pickle score is kept as ws_bulk_pickle for
    traceability / cross-ref with repro+baselines (which use the normalized wasserstein[0]).
    PREVIOUS BUG: ws_bulk used scores[f]['wasserstein'][0] (normalized, ~0.018) while the
    tails are raw (~50); the ratio column was therefore not computable on a consistent scale."""
    feats = ["ask_volume","bid_volume","ask_volume_touch","bid_volume_touch","ofi",
             "spread","orderbook_imbalance","log_inter_arrival_time",
             "limit_bid_order_depth","limit_ask_order_depth"]
    res = {}
    for f in feats:
        if f not in dfs: continue
        d = dfs[f]; r = d[d.type=="real"]["score"].values; g = d[d.type=="generated"]["score"].values
        if r.size == 0 or g.size == 0: continue
        ws_bulk = _w1(r, g)                                  # raw scale, comparable to tails
        ws_bulk_pickle = (float(scores[f]["wasserstein"][0])
                          if (f in scores and "wasserstein" in scores[f]) else None)
        rbar = float(np.mean(np.abs(r))) if r.size else float("nan")        # real bulk scale
        ws_bulk_norm = (ws_bulk / rbar) if (rbar and rbar > 1e-12) else None  # scale-free bulk discrepancy
        row = {"ws_bulk": float(ws_bulk), "ws_bulk_pickle": ws_bulk_pickle,
               "ws_bulk_norm": ws_bulk_norm}
        for q in quantiles:
            thr = np.quantile(r, q)
            rt, gt = r[r >= thr], g[g >= thr]
            wt = _w1(rt, gt)
            tbar = float(np.mean(np.abs(rt))) if rt.size else float("nan")   # real tail scale
            ws_tail_norm = (wt / tbar) if (tbar and tbar > 1e-12) else None   # scale-free tail discrepancy = wt / real-tail-mean
            qr, qg = float(np.quantile(r, q)), float(np.quantile(g, q))
            row[f"ws_tail_{q}"] = wt
            row[f"ratio_{q}"] = (float(wt / ws_bulk) if (ws_bulk and ws_bulk > 1e-12) else None)   # RAW ratio (scale-confounded)
            row[f"ws_tail_{q}_norm"] = ws_tail_norm
            row[f"ratio_{q}_norm"] = (float(ws_tail_norm / ws_bulk_norm)                            # SCALE-FREE: relative tail vs relative bulk
                                      if (ws_bulk_norm and ws_tail_norm is not None and ws_bulk_norm > 1e-12) else None)
            row[f"qratio_{q}"] = (float(qg / qr) if abs(qr) > 1e-12 else None)                       # gen p_q / real p_q (1.0 = matched magnitude)
            row[f"n_real_tail_{q}"] = int(rt.size); row[f"n_gen_tail_{q}"] = int(gt.size)
        res[f] = row
    return stamp({"eval":"tail_conditioned","features":res,
                  "note":"ws_bulk = raw _w1(r,g) (deck WS_bulk col); ratio_q = ws_tail_q/ws_bulk (RAW, "
                         "scale-confounded for heavy tails). SCALE-FREE additions: ws_tail_q_norm = "
                         "ws_tail_q / mean(|real tail|); ratio_q_norm = (relative tail disc)/(relative bulk disc) "
                         "-- if >1 the model is genuinely worse in the tail, not just facing bigger numbers; "
                         "qratio_q = gen_p_q / real_p_q (magnitude of gen's q-quantile vs real's). "
                         "ws_bulk_pickle = normalized scores[f]['wasserstein'][0]; tail conditions on real q-quantile"}, origin)

def eval_regime_stratified(scores, dfs, books_real, books_gen, origin,
                           ws_feats=("spread","ask_volume","bid_volume","ask_volume_touch",
                                     "bid_volume_touch","orderbook_imbalance")):
    """Per-tertile fidelity (WS) and skill (IC), stratified by real-continuation regime.

    Stratifiers (per real window): realised vol (sum dlog-mid^2) and median spread.
    Each gets vol/spread tertiles. Then:
      - WS per tertile = mean over dense features of W1(real-windows-in-tertile,
        gen-windows-in-tertile), using the per-window dense reshape (n_windows x win_len)
        rather than the pooled pickle (which has no per-window WS). The reshape is aligned
        to the SAME sorted file order as the books, so window i's tertile labels its rows.
      - IC per tertile = Spearman across windows of (gen window log-return, real window
        log-return); gen/real windows paired by sorted index (matched conditioning).
      - regime-robustness summary = max_r IC(r) - min_r IC(r).
    """
    from scipy.stats import spearmanr
    rkeys = sorted(books_real.keys()); gkeys = sorted(books_gen.keys())
    n = min(len(rkeys), len(gkeys))
    if n == 0:
        return stamp({"eval":"regime_stratified","status":"no windows"}, origin)
    # ---- stratifiers + paired window returns, in sorted file order ----
    spread_lvl, vol, ret_real, ret_gen = [], [], [], []
    for i in range(n):
        rdf, gdf = books_real[rkeys[i]], books_gen[gkeys[i]]
        if rdf.shape[1] < 3 or gdf.shape[1] < 3:
            spread_lvl.append(np.nan); vol.append(np.nan); ret_real.append(np.nan); ret_gen.append(np.nan); continue
        sp = (rdf.iloc[:,0]-rdf.iloc[:,2]).values
        mr = _mid_from_book(rdf); mg = _mid_from_book(gdf)
        lr = np.diff(np.log(np.clip(mr,1e-9,None)))
        spread_lvl.append(float(np.median(sp))); vol.append(float(np.sum(lr**2)))
        ret_real.append(float(np.log(max(mr[-1],1e-9)) - np.log(max(mr[0],1e-9))))
        ret_gen.append(float(np.log(max(mg[-1],1e-9)) - np.log(max(mg[0],1e-9))))
    spread_lvl = np.array(spread_lvl); vol = np.array(vol)
    ret_real = np.array(ret_real); ret_gen = np.array(ret_gen)

    def tertiles(x):
        good = ~np.isnan(x); q1,q2 = np.percentile(x[good],[33.3,66.7])
        lab = np.full(len(x), -1); lab[good] = np.where(x[good]<=q1,0,np.where(x[good]<=q2,1,2))
        return lab

    # ---- reshape dense features once (real & gen), guard the 3136xwin assumption ----
    win_len = None
    for f in ws_feats:
        rr = _reshape_dense(dfs, f, "real", GMSG)
        if rr is not None: win_len = GMSG; break
    dense = {}
    if win_len is not None:
        for f in ws_feats:
            R = _reshape_dense(dfs, f, "real", win_len); G = _reshape_dense(dfs, f, "generated", win_len)
            if R is not None and G is not None: dense[f] = (R, G)

    def ws_for_windows(idx):
        """mean over dense feats of W1(real rows of these windows, gen rows of these windows)."""
        vals = []
        for f,(R,G) in dense.items():
            ri = idx[idx < R.shape[0]]; gi = idx[idx < G.shape[0]]
            if ri.size==0 or gi.size==0: continue
            vals.append(_w1(R[ri].ravel(), G[gi].ravel()))
        return float(np.mean(vals)) if vals else float("nan")

    def ic_for_windows(idx):
        a, b = ret_gen[idx], ret_real[idx]; m = (~np.isnan(a))&(~np.isnan(b))
        return float(spearmanr(a[m], b[m]).statistic) if m.sum()>2 else float("nan")

    def ic_ci_for_windows(idx, n=N_BOOT, seed=RNG_SEED):
        """Window-level bootstrap CI on the tertile IC (resample the tertile's windows).
        NOT day-clustered: if windows within a trading day are correlated this is
        anti-conservative (CI too tight). Honest window-level interval; tighten to a
        day-cluster bootstrap if window->day labels become available."""
        a, b = ret_gen[idx], ret_real[idx]; m = (~np.isnan(a))&(~np.isnan(b))
        a, b = a[m], b[m]
        if a.size <= 2: return [float("nan"), float("nan")]
        rng = np.random.default_rng(seed); out = []
        for _ in range(n):
            bi = rng.integers(0, a.size, a.size)
            s = spearmanr(a[bi], b[bi]).statistic
            if not np.isnan(s): out.append(s)
        if not out: return [float("nan"), float("nan")]
        lo, hi = np.percentile(out, [2.5, 97.5]); return [float(lo), float(hi)]

    out = {"n_windows": int(n), "win_len": win_len,
           "dense_features_used": list(dense.keys()),
           "ws_note": ("per-tertile WS over reshaped dense features (approximates the 21-feature "
                       "WS-21; uses the same W1 machinery + real per-tertile split). Empty if the "
                       "pickle long-form isn't window-divisible."),
           "ic_note": ("IC = Spearman(gen window log-return, real window log-return); paired by "
                       "sorted index. ic_ci = window-level bootstrap (NOT day-clustered; "
                       "anti-conservative if intra-day windows are correlated).")}
    for sname, strat in [("spread", spread_lvl), ("volatility", vol)]:
        lab = tertiles(strat); block = {}
        for r in (0,1,2):
            idx = np.where(lab==r)[0]
            block[["low","mid","high"][r]] = {"n":int(idx.size),
                "ws": ws_for_windows(idx) if dense else None,
                "ic": ic_for_windows(idx),
                "ic_ci": ic_ci_for_windows(idx)}
        ics = [block[t]["ic"] for t in ("low","mid","high") if not np.isnan(block[t]["ic"])]
        block["ic_robustness_spread"] = (float(max(ics)-min(ics)) if ics else None)
        out[sname] = block
    return stamp({"eval":"regime_stratified","data":out,
                  "note":"regime = real-continuation tertile (persistent over the window)"}, origin)

def eval_joint_canonical(msgs_real, msgs_gen, books_real, books_gen, origin):
    """Canonical vol-volume: per-window realised vol (mid returns) x executed volume
    (sum of size where event_type==EVENT_EXEC). Rebuilt from raw messages."""
    def per_window(msgs, books):
        sig, vexec, n_exec = [], [], []
        common = set(msgs) & {k.replace("orderbook","message") for k in books} if False else None
        for mk, mdf in msgs.items():
            bk = mk.replace("message","orderbook")
            if bk not in books: continue
            bdf = books[bk]
            mid = _mid_from_book(bdf); lr = np.diff(np.log(np.clip(mid,1e-9,None)))
            sig.append(float(np.sum(lr**2)))
            ex = mdf[mdf.event_type==EVENT_EXEC]["size"].sum()
            vexec.append(float(ex)); n_exec.append(int((mdf.event_type==EVENT_EXEC).sum()))
        return np.array(sig), np.array(vexec), np.array(n_exec)
    sr, vr, nr = per_window(msgs_real, books_real)
    sg, vg, ng = per_window(msgs_gen, books_gen)
    from scipy.stats import spearmanr
    def rho(x,y):
        m = (~np.isnan(x))&(~np.isnan(y))&(y>0)
        return float(spearmanr(x[m], y[m]).statistic) if m.sum()>2 else float("nan")
    rr, rg = rho(sr,vr), rho(sg,vg)
    # ---- Delta-rho bootstrap CI (resample windows on each side) ----
    rng = np.random.default_rng(RNG_SEED); dr_boot = []
    if len(sr) > 2 and len(sg) > 2:
        for _ in range(N_BOOT):
            ri = rng.integers(0, len(sr), len(sr)); gi = rng.integers(0, len(sg), len(sg))
            dr_boot.append(rho(sg[gi],vg[gi]) - rho(sr[ri],vr[ri]))
        dr_boot = np.array(dr_boot); dr_lo, dr_hi = np.nanpercentile(dr_boot,[2.5,97.5])
    else:
        dr_lo = dr_hi = float("nan")
    # ---- sliced-W2 between joint (sigma, V) clouds + real-vs-real noise floor ----
    def clean2d(s,v):
        m = (~np.isnan(s))&(~np.isnan(v))&(v>0); return np.column_stack([s[m], v[m]])
    Xr, Xg = clean2d(sr,vr), clean2d(sg,vg)
    sw2 = floor = float("nan")
    if len(Xr) > 4 and len(Xg) > 4:
        Xr_s, Xg_s = _std_jointly(Xr, Xg); sw2 = _sliced_w2(Xr_s, Xg_s)
        half = len(Xr)//2; perm = rng.permutation(len(Xr))
        A, B = Xr[perm[:half]], Xr[perm[half:2*half]]
        A_s, B_s = _std_jointly(A, B); floor = _sliced_w2(A_s, B_s)
    closure = (1 - abs(rr-rg)/abs(rr)) if (not np.isnan(rr) and abs(rr)>1e-9 and not np.isnan(rg)) else None
    return stamp({"eval":"joint_canonical",
        "rho_real":rr, "rho_gen":rg,
        "delta_rho": (rg-rr) if not (np.isnan(rr) or np.isnan(rg)) else None,
        "delta_rho_ci":[float(dr_lo), float(dr_hi)],
        "sliced_w2": float(sw2), "sliced_w2_floor_real_vs_real": float(floor),
        "closure": closure,
        "n_windows_real":int(len(sr)), "n_windows_gen":int(len(sg)),
        "exec_sparsity": {"mean_exec_per_window_real":float(np.mean(nr)) if nr.size else None,
                          "mean_exec_per_window_gen":float(np.mean(ng)) if ng.size else None,
                          "windows_with_zero_exec_real":int((nr==0).sum()),
                          "windows_with_zero_exec_gen":int((ng==0).sum())},
        "note":"executed volume = sum(size | event_type==4); realised vol = sum(dlog mid)^2; "
               "rho is window-level pooled (canonical pair). closure=1-|drho|/|rho_real|; "
               "SW2 floor = real split-half (suggestive if SW2 within floor's band)"}, origin)

def eval_otm_observational(msgs_real, books_real, msgs_gen, books_gen, origin, H=100):
    """Observational response R(k): mean |d mid| over H msgs after a new limit order
    at tick-distance k from mid. Real vs gen. (Observational proxy, not causal injection.)"""
    # Contiguous tick-distance partition: every new order lands in exactly one bin, so the
    # decay curve R(k) is read off a true coordinate (no dropped orders, no gap cherry-pick
    # question). Existing edges 0/5/15/35/75/150/500/1500 preserved; 5-15, 35-75, 500-1500
    # added to fill the former holes. 1500+ isolates the round-$ placements.
    bins = [(0,5),(5,15),(15,35),(35,75),(75,150),(150,500),(500,1500),(1500,10**9)]
    def response(msgs, books):
        acc = {b:[] for b in bins}
        for mk, mdf in msgs.items():
            bk = mk.replace("message","orderbook")
            if bk not in books: continue
            mid = _mid_from_book(books[bk])
            price = mdf["price"].values; et = mdf["event_type"].values
            news = np.where(et == EVENT_LIMIT)[0]
            news = news[(news + H < len(mid)) & (news < len(mid))]
            if news.size == 0: continue
            k = np.abs(price[news] - mid[news]) / 100.0          # ticks (price 1/10000$, tick=100)
            dmid = np.abs(mid[news + H] - mid[news]) / 100.0
            for b in bins:
                sel = (k >= b[0]) & (k < b[1])
                if sel.any(): acc[b].extend(dmid[sel].tolist())
        return {f"{b[0]}-{b[1]}": (float(np.mean(v)) if v else None, len(v)) for b,v in acc.items()}
    return stamp({"eval":"otm_observational","H":H,
                  "R_real":response(msgs_real,books_real),
                  "R_gen":response(msgs_gen,books_gen),
                  "note":"observational proxy (contiguous tick-distance partition); "
                         "clean test needs delta_k injection (model-side)"}, origin)

def eval_stylized_facts(msgs_real, books_real, msgs_gen, books_gen, origin):
    """Return autocorr, vol clustering, Hill tail index, leverage - real vs gen, event-time."""
    def series(books):
        rets = []
        for k, df in books.items():
            mid = _mid_from_book(df); lr = np.diff(np.log(np.clip(mid,1e-9,None)))
            rets.append(lr)
        return rets
    def acf(rlist, lag, sq=False):
        num=den=0.0
        for r in rlist:
            x = r**2 if sq else r
            if len(x)<=lag: continue
            x = x - x.mean(); num += np.sum(x[:-lag]*x[lag:]); den += np.sum(x*x)
        return float(num/den) if den>0 else float("nan")
    def hill(rlist, frac=0.05):
        allr = np.abs(np.concatenate([r for r in rlist if len(r)]))
        allr = allr[allr>0]; 
        if allr.size<50: return float("nan")
        k = max(10, int(frac*allr.size)); top = np.sort(allr)[-k:]
        return float(1.0/np.mean(np.log(top/top[0]+1e-12)))
    def leverage(rlist, lag):
        """Pooled Corr(r_t, r^2_{t+lag}) across windows; <0 = leverage effect. Event-time.
        (Leverage is a calendar-time effect; at event-time it is typically ~0 -- report,
        read as inconclusive rather than a failure if both real and gen sit near zero.)"""
        xs, ys = [], []
        for r in rlist:
            if len(r) <= lag: continue
            xs.append(r[:-lag]); ys.append(r[lag:]**2)
        if not xs: return float("nan")
        x = np.concatenate(xs); y = np.concatenate(ys)
        if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12: return float("nan")
        return float(np.mean((x-x.mean())*(y-y.mean())) / (x.std()*y.std()))
    rr, rg = series(books_real), series(books_gen)
    facts = {}
    for lag in (1,5,30):
        facts[f"ret_acf_lag{lag}"]  = {"real":acf(rr,lag), "gen":acf(rg,lag)}
        facts[f"vol_acf_lag{lag}"]  = {"real":acf(rr,lag,sq=True), "gen":acf(rg,lag,sq=True)}
    for lag in (1,5):
        facts[f"leverage_lag{lag}"] = {"real":leverage(rr,lag), "gen":leverage(rg,lag)}
    facts["hill_tail_index"] = {"real":hill(rr), "gen":hill(rg)}
    return stamp({"eval":"stylized_facts","facts":facts,
                  "note":"event-time (per-message) returns; real-vs-gen, not calendar exponents; "
                         "leverage = pooled Corr(r_t, r^2_{t+lag}), <0 = leverage effect"}, origin)

def eval_reproducibility(scores, dfs, origin, baseline_pickle):
    """Compare this run's WS to the baseline pickle, like-for-like, per feature."""
    if not baseline_pickle or not os.path.exists(baseline_pickle):
        return stamp({"eval":"reproducibility","status":"no baseline available"}, origin)
    bs, bd = load_scores(baseline_pickle)
    comp = {}
    for f in scores:
        if f in bs and "wasserstein" in scores[f] and "wasserstein" in bs[f]:
            comp[f] = {"this":float(scores[f]["wasserstein"][0]),
                       "baseline":float(bs[f]["wasserstein"][0])}
    return stamp({"eval":"reproducibility","per_feature":comp,
                  "note":"compare WS-21 like-for-like; expect within bootstrap CI if reproducing"}, origin)

def eval_tstr(msgs_real, books_real, msgs_gen, books_gen, origin):
    """Predictive utility (train-synthetic-test-real). Train a mid-direction predictor on
    [windowed OFI, book imbalance, spread]; label sign(mid_{t+H}-mid_t). Compare:
      r->r : train on REAL train windows, test on REAL held-out windows
      s->r : train on GEN windows,        test on the SAME REAL held-out windows
    Report AUC ratio s->r / r->r with a window-bootstrap CI. Logistic always; GBM if sklearn.
    """
    Xr, yr, sidr = _tstr_features(books_real)
    Xg, yg, _    = _tstr_features(books_gen)
    if Xr.shape[0] == 0 or Xg.shape[0] == 0:
        return stamp({"eval":"tstr","status":"insufficient features",
                      "n_real":int(Xr.shape[0]),"n_gen":int(Xg.shape[0])}, origin)
    # split REAL by window id (no leakage); shared real test set for both predictors
    rng = np.random.default_rng(RNG_SEED)
    uw = np.unique(sidr); rng.shuffle(uw)
    n_test = max(1, int(TSTR_TEST_FRAC * len(uw)))
    test_w = set(uw[:n_test].tolist()); is_test = np.array([s in test_w for s in sidr])
    Xtr_r, ytr_r = Xr[~is_test], yr[~is_test]
    Xte,  yte    = Xr[is_test],  yr[is_test]
    # standardize on each TRAIN set, apply to the shared real test set
    def zfit(X): mu=X.mean(0); sd=X.std(0)+1e-9; return mu,sd
    def zapp(X,mu,sd): return (X-mu)/sd
    def auc_linear(Xtr,ytr):
        mu,sd=zfit(Xtr); w,b=_fit_logistic(zapp(Xtr,mu,sd),ytr)
        return _auc(_predict_logistic(zapp(Xte,mu,sd),w,b), yte), (mu,sd,w,b)
    auc_rr, _ = auc_linear(Xtr_r, ytr_r)
    auc_sr, _ = auc_linear(Xg,    yg)
    lin = {"auc_rr":float(auc_rr),"auc_sr":float(auc_sr),
           "ratio": float(auc_sr/auc_rr) if auc_rr>0 else None}
    # GBM (nonlinear) row if sklearn present; certifies the linear claim is robust
    gbm = {"status":"skipped (scikit-learn not installed in venv)"}
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        def auc_gbm(Xtr,ytr):
            if len(Xtr) > 60000:
                idx = np.random.default_rng(RNG_SEED).choice(len(Xtr), 60000, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]
            m=HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=0.08,
                                             early_stopping=True,random_state=0)
            m.fit(Xtr,ytr); return _auc(m.predict_proba(Xte)[:,1], yte)
        g_rr=auc_gbm(Xtr_r,ytr_r); g_sr=auc_gbm(Xg,yg)
        gbm={"auc_rr":float(g_rr),"auc_sr":float(g_sr),
             "ratio":float(g_sr/g_rr) if g_rr>0 else None,
             "caveat":"if GBM auc_rr <= logistic auc_rr, real signal is ~linear; nonlinear row "
                      "certifies robustness but can't strongly probe nonlinear fidelity"}
    except Exception as e:
        gbm={"status":f"skipped ({type(e).__name__})"}
    # window-bootstrap CI on the linear ratio (resample test windows; shared resamples for both)
    test_idx_by_w = {w: np.where(sidr == w)[0] for w in test_w}
    mu_r,sd_r=zfit(Xtr_r); w_r,b_r=_fit_logistic(zapp(Xtr_r,mu_r,sd_r),ytr_r)
    mu_g,sd_g=zfit(Xg);    w_g,b_g=_fit_logistic(zapp(Xg,mu_g,sd_g),yg)
    pr=_predict_logistic(zapp(Xte,mu_r,sd_r),w_r,b_r)
    pg=_predict_logistic(zapp(Xte,mu_g,sd_g),w_g,b_g)
    # local index of each test window's rows within (Xte, yte, pr, pg)
    te_global=np.where(is_test)[0]; g2local={g:i for i,g in enumerate(te_global)}
    twl=list(test_w); win_rows=[np.array([g2local[r] for r in test_idx_by_w[w]]) for w in twl]
    ratios=[]
    for _ in range(N_BOOT):
        if not twl: break
        pick=rng.integers(0,len(twl),len(twl))
        sel=np.concatenate([win_rows[i] for i in pick])
        if sel.size==0: continue
        a_rr=_auc(pr[sel], yte[sel]); a_sr=_auc(pg[sel], yte[sel])
        if a_rr and a_rr>0 and not np.isnan(a_rr): ratios.append(a_sr/a_rr)
    ci=[float(np.nanpercentile(ratios,2.5)), float(np.nanpercentile(ratios,97.5))] if ratios else [float("nan")]*2
    lin["ratio_ci"]=ci
    return stamp({"eval":"tstr","H":TSTR_H,"ofi_window":TSTR_OFI_W,
                  "n_test_examples":int(yte.size),"frac_up":float(yte.mean()),
                  "n_test_windows":len(test_w),
                  "linear":lin,"gbm":gbm,
                  "note":"features=[windowed OFI (CKS), book imbalance, spread]; split by window; "
                         "shared real test set; window-bootstrap CI on linear ratio"}, origin)

def eval_baselines(scores, dfs, origin):
    """Empirical-baseline anchor: gap-closure g = (WS_baseline - WS_model)/WS_baseline.

    Baseline (i) iid message resampler: draws the feature's marginal i.i.d. -> its WS vs real
    is just the finite-sample real-vs-real floor. So g<0 on pure-marginal features means the
    model is WORSE than a trivial marginal resampler there (the intended cautionary anchor:
    WS-21 is a marginal metric a resampler can game). Computed exactly from the dfs long-form.

    Baseline (ii) (omega,d) bigram: needs message-sequence simulation run back through the
    pipeline DECODER to produce book features -> a GENERATION-side task, not analyze-only.
    The message-level structural part (event-type/direction transition match) is reported here;
    the book-feature gap-closure for the bigram is left to a gen_launch baseline row.
    """
    feats = ["spread","orderbook_imbalance","ask_volume","bid_volume",
             "ask_volume_touch","bid_volume_touch","ofi","log_inter_arrival_time"]
    rng = np.random.default_rng(RNG_SEED); res = {}
    for f in feats:
        if f not in dfs: continue
        d = dfs[f]; r = d[d.type=="real"]["score"].values; g = d[d.type=="generated"]["score"].values
        if r.size==0 or g.size==0: continue
        ws_model = float(scores[f]["wasserstein"][0]) if f in scores and "wasserstein" in scores[f] else _w1(r,g)
        # resampler WS = W1(real, iid-resample-of-real of size = gen count) ~ real-vs-real floor
        samp = rng.choice(r, size=min(len(g), len(r)), replace=True)
        ws_resampler = _w1(r, samp)
        g_close = (ws_resampler - ws_model)/ws_resampler if ws_resampler>1e-12 else None
        res[f] = {"ws_model":ws_model,"ws_resampler":float(ws_resampler),
                  "gap_closure":(float(g_close) if g_close is not None else None)}
    return stamp({"eval":"baselines","resampler_gap_closure":res,
                  "bigram":{"status":"book-feature bigram requires pipeline decoder (gen-side); "
                            "not computable analyze-only"},
                  "note":"g->1 model closes gap; g<0 model worse than trivial marginal resampler "
                         "(expected on pure marginals -- the WS-21-gameability anchor)"}, origin)

def eval_position_in_window(scores, dfs, books_real, origin,
                            feats=("spread","ask_volume","bid_volume","orderbook_imbalance"),
                            bin_width=TAU_BIN_WIDTH):
    """Within-window drift WS_tau at this run's single conditioning length c.
    FIXED-WIDTH bins: tau in [0,GMSG) is split into bins of `bin_width` MESSAGES, so a bin
    means the same physical position regardless of g (n_bins = GMSG // bin_width). This makes
    bins, slopes, growth, and the (c,g) heatmap directly comparable across different g.
    MATCHED comparison: per bin, WS = mean over dense feats of W1(gen rows AT tau, real rows AT tau),
    isolating generator drift from the real data's own within-window position structure.
    Also emits ws_tau_pooled = W1(gen-at-tau, real-pooled) for reference. Fit WS_tau = alpha +
    beta*tau^p on the matched curve, with tau in MESSAGE units (not tau/g)."""
    n_bins = max(1, GMSG // bin_width)
    edges = (np.arange(n_bins + 1) * bin_width).astype(int)
    edges[-1] = GMSG  # last edge snaps to GMSG (handles GMSG not divisible by bin_width)
    centers = ((edges[:-1] + edges[1:]) / 2.0)
    per_feat = {}; per_feat_pooled = {}
    ws_tau = np.full(n_bins, np.nan); ws_tau_pooled = np.full(n_bins, np.nan)
    acc = [[] for _ in range(n_bins)]; acc_pooled = [[] for _ in range(n_bins)]
    for f in feats:
        G = _reshape_dense(dfs, f, "generated", GMSG)
        R = _reshape_dense(dfs, f, "real", GMSG)   # real reshaped to (n_win, GMSG), sliced per tau
        Rpool = dfs[f][dfs[f].type=="real"]["score"].values if f in dfs else np.array([])
        if G is None or R is None or Rpool.size==0: continue
        fvals=[]; fvals_pool=[]
        for bi in range(n_bins):
            cols = slice(edges[bi], edges[bi+1])
            gv = G[:, cols].ravel()
            rv = R[:, cols].ravel()                # real AT THIS POSITION
            w_matched = _w1(gv, rv); w_pooled = _w1(gv, Rpool)
            fvals.append(w_matched); acc[bi].append(w_matched)
            fvals_pool.append(w_pooled); acc_pooled[bi].append(w_pooled)
        per_feat[f]=[float(x) for x in fvals]
        per_feat_pooled[f]=[float(x) for x in fvals_pool]
    for bi in range(n_bins):
        if acc[bi]: ws_tau[bi]=float(np.mean(acc[bi]))
        if acc_pooled[bi]: ws_tau_pooled[bi]=float(np.mean(acc_pooled[bi]))
    fit = {"status":"not enough bins"}
    good = ~np.isnan(ws_tau)
    if good.sum() >= 3:
        try:
            from scipy.optimize import curve_fit
            tau = centers[good]; y = ws_tau[good]   # tau in MESSAGE units now
            f_model = lambda t,a,b,p: a + b*np.power(np.clip(t,1e-6,None), p)
            popt,_ = curve_fit(f_model, tau, y, p0=[y.min(), max(y.max()-y.min(),1e-3), 1.0],
                               bounds=([0,0,0.2],[np.inf,np.inf,3.0]), maxfev=10000)
            fit = {"alpha":float(popt[0]),"beta":float(popt[1]),"p":float(popt[2]),
                   "model":"WS_tau = alpha + beta*tau^p  [MATCHED, fixed-width bins, tau in messages]"}
        except Exception as e:
            fit = {"status":f"fit failed ({type(e).__name__})"}
    return stamp({"eval":"position_in_window","c":CMSG,"g":GMSG,"n_bins":n_bins,
                  "bin_width":bin_width,
                  "tau_centers":[float(x) for x in centers],
                  "ws_tau":[float(x) if not np.isnan(x) else None for x in ws_tau],
                  "ws_tau_pooled":[float(x) if not np.isnan(x) else None for x in ws_tau_pooled],
                  "per_feature_ws_tau":per_feat,
                  "per_feature_ws_tau_pooled":per_feat_pooled,
                  "fit_single_c":fit,
                  "method":"MATCHED + FIXED-WIDTH bins (bin_width messages); bins comparable across g. "
                           "ws_tau_pooled retains legacy W1(gen-at-tau, real-pooled).",
                  "note":"single-c drift curve; fixed-width bins so tau positions align across g cells."}, origin)

def fit_cg_surface(position_jsons):
    """Aggregate per-c position_in_window results into the alpha(c)/c* surface (deck slide 2).
    position_jsons: list of dicts (loaded position.json from several c cells, same g).
    Fits alpha(c) across cells and reports c* (10%-closure point), alpha(0), alpha(cmax)."""
    rows = []
    for j in position_jsons:
        c = j.get("c"); fit = j.get("fit_single_c",{})
        if c is None or "alpha" not in fit: continue
        rows.append((float(c), fit["alpha"], fit.get("beta"), fit.get("p")))
    if len(rows) < 2: return {"status":"need >=2 c cells with successful single-c fits"}
    rows.sort(); cs = np.array([r[0] for r in rows]); al = np.array([r[1] for r in rows])
    a0, amax = float(al[np.argmin(cs)]), float(al[np.argmax(cs)])
    # c* = smallest c whose normalized residual fidelity drops below 0.1
    denom = (a0 - amax)
    cstar = None
    if abs(denom) > 1e-12:
        norm = (al - amax)/denom
        below = cs[norm < 0.1]
        cstar = float(below.min()) if below.size else None
    return {"alpha_of_c":{str(int(c)):float(a) for c,a in zip(cs,al)},
            "alpha_0":a0,"alpha_cmax":amax,"c_star":cstar,
            "note":"c* = min c with (alpha(c)-alpha(cmax))/(alpha(0)-alpha(cmax)) < 0.1"}

# =============================================================================
# DRIVER
# =============================================================================
ALL_EVALS = ["tail","regime","joint","otm","stylized","repro","tstr","baselines","position"]

def run_ticker(experiment, ticker, which, dry=False, run_dir_override=None, scores_dir_override=None):
    cfg = EXPERIMENTS[experiment]
    run_dir = run_dir_override or cfg["run_dirs"].get(ticker, "")
    scores_dir = scores_dir_override or cfg["scores_dirs"].get(ticker, "")
    origin = build_origin(experiment, ticker, run_dir_override=run_dir)
    label = os.path.basename(run_dir.rstrip("/")) or ticker      # per-cell, so cells don't collide
    out_dir = os.path.join(OUT_ROOT, experiment, label)
    print(f"\n=== {experiment} / {ticker} (depth L{cfg['depth']}) ===")
    print(f"  run_dir   : {run_dir}")
    print(f"  scores_dir: {scores_dir}")
    if dry:
        print(f"  data_gen exists : {os.path.isdir(os.path.join(run_dir,'data_gen'))}")
        print(f"  scores exist    : {bool(glob.glob(os.path.join(scores_dir,'scores_uncond_*')))}")
        return
    os.makedirs(out_dir, exist_ok=True)
    dg, dr = os.path.join(run_dir,"data_gen"), os.path.join(run_dir,"data_real")
    books_real = load_books(dr,"real"); books_gen = load_books(dg,"gen")
    # SANITY GATE first - refuse to score corrupt generation.
    diag = assert_generation_sane(books_real, books_gen, origin)
    print(f"  sanity OK: gen {diag['gen_frac_above']:.1%} above real p99.9 "
          f"(med gen {diag['gen_spread_med']:.0f} vs real {diag['real_spread_med']:.0f})")
    write_provenance(out_dir, origin, extra={"sanity":diag})
    msgs_real = load_messages(dr,"real"); msgs_gen = load_messages(dg,"gen")
    spk = sorted(glob.glob(os.path.join(scores_dir,"scores_uncond_*")))
    scores, dfs = load_scores(spk[-1]) if spk else ({},{})
    print(f"  loaded {len(books_real)} real / {len(books_gen)} gen books; "
          f"{len(msgs_real)} real / {len(msgs_gen)} gen msgs"
          + (f"; scores {len(dfs)} feats" if dfs else "; NO pickle (raw-only)"), flush=True)
    import time as _time
    results = {}
    def _emit(name, fn):
        t0 = _time.time(); r = fn()
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f: json.dump(r, f, indent=2)
        print(f"  wrote {name}.json  ({_time.time()-t0:.0f}s)", flush=True)
        results[name] = r
    if "tail" in which and dfs:      _emit("tail",      lambda: eval_tail_conditioned(scores, dfs, origin))
    if "regime" in which:            _emit("regime",    lambda: eval_regime_stratified(scores, dfs, books_real, books_gen, origin))
    if "joint" in which:             _emit("joint",     lambda: eval_joint_canonical(msgs_real, msgs_gen, books_real, books_gen, origin))
    if "otm" in which:               _emit("otm",       lambda: eval_otm_observational(msgs_real, books_real, msgs_gen, books_gen, origin))
    if "stylized" in which:          _emit("stylized",  lambda: eval_stylized_facts(msgs_real, books_real, msgs_gen, books_gen, origin))
    if "repro" in which and dfs:     _emit("repro",     lambda: eval_reproducibility(scores, dfs, origin, cfg["baseline_pickle"]))
    if "tstr" in which:              _emit("tstr",      lambda: eval_tstr(msgs_real, books_real, msgs_gen, books_gen, origin))
    if "baselines" in which and dfs: _emit("baselines", lambda: eval_baselines(scores, dfs, origin))
    if "position" in which and dfs:  _emit("position",  lambda: eval_position_in_window(scores, dfs, books_real, origin))
    with open(os.path.join(out_dir,"summary.json"),"w") as f:
        json.dump({"origin":asdict(origin),"evals":list(results)}, f, indent=2)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--tickers", nargs="*", default=None, help="subset; default = experiment's tickers")
    ap.add_argument("--evals", nargs="*", default=["all"])
    ap.add_argument("--run-dir", default=None, help="override EXPERIMENTS run_dir (eval one specific cell)")
    ap.add_argument("--scores-dir", default=None, help="override EXPERIMENTS scores_dir for that cell")
    ap.add_argument("--dry-run", action="store_true", help="print resolved paths, don't compute")
    args = ap.parse_args()
    # --- root-cause c/g resolution: derive from run-dir name; env vars override if set ---
    global CMSG, GMSG
    if args.run_dir:
        pc, pg = derive_cg_from_rundir(args.run_dir)
        if pc is not None:
            CMSG = int(_CMSG_ENV) if _CMSG_ENV is not None else pc
            GMSG = int(_GMSG_ENV) if _GMSG_ENV is not None else pg
            if _CMSG_ENV is not None and int(_CMSG_ENV) != pc:
                print(f"  !! NOTE: CMSG env={_CMSG_ENV} overrides run-dir c={pc}")
            if _GMSG_ENV is not None and int(_GMSG_ENV) != pg:
                print(f"  !! NOTE: GMSG env={_GMSG_ENV} overrides run-dir g={pg}")
        else:
            print(f"  !! WARNING: no -c<>g<> in run-dir name; using c={CMSG} g={GMSG} (may be wrong)")
    which = ALL_EVALS if "all" in args.evals else args.evals
    tickers = args.tickers or EXPERIMENTS[args.experiment]["tickers"]
    print(f"experiment={args.experiment}  depth=L{EXPERIMENTS[args.experiment]['depth']}  "
          f"tickers={tickers}  evals={which}  c={CMSG} g={GMSG}")
    for t in tickers:
        try:
            run_ticker(args.experiment, t, which, dry=args.dry_run,
                       run_dir_override=args.run_dir, scores_dir_override=args.scores_dir)
        except GenerationCorrupt as e:
            print(f"  !! SKIPPED {t}: {e}")

if __name__ == "__main__":
    main()
