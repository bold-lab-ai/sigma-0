"""Event-anchored return benchmark: mid-price return N milliseconds after a trade.

The existing return bench (``AlphaTrade/lob_pipeline/return_bench/run_return_bench.py``)
counts its horizon in *messages* and anchors at a fixed offset inside the conditioning
window. This module anchors at the instant a real trade executes and measures the
horizon in *wall-clock milliseconds*.

Counting in milliseconds is a deliberate departure from ``mid_training/return_alignment/
returns.py``, whose docstring argues that event counting keeps a slow generator clock
from masquerading as a return-distribution improvement. Both are right about different
questions. Event counting holds the clock fixed and asks only about price dynamics.
Millisecond counting puts the clock itself under test, because a trader cares where the
price is 10 ms after a fill and not how many messages arrived meanwhile. The cost is
that a degraded IC no longer separates "bad price prediction" from "bad timestamps",
so :func:`clock_diagnostics` runs alongside the metrics and every report carries it.

Direction encoding differs between layers and the two must not be conflated:

======================================  ==========  ==============================
layer                                   values      set by
======================================  ==========  ==============================
raw LOBSTER csv                         -1 / +1     data source
preprocessed ``*_proc.npy``             0 / 1       ``encoding_26tok.py`` ``(d+1)/2``
inference output csv (cond/gen/real)    -1 / +1     ``inference_no_errcorr.py``
======================================  ==========  ==============================

``direction`` names the side the *resting* order sat on, not the aggressor. A trade
against a resting bid means a seller crossed the spread, so the mid should fall.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Inference output csv is LOBSTER-shaped: one message per row.
LOBSTER_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]

# 14-column npy message arrays (inference_no_errcorr.py:50-64).
NPY_TIME_S = 8
NPY_TIME_NS = 9

EVENT_TYPE_TRADE = 4

# Resting side, inference-output convention. See module docstring: these are NOT the
# values found in *_proc.npy, which uses {0, 1}.
CSV_RESTING_BID = 1  # trade hit the bid  -> seller-initiated -> mid should fall
CSV_RESTING_ASK = -1  # trade lifted the ask -> buyer-initiated  -> mid should rise

# Same thing one layer down, for anchor selection over preprocessed arrays.
PROC_RESTING_BID = 1
PROC_RESTING_ASK = 0

# A realised return is "zero" only when the mid did not move at all. The smallest
# move possible is half a tick, which at a mid of 3.15e6 (GOOG, 1e-4 USD units) is
# log(1 + 50/3.15e6) = 1.59e-5 -- so anything between 1e-8 and 1e-5 cannot occur and
# the threshold has three orders of magnitude of room. It must stay well below half a
# tick: at a mid of 5e6 half a tick is exactly 1e-5, so a 1e-5 threshold would start
# discarding real moves as if the price had held still.
ZERO_RETURN_THRESH = 1e-8


def trade_sign_baseline(directions):
    """The return sign implied by the anchoring trade alone, using no model output.

    A trade against a resting bid is seller-initiated and the mid should fall; one
    against a resting ask should raise it. Both facts are already in the conditioning
    window, so this predicts without generating anything.

    It belongs next to every reported IC because it is the level a generative model
    has to clear to have contributed. Measured on the SP500 ladder it scores Ranked IC
    0.77-0.80 and 93-94% direction accuracy, above all four models -- a comparison that
    changes what a model IC of 0.68 means, and that no amount of noise floor or
    confidence interval would have revealed.

    ``directions`` uses the inference-output convention (CSV_RESTING_BID = +1).
    """
    d = np.asarray(directions, dtype=np.float64)
    return -d


def load_mid(book_path: str) -> np.ndarray:
    """Mid price per book row. Orderbook is [ask_p1, ask_v1, bid_p1, bid_v1, ...]."""
    if str(book_path).endswith(".npy"):
        book = np.load(book_path)
    else:
        book = pd.read_csv(book_path, header=None).to_numpy()
    ask = book[:, 0].astype(np.float64)
    bid = book[:, 2].astype(np.float64)
    mid = (ask + bid) / 2.0
    # LOBSTER pads missing levels with +/-9999999999; a non-positive quote on either
    # side means the level is absent, not that the price is zero.
    mid[(ask <= 0) | (bid <= 0)] = np.nan
    return mid


def load_times(msg_path: str) -> np.ndarray:
    """Wall-clock seconds past midnight, one entry per message."""
    if str(msg_path).endswith(".npy"):
        msg = np.load(msg_path)
        return msg[:, NPY_TIME_S].astype(np.float64) + msg[:, NPY_TIME_NS].astype(np.float64) * 1e-9
    return pd.read_csv(msg_path, header=None, names=LOBSTER_COLS)["time"].to_numpy(np.float64)


def price_at_horizon(
    times: np.ndarray,
    mid: np.ndarray,
    t0: float,
    p0: float,
    horizon_ms: float,
) -> tuple[bool, float]:
    """Mid price as of ``t0 + horizon_ms``, last observation carried forward.

    Returns ``(covered, price)``.

    ``covered=False`` means the sequence ended before ``t0 + horizon`` and the caller
    must drop this sample for this horizon. Without that check ``np.searchsorted``
    quietly returns the final row, which turns a 500 ms return into "the return at the
    end of the sequence" while raising nothing and producing no NaN. Measured on 2000
    real sequences, 42% of 500 ms samples end early, so the failure is common rather
    than a corner case.

    ``j < 0`` is a different situation and is *not* a coverage failure: no message
    landed at or before the target, so the book is untouched and the mid is still p0.
    """
    times = np.asarray(times, dtype=np.float64)
    t_target = t0 + horizon_ms / 1000.0

    if times.size == 0 or times[-1] < t_target:
        return False, float("nan")

    # side='right' then -1 lands on the last message at or before the target. Using
    # 'left' would skip a message that shares the target timestamp, and dropping the
    # -1 would read a message from the future.
    j = int(np.searchsorted(times, t_target, side="right")) - 1
    if j < 0:
        return True, float(p0)
    return True, float(mid[j])


def sample_returns(
    horizons_ms: Sequence[float],
    t0: float,
    p0: float,
    gen_times: np.ndarray,
    gen_mid: np.ndarray,
    real_times: np.ndarray,
    real_mid: np.ndarray,
) -> dict[float, tuple[float, float]]:
    """Log returns for one anchored sample, keyed by horizon.

    A horizon is present in the result only when *both* sides cover it, so the paired
    statistics always compare the same sample set.
    """
    out: dict[float, tuple[float, float]] = {}
    if not np.isfinite(p0) or p0 <= 0:
        return out

    for h in horizons_ms:
        cov_g, pg = price_at_horizon(gen_times, gen_mid, t0, p0, h)
        cov_r, pr = price_at_horizon(real_times, real_mid, t0, p0, h)
        if not (cov_g and cov_r):
            continue
        if not (np.isfinite(pg) and np.isfinite(pr)) or pg <= 0 or pr <= 0:
            continue
        out[h] = (float(np.log(pg / p0)), float(np.log(pr / p0)))
    return out


def compute_metrics(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """IC, Ranked IC and Direction Accuracy for one horizon.

    Direction Accuracy drops samples whose realized return is zero; those carry no
    directional information and would otherwise be scored against ``sign(0) == 0``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    n = int(pred.size)

    if n < 3:
        return {"IC": np.nan, "ranked_IC": np.nan, "direction_acc": np.nan,
                "N_total": n, "N_eff": 0}

    nonzero = np.abs(real) > ZERO_RETURN_THRESH
    n_eff = int(nonzero.sum())
    if n_eff > 0:
        acc = float(np.mean(np.sign(pred[nonzero]) == np.sign(real[nonzero])))
    else:
        acc = np.nan

    return {
        "IC": float(stats.pearsonr(pred, real)[0]),
        "ranked_IC": float(stats.spearmanr(pred, real)[0]),
        "direction_acc": acc,
        "N_total": n,
        "N_eff": n_eff,
    }


def shuffle_null(
    pred: np.ndarray,
    real: np.ndarray,
    n_draws: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Metric distribution under randomly re-paired samples.

    This is what says whether an observed IC means anything. It also corrects a
    tempting assumption: Direction Accuracy under the null is not 50%, because the
    realized returns are not sign-balanced. Measured on 39 real anchors the null sits
    at 43-49%, so reading 47% as "worse than chance" would be wrong.
    """
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if pred.size < 5:
        return {"IC_mean": np.nan, "IC_sd": np.nan, "dir_acc_mean": np.nan}

    rng = np.random.default_rng(seed)
    ics, accs = [], []
    for _ in range(n_draws):
        m = compute_metrics(pred[rng.permutation(pred.size)], real)
        ics.append(m["IC"])
        accs.append(m["direction_acc"])
    return {
        "IC_mean": float(np.nanmean(ics)),
        "IC_sd": float(np.nanstd(ics)),
        "dir_acc_mean": float(np.nanmean(accs)),
    }


def clock_diagnostics(
    gen_times: np.ndarray,
    real_times: np.ndarray,
    t0: float,
    horizon_ms: float,
) -> dict[str, float]:
    """How many messages each side emitted inside the same wall-clock window.

    A millisecond horizon scores the generator's clock as well as its prices. When
    these counts diverge, a degraded IC has two possible causes and this is what tells
    them apart.
    """
    t_target = t0 + horizon_ms / 1000.0
    n_gen = int(np.sum((gen_times > t0) & (gen_times <= t_target)))
    n_real = int(np.sum((real_times > t0) & (real_times <= t_target)))
    return {
        "n_msg_gen": n_gen,
        "n_msg_real": n_real,
        "msg_ratio": (n_gen / n_real) if n_real > 0 else np.nan,
    }


def magnitude_skill(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """Rank |predicted| against |realised|, among samples whose sign is already right.

    Separated out because the three headline metrics are all dominated by direction,
    and direction is largely mechanical here: a trade removes liquidity from the side
    it hit, so the mid moves that way. The trade-sign baseline reaches 93% direction
    accuracy without a model. What it cannot do is say *how far* -- it emits one
    magnitude for every sample, so this statistic is undefined for it.

    That makes this the part a generative model can actually contribute. On the SP500
    ladder it is weak but non-zero (~0.11), which is a more honest summary of the
    model's edge than a Ranked IC that the baseline beats.
    """
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    nz = np.abs(real) > ZERO_RETURN_THRESH
    if nz.sum() < 3:
        return {"magnitude_ric": np.nan, "n_signed_correct": 0}
    right = np.sign(pred[nz]) == np.sign(real[nz])
    if right.sum() < 3 or np.unique(np.abs(pred[nz][right])).size < 2:
        # A constant-magnitude predictor cannot rank. nan is the correct answer, not 0.
        return {"magnitude_ric": np.nan, "n_signed_correct": int(right.sum())}
    return {"magnitude_ric": float(stats.spearmanr(np.abs(pred[nz][right]),
                                                   np.abs(real[nz][right]))[0]),
            "n_signed_correct": int(right.sum())}


def distribution_comparison(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """Whether the generated returns have the right spread, not just the right sign.

    IC is scale free, so a generator that gets every direction right while moving the
    price half as far still scores well. These numbers catch that.
    """
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if pred.size < 3:
        return {"std_gen_bp": np.nan, "std_real_bp": np.nan, "std_ratio": np.nan,
                "abs_median_gen_bp": np.nan, "abs_median_real_bp": np.nan,
                "ks_stat": np.nan, "ks_pvalue": np.nan}

    sd_g, sd_r = float(pred.std()), float(real.std())
    ks = stats.ks_2samp(pred, real)
    return {
        "std_gen_bp": sd_g * 1e4,
        "std_real_bp": sd_r * 1e4,
        "std_ratio": (sd_g / sd_r) if sd_r > 0 else np.nan,
        "abs_median_gen_bp": float(np.median(np.abs(pred))) * 1e4,
        "abs_median_real_bp": float(np.median(np.abs(real))) * 1e4,
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }
