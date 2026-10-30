"""score(traces, terminal, cost): a pure function of the traces.

Per episode: pnl = cash + terminal value of the open position - fees, and its
return in bps of the notional traded (shares x mid at each fill). Episodes
weigh equally, as nano-sigma's PnL bench weighs its equal $1 bets
(tools/return_bench.py, pnl()): bps_per_bet is the mean over episodes that
traded, and the daily figures average the episodes of one date.
"""

from __future__ import annotations

import numpy as np

from post_training.policy_bench.costs import UNITS_PER_DOLLAR
from post_training.policy_bench.trace import Trace


def episode_pnl(tr: Trace, terminal, cost) -> tuple[float, float]:
    """(pnl in $, notional traded in $) of one trace."""
    value, taker = terminal(tr, tr.position())
    pnl = tr.cash() + value - cost(tr.fills, taker)
    notional = float((tr.fills[:, 3] * tr.fills[:, 5]).sum())
    return pnl / UNITS_PER_DOLLAR, notional / UNITS_PER_DOLLAR


def score(traces: list[Trace], terminal, cost) -> dict:
    pnl, notional = np.array([episode_pnl(t, terminal, cost) for t in traces]).reshape(-1, 2).T
    traded = (notional > 0) & np.isfinite(pnl)
    bps = np.where(traded, pnl / np.where(traded, notional, 1.0) * 1e4, np.nan)
    dates = np.array([t.date for t in traces])
    days = np.unique(dates[traded])
    daily = np.array([np.mean(bps[traded & (dates == d)]) for d in days])
    sd = daily.std(ddof=1) if len(daily) > 1 else float("nan")
    fills = np.concatenate([t.fills for t in traces]) if traces else np.zeros((0, 6))
    return dict(
        episodes=len(traces), traded=int(traded.sum()), unpriced=int(((notional > 0) & ~np.isfinite(pnl)).sum()),
        bps_per_bet=float(np.mean(bps[traded])) if traded.any() else float("nan"),
        hit_rate=float(np.mean(pnl[traded] > 0)) if traded.any() else float("nan"),
        pnl_usd=float(np.nansum(pnl)), notional_usd=float(notional.sum()),
        daily_mean_bps=float(daily.mean()) if len(daily) else float("nan"),
        daily_t=float(daily.mean() / (sd / np.sqrt(len(daily)))) if sd > 0 else float("nan"),
        sharpe_annual=float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
        n_days=int(len(days)), fills=int(len(fills)),
        maker_share=float(fills[:, 4].mean()) if len(fills) else float("nan"),
        refused=int(sum(t.refused for t in traces)))
