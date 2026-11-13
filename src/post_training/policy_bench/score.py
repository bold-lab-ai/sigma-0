"""score(traces, terminal, cost): a pure function of the traces.

Per episode: pnl = cash + terminal value of the open position - fees, and its
return in bps of the notional traded (shares x mid at each fill). Episodes
weigh equally, as nano-sigma's PnL bench weighs its equal $1 bets
(tools/return_bench.py, pnl()): bps_per_bet is the mean over episodes that
traded, and the daily figures average the episodes of one date.

The ranking figure is daily_pnl_usd: the dollar PnL of every priced episode
(0 if it did not trade), averaged within each date, then over dates, with its
t over dates (daily_pnl_t). Episodes of one date share the market, so the date
is the unit of the t. episode_pnl_usd and episode_pnl_t treat episodes as
independent, for reference.
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


def _t(x: np.ndarray) -> float:
    """Mean over its standard error; nan for fewer than two values or no spread."""
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else float("nan")


def score(traces: list[Trace], terminal, cost) -> dict:
    pnl, notional = np.array([episode_pnl(t, terminal, cost) for t in traces]).reshape(-1, 2).T
    traded = (notional > 0) & np.isfinite(pnl)
    bps = np.where(traded, pnl / np.where(traded, notional, 1.0) * 1e4, np.nan)
    dates = np.array([t.date for t in traces])
    days = np.unique(dates[traded])
    daily = np.array([np.mean(bps[traded & (dates == d)]) for d in days])
    sd = daily.std(ddof=1) if len(daily) > 1 else float("nan")
    fills = np.concatenate([t.fills for t in traces]) if traces else np.zeros((0, 6))
    priced = np.isfinite(pnl)
    ep_pnl = pnl[priced]
    pdays = np.unique(dates[priced])
    dpnl = np.array([np.mean(pnl[priced & (dates == d)]) for d in pdays])
    return dict(
        daily_pnl_usd=float(dpnl.mean()) if len(dpnl) else float("nan"),
        daily_pnl_t=_t(dpnl), n_pnl_days=int(len(pdays)),
        episode_pnl_usd=float(ep_pnl.mean()) if len(ep_pnl) else float("nan"),
        episode_pnl_t=_t(ep_pnl),
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
