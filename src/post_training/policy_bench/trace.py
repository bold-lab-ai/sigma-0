"""What one episode leaves behind: enough to score it without re-running it.

A Trace keeps the mid after every background message, the book at the end
(its L10 view and the full depth of both sides), every agent fill, and the
decisions with their signal. score.py applies the
terminal rule and the costs to it after the fact. `save` and `load` keep a
list of traces in one npz file, no pickle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from post_training.policy_bench.book import AGENT, ASK, BID

FILL_COLS = ("k", "side", "price", "size", "maker", "mid")


@dataclass
class Trace:
    message: str              # manifest message file of the stock-day
    date: str
    start: int
    seed: int
    mid: np.ndarray           # (n + 1,) mid after k background messages; nan if a side is empty
    l10_end: np.ndarray       # (40,) book at the end
    fills: np.ndarray         # (n_fills, 6) float64, columns FILL_COLS
    decisions: np.ndarray     # (n_decisions,) k of each decision
    signal: np.ndarray        # (n_decisions, H) predicted returns, bps (empty if no signal)
    refused: int              # orders refused by the inventory cap or an empty side
    gen: dict = field(default_factory=dict)   # generation counts of a simulated episode
    depth_end: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    # (levels, 3) side (BID / ASK), price, background shares: every level of the end book, best first
    # per side, the agent's own orders left out. Empty for traces saved before it existed.

    @classmethod
    def build(cls, ep, mids, l10_end, fills, decisions, signals, refused, gen=None, book=None) -> "Trace":
        return cls(
            message=ep.entry["message"], date=ep.entry.get("date", ""), start=int(ep.start), seed=int(ep.seed),
            mid=np.array([np.nan if m is None else m for m in mids], np.float64),
            l10_end=np.asarray(l10_end, np.int64).copy(),
            fills=np.array([[f.k, f.side, f.price, f.size, f.maker, f.mid] for f in fills], np.float64).reshape(-1, 6),
            decisions=np.asarray(decisions, np.int64),
            signal=np.array(signals, np.float64).reshape(len(signals), -1) if signals else np.zeros((0, 0)),
            refused=int(refused), gen=dict(gen or {}),
            depth_end=book_depth(book) if book is not None else np.zeros((0, 3), np.int64))

    def position(self) -> int:
        return int((self.fills[:, 1] * self.fills[:, 3]).sum())

    def cash(self) -> float:
        """LOBSTER units x shares received (negative after buying)."""
        return float(-(self.fills[:, 1] * self.fills[:, 3] * self.fills[:, 2]).sum())


def book_depth(book) -> np.ndarray:
    """(levels, 3) side, price, shares of every level of `book`, bids then asks, best first,
    without the agent's orders (levels holding only agent orders are left out)."""
    out = []
    for side in (BID, ASK):
        for p in book.prices(side):
            q = sum(o.size for o in book.queue(side, p) if o.owner != AGENT)
            if q > 0:
                out.append((side, p, q))
    return np.array(out, np.int64).reshape(-1, 3)


def save(path: str, traces: list[Trace]) -> None:
    def ptr(xs):
        return np.cumsum([0] + [len(x) for x in xs])

    H = max((t.signal.shape[1] for t in traces if t.signal.size), default=0)
    np.savez(path,
             message=np.array([t.message for t in traces]), date=np.array([t.date for t in traces]),
             start=np.array([t.start for t in traces], np.int64), seed=np.array([t.seed for t in traces], np.int64),
             refused=np.array([t.refused for t in traces], np.int64),
             l10_end=np.stack([t.l10_end for t in traces]) if traces else np.zeros((0, 40), np.int64),
             mid=np.concatenate([t.mid for t in traces]) if traces else np.zeros(0),
             mid_ptr=ptr([t.mid for t in traces]),
             fills=np.concatenate([t.fills for t in traces]) if traces else np.zeros((0, 6)),
             fills_ptr=ptr([t.fills for t in traces]),
             decisions=np.concatenate([t.decisions for t in traces]) if traces else np.zeros(0, np.int64),
             signal=np.concatenate([t.signal.reshape(-1, H) if t.signal.size else np.zeros((len(t.decisions), H))
                                    for t in traces]) if traces else np.zeros((0, H)),
             dec_ptr=ptr([t.decisions for t in traces]),
             gen=np.array([json.dumps(t.gen, sort_keys=True) for t in traces]),
             depth_end=np.concatenate([t.depth_end for t in traces]) if traces else np.zeros((0, 3), np.int64),
             depth_ptr=ptr([t.depth_end for t in traces]))


def load(path: str) -> list[Trace]:
    with np.load(path) as f:          # NpzFile reads a member again on every access: read each once
        z = {k: f[k] for k in f.files}
    mp, fp, dp = z["mid_ptr"], z["fills_ptr"], z["dec_ptr"]
    qp = z.get("depth_ptr")
    return [Trace(message=str(z["message"][i]), date=str(z["date"][i]), start=int(z["start"][i]),
                  seed=int(z["seed"][i]), mid=z["mid"][mp[i]:mp[i + 1]], l10_end=z["l10_end"][i],
                  fills=z["fills"][fp[i]:fp[i + 1]], decisions=z["decisions"][dp[i]:dp[i + 1]],
                  signal=z["signal"][dp[i]:dp[i + 1]], refused=int(z["refused"][i]),
                  gen=json.loads(str(z["gen"][i])) if "gen" in z else {},
                  depth_end=z["depth_end"][qp[i]:qp[i + 1]] if qp is not None else np.zeros((0, 3), np.int64))
            for i in range(len(z["start"]))]
