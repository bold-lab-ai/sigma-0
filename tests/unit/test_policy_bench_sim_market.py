from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from post_training.policy_bench.book import AGENT, ASK, BID  # noqa: E402
from post_training.policy_bench.codec import NS, Message  # noqa: E402
from post_training.policy_bench.nanosigma.model_newtok import NanoSigmaNewTok  # noqa: E402
from post_training.policy_bench.policies import NoTrade  # noqa: E402
from post_training.policy_bench.rollout import rollout  # noqa: E402
from post_training.policy_bench.sim_market import SimMarket  # noqa: E402
from post_training.policy_bench.types import BUY, Cancel, Episode, LimitOrder, MarketOrder  # noqa: E402
from policy_bench_days import write_day  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoSigmaNewTok(d=64, n_heads=4, n_fused=2).eval()


def _market(tmp_path, model, batch=2, **kw):
    kw.setdefault("n_messages", 6)
    # a random model's draws are almost never legal: fall back to the masks after two
    return SimMarket(model, str(tmp_path / "shard"), str(tmp_path / "adds"), batch, history=2,
                     dtype=torch.float32, max_rejections=2, **kw)


def test_no_trade_rollout_is_legal_and_reproducible(tmp_path, model):
    entry = write_day(tmp_path)
    eps = [Episode(entry, start=2, seed=7), Episode(entry, start=2, seed=7)]
    a = rollout(NoTrade(), None, _market(tmp_path, model, n_messages=4), eps)
    b = rollout(NoTrade(), None, _market(tmp_path, model, n_messages=4), eps)
    for t in a:
        assert t.gen["messages"] == 4 and len(t.mid) == 5 and len(t.fills) == 0
    for x, y in [(a[0], a[1]), (a[0], b[0])]:            # same seed: same draws, row by row and run by run
        np.testing.assert_array_equal(x.mid, y.mid)
        np.testing.assert_array_equal(x.l10_end, y.l10_end)
        assert x.gen == y.gen


class _Trader:
    """k = 0: a resting bid and a market buy; k = 2: cancel the bid."""

    def reset(self, ep):
        self.sent = 0

    def step(self, obs):
        out = []
        if obs.k == 0:
            out = [LimitOrder(BUY, 9_900, 2), MarketOrder(BUY, 1)]
        elif obs.k == 2:
            out = [Cancel(oid) for oid, _ in obs.open_orders]
        self.sent += len(out)
        return out


def test_agent_orders_reach_the_book_and_the_simulator(tmp_path, model):
    entry = write_day(tmp_path)
    m = _market(tmp_path, model, batch=1, latency=1)
    m.reset([Episode(entry, start=2, seed=3)])
    pol = _Trader()
    pol.reset(None)
    seen_resting = False
    while not m.done():
        k = m.rows[0].k
        obs = m.observe(0)
        m.record_decision(0, None)
        m.submit(0, pol.step(obs))
        m.advance()
        r = m.rows[0]
        if k == 0:                                          # active at k = 1, fed before message 1
            continue
        seen_resting |= any(o.owner == AGENT for p in r.book.prices(BID) for o in r.book.queue(BID, p))
    tr = m.trace(0)
    assert seen_resting
    assert tr.gen["agent_messages"] + tr.refused == pol.sent and tr.gen["messages"] == 6
    taker = tr.fills[tr.fills[:, 4] == 0]
    assert len(taker) == 1 and taker[0, 1] == BUY and taker[0, 3] == 1
    assert tr.position() == int((tr.fills[:, 1] * tr.fills[:, 3]).sum())
    assert not m.rows[0].resting                           # cancelled or filled
    assert not any(o.owner == AGENT for p in m.rows[0].book.prices(BID) for o in m.rows[0].book.queue(BID, p))


def test_background_execution_fills_the_agent_front_of_queue(tmp_path, model):
    entry = write_day(tmp_path)
    m = _market(tmp_path, model, batch=1)
    m.reset([Episode(entry, start=2, seed=1)])
    r = m.rows[0]
    r.book.add(ASK, 10_050, 99, 4, AGENT)                   # the best ask, the agent's
    r.resting[99] = -BUY
    t = m._t_prev(r) + 5
    m._apply_background(r, Message(4, 0, 0, 5, t // NS, t % NS, ((10_050, 4), (10_100, 1))))
    assert [(f.side, f.price, f.size, f.maker) for f in r.fills] == [(-BUY, 10_050.0, 4, True)]
    assert r.position == -4 and 99 not in r.resting
    rows = r.log.rows(2)
    assert rows[:, 0].tolist() == [99, 10] and rows[:, 5].tolist() == [4, 1] and rows[:, 6:8].tolist() == [[0, 5], [0, 0]]
