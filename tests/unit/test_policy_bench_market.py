from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from lobpipeline.policy_bench import trace as trace_io  # noqa: E402
from lobpipeline.policy_bench.book import AGENT, ASK, BID, Book  # noqa: E402
from lobpipeline.policy_bench.costs import NO_COST, MakerTaker  # noqa: E402
from lobpipeline.policy_bench.market import BookFill, MidFill, ReplayMarket  # noqa: E402
from lobpipeline.policy_bench.policies import NoTrade, SignRule  # noqa: E402
from lobpipeline.policy_bench.rollout import rollout  # noqa: E402
from lobpipeline.policy_bench.score import score  # noqa: E402
from lobpipeline.policy_bench.terminal import FlattenTaker, MarkToMicro, MarkToMid  # noqa: E402
from lobpipeline.policy_bench.trace import Trace  # noqa: E402
from lobpipeline.policy_bench.types import BUY, SELL, Episode, Fill, LimitOrder, MarketOrder  # noqa: E402
from policy_bench_days import write_day  # noqa: E402


def _book():
    bk = Book()
    bk.add(ASK, 10_100, 1, 3)
    bk.add(ASK, 10_200, 2, 5)
    bk.add(BID, 10_000, 3, 4)
    return bk


def test_book_fill_walks_the_opposite_side():
    fills = BookFill().fill(MarketOrder(BUY, 6), _book(), k=7)
    assert [(f.price, f.size, f.maker, f.mid, f.k) for f in fills] == [(10_100.0, 3, False, 10_050.0, 7),
                                                                       (10_200.0, 3, False, 10_050.0, 7)]
    assert BookFill().fill(MarketOrder(BUY, 9), _book(), 0) == []                 # deeper than the side
    assert [(f.price, f.size) for f in BookFill().fill(MarketOrder(SELL, 4), _book(), 0)] == [(10_000.0, 4)]


def test_mid_fill_and_an_empty_side():
    assert [(f.price, f.size) for f in MidFill().fill(MarketOrder(SELL, 9), _book(), 0)] == [(10_050.0, 9)]
    one_sided = Book()
    one_sided.add(ASK, 10_100, 1, 3)
    assert MidFill().fill(MarketOrder(BUY, 1), one_sided, 0) == []
    assert BookFill().fill(MarketOrder(BUY, 1), one_sided, 0) == []


def _l10(ask, aq, bid, bq):
    body = np.zeros(40, np.int64)
    body[0::4], body[2::4] = 9_999_999_999, -9_999_999_999
    body[:4] = ask, aq, bid, bq
    return body


def _trace(fills, end_mid, l10_end, date="2026-01-05"):
    ep = Episode(entry=dict(message="X.npy", date=date), start=0)
    return Trace.build(ep, [10_050.0, end_mid], l10_end, fills, [0], [np.array([1.0])], 0)


def test_terminal_rules_and_fees():
    tr = _trace([Fill(0, BUY, 10_100.0, 3, False, 10_050.0)], 10_150.0, _l10(10_200, 1, 10_100, 2))
    assert tr.position() == 3 and tr.cash() == -30_300.0
    assert MarkToMid()(tr, 3) == (30_450.0, 0.0)
    assert MarkToMicro()(tr, 3) == (3 * (10_200 * 2 + 10_100 * 1) / 3, 0.0)
    assert math.isnan(FlattenTaker()(tr, 3)[0])                                   # only 2 shares bid
    assert FlattenTaker()(tr, 2) == (20_200.0, 2.0)
    assert MakerTaker()(tr.fills, taker_shares=2) == pytest.approx(5 * 0.003 * 10_000)
    maker = _trace([Fill(0, SELL, 10_100.0, 4, True, 10_050.0)], 10_050.0, _l10(10_100, 1, 10_000, 1))
    assert MakerTaker()(maker.fills) == pytest.approx(-4 * 0.002 * 10_000)


def test_flatten_taker_walks_the_whole_end_book():
    bk = Book()
    for i, (p, q) in enumerate([(10_100, 1), (10_000, 2), (9_000, 5)]):
        bk.add(BID if p < 10_050 else ASK, p, i, q)
    bk.add(BID, 10_000, 9, 50, AGENT)                     # the agent's own bid is not taken
    ep = Episode(entry=dict(message="X.npy", date="2026-01-05"), start=0)
    tr = Trace.build(ep, [10_050.0, 10_050.0], _l10(10_100, 1, 10_000, 2),
                     [Fill(0, BUY, 10_100.0, 6, False, 10_050.0)], [0], [], 0, book=bk)
    assert tr.depth_end.tolist() == [[BID, 10_000, 2], [BID, 9_000, 5], [ASK, 10_100, 1]]
    assert FlattenTaker()(tr, 6) == (2 * 10_000.0 + 4 * 9_000.0, 6.0)   # past the L10 view
    assert math.isnan(FlattenTaker()(tr, 8)[0])                         # more than the side: unpriced
    assert FlattenTaker()(tr, -1) == (-10_100.0, 1.0)


def test_score_weighs_episodes_equally():
    up = _trace([Fill(0, BUY, 10_050.0, 1, False, 10_050.0)], 10_150.5, _l10(10_200, 1, 10_100, 1))
    big = _trace([Fill(0, SELL, 10_050.0, 100, False, 10_050.0)], 10_000.0, _l10(10_100, 1, 9_900, 1))
    flat = _trace([], 10_000.0, _l10(10_100, 1, 9_900, 1))
    r = score([up, big, flat], MarkToMid(), NO_COST)
    one = 100.5 / 10_050 * 1e4
    two = 50 / 10_050 * 1e4
    assert r["traded"] == 2 and r["bps_per_bet"] == pytest.approx((one + two) / 2)
    assert r["pnl_usd"] == pytest.approx((100.5 + 5_000) / 10_000) and r["hit_rate"] == 1.0


def test_traces_survive_save_and_load(tmp_path):
    a = _trace([Fill(0, BUY, 10_100.0, 3, False, 10_050.0)], 10_150.0, _l10(10_200, 1, 10_100, 2))
    b = _trace([], 10_000.0, _l10(10_100, 1, 9_900, 1), date="2026-01-06")
    a.depth_end = np.array([[BID, 10_100, 2], [ASK, 10_200, 1]], np.int64)
    trace_io.save(str(tmp_path / "t.npz"), [a, b])
    a2, b2 = trace_io.load(str(tmp_path / "t.npz"))
    for x, y in ((a, a2), (b, b2)):
        assert (x.message, x.date, x.start, x.refused) == (y.message, y.date, y.start, y.refused)
        for f in ("mid", "l10_end", "fills", "decisions", "signal", "depth_end"):
            np.testing.assert_array_equal(getattr(x, f), getattr(y, f))


# ---------------------------------------------------------------- a replayed day
class _Signal:
    """Stands in for signal.Signal: one horizon, always predicting `value` bps."""
    prefill, horizons = 2, [128]

    def __init__(self, value):
        self.value, self.seen = value, []

    def inputs(self, msg, l10):
        self.seen.append((msg.copy(), l10.copy()))
        return np.zeros(1), np.zeros(1)

    def predict(self, tok, book):
        return np.full((len(tok), 1), self.value)


def _market(tmp_path, fill, **kw):
    return ReplayMarket(str(tmp_path / "shard"), str(tmp_path / "adds"), fill, history=2, **kw)


def test_replay_sign_rule_buys_at_the_ask_and_marks_to_mid(tmp_path):
    entry = write_day(tmp_path)
    sig = _Signal(+3.0)
    m = _market(tmp_path, BookFill(), n_messages=4, latency=0)
    [tr] = rollout(SignRule(0, size=2), sig, m, [Episode(entry, start=2)], every=64)
    # the decision sees rows 0-1 and the L10 rows before rows 0, 1 and 2
    msg, l10 = sig.seen[0]
    assert list(msg[:, 0]) == [20, 21] and len(l10) == 3
    assert tr.fills.tolist() == [[0, BUY, 10_100.0, 2, 0, 10_050.0]]
    # after rows 2-5: rows 3 and 4 take the 10,100 ask (2 then 3 shares), row 5 adds an ask at 10,400
    assert tr.mid.tolist() == [10_050.0, 10_050.0, 10_050.0, 10_150.0, 10_150.0]
    r = score([tr], MarkToMid(), NO_COST)
    assert r["bps_per_bet"] == pytest.approx(2 * (10_150 - 10_100) / (2 * 10_050) * 1e4)


def test_replay_latency_and_inventory_cap(tmp_path):
    entry = write_day(tmp_path)
    m = _market(tmp_path, MidFill(), n_messages=4, latency=1, inventory_cap=1)
    [tr] = rollout(SignRule(0, size=1), _Signal(-1.0), m, [Episode(entry, start=2)], every=2)
    # decisions at k = 0 and 2; the sell fills one message later, at the mid after row 2
    assert tr.decisions.tolist() == [0, 2] and tr.fills.tolist() == [[1, SELL, 10_050.0, 1, 0, 10_050.0]]
    m = _market(tmp_path, MidFill(), n_messages=4, latency=0, inventory_cap=1)
    [tr] = rollout(SignRule(0, size=3), _Signal(-1.0), m, [Episode(entry, start=2)], every=2)
    assert len(tr.fills) == 0 and tr.refused == 2


def test_replay_latency_in_nanoseconds(tmp_path):
    # rows 2-5 at +1,000, +2,000, +3,000, +4,000 ns; the decision at k = 0 sees row 1, at +0 ns
    entry = write_day(tmp_path, ns=(0, 0, 1_000, 2_000, 3_000, 4_000))

    def fill_k(latency_ns):
        m = _market(tmp_path, MidFill(), n_messages=4, latency_ns=latency_ns)
        [tr] = rollout(SignRule(0, size=1), _Signal(-1.0), m, [Episode(entry, start=2)], every=64)
        return tr.fills[:, 0].tolist()
    assert fill_k(0) == [0]              # before row 2, as latency 0
    assert fill_k(1_000) == [0]          # due at row 2's time: before row 2
    assert fill_k(1_500) == [1]          # after row 2, before row 3
    assert fill_k(10_000) == []          # never due within the episode
    m = _market(tmp_path, MidFill(), n_messages=4, latency=0)
    [tr] = rollout(SignRule(0, size=1), _Signal(-1.0), m, [Episode(entry, start=2)], every=64)
    assert tr.fills[:, 0].tolist() == [0]


def test_replay_refuses_limit_orders_and_no_trade_is_flat(tmp_path):
    entry = write_day(tmp_path)
    m = _market(tmp_path, BookFill(), n_messages=4)
    m.reset([Episode(entry, start=2)])
    with pytest.raises(NotImplementedError):
        m.submit(0, [LimitOrder(BUY, 10_000, 1)])
    [tr] = rollout(NoTrade(), None, m, [Episode(entry, start=2)])
    assert len(tr.fills) == 0 and score([tr], MarkToMid(), MakerTaker())["traded"] == 0


def test_replay_reuses_the_days_book_with_the_same_result(tmp_path):
    entry = write_day(tmp_path)
    eps = [Episode(entry, start=1), Episode(entry, start=4)]
    def market():
        return ReplayMarket(str(tmp_path / "shard"), str(tmp_path / "adds"), MidFill(), history=1,
                            n_messages=2, latency=0)

    reused = rollout(SignRule(0, size=1), _Signal(+1.0), market(), eps)       # the second replays forward
    fresh = rollout(SignRule(0, size=1), _Signal(+1.0), market(), eps[1:])    # from the snapshot
    for f in ("mid", "l10_end", "fills"):
        np.testing.assert_array_equal(getattr(reused[1], f), getattr(fresh[0], f))
    assert reused[1].mid.tolist() == [10_050.0, 10_150.0, 10_150.0]


def test_policies_count_orders_on_their_way():
    from lobpipeline.policy_bench.policies import SkewQuoter
    from lobpipeline.policy_bench.types import Cancel, Obs
    l10 = _l10(10_100, 5, 10_000, 5)
    rule = SignRule(0, size=100)
    sig = np.array([1.0])
    assert rule.step(Obs(0, 0, 0, l10, 0, 0.0, signal=sig)) == [MarketOrder(BUY, 100)]
    assert rule.step(Obs(0, 0, 0, l10, 0, 0.0, signal=sig, pending=[MarketOrder(BUY, 100)])) == []
    assert rule.step(Obs(0, 0, 0, l10, 100, 0.0, signal=-sig, pending=[MarketOrder(SELL, 100)])) == \
        [MarketOrder(SELL, 100)]
    q = SkewQuoter(index=0, size=100, threshold=0.5)
    obs = Obs(0, 0, 0, l10, 0, 0.0, open_orders=[(7, BUY, 9_900, 100)], signal=np.array([0.0]),
              pending=[LimitOrder(BUY, 10_000, 100), Cancel(7)])
    assert q.step(obs) == [LimitOrder(SELL, 10_100, 100)]   # the bid is on its way, its old bid's cancel too


def test_replay_shows_orders_on_their_way(tmp_path):
    entry = write_day(tmp_path, ns=(0, 0, 1_000, 2_000, 3_000, 4_000))
    m = _market(tmp_path, MidFill(), n_messages=4, latency_ns=2_500)
    m.reset([Episode(entry, start=2)])
    m.record_decision(0, None)
    m.submit(0, [MarketOrder(BUY, 1)])
    assert m.observe(0).pending == [MarketOrder(BUY, 1)]
    m.advance()
    m.advance()
    assert m.observe(0).pending == [MarketOrder(BUY, 1)]       # due at +2,500 ns, row 3 is at +2,000
    m.advance()
    assert m.observe(0).pending == [] and len(m.trace(0).fills) == 1


def test_skew_quoter_keeps_good_quotes_and_leans_with_the_signal():
    from lobpipeline.policy_bench.policies import SkewQuoter
    from lobpipeline.policy_bench.types import Cancel, Obs

    book = _l10(10_100, 5, 10_000, 5)
    q = SkewQuoter(index=0, size=100, threshold=0.5, cap=150)
    both = q.step(Obs(0, 0, 0, book, 0, 0.0, [], [], np.array([0.1])))
    assert both == [LimitOrder(BUY, 10_000, 100), LimitOrder(SELL, 10_100, 100)]
    # a rise is predicted: keep the bid that is still at the best, cancel the ask
    up = q.step(Obs(1, 0, 0, book, 0, 0.0, [(7, BUY, 10_000, 100), (8, SELL, 10_100, 100)], [], np.array([2.0])))
    assert up == [Cancel(8)]
    # long 100 with a cap of 150: no more bids; the stale ask at 10,200 is replaced
    full = q.step(Obs(2, 0, 0, book, 100, 0.0, [(8, SELL, 10_200, 100)], [], np.array([0.0])))
    assert full == [Cancel(8), LimitOrder(SELL, 10_100, 100)]
