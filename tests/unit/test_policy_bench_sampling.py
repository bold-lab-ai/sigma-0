from __future__ import annotations

import numpy as np

from lobpipeline.policy_bench import rules
from lobpipeline.policy_bench.book import AGENT, ASK, BID, Book
from lobpipeline.policy_bench.codec import NS, Message, encode, mid_floor
from lobpipeline.policy_bench.nanosigma.data import SIGN0, TIME0
from lobpipeline.policy_bench.sampling import MessageState, empties_side, file_end_ns, finish, legal_tokens

T0 = 36_000 * NS + 123_456_789
END = 57_600 * NS


def _sample(book, rng, t_prev=T0, end=END):
    mid = mid_floor(book.best(ASK), book.best(BID))
    st = MessageState(t_prev)
    while not st.done():
        f = st.forced()
        ok = legal_tokens(st, book, mid, end)
        assert len(ok), f"no legal token after {st.toks}"
        st.push(f if f is not None else rng.choice(ok))
    return st, mid


def _random_book(rng):
    bk, oid = Book(), 1
    ask0 = 10_000 + 100 * int(rng.integers(1, 4))
    for side, base, step in ((ASK, ask0, 100), (BID, 10_000, -100)):
        for lvl in range(int(rng.integers(1, 4))):
            for _ in range(int(rng.integers(1, 4))):
                owner = AGENT if rng.random() < 0.2 else 0
                bk.add(side, base + step * lvl, oid, int(rng.integers(1, 300)), owner)
                oid += 1
    if rng.random() < 0.3:                     # an order off the tick grid
        bk.add(BID, 9_955, oid, 7)
    return bk


def test_masked_sampling_always_gives_a_legal_message():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(400):
        bk = _random_book(rng)
        st, mid = _sample(bk, rng)
        m, reason = finish(st, bk, mid, END)
        assert reason is None, (reason, st.toks)
        assert m.time == T0 + m.dt
        seen.add(m.evt)
        slot = rules.slot_candidates(m, bk)[0] if m.evt in (2, 3) else None
        rules.apply(m, bk, oid=10**6, slot=slot)
    assert seen == {1, 2, 3, 4}


def test_time_is_forced_from_the_previous_message():
    st = MessageState(T0)
    for t in [1004, 2110, TIME0 + 2, TIME0 + 999, TIME0 + 0, TIME0 + 1]:
        st.push(t)
    t = T0 + 2 * NS + 999 * 10**6 + 1
    want = encode(Message(1, 0, 2, 999 * 10**6 + 1, t // NS, t % NS, ((10_100, 1),)), 10_000)[6:11]
    got = []
    while st.forced() is not None:
        got.append(st.forced())
        st.push(got[-1])
    assert got == list(want) and st.kind() == "sign"


def test_dt_is_masked_to_the_close():
    bk = _random_book(np.random.default_rng(1))
    mid = mid_floor(bk.best(ASK), bk.best(BID))
    st = MessageState(END - 2 * NS - 5)            # 2 s and 5 ns before the close
    st.push(1004)
    st.push(2110)
    assert legal_tokens(st, bk, mid, END).max() == TIME0 + 2
    st.push(TIME0 + 2)
    assert legal_tokens(st, bk, mid, END).tolist() == [TIME0]
    for _ in range(2):
        st.push(TIME0)
    assert legal_tokens(st, bk, mid, END).tolist() == list(range(TIME0, TIME0 + 6))


def test_finish_rejects_what_the_rules_reject():
    bk = Book()
    bk.add(ASK, 10_100, 1, 5)
    bk.add(BID, 10_000, 2, 5, AGENT)
    mid = mid_floor(10_100, 10_000)

    def state(m):
        st = MessageState(m.time - m.dt)
        for t in encode(m, mid):
            st.push(t)
        return st

    t = T0 + 1
    cross = Message(1, 1, 0, 1, t // NS, t % NS, ((10_100, 1),))           # a bid at the best ask
    agent = Message(3, 1, 0, 1, t // NS, t % NS, ((10_000, 5),), qslot=0)  # deletes the agent's order
    good = Message(1, 1, 0, 1, t // NS, t % NS, ((10_000, 3),))
    assert finish(state(cross), bk, mid, END)[1] == "new_crosses"
    assert finish(state(agent), bk, mid, END)[1] == "slot_agent"
    assert finish(state(good), bk, mid, END) == (good, None)
    st = state(good)
    st.toks[11] = SIGN0 + 1                       # +0 ticks: decodes, but is not how encode writes it
    st.toks[12:14] = [1108, 1108]
    assert finish(st, bk, mid, END)[1] == "noncanonical"
    late = Message(1, 1, 0, 1, END // NS, END % NS + 1, ((10_000, 3),))
    assert finish(state(late), bk, mid, END)[1] == "after_close"


def test_file_end():
    assert file_end_ns("AAPL/AAPL_2026-01-05_34200000_57600000_message_10_proc.npy") == 57_600 * NS
    assert file_end_ns("ABT/ABT_2026-01-13_24900000_57900000_message_10_proc.npy") == 57_900 * NS


def test_no_message_may_empty_a_side():
    bk = Book()
    bk.add(ASK, 10_100, 1, 5)
    bk.add(BID, 10_000, 2, 5)
    bk.add(BID, 9_900, 3, 2)
    t = T0 + 1
    take_ask = Message(4, 0, 0, 1, t // NS, t % NS, ((10_100, 5),))
    take_bids = Message(4, 1, 0, 1, t // NS, t % NS, ((10_000, 5), (9_900, 1)))
    delete_ask = Message(3, 0, 0, 1, t // NS, t % NS, ((10_100, 5),), qslot=0)
    assert empties_side(take_ask, bk) and empties_side(delete_ask, bk) and not empties_side(take_bids, bk)
    mid = mid_floor(10_100, 10_000)
    st = MessageState(T0)
    for tok in encode(take_ask, mid):
        st.push(tok)
    assert finish(st, bk, mid, END)[1] == "empties_side"
    rng = np.random.default_rng(3)
    for _ in range(200):                           # the masks never offer it either
        m, reason = finish(_sample(bk, rng)[0], bk, mid, END)
        assert reason is None and not (m.evt == 4 and m.d == 0 and m.levels[0][1] >= 5)
