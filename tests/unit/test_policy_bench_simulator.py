import numpy as np
import pytest

torch = pytest.importorskip("torch")

from post_training.policy_bench.book import ASK, BID, Book, EMPTY_ASK_PRICE, L10Writer  # noqa: E402
from post_training.policy_bench.nanosigma.data import D_BOOK, START  # noqa: E402
from post_training.policy_bench.nanosigma.model_newtok import NanoSigmaNewTok  # noqa: E402
from post_training.policy_bench.nanosigma.newtok import VOCAB_NEW  # noqa: E402
from post_training.policy_bench.simulator import BatchKVSimulator, KVSimulator  # noqa: E402


def _window(T=60, M=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randint(4, VOCAB_NEW, (T,), generator=g)
    mi = torch.sort(torch.randint(0, M, (T,), generator=g)).values
    book = torch.randn(M, D_BOOK, generator=g)
    inp = torch.cat([torch.tensor([START]), tok[:-1]])
    return inp, mi, book


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return NanoSigmaNewTok(d=64, n_heads=4, n_fused=2).eval()


def test_kv_cache_equals_full_forward(model):
    inp, mi, book = _window()
    with torch.no_grad():
        full = model(inp[None], mi[None], book[None])[0]
    sim = KVSimulator(model, max_len=128, dtype=torch.float32)
    out = [sim.feed(inp[:20], book[mi[:20]])]
    for k in range(20, 30):                                    # one token at a time
        out.append(sim.feed(inp[k:k + 1], book[mi[k:k + 1]]))
    out.append(sim.feed(inp[30:], book[mi[30:]]))              # a chunk after the cache
    torch.testing.assert_close(torch.cat(out), full, rtol=1e-5, atol=1e-5)


def test_truncate_rolls_back(model):
    inp, mi, book = _window(seed=1)
    with torch.no_grad():
        full = model(inp[None], mi[None], book[None])[0]
    sim = KVSimulator(model, max_len=128, dtype=torch.float32)
    sim.feed(inp[:40], book[mi[:40]])
    sim.truncate(25)
    junk = torch.randint(4, VOCAB_NEW, (9,))
    sim.feed(junk, book[mi[25:34]])                            # a rejected message
    sim.truncate(25)
    torch.testing.assert_close(sim.feed(inp[25:], book[mi[25:]]), full[25:], rtol=1e-5, atol=1e-5)


def test_feed_past_max_len_raises(model):
    inp, mi, book = _window(T=10)
    sim = KVSimulator(model, max_len=8, dtype=torch.float32)
    with pytest.raises(ValueError):
        sim.feed(inp, book[mi])


def test_batch_step_equals_each_rows_full_forward(model):
    wins = [_window(seed=10 + b) for b in range(3)]
    with torch.no_grad():
        full = [model(inp[None], mi[None], book[None])[0] for inp, mi, book in wins]
    sim = BatchKVSimulator(model, batch=3, max_len=128, dtype=torch.float32)
    starts = [12, 25, 18]                                      # rows at different lengths
    got = [[sim.prefill(b, inp[:s], book[mi[:s]])] for b, ((inp, mi, book), s) in enumerate(zip(wins, starts))]
    pos = list(starts)
    for t in range(14):
        active = [True, t % 3 != 1, True]                      # row 1 sits out some steps
        tok = torch.stack([w[0][p] for w, p in zip(wins, pos)])
        bk = torch.stack([w[2][w[1][p]] for w, p in zip(wins, pos)])
        out = sim.step(tok, bk, active)
        for b in range(3):
            if active[b]:
                got[b].append(out[b:b + 1])
                pos[b] += 1
    assert sim.n == pos
    for b in range(3):
        torch.testing.assert_close(torch.cat(got[b]), full[b][:pos[b]], rtol=1e-5, atol=1e-5)


def test_batch_truncate_rolls_back_one_row(model):
    wins = [_window(seed=20 + b) for b in range(2)]
    with torch.no_grad():
        full = [model(inp[None], mi[None], book[None])[0] for inp, mi, book in wins]
    sim = BatchKVSimulator(model, batch=2, max_len=128, dtype=torch.float32)
    for b, (inp, mi, book) in enumerate(wins):
        sim.prefill(b, inp[:30], book[mi[:30]])
    junk = torch.randint(4, VOCAB_NEW, (2,))
    sim.step(junk, torch.randn(2, D_BOOK))                     # row 0's token is rejected
    sim.truncate(0, 30)
    nxt = [30, 31]
    tok = torch.stack([w[0][p] for w, p in zip(wins, nxt)])
    tok[1] = junk[1]                                           # row 1 keeps its junk token
    out = sim.step(tok, torch.stack([wins[0][2][wins[0][1][30]], torch.zeros(D_BOOK)]))
    torch.testing.assert_close(out[0], full[0][30], rtol=1e-5, atol=1e-5)
    assert sim.n == [31, 32]
    with pytest.raises(ValueError):
        sim.truncate(1, 40)


def test_batch_step_on_a_full_row_raises(model):
    sim = BatchKVSimulator(model, batch=2, max_len=4, dtype=torch.float32)
    inp, mi, book = _window(T=4)
    sim.prefill(0, inp, book[mi])
    with pytest.raises(ValueError):
        sim.step(inp[:2], book[mi[:2]])
    sim.step(inp[:2], book[mi[:2]], active=[False, True])     # the full row sits out


def _row(ask, bid, t=(0, 0)):
    r = np.zeros(43, np.int64)
    r[1:3] = t
    r[3], r[5] = ask, bid
    return r


def test_l10_writer_columns():
    bk = Book()
    bk.add(ASK, 10_100, 1, 5)
    bk.add(BID, 10_000, 2, 7)
    w = L10Writer(_row(10_100, 10_000))
    r = w.next_row(bk, 34_200, 17)
    assert list(r[:3]) == [0, 34_200, 17] and np.array_equal(r[3:], bk.l10())
    bk.remove_at(ASK, 10_100, 0)
    bk.add(ASK, 10_300, 3, 1)       # mid 100.5 -> 101.5 ticks; half to even: 100 -> 102
    assert w.next_row(bk, 1, 2)[0] == 2


def test_l10_writer_carries_an_empty_side_forward():
    bk = Book()
    bk.add(ASK, 10_200, 1, 5)
    bk.add(BID, 10_000, 2, 7)
    w = L10Writer(_row(10_200, 10_000))
    bk.remove_at(BID, 10_000, 0)                                # bid side empties
    r = w.next_row(bk, 0, 0)
    assert r[5] == -9_999_999_999 and r[0] == 0                 # last bid carried: mid unchanged
    bk.add(BID, 10_200 - 400, 3, 1)
    assert w.next_row(bk, 0, 0)[0] == -1


def test_l10_writer_unknown_side_gives_zero():
    bk = Book()
    bk.add(BID, 10_000, 2, 7)
    w = L10Writer(_row(EMPTY_ASK_PRICE, 10_000))
    assert w.next_row(bk, 0, 0)[0] == 0
    bk.add(ASK, 10_200, 1, 1)
    assert w.next_row(bk, 0, 0)[0] == 0                         # first known mid
