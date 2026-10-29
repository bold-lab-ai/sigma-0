import numpy as np
import pytest

torch = pytest.importorskip("torch")

from post_training.policy_bench.book import ASK, BID, Book, EMPTY_ASK_PRICE, L10Writer  # noqa: E402
from post_training.policy_bench.nanosigma.data import D_BOOK, START  # noqa: E402
from post_training.policy_bench.nanosigma.model_newtok import NanoSigmaNewTok  # noqa: E402
from post_training.policy_bench.nanosigma.newtok import VOCAB_NEW  # noqa: E402
from post_training.policy_bench.simulator import KVSimulator  # noqa: E402


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
