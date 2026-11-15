"""newtok codec of the policy bench against nano-sigma's own encoder.

The simulator was trained on nano-sigma's `encode_window`. Any difference in
how the bench writes a message is a distribution shift the model never saw, so
`encode_stream` has to give the same tokens id for id, and `decode` has to
undo `encode` up to the documented clipping.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from lobpipeline.policy_bench import codec
from lobpipeline.policy_bench.codec import SLOT_NA, SLOT_NONE, DecodeError, Message, decode, encode
from lobpipeline.policy_bench.nanosigma.data import NA, TIME0
from lobpipeline.policy_bench.nanosigma.newtok import EOE, QSLOT0, encode_window

VENDORED = Path(codec.__file__).parent / "nanosigma"
MID = 1_000_000


def test_vendored_files_match_their_provenance_hashes():
    table = dict(re.findall(r"\| `(\w+\.py)` \| [^|]+ \| `([0-9a-f]{64})` \|",
                            (VENDORED / "PROVENANCE.md").read_text()))
    assert set(table) == {"data.py", "newtok.py", "model.py", "model_newtok.py"}
    for name, sha in table.items():
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == sha, name


def _stream(rng, n):
    """Synthetic processed rows with bursts, clipped fields and every slot code."""
    msg = np.zeros((n, 14), np.int64)
    book = np.zeros((n, 43), np.int64)
    t = 34_200 * 10**9
    for r in range(n):
        burst = r > 0 and msg[r - 1, 1] == 4 and rng.random() < 0.5
        if burst:
            msg[r, [1, 2]] = 4, msg[r - 1, 2]
            dt = 0
        else:
            msg[r, 1] = rng.integers(1, 5)
            msg[r, 2] = rng.integers(0, 2)
            dt = int(rng.choice([0, rng.integers(1, 10**6), rng.integers(1, 10**10)]))
            if rng.random() < 0.02:
                dt = int(rng.integers(1_000, 1_100)) * 10**9   # dt_s above 999 is clipped
        t += dt
        msg[r, 6], msg[r, 7] = divmod(dt, 10**9)
        msg[r, 8], msg[r, 9] = divmod(t, 10**9)
        msg[r, 3] = MID + 100 * int(rng.integers(-40, 40)) + int(rng.choice([0, 0, 0, 37]))
        msg[r, 5] = int(rng.choice([rng.integers(1, 500), 9999, 0]))
        book[r, 3] = MID + 100 * int(rng.integers(1, 4))
        book[r, 5] = MID - 100 * int(rng.integers(0, 4)) - int(rng.choice([0, 50]))
    qslot = rng.choice([-2, -1, 0, 1, 7, 98, 99, 150], size=n)
    return msg, book, qslot


@pytest.mark.parametrize("seed", range(5))
def test_encode_stream_equals_encode_window(seed):
    msg, book, qslot = _stream(np.random.default_rng(seed), 400)
    ref = encode_window(msg, book, qslot)
    got = codec.encode_stream(msg, book, qslot)
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(a, b)


def _clipped(m: Message, mid: int) -> Message:
    levels = tuple((mid + (p - mid) // 100 * 100, min(q, 9999)) for p, q in m.levels)
    q = min(m.qslot, 99) if m.qslot >= 0 else m.qslot
    return Message(m.evt, m.d, min(m.dt_s, 999), m.dt_ns, m.t_s, m.t_ns, levels, q)


@pytest.mark.parametrize("seed", range(5))
def test_decode_inverts_encode_up_to_clipping(seed):
    msg, book, qslot = _stream(np.random.default_rng(seed), 400)
    for i, j in codec.burst_bounds(msg):
        m = codec.message_from_rows(msg[i:j], int(qslot[i]))
        mid = codec.mid_floor(book[i, 3], book[i, 5])
        assert decode(encode(m, mid), mid) == _clipped(m, mid)


def _new(**kw):
    base = dict(evt=1, d=1, dt_s=0, dt_ns=5, t_s=34_200, t_ns=1, levels=((MID - 100, 10),))
    base.update(kw)
    return Message(**base)


def test_token_counts_follow_the_newtok_layout():
    assert len(encode(_new(), MID)) == 16
    assert len(encode(_new(evt=3, qslot=4), MID)) == 17
    ex = encode(_new(evt=4, levels=((MID - 100, 5), (MID - 200, 7))), MID)
    assert len(ex) == 12 + 5 * 2 and ex[-1] == EOE


def test_qslot_tokens():
    assert encode(_new(evt=2, qslot=SLOT_NA), MID)[-1] == NA
    assert encode(_new(evt=2, qslot=150), MID)[-1] == QSLOT0 + 99


@pytest.mark.parametrize("bad", [
    lambda t: t[:-1],                                   # truncated
    lambda t: np.append(t, t[-1]),                      # trailing token
    lambda t: np.concatenate([[TIME0], t[1:]]),         # event slot holds a time digit
])
def test_decode_rejects_malformed_new_orders(bad):
    with pytest.raises(DecodeError):
        decode(bad(encode(_new(), MID)), MID)


def test_decode_rejects_execution_without_eoe_or_levels():
    ex = encode(_new(evt=4), MID)
    with pytest.raises(DecodeError):
        decode(ex[:-1], MID)
    with pytest.raises(DecodeError):
        decode(np.append(ex[:11], EOE), MID)


def test_decoded_message_without_qslot_is_slot_none():
    assert decode(encode(_new(), MID), MID).qslot == SLOT_NONE
