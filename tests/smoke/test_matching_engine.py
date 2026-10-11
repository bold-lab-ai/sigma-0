"""Smoke test for the migrated JAX-LOB matching engine (``matching_engine.jaxob``).

This verifies the CORE order-book matching path actually *runs and is numerically
correct*, not merely that the package imports (which ``tests/unit/test_imports.py``
already covers via an empty ``jaxob/__init__.py``).

Login-node safety
-----------------
The Isambard login nodes impose a cgroup-v2 ``pids.max=500`` per-user limit.
JAX's XLA CPU backend sizes its thread pools to the full visible core count
(144 on a login node) and creates several such pools (Eigen intra-op, the
compile-time constant-folding TSL pool, the PjRt async pool, ...). Together they
exceed 500 PIDs and XLA aborts with ``pthread_create ... EAGAIN`` (SIGABRT).
We cap CPU affinity to <=4 cores *before JAX initialises* so XLA spawns only a
handful of threads. This keeps the test self-contained: a plain ``pytest`` run on
a login node will not trigger the thread storm.
"""
from __future__ import annotations

import os

# --- Login-node thread-storm guard: cap affinity BEFORE importing jax. ---
try:  # pragma: no cover - platform dependent
    _cpus = sorted(os.sched_getaffinity(0))
    if len(_cpus) > 4:
        os.sched_setaffinity(0, set(_cpus[:4]))
except (AttributeError, OSError):
    pass  # non-Linux or not permitted; harmless.

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # force CPU; skip GPU plugin on login node.

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")  # skip cleanly when JAX is absent (e.g. base home env).
pytest.importorskip("chex")

from matching_engine.jaxob.jorderbook import OrderBook

# L2 snapshot: alternating (ask_price, ask_qty, bid_price, bid_qty) per level.
L2_INIT = [354200, 452, 350100, 89, 361200, 100, 344000, 400, 362900, 100,
           343100, 100, 364000, 400, 338700, 100, 371900, 1100, 337100, 1000]
BEST_ASK, BEST_ASK_QTY = 354200, 452
BEST_BID, BEST_BID_QTY = 350100, 89


@pytest.fixture(scope="module")
def book_and_state():
    ob = OrderBook()
    state = ob.reset(jnp.array(L2_INIT))
    return ob, state


def test_reset_reports_correct_top_of_book(book_and_state):
    ob, state = book_and_state
    assert int(ob.get_best_ask(state)) == BEST_ASK
    assert int(ob.get_best_bid(state)) == BEST_BID
    ask, bid = ob.get_best_bid_and_ask_inclQuants(state)
    assert [int(x) for x in np.asarray(ask)] == [BEST_ASK, BEST_ASK_QTY]
    assert int(np.asarray(bid)[0]) == BEST_BID


def test_l2_snapshot_shape_and_values(book_and_state):
    ob, state = book_and_state
    n = 4
    l2 = np.asarray(ob.get_L2_state(state, n))
    assert l2.shape == (4 * n,)  # ask_p, ask_q, bid_p, bid_q per level.
    assert int(l2[0]) == BEST_ASK and int(l2[1]) == BEST_ASK_QTY
    assert int(l2[2]) == BEST_BID


def test_crossing_bid_matches_ask_records_trade_and_consumes_book(book_and_state):
    ob, state = book_and_state
    # Aggressive crossing BID:
    #   [type=1 limit, side=1 bid, qty=100, price=355000>=best_ask, trade_id, order_id, sec, ns]
    msg = jnp.array([1, 1, 100, 355000, 7777, 7777, 3402, 1000], dtype=jnp.int32)
    new = ob.process_order_array(state, msg)

    # Book consumed: best ask price unchanged, resting qty drops 452 -> 352.
    assert int(ob.get_best_ask(new)) == BEST_ASK
    assert int(ob.get_volume_at_price(new, 0, BEST_ASK)) == BEST_ASK_QTY - 100

    # Exactly one filled trade row, at the passive ask price, magnitude 100, our id recorded.
    trades = np.asarray(new.trades)
    filled = trades[(trades != -1).any(axis=1)]
    assert filled.shape[0] == 1
    row = filled[0]
    assert int(row[0]) == BEST_ASK        # trade price == passive (resting ask) price.
    assert abs(int(row[1])) == 100        # qty is signed in this engine; magnitude == matched.
    assert 7777 in (int(row[3]), int(row[7]))  # aggressor order/trade id recorded.


def test_non_crossing_bid_rests_without_trade(book_and_state):
    ob, state = book_and_state
    # Passive BID well below the best ask -> no match, no trade, best bid improves to 351000.
    msg = jnp.array([1, 1, 50, 351000, 6666, 6666, 3403, 2000], dtype=jnp.int32)
    new = ob.process_order_array(state, msg)
    assert int(ob.get_best_bid(new)) == 351000
    trades = np.asarray(new.trades)
    assert int((trades != -1).any(axis=1).sum()) == 0  # no fills.


def test_get_next_executable_order_returns_top_of_book(book_and_state):
    # Was xfail: the OrderBook wrapper used a plain @jax.jit, so `side` reached the
    # inner static_argnums function as a tracer and `if side == 0` raised
    # TracerBoolConversionError. Fixed by marking `side` static (static_argnums=(2,))
    # in the wrapper. side 0 -> top ASK, side 1 -> top BID; row = [price, qty, oid, tid, sec, ns].
    ob, state = book_and_state
    top_ask = np.asarray(ob.get_next_executable_order(state, 0))
    top_bid = np.asarray(ob.get_next_executable_order(state, 1))
    assert int(top_ask[0]) == BEST_ASK and int(top_ask[1]) == BEST_ASK_QTY
    assert int(top_bid[0]) == BEST_BID and int(top_bid[1]) == BEST_BID_QTY
