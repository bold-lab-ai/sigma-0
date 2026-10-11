"""Replay one frozen episode with a market-making policy in the loop.

Each decision step does the same four things:

    read the book        -> build the observation the policy sees
    ask the policy       -> cancel / post at the touch
    hand both streams    -> [my cancels, my quotes, background slice]
      to the engine         in that order, through one scan
    read what happened   -> fills attributed to my trader id, new book

The ordering inside a step is a modelling choice, not an accident. Cancels go
first so a policy that re-quotes does not race its own resting order; my new
quotes go before the background slice so that posting at the touch actually
earns queue priority against flow that arrives afterwards. Putting my quotes
last would silently make every passive fill impossible.

What is NOT modelled: my own market impact. The background flow was generated
without knowledge of my quotes and is replayed unchanged, so other participants
never react to me. That is the known reason simulated PnL runs optimistic, and
it is why the same replay against the real continuation is the control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from post_training.heuristic_learning.episode_builder import (
    AGENT_TRADER_ID,
    mid_from_book,
)


TICK = 100  # LOBSTER prices are in 1/100 cent


def _empty_ask_sentinel(cfg) -> int:
    """The value `get_L2_state` writes into an ask level that does not exist.

    Empty ASK levels come back as `cfg.maxint`; the bid side uses -1
    (JaxOrderBookArrays.py:283). A `price <= 0` guard therefore does not detect a
    one-sided book on the ask side, and a policy would happily quote against a
    price of two billion.

    This reads the engine's own constant rather than hard-coding a threshold.
    An earlier version compared against 2**30, which worked only by coincidence:
    `MaxInt._64_Bit_Signed` is misnamed and actually holds 2**31-1, so the
    comparison happened to land on the right side of it. Change that enum to a
    genuinely 64-bit value and the hard-coded version still works; shrink it
    below 2**30 and the guard silently stops firing.
    """
    return int(cfg.maxint)


@dataclass
class SimConfig:
    n_levels: int = 10
    n_steps: int = 60                 # decision points across the window
    warmup_s: float = 1.0             # no quoting for the first second: the
                                      # formulation's latency constraint, since
                                      # inference over a 1-min window takes time
    size_cap: int = 10                # max shares per quote
    inventory_cap: int = 50           # |inventory| beyond this forces abstain
    unwind_penalty_ticks: int = 1     # crossing cost when flattening at the end


@dataclass
class StepRecord:
    t_ns: int
    mid: float
    best_bid: int
    best_ask: int
    inventory: int
    cash: float
    action: Dict[str, Any]
    fills: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Trajectory:
    steps: List[StepRecord]
    realized_pnl: float
    final_inventory: int
    unwind_pnl: float
    total_pnl: float
    n_fills: int
    error: Optional[str] = None


def _engine_msg(event_type: int, side: int, qty: int, price: int,
                order_id: int, t_s: int, t_ns: int) -> np.ndarray:
    """One engine message: [type, side, quant, price, order_id, trader_id, s, ns].

    `side` is already the engine's {-1, 1}; it is NOT re-mapped here. The CSV
    writer and the engine's own msg_to_jnp both land on {-1, 1}, so a second
    mapping would produce {-3, 1}, which the engine accepts and then matches
    against a side that does not exist.
    """
    return np.array([event_type, side, qty, price, order_id, AGENT_TRADER_ID,
                     t_s, t_ns], dtype=np.int32)


def build_observation(book: np.ndarray, t_ns: int, window_end_ns: int,
                      inventory: int, cash: float, realized: float,
                      my_bid: Optional[int], my_ask: Optional[int],
                      fills: List[Dict[str, Any]],
                      signal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    best_ask, ask_sz, best_bid, bid_sz = (int(book[0]), int(book[1]),
                                          int(book[2]), int(book[3]))
    return {
        "time_left_s": max(0.0, (window_end_ns - t_ns) / 1e9),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": (best_ask + best_bid) / 2.0,
        "spread_ticks": (best_ask - best_bid) / TICK,
        "bid_sizes": book[3::4].tolist(),
        "ask_sizes": book[1::4].tolist(),
        "inventory": inventory,
        "cash": cash,
        "realized_pnl": realized,
        "my_bid_px": my_bid,
        "my_ask_px": my_ask,
        "fills_since_last": fills,
        "signal": signal or {},
    }


def baseline_symmetric_touch(obs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Quote both sides at the touch, always, subject only to the hard caps.

    This is the baseline every candidate is scored against, and it is
    deliberately the dumbest thing that still counts as market making: it never
    reads the signal, never skews for inventory, never cancels early. A
    candidate that cannot beat it has not learned to use the signal at all, so
    a baseline-relative delta is a cleaner thing to hand an optimizer than an
    absolute PnL whose scale is a property of the ticker.
    """
    inv = obs["inventory"]
    cap = state.get("inventory_cap", 50)
    return {
        "post_bid": inv < cap,
        "post_ask": inv > -cap,
        "cancel_bid": True,
        "cancel_ask": True,
        "size": state.get("size", 10),
    }


def run_episode(policy_fn: Callable[[Dict, Dict], Dict],
                episode: Dict[str, Any],
                cfg: SimConfig = SimConfig(),
                policy_state: Optional[Dict[str, Any]] = None) -> Trajectory:
    """Replay one episode with `policy_fn` quoting into it."""
    import jax.numpy as jnp
    from matching_engine.jaxob.jaxob_config import JAXLOB_Configuration
    from matching_engine.jaxob.jorderbook import LobState, OrderBook

    background = np.asarray(episode["background"], dtype=np.int32)
    init_book = np.asarray(episode["init_book"], dtype=np.int32)
    n_levels = init_book.shape[0] // 4

    t_all = background[:, 6].astype(np.int64) * 1_000_000_000 + background[:, 7].astype(np.int64)
    t0, t_end = int(t_all[0]), int(t_all[-1])
    # Decision times are evenly spaced in wall-clock, not in message count: a
    # market maker's cadence is set by the clock, while message density varies.
    bounds = np.linspace(t0 + int(cfg.warmup_s * 1e9), t_end, cfg.n_steps + 1).astype(np.int64)

    engine_cfg = JAXLOB_Configuration()
    empty_ask = _empty_ask_sentinel(engine_cfg)
    sim = OrderBook(cfg=engine_cfg)
    state = sim.reset(jnp.asarray(init_book, dtype=jnp.int32),
                      jnp.array([int(background[0, 6]), 0], dtype=jnp.int32))

    pstate: Dict[str, Any] = dict(policy_state or {})
    pstate.setdefault("size", cfg.size_cap)
    pstate.setdefault("inventory_cap", cfg.inventory_cap)

    inventory, cash, realized = 0, 0.0, 0.0
    # Live agent orders: oid -> {side, px, qty}. Needed for three separate
    # reasons: to recover a fill's direction, to avoid cancelling an order the
    # book no longer holds, and to know whether a quote already sits where the
    # policy wants it.
    live: Dict[int, Dict[str, int]] = {}
    my_bid_oid = my_ask_oid = None
    next_oid = -1000                      # negative ids mark agent orders
    steps: List[StepRecord] = []
    cursor = int(np.searchsorted(t_all, bounds[0], side="left"))

    # Padding to a fixed width: msgs.shape[0] is part of the jit signature, so a
    # a different slice length every step would recompile. Rounding to a grid also lets
    # different episodes share compilations.
    slice_lens = np.diff(np.searchsorted(t_all, bounds, side="left"))
    pad_to = int(np.ceil((int(slice_lens.max(initial=0)) + 8) / 64.0) * 64)
    n_rejected = 0

    def _push(st, msgs_np):
        """One batched engine call, with this step's trades isolated.

        Resetting `trades` to all -1 first is not an optimisation. The engine's
        trade buffer does not wrap: once `nTrades` rows are used, `match_order`'s
        empty-slot search returns -1 and every later trade is written to the LAST
        row, destroying its predecessor. Reading a cumulative buffer therefore
        silently loses fills part-way through a window. The production env does
        the same reset for the same reason (marl_env.py:347).
        """
        if len(msgs_np) == 0:
            return st
        block = np.zeros((pad_to, 8), dtype=np.int32)      # [0]*8 is the engine no-op
        block[:len(msgs_np)] = msgs_np
        fresh = jnp.full_like(st.trades, -1)
        st = LobState(st.asks, st.bids, fresh, st.key)
        return sim.process_orders_array(st, jnp.asarray(block, dtype=jnp.int32))

    # Warm-up: background only, no quoting yet. Pushed in pad_to-sized chunks so
    # it reuses the same compiled shape as the decision steps rather than
    # forcing a second, much wider one.
    for i in range(0, cursor, pad_to):
        state = _push(state, background[i:min(i + pad_to, cursor)])

    for k in range(cfg.n_steps):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        book = np.asarray(sim.get_L2_state(state, n_levels))
        best_ask, best_bid = int(book[0]), int(book[2])
        # Empty ask levels come back as cfg.maxint, not -1, so a `<= 0` guard
        # alone would treat a one-sided book as tradeable.
        if best_bid <= 0 or best_ask <= 0 or best_ask >= empty_ask:
            continue

        t_s, t_ns_part = int(lo // 1_000_000_000), int(lo % 1_000_000_000)
        prev_fills = steps[-1].fills if steps else []
        obs = build_observation(book, lo, t_end, inventory, cash, realized,
                                live.get(my_bid_oid, {}).get("px"),
                                live.get(my_ask_oid, {}).get("px"),
                                prev_fills, episode.get("signal"))
        try:
            action = policy_fn(obs, pstate) or {}
        except Exception as exc:                          # a bad policy must not
            return Trajectory(steps, realized, inventory, 0.0, realized,        # kill the harness
                              sum(len(s.fills) for s in steps),
                              error=f"policy raised at step {k}: {exc!r}")

        size = int(min(abs(int(action.get("size", 0))), cfg.size_cap))
        outgoing: List[np.ndarray] = []

        # A cancel is only sent for an order the book still holds. Cancelling a
        # dead id is not a harmless no-op: the engine falls back to
        # get_init_id_match, which strips `size` shares from a book-init order at
        # that price -- i.e. it eats the very queue the agent is standing in.
        def _cancel(oid_attr, want, side):
            oid = oid_attr
            if not want or oid is None or oid not in live:
                return oid
            o = live[oid]
            outgoing.append(_engine_msg(2, side, o["qty"], o["px"], oid, t_s, t_ns_part))
            del live[oid]
            return None

        # Keeping an unchanged quote alive is what makes "market making at the
        # touch" mean anything: the engine preserves (price, time) priority
        # across partial fills, so a quote that is cancelled and re-posted every
        # step is permanently at the back of the queue and can never accrue
        # position. Only re-quote when the price actually has to move.
        want_bid = bool(action.get("post_bid")) and size > 0 and inventory < cfg.inventory_cap
        want_ask = bool(action.get("post_ask")) and size > 0 and inventory > -cfg.inventory_cap
        bid_ok = my_bid_oid in live and live.get(my_bid_oid, {}).get("px") == best_bid
        ask_ok = my_ask_oid in live and live.get(my_ask_oid, {}).get("px") == best_ask

        my_bid_oid = _cancel(my_bid_oid, action.get("cancel_bid") and not (want_bid and bid_ok), 1)
        my_ask_oid = _cancel(my_ask_oid, action.get("cancel_ask") and not (want_ask and ask_ok), -1)

        if want_bid and my_bid_oid is None:
            next_oid -= 1
            my_bid_oid = next_oid
            live[next_oid] = {"side": 1, "px": best_bid, "qty": size}
            outgoing.append(_engine_msg(1, 1, size, best_bid, next_oid, t_s, t_ns_part))
        if want_ask and my_ask_oid is None:
            next_oid -= 1
            my_ask_oid = next_oid
            live[next_oid] = {"side": -1, "px": best_ask, "qty": size}
            outgoing.append(_engine_msg(1, -1, size, best_ask, next_oid, t_s, t_ns_part))

        # One batched call per step: [my cancels, my quotes, background slice].
        # My quotes precede the background slice so posting at the touch actually
        # earns priority against flow arriving afterwards.
        nxt = int(np.searchsorted(t_all, hi, side="left"))
        step_msgs = np.concatenate(
            [np.asarray(outgoing, dtype=np.int32).reshape(-1, 8), background[cursor:nxt]]
        ) if outgoing else background[cursor:nxt]
        if len(step_msgs) > pad_to:
            n_rejected += len(step_msgs) - pad_to
            step_msgs = step_msgs[:pad_to]
        state = _push(state, step_msgs)
        cursor = nxt

        fills = _collect_fills(state.trades, live)
        for f in fills:
            signed = f["qty"] if f["side"] > 0 else -f["qty"]
            inventory += signed
            cash -= signed * f["px"]
            realized = cash + inventory * ((best_ask + best_bid) / 2.0)

        steps.append(StepRecord(t_ns=lo, mid=(best_ask + best_bid) / 2.0,
                                best_bid=best_bid, best_ask=best_ask,
                                inventory=inventory, cash=cash,
                                action=dict(action), fills=fills))

    # End of window: flatten. Inventory risk is not penalised by a lambda term;
    # it shows up here, as the cost of getting out at the far touch.
    final_book = np.asarray(sim.get_L2_state(state, n_levels))
    fa, fb = int(final_book[0]), int(final_book[2])
    final_mid = (fa + fb) / 2.0 if fa > 0 and fb > 0 else (steps[-1].mid if steps else 0.0)
    if inventory > 0:
        unwind_px = final_mid - cfg.unwind_penalty_ticks * TICK
    elif inventory < 0:
        unwind_px = final_mid + cfg.unwind_penalty_ticks * TICK
    else:
        unwind_px = final_mid
    unwind_pnl = inventory * unwind_px
    total = cash + unwind_pnl

    return Trajectory(steps=steps, realized_pnl=cash, final_inventory=inventory,
                      unwind_pnl=unwind_pnl, total_pnl=total,
                      n_fills=sum(len(s.fills) for s in steps))


def _collect_fills(trades, live: Dict[int, Dict[str, int]]) -> List[Dict[str, Any]]:
    """Our fills from a trades buffer that holds only this step's trades.

    No de-duplication happens here, and that is deliberate. `run_episode` hands
    the engine a freshly blanked trades array each step, so everything present
    belongs to the step just processed. De-duplicating a cumulative buffer
    instead would look like it worked while silently losing fills, because the
    buffer overwrites its last row once full rather than wrapping.

    Attribution is delegated to the engine's `get_agent_trades`, which masks on
    the two trader-id columns (`PASS_TID` for the resting side, `AGRS_TID` for
    the aggressor). Scanning the whole row for our id would also match an order
    id that happens to equal it.

    Direction is recovered from the order id, not from the quantity: a trade row
    carries an unsigned quantity, so its sign says nothing about whether we
    bought or sold. A fill on an order we posted as a bid is a buy.
    """
    import numpy as _np
    import matching_engine.jaxob.JaxOrderBookArrays as job
    from matching_engine.jaxob.jaxob_constants import TradesFeat as TF

    agent_trades, _ = job.get_agent_trades(trades, AGENT_TRADER_ID)
    out: List[Dict[str, Any]] = []
    for row in _np.asarray(agent_trades):
        px, qty = int(row[TF.P.value]), int(row[TF.Q.value])
        if px <= 0 or qty == 0:                                # blanked slot
            continue
        pass_oid, agrs_oid = int(row[TF.PASS_OID.value]), int(row[TF.AGRS_OID.value])
        our_oid = pass_oid if pass_oid in live else (agrs_oid if agrs_oid in live else None)
        if our_oid is None:
            continue
        rec = live[our_oid]
        filled = min(abs(qty), rec["qty"])
        rec["qty"] -= filled
        if rec["qty"] <= 0:
            del live[our_oid]                                  # fully filled: no longer cancellable
        out.append({"px": px, "qty": filled, "side": rec["side"], "oid": our_oid})
    return out
