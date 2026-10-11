from jax import config
config.update("jax_disable_jit", False)
#config.update("jax_disable_jit", True)

from datetime import datetime
import functools
from glob import glob
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.nn import one_hot
import flax.linen as nn
from flax.training.train_state import TrainState
from base_model.training import train_helpers
import numpy as onp
import os
import sys
import pandas as pd
import pickle
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
from base_model.utils.log import debug, info
import math
import time

import base_model.inference.validation as valh
import base_model.inference.evaluation as eval
import base_model.data.preproc as preproc
# from preproc import transform_L2_state_gpu
import base_model.encoding.encoding as encoding
from base_model.encoding.encoding import Message_Tokenizer, Vocab
from base_model.data.lobster_dataloader import LOBSTER_Dataset
# Single source of truth for file ordering; see lob/data_files.py.
from base_model.data.data_files import resolve_data_files
from base_model.data.wide_book import select_wide_book_levels
from base_model.encoding.encoding_1tok import (
    local_to_global_jax, global_to_local_jax, N_FIELDS, N_SPECIAL_TOKENS,
)
import chex

from matching_engine.jaxob.jorderbook import OrderBook, LobState
from matching_engine.jaxob.jaxob_config import JAXLOB_Configuration
import matching_engine.jaxob.jaxob_constants as cst
import matching_engine.jaxob.JaxOrderBookArrays as job
# from gym_exchange.environment.base_env.assets.action import OrderIdGenerator


# lazy for the same reason as train_helpers: the tokenizer is switchable
def __getattr__(name):
    if name == "REF_LEN":
        return Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# indices for DECODED message fields
ORDER_ID_i = 0
EVENT_TYPE_i = 1
DIRECTION_i = 2
PRICE_ABS_i = 3
PRICE_i = 4
SIZE_i = 5
DTs_i = 6
DTns_i = 7
TIMEs_i = 8
TIMEns_i = 9
PRICE_REF_i = 10
SIZE_REF_i = 11
TIMEs_REF_i = 12
TIMEns_REF_i = 13

l2_state_n = 10

# ENCODED TOKEN INDICES
# time tokens aren't generated but calculated using delta_t
# hence, skip generation from TIME_START_I (inclusive) to TIME_END_I (exclusive)
TIME_START_I, _ = valh.get_idx_from_field('time_s')
_, TIME_END_I = valh.get_idx_from_field('time_ns')

# @jax.jit
# def init_msgs_from_l2(book: Union[pd.Series, onp.ndarray]) -> jnp.ndarray:
#     """"""
#     orderbookLevels = len(book) // 4  # price/quantity for bid/ask
#     data = jnp.array(book).reshape(int(orderbookLevels*2),2)
#     newarr = jnp.zeros((int(orderbookLevels*2),8))
#     initOB = newarr \
#         .at[:,3].set(data[:,0]) \
#         .at[:,2].set(data[:,1]) \
#         .at[:,0].set(1) \
#         .at[0:orderbookLevels*4:2,1].set(-1) \
#         .at[1:orderbookLevels*4:2,1].set(1) \
#         .at[:,4].set(0) \
#         .at[:,5].set(job.INITID) \
#         .at[:,6].set(34200) \
#         .at[:,7].set(0).astype('int32')
#     return initOB


def df_msgs_to_jnp(m_df: pd.DataFrame) -> jnp.ndarray:
    """"""
    m_df = m_df.copy()
    cols = ['Time', 'Type', 'OrderID', 'Quantity', 'Price', 'Side']
    if m_df.shape[1] == 7:
        cols += ["TradeID"]
    m_df.columns = cols
    m_df['TradeID'] = 0  #  TODO: should be TraderID for multi-agent support
    col_order=['Type','Side','Quantity','Price','TradeID','OrderID','Time']
    m_df = m_df[col_order]
    m_df = m_df[(m_df['Type'] != 6) & (m_df['Type'] != 7) & (m_df['Type'] != 5)]
    time = m_df["Time"].astype('string').str.split('.',expand=True)
    m_df[["TimeWhole","TimeDec"]] = time.astype('int32')
    m_df = m_df.drop("Time", axis=1)
    mJNP = jnp.array(m_df)
    return mJNP

@jax.jit
def msg_to_jnp(
        m_raw: jax.Array,
    ) -> jax.Array:
    """ Select only the relevant columns from the raw messages
        and rearrange for simulator.
    """
    m = m_raw.copy()

    return jnp.array([
        m[EVENT_TYPE_i],
        (m[DIRECTION_i] * 2) - 1,
        m[SIZE_i],
        m[PRICE_ABS_i],
        m[ORDER_ID_i],
        0,  # TraderID
        m[TIMEs_i],
        m[TIMEns_i],
    ])

msgs_to_jnp = jax.jit(jax.vmap(msg_to_jnp))

# # NOTE: cannot jit due to side effects --> resolve later
# @jax.jit
# def reset_orderbook(
#         b: OrderBook,
#         l2_book: Optional[Union[pd.Series, onp.ndarray]] = None,
#     ) -> OrderBook:
#     """"""
#     b.bids = b.bids.at[:].set(-1)
#     b.asks = b.asks.at[:].set(-1)
#     b.trades = b.trades.at[:].set(-1)
#     if l2_book is not None:
#         msgs = init_msgs_from_l2(l2_book)
#         # NOTE: cannot jit due to side effects --> resolve later
#         # CONTINUE HERE....
#         b.process_orders_array(msgs)
#     return b

def copy_orderbook(
        b: OrderBook
    ) -> OrderBook:
    b_copy = OrderBook(cfg=b.cfg)
    b_copy.bids = b.bids.copy()
    b_copy.asks = b.asks.copy()
    b_copy.trades = b.trades.copy()
    return b_copy

def get_sim(
        init_l2_book: jax.Array,
        replay_msgs_raw: jax.Array,
        start_time: jax.Array,
        sim: OrderBook,
        # nOrders: int = 100,
        # nTrades: int = 100
        # sim_book_levels: int,
        # sim_queue_len: int,
    ) -> Tuple[OrderBook, jax.Array]:
    """
    """

    # reset simulator : args are (nOrders, nTrades)
    #Set the ns component of the start time to 0 to ensure that init messages are before first message.
    # Only edge case is if first message and init are 0 ns - unlikely.
    start_time=start_time.at[1].set(0)
    # init simulator at the start of the sequence
    sim_state = sim.reset(init_l2_book,start_time)
    # return sim, sim_state
    # replay sequence in simulator (actual)
    # so that sim is at the same state as the model
    replay = msgs_to_jnp(replay_msgs_raw)
    sim_state = sim.process_orders_array(sim_state, replay)
    return sim_state

get_sims_vmap = jax.jit(
    jax.vmap(
        get_sim,
        in_axes=(0, 0,0,None),
        out_axes=(0),
    ),
    static_argnums=(3,)
)


def replay_real_msgs_single(sim, sim_state, real_msgs_raw, n_levels):
    """Replay a real message stream and return the L2 book state after each message.

    Ported from LOBelia (alexandre-bismuth/LOBelia @ 380b35f), where it is
    `_replay_real_msgs_single`. Renamed to a public name because callers outside
    this module need it: it is the counterfactual arm of the sim-vs-real
    comparison, i.e. the same deterministic engine driven by the real
    continuation instead of a generated one.

    Args:
        sim: OrderBook instance (only used for its jitted methods).
        sim_state: LobState to start from, typically the state returned by
            `get_sim` after replaying the conditioning window.
        real_msgs_raw: (N, ...) raw messages, same layout `msgs_to_jnp` expects.
        n_levels: book depth to extract per step.

    Returns:
        (N, 4 * n_levels) array of L2 states, one row per replayed message.
    """
    msgs_jnp = msgs_to_jnp(real_msgs_raw)

    def scan_body(carry, msg):
        state = sim.process_order_array(carry, msg)
        l2 = sim.get_L2_state(state, n_levels)
        return state, l2

    _, l2_states = jax.lax.scan(scan_body, sim_state, msgs_jnp)
    return l2_states


def compute_gt_divergence(gen_l2, real_l2, tick_size):
    """Per-step divergence between a generated book path and the real one.

    Ported from LOBelia (alexandre-bismuth/LOBelia @ 380b35f).

    L2 column layout is (ask_px, ask_vol, bid_px, bid_vol) repeated per level,
    so mid = (col0 + col2) / 2, matching the convention used by return_bench.

    Divergences are reported in ticks (mid, spread) and in shares (top-5 volume
    L1), so they stay comparable across tickers and across price levels.

    Returns:
        dict of (N,) arrays: mid_divergence, spread_divergence,
        vol_l1_ask, vol_l1_bid.
    """
    gen_best_ask = gen_l2[:, 0]
    gen_best_bid = gen_l2[:, 2]
    real_best_ask = real_l2[:, 0]
    real_best_bid = real_l2[:, 2]

    gen_mid = (gen_best_ask + gen_best_bid) / 2.0
    real_mid = (real_best_ask + real_best_bid) / 2.0
    gen_spread = gen_best_ask - gen_best_bid
    real_spread = real_best_ask - real_best_bid

    mid_divergence = jnp.abs(gen_mid - real_mid) / tick_size
    spread_divergence = jnp.abs(gen_spread - real_spread) / tick_size

    gen_ask_vols = gen_l2[:, 1::4][:, :5]
    gen_bid_vols = gen_l2[:, 3::4][:, :5]
    real_ask_vols = real_l2[:, 1::4][:, :5]
    real_bid_vols = real_l2[:, 3::4][:, :5]

    vol_l1_ask = jnp.abs(gen_ask_vols - real_ask_vols).sum(axis=1)
    vol_l1_bid = jnp.abs(gen_bid_vols - real_bid_vols).sum(axis=1)

    return {
        'mid_divergence': mid_divergence,
        'spread_divergence': spread_divergence,
        'vol_l1_ask': vol_l1_ask,
        'vol_l1_bid': vol_l1_bid,
    }


def get_dataset(
        data_dir: str,
        n_messages: int,
        n_eval_messages: int,
        *,
        n_cache_files: int = 500,
        seed: int = 42,
        book_depth: int = 500,
        day_indeces: Optional[List[int]] = None,
        limit_seq: int = math.inf,
        test_split: float = 0.1,
        wide_book_dir: Optional[str] = None,
        explicit_windows=None,
    ):
    msg_files, book_files, wide_book_files = resolve_data_files(
        data_dir,
        day_indeces=day_indeces,
        test_split=test_split,
        wide_book_dir=wide_book_dir,
    )

    ds = LOBSTER_Dataset(
        msg_files,
        n_messages=n_messages + n_eval_messages,
        mask_fn=LOBSTER_Dataset.inference_mask,
        seed=seed,
        n_cache_files=n_cache_files,
        randomize_offset=False,
        book_files=book_files,
        use_simple_book=True,
        book_transform=False,
        book_depth=book_depth,
        return_raw_msgs=True,
        inference=True, #this flag shifts the book to exclude the very first state b4 the 1st message.
        limit_seq_per_file=limit_seq,
        wide_book_files=wide_book_files,
        explicit_windows=explicit_windows,
    )
    return ds

def switch(
        condlist: Sequence[jax.Array],
        funclist: Sequence[Callable],
        operands: Any = None,
        *args, **kw
    ) -> Any:
    """ Convenience function for jax.lax.switch, assuming conditions in condlist are
        mutually exclusive cases. If an extra function is given in funclist,
        this will be applied if no condition is true.
    """
    # unroll the loop over a few args
    switch_i = sum([(i+1) * condlist[i] for i in range(len(condlist))])
    # last funclist element is the default function if no condition is true
    if len(condlist) == len(funclist) - 1:
        return jax.lax.switch(
            switch_i,
            (funclist[-1],
            *funclist[:-1]),
            *operands
        )
    elif len(condlist) == len(funclist):
        return jax.lax.switch(
            switch_i - 1,
            funclist,
            *operands
        )
    else:
        raise ValueError(f'Invalid number of conditions and functions, got {len(condlist)} and {len(funclist)}')


@jax.jit
def find_order_at_price_closest_time(
        side_array: jax.Array,    # (nOrders, 6)
        price: int,
        time_s: int,
        time_ns: int,
    ) -> jax.Array:
    """Find the active order at `price` whose timestamp is closest to
    (time_s, time_ns).  Returns the 6-element order row, or a
    NEGATIVE_RETURN_ID dummy if no active orders exist at that price.

    Column layout: 0=Price, 1=Qty, 2=OID, 3=TID, 4=Time_s, 5=Time_ns.
    Empty slots have all columns set to -1.

    Uses millisecond-precision time distance (int32-safe: max intraday
    diff ~23.4M ms, well within int32 range of ~2.1B).
    """
    # Mask: active orders at the target price (qty > 0 implies non-empty)
    price_match = (side_array[:, 0] == price) & (side_array[:, 1] > 0)

    # Time distance in milliseconds (int32-safe, no x64 dependency).
    # total_diff = (order_s - target_s) * 1000 + (order_ns - target_ns) // 1e6
    # Max |total_diff| for intraday LOB: ~23,400,000 ms ≪ 2,147,483,647
    diff_ms = (
        (side_array[:, 4] - time_s) * jnp.int32(1000)
        + (side_array[:, 5] - time_ns) // jnp.int32(1_000_000)
    )
    abs_diff = jnp.abs(diff_ms)

    # Non-matching rows get max distance so argmin ignores them
    masked_diff = jnp.where(price_match, abs_diff, jnp.iinfo(jnp.int32).max)
    best_idx = jnp.argmin(masked_diff)

    return jax.lax.cond(
        price_match.any(),
        lambda idx: side_array[idx],
        lambda idx: cst.NEGATIVE_RETURN_ID * jnp.ones((6,), dtype=jnp.int32),
        best_idx,
    )


def get_sim_msg(
        pred_msg_enc: jax.Array,
        sim: OrderBook,
        sim_state: LobState,
        mid_price: int,
        new_order_id: int,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Dict[str, Any]:
    """"""
    # decoded predicted message
    # pred_msg = tok.decode(pred_msg_enc, v).squeeze()
    msg_decoded = encoding.decode_msg(pred_msg_enc, encoder)
    # jax.debug.print('decoded predicted message: \n {}', msg_decoded)
    new_part = msg_decoded[: Message_Tokenizer.N_NEW_FIELDS]
    # ref part is not needed for the simulator logic
    # ref_part = pred_msg[Message_Tokenizer.N_NEW_FIELDS: ]

    event_type = msg_decoded[EVENT_TYPE_i]
    quantity = msg_decoded[SIZE_i]
    side = msg_decoded[DIRECTION_i]
    rel_price = msg_decoded[PRICE_i]
    delta_t_s = msg_decoded[DTs_i]
    delta_t_ns = msg_decoded[DTns_i]
    time_s = msg_decoded[TIMEs_i]
    time_ns = msg_decoded[TIMEns_i]

    rel_price_ref = msg_decoded[PRICE_REF_i]
    quantity_ref = msg_decoded[SIZE_REF_i]
    time_s_ref = msg_decoded[TIMEs_REF_i]
    time_ns_ref = msg_decoded[TIMEns_REF_i]

    p_abs = mid_price + rel_price * tick_size

    # get message for jax lob simulator
    # sim_msg = switch(
    #     (event_type == 1, (event_type == 2) | (event_type == 3), event_type == 4),
    #     (get_sim_msg_new, get_sim_msg_mod, get_sim_msg_exec, construct_dummy_sim_msg),
    #     (event_type, quantity, side, p_abs, time_s, time_ns,
    #             rel_price_ref, quantity_ref, time_s_ref, time_ns_ref,
    #             new_order_id, sim, sim_state,
    #     )
    # )
    # --- Progressive order-ID resolution for cancellations ---
    # Level 1: exact timestamp match (original behavior)
    orig_order_L1 = sim.get_order_at_time(sim_state, side, time_s_ref, time_ns_ref)
    order_id_L1 = orig_order_L1[2]  # col 2 = OID

    # Level 2: price-based closest-time match (fallback)
    side_array = jax.lax.cond(
        side == 1,
        lambda a, b: b,   # side 1 = bids
        lambda a, b: a,   # side 0 = asks
        sim_state.asks, sim_state.bids,
    )
    orig_order_L2 = find_order_at_price_closest_time(
        side_array, p_abs, time_s_ref, time_ns_ref,
    )
    order_id_L2 = orig_order_L2[2]

    # Use L1 if it succeeded, otherwise fall back to L2
    order_id_ref = jnp.where(
        order_id_L1 != cst.NEGATIVE_RETURN_ID,
        order_id_L1,
        order_id_L2,
    )

    # Cancel/delete (type 2/3): use resolved ref ID; otherwise new_order_id
    order_id = jax.lax.cond(
        (event_type == 2) | (event_type == 3),
        lambda new_id, ref_id: ref_id,
        lambda new_id, ref_id: new_id,
        new_order_id, order_id_ref,
    )
    # jax.debug.print("{}",order_id)

    sim_msg = construct_sim_msg(
        event_type,  # type: execution
        side,  # side of execution
        quantity,
        p_abs,
        order_id,
        time_s,
        time_ns,
    )

    msg_decoded = msg_decoded.at[PRICE_ABS_i].set(p_abs) \
                             .at[ORDER_ID_i].set(order_id)

    # --- provenance (D-R5) ---------------------------------------------------
    # How this message's order id was arrived at, as a bit field, so the record
    # says what it did rather than leaving a reader to infer it from an id that
    # looks like any other. One integer answers three registered defects:
    #
    #   bits 0-1  resolution: 0 not a cancel, 1 exact-timestamp hit,
    #             2 price fallback, 3 total miss (sentinel written)   D-R5
    #   bit 2 (4) the NaN branch fired: a dummy went to the simulator D-R1
    #   bit 3 (8) rel_price_ref disagrees with the message's own price D-R4
    #
    # Bit 3 is what finally uses `rel_price_ref` and `quantity_ref`. They were
    # decoded and discarded, so the model's own prediction about the referenced
    # order was thrown away instead of checked against the message that claims
    # to cancel it. It is a check rather than an input on purpose: making the
    # lookup follow the ref price would change the generated market, and that
    # is a separate decision needing its own evidence.
    is_cd = (event_type == 2) | (event_type == 3)
    resolution = jnp.where(
        ~is_cd, 0,
        jnp.where(order_id_L1 != cst.NEGATIVE_RETURN_ID, 1,
                  jnp.where(order_id_L2 != cst.NEGATIVE_RETURN_ID, 2, 3)))
    nan_fired = jnp.where(jnp.isnan(new_part).any(), 4, 0)
    # `rel_price != rel_price_ref` rather than comparing the absolute prices.
    # The absolute form adds the same mid and the same tick to both sides, so
    # they cancel; writing it this way makes the comparison literally identical
    # to the one the real-message control runs, instead of merely equivalent to
    # it. A control whose arithmetic differs from the thing it controls for is
    # not a control.
    ref_disagrees = jnp.where(
        is_cd & ~jnp.isnan(rel_price_ref) & ~jnp.isnan(rel_price)
        & (rel_price != rel_price_ref), 8, 0)
    provenance = (resolution + nan_fired + ref_disagrees).astype(jnp.int32)

    # return dummy message instead if new_part contains NaNs
    sim_msg, msg_decoded = jax.lax.cond(
        jnp.isnan(new_part).any(),
        lambda sim_msg, msg_decoded: (construct_dummy_sim_msg(), msg_decoded),
        lambda sim_msg, msg_decoded: (sim_msg, msg_decoded),
        sim_msg, msg_decoded
    )
    return sim_msg, msg_decoded, provenance

# event_type, side, quantity, price,order_id,trade(r)_id, time_s, time_ns
@jax.jit
def construct_sim_msg(
        event_type: int,
        side: int,
        quantity: int,
        price: int,
        order_id: int,
        time_s: int,
        time_ns: int,
    ):
    """ NOTE: trader ID is set to 0
    """
    return jnp.array([
        event_type,
        (side * 2) - 1,
        quantity,
        price,
        order_id, # order_id
        -88,
        time_s,
        time_ns,
    ], dtype=jnp.int32)

@jax.jit
def construct_dummy_sim_msg(*args) -> jax.Array:
    return jnp.ones((8,), dtype=jnp.int32) * (-1)

@jax.jit
def construct_raw_msg(
        oid: Optional[int] = encoding.NA_VAL,
        event_type: Optional[int] = encoding.NA_VAL,
        direction: Optional[int] = encoding.NA_VAL,
        price_abs: Optional[int] = encoding.NA_VAL,
        price: Optional[int] = encoding.NA_VAL,
        size: Optional[int] = encoding.NA_VAL,
        delta_t_s: Optional[int] = encoding.NA_VAL,
        delta_t_ns: Optional[int] = encoding.NA_VAL,
        time_s: Optional[int] = encoding.NA_VAL,
        time_ns: Optional[int] = encoding.NA_VAL,
        price_ref: Optional[int] = encoding.NA_VAL,
        size_ref: Optional[int] = encoding.NA_VAL,
        time_s_ref: Optional[int] = encoding.NA_VAL,
        time_ns_ref: Optional[int] = encoding.NA_VAL,
    ):
    msg_raw = jnp.array([
        oid,
        event_type,
        direction,
        price_abs,
        price,
        size,
        delta_t_s,
        delta_t_ns,
        time_s,
        time_ns,
        price_ref,
        size_ref,
        time_s_ref,
        time_ns_ref,
    ])
    return msg_raw

@jax.jit
def rel_to_abs_price(
        p_rel: jax.Array,
        best_bid: jax.Array,
        best_ask: jax.Array,
        tick_size: int = 100,
    ) -> jax.Array:

    p_ref = (best_bid + best_ask) / 2
    p_ref = ((p_ref // tick_size) * tick_size).astype(jnp.int32)
    return p_ref + p_rel * tick_size

@jax.jit
def construct_orig_msg_enc(
        pred_msg_enc: jax.Array,
        #v: Vocab,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> jax.Array:
    """ Reconstructs encoded original message WITHOUT Delta t
        from encoded message string --> delta_t field is filled with NA_TOK
    """
    return jnp.concatenate([
        encoding.encode(jnp.array([1]), *encoder['event_type']),
        pred_msg_enc[slice(*valh.get_idx_from_field('direction'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('price_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size_ref'))],
        # NOTE: no delta_t here
        jnp.full(
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_s']] + \
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_ns']],
            Vocab.NA_TOK
        ),
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns_ref'))],
    ])

@jax.jit
def convert_msg_to_ref(
        pred_msg_enc: jax.Array,
    ) -> jax.Array:
    """ Converts encoded message to reference message part,
        i.e. (price, size, time) tokens
    """
    return jnp.concatenate([
        pred_msg_enc[slice(*valh.get_idx_from_field('price'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns'))],
    ])

def search_orig_msg(
        sim, sim_state, side, p_mod_raw, m_seq, pred_msg_enc, encoder, m_seq_raw
    ):
    vol = sim.get_volume_at_price(sim_state, side, p_mod_raw)
    ret_none = (vol==0)

    # if sim.get_volume_at_price(sim_state, side, p_mod_raw) == 0:
    #     debug('No volume at given price, discarding...')
    #     return None, None, None

    m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
    # ref part is only needed to match to an order ID
    # find original msg index location in the sequence (if it exists)
    orig_enc = construct_orig_msg_enc(pred_msg_enc, encoder)
    debug('reconstruct. orig_enc \n', orig_enc)

    sim_ids = sim.get_side_ids(sim_state, side)
    debug('sim IDs', sim_ids[sim_ids > 1])
    mask = get_invalid_ref_mask(m_seq_raw, p_mod_raw, sim_ids)
    orig_i, n_fields_removed = valh.try_find_msg(orig_enc, m_seq, mask)

    # didn't find matching original message
    if orig_i is None:
        if sim.get_volume_at_price(sim_state, side, p_mod_raw, True) == 0:
            debug('No init volume found', side, p_mod_raw)
            return None, None, None
        order_id = job.INITID
        # keep generated ref part, which we cannot validate
        orig_msg_found = orig_enc[-(Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN): ]

    # found matching original message
    else:
        # get order ID from raw data for simulator
        ORDER_ID_i = 0
        order_id = m_seq_raw[orig_i, ORDER_ID_i]
        # found original message: convert to ref part
        EVENT_TYPE_i = 1
        if m_seq_raw[orig_i, EVENT_TYPE_i] == 1:
            orig_msg_found = convert_msg_to_ref(m_seq[orig_i])
        # found reference to original message
        else:
            # take ref fields from matching message
            orig_msg_found = jnp.array(m_seq[orig_i, -(Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN): ])

    return order_id, orig_msg_found, ret_none

@jax.jit
def get_invalid_ref_mask(
        m_seq_raw: jax.Array,
        p_mod_raw: int,
        sim_ids: jax.Array
    ):
    """
    """
    PRICE_ABS_i = 3
    # filter sequence to prices matching the correct price level
    wrong_price_mask = (m_seq_raw[:, PRICE_ABS_i] != p_mod_raw)
    # filter to orders still in the book: order IDs from sim
    ORDER_ID_i = 0
    not_in_book_mask = jnp.isin(m_seq_raw[:, ORDER_ID_i], sim_ids, invert=True)
    mask = not_in_book_mask | wrong_price_mask
    return mask

@jax.jit
def add_times(
        a_s: jax.Array,
        a_ns: jax.Array,
        b_s: jax.Array,
        b_ns: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
    """ Adds two timestamps given as seconds and nanoseconds each (both fit in int32)
        and returns new timestamp, split into time_s and time_ns
    """
    a_ns = b_ns + a_ns
    extra_s = a_ns // 1000000000
    a_ns = a_ns % 1000000000
    a_s = a_s + b_s + extra_s
    return a_s, a_ns

def _get_safe_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        tick_size: int,
    ) -> int:
    """
    """
    # get current mid price from simulator
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)

    # both negative: 0 ~> (ask + bid) / 2
    # ask negative:  1 ~> bid + tick_size
    # bid negative:  2 ~> ask - tick_size
    # both negative: 3 ~> 0
    case_i = (ask <= 0) * 1 + (bid <= 0) * 2

    p_mid = jax.lax.switch(
        case_i,
        (
            lambda ask, bid: (ask + bid) // 2,
            lambda ask, bid: bid + tick_size,
            lambda ask, bid: ask - tick_size,
            lambda ask, bid: 0,
        ),
        ask, bid
    )
    # round down to next valid tick
    p_mid = (p_mid // tick_size) * tick_size
    return p_mid

@partial(jax.jit, static_argnums=(0,))
def _get_new_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        p_mid_old: jax.Array,
        tick_size: int,
    ) -> jax.Array:
    """
    """
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)
    mid = ((((ask + bid) // 2) // tick_size) * tick_size)
    return jax.lax.cond(
        (ask <= 0) | (bid <= 0),
        lambda new, old: old,
        lambda new, old: new,
        mid, p_mid_old
    )

def _add_time_tokens(
        tok_seq_A: jax.Array,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        time_init_s: int,
        time_init_ns: int,
        delta_t_s_start_i: int,
        delta_t_s_end_i: int,
        delta_t_ns_start_i: int,
        delta_t_ns_end_i: int,
    ):
    """
    """
    # TODO: simplify --> separate function
    delta_t_s_toks = tok_seq_A[delta_t_s_start_i: delta_t_s_end_i]
    delta_t_ns_toks = tok_seq_A[delta_t_ns_start_i: delta_t_ns_end_i]
    # debug('delta_t_toks', delta_t_s_toks, delta_t_ns_toks)
    delta_t_s = encoding.decode(delta_t_s_toks, *encoder['time'])
    delta_t_s = encoding.combine_field(delta_t_s, 3)
    delta_t_ns = encoding.decode(delta_t_ns_toks, *encoder['time'])
    delta_t_ns = encoding.combine_field(delta_t_ns, 3)

    # debug('delta_t', delta_t_s, delta_t_ns)
    time_s_ret, time_ns_ret = add_times(time_init_s, time_init_ns, delta_t_s, delta_t_ns)
    # debug('time', time_s, time_ns)

    # encode time and add to sequence
    time_s = encoding.split_field(time_s_ret, 2, 3)
    time_s_toks = encoding.encode(time_s, *encoder['time'])
    time_ns = encoding.split_field(time_ns_ret, 3, 3)
    time_ns_toks = encoding.encode(time_ns, *encoder['time'])

    # debug('time_toks', time_s_toks, time_ns_toks)
    time_tokens=jnp.hstack([time_s_toks, time_ns_toks])
    return time_tokens, time_s_ret, time_ns_ret


def _generate_token(
        train_state : TrainState,
        model : nn.module,
        batchnorm : bool,
        valid_mask_array : jax.Array ,
        sample_top_n : int,

        m_tok: jax.Array ,
        b_tok: jax.Array ,
        hidden: Tuple,
        token_index : int,
        rng,
    ):
    # syntactically valid tokens for current message position
    valid_mask = valh.get_valid_mask(valid_mask_array, token_index)
    # jax.debug.print("Calling apply model with token {} at index {}",m_tok,token_index)


    # TODO Turn hidden[4] to none here, and use dummy 0 for input.
    # if start_ema:
    #     hidden=hidden[:3]+(None,)

    # jax.debug.print("Start ema{}",start_ema)
    # print(start_ema)
    hidden, logits = valh.apply_model(hidden,
                              m_tok,
                              b_tok,
                              train_state,
                              model,
                              batchnorm,
                              False)
    # jax.debug.print("{}",logits.shape)
    logits=logits[0]
    argsortedlogits=jnp.argsort(logits,descending=True)
    # jax.debug.print("Best tokens for index {} before the mask: \n {} \n best logits: \n {}",token_index,argsortedlogits,logits[0][argsortedlogits])


    # filter out (syntactically) invalid tokens for current position
    #TODO: check that the masking works correctly

    if valid_mask is not None:
        logits = valh.filter_valid_pred(logits, valid_mask)



    # jax.debug.print("Best logits for index {} after the mask: \n {}",token_index,jnp.flip(jnp.argsort(logits)))


    # update sequence
    # NOTE: rng arg expects one element per batch element
    rng, rng_ = jax.random.split(rng)
    m_tok = valh.fill_predicted_tok( logits, sample_top_n, jnp.array([rng_]))
    return m_tok, hidden, token_index + 1, rng

def _make_generate_token_scannable(
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        valid_mask_array: jax.Array,
        sample_top_n: int,
    ):
    """
    """
    __generate_token = jax.jit(functools.partial(
        _generate_token, train_state, model, batchnorm, valid_mask_array, sample_top_n
    ))

    def _generate_token_scannable(carry, xs):
        # m_seq, b_tok, mask_i, rng = carry
        m_tok, hidden, mask_i, rng = __generate_token(*carry)
        return (m_tok, carry[1], hidden, mask_i, rng), m_tok

    return _generate_token_scannable


def _generate_msg(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,

        m_init: jax.Array, #last token from prev message, or start tok.
        b_init: jax.Array, #last book state after prev message, or start book.
        n_msg_todo: int,
        p_mid: jax.Array,
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        hidden: Tuple,
        time_i: jax.Array,
        # ema_start:bool,
        b_seq_real:Optional[jax.Array]=None,
        inject_flag: Optional[jax.Array]=None,   # scalar 0/1 for THIS step
        inject_msg: Optional[jax.Array]=None,    # (4,) = (event_type, side, qty, offset_ticks) for THIS step
    ) -> Tuple[jax.Array, LobState, jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """
    """
    # jax.debug.print("Input token with {} msg todo \n {}",n_msg_todo, m_init)

    rng, rng_ = jax.random.split(rng)
    # treat as compile time constants
    with jax.ensure_compile_time_eval():
        l = Message_Tokenizer.MSG_LEN
        time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
        delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
        delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    #
    time_init_s = time_i[0]
    time_init_ns = time_i[1]

    # TODO: calculating time in case where generation is not sequentially left to right
    #       --> check if delta_t complete --> calc time once

    generate_token_scannable = _make_generate_token_scannable(
        train_state, model, batchnorm, valid_mask_array, sample_top_n
    )

    if debug_book:
        b_init=jnp.expand_dims(b_seq_real,0)

    # get next message: generate l tokens:
    # generate tokens until time is reached
    token_idx = 0
    # Pass the first token & book (last from prev msg or START)
    #Generate tokens up to the last delta t (before first abs time)
    gen_token_carry = (m_init, b_init,hidden, token_idx, rng_)
    # jax.debug.print("Book  going into scan token for with {} msg todo \n {}",n_msg_todo, b_init)
    (m_inter, b_inter, hidden, token_idx, rng_), tok_seq_A = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=time_s_start_i
    )
    tok_seq_A=jnp.squeeze(tok_seq_A)
    # fill the time tokens, retain the actual times, to generate the next message.
    tok_seq_T, time_s, time_ns= _add_time_tokens(
        tok_seq_A,
        encoder,
        time_init_s,
        time_init_ns,
        delta_t_s_start_i,
        delta_t_s_end_i,
        delta_t_ns_start_i,
        delta_t_ns_end_i,
    )
    time_f=jnp.array([time_s, time_ns])

    # jax.debug.print("Calling apply model with time tokens {} with {} msgs to go",tok_seq_T,n_msg_todo)

    tok_seq_roll_thru_hidden=jnp.concatenate([tok_seq_A[-1:],tok_seq_T[:-1]])
    # jax.debug.print("Calling apply model with 'time' tokens {} with {} msgs to go",tok_seq_roll_thru_hidden,n_msg_todo)

    hidden,_=valh.apply_model(hidden,
                            tok_seq_roll_thru_hidden,
                            b_init,
                            train_state,
                            model,
                            batchnorm,
                            False)

    # update mask index to skip time token positions
    token_idx = time_ns_end_i
    gen_token_carry = (tok_seq_T[-1:], b_init, hidden, token_idx, rng_)

    # finish message generation
    (m_final, b_final, hidden, token_idx, rng_), tok_seq_B = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=l-time_ns_end_i
    )

    tok_seq_B=jnp.squeeze(tok_seq_B)


    # Fully generated message.
    tok_seq_gen=jnp.concatenate([tok_seq_A,tok_seq_T,tok_seq_B])
    # order_id = id_gen.step()  # no order ID generator any more in v3 sim?
    order_id = n_msg_todo

    sim_msg, msg_decoded, provenance = get_sim_msg(
        tok_seq_gen,  # the generated message
        # m_seq[:-l],  # sequence without generated message
        # m_seq_raw[1:],   # raw data (same length as sequence without generated message)
        # None,
        sim,
        sim_state,
        mid_price = p_mid,
        new_order_id = order_id,
        tick_size = tick_size,
        encoder = encoder,
    )

    # --- do(Q) injection ---------------------------------------------------
    # At the flagged step, replace the generated sim_msg with a hand-built
    # order, so the rollout answers p(market | do(our order)) rather than the
    # observational p(market). inject_msg = (event_type, side, qty,
    # offset_ticks); the absolute price is derived in-kernel from the live
    # p_mid, so it is correct per rollout rather than fixed at trace time.
    # Defaults (None) => the branch is not traced at all => the generator
    # program is byte-identical to the baseline.
    if inject_flag is not None:
        def _injected(_sim_msg):
            p_abs_inj = p_mid + inject_msg[3] * tick_size
            return construct_sim_msg(
                inject_msg[0].astype(jnp.int32),   # event_type
                inject_msg[1].astype(jnp.int32),   # side
                inject_msg[2].astype(jnp.int32),   # quantity
                p_abs_inj.astype(jnp.int32),       # p_abs
                order_id,                          # order_id
                time_f[0].astype(jnp.int32),       # time_s
                time_f[1].astype(jnp.int32),       # time_ns
            )
        sim_msg = jax.lax.cond(
            inject_flag > 0,
            _injected,
            lambda s: s,
            sim_msg,
        )
    # -----------------------------------------------------------------------

    # def print_cond(string_,msg,n_msg_todo):
    #     if n_msg_todo==500:
    #         print(f"{string_} with {n_msg_todo} msg todo \n {msg}")

    # jax.debug.callback(print_cond, "sim_msg", sim_msg,n_msg_todo)

    # feed message to simulator, updating book state
    # jax.debug.callback(print_cond, "sim_state before", sim_state,n_msg_todo)

    sim_state = sim.process_order_array(sim_state, sim_msg)

    # jax.debug.callback(print_cond, "sim_state after", sim_state,n_msg_todo)


    # debug('trades', _trades)

    # get current mid price from simulator
    p_mid_new = _get_new_mid_price(sim, sim_state, p_mid, tick_size)
    # jax.debug.print('p_mid_new {}', p_mid_new)

    # price change in ticks
    p_change = ((p_mid_new - p_mid) // tick_size)#.astype(jnp.int32)

    # get new book state
    book_l2 = sim.get_L2_state(sim_state, l2_state_n)
    # l2_book_states.append(book_l2)

    # error if the new message does not change the book state
    # is_error = (book_l2 == b_seq[-1, 1:]).all()

    new_book_raw = jnp.concatenate([jnp.array([p_change]),time_f, book_l2[0:40]]).reshape(1,-1)
    # jax.debug.print("book shape with time and midprice sim for with {} msg todo \n {}",n_msg_todo,new_book_raw)

    b_final = preproc.transform_L2_state_gpu(new_book_raw, 500, 100)
    # jax.debug.print("book after transform after message, for with {} msg todo \n {}",n_msg_todo,b_final)
    # update book sequence

    n_msg_todo -= 1

    return msg_decoded, sim_state, m_final, tok_seq_gen, b_final, book_l2, p_mid_new, n_msg_todo, hidden, time_f, provenance


def _make_generate_msg_scannable(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
    ):
    """
    """
    __generate_msg = jax.jit(functools.partial(
        _generate_msg, sim, train_state, model, batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size,debug_book
    ),device=jax.devices()[0])

    def _generate_msg_scannable(gen_state, input):
        """ Wrapper for _generate_msg to be used with jax.lax.scan
        """
        # input is either b_seq_real (legacy/replay) or
        # (inject_flag, inject_msg) when injecting in pure-generation mode.
        if isinstance(input, tuple):
            inject_flag, inject_msg = input
            b_seq_real = None
        else:
            b_seq_real = input
            inject_flag, inject_msg = None, None
        m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time= gen_state
        rng, rng_ = jax.random.split(rng)

        msg_decoded, sim_state, m_seq, msg_token, b_seq, book_l2, p_mid, n_msg_todo,hidden, time, provenance = __generate_msg(
            m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng_, hidden, time, b_seq_real, inject_flag, inject_msg
        )
        return (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng,hidden, time), (msg_decoded, book_l2, msg_token, provenance)
    return _generate_msg_scannable


# ─────────────────────────────────────────────────────────────────────
# 1tok inference path: one model call per message → 24 field logits
# ─────────────────────────────────────────────────────────────────────

def _sample_fields_1tok(field_logits_list, sample_top_n, rng):
    """Sample from 24 per-field logit distributions independently.

    Args:
        field_logits_list: list of 24 arrays, each (1, V_i) log-softmax
        sample_top_n: 1 for argmax, >1 for top-k, -1 for full distribution
        rng: JAX PRNG key
    Returns:
        (sampled_local, rng) where sampled_local is (24,) int32 local indices
    """
    samples = []
    for i, logits in enumerate(field_logits_list):
        rng, rng_ = jax.random.split(rng)
        logits_i = jnp.squeeze(logits)  # (1, 1, V_i) or (1, V_i) → (V_i,)
        # Block special tokens during generation
        # logits_i is already log_softmax from MultiFieldDecoder
        logits_i = logits_i.at[:N_SPECIAL_TOKENS].set(-jnp.inf)
        if sample_top_n == 1:
            chosen = jnp.argmax(logits_i)
        elif sample_top_n > 0:
            top_k_vals, top_k_idx = jax.lax.top_k(logits_i, sample_top_n)
            probs = jnp.exp(top_k_vals)
            probs = probs / probs.sum()  # renormalize after top-k
            chosen = jax.random.choice(rng_, top_k_idx, p=probs)
        else:
            probs = jnp.exp(logits_i)
            probs = probs / probs.sum()  # renormalize after blocking specials
            chosen = jax.random.choice(rng_, jnp.arange(logits_i.shape[0]), p=probs)
        samples.append(chosen)
    return jnp.stack(samples), rng


def _generate_msg_1tok(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,

        m_init: jax.Array,      # (1, 24) local indices — last generated or last cond msg
        b_init: jax.Array,      # (1, book_dim) — last book state
        n_msg_todo: int,
        p_mid: jax.Array,
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        hidden: Tuple,
        time_i: jax.Array,
        b_seq_real: Optional[jax.Array] = None,
    ) -> Tuple:
    """Generate one complete message in a single model call (1tok mode)."""
    rng, rng_ = jax.random.split(rng)

    with jax.ensure_compile_time_eval():
        time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
        delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
        delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    time_init_s = time_i[0]
    time_init_ns = time_i[1]

    if debug_book:
        b_init = jnp.expand_dims(b_seq_real, 0)

    # One model call → all 24 field logits
    hidden, field_logits = valh.apply_model_1tok(
        hidden, m_init, b_init, train_state, model, batchnorm, False)

    # Sample all 24 fields at once
    sampled_local, rng_ = _sample_fields_1tok(field_logits, sample_top_n, rng_)

    # Convert to global token IDs (same space as 24tok)
    sampled_global = local_to_global_jax(sampled_local)

    # Compute absolute time from delta_t + previous time (override model's time prediction)
    time_tokens, time_s, time_ns = _add_time_tokens(
        sampled_global,  # full message — _add_time_tokens indexes into it
        encoder,
        time_init_s, time_init_ns,
        delta_t_s_start_i, delta_t_s_end_i,
        delta_t_ns_start_i, delta_t_ns_end_i,
    )
    # Overwrite absolute time fields with computed values
    sampled_global = sampled_global.at[time_s_start_i:time_ns_end_i].set(time_tokens)
    time_f = jnp.array([time_s, time_ns])

    order_id = n_msg_todo

    # Decode and process through simulator (same as 24tok from here)
    sim_msg, msg_decoded, provenance = get_sim_msg(
        sampled_global,
        sim,
        sim_state,
        mid_price=p_mid,
        new_order_id=order_id,
        tick_size=tick_size,
        encoder=encoder,
    )

    sim_state = sim.process_order_array(sim_state, sim_msg)

    p_mid_new = _get_new_mid_price(sim, sim_state, p_mid, tick_size)
    p_change = ((p_mid_new - p_mid) // tick_size)

    book_l2 = sim.get_L2_state(sim_state, l2_state_n)
    new_book_raw = jnp.concatenate([jnp.array([p_change]), time_f, book_l2[0:40]]).reshape(1, -1)
    b_final = preproc.transform_L2_state_gpu(new_book_raw, 500, 100)

    # Next step input: convert back to local for the model
    sampled_local_out = global_to_local_jax(sampled_global)
    m_final = sampled_local_out.reshape(1, N_FIELDS)  # (1, 24)

    n_msg_todo -= 1

    return msg_decoded, sim_state, m_final, sampled_global, b_final, book_l2, p_mid_new, n_msg_todo, hidden, time_f, provenance


def _make_generate_msg_1tok_scannable(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
    ):
    __generate_msg = jax.jit(functools.partial(
        _generate_msg_1tok, sim, train_state, model, batchnorm,
        encoder, sample_top_n, tick_size, debug_book
    ), device=jax.devices()[0])

    def _generate_msg_scannable(gen_state, input):
        b_seq_real = input
        m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time = gen_state
        rng, rng_ = jax.random.split(rng)

        msg_decoded, sim_state, m_seq, msg_token, b_seq, book_l2, p_mid, n_msg_todo, hidden, time, provenance = __generate_msg(
            m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng_, hidden, time, b_seq_real
        )
        return (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time), (msg_decoded, book_l2, msg_token, provenance)
    return _generate_msg_scannable


@partial(jax.jit, static_argnums=(0, 2, 3, 5, 6, 9, 13, 15), backend='gpu')
def generate_1tok(
        sim: OrderBook,              # static
        train_state: TrainState,
        model: nn.Module,            # static
        batchnorm: bool,             # static
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,           # static
        tick_size: int,              # static
        m_seq_cond: jax.Array,       # (n_cond_msgs+1, 24) local indices
        b_seq_cond: jax.Array,       # (n_cond_msgs+1, book_dim)
        n_msg_todo: int,             # static
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        init_hidden: Tuple,
        conditional: bool,           # static
        init_time: jax.Array,
        debug_book: bool = False,
        b_seq_real: Optional[jax.Array] = None,
        valid_mask_array: Optional[jax.Array] = None,  # unused for 1tok, kept for API compat
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    print("WARNING: Compiling generate_1tok, you should only see this once.")

    if not debug_book:
        b_seq_real = None

    if conditional:
        def roll_hidden_scan_1tok(carry, xs):
            m_seq, b_seq = xs
            h = carry
            h, _ = valh.apply_model_1tok(
                h, m_seq, b_seq, train_state, model, batchnorm, True)
            return h, None

        # Split conditioning: one message per scan step
        N = m_seq_cond[:-1].shape[0]  # n_cond_msgs
        m_seq_cond_split = m_seq_cond[:-1].reshape((N, 1, N_FIELDS))
        b_seq_cond_split = b_seq_cond[:-1].reshape((N, 1) + b_seq_cond.shape[1:])

        hidden_state, _ = jax.lax.scan(
            roll_hidden_scan_1tok, init_hidden, (m_seq_cond_split, b_seq_cond_split))
        init_token = m_seq_cond[-1:]   # (1, 24)
        init_book = b_seq_cond[-1:]    # (1, book_dim)
        init_time = jnp.asarray(valh.get_first_time_1tok(m_seq_cond, encoder))
    else:
        # Unconditional 1tok generation: start from START token and initial book.
        hidden_state = init_hidden
        # Ensure shapes are (1, 24) / (1, book_dim) for scan compatibility
        init_token = m_seq_cond if m_seq_cond.ndim == 2 else m_seq_cond.reshape(1, -1)
        init_book = b_seq_cond if b_seq_cond.ndim >= 2 else b_seq_cond.reshape(1, -1)
        # init_time passed in as parameter; default to market open if all zeros
        init_time = jnp.where(
            jnp.all(init_time == 0),
            jnp.asarray((34200, 0)),
            init_time,
        )

    p_mid = _get_safe_mid_price(sim, sim_state, tick_size)

    generate_msg_scannable = _make_generate_msg_1tok_scannable(
        sim, train_state, model, batchnorm,
        encoder, sample_top_n, tick_size, debug_book,
    )
    gen_state, (msgs_decoded, l2_book_states, msgs_tokens, provenances) = jax.lax.scan(
        generate_msg_scannable,
        (init_token, init_book, n_msg_todo, p_mid, sim_state, rng, hidden_state, init_time),
        length=n_msg_todo,
        xs=b_seq_real,
    )

    num_errors = (l2_book_states[1:] == l2_book_states[:-1]).all(axis=1).sum()

    return msgs_decoded, l2_book_states, num_errors, msgs_tokens, provenances


generate_batched_1tok = jax.jit(
    jax.vmap(
        generate_1tok,
        in_axes=(
            None, None, None, None, None,
            None, None,    0,    0, None,
            0,       0,    0, None,    0,
            None,    0, None,
        )
    ),
    static_argnums=(0, 2, 3, 5, 6, 9, 13, 15), backend='gpu'
)


@partial(jax.jit, static_argnums=(0, 2, 3, 5, 6, 9,13,15),backend='gpu')
def generate(
        sim: OrderBook,  # static
        train_state: TrainState,
        model: nn.Module,  # static
        batchnorm: bool,  # static
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,  # static
        tick_size: int,  # static
        m_seq_cond: jax.Array,
        b_seq_cond: jax.Array,
        n_msg_todo: int,  # static
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        init_hidden : Tuple,
        conditional : bool, # static
        init_time : jax.Array,
        debug_book: bool=False,
        b_seq_real: Optional[jax.Array]=None, #Must be very careful, these should only be used for debugging.
        valid_mask_array: Optional[jax.Array]=None,  # pre-computed syntax mask
        inject_flag_seq: Optional[jax.Array]=None,   # (n_msg_todo,) traced 0/1
        inject_msg_seq: Optional[jax.Array]=None,    # (n_msg_todo, 4) traced
        # if eval_msgs given, also returns loss of predictions
        # e.g. to calculate perplexity
        # m_seq_eval: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:

    # id_gen = OrderIdGenerator()
    # l = Message_Tokenizer.MSG_LEN
    # v = Vocab()
    # vocab_len = len(v)
    # last_start_i = m_seq.shape[0] - l
    # l2_book_states = []
    # m_seq_raw = m_seq_raw.copy()
    # num_errors = 0
    print("WARNING: Compiling the generate function, you should only see this once.")
    # m_seq_cond=m_seq_cond.copy()
    # b_seq_cond=b_seq_cond.copy()

    if valid_mask_array is None:
        with jax.ensure_compile_time_eval():
            valid_mask_array = valh.syntax_validation_matrix()

    # valid_mask_array=None
    # jax.debug.print("Note: Valid mask turned off in generate_token")

    if not debug_book:
        b_seq_real=None


    if conditional:
        # Conditioning via scan (one message per step through __call_inference__)
        # NOTE: prefill (training-identical conditioning) is done in sample_new()
        # BEFORE generate() is called. init_hidden already contains the prefilled
        # KV cache when prefill is enabled.
        def roll_hidden_scan(carry, xs):
            m_seq, b_seq = xs
            h = carry
            h, log = valh.apply_model(h, m_seq, b_seq, train_state, model,
                                       batchnorm, True)
            carry = h
            return carry, None

        l = Message_Tokenizer.MSG_LEN
        total_cond_tokens = m_seq_cond[:-1].shape[0]
        N = total_cond_tokens // l

        if N > 0:
            # Standard conditioning scan (one message per step)
            chex.assert_is_divisible(total_cond_tokens, N)
            chex.assert_is_divisible(b_seq_cond[:-1].shape[0], N)
            m_seq_cond_split = m_seq_cond[:-1].reshape((N, -1))
            b_seq_cond_split = b_seq_cond[:-1].reshape((N, -1) + b_seq_cond[:-1].shape[1:])

            hidden_state, _ = jax.lax.scan(roll_hidden_scan, init_hidden,
                                            (m_seq_cond_split, b_seq_cond_split))
        else:
            # Prefilled case: init_hidden already contains conditioned KV cache.
            # No conditioning tokens to scan — just use init_hidden as-is.
            hidden_state = init_hidden

        init_token = m_seq_cond[-1:]
        init_book = b_seq_cond[-1:]
        init_time = jnp.asarray(valh.get_first_time(m_seq_cond, encoder))
    else:
        # Unconditional generation: start from START token and initial book state.
        # m_seq_cond should be the START token (scalar or 1-element array).
        # b_seq_cond should be the initial book state.
        hidden_state=init_hidden
        # Ensure init_token is (1,) shaped for scan compatibility
        if m_seq_cond.ndim == 0:
            init_token = m_seq_cond.reshape(1)
        elif m_seq_cond.ndim == 1 and m_seq_cond.shape[0] == 1:
            init_token = m_seq_cond
        else:
            # If passed a full message, use first token as START
            init_token = m_seq_cond[:1]
        init_book = b_seq_cond if b_seq_cond.ndim >= 2 else b_seq_cond.reshape(1, -1)
        # init_time passed in as parameter; default to market open if None-like
        init_time = jnp.where(
            jnp.all(init_time == 0),
            jnp.asarray((34200, 0)),  # market open: 09:30:00
            init_time,
        )

    # jax.debug.print("hidden_state vs init hidden state {}",hidden_state==init_hidden)

    # get current mid price from simulator
    p_mid = _get_safe_mid_price(sim, sim_state, tick_size)
    # jax.debug.print('generate - p_mid {}', p_mid)

    generate_msg_scannable = _make_generate_msg_scannable(
        sim, train_state, model, batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size, debug_book,
    )
    # do(Q) injection rides xs as a (flag, msg) tuple; else legacy b_seq_real.
    # Both halves or neither: one alone is a mis-wiring that would otherwise
    # fail deep inside the scan (a None leaf reaching inject_msg[3]), or worse,
    # silently ignore the message it was handed.
    if (inject_flag_seq is None) != (inject_msg_seq is None):
        raise ValueError(
            "inject_flag_seq and inject_msg_seq must be given together "
            f"(flag={'set' if inject_flag_seq is not None else 'None'}, "
            f"msg={'set' if inject_msg_seq is not None else 'None'})"
        )
    if inject_flag_seq is not None:
        _xs = (inject_flag_seq, inject_msg_seq)
    else:
        _xs = b_seq_real
    gen_state, (msgs_decoded, l2_book_states, msgs_tokens, provenances) = jax.lax.scan(
        generate_msg_scannable,
        (init_token, init_book, n_msg_todo, p_mid, sim_state,rng, hidden_state,init_time),
        length=n_msg_todo,
        xs=_xs,
    )
    (final_token, final_book,n_msg_todo, p_mid, sim_state, rng, hidden_state,final_time) = gen_state

    # all_msg_toks_gen=jnp.concatenate(msgs_tokens)



    # count errors when the message does not change the (visible) book state
    num_errors = (l2_book_states[1:] == l2_book_states[:-1]).all(axis=1).sum()

    return msgs_decoded, l2_book_states, num_errors, msgs_tokens, provenances

generate_batched = jax.jit(
    jax.vmap(
        generate,
        in_axes=(
            None, None, None, None, None,
            None, None,    0,    0, None,
            0,       0,    0, None,    0,
            None,    0, None, None, None,
        )
    ),
    static_argnums=(0, 2, 3, 5, 6, 9,13,15),backend='gpu'
)

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def calc_sequence_losses(
        m_seq,
        b_seq,
        state,
        model,
        batchnorm,
        n_inp_msgs,  # length of input sequence in messages
        valid_mask_array
    ):
    """ Takes a sequence of messages, and calculates cross-entropy loss for each message,
        based on the next message in the sequence.
    """
    @partial(jax.jit, static_argnums=(1,2))
    def moving_window(a: jax.Array, size: int, stride: int = 1):
        starts = jnp.arange(0, len(a) - size + 1, stride)
        return jax.vmap(
            lambda start: jax.lax.dynamic_slice(
                a,
                (start, *jnp.zeros(a.ndim-1, dtype=jnp.int32)),
                (size, *a.shape[1:])
            )
        )(starts)

    l = Message_Tokenizer.MSG_LEN

    @jax.jit
    def prep_single_inp(
            mask_i,
            na_mask,
            m_seq,
            b_seq,
        ):
        m_seq = m_seq.copy().reshape((-1, l))
        last_msg = jnp.where(
            na_mask,
            Vocab.HIDDEN_TOK,#Vocab.NA_TOK,
            m_seq[-1]
        )
        m_seq = m_seq.at[-1, :].set(last_msg).reshape(-1)
        m_seq, y = valh.mask_last_msg_in_seq(m_seq, mask_i)

        input = (m_seq, b_seq)
        integration_timesteps = (
            jnp.ones(len(m_seq), dtype=jnp.float32),
            jnp.ones(len(b_seq), dtype=jnp.float32)
        )
        return input, integration_timesteps, y.astype(jnp.float32)
    prep_multi_input = jax.vmap(prep_single_inp, in_axes=(0, 0, None, None))

    @jax.jit
    def single_msg_losses(carry, inp):
        @partial(jax.jit, static_argnums=(0,))
        def na_mask_slice(last_non_masked_i):
            a = jnp.ones((l,), dtype=jnp.bool_)
            a = a.at[: last_non_masked_i+1].set(False)
            return a

        m_seq, b_seq, valid_mask = inp
        mask_idxs = jnp.concatenate([jnp.arange(0, TIME_START_I), jnp.arange(TIME_END_I, l)])
        na_masks = jnp.array([na_mask_slice(i) for i in range(TIME_START_I)] \
            + [na_mask_slice(i) for i in range(TIME_END_I, l)])

        bsz = 10
        assert 2*bsz >= mask_idxs.shape[0], f'bsz:{bsz}; msg len:{mask_idxs.shape[0]}'
        # split inference into two batches to avoid OOM
        input, integration_timesteps, y1 = prep_multi_input(mask_idxs[:bsz], na_masks[:bsz], m_seq, b_seq)
        logits1 = valh.predict(
            input,
            integration_timesteps, state, model, batchnorm)
        input, integration_timesteps, y2 = prep_multi_input(mask_idxs[-bsz:], na_masks[-bsz:], m_seq, b_seq)
        logits2 = valh.predict(
            input,
            integration_timesteps, state, model, batchnorm)

        logits = jnp.concatenate([logits1, logits2[2*bsz - mask_idxs.shape[0] : ]], axis=0)
        y = jnp.concatenate([y1, y2[2*bsz - mask_idxs.shape[0] : ]], axis=0)

        # filter out (syntactically) invalid tokens for current position
        if valid_mask is not None:
            logits = valh.filter_valid_pred(logits, valid_mask)

        losses = train_helpers.cross_entropy_loss(logits, y)
        return carry, losses

    m_seq = m_seq.reshape((-1, l))
    inputs = (
        moving_window(m_seq, n_inp_msgs),
        moving_window(b_seq, n_inp_msgs),
        jnp.repeat(
            jnp.expand_dims(
                jnp.delete(valid_mask_array, slice(TIME_START_I, TIME_END_I), axis=0),
                axis=0
            ),
            m_seq.shape[0] - n_inp_msgs + 1,
            axis=0
        )
    )
    last_i, losses = jax.lax.scan(
        single_msg_losses,
        init=0,
        xs=inputs
    )
    return losses

# def generate_single_rollout(
#         m_seq_inp,
#         b_seq_inp,
#         n_gen_msgs,
#         sim,
#         sim_state,
#         state,
#         model,
#         batchnorm,
#         encoder,
#         rng,
#     ):

#     rng, rng_ = jax.random.split(rng)

#     # generate predictions
#     m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = generate(
#         m_seq_inp,
#         b_seq_inp,
#         n_gen_msgs,
#         sim,
#         sim_state,
#         state,
#         model,
#         batchnorm,
#         encoder,
#         rng_,
#         sample_top_n=-1,  # sample from entire distribution
#     )

#     return (
#         m_seq_gen,
#         b_seq_gen,
#         {
#             'num_errors': num_errors,
#             'l2_book_states': l2_book_states,
#         }
#     )

# # sample from distribution of rollouts with same input an different rng keys
# generate_repeated_rollouts = jax.vmap(generate_single_rollout, in_axes=((None,)*9 + (0,)))
# # sample different rollouts with different input sequences (and different rng keys)
# generate_multiple_rollouts = jax.vmap(generate_single_rollout, in_axes=(0, 0, None, None, 0, None, None, None, None, 0))

def sample_new(
        n_samples: int,  # draw n random samples from dataset for evaluation
        batch_size: int,  # how many samples to process in parallel
        ds: LOBSTER_Dataset,
        rng: jax.dtypes.prng_key,
        seq_len_cond: int, #cond: should be 0 if uncond.
        n_cond_msgs: int, #cond: should be 0 if uncond
        n_gen_msgs: int, #gen
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        stock_symbol: str,
        n_vol_series: int = 500,
        # sim_book_levels: int = 20,
        # sim_queue_len: int = 100,
        # data_levels: int = 10,
        save_folder: str = './data_saved/',
        tick_size: int = 100,
        sample_top_n: int = -1,
        init_hidden: Optional[Tuple] = None,
        args: Optional[Any] = None,
        conditional: bool = True,
        v: Vocab = Vocab(),
        overfit_debug: bool = False,
        sample_indices: Optional[List[int]] = None,
        wide_levels: int = 10,
        token_mode: str = '26tok',
        gt_compare: bool = False,  # accepted for pipeline compat, not implemented in O2b
        inject_step: Optional[int] = None,        # which gen step to inject at (None=off)
        inject_event_type: int = 4,               # 4 = marketable execution; 1 = new limit
        inject_side: int = 1,                     # 1 = bid side, 0 = ask side
        inject_qty: int = 0,                      # Q (shares)
        inject_price_offset_ticks: int = 0,       # p_abs = mid + offset*tick (0 = at touch)
    ):
    """
    """
    is_1tok = (token_mode == '1tok')
    assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'
    if conditional is False:
        assert n_cond_msgs==0, "If conditional flag is false then cannot expect to have any messages for conditioning."
        if not is_1tok:
            assert seq_len_cond==0, "If conditional flag is false, then cannot have any tokens for conditioning."

    rng, rng_ = jax.random.split(rng)
    if sample_indices is not None:
        # Pre-determined indices (from multi-GPU rank splitting or HF-matched mode)
        flat = list(sample_indices)
        sample_i = [flat[i:i+batch_size] for i in range(0, len(flat), batch_size)]
    elif overfit_debug:
        sample_i = [list(range(batch_size))]
    else:
        sample_i = jax.random.choice(
            rng_,
            jnp.arange(len(ds), dtype=jnp.int32),
            shape=(n_samples // batch_size, batch_size),
            replace=False
        ).tolist()
    rng, rng_ = jax.random.split(rng)

    # create folders to save the data if they don't exist yet
    Path(save_folder + f'/data_cond/').mkdir(exist_ok=True, parents=True)
    Path(save_folder + f'/data_real/').mkdir(exist_ok=True, parents=True)
    Path(save_folder + f'/data_gen/').mkdir(exist_ok=True, parents=True)


    if (init_hidden == None):
        if is_1tok:
            # 1tok: 3-tuple carry (book, fused, ema) — no message_encoder
            init_hidden = model.initialize_carry(1,
                                            hidden_size=(args.ssm_size_base // pow(2, int(args.conj_sym))),
                                            n_book_pre_layers=args.n_book_pre_layers,
                                            n_book_post_layers=args.n_book_post_layers,
                                            n_fused_layers=args.n_layers,
                                            h_size_ema=args.ssm_size_base)
        else:
            from base_model.models.registry import attention_config_for_width, build_backbone

            backbone = build_backbone(args)
            carry_kwargs = dict(backbone.carry_kwargs)
            if backbone.is_attention:
                msg_len = Message_Tokenizer.MSG_LEN
                max_cache_len = seq_len_cond + n_gen_msgs * msg_len + msg_len
                transformer_config = attention_config_for_width(
                    backbone, args.d_model, max_cache_len=max_cache_len)
                d_book = getattr(args, 'd_book', 503)
                transformer_config_book = attention_config_for_width(
                    backbone, d_book, max_cache_len=max_cache_len)
                carry_kwargs['transformer_config'] = transformer_config
                carry_kwargs['transformer_config_book'] = transformer_config_book
            elif backbone.is_hybrid:
                # A hybrid trunk carries two different things: the recurrent
                # positions take the mamba3 state tuple that carry_kwargs
                # already describes, and the attention positions take a KV
                # cache. Without this branch every layer gets the recurrent
                # carry and the attention layer unpacks a four-tuple into three
                # names. Training never touches this path, because the parallel
                # scan has no carry, so the gap only appears at generation.
                # Attention lives in the fused trunk only, so d_model is the
                # only width that needs a cache.
                msg_len = Message_Tokenizer.MSG_LEN
                max_cache_len = seq_len_cond + n_gen_msgs * msg_len + msg_len
                carry_kwargs['attn_layers'] = backbone.attn_layers
                carry_kwargs['attn_config'] = attention_config_for_width(
                    backbone, args.d_model, max_cache_len=max_cache_len)
            hidden_size = int(args.ssm_size_base)
            if backbone.definition.name == 's5' and args.conj_sym:
                hidden_size //= 2
            init_hidden = model.initialize_carry(
                1,
                hidden_size=hidden_size,
                n_message_layers=args.n_message_layers,
                n_book_pre_layers=args.n_book_pre_layers,
                n_book_post_layers=args.n_book_post_layers,
                n_fused_layers=args.n_layers,
                h_size_ema=args.d_model,
                d_book=getattr(args, 'd_book', 503),
                **carry_kwargs)

    # jax.debug.print("Init hidden is: \n {}",len(init_hidden))
    # Assumes only a single hidden state is given and needs to be duplicated. TODO Add a flag.
    init_hidden_batched=jax.tree_util.tree_map(lambda x : jnp.resize(x,(batch_size,)+x.shape),init_hidden)


    #TODO: complete these options to make sure every case works and add some asserts.
    # print(jax.tree_util.tree_map(lambda x : x.shape, init_hidden ))
    # print(jax.tree_util.tree_map(lambda x : x.shape, init_hidden_batched ))


    # init_time_batched=jax.tree_util.tree_map(lambda x : jnp.resize(x,(batch_size,)+x.shape),init_time)
    # print(jax.tree_util.tree_map(lambda x : x.shape, init_time ))
    # print(jax.tree_util.tree_map(lambda x : x.shape, init_time_batched ))
    # nOrders must fit all init orders (2 * wide_levels) plus headroom for
    # generated orders during the sequence. book_depth controls the init_id
    # range so get_init_volume_at_price recognises init orders correctly.
    sim_nOrders = max(100, wide_levels * 2 + 50)
    sim_init = OrderBook(cfg=JAXLOB_Configuration(
        nOrders=sim_nOrders,
        book_depth=wide_levels,
        cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value,
    ))
    if wide_levels > 10:
        print(f"[sample_new] Wide init: {wide_levels} levels, nOrders={sim_nOrders}, "
              f"init_id_range=[{sim_init.cfg.init_id}, {sim_init.cfg.init_id - wide_levels*2}]")
    # all_metrics = []
    initial=True
    conditional_orig = conditional  # save before prefill modifies it
    for batch_i in tqdm(sample_i):
        # print('BATCH', batch_i)
        # TODO: check if we can init the dataset without the raw data
        #       if it's not needed
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        # print("sample_new: M_seq_inputs:", m_seq)
        # print('m_seq.shape before jnp.array', onp.array(m_seq).shape)
        m_seq = jnp.array(m_seq)
        b_seq_pv = jnp.array(b_seq_pv)
        msg_seq_raw = jnp.array(msg_seq_raw)
        book_l2_init = select_wide_book_levels(
            jnp.array(book_l2_init),
            wide_levels,
        )

        # transform book to volume image representation for model
        b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)
        init_time_batched=b_seq_pv[:,0,1:3]


        #Add the start token
        #FIXME: Move this to the data loader using the inference mask.
            # Done?
        # m_seq=jnp.concatenate([jnp.ones((batch_size,1),dtype=int)*v.START_TOK,m_seq],axis=1)

        print(m_seq.shape)
        # encoded data
        if is_1tok:
            # Dataset prepends a single START token → strip it before reshaping
            # to (batch, n_msgs, 24) so field positions align correctly.
            m_seq_flat = m_seq[:, 1:]  # drop START token at position 0
            n_clean = (m_seq_flat.shape[1] // N_FIELDS) * N_FIELDS
            n_total_msgs = n_clean // N_FIELDS
            m_seq_2d = m_seq_flat[:, :n_clean].reshape(batch_size, n_total_msgs, N_FIELDS)
            m_seq_2d = global_to_local_jax(m_seq_2d)  # broadcasts over (batch, n_msgs, 24)
            m_seq_inp = m_seq_2d[:, :n_cond_msgs+1]   # (batch, n_cond+1, 24)
            m_seq_eval = m_seq_flat[:, (n_cond_msgs+1)*N_FIELDS:]  # keep flat for debug/save
        else:
            # For NSA/Transformer: strip START token to align message boundaries
            # with 26-token blocks (START shifts all positions by 1).
            # S5 SSM doesn't need this (no position encoding), but NSA does.
            model_type = getattr(args, 'model_type', 's5')
            if model_type in ('nsa', 'transformer'):
                # Strip START, take n_cond complete messages + 1 init token
                m_seq_no_start = m_seq[:, 1:]  # drop START at position 0
                m_seq_inp = m_seq_no_start[:, :seq_len_cond+1]
                m_seq_eval = m_seq_no_start[:, (seq_len_cond+1):]
            else:
                # S5 SSM: keep START (traditional behavior)
                m_seq_inp = m_seq[:, : seq_len_cond+1]
                m_seq_eval = m_seq[:, (seq_len_cond+1): ]
        # Debug prints to file
        # Set print options to show all array elements
        if overfit_debug:
            with open(f'debug_m_seq_inp_batch_{batch_i[0]}.txt', 'w') as f:
                print(f"m_seq_inp shape: {m_seq_inp.shape}", file=f)
                print(f"m_seq_inp:\n{m_seq_inp}", file=f)

            with open(f'debug_m_seq_eval_batch_{batch_i[0]}.txt', 'w') as f:
                print(f"m_seq_eval shape: {m_seq_eval.shape}", file=f)
                print(f"m_seq_eval:\n{m_seq_eval}", file=f)

        # Reset print options to default
        b_seq_inp = b_seq[: , : n_cond_msgs+1]
        b_seq_eval = b_seq[:, (n_cond_msgs+1):]
        # true L2 data: remove price change column
        # shape: [batch, messages, levels]
        b_seq_pv_inp = onp.array(b_seq_pv[:, : n_cond_msgs+1, 3:])
        b_seq_pv_eval = onp.array(b_seq_pv[:, (n_cond_msgs+1):, 3:]) #Drop the midprice and times for logging purposes in lobster.

        # raw LOBSTER data
        m_seq_raw_inp = msg_seq_raw[:, : n_cond_msgs]
        m_seq_raw_eval = msg_seq_raw[:, n_cond_msgs: ]

        # initialise simulator
        sim_states_init = get_sims_vmap(
            book_l2_init,  # book state before any messages
            m_seq_raw_inp, # messages to replay to init sim
            init_time_batched,
            sim_init,
            # TODO: consider passing nOrders, nTrades
        )

        # book state after initialisation (replayed messages)
        # actually, this is already part of the input data --> only needed for comparison
        # l2_book_states_init = sim_init.get_L2_states_vmap(sim_states_init, l2_state_n)

        # run actual messages on sim_eval (once) to compare
        # convert m_seq_raw_eval to sim_msgs
        # msgs_eval = msgs_to_jnp(m_seq_raw_eval[: n_gen_msgs])
        # sim_state_eval, l2_book_states_eval, _ = sim_init.process_orders_array_l2(sim_state_init, msgs_eval, l2_state_n)

        if overfit_debug:
            debug_book=True
        else:
            debug_book=False
        if debug_book:
            real_book=jnp.concatenate([jnp.expand_dims(b_seq_inp[:,-1],axis=1),b_seq_eval[:,:-1]],axis=1)
            print(real_book.shape)
        else:
            real_book=None

        # --- do(Q) injection arrays (None => no injection => baseline) ------
        # inject_step: which generation step to inject at (0-indexed).
        # inject_msg fields per step: (event_type, side, qty, offset_ticks).
        inject_flag_seq = None
        inject_msg_seq = None
        if inject_step is not None:
            _n = n_gen_msgs
            if not 0 <= inject_step < _n:
                # .at[i].set() drops out-of-bounds scatter updates and wraps
                # negative indices, so an off-by-one here would not raise: it
                # would produce a rollout that reports an injection and does
                # not contain one.
                raise ValueError(
                    f"inject_step={inject_step} is outside the generation "
                    f"horizon [0, {_n})"
                )
            inject_flag_seq = jnp.zeros((_n,), jnp.int32).at[inject_step].set(1)
            _msg_row = jnp.array(
                [inject_event_type, inject_side, inject_qty, inject_price_offset_ticks],
                dtype=jnp.int32,
            )
            inject_msg_seq = jnp.zeros((_n, 4), jnp.int32).at[inject_step].set(_msg_row)
            print(f"[inject] step={inject_step} etype={inject_event_type} "
                  f"side={inject_side} qty={inject_qty} offset={inject_price_offset_ticks}")
            if is_1tok:
                raise NotImplementedError(
                    "do(Q) injection is wired for the token-per-field generator only; "
                    "generate_1tok has no injection point."
                )
        # generate_batched's in_axes has one entry per positional argument, so
        # the two injection slots are always passed on the token-per-field path
        # (as None when off — an empty pytree, contributing no traced values and
        # taking no branch inside generate). The 1tok generator has no injection
        # slots, so its trace and its call both stay at the old arity.
        _inject_args = () if is_1tok else (inject_flag_seq, inject_msg_seq)
        # --------------------------------------------------------------------
        # print('m_seq_inp.shape', m_seq_inp.shape)
        # print('b_seq_inp.shape', b_seq_inp.shape)
        # print('sim_states_init.asks.shape', sim_states_init.asks.shape)
        # print('sim_states_init.bids.shape', sim_states_init.bids.shape)
        # print('sim_states_init.trades.shape', sim_states_init.trades.shape)
        # init_hidden_batched,init_time_batched,init_token_batched,init_book_batched=roll_batched(
        #     conditional, #Static
        #     train_state,  # None map, static?
        #     model, # static
        #     batchnorm, # static
        #     encoder,
        #     init_hidden_batched,
        #     m_seq_inp[:], # in_axis = 0
        #     b_seq_inp, # in_axis = 0
        #     init_time_batched,
        # )



        # === PREFILL: conditioning via training forward pass ===
        # Runs SAME flash attention as training (cuDNN, parallel) to build
        # KV cache with numerically identical K,V. Eliminates Msg 17 top-k
        # butterfly effect by starting generation from correct hidden state.
        # Reset conditional for this batch (prefill sets it to False)
        conditional = conditional_orig
        use_prefill = (not is_1tok and conditional and
                       getattr(args, 'model_type', 's5') in ('transformer', 'nsa'))
        if use_prefill:
            l = Message_Tokenizer.MSG_LEN
            # Conditioning tokens (complete messages only, multiple of block_size)
            cond_len = (m_seq_inp.shape[1] - 1) // l * l  # exclude last token, round down
            m_cond_batch = m_seq_inp[:, :cond_len]  # (batch, cond_len)
            # Book: repeat each book state for all tokens in its message
            n_cond = cond_len // l
            b_cond_batch = jnp.repeat(b_seq_inp[:, :n_cond], l, axis=1)  # (batch, cond_len, book_dim)
            ts_cond = jnp.ones((batch_size, cond_len))

            max_cache_len = cond_len + n_gen_msgs * l + l

            prefilled_hidden, _ = model.apply(
                {"params": train_state.params},
                m_cond_batch, b_cond_batch, ts_cond, ts_cond, max_cache_len,
                method='prefill')
            # Use prefilled hidden instead of empty cache
            init_hidden_batched = prefilled_hidden
            # Skip conditioning scan inside generate() by setting conditional=False.
            # But we still need init_token and init_time from the conditioning data.
            # Compute them here, then pass them via m_seq_inp/b_seq_inp as 1-token inputs.
            # generate() with conditional=False uses m_seq_cond directly as init_token.
            # Extract init_time from last conditioning message BEFORE trimming
            # (time is encoded in the last 26 tokens of conditioning)
            last_msg_tokens = m_seq_inp[:, cond_len-l:cond_len]  # (batch, 26)
            # Decode time from last message (per batch element)
            # init_time_batched will be used by generate() for time tracking
            init_time_batched = jnp.stack([
                jnp.asarray(valh.get_first_time(last_msg_tokens[b], encoder))
                for b in range(batch_size)])

            m_seq_inp = m_seq_inp[:, cond_len:cond_len+1]  # just the init_token
            b_seq_inp = b_seq_inp[:, n_cond:n_cond+1]      # just the init book
            conditional = False  # skip conditioning scan in generate()

        print('Before generation, real book is (should be none):', real_book)
        if initial:
            initial = False
            if is_1tok:
                # 1tok: no syntax mask needed, use generate_batched_1tok
                valid_mask_array = None
                generate_traced = generate_batched_1tok.trace(
                    sim_init,
                    train_state,
                    model,
                    batchnorm,
                    encoder,
                    sample_top_n,
                    tick_size,
                    m_seq_inp[:],       # (batch, n_cond+1, 24)
                    b_seq_inp,
                    n_gen_msgs,
                    sim_states_init,
                    jax.random.split(rng_, batch_size),
                    init_hidden_batched,
                    conditional,
                    init_time_batched,
                    debug_book,
                    real_book,
                    valid_mask_array,
                )
            else:
                is_transformer = getattr(args, 'model_type', 's5') in ('transformer', 'nsa')
                valid_mask_array = valh.syntax_validation_matrix(
                    block_start_tok=is_transformer)
                generate_traced = generate_batched.trace(
                    sim_init,
                    train_state,
                    model,
                    batchnorm,
                    encoder,
                    sample_top_n,
                    tick_size,
                    m_seq_inp[:],
                    b_seq_inp,
                    n_gen_msgs,
                    sim_states_init,
                    jax.random.split(rng_, batch_size),
                    init_hidden_batched,
                    conditional,
                    init_time_batched,
                    debug_book,
                    real_book,
                    valid_mask_array,
                    *_inject_args,
                )
            generate_lowered = generate_traced.lower()
            generate_compiled = generate_lowered.compile()

        start_time = time.time()
        msgs_decoded, l2_book_states, num_errors, mgs_tokens, provenances = generate_compiled(
            train_state,
            encoder,
            m_seq_inp[:],
            b_seq_inp,
            sim_states_init,
            jax.random.split(rng_, batch_size),
            init_hidden_batched,
            init_time_batched,
            real_book,
            valid_mask_array,
            *_inject_args,
        )
        end_time = time.time()
        print(f"Generation time for batch of size {batch_size}: {(end_time - start_time):.2f} seconds")
        rng, rng_ = jax.random.split(rng)
        # TODO: save as metadata
        print('num_errors', num_errors)

        # only keep actually newly generated messages
        # m_seq_raw_gen = m_seq_raw_gen[-n_gen_msgs:]

        # Detect if msg_seq_raw is encoded (22 cols) vs raw (14 cols)
        msg_is_encoded = (m_seq_raw_eval.shape[-1] == Message_Tokenizer.MSG_LEN)

        # save data for all elements in the batch
        for i, cond_msg, cond_book, real_msg, real_book, gen_msg, gen_book,msg_tok,msg_tok_eval, \
            book_init_i, t_init_i, prov_i \
            in zip(
                batch_i,
                m_seq_raw_inp, b_seq_pv_inp,
                m_seq_raw_eval, b_seq_pv_eval,
                msgs_decoded, l2_book_states,
                mgs_tokens,m_seq_eval,
                book_l2_init, init_time_batched,
                provenances,
            ):

            # get date from filename
            date = ds.get_date(i)
            if overfit_debug:
                jnp.set_printoptions(threshold=sys.maxsize)
                with open(save_folder+f'/tokens/{stock_symbol}_{date}_real_{i}.txt', 'w') as f:
                    print( onp.reshape(msg_tok_eval,(-1,Message_Tokenizer.MSG_LEN)), file=f)

                with open(save_folder+f'/tokens/{stock_symbol}_{date}_gen_{i}.txt', 'w') as f:
                    print( msg_tok, file=f)
                jnp.set_printoptions()

            # Decode encoded messages to 14-col LOBSTER format if needed
            if msg_is_encoded:
                _decode_batch = jax.vmap(encoding.decode_msg, in_axes=(0, None))
                real_msg_dec = onp.array(_decode_batch(jnp.array(real_msg), encoder))
                cond_msg_dec = onp.array(_decode_batch(jnp.array(cond_msg), encoder))
            else:
                real_msg_dec = real_msg
                cond_msg_dec = cond_msg

            # input / cond data
            msg_to_lobster_format(cond_msg_dec).to_csv(
                save_folder + f'/data_cond/{stock_symbol}_{date}_message_real_id_{i}.csv',
                index=False, header=False
            )
            book_to_lobster_format(cond_book).to_csv(
                save_folder + f'/data_cond/{stock_symbol}_{date}_orderbook_real_id_{i}.csv',
                index=False, header=False
            )

            # The exact input to the simulator's initialisation (D-I3). Without
            # it a replay cannot reproduce this run: the generator resets on
            # `book_l2_init`, the WIDE book from before the conditioning
            # window, and then replays the conditioning messages, so orders
            # created during that window exist individually with their real
            # ids, timestamps and queue positions. A replay starting from the
            # 10-level book at the END of the window instead collapses each
            # price level into one synthetic order, which is a different
            # initial state, not a coarser view of the same one.
            #
            # One row: time_s, time_ns, then 4 * wide_levels book columns. The
            # width is therefore self-describing, which is the point (D-I2):
            # the record should not depend on a reader knowing what
            # --wide_levels was set to.
            onp.savetxt(
                save_folder + f'/data_cond/{stock_symbol}_{date}_bookinit_real_id_{i}.csv',
                onp.concatenate([
                    onp.asarray(t_init_i, dtype=onp.int64).reshape(-1)[:2],
                    onp.asarray(book_init_i, dtype=onp.int64).reshape(-1),
                ])[None, :],
                fmt='%d', delimiter=',',
            )

            # The control for D-R4. Provenance bit 3 fires on 60.1% of
            # GENERATED cancels, and on its own that number cannot be
            # interpreted: it means either the model emits inconsistent
            # (price, price_ref) pairs, or the two fields are not comparable
            # the way the check compares them. Nothing in the generated data
            # separates those.
            #
            # Real market messages settle it, because a LOBSTER cancel's own
            # price IS the price of the order it cancels, so real data is
            # consistent by construction. Running the identical check on the
            # real conditioning and continuation streams therefore gives:
            #
            #   a high rate   -> the fields are not comparable, my check is wrong
            #   a near-zero rate -> the model really does emit inconsistent pairs
            #
            # The check reduces to `price != price_ref` on the decoded fields:
            # the absolute form adds the same mid and tick to both sides, so
            # they cancel. That also means the result cannot depend on the mid,
            # only on whether the two fields share a reference frame at all.
            for _tag, _dec, _dir in (("cond", cond_msg_dec, "data_cond"),
                                     ("real", real_msg_dec, "data_real")):
                _d = onp.asarray(_dec)
                _is_cd = onp.isin(_d[:, EVENT_TYPE_i], (2, 3))
                _p, _pr = _d[:, PRICE_i], _d[:, PRICE_REF_i]
                _known = ~(onp.isnan(_p) | onp.isnan(_pr))
                _bit = (_is_cd & _known & (_p != _pr)).astype(onp.int32)
                onp.savetxt(
                    save_folder + f'/{_dir}/{stock_symbol}_{date}_refcheck_real_id_{i}.csv',
                    onp.stack([_is_cd.astype(onp.int32),
                               _known.astype(onp.int32), _bit], axis=1),
                    fmt='%d', delimiter=',',
                )

            # real data
            msg_to_lobster_format(real_msg_dec).to_csv(
                save_folder + f'/data_real/{stock_symbol}_{date}_message_real_id_{i}.csv',
                index=False, header=False
            )
            book_to_lobster_format(real_book).to_csv(
                save_folder + f'/data_real/{stock_symbol}_{date}_orderbook_real_id_{i}.csv',
                index=False, header=False
            )

            # gen data
            msg_to_lobster_format(gen_msg).to_csv(
                save_folder + f'/data_gen/{stock_symbol}_{date}_message_real_id_{i}_gen_id_0.csv',
                index=False, header=False
            )
            book_to_lobster_format(gen_book).to_csv(
                save_folder + f'/data_gen/{stock_symbol}_{date}_orderbook_real_id_{i}_gen_id_0.csv',
                index=False, header=False
            )

            # How each generated message's order id was arrived at (D-R5). A
            # sidecar rather than a seventh column, so every existing reader of
            # the message CSV keeps working and this stays optional.
            #
            #   bits 0-1  0 not a cancel, 1 exact-timestamp hit,
            #             2 price fallback, 3 total miss
            #   bit 2 (4) the NaN branch fired: the simulator got a dummy
            #   bit 3 (8) rel_price_ref disagrees with the message's own price
            #
            # Without this the record shows an order id and nothing about how
            # confident it is, so a resolution failure is indistinguishable
            # from a resolution success.
            onp.savetxt(
                save_folder + f'/data_gen/{stock_symbol}_{date}_provenance_real_id_{i}_gen_id_0.csv',
                onp.asarray(prov_i, dtype=onp.int32).reshape(-1, 1),
                fmt='%d', delimiter=',',
            )

def msg_to_lobster_format(
        m_seq: jax.Array,
) -> pd.DataFrame:
    """
    message format: [time, event_type, order_id, size, price, direction]
    """
    m_seq_ = onp.array(m_seq)[:, [TIMEs_i, TIMEns_i, EVENT_TYPE_i, ORDER_ID_i, SIZE_i, PRICE_ABS_i, DIRECTION_i]]
    m_seq_ = pd.DataFrame(m_seq_, columns=['time_s', 'time_ns', 'event_type', 'order_id', 'size', 'price', 'direction'])

    # combine time field to single field
    m_seq_.insert(
        column = 'time',
        loc = 0,
        value = m_seq_['time_s'].astype(str) \
              + '.' \
              + m_seq_['time_ns'].astype(str).str.pad(width=9, side='left', fillchar='0')
    )
    m_seq_.drop(columns=['time_s', 'time_ns'], inplace=True)

    # convert direction {0,1} to {-1,1}
    m_seq_['direction'] = m_seq_['direction'].replace({0: -1})
    return m_seq_

def book_to_lobster_format(
        b_seq: jax.Array,
    ) -> pd.DataFrame:
    """
    """
    b_seq_ = pd.DataFrame(b_seq)

    return b_seq_


transform_L2_state_batch = jax.jit(
    jax.vmap(
        preproc.transform_L2_state_gpu,
        in_axes=(0, None, None)
    ),
    static_argnums=(1, 2)
)
