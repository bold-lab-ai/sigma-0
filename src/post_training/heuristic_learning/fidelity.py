"""Is the simulator faithful, and to what?

Two arms, kept separate on purpose.

    real arm   replay the REAL continuation messages from the real conditioning
               book and compare against the real book the dataset recorded.
               This tests the engine and this code, against actual market data.

    gen arm    replay the GENERATED messages and compare against the book the
               generator recorded while producing them. This tests the world
               model's record, not the engine.

Separating them is what located the problem. Measured on GS 2026-01, eight
windows of 900 messages: the real arm reproduces the recorded book exactly in
5 of 8 windows (mean exact-match 0.9614), while the gen arm reproduces it in
0 of 8 (mean 0.5511). An engine defect could not produce that asymmetry, since
both arms run the identical code over identically-shaped inputs.

Where each arm's divergence comes from
--------------------------------------
Located by `autopsy.py`, which compares the two book paths as price to
quantity maps at the step where they first disagree. Full detail is in
docs/fidelity_defect_taxonomy.md section 5.5.

    real arm   8 of 8   D-I3, initialisation. The replay did not reproduce the
                        generator's procedure, which is to reset on the wide
                        book from BEFORE the conditioning window and replay the
                        conditioning messages. Narrowing the comparison window
                        moves the median onset 106 -> 346 -> 588 -> 900.

    gen arm    5 of 5   D-R3, of the divergences whose price is inside the
               in-window initial visible range: the recorded message price is not
                        where the generator removed quantity, by 1 to 11 ticks.
                        Depth-invariant, so no initialisation fixes it.

    gen arm    3 of 8   D-I3 as well.

Two earlier attributions in this file were wrong and are corrected below:
D-O1 was credited with the real arm's residual and has zero measured
instances, and D-R5 was called a prerequisite for separating D-R1 from D-R3,
which it is not once you use the replay side's information rather than only
the CSV.

The market a policy trades in is defined as **the engine's evolution of the
saved message stream**: deterministic, self-consistent, byte-identical across
candidate policies. The recorded `gen_book` is a cross-check whose divergence
is reported as world-model record quality, not as simulator error.

Hypotheses tested and refuted along the way
-------------------------------------------
Kept, because a refuted hypothesis stops it being re-investigated, and because
one of these refutations was itself invalid:

    engine order capacity      nOrders 100 vs 1050        bit-identical  (*)
    type-4 semantics           IOC / LIM / MKT            IOC (default) best; MKT 0/8
    cancel fallback strategy   cancel_mode 0..3 x depth   all eight bit-identical
    hidden orders              type-5 in the stream       none present
    index alignment            shift -1 / 0 / +1          shift 0 strictly best on 8/8

(*) Mis-specified, and kept with the correction attached rather than deleted
(D-V1). Varying `nOrders` alone cannot change anything here, because the replay
built `2 * n_levels` init orders either way and never approached either
capacity. The negative result is true and carries no information about
initialisation depth, which is a different parameter the experiment did not
vary. A refuted hypothesis is only as good as the experiment that refuted it.

`cancel_mode` having no effect is the informative one: divergence never reaches
the cancel-fallback path. That, plus the type-4 trigger, was read as evidence
for the within-level queue-order bound (D-O1). **The reasoning was sound and
the conclusion was wrong.** Direct measurement attributes the real arm to D-I3
instead, and D-O1 now stands as a true bound with zero measured instances.
Calling a fixable defect an information bound stops people trying, which is the
failure mode class V exists to catch.

Two criteria are reported, not one. Mid moves only when the touch moves, so a
book can be wrong at every level below it and still score 1.0 (D-O2). For a
market-making PnL the mid criterion is the relevant one; for "did the replay
reproduce the book" it is not.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from post_training.heuristic_learning.episode_builder import (
    fixture_paths,
    mid_from_book,
    read_book_csv,
    read_message_csv,
)


HANDLED_EVENT_TYPES = frozenset({1, 2, 3, 4})

# Mirrored from matching_engine/jaxob/jaxob_constants.py rather than imported,
# so that diagnosing a stream needs no engine (and no jax) at all. A test
# asserts they still agree.
INIT_ID = -2            # cst.INITID
LOOKUP_MISS = -999_999_999   # cst.NEGATIVE_RETURN_ID

# A constant that reaches persisted records is part of the file format, not
# only of the code. Records written before D-R6 was fixed carry -99, and a
# diagnostic that only knew the current value would both undercount them and
# misclassify them as out-of-depth init references. Both are recognised.
LEGACY_LOOKUP_MISS = -99
MISS_SENTINELS = (LOOKUP_MISS, LEGACY_LOOKUP_MISS)


def init_id_span(n_levels: int) -> Tuple[int, int]:
    """The closed range of synthetic ids `init_msgs_from_l2` creates.

    It writes `init_id - arange(0, 2*n_levels)` into the id column, so the ids
    run from `INIT_ID` down to `INIT_ID - (2*n_levels - 1)`. Row j maps to the
    same physical (side, level) at any width, because the reshape is
    `(2L, 2)` over a `[ask_p, ask_q, bid_p, bid_q, ...]` layout. That is why an
    id inside this span names the same order however wide the snapshot was, and
    why one below it names nothing at all.
    """
    return INIT_ID - (2 * n_levels - 1), INIT_ID


@dataclass
class ReplayReport:
    """Everything about a replay that a PnL number alone would hide.

    This is reported next to every PnL rather than on request. A simulator that
    quietly absorbs messages it could not apply produces a plausible number from
    a market that never existed, and nothing downstream can tell.
    """

    n_messages: int = 0
    span_s: float = 0.0
    monotonic: bool = True
    n_negative_deltas: int = 0
    event_types: Dict[int, int] = field(default_factory=dict)
    n_unhandled_types: int = 0
    n_cancel_delete: int = 0
    n_unresolvable_ids: int = 0      # cancel/delete naming an order we never saw created
    n_truncated: int = 0             # messages dropped because a step block overflowed

    # The breakdown of n_unresolvable_ids (D-V3). One counter used to hold all
    # of these, which made a benign phenomenon and a fatal one indistinguishable:
    # a real stream carries 12 to 38 cancels of orders created before the
    # conditioning window and still replays exactly, because the engine's
    # price-based fallback lands where the real cancel landed. A generated
    # stream naming an init order below the replay's depth cannot resolve at
    # all. Reporting one number for both hid that difference for weeks.
    n_unresolved_positive: int = 0    # oid >= 0, unseen: benign, fallback lands right
    n_lookup_miss_sentinel: int = 0   # oid == LOOKUP_MISS: the generator's own lookup failed
    n_init_ref_in_depth: int = 0      # names an init order this replay does create
    n_init_ref_out_of_depth: int = 0  # names an init order that cannot exist here

    # Onset, not just volume. Bad ids that arrive after a divergence has already
    # started explain magnitude; only ones that arrive before it can explain
    # cause. Keeping the step index makes that distinction routine rather than
    # something to rediscover by hand.
    first_bad_id_step: int = -1       # -1 when no non-benign id occurs

    @property
    def clean(self) -> bool:
        return self.monotonic and self.n_unhandled_types == 0 and self.n_truncated == 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_messages": self.n_messages,
            "span_s": round(self.span_s, 3),
            "monotonic": self.monotonic,
            "n_negative_deltas": self.n_negative_deltas,
            "event_types": self.event_types,
            "n_unhandled_types": self.n_unhandled_types,
            "n_cancel_delete": self.n_cancel_delete,
            "n_unresolvable_ids": self.n_unresolvable_ids,
            "n_unresolved_positive": self.n_unresolved_positive,
            "n_lookup_miss_sentinel": self.n_lookup_miss_sentinel,
            "n_init_ref_in_depth": self.n_init_ref_in_depth,
            "n_init_ref_out_of_depth": self.n_init_ref_out_of_depth,
            "first_bad_id_step": self.first_bad_id_step,
            "n_truncated": self.n_truncated,
            "clean": self.clean,
        }


def describe_stream(msgs: np.ndarray, prior_ids: Optional[Sequence[int]] = None,
                    n_levels: int = 10) -> ReplayReport:
    """Diagnose a message stream without running it.

    `prior_ids` are order ids known to exist before the stream starts (typically
    the ids created during the conditioning window). Cancels naming anything
    outside that set and outside the stream's own new orders are counted as
    unresolvable, and then split by *why*, because unresolvable covers two
    phenomena of opposite significance (D-V3):

        oid >= 0, unseen        benign. Real streams carry 12 to 38 per window
                                and still replay exactly: the engine's
                                price-based fallback lands where the real
                                cancel landed.
        oid == LOOKUP_MISS      the generator's own lookup failed and the
                                sentinel was written out as if it were an id.
        init id, in depth       names an order this replay does create, at the
                                same physical level the generator meant.
        init id, out of depth   names an order that cannot exist here at all.

    `n_levels` is the replay's initialisation width and is what separates the
    last two. Getting it wrong silently reclassifies fatal references as benign.
    """
    t = msgs[:, 6].astype(np.int64) * 1_000_000_000 + msgs[:, 7].astype(np.int64)
    deltas = np.diff(t)
    types, counts = np.unique(msgs[:, 0], return_counts=True)
    known = set(int(i) for i in (prior_ids or ()))
    known |= set(msgs[msgs[:, 0] == 1, 4].tolist())

    is_cd = np.isin(msgs[:, 0], [2, 3])
    cd_steps = np.flatnonzero(is_cd)
    lo, hi = init_id_span(n_levels)

    n_unres = n_pos = n_miss = n_in = n_out = 0
    first_bad = -1
    for step in cd_steps:
        oid = int(msgs[step, 4])
        if oid in known:
            continue
        n_unres += 1
        if oid in MISS_SENTINELS:
            n_miss += 1
        elif oid >= 0:
            n_pos += 1
            continue                       # benign: does not count as a bad id
        elif lo <= oid <= hi:
            n_in += 1
        else:
            n_out += 1
        if first_bad < 0:
            first_bad = int(step)

    return ReplayReport(
        n_messages=int(msgs.shape[0]),
        span_s=float(t[-1] - t[0]) / 1e9 if len(t) > 1 else 0.0,
        monotonic=bool((deltas >= 0).all()) if len(deltas) else True,
        n_negative_deltas=int((deltas < 0).sum()) if len(deltas) else 0,
        event_types={int(k): int(v) for k, v in zip(types, counts)},
        n_unhandled_types=int(sum(int(v) for k, v in zip(types, counts)
                                  if int(k) not in HANDLED_EVENT_TYPES)),
        n_cancel_delete=int(cd_steps.size),
        n_unresolvable_ids=n_unres,
        n_unresolved_positive=n_pos,
        n_lookup_miss_sentinel=n_miss,
        n_init_ref_in_depth=n_in,
        n_init_ref_out_of_depth=n_out,
        first_bad_id_step=first_bad,
    )


def replay_stream(msgs: np.ndarray, init_book: np.ndarray, n_levels: int,
                  prior_ids: Optional[Sequence[int]] = None,
                  engine_cfg: Optional[Dict[str, Any]] = None,
                  warmup_msgs: Optional[np.ndarray] = None,
                  init_time: Optional[Tuple[int, int]] = None,
                  ) -> Tuple[np.ndarray, ReplayReport]:
    """Replay a message stream; return the book path at `n_levels` and a report.

    Two initialisation modes, and which one is used matters more than anything
    else this module does (D-I3).

    **Faithful** (`warmup_msgs` given): reproduce `get_sim` in
    lob/inference_no_errcorr.py exactly. Reset on the wide book from *before*
    the conditioning window with the ns component of the timestamp zeroed so
    the synthetic init messages sort ahead of the first real one, then replay
    the conditioning messages. Orders created during that window then exist
    individually, with their real ids, timestamps and queue positions.

    **Degraded** (no `warmup_msgs`): reset on the L2 snapshot at the *end* of
    the conditioning window. `init_msgs_from_l2` collapses each price level
    into one synthetic order, and everything below the snapshot's depth simply
    does not exist. That is a different initial state, not a coarser view of
    the same one, and it is what made the real arm diverge in 8 of 8 windows.

    `n_levels` is the width of the *output*, which must match the reference
    book being compared against. The engine's `book_depth` and `nOrders` come
    from the *input* width, which is much larger in faithful mode.

    One batched call per stream, not one per message: the length is part of the
    jit signature, so a fixed-shape call compiles once and runs ~70x faster
    than the per-message loop it replaces.
    """
    import jax.numpy as jnp
    from matching_engine.jaxob.jaxob_config import JAXLOB_Configuration
    from matching_engine.jaxob.jorderbook import OrderBook

    report = describe_stream(msgs, prior_ids, n_levels)

    init_levels = int(np.asarray(init_book).reshape(-1).shape[0] // 4)
    cfg = dict(engine_cfg or {})
    if warmup_msgs is not None:
        # Mirror inference_no_errcorr.sample_new's sim_init: capacity has to
        # hold 2 * init_levels synthetic orders plus everything the warmup
        # creates, and book_depth decides which ids count as init orders.
        cfg.setdefault("book_depth", init_levels)
        cfg.setdefault("nOrders", max(100, init_levels * 2 + 50 + len(warmup_msgs)))
    sim = OrderBook(cfg=JAXLOB_Configuration(**cfg))

    if init_time is None:
        t0 = (int(msgs[0, 6]), 0)
    else:
        t0 = (int(init_time[0]), 0)   # ns zeroed, as get_sim does
    state = sim.reset(jnp.asarray(init_book, dtype=jnp.int32),
                      jnp.array(t0, dtype=jnp.int32))

    if warmup_msgs is not None and len(warmup_msgs):
        state = sim.process_orders_array(
            state, jnp.asarray(warmup_msgs, dtype=jnp.int32))

    _, l2 = sim.process_orders_array_l2(state, jnp.asarray(msgs, dtype=jnp.int32), n_levels)
    return np.asarray(l2), report


def compare_paths(ours: np.ndarray, reference: np.ndarray, tick_size: int = 100) -> Dict[str, float]:
    """Per-step agreement between two book paths, on two criteria.

    Mid is `(col0 + col2) / 2`, the same convention return_bench uses, so a
    divergence there is directly comparable to a divergence in a return series.
    But mid moves only when the touch moves, so a book can be wrong at every
    level below the touch and still score 1.0 (D-O2). The full-book criterion
    is reported alongside it, and it is the stricter of the two by
    construction: any step where the mids differ also has differing rows.

    Which one to use depends on the question. For a market-making PnL the mid
    criterion is the relevant one, because that is what the strategy trades
    against. For "did the replay reproduce the generator's book", it is not.
    """
    n = min(len(ours), len(reference))
    d = np.abs(mid_from_book(ours[:n]) - mid_from_book(reference[:n])) / tick_size
    first = int(np.argmax(d != 0)) if (d != 0).any() else -1

    row_equal = np.all(ours[:n] == reference[:n], axis=1)
    first_full = int(np.argmax(~row_equal)) if (~row_equal).any() else -1
    return {
        "exact_full_frac": float(row_equal.mean()) if n else 0.0,
        "first_full_divergence_step": float(first_full),
        "n_compared": float(n),
        "exact_mid_frac": float((d == 0).mean()) if n else 0.0,
        "mean_abs_ticks": float(d.mean()) if n else 0.0,
        "max_abs_ticks": float(d.max()) if n else 0.0,
        "first_divergence_step": float(first),
    }


# ------------------------------------------------------------------ report --
def discover_windows(run_dir: Path, stock: str, limit: Optional[int] = None) -> List[Tuple[str, int]]:
    """(date, index) pairs in a generated run directory.

    `os.scandir` is lazy and the loop breaks early when a limit is given, so
    this reads only as many directory entries as it needs. These directories can
    hold thousands of files and this filesystem's policy forbids scanning them.
    """
    out: List[Tuple[str, int]] = []
    pat = re.compile(rf"{re.escape(stock)}_(\d{{4}}-\d{{2}}-\d{{2}})_message_real_id_(\d+)_gen_id_0\.csv")
    with os.scandir(run_dir / "data_gen") as it:
        for entry in it:
            m = pat.match(entry.name)
            if not m:
                continue
            out.append((m.group(1), int(m.group(2))))
            if limit and len(out) >= limit:
                break
    return sorted(out)


#: Engine parameters a replay must match, and which the record may declare.
#: `book_depth` and `nOrders` are deliberately absent: faithful mode derives
#: them from the width of the initialisation book, which is better than a
#: manifest because the data describes itself and cannot disagree with itself.
THETA_KEYS = ("tick_size", "init_id")


def load_theta(run_dir: Path) -> Dict[str, Any]:
    """Engine config declared by the run, if it declared any (D-T1).

    The replay used to construct `JAXLOB_Configuration()` with defaults and
    hope. Any generator-side change to `theta` then desynchronised the two
    sides silently, which is the whole of class T. Reading it makes the
    dependency explicit; the manifest not carrying it yet (D-T3) is why this
    returns an empty dict rather than raising.
    """
    for candidate in (run_dir / "manifest.json",
                      run_dir.parent / "manifest.json",
                      run_dir.parent.parent / "manifest.json"):
        if candidate.is_file():
            try:
                m = json.loads(candidate.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
            engine = m.get("engine", m)
            return {k: engine[k] for k in THETA_KEYS if k in engine}
    return {}


def window_report(run_dir: Path, stock: str, date: str, index: int,
                  engine_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    p = fixture_paths(run_dir, stock, date, index)
    # Declared theta first, caller override on top: an explicit argument is an
    # experiment, the manifest is the record of what actually ran.
    theta = {**load_theta(run_dir), **(engine_cfg or {})}
    tick = int(theta.pop("tick_size", 100))
    cond_book = read_book_csv(p["cond_book"])
    n_levels = cond_book.shape[1] // 4
    cond_msgs = read_message_csv(p["cond_msg"])
    prior_ids = cond_msgs[cond_msgs[:, 0] == 1, 4].tolist()

    # Faithful initialisation when the run recorded what the generator actually
    # initialised from; the degraded path otherwise, so older runs still replay
    # and the two can be compared (D-I3).
    init_book, warmup, init_time = cond_book[-1], None, None
    if p["cond_book_init"].is_file():
        row = np.loadtxt(p["cond_book_init"], delimiter=",", dtype=np.int64).reshape(-1)
        init_time, init_book = (int(row[0]), int(row[1])), row[2:]
        warmup = cond_msgs

    out: Dict[str, Dict[str, Any]] = {}
    for arm, mk, bk in (("real", "real_msg", "real_book"), ("gen", "gen_msg", "gen_book")):
        if not (p[mk].is_file() and p[bk].is_file()):
            continue
        msgs = read_message_csv(p[mk])
        ref = read_book_csv(p[bk])
        ours, rep = replay_stream(msgs, init_book, ref.shape[1] // 4, prior_ids,
                                  theta, warmup_msgs=warmup, init_time=init_time)
        out[arm] = {**compare_paths(ours, ref, tick), **rep.as_dict()}
    return out


def _cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.episodes)
    windows = discover_windows(run_dir, args.stock, args.n)
    if not windows:
        print(f"no generated windows under {run_dir / 'data_gen'}")
        return 1

    rows: Dict[str, List[float]] = {"real": [], "gen": []}
    rows_full: Dict[str, List[float]] = {"real": [], "gen": []}
    # `onset` is the column that matters for causation: it says whether the
    # first non-benign id arrived BEFORE the first divergence. A bad id that
    # arrives after divergence has begun can only have amplified it.
    print(f"{'window':22s} {'arm':5s} {'msgs':>5s} {'mid':>7s} {'full':>7s} "
          f"{'mean_tk':>8s} {'1st_div':>8s} {'1st_bad':>8s} {'onset':>6s} "
          f"{'pos':>4s} {'-99':>4s} {'in':>4s} {'out':>4s} {'clean':>6s}")
    for date, idx in windows:
        rep = window_report(run_dir, args.stock, date, idx)
        for arm in ("real", "gen"):
            if arm not in rep:
                continue
            r = rep[arm]
            rows[arm].append(r["exact_mid_frac"])
            rows_full[arm].append(r["exact_full_frac"])
            div, bad = int(r["first_divergence_step"]), int(r["first_bad_id_step"])
            if div < 0:
                onset = "-"                      # nothing diverged
            elif bad < 0 or bad >= div:
                onset = "NO"                     # bad ids cannot have caused it
            else:
                onset = "maybe"                  # a bad id precedes the divergence
            print(f"{date + ' #' + str(idx):22s} {arm:5s} {int(r['n_messages']):5d} "
                  f"{r['exact_mid_frac']:7.4f} {r['exact_full_frac']:7.4f} "
                  f"{r['mean_abs_ticks']:8.2f} "
                  f"{div:8d} {bad:8d} {onset:>6s} "
                  f"{r['n_unresolved_positive']:4d} {r['n_lookup_miss_sentinel']:4d} "
                  f"{r['n_init_ref_in_depth']:4d} {r['n_init_ref_out_of_depth']:4d} "
                  f"{str(r['clean']):>6s}")

    print()
    for arm in ("real", "gen"):
        v = rows[arm]
        if not v:
            continue
        f = rows_full[arm]
        n_exact = sum(1 for x in v if x == 1.0)
        n_full = sum(1 for x in f if x == 1.0)
        print(f"{arm:5s} arm: mid {n_exact}/{len(v)} exact, mean {np.mean(v):.4f}   |   "
              f"full book {n_full}/{len(f)} exact, mean {np.mean(f):.4f}")
    print("\nreal arm tests the engine against market data; gen arm tests the "
          "world model's record. They are not comparable acceptance bars.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="real-arm and gen-arm fidelity across a run's windows")
    r.add_argument("--episodes", required=True, help="a generated run dir (holds data_cond/data_gen/data_real)")
    r.add_argument("--stock", default="GS")
    r.add_argument("--n", type=int, default=None, help="limit the number of windows")
    r.set_defaults(func=_cmd_report)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------- provenance --
#: How a generated message's order id was arrived at (D-R5). A bit field
#: written by `get_sim_msg`, so the record states what its resolution did
#: instead of leaving a reader to infer it from an id that looks like any
#: other. Runs generated before D-R5 have no provenance file at all.
RESOLUTION = {0: "not-a-cancel", 1: "exact-timestamp-hit",
              2: "price-fallback", 3: "total-miss"}
PROV_NAN_FIRED = 4        # the simulator was handed a dummy (D-R1)
#: `price != price_ref` on the decoded fields. NOT a defect indicator, and it
#: was labelled as one until the control ran. Real market messages are
#: consistent by construction, and the same check fires on 55.8% of their
#: cancels (63.2% conditioning, 51.7% continuation) against 63.4% of generated
#: ones. The two fields simply do not share a reference frame, so their
#: inequality means nothing about the model. Kept because it is a real,
#: well-defined datum and because deleting it would delete the evidence that
#: the control was run; renamed because the old name asserted a conclusion the
#: control refuted.
PROV_PRICE_DIFFERS_FROM_REF = 8


def decode_provenance(code: int) -> Dict[str, Any]:
    """One integer into the three facts it encodes."""
    return {
        "resolution": RESOLUTION[int(code) & 0b11],
        "nan_branch_fired": bool(int(code) & PROV_NAN_FIRED),
        "price_differs_from_ref": bool(int(code) & PROV_PRICE_DIFFERS_FROM_REF),
    }


def summarise_provenance(codes: Sequence[int]) -> Dict[str, int]:
    """Counts a reader can act on, rather than a histogram of opaque integers.

    `nan_branch_fired` is D-R1's firing rate, which had never been measured
    because the record carried nothing to measure it with.
    `price_differs_from_ref` is NOT a defect count. It was introduced as one
    and the control refuted it: real messages, consistent by construction,
    trip it at the same rate. See D-R4.
    """
    out = {f"resolution_{v}": 0 for v in RESOLUTION.values()}
    out["nan_branch_fired"] = 0
    out["price_differs_from_ref"] = 0
    for c in codes:
        d = decode_provenance(c)
        out[f"resolution_{d['resolution']}"] += 1
        out["nan_branch_fired"] += d["nan_branch_fired"]
        out["price_differs_from_ref"] += d["price_differs_from_ref"]
    return out
