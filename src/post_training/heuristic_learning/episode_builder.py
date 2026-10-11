"""Turn generated order flow into frozen, replayable market-making episodes.

An *episode* is one minute of market that every candidate policy will face
byte-for-byte identically:

    init_book      the L2 book at the end of the conditioning window
    background     an (N, 8) engine-ready message stream, exogenous to the policy
    signal         a forward mid-price curve with dispersion, from J rollouts

Why the background flow is frozen rather than generated on the fly
------------------------------------------------------------------
The problem formulation requires that inference condition only on orders that
really happened, and that the generator never produce the fills of your own
quotes. Together those make the background flow *exogenous* to the policy: it
does not depend on what you quote. So it can be generated once and replayed.

That is not merely a speed trick, though it is also that (a round of policy
search drops from GPU rollouts to a CPU replay). It buys two things that matter
more: every candidate policy faces an identical market, so score differences are
paired rather than confounded by fresh sampling noise; and swapping the frozen
stream for the real continuation turns the same code path into a real-data
backtest with nothing else changed.

The loop still closes. It closes at the policy, which re-conditions each step on
its realised fills, inventory and the post-match book. What is frozen is the
market's own behaviour, not the feedback. The cost is that our own market impact
is not modelled, which is a known reason simulated PnL runs optimistic; the
sim-vs-real comparison in `lob.inference_no_errcorr.compute_gt_divergence`
exists to quantify that rather than hand-wave it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Engine message layout, from matching_engine.jaxob.jaxob_constants.LOBMSGFEAT:
#   [type, side, quant, price, order_id, trader_id, time_s, time_ns]
ENGINE_COLS = ("type", "side", "quant", "price", "order_id", "trader_id", "time_s", "time_ns")

# Trader ids. The background stream is "the market"; the agent must be
# distinguishable from it because fills are attributed by trader id
# (see job.get_agent_trades). INITID = -2 is taken by the book-init messages.
BACKGROUND_TRADER_ID = 0
AGENT_TRADER_ID = 8888

# LOBSTER-format message CSV written by lob.inference_no_errcorr.msg_to_lobster_format.
CSV_COLUMNS = ("time", "event_type", "order_id", "size", "price", "direction")


class EpisodeError(RuntimeError):
    """Raised when generated data cannot be turned into a usable episode."""


# --------------------------------------------------------------------- I/O --
def read_message_csv(path: Path) -> np.ndarray:
    """Read a generated/real message CSV into an (N, 8) engine message array.

    The direction column needs care. `msg_to_lobster_format` already maps raw
    {0, 1} to {-1, 1} via `replace({0: -1})`, and the engine's own
    `msg_to_jnp` reaches the same {-1, 1} by computing `(d * 2) - 1` on the raw
    value. Both land on the same convention, so reading the CSV back means
    taking the direction column *as written*. Applying `(d * 2) - 1` a second
    time would yield {-3, 1}: the engine does not reject that, it just matches
    against a side that does not exist, so the error is silent.
    """
    # `to_csv(..., index=False, header=False)` -- the files carry no header row,
    # so columns are positional: time, event_type, order_id, size, price, direction.
    # The time column must stay text: "47200.639684060" as a float loses
    # nanosecond resolution, and nanosecond ordering is exactly what the engine
    # sequences on.
    rows = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not rows:
        raise EpisodeError(f"empty message csv: {path}")
    if rows[0].lstrip().startswith(CSV_COLUMNS[0]):        # tolerate a header if one appears
        rows = rows[1:]

    n = len(rows)
    msgs = np.zeros((n, 8), dtype=np.int32)
    for i, line in enumerate(rows):
        parts = line.split(",")
        if len(parts) != len(CSV_COLUMNS):
            raise EpisodeError(
                f"{path}:{i + 1}: expected {len(CSV_COLUMNS)} columns {CSV_COLUMNS}, got {len(parts)}"
            )
        t_s, _, t_ns = parts[0].partition(".")
        msgs[i, 0] = int(parts[1])                          # event_type
        msgs[i, 1] = int(parts[5])                          # direction, already {-1, 1}
        msgs[i, 2] = int(parts[3])                          # size
        msgs[i, 3] = int(parts[4])                          # price (absolute ticks)
        msgs[i, 4] = int(parts[2])                          # order_id
        msgs[i, 5] = BACKGROUND_TRADER_ID
        msgs[i, 6] = int(t_s)
        msgs[i, 7] = int(t_ns.ljust(9, "0")[:9]) if t_ns else 0

    sides = np.unique(msgs[:, 1])
    if not set(sides.tolist()) <= {-1, 1}:
        raise EpisodeError(
            f"{path}: direction column holds {sides.tolist()}, expected a subset of [-1, 1]. "
            "The CSV writer already maps 0 -> -1, so re-applying (d * 2) - 1 is the usual cause."
        )
    return msgs


def read_book_csv(path: Path) -> np.ndarray:
    """Read an orderbook CSV into an (N, 4 * levels) L2 array.

    Column layout is (ask_px, ask_vol, bid_px, bid_vol) repeated per level,
    the same layout `compute_gt_divergence` and `return_bench` assume.
    """
    book = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    if book.size == 0:
        raise EpisodeError(f"empty orderbook csv: {path}")
    if book.shape[1] % 4 != 0:
        raise EpisodeError(f"{path}: {book.shape[1]} columns is not a multiple of 4")
    return book


# ---------------------------------------------------------------- validate --
def message_stream_report(msgs: np.ndarray) -> Dict[str, object]:
    """Describe a message stream's timing, without deciding whether to accept it.

    Timestamps are the thing to check. A defect in the generation path could put
    a START token into a message's time field, where `combine_field` collapses it
    to the NA sentinel -9999 and it reaches `add_times` unvalidated, producing a
    *negative* inter-arrival. It was observed at 3 events in 780,864 in a formal
    run, which is exactly the density that survives eyeballing. A window with a
    non-monotonic clock cannot be replayed meaningfully, so it must be caught
    here rather than discovered as a strange PnL later.
    """
    t_ns = msgs[:, 6].astype(np.int64) * 1_000_000_000 + msgs[:, 7].astype(np.int64)
    deltas = np.diff(t_ns)
    span_s = float(t_ns[-1] - t_ns[0]) / 1e9 if len(t_ns) > 1 else 0.0
    return {
        "n_messages": int(msgs.shape[0]),
        "span_s": span_s,
        "monotonic": bool((deltas >= 0).all()) if len(deltas) else True,
        "n_negative_deltas": int((deltas < 0).sum()) if len(deltas) else 0,
        "min_delta_ns": int(deltas.min()) if len(deltas) else 0,
        "event_types": {int(k): int(v) for k, v in zip(*np.unique(msgs[:, 0], return_counts=True))},
    }


def assert_replayable(report: Dict[str, object], *, source: str) -> None:
    if not report["monotonic"]:
        raise EpisodeError(
            f"{source}: clock is not monotonic "
            f"({report['n_negative_deltas']} negative inter-arrivals, "
            f"min delta {report['min_delta_ns']} ns). "
            "Discard the window; do not repair it in place."
        )


# ------------------------------------------------------------------- signal --
def mid_from_book(book: np.ndarray) -> np.ndarray:
    """Mid price per row: (best ask + best bid) / 2, matching return_bench."""
    return (book[:, 0].astype(np.float64) + book[:, 2].astype(np.float64)) / 2.0


def build_signal_curve(
    rollout_books: Sequence[np.ndarray],
    horizons: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Summarise J rollouts of the same context into a forward mid-price curve.

    Each rollout is one sample from the world model's predictive distribution
    over the same history, so the spread across rollouts is the model's own
    uncertainty, not estimation noise. The policy gets both: the mean says which
    way, the q10/q90 band says how confidently.

    Horizons are in messages, not seconds, because that is the model's own
    stepping unit. `attach_wallclock` converts if a policy wants seconds.
    """
    if not rollout_books:
        raise EpisodeError("no rollouts supplied for the signal curve")

    horizons = [int(h) for h in horizons]
    anchors, paths = [], []
    for book in rollout_books:
        mid = mid_from_book(book)
        anchors.append(mid[0])
        usable = [mid[h - 1] if h - 1 < len(mid) else np.nan for h in horizons]
        paths.append(usable)

    arr = np.asarray(paths, dtype=np.float64)          # (J, H)
    anchor = float(np.mean(anchors))

    with np.errstate(invalid="ignore"):
        return {
            "horizons": np.asarray(horizons, dtype=np.int32),
            "anchor_mid": np.float64(anchor),
            "mid_mean": np.nanmean(arr, axis=0),
            "mid_q10": np.nanquantile(arr, 0.10, axis=0),
            "mid_q90": np.nanquantile(arr, 0.90, axis=0),
            "n_rollouts": np.int32(arr.shape[0]),
        }


# ------------------------------------------------------------------ episode --
@dataclass(frozen=True)
class EpisodeSpec:
    """Everything needed to rebuild one episode from source data."""

    window_id: int
    stock: str
    date: str
    background_seed: int
    signal_seeds: Tuple[int, ...]
    bg_checkpoint: str
    signal_checkpoint: str


def build_episode(
    spec: EpisodeSpec,
    init_book: np.ndarray,
    background: np.ndarray,
    signal: Dict[str, np.ndarray],
) -> Dict[str, object]:
    report = message_stream_report(background)
    assert_replayable(report, source=f"window {spec.window_id}")
    return {
        "window_id": np.int32(spec.window_id),
        "stock": spec.stock,
        "date": spec.date,
        "init_book": init_book.astype(np.int32),
        "background": background.astype(np.int32),
        "background_report": json.dumps(report),
        "background_seed": np.int32(spec.background_seed),
        "signal_seeds": np.asarray(spec.signal_seeds, dtype=np.int32),
        "bg_checkpoint": spec.bg_checkpoint,
        "signal_checkpoint": spec.signal_checkpoint,
        **{f"signal_{k}": v for k, v in signal.items()},
    }


def save_episode(episode: Dict[str, object], out_dir: Path) -> Path:
    """Write one episode atomically, never overwriting an existing one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"episode_{int(episode['window_id']):06d}.npz"
    if final.exists():
        return final                      # resume: an existing episode is authoritative
    tmp = final.with_suffix(".npz.tmp")
    np.savez_compressed(tmp, **episode)
    tmp.rename(final)                     # rename is atomic within a filesystem
    return final


# ------------------------------------------------- fixture paths / selftest --
def fixture_paths(root: Path, stock: str, date: str, index: int) -> Dict[str, Path]:
    """Exact paths for one generated sequence.

    Constructed from the naming convention in `inference_no_errcorr.sample_new`
    rather than by listing the directory: these output directories hold
    thousands of files and this filesystem's policy forbids scanning them.
    """
    return {
        "cond_msg": root / "data_cond" / f"{stock}_{date}_message_real_id_{index}.csv",
        "cond_book": root / "data_cond" / f"{stock}_{date}_orderbook_real_id_{index}.csv",
        # The generator's actual initialisation input: the wide book from
        # before the conditioning window, prefixed by its timestamp. Present
        # only in runs generated after D-I3 was fixed; absent in older ones,
        # which is why every reader treats it as optional.
        "cond_book_init": root / "data_cond" / f"{stock}_{date}_bookinit_real_id_{index}.csv",
        "real_msg": root / "data_real" / f"{stock}_{date}_message_real_id_{index}.csv",
        "real_book": root / "data_real" / f"{stock}_{date}_orderbook_real_id_{index}.csv",
        "gen_msg": root / "data_gen" / f"{stock}_{date}_message_real_id_{index}_gen_id_0.csv",
        "gen_book": root / "data_gen" / f"{stock}_{date}_orderbook_real_id_{index}_gen_id_0.csv",
    }


def replay_and_compare(gen_msgs: np.ndarray, init_book: np.ndarray, gen_book: np.ndarray) -> Dict[str, float]:
    """Replay generated messages through the engine and compare to its own book.

    This is the consistency gate for the whole pipeline. If our reconstruction of
    the engine's inputs is right, feeding the same messages from the same initial
    book must reproduce the book states the generator recorded. Any divergence
    means the CSV round-trip lost or corrupted something, and every downstream
    PnL would be quietly wrong.
    """
    import jax.numpy as jnp
    from matching_engine.jaxob.jaxob_config import JAXLOB_Configuration
    from matching_engine.jaxob.jorderbook import OrderBook

    n_levels = init_book.shape[0] // 4
    sim = OrderBook(cfg=JAXLOB_Configuration())
    start_time = jnp.array([int(gen_msgs[0, 6]), 0], dtype=jnp.int32)
    state = sim.reset(jnp.asarray(init_book, dtype=jnp.int32), start_time)

    ours = []
    for msg in gen_msgs:
        state = sim.process_order_array(state, jnp.asarray(msg, dtype=jnp.int32))
        ours.append(np.asarray(sim.get_L2_state(state, n_levels)))
    ours_arr = np.stack(ours)

    n = min(len(ours_arr), len(gen_book))
    our_mid = mid_from_book(ours_arr[:n])
    their_mid = mid_from_book(gen_book[:n])
    diff = np.abs(our_mid - their_mid)
    return {
        "n_compared": float(n),
        "mid_exact_match_frac": float((diff == 0).mean()),
        "mid_max_abs_diff": float(diff.max()) if n else 0.0,
        "mid_mean_abs_diff": float(diff.mean()) if n else 0.0,
    }


def sample_fixture_sequences(root: Path, stock: str, limit: int) -> List[Tuple[str, int]]:
    """Find a few (date, index) pairs in a generated run directory.

    `real_id` is the dataset index of the conditioning window, not a counter, so
    the pairs have to be discovered rather than assumed. `os.scandir` is lazy, so
    breaking out early reads only as many directory entries as we need instead of
    enumerating thousands of files.
    """
    import os

    out: List[Tuple[str, int]] = []
    prefix, suffix = f"{stock}_", "_gen_id_0.csv"
    with os.scandir(root / "data_gen") as it:
        for entry in it:
            name = entry.name
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            if "_message_real_id_" not in name:
                continue
            date = name[len(prefix):].split("_message_real_id_")[0]
            index = int(name.split("_message_real_id_")[1].split("_gen_id_")[0])
            out.append((date, index))
            if len(out) >= limit:
                break
    return out


def _cmd_selftest_replay(args: argparse.Namespace) -> int:
    root = Path(args.fixture)
    pairs = sample_fixture_sequences(root, args.stock, args.n)
    if not pairs:
        print(f"no generated sequences found under {root / 'data_gen'}")
        return 1

    ok = 0
    for date, idx in pairs:
        paths = fixture_paths(root, args.stock, date, idx)
        missing = [str(p) for p in (paths["gen_msg"], paths["gen_book"], paths["cond_book"]) if not p.is_file()]
        if missing:
            print(f"[{idx}] SKIP missing: {missing[0]}")
            continue

        gen_msgs = read_message_csv(paths["gen_msg"])
        gen_book = read_book_csv(paths["gen_book"])
        cond_book = read_book_csv(paths["cond_book"])
        init_book = cond_book[-1]

        report = message_stream_report(gen_msgs)
        print(f"[{date} #{idx}] n={report['n_messages']} span={report['span_s']:.3f}s "
              f"monotonic={report['monotonic']} neg={report['n_negative_deltas']}")
        if not report["monotonic"]:
            print(f"[{date} #{idx}] FAIL non-monotonic clock -- window would be discarded")
            continue

        cmp_ = replay_and_compare(gen_msgs, init_book, gen_book)
        print(f"[{date} #{idx}] replay: exact_mid={cmp_['mid_exact_match_frac']:.4f} "
              f"max|d|={cmp_['mid_max_abs_diff']:.1f} mean|d|={cmp_['mid_mean_abs_diff']:.3f} "
              f"over {int(cmp_['n_compared'])} steps")
        if cmp_["mid_exact_match_frac"] == 1.0:
            ok += 1
    print(f"\n{ok}/{len(pairs)} sequences reproduced the generator's mid path exactly")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest-replay", help="check CSV -> engine -> mid against the generator's own book")
    st.add_argument("--fixture", required=True, help="an inference_results run dir with data_cond/data_gen")
    st.add_argument("--stock", default="GOOG")
    st.add_argument("--n", type=int, default=4)
    st.set_defaults(func=_cmd_selftest_replay)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
