"""Run the whole bench on synthetic inference output, and on deliberately broken output.

Why this exists, separately from the recorded evidence:

`ci/evidence/return_bench.json` carries both the measured numbers and the
assertions they must satisfy, and both were written by the same run. If the
implementation were wrong, the recorded number would be wrong too, and the
assertion could have been written to accommodate it. Auditing that file proves
the numbers have not drifted from the code that produced them -- which is worth
having, and has already caught real drift -- but it cannot prove the numbers
were right in the first place.

Nothing here reads the evidence. These tests build inference directories from
scratch, run the bench over them, and check the answers against what the inputs
make true by construction. Crucially they also break the inputs in specific ways
and require the bench to notice; a check that only ever sees correct input
cannot distinguish a working implementation from one that returns 1.0
unconditionally.

No model, no GPU, no cluster, no market data. numpy, pandas and scipy only, so
this runs on a bare CI runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lobpipeline.return_bench.run_event_bench import build_table, collect, find_paired_files

N_COND, N_GEN = 20, 30
HORIZONS = [1.0, 5.0, 10.0]
TICKER, DATE = "SYN", "2026-03-02"


def _write(path, rows, ncols):
    """LOBSTER-shaped csv: message rows are 6 wide, book rows are 40."""
    pd.DataFrame(rows).to_csv(path, header=False, index=False)
    assert len(rows[0]) == ncols


def _messages(n, t0, dt_ms, last_is_trade, direction=1, seed=0):
    """n messages spaced dt_ms apart; optionally force the final one to be a trade."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t = t0 + i * dt_ms / 1000.0
        et = int(rng.choice([1, 2, 3]))
        d = int(rng.choice([-1, 1]))
        if last_is_trade and i == n - 1:
            et, d = 4, direction
        rows.append([f"{t:.9f}", et, 1000 + i, 100, 3000000, d])
    return rows


def _book(n, mid_path):
    """Book rows whose top of book reproduces a given mid-price path."""
    rows = []
    for m in mid_path[:n]:
        ask, bid = m + 50, m - 50          # mid = (ask + bid) / 2 exactly
        rows.append([ask, 100, bid, 100] + [0] * 36)
    return rows


def _mid_path(n, start=3000000, step=100, seed=1):
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.choice([-step, 0, step], size=n))


def build_infer_dir(tmp_path, n_seq=8, gen_book_shift=0, gen_time_offset_ms=0.0,
                    anchor_is_trade=True):
    """A minimal inference directory the bench can score.

    gen_book_shift      offset the generated book by this many rows against its own
                        messages, i.e. a book/message misalignment
    gen_time_offset_ms  shift every generated timestamp, i.e. a clock error
    anchor_is_trade     whether the last conditioning message is an execution
    """
    for sub in ("data_cond", "data_gen", "data_real"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    for k in range(n_seq):
        rid = 10000 + k
        t_start = 34200.0 + 60 * k
        cond_msgs = _messages(N_COND, t_start, dt_ms=2.0, last_is_trade=anchor_is_trade,
                              direction=1 if k % 2 == 0 else -1, seed=k)
        t0 = float(cond_msgs[-1][0])

        # conditioning book carries one extra row: the state after the last message
        cond_mid = _mid_path(N_COND + 1, seed=100 + k)
        real_mid = _mid_path(N_GEN, start=int(cond_mid[-1]), seed=200 + k)
        gen_mid = _mid_path(N_GEN, start=int(cond_mid[-1]), seed=300 + k)

        real_msgs = _messages(N_GEN, t0 + 0.0005, dt_ms=1.5, last_is_trade=False, seed=400 + k)
        gen_msgs = _messages(N_GEN, t0 + 0.0005 + gen_time_offset_ms / 1000.0,
                             dt_ms=1.5, last_is_trade=False, seed=400 + k)

        base = f"{TICKER}_{DATE}"
        _write(tmp_path / "data_cond" / f"{base}_message_real_id_{rid}.csv", cond_msgs, 6)
        _write(tmp_path / "data_cond" / f"{base}_orderbook_real_id_{rid}.csv",
               _book(N_COND + 1, cond_mid), 40)
        _write(tmp_path / "data_real" / f"{base}_message_real_id_{rid}.csv", real_msgs, 6)
        _write(tmp_path / "data_real" / f"{base}_orderbook_real_id_{rid}.csv",
               _book(N_GEN, real_mid), 40)
        _write(tmp_path / "data_gen" / f"{base}_message_real_id_{rid}_gen_id_0.csv", gen_msgs, 6)

        gb = gen_mid if gen_book_shift == 0 else np.roll(gen_mid, gen_book_shift)
        _write(tmp_path / "data_gen" / f"{base}_orderbook_real_id_{rid}_gen_id_0.csv",
               _book(N_GEN, gb), 40)
    return tmp_path


def score(d, identity=False, require_trade=True):
    pairs = find_paired_files(str(d))
    rec, st, _ = collect(pairs, HORIZONS, workers=4, identity=identity,
                         require_trade=require_trade)
    if st["n_used"] == 0:
        return None, st
    return build_table(rec, HORIZONS, st["n_used"]), st


# --------------------------------------------------------------------------------
# The bench answers correctly on inputs whose answer is known by construction
# --------------------------------------------------------------------------------

def test_identity_is_exact_on_synthetic_data(tmp_path):
    """The claim the recorded evidence rests on, recomputed here from scratch."""
    d = build_infer_dir(tmp_path, n_seq=8)
    df, st = score(d, identity=True)
    assert st["n_used"] == 8
    for _, r in df.iterrows():
        assert r["IC"] == pytest.approx(1.0, abs=1e-12)
        assert r["ranked_IC"] == pytest.approx(1.0, abs=1e-12)
        assert r["direction_acc"] == pytest.approx(1.0, abs=1e-12)
        assert r["ks_stat"] == pytest.approx(0.0, abs=1e-12)
        assert r["std_ratio"] == pytest.approx(1.0, abs=1e-12)


def test_every_scored_window_is_anchored_on_a_trade(tmp_path):
    d = build_infer_dir(tmp_path, n_seq=8, anchor_is_trade=True)
    _, st = score(d)
    assert st["n_used"] == 8
    assert st["n_not_trade"] == 0


# --------------------------------------------------------------------------------
# And it notices when the inputs are wrong. Without these, an implementation that
# returned 1.0 unconditionally would pass everything above.
# --------------------------------------------------------------------------------

def test_windows_not_anchored_on_a_trade_are_rejected(tmp_path):
    d = build_infer_dir(tmp_path, n_seq=8, anchor_is_trade=False)
    df, st = score(d)
    assert st["n_used"] == 0 and st["n_not_trade"] == 8
    assert df is None


def test_a_misaligned_generated_book_does_not_score_as_identical(tmp_path):
    """Roll the generated book against its own messages by one row.

    This is the off-by-one that produces perfectly plausible numbers on real data.
    Scoring gen against real must not be blind to it.
    """
    ok = build_infer_dir(tmp_path / "ok", n_seq=8)
    bad = build_infer_dir(tmp_path / "bad", n_seq=8, gen_book_shift=1)
    df_ok, _ = score(ok, identity=True)
    df_bad, _ = score(bad, identity=True)
    # identity mode replaces gen with real, so it stays exact either way ...
    assert df_ok["IC"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    # ... but the real comparison must differ between the two directories.
    a, _ = score(ok)
    b, _ = score(bad)
    assert not np.allclose(a["IC"].to_numpy(), b["IC"].to_numpy(), atol=1e-9), (
        "a one-row book misalignment left every IC unchanged; the bench is not "
        "reading the generated book it was given")


def test_a_generated_clock_offset_is_visible(tmp_path):
    """Shift every generated timestamp and require the clock diagnostic to move."""
    ok = build_infer_dir(tmp_path / "ok", n_seq=8)
    slow = build_infer_dir(tmp_path / "slow", n_seq=8, gen_time_offset_ms=4.0)
    a, _ = score(ok)
    b, _ = score(slow)
    # Same real stream on both sides, so any change is the generated clock.
    assert (b["gen_msg"] < a["gen_msg"]).any(), (
        "offsetting every generated timestamp changed no message count; the clock "
        "diagnostic is not reading the generated timestamps")
    assert np.allclose(a["real_msg"].to_numpy(), b["real_msg"].to_numpy())


def test_coverage_falls_when_a_horizon_outruns_the_sequence(tmp_path):
    """A horizon past the end of the data must lose samples, not reuse the last row."""
    d = build_infer_dir(tmp_path, n_seq=8)
    pairs = find_paired_files(str(d))
    # The synthetic streams span ~45 ms; 10 s is far beyond them.
    rec, st, _ = collect(pairs, [1.0, 10_000.0], workers=4, require_trade=True)
    df = build_table(rec, [1.0, 10_000.0], st["n_used"])
    near, far = df.iloc[0], df.iloc[1]
    assert near["N_cov"] == 8
    assert far["N_cov"] == 0, (
        "a 10 s horizon over a 45 ms sequence retained samples; searchsorted is "
        "silently returning the final row")
