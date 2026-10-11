"""Replay fidelity against real market data, pinned as a regression baseline.

The real arm is the one that means something: real continuation messages
replayed from the real conditioning book, compared against the book the dataset
recorded. `real_book` comes from the dataset (lob/inference_no_errcorr.py:1817
builds it from b_seq_inp / b_seq_eval), not from the engine, so agreement here
is a genuine check rather than the engine agreeing with itself.

The numbers are pinned rather than merely reported because both directions
matter. A drop means the engine or this code regressed. A jump means someone
changed replay semantics and should say which of the refuted explanations they
overturned: order capacity, type-4 interpretation, cancel fallback, hidden
orders, and index alignment were each tested and ruled out by measurement.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from post_training.heuristic_learning.fidelity import discover_windows, window_report


# Not welded to one timestamped directory (D-V4). Override with
# SIGMA_EPISODES_DIR to point at another run.
DEFAULT_RUN_DIR = (
    "/lus/lfs1aip2/projects/public/u6gb/tasks/agentic_mm_20260801T144426Z"
    "/generated/background_seed2026"
)
RUN_DIR = Path(os.environ.get("SIGMA_EPISODES_DIR", DEFAULT_RUN_DIR))
STOCK = "GS"

# The bar moved, because the fix landed (D-I3). It used to be "mid 5 of 8",
# a criterion that only watches the touch. It is now "full book 8 of 8": every
# level, every step, 900 messages, all eight windows, on both arms.
#
# Keeping the old bar after the fix would have made these tests pass while
# measuring nothing, which is how a regression pin quietly stops being one.
EXPECTED_REAL_EXACT = 8
EXPECTED_REAL_MEAN_MIN = 1.0
EXPECTED_GEN_MEAN_MIN = 1.0
EXPECTED_FULL_EXACT = 8

# Fidelity 1.0 does NOT mean the market is correct. It means the replay
# reproduces what the generator did, including its mistakes. Both sides commit
# this one identically, so it costs no fidelity and is invisible to every other
# assertion in this file, which is why it gets a band of its own.
#
# The band moved once, and the move is the point of having it. It was
# (600, 1200) around 880 sentinels. D-R6's fix -- moving the sentinel out of
# the init id space so a miss falls through to the price fallback instead of
# cancelling a real order -- changed the generated market and the count fell to
# 355. A one-sided cap would have let that pass silently and the reason would
# have been lost. Cross-checked independently: the provenance field counts
# exactly 355 total misses over the same 7200 messages.
# Baseline-specific, and now known to be so (D-X7): generation is
# byte-identical only on the node that produced it, so this count is a property
# of THIS recorded run and not of the code. 588 on the current baseline; it was
# 355 on a run of the same code on another node, which is what surfaced D-X7.
EXPECTED_SENTINEL_BAND = (400, 800)

pytestmark = pytest.mark.skipif(
    not (RUN_DIR / "data_gen").is_dir(),
    reason=f"generated episodes not present at {RUN_DIR}",
)


@pytest.fixture(scope="module")
def arms():
    windows = discover_windows(RUN_DIR, STOCK)
    assert windows, "no windows discovered"
    out = {"real": [], "gen": []}
    for date, idx in windows:
        rep = window_report(RUN_DIR, STOCK, date, idx)
        for arm in ("real", "gen"):
            if arm in rep:
                out[arm].append((f"{date} #{idx}", rep[arm]))
    return out


def test_real_arm_reproduces_the_recorded_book_on_the_expected_windows(arms):
    exact = [name for name, r in arms["real"] if r["exact_mid_frac"] == 1.0]
    assert len(exact) == EXPECTED_REAL_EXACT, (
        f"real arm exact on {len(exact)}/{len(arms['real'])} "
        f"(expected {EXPECTED_REAL_EXACT}): {exact}"
    )


def test_real_arm_mean_agreement_holds(arms):
    mean = float(np.mean([r["exact_mid_frac"] for _, r in arms["real"]]))
    assert mean >= EXPECTED_REAL_MEAN_MIN, f"real arm mean {mean:.4f}"


def test_real_arm_beats_gen_arm_on_every_window(arms):
    """The asymmetry is the finding. One codebase over identically-shaped inputs
    cannot produce it from an engine defect, which is what located the problem
    in the generated record rather than in the engine."""
    gen = dict(arms["gen"])
    for name, r in arms["real"]:
        assert name in gen
        assert r["exact_mid_frac"] >= gen[name]["exact_mid_frac"], name


def test_every_replayed_stream_is_clean(arms):
    """No non-monotonic clocks, no unhandled event types, nothing truncated.
    A stream that fails this cannot be replayed meaningfully at all."""
    for arm in ("real", "gen"):
        for name, r in arms[arm]:
            assert r["monotonic"], f"{arm} {name}: clock runs backwards"
            assert r["n_unhandled_types"] == 0, f"{arm} {name}: event types outside 1-4"
            assert r["clean"], f"{arm} {name}: {r}"


def test_gen_arm_has_not_collapsed(arms):
    mean = float(np.mean([r["exact_mid_frac"] for _, r in arms["gen"]]))
    assert mean >= EXPECTED_GEN_MEAN_MIN, f"gen arm mean {mean:.4f}"


def test_the_real_arm_never_carries_a_lookup_miss_sentinel(arms):
    """Structural, not empirical: the sentinel is written by the generator's
    order resolution. Real LOBSTER messages carry real positive ids and never
    pass through it. If this ever fires, the real arm has stopped being an
    independent check on the engine."""
    for name, r in arms["real"]:
        assert r["n_lookup_miss_sentinel"] == 0, name
        assert r["n_init_ref_out_of_depth"] == 0, name


def test_lookup_miss_sentinel_volume_is_pinned(arms):
    """A tripwire in both directions, which is why it is a range and not a cap.

    355 of 7200 messages carry the sentinel. The history is worth keeping: 21
    before the exact-timestamp lookup landed, 880 after it (a miss started
    being reported instead of silently resolving to an unrelated order), then
    355 once D-R6 moved the sentinel out of the init id space, which changed
    what a miss does and therefore the market that follows it.

    It costs no fidelity, because both sides apply it identically, which is
    exactly why it needs its own tripwire: no other assertion here can see it.

    A large drop means someone changed the sentinel or the resolution and
    should say which. A large rise means generation changed. Neither should
    pass unnoticed."""
    total = sum(r["n_lookup_miss_sentinel"] for _, r in arms["gen"])
    lo, hi = EXPECTED_SENTINEL_BAND
    assert lo <= total <= hi, (
        f"lookup-miss sentinels total {total}, outside the pinned band. "
        "This is not a failure by itself; it means generation semantics moved "
        "and the change needs to be stated."
    )


def test_unresolvable_cancels_do_not_predict_failure(arms):
    """Recorded because it refutes the obvious hypothesis. Real streams carry
    12-38 unresolvable cancels per window and still replay exactly, so the
    engine's price-based fallback lands where the real cancel landed."""
    exact_with_unresolvable = [
        name for name, r in arms["real"]
        if r["exact_mid_frac"] == 1.0 and r["n_unresolvable_ids"] > 0
    ]
    assert exact_with_unresolvable, (
        "expected at least one window that replays exactly despite unresolvable "
        "cancels; if none remain, the mechanism has changed"
    )


def test_both_arms_are_exact_on_the_full_book(arms):
    """The strict criterion, and the one the D-I3 fix was aimed at. Mid only
    watches the touch: before faithful initialisation the real arm scored
    5/8 on mid and 0/8 here, and the gen arm 0/8 and 0/8."""
    for arm in ("real", "gen"):
        exact = [name for name, r in arms[arm] if r["exact_full_frac"] == 1.0]
        assert len(exact) == EXPECTED_FULL_EXACT, (
            f"{arm} arm exact on the full book in {len(exact)}/{len(arms[arm])} "
            f"windows (expected {EXPECTED_FULL_EXACT})"
        )


def test_no_window_diverges_at_any_step(arms):
    """Stated separately from the fraction because 'never diverged' and
    'agreed on average' are different claims and only the first one licenses
    using this simulator as ground truth."""
    for arm in ("real", "gen"):
        for name, r in arms[arm]:
            assert r["first_full_divergence_step"] == -1, (
                f"{arm} {name} diverges at step {int(r['first_full_divergence_step'])}"
            )


def test_faithful_initialisation_is_actually_in_use(arms):
    """Guards against the baseline silently reverting to the degraded path,
    which would still pass some assertions above while measuring a different
    thing entirely. The init file is what selects faithful mode."""
    from post_training.heuristic_learning.episode_builder import fixture_paths

    windows = discover_windows(RUN_DIR, STOCK)
    for date, idx in windows:
        p = fixture_paths(RUN_DIR, STOCK, date, idx)
        assert p["cond_book_init"].is_file(), (
            f"{date} #{idx} has no bookinit file, so this run is being replayed "
            "through the degraded path and these numbers do not mean what they say"
        )
