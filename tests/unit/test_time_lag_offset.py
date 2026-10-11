"""The time-lagged mode must put its two statistics `lag` apart.

As merged, it did not. `_align_and_slice` dropped the first `lag` rows of the
merged frame and both statistics were then evaluated on that same frame, so the
conditioning value and the evaluated value came from the same message. `lag`
truncated; it never offset. The mode's whole purpose is the offset, and a
contemporaneous conditional is what `--conditional` already computes.

The test below makes each statistic return its own row index, so the offset is
readable directly: `evaluated - conditioning` must equal `lag` at every position.
Under the merged behaviour it equals 0 everywhere, which is the failure this
pins down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

SCORER = Path(__file__).resolve().parents[2] / "src" / "lobpipeline" / "scorer" / "lob_bench"


@pytest.fixture(scope="module")
def scoring():
    """Import the scorer by path; it uses bare module names internally."""
    sys.path.insert(0, str(SCORER))
    try:
        import scoring as module
    except ImportError as exc:  # the scoring dependency set is not installed here
        pytest.skip(f"scorer not importable: {exc}")
    finally:
        pass
    return module


class _Seq:
    """The attributes the lagged scorers read off a sequence.

    The prompt and continuation frames are built independently, each with its
    own 0-based RangeIndex, because that is what ``data_loading`` produces
    from a pair of CSVs (``index_col=False``).  An earlier fixture sliced one
    frame in two, giving the merged frame a duplicate-free index that real
    data never has -- and hiding that the merge kept the duplicate labels,
    which made every statistic that aligns by index raise
    ``cannot reindex on an axis with duplicate labels`` on the first real run.
    """

    def __init__(self, n_cond, n_cont):
        self.m_cond = pd.DataFrame({"row": np.arange(n_cond, dtype=float)})
        # The prompt book carries one more row than the prompt messages (the
        # snapshot before any message), as in the real inference inventory.
        # After the merge is cut to a common length this leaves the message
        # and book indexes disagreeing row for row, which is what forces
        # pandas to actually align them -- identical indexes take a fast path
        # that hides the duplicate labels.
        self.b_cond = pd.DataFrame({"row": np.arange(n_cond + 1, dtype=float)})
        continuation = pd.DataFrame(
            {"row": np.arange(n_cond, n_cond + n_cont, dtype=float)}
        )
        self.m_real = continuation
        self.b_real = continuation.copy()
        self.m_gen = [continuation.copy()]
        self.b_gen = [continuation.copy()]


def _row_index(m, b):
    """A statistic that simply reports which message it was evaluated on."""
    return m["row"].values


def test_the_two_statistics_end_up_lag_apart(scoring):
    lag = 40
    seq = _Seq(n_cond=100, n_cont=100)

    conditioning, evaluated = scoring._score_one_seq_lagged(
        seq, _row_index, _row_index, lag, is_real=True
    )

    assert len(conditioning) == len(evaluated)
    offsets = np.unique(evaluated - conditioning)
    assert offsets.tolist() == [lag], (
        f"conditioning and evaluated values are {offsets} apart, expected [{lag}]. "
        "An offset of [0] means lag is truncating both statistics instead of "
        "putting them lag apart, which makes this mode a contemporaneous "
        "conditional -- the measurement --conditional already provides."
    )


def test_the_conditioning_value_reaches_back_into_the_prompt(scoring):
    """With lag >= the continuation length, every conditioner is prompt-side.

    This is the property the mode exists for: the value being conditioned on is
    one the model was given rather than one it produced.
    """
    n_cond, n_cont, lag = 100, 60, 60
    seq = _Seq(n_cond=n_cond, n_cont=n_cont)

    conditioning, evaluated = scoring._score_one_seq_lagged(
        seq, _row_index, _row_index, lag, is_real=True
    )

    # Rows below n_cond are prompt; rows at or above it are the continuation.
    assert evaluated.max() == n_cond + n_cont - 1
    assert conditioning.max() < n_cond, (
        "some conditioning values come from the continuation, so the mode is "
        "conditioning the model's output on the model's own output"
    )


def test_a_sequence_shorter_than_the_lag_is_dropped_not_truncated(scoring):
    seq = _Seq(n_cond=10, n_cont=10)
    assert scoring._score_one_seq_lagged(seq, _row_index, _row_index, 40, is_real=True) is None


def test_the_generated_path_offsets_the_same_way(scoring):
    """The generated branch of `_score_one_seq_lagged` matches the real one.

    This exercises the is_real=False branch of the same module-level copy;
    the module's *other* copy of the logic is covered by the
    `..._in_the_inner_copy_too` tests below.
    """
    lag = 40
    seq = _Seq(n_cond=100, n_cont=100)

    conditioning, evaluated = scoring._score_one_seq_lagged(
        seq, _row_index, _row_index, lag, is_real=False
    )
    assert np.unique(evaluated[0] - conditioning[0]).tolist() == [lag]


def test_the_offset_holds_in_the_inner_copy_too(scoring):
    """`score_data_time_lagged` carries a second, inline copy of the offset.

    With n_workers=1 the driver takes `merge_and_score_lagged`, not the
    module-level `_score_one_seq_lagged`, so the module-level tests above
    cannot see this copy diverge -- reversing its offset direction fails
    nothing else in this file.  The row-index statistic makes the offset
    readable straight off the returned table: every evaluated row must sit
    exactly `lag` messages after the row it is conditioned on, on both the
    real and the generated side.
    """
    lag = 40
    loader = [_Seq(n_cond=100, n_cont=100)]

    score_df = scoring.score_data_time_lagged(
        loader, _row_index, _row_index, lag,
        n_workers=1,
        score_kwargs={"n_bins": 5},
        score_lagged_kwargs={"n_bins": 5},
    )

    for side in ("real", "generated"):
        rows = score_df[score_df["type"] == side]
        assert len(rows) > 0, f"no {side} rows scored"
        offsets = np.unique(rows["score"].values - rows["score_cond"].values)
        assert offsets.tolist() == [lag], (
            f"{side}: conditioning and evaluated values are {offsets} apart, "
            f"expected [{lag}]; the inner copy has diverged from "
            "_score_one_seq_lagged"
        )


def test_the_inner_copy_conditions_on_the_prompt_at_full_lag(scoring):
    """Inner-copy version of the reach-back property: at lag >= continuation
    length every conditioning value must be prompt-side."""
    n_cond, n_cont, lag = 100, 60, 60
    loader = [_Seq(n_cond=n_cond, n_cont=n_cont)]

    score_df = scoring.score_data_time_lagged(
        loader, _row_index, _row_index, lag,
        n_workers=1,
        score_kwargs={"n_bins": 5},
        score_lagged_kwargs={"n_bins": 5},
    )

    real = score_df[score_df["type"] == "real"]
    assert real["score"].max() == n_cond + n_cont - 1
    assert real["score_cond"].max() < n_cond, (
        "the inner copy conditions on the continuation at full lag, so the "
        "prompt is not what the mode is reading"
    )

def _index_aligned_stat(m, b):
    """A statistic that aligns by index, the way `eval.l1_volume` does.

    `l1_volume` builds a frame from the book and assigns a Series derived
    from the messages into it; pandas aligns that assignment by index.  The
    row-index statistic above never does this, so only a statistic shaped
    like this one can notice a merged frame that kept duplicate labels.
    """
    df = b.copy()
    df["w"] = m["row"].diff()
    return df["w"].values


def test_merged_frames_are_safe_for_index_aligned_statistics(scoring):
    """The real scoring functions align by index; the merge must allow it.

    Concatenating the prompt and continuation with their own 0-based indexes
    kept the duplicate labels, so the first statistic that assigned a Series
    into a frame raised `cannot reindex on an axis with duplicate labels` --
    on every sequence of the first real run of this mode.
    """
    lag = 40
    seq = _Seq(n_cond=100, n_cont=100)

    result = scoring._score_one_seq_lagged(
        seq, _index_aligned_stat, _index_aligned_stat, lag, is_real=True
    )
    assert result is not None

    score_df = scoring.score_data_time_lagged(
        [_Seq(n_cond=100, n_cont=100)],
        _index_aligned_stat,
        _index_aligned_stat,
        lag,
        n_workers=1,
        score_kwargs={"n_bins": 5},
        score_lagged_kwargs={"n_bins": 5},
    )
    assert (score_df["type"] == "real").any()
