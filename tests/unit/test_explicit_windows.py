"""Tests for the explicit-window bypass in LOBSTER_Dataset.

The bypass exists because window placement is otherwise decided by a fixed
non-overlapping grid, ``seq_offsets[file] + seq_idx * n_messages``. Which kind of
message a conditioning window ends on is therefore a property of the grid, not a
choice: measured over 2000 GOOG windows only 1.95% end on a trade. Event-anchored
evaluation needs to place windows deliberately.

The regression risk is that the bypass changes behaviour for callers who do not use
it, so the first test drives both paths to the same physical window and requires
identical bounds.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# Importable without torchvision as of the s5.dataloaders fix that ships with this
# change. Before it, s5/dataloaders/{__init__,base}.py pulled torchvision in at module
# scope for image-benchmark plumbing unrelated to order books, which took
# lob.lobster_dataloader down with it and is why nothing under tests/ had ever
# constructed a LOBSTER_Dataset.
from base_model.data.lobster_dataloader import LOBSTER_Dataset


def _grid_probe(n_messages=250, offsets=(0, 7), cumsum=(0, 4, 9)):
    """Stand-in carrying just the state _window_bounds reads, on the grid path."""
    seqs_cumsum = np.asarray(cumsum)

    def get_seq_location(idx):
        file_idx = int(np.searchsorted(seqs_cumsum, idx + 1) - 1)
        return file_idx, int(idx - seqs_cumsum[file_idx])

    return SimpleNamespace(
        explicit_windows=None,
        n_messages=n_messages,
        seq_offsets=np.asarray(offsets, dtype=np.int64),
        _get_seq_location=get_seq_location,
    )


def _explicit_probe(windows, n_messages=250):
    return SimpleNamespace(
        explicit_windows=np.asarray(windows, dtype=np.int64),
        n_messages=n_messages,
        seq_offsets=None,
        _get_seq_location=None,
    )


def test_T8_explicit_path_reproduces_grid_bounds_exactly():
    """Same physical window through both paths must give identical bounds."""
    n_messages = 250
    grid = _grid_probe(n_messages=n_messages)

    for idx in range(9):
        f, s_idx, start, end = LOBSTER_Dataset._window_bounds(grid, idx)

        # Ask the explicit path for that same (file_idx, seq_start).
        expl = _explicit_probe([(f, start)], n_messages=n_messages)
        f2, _, start2, end2 = LOBSTER_Dataset._window_bounds(expl, 0)

        assert (f2, start2, end2) == (f, start, end)
        # And the grid really is offset + k * n_messages, so the bypass is not
        # accidentally reproducing a formula it also computes.
        assert start == int(grid.seq_offsets[f]) + s_idx * n_messages


def test_T8b_explicit_windows_are_returned_in_the_order_supplied():
    windows = [(1, 5000), (0, 12), (1, 3), (0, 999)]
    probe = _explicit_probe(windows, n_messages=100)
    for i, (f, s) in enumerate(windows):
        assert LOBSTER_Dataset._window_bounds(probe, i) == (f, i, s, s + 100)


def test_T8c_explicit_path_reaches_starts_the_grid_cannot():
    """A start that is not a multiple of n_messages is the entire point."""
    probe = _explicit_probe([(0, 137)], n_messages=250)
    _, _, start, end = LOBSTER_Dataset._window_bounds(probe, 0)
    assert start == 137 and end == 387
    assert start % 250 != 0


def test_T8d_out_of_range_index_raises():
    probe = _explicit_probe([(0, 10), (0, 20)], n_messages=50)
    with pytest.raises(IndexError):
        LOBSTER_Dataset._window_bounds(probe, 2)
    with pytest.raises(IndexError):
        LOBSTER_Dataset._window_bounds(probe, -1)


# --------------------------------------------------------------------------------
# Constructor validation. These all raise before any data file is opened, so bogus
# paths are fine.
# --------------------------------------------------------------------------------

def _make(**kw):
    base = dict(
        message_files=["/nonexistent_a.npy", "/nonexistent_b.npy"],
        n_messages=250,
        mask_fn=LOBSTER_Dataset.inference_mask,
        randomize_offset=False,
    )
    base.update(kw)
    return LOBSTER_Dataset(**base)


@pytest.mark.parametrize("windows, msg", [
    (np.zeros((3, 3), dtype=np.int64), "shape"),
    (np.zeros((0, 2), dtype=np.int64), "empty"),
    ([(2, 0)], "file_idx out of range"),      # only 2 files, so index 2 is invalid
    ([(0, -1)], "non-negative"),
])
def test_explicit_windows_validation(windows, msg):
    with pytest.raises(ValueError, match=msg):
        _make(explicit_windows=windows)


def test_randomize_offset_must_be_off():
    """The starts are absolute rows; a per-file offset would shift them silently."""
    with pytest.raises(ValueError, match="randomize_offset=False"):
        _make(explicit_windows=[(0, 100)], randomize_offset=True)


def test_npz_wide_book_combination_is_refused():
    """.npz snapshots are keyed by grid seq_idx, which an explicit window lacks.

    Without this check __getitem__ silently falls back to a zero-padded L10 book,
    handing back a wrong initial order book with no error raised.
    """
    with pytest.raises(ValueError, match="npz wide book"):
        _make(explicit_windows=[(0, 100)],
              wide_book_files=["/w_a.npz", "/w_b.npz"])
