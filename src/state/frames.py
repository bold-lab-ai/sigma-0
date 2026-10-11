"""Wall-clock resampling of event-level book states into fixed-interval frames.

The state model treats the order book the way a video model treats a scene:
a sequence of frames on a uniform clock.  The message-level models in this
repository advance one event at a time; BookDiT advances one frame per ten
messages (an event clock).  Here the frame clock is *wall time*: one frame
every ``interval_ns`` nanoseconds, 1 ms by default.  That choice is the model's
horizon definition — a window of H frames always spans exactly H * interval_ns
of market time, independent of how bursty message arrival was.

Resampling semantics (state variables, not flows):

    frame[k] = book state of the LAST event with time <= grid[k]

i.e. forward-fill.  A frame with many events inside its interval keeps only the
final state (intermediate churn is folded away); a frame with none repeats the
previous state.  ``events_per_frame`` reports how much was folded, because that
is the information the clock throws away and any honest evaluation should know
its size.

Input layout is the repository's wide-book npy contract (see
``src/base_model/data/wide_book.py`` and ``src/base_model/data/preproc.py``): 3 metadata columns
``[delta_p_mid, time_s, time_ns]`` followed by 4 columns per level
``[ask_p, ask_s, bid_p, bid_s]`` — 43 columns for the 10-level GOOG corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

META_COLS = 3
TIME_S_COL = 1
TIME_NS_COL = 2

DEFAULT_INTERVAL_NS = 1_000_000  # 1 ms — the default frame clock


def absolute_time_ns(wide: np.ndarray) -> np.ndarray:
    """Recover int64 absolute nanoseconds from the two metadata time columns."""
    if wide.ndim != 2 or wide.shape[1] <= META_COLS:
        raise ValueError(f"expected (N, >3) wide-book array, got {wide.shape}")
    sec = wide[:, TIME_S_COL].astype(np.int64)
    ns = wide[:, TIME_NS_COL].astype(np.int64)
    return sec * 1_000_000_000 + ns


@dataclass
class FrameSeries:
    """A uniformly sampled book-state sequence."""

    frames: np.ndarray  # (T, C) book columns only, float32
    times_ns: np.ndarray  # (T,) int64 grid timestamps
    events_per_frame: np.ndarray  # (T,) int64, events folded into each frame
    interval_ns: int

    @property
    def horizon_ns(self) -> int:
        return len(self.frames) * self.interval_ns


def resample_events(
    times_ns: np.ndarray,
    states: np.ndarray,
    interval_ns: int = DEFAULT_INTERVAL_NS,
) -> FrameSeries:
    """Forward-fill event states onto a uniform ``interval_ns`` grid.

    times_ns must be non-decreasing.  The grid starts at the first multiple of
    ``interval_ns`` at or after the first event (so every frame has a defined
    state) and ends at the last event.
    """
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")
    times_ns = np.asarray(times_ns, dtype=np.int64)
    if times_ns.ndim != 1 or len(times_ns) == 0:
        raise ValueError("times_ns must be a non-empty 1-D array")
    if np.any(np.diff(times_ns) < 0):
        raise ValueError("event times must be non-decreasing")
    states = np.asarray(states)
    if states.shape[0] != times_ns.shape[0]:
        raise ValueError("times_ns and states disagree on the number of events")

    start = -(-times_ns[0] // interval_ns) * interval_ns  # ceil to the grid
    stop = times_ns[-1]
    if start > stop:
        # All events fall inside one interval before the first grid point:
        # a single frame at the ceiling carries the final state.
        grid = np.array([start], dtype=np.int64)
    else:
        grid = np.arange(start, stop + 1, interval_ns, dtype=np.int64)

    # Index of the last event at or before each grid point (forward fill).
    idx = np.searchsorted(times_ns, grid, side="right") - 1
    # start >= times_ns[0] guarantees idx >= 0 for every grid point.
    frames = states[idx].astype(np.float32, copy=True)

    boundary = np.concatenate([[-1], idx])
    events_per_frame = np.diff(boundary + 1).astype(np.int64)

    return FrameSeries(
        frames=frames,
        times_ns=grid,
        events_per_frame=events_per_frame,
        interval_ns=int(interval_ns),
    )


def resample_wide_book(
    wide: np.ndarray,
    interval_ns: int = DEFAULT_INTERVAL_NS,
) -> FrameSeries:
    """Resample a wide-book array (43-column contract) onto the frame clock.

    Metadata columns are consumed for the clock and dropped from the frames;
    the returned channels are the 4-per-level book columns only.
    """
    times = absolute_time_ns(wide)
    book = wide[:, META_COLS:]
    if book.shape[1] % 4 != 0:
        raise ValueError(
            f"book width must be divisible by 4, found {book.shape[1]}"
        )
    return resample_events(times, book, interval_ns=interval_ns)
