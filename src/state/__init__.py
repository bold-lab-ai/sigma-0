"""State model: the order book as a video on a wall-clock frame grid.

Where the message-level models in this repository emit one event at a time,
the state model denoises whole blocks of future book snapshots at once.  Frames
sit on a fixed wall-clock grid — 1 ms per frame by default (``frames.py``), so
a window of H frames always spans H milliseconds of market time: the model's
horizon is defined in time, not in event count.

    frames.py          event stream -> fixed-interval frames (forward fill)
    representation.py  structured chart; decode cannot produce an invalid book
    dataset.py         windowing + normalisation
    dit.py             FrameDiT backbone (WanDiT block, mask-concat cond)
    flow.py            rectified-flow core: losses, ODE/SDE samplers, log-probs
    rewards.py         rule-based and learned rewards over decoded books
    train.py           conditional flow-matching pretraining
    posttraining/      ReFL, Flow-GRPO, AWM, RAM, DMD — see its README
"""

from .dit import FrameDiT, FrameDiTConfig, bind_condition, bind_unconditional
from .flow import RectifiedFlow, fm_loss, interpolate, velocity_target, x0_from_v
from .frames import DEFAULT_INTERVAL_NS, FrameSeries, resample_events, resample_wide_book
from .representation import ChartNormalizer, decode_frames, decode_frames_torch, encode_frames

__all__ = [
    "FrameDiT",
    "FrameDiTConfig",
    "bind_condition",
    "bind_unconditional",
    "RectifiedFlow",
    "fm_loss",
    "interpolate",
    "velocity_target",
    "x0_from_v",
    "DEFAULT_INTERVAL_NS",
    "FrameSeries",
    "resample_events",
    "resample_wide_book",
    "ChartNormalizer",
    "decode_frames",
    "decode_frames_torch",
    "encode_frames",
]
