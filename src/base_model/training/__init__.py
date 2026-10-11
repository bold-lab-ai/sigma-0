"""Training: the loop (``train``, ``train_helpers``), state init and Orbax
checkpoint restore (``state``), checkpoint inspection, selection, and the
Slurm launch-plan builder used by ``run/base_model/train_base_model.py``.

``train`` and ``state`` import jax; import them explicitly.
"""

from .checkpoint_inspect import (
    CheckpointArrayInfo,
    CheckpointLoadError,
    LoadedArraySummary,
    ResolvedCheckpoint,
    checkpoint_config,
    iter_checkpoint_arrays,
    load_checkpoint_metadata,
    load_state_metadata,
    resolve_checkpoint_path,
    sample_checkpoint_arrays,
    summarize_checkpoint,
)
from .launch_plan import TrainingLaunchPlan, build_train_plan, run_train

__all__ = [
    "CheckpointArrayInfo",
    "CheckpointLoadError",
    "LoadedArraySummary",
    "ResolvedCheckpoint",
    "TrainingLaunchPlan",
    "build_train_plan",
    "checkpoint_config",
    "iter_checkpoint_arrays",
    "load_checkpoint_metadata",
    "load_state_metadata",
    "resolve_checkpoint_path",
    "run_train",
    "sample_checkpoint_arrays",
    "summarize_checkpoint",
]
