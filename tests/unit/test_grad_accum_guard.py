"""D-X10 detector: gradient accumulation must refuse topologies that drop K.

create_jit_train_step builds the (micro_step_fn, apply_step_fn, K) pair only
on the shard_map path. Before the guard, grad_accum_steps>1 on a flat
single-node mesh fell through to the plain per-micro-batch path and K was
silently ignored: the 2026-08-27 single-node baseline segment trained at
effective batch 4 while its launch banner printed "K=20 (effective BSZ = 80)".
The guard must fire before the mesh is used for anything else, so a bare
namespace carrying only axis_names is enough to pin the behaviour.
"""
from types import SimpleNamespace

import pytest

from base_model.training.train_helpers import create_jit_train_step


def test_grad_accum_without_shard_map_raises():
    mesh = SimpleNamespace(axis_names=('data',))
    with pytest.raises(ValueError, match="grad_accum_steps"):
        create_jit_train_step(mesh, state=None, hierarchical=False,
                              grad_accum_steps=20)


def test_grad_accum_of_one_passes_the_guard():
    # K=1 must not trip the guard. The call may still fail later for
    # unrelated reasons (the namespace is not a real mesh); only the guard's
    # ValueError naming grad_accum_steps would be a regression here.
    mesh = SimpleNamespace(axis_names=('data',))
    try:
        create_jit_train_step(mesh, state=None, hierarchical=False,
                              grad_accum_steps=1)
    except ValueError as e:
        assert "grad_accum_steps" not in str(e)
    except Exception:
        pass
