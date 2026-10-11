"""Splitting the sequence across devices must not change the numbers.

The sharded phase 5 reassociates a sum, it does not approximate one: the local
all-pairs contraction covers pairs inside a device and the device-level
contraction covers pairs across devices, and together they cover each ordered
pair (c <= z) exactly once. So the only admissible difference from the
unsharded reference is floating-point reassociation.

Runs on four host devices, so it needs no GPU and can gate a change before any
allocation is available. It checks two things that fail differently: agreement
with the reference, and that the device count does not matter, since a bug in
the carry usually shows up as a result that drifts as D changes while still
looking plausible at any single D.
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
import numpy as np                                             # noqa: E402
import pytest                                                  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P              # noqa: E402
# jax.shard_map (0.8+) dropped check_rep, which this needs because the
# all_gather inside makes the output's replication non-obvious to the checker.
# The experimental entry point still accepts it and still works, so the
# deprecation warning is the cheaper of the two costs.
from jax.experimental.shard_map import shard_map               # noqa: E402

from base_model.models.mamba3_jax import _phases_4_5_6_jax                    # noqa: E402
from base_model.models.mamba3_seqpar import phases_4_5_6_seqpar               # noqa: E402

NC, CS, NH, HD, N = 16, 8, 3, 4, 6


def _inputs(seed=0):
    """Chunked inputs shaped as phases 1-3 hand them over.

    Log-decays must be negative, which is what the real dt * A produces. With
    positive values exp(cumA[z] - cumA[c]) grows without bound over nc chunks
    and any comparison degenerates into comparing two overflows.
    """
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    Q_c = jax.random.normal(k[0], (NC, CS, NH, N))
    K_c = jax.random.normal(k[1], (NC, CS, NH, N))
    V_c = jax.random.normal(k[2], (NC, CS, NH, HD))
    ADT_c = -jax.random.uniform(k[3], (NH, NC, CS), minval=0.02, maxval=0.4)
    A_cumsum = jnp.cumsum(ADT_c, axis=-1)
    return Q_c, K_c, V_c, ADT_c, A_cumsum


def _sharded(inputs, n_dev):
    mesh = Mesh(np.array(jax.devices()[:n_dev]), axis_names=("sp",))
    fn = shard_map(
        lambda *a: phases_4_5_6_seqpar(*a, axis_name="sp"),
        mesh=mesh,
        # Q/K/V carry the chunk axis first; ADT and A_cumsum carry it second.
        in_specs=(P("sp"), P("sp"), P("sp"), P(None, "sp"), P(None, "sp")),
        out_specs=P("sp"),
        check_rep=False,
    )
    return jax.jit(fn)(*inputs)


@pytest.mark.parametrize("n_dev", [2, 4])
def test_matches_unsharded_reference(n_dev):
    inputs = _inputs()
    ref = np.asarray(_phases_4_5_6_jax(*inputs))
    got = np.asarray(_sharded(inputs, n_dev))
    err = np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-12)
    assert err < 1e-5, (
        f"D={n_dev}: sharded phase 5 differs from the reference by {err:.3e} "
        f"relative. The split reassociates the sum, it does not change it, so "
        f"anything above rounding means the cross-device carry is wrong."
    )


def test_result_is_independent_of_device_count():
    """D=2 and D=4 must agree, since D is a partitioning choice, not a model one.

    Kept separate from the reference check because a carry that is dropped or
    double-counted often still matches the reference at one particular D.
    """
    inputs = _inputs(seed=1)
    a = np.asarray(_sharded(inputs, 2))
    b = np.asarray(_sharded(inputs, 4))
    err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-12)
    assert err < 1e-5, f"D=2 and D=4 disagree by {err:.3e} relative"


def test_single_device_is_the_reference():
    """D=1 exercises the carry path with an all-zero carry; it must be exact."""
    inputs = _inputs(seed=2)
    ref = np.asarray(_phases_4_5_6_jax(*inputs))
    got = np.asarray(_sharded(inputs, 1))
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
