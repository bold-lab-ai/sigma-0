"""GRAD_ALLREDUCE_DTYPE: the cross-node gradient mean may travel in bf16.

Runs the helper under shard_map on four host devices arranged as the
(nodes, gpus) mesh the hierarchical step uses. The reduced tree keeps its
float32 dtype, the float32 path is exactly jax.lax.pmean, and the bf16 path
differs from it only by bf16 rounding.
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import numpy as np                                           # noqa: E402
import pytest                                                # noqa: E402
from jax.experimental.shard_map import shard_map             # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P            # noqa: E402

from base_model.training.train_helpers import _pmean_in, cross_node_reduce_dtype  # noqa: E402

pytestmark = pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 host devices")


def _run(dtype):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), axis_names=("nodes", "gpus"))
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 64), dtype=jnp.float32)
    f = shard_map(lambda a: _pmean_in({"w": a, "b": a * 3.0}, "nodes", dtype), mesh=mesh,
                  in_specs=P(("nodes", "gpus")), out_specs=P(("nodes", "gpus")), check_rep=False)
    return jax.jit(f)(x), x


def test_float32_path_is_plain_pmean():
    out, x = _run(jnp.float32)
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), axis_names=("nodes", "gpus"))
    ref = jax.jit(shard_map(lambda a: jax.lax.pmean({"w": a, "b": a * 3.0}, "nodes"), mesh=mesh,
                            in_specs=P(("nodes", "gpus")), out_specs=P(("nodes", "gpus")), check_rep=False))(x)
    assert out["w"].dtype == jnp.float32
    assert np.array_equal(np.asarray(out["w"]), np.asarray(ref["w"]))
    assert np.array_equal(np.asarray(out["b"]), np.asarray(ref["b"]))


def test_bf16_path_keeps_float32_leaves_within_bf16_rounding():
    out16, x = _run(jnp.bfloat16)
    out32, _ = _run(jnp.float32)
    assert out16["w"].dtype == jnp.float32 and out16["b"].dtype == jnp.float32
    rel = np.abs(np.asarray(out16["w"]) - np.asarray(out32["w"])).max() / np.abs(np.asarray(out32["w"])).max()
    assert 0 < rel < 1e-2, rel


def test_knob_default_and_validation(monkeypatch):
    monkeypatch.delenv("GRAD_ALLREDUCE_DTYPE", raising=False)
    assert cross_node_reduce_dtype() == jnp.float32
    monkeypatch.setenv("GRAD_ALLREDUCE_DTYPE", "bfloat16")
    assert cross_node_reduce_dtype() == jnp.bfloat16
    monkeypatch.setenv("GRAD_ALLREDUCE_DTYPE", "float16")
    with pytest.raises(ValueError, match="GRAD_ALLREDUCE_DTYPE"):
        cross_node_reduce_dtype()
