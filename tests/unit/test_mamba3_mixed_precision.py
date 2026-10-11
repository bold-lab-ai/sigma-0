"""--dtype bfloat16 on Mamba3: two matmuls change, nothing else does.

What the flag must do: run in_proj, out_proj and the GLU projection in bf16
with float32 parameters, hand the Pallas kernels bf16 operands, and leave the
decay math, the norms, the recurrence and the parameters in float32. What it
must not do: change a float32 layer at all. The checks read the traced
program, not the config, because a dtype that is set, forwarded and logged
while reaching no code is exactly the defect this flag had before.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MAMBA3_CONTRACTION_PRECISION", "highest")

import jax                                                   # noqa: E402
from jax.flatten_util import ravel_pytree                    # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import numpy as np                                           # noqa: E402
import pytest                                                # noqa: E402

from base_model.models.layers import SequenceLayer          # noqa: E402
from base_model.models.mamba3 import Mamba3SSM, init_Mamba3SSM  # noqa: E402

H, L = 64, 70


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


def _layer(dtype):
    # d_state 64 -> 16 rotation angles: the smallest state the fused kernels take.
    return Mamba3SSM(H=H, d_state=64, expand=2, headdim=16, chunk_size=16,
                     rope_fraction=0.5, dtype=dtype)


def _find_eqns(jaxpr, name):
    """Every equation of primitive `name`, descending into sub-jaxprs
    (custom_vjp_call, pjit, scan ...)."""
    out = []
    def walk(jp):
        for eqn in jp.eqns:
            if eqn.primitive.name == name:
                out.append(eqn)
            for sub in eqn.params.values():
                if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                    walk(sub.jaxpr)
                elif hasattr(sub, "eqns"):
                    walk(sub)
    walk(jaxpr)
    return out


def _dot_operand_dtypes(fn, *args):
    """(lhs, rhs) dtypes of every dot_general in the traced program, at any depth
    outside Pallas kernels (their bodies are opaque to this walk on purpose)."""
    jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
    return [tuple(str(v.aval.dtype) for v in e.invars) for e in _find_eqns(jaxpr, "dot_general")]


def test_parameters_stay_float32_under_bf16_compute():
    x = jax.random.normal(jax.random.PRNGKey(0), (L, H))
    variables = _layer(jnp.bfloat16).init(jax.random.PRNGKey(1), x)
    dtypes = {str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(variables)}
    assert dtypes == {"float32"}, dtypes


def test_projections_are_bf16_and_everything_else_float32(monkeypatch):
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "xla")
    x = jax.random.normal(jax.random.PRNGKey(0), (L, H))
    layer32, layer16 = _layer(jnp.float32), _layer(jnp.bfloat16)
    variables = layer32.init(jax.random.PRNGKey(1), x)

    dots32 = _dot_operand_dtypes(lambda v: layer32.apply(v, x), variables)
    dots16 = _dot_operand_dtypes(lambda v: layer16.apply(v, x), variables)
    assert dots32 and all(d == ("float32", "float32") for d in dots32)
    bf16 = [d for d in dots16 if d == ("bfloat16", "bfloat16")]
    other = [d for d in dots16 if d != ("bfloat16", "bfloat16")]
    assert len(bf16) == 2, f"expected in_proj and out_proj in bf16, got {dots16}"
    assert all(d == ("float32", "float32") for d in other), other
    assert len(dots16) == len(dots32)


def test_bf16_layer_tracks_the_float32_layer(monkeypatch):
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "xla")
    x = jax.random.normal(jax.random.PRNGKey(0), (L, H))
    cot = jax.random.normal(jax.random.PRNGKey(2), (L, H))
    variables = _layer(jnp.float32).init(jax.random.PRNGKey(1), x)

    def run(layer):
        out = layer.apply(variables, x)
        grads = jax.grad(lambda v: jnp.sum(layer.apply(v, x).astype(jnp.float32) * cot))(variables)
        return out, grads

    out32, g32 = run(_layer(jnp.float32))
    out16, g16 = run(_layer(jnp.bfloat16))
    assert out16.dtype == jnp.bfloat16 and out32.dtype == jnp.float32
    assert _rel(out16, out32) < 5e-2
    for a, b in zip(jax.tree_util.tree_leaves(g16), jax.tree_util.tree_leaves(g32)):
        assert a.dtype == jnp.float32
        assert np.isfinite(np.asarray(a)).all()
    assert _rel(ravel_pytree(g16)[0], ravel_pytree(g32)[0]) < 0.25


def test_pallas_path_takes_bf16_operands(monkeypatch):
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "pallas")
    monkeypatch.delenv("MAMBA3_PALLAS_DTYPE", raising=False)
    x = jax.random.normal(jax.random.PRNGKey(0), (L, H))
    variables = _layer(jnp.float32).init(jax.random.PRNGKey(1), x)
    out16 = _layer(jnp.bfloat16).apply(variables, x)
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "xla")
    out32 = _layer(jnp.float32).apply(variables, x)
    assert out16.dtype == jnp.bfloat16
    assert _rel(out16, out32) < 5e-2
    # The kernels' tile inputs are bf16 when the layer is: the pallas_call's
    # operands carry the I/O dtype.
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "pallas")
    jaxpr = jax.make_jaxpr(lambda v: _layer(jnp.bfloat16).apply(v, x))(variables).jaxpr
    calls = _find_eqns(jaxpr, "pallas_call")
    assert calls, "no pallas_call traced"
    tile_dtypes = {str(v.aval.dtype) for v in calls[-1].invars if v.aval.ndim == 4}
    assert "bfloat16" in tile_dtypes, tile_dtypes


def test_sequence_layer_glu_follows_the_mixer_dtype():
    factory = init_Mamba3SSM(H, d_state=16, headdim=16, chunk_size=16, dtype=jnp.bfloat16)
    layer = SequenceLayer(ssm=factory, dropout=0.0, d_model=H, activation="half_glu1",
                          training=False, prenorm=True, batchnorm=False)
    x = jax.random.normal(jax.random.PRNGKey(0), (L, H))
    variables = layer.init(jax.random.PRNGKey(1), x)
    out = layer.apply(variables, x)
    assert out.dtype == jnp.float32, "the residual stream stays float32"
    dots = _dot_operand_dtypes(lambda v: layer.apply(v, x), variables)
    assert [d for d in dots if d == ("bfloat16", "bfloat16")].__len__() == 3, dots
