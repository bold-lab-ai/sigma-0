"""MAMBA3_SSD_IMPL=pallas must leave the Mamba3 layer's numbers unchanged.

The SSD-level parity lives in test_mamba3_pallas_ssd.py. This file checks the
wiring through the Flax layer: the knob selects the fused path, forward and
parameter gradients agree with the reference, a layer whose tiles Triton
cannot take falls back to the reference with a warning, and the default is
the reference.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MAMBA3_CONTRACTION_PRECISION", "highest")

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import numpy as np                                           # noqa: E402
import pytest                                                # noqa: E402

from base_model.models.mamba3 import Mamba3SSM               # noqa: E402


def _layer(H=64, d_state=64, headdim=16, chunk_size=16):
    return Mamba3SSM(H=H, d_state=d_state, expand=2, headdim=headdim,
                     chunk_size=chunk_size, rope_fraction=0.5)


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


def _run(monkeypatch, impl, layer, variables, x, cot):
    monkeypatch.setenv("MAMBA3_SSD_IMPL", impl)

    def loss(v):
        return jnp.sum(layer.apply(v, x) * cot)

    out = jax.jit(lambda v: layer.apply(v, x))(variables)
    grads = jax.jit(jax.grad(loss))(variables)
    return out, grads


def test_pallas_layer_matches_reference_forward_and_grads(monkeypatch):
    layer = _layer()
    x = jax.random.normal(jax.random.PRNGKey(0), (70, 64))
    cot = jax.random.normal(jax.random.PRNGKey(1), (70, 64))
    variables = layer.init(jax.random.PRNGKey(2), x)

    out_ref, g_ref = _run(monkeypatch, "xla", layer, variables, x, cot)
    out_got, g_got = _run(monkeypatch, "pallas", layer, variables, x, cot)

    assert _rel(out_got, out_ref) < 1e-5
    flat_ref = jax.tree_util.tree_leaves_with_path(g_ref)
    flat_got = jax.tree_util.tree_leaves(g_got)
    assert len(flat_ref) == len(flat_got)
    for (path, a), b in zip(flat_ref, flat_got):
        err = _rel(b, a)
        assert err < 5e-5, f"grad {jax.tree_util.keystr(path)} differs by {err:.3e}"


def test_unsupported_tiles_fall_back_to_the_reference(monkeypatch):
    # H=5 gives d_inner 10 and headdim 10: not a power of two, so no kernel.
    layer = _layer(H=5, d_state=64, headdim=16, chunk_size=16)
    x = jax.random.normal(jax.random.PRNGKey(0), (40, 5))
    variables = layer.init(jax.random.PRNGKey(2), x)
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "xla")
    ref = layer.apply(variables, x)
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "pallas")
    from base_model.models import mamba3_pallas
    mamba3_pallas._warned_fallback.clear()
    with pytest.warns(RuntimeWarning, match="uses the XLA reference"):
        got = layer.apply(variables, x)
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_default_is_the_reference_and_bad_values_are_refused(monkeypatch):
    from base_model.models.mamba3_pallas import ssd_impl_mode, pallas_compute_dtype
    monkeypatch.delenv("MAMBA3_SSD_IMPL", raising=False)
    assert ssd_impl_mode() == "xla"
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "triton")
    with pytest.raises(ValueError, match="MAMBA3_SSD_IMPL"):
        ssd_impl_mode()
    monkeypatch.setenv("MAMBA3_PALLAS_DTYPE", "float16")
    with pytest.raises(ValueError, match="MAMBA3_PALLAS_DTYPE"):
        pallas_compute_dtype()


def test_small_rope_ranges_fall_back_to_the_reference(monkeypatch):
    # d_state 16 at rope_fraction 0.5 gives 4 rotation angles: ranges of 4 are
    # below the tensor-core tile, so the layer keeps the reference.
    layer = _layer(H=64, d_state=16, headdim=16, chunk_size=16)
    x = jax.random.normal(jax.random.PRNGKey(0), (40, 64))
    variables = layer.init(jax.random.PRNGKey(2), x)
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "xla")
    ref = layer.apply(variables, x)
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "pallas")
    from base_model.models import mamba3_pallas
    mamba3_pallas._warned_fallback.clear()
    with pytest.warns(RuntimeWarning, match="num_rope_angles=4"):
        got = layer.apply(variables, x)
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_fused_path_reads_b_and_c_once_for_all_heads(monkeypatch):
    """The phase-1 kernel takes B and C as (L_pad, N); no (L, heads, N) tensor
    (the reference's head-repeated layout) exists anywhere in the forward."""
    layer = _layer()
    x = jax.random.normal(jax.random.PRNGKey(0), (70, 64))
    variables = layer.init(jax.random.PRNGKey(2), x)
    monkeypatch.setenv("MAMBA3_SSD_IMPL", "pallas")
    jaxpr = jax.make_jaxpr(lambda v: layer.apply(v, x))(variables).jaxpr

    calls, outvars = [], []
    def walk(jp):
        for eqn in jp.eqns:
            if eqn.primitive.name == "pallas_call":
                calls.append(eqn)
            outvars.extend(eqn.outvars)
            for sub in eqn.params.values():
                if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                    walk(sub.jaxpr)
                elif hasattr(sub, "eqns"):
                    walk(sub)
    walk(jaxpr)
    assert len(calls) == 3, f"phase1, chunk_state and chunk_scan expected, got {len(calls)} pallas_calls"
    N = layer.d_state
    shared = [v.aval.shape for v in calls[0].invars
              if v.aval.ndim == 2 and v.aval.shape == (80, N)]     # 70 tokens padded to 5 chunks of 16
    assert len(shared) == 2, f"phase1 should read the shared (L_pad, N) C and B tiles, got {shared}"
    repeated = [v.aval.shape for v in outvars
                if hasattr(v, "aval") and v.aval.ndim == 3 and v.aval.shape[-1] == N]
    assert not repeated, f"a head-repeated (L, heads, N) tensor is traced: {repeated}"
