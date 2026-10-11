"""The compiled Pallas SSD kernels must match the XLA reference on a real GPU.

The CPU tests run the same kernels through the Pallas interpreter, which
checks the arithmetic but not the Triton lowering. This runs the compiled
kernels and is skipped without a GPU backend (the suite record is CPU-only).
Tolerances come from job 6393204 at the production shape: full-f32 dots agree
to about 1e-7 forward and 2e-6 on gradients; bf16 operands to about 4e-3
forward and 4e-2 on the worst gradient. The fused path takes B, C and the
angle rates shared over heads; the reference gets them repeated.
"""
import os

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
import numpy as np  # noqa: E402

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="compiled Pallas kernels exist for GPU only; the CPU tests use the interpreter",
)

from base_model.models.mamba3_jax import mamba3_ssd_chunked_jax          # noqa: E402
from base_model.models.mamba3_pallas import mamba3_ssd_chunked_pallas    # noqa: E402

L, NH, N, R, P, CS = 1024, 4, 128, 32, 64, 64


def _inputs(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 11)
    Q = jax.random.normal(k[0], (L, N))
    K = jax.random.normal(k[1], (L, N))
    V = jax.random.normal(k[2], (L, NH, P))
    Z = jax.random.normal(k[3], (L, NH, P))
    DT = jax.nn.softplus(jax.random.normal(k[4], (NH, L)) - 2.0)
    A = -jax.nn.softplus(jax.random.normal(k[5], (NH, L)))
    Trap = jax.random.normal(k[6], (NH, L))
    angles = jax.random.normal(k[7], (L, 1, R))
    Q_bias = 0.1 * jax.random.normal(k[8], (NH, N))
    K_bias = 0.1 * jax.random.normal(k[9], (NH, N))
    D = jax.random.normal(k[10], (NH,))
    return (Q, K, V, A * DT, DT, Trap, angles, Q_bias, K_bias, D, Z)


def _ref_args(args):
    Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z = args
    rep = lambda x: jnp.repeat(x[:, None, :] if x.ndim == 2 else x, NH, axis=1)
    return (rep(Q), rep(K), V, ADT, DT, Trap, rep(angles), Q_bias, K_bias, D, Z)


def _ref_grads_to_shared(g):
    g = list(g)
    g[0] = jnp.sum(g[0], axis=1)
    g[1] = jnp.sum(g[1], axis=1)
    g[6] = jnp.sum(g[6], axis=1, keepdims=True)
    return tuple(g)


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


@pytest.mark.parametrize("cdt,tol_fwd,tol_grad", [
    (jnp.float32, 1e-5, 1e-4),
    (jnp.bfloat16, 2e-2, 1e-1),
])
def test_compiled_kernels_match_reference(monkeypatch, cdt, tol_fwd, tol_grad):
    monkeypatch.setenv("MAMBA3_CONTRACTION_PRECISION", "highest")
    args = _inputs()
    cot = jax.random.normal(jax.random.PRNGKey(9), (L, NH, P))
    ref = jax.jit(lambda *a: mamba3_ssd_chunked_jax(*a, chunk_size=CS))
    got = jax.jit(lambda *a: mamba3_ssd_chunked_pallas(
        *a, chunk_size=CS, compute_dtype=cdt, interpret=False))
    y_ref = ref(*_ref_args(args))
    y_got = got(*args)
    assert np.isfinite(np.asarray(y_got)).all()
    assert _rel(y_got, y_ref) < tol_fwd
    n = len(args)
    g_ref = _ref_grads_to_shared(jax.jit(jax.grad(
        lambda *a: jnp.sum(ref(*a) * cot), argnums=tuple(range(n))))(*_ref_args(args)))
    g_got = jax.jit(jax.grad(lambda *a: jnp.sum(got(*a) * cot), argnums=tuple(range(n))))(*args)
    names = ["Q", "K", "V", "ADT", "DT", "Trap", "angles", "Q_bias", "K_bias", "D", "Z"]
    errs = {nm: _rel(a, b) for nm, a, b in zip(names, g_got, g_ref)}
    worst = max(errs.values())
    assert worst < tol_grad, f"gradient disagreement {errs}"
