"""The fused Pallas SSD must compute what the XLA reference computes.

Runs the kernels through the Pallas interpreter on CPU, so it gates a change
before any GPU is available. Two families of checks: the forward matches the
reference at f32 to rounding, and every input gradient matches jax.grad of
the reference. The fused path takes B, C and the rotation angle rates once
for all heads (the model shares them); the reference takes them repeated, so
its Q, K and angle gradients are summed over heads before the comparison.
Shapes use power-of-two tiles (the Triton constraint, including the rotation
angles and the unrotated rest of the state) and a sequence length that is not
a multiple of the chunk, so padding is covered.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MAMBA3_CONTRACTION_PRECISION", "highest")

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import numpy as np                                           # noqa: E402
import pytest                                                # noqa: E402

from base_model.models.mamba3_jax import mamba3_ssd_chunked_jax          # noqa: E402
from base_model.models.mamba3_pallas import (                            # noqa: E402
    mamba3_ssd_chunked_pallas,
    pallas_tiles_supported,
)

jax.config.update("jax_enable_x64", False)

# (L, nh, N, R, P, CS): every tile >= 16 where Triton needs it.
SHAPES = [
    (70, 2, 64, 16, 16, 16),     # L not a multiple of CS -> padding path; 16 angles, rest 32
    (128, 3, 64, 32, 16, 32),    # rope on every state dim: no rest
    (200, 2, 128, 32, 32, 64),   # the production shape's 32 angles and rest 64; several chunks
]
NAMES = ["Q", "K", "V", "ADT", "DT", "Trap", "angles", "Q_bias", "K_bias", "D", "Z"]


def _inputs(L, nh, N, R, P, seed=0):
    """Fused-path arguments: Q, K (L, N) and angles (L, 1, R) shared over heads."""
    k = jax.random.split(jax.random.PRNGKey(seed), 11)
    Q = jax.random.normal(k[0], (L, N))
    K = jax.random.normal(k[1], (L, N))
    V = jax.random.normal(k[2], (L, nh, P))
    Z = jax.random.normal(k[3], (L, nh, P))
    DT = jax.nn.softplus(jax.random.normal(k[4], (nh, L)) - 2.0)          # positive
    A = -jax.nn.softplus(jax.random.normal(k[5], (nh, L)))                # negative
    ADT = A * DT
    Trap = jax.random.normal(k[6], (nh, L))
    angles = jax.random.normal(k[7], (L, 1, R))
    Q_bias = 0.1 * jax.random.normal(k[8], (nh, N))
    K_bias = 0.1 * jax.random.normal(k[9], (nh, N))
    D = jax.random.normal(k[10], (nh,))
    return (Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z)


def ref_args(args):
    """The same problem as the reference sees it: Q, K, angles repeated over heads."""
    Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z = args
    nh = V.shape[1]
    rep = lambda x: jnp.repeat(x[:, None, :] if x.ndim == 2 else x, nh, axis=1)
    return (rep(Q), rep(K), V, ADT, DT, Trap, rep(angles), Q_bias, K_bias, D, Z)


def ref_grads_to_shared(grads):
    """Reference gradients of the repeated Q, K, angles summed over heads."""
    g = list(grads)
    g[0] = jnp.sum(g[0], axis=1)                       # dQ (L, nh, N) -> (L, N)
    g[1] = jnp.sum(g[1], axis=1)
    g[6] = jnp.sum(g[6], axis=1, keepdims=True)        # dangles -> (L, 1, R)
    return tuple(g)


def reference(*args, chunk_size):
    return mamba3_ssd_chunked_jax(*ref_args(args), chunk_size=chunk_size)


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


@pytest.mark.parametrize("L,nh,N,R,P,CS", SHAPES)
def test_forward_matches_reference(L, nh, N, R, P, CS):
    args = _inputs(L, nh, N, R, P)
    ref = jax.jit(lambda *a: reference(*a, chunk_size=CS))(*args)
    got = jax.jit(lambda *a: mamba3_ssd_chunked_pallas(
        *a, chunk_size=CS, compute_dtype=jnp.float32, interpret=True))(*args)
    assert got.shape == ref.shape == (L, nh, P)
    assert np.isfinite(np.asarray(got)).all()
    err = _rel(got, ref)
    assert err < 1e-5, f"forward differs from the reference by {err:.3e} relative"


@pytest.mark.parametrize("L,nh,N,R,P,CS", SHAPES)
def test_gradients_match_reference(L, nh, N, R, P, CS):
    args = _inputs(L, nh, N, R, P, seed=1)
    cot = jax.random.normal(jax.random.PRNGKey(99), (L, nh, P))

    def loss_ref(*a):
        return jnp.sum(mamba3_ssd_chunked_jax(*a, chunk_size=CS) * cot)

    def loss_pallas(*a):
        return jnp.sum(mamba3_ssd_chunked_pallas(
            *a, chunk_size=CS, compute_dtype=jnp.float32, interpret=True) * cot)

    n = len(args)
    g_ref = ref_grads_to_shared(
        jax.jit(jax.grad(loss_ref, argnums=tuple(range(n))))(*ref_args(args)))
    g_got = jax.jit(jax.grad(loss_pallas, argnums=tuple(range(n))))(*args)
    for name, a, b in zip(NAMES, g_got, g_ref):
        assert a.shape == b.shape, name
        assert np.isfinite(np.asarray(a)).all(), f"d{name} has non-finite entries"
        err = _rel(a, b)
        assert err < 2e-5, f"d{name} differs from jax.grad of the reference by {err:.3e}"


def test_every_gradient_is_exercised():
    """Each kernel-produced gradient must be non-trivial in the test problem,
    or a wrong formula that returns zeros would pass the parity check."""
    L, nh, N, R, P, CS = 200, 2, 128, 32, 32, 64
    args = _inputs(L, nh, N, R, P, seed=11)
    cot = jax.random.normal(jax.random.PRNGKey(12), (L, nh, P))
    g = jax.grad(lambda *a: jnp.sum(mamba3_ssd_chunked_pallas(
        *a, chunk_size=CS, compute_dtype=jnp.float32, interpret=True) * cot),
        argnums=tuple(range(len(args))))(*args)
    for name, t in zip(NAMES, g):
        assert float(jnp.abs(t).max()) > 1e-3, f"d{name} is (nearly) zero in the test problem"


def test_angles_accepted_as_2d_or_per_head():
    """angles may come as (L, R), (L, 1, R) or already repeated (L, nh, R)."""
    L, nh, N, R, P, CS = 70, 2, 64, 16, 16, 16
    args = _inputs(L, nh, N, R, P, seed=13)
    f = lambda ang: mamba3_ssd_chunked_pallas(
        *args[:6], ang, *args[7:], chunk_size=CS, compute_dtype=jnp.float32, interpret=True)
    base = f(args[6])
    assert _rel(f(args[6][:, 0, :]), base) < 1e-6
    assert _rel(f(jnp.repeat(args[6], nh, axis=1)), base) < 1e-6


def test_vmap_over_batch_matches_per_sequence():
    """LOBS5 batches with vmap, with the parameters (biases, D) unbatched: the
    pallas_call and custom_vjp must map, forward and gradients."""
    L, nh, N, R, P, CS = 70, 2, 64, 16, 16, 16
    a0 = _inputs(L, nh, N, R, P, seed=2)
    a1 = _inputs(L, nh, N, R, P, seed=3)
    shared = {7, 8, 9}                                  # Q_bias, K_bias, D from a0 only
    batched = tuple(a0[i] if i in shared else jnp.stack([x, y])
                    for i, (x, y) in enumerate(zip(a0, a1)))
    a1 = tuple(a0[i] if i in shared else a1[i] for i in range(len(a1)))
    in_axes = tuple(None if i in shared else 0 for i in range(len(a0)))
    f = lambda *a: mamba3_ssd_chunked_pallas(
        *a, chunk_size=CS, compute_dtype=jnp.float32, interpret=True)
    out = jax.jit(jax.vmap(f, in_axes=in_axes))(*batched)
    assert _rel(out[0], f(*a0)) < 1e-6
    assert _rel(out[1], f(*a1)) < 1e-6
    cot = jax.random.normal(jax.random.PRNGKey(5), (2, L, nh, P))
    g = jax.jit(jax.grad(lambda *a: jnp.sum(jax.vmap(f, in_axes=in_axes)(*a) * cot),
                         argnums=(0, 7, 9)))(*batched)
    g0 = jax.grad(lambda *a: jnp.sum(f(*a) * cot[0]), argnums=(0, 7, 9))(*a0)
    g1 = jax.grad(lambda *a: jnp.sum(f(*a) * cot[1]), argnums=(0, 7, 9))(*a1)
    assert _rel(g[0][0], g0[0]) < 1e-5 and _rel(g[0][1], g1[0]) < 1e-5     # dQ per sequence
    assert _rel(g[1], g0[1] + g1[1]) < 1e-5                                # dQ_bias summed
    assert _rel(g[2], g0[2] + g1[2]) < 1e-5                                # dD summed


def test_bf16_compute_stays_within_bf16_rounding():
    L, nh, N, R, P, CS = 128, 2, 64, 16, 16, 32
    args = _inputs(L, nh, N, R, P, seed=4)
    ref = reference(*args, chunk_size=CS)
    got = mamba3_ssd_chunked_pallas(*args, chunk_size=CS,
                                    compute_dtype=jnp.bfloat16, interpret=True)
    err = _rel(got, ref)
    assert err < 2e-2, f"bf16 tensor-core inputs differ by {err:.3e}; bf16 floor is ~4e-3"
    assert err > 1e-7, "bf16 path produced f32-exact numbers: the cast is not applied"


def test_tile_rule():
    assert pallas_tiles_supported(64, 128, 64, 32)       # the 75m preset
    assert pallas_tiles_supported(16, 64, 16, 16)
    assert pallas_tiles_supported(32, 64, 16, 32)        # rope on every state dim
    assert not pallas_tiles_supported(64, 128, 10, 32)   # book layer headdim
    assert not pallas_tiles_supported(48, 128, 64, 32)   # chunk not a power of two
    assert not pallas_tiles_supported(64, 8, 64, 2)      # below the tensor-core tile
    assert not pallas_tiles_supported(64, 128, 64, 8)    # 8 angles: below the tile
    assert not pallas_tiles_supported(64, 128, 64, 48)   # 48 angles: not a power of two
    assert not pallas_tiles_supported(64, 128, 64, 0)    # no rope: the kernels always rotate
    assert not pallas_tiles_supported(64, 32, 64, 8)     # Kang's d_state 32 configuration


def test_bf16_tiles_forward_and_backward():
    """io_dtype=bf16 stores the per-head tiles the kernels exchange (rotated Q,
    scaled K, V, Z, Y and the gradient partials) in bf16. The kernels must upcast for their elementwise math (the Triton
    lowering has no bf16 elementwise ops; job 6393377) and the gradients must
    come back finite, within bf16 rounding of the reference."""
    L, nh, N, R, P, CS = 128, 2, 64, 16, 16, 32
    args = _inputs(L, nh, N, R, P, seed=5)
    cot = jax.random.normal(jax.random.PRNGKey(6), (L, nh, P))
    ref = reference(*args, chunk_size=CS)
    f = lambda *a: mamba3_ssd_chunked_pallas(*a, chunk_size=CS, compute_dtype=jnp.bfloat16,
                                             io_dtype=jnp.bfloat16, interpret=True)
    got = f(*args)
    assert got.dtype == jnp.bfloat16
    assert _rel(got, ref) < 3e-2
    n = len(args)
    g_ref = ref_grads_to_shared(jax.grad(
        lambda *a: jnp.sum(mamba3_ssd_chunked_jax(*a, chunk_size=CS) * cot),
        argnums=tuple(range(n)))(*ref_args(args)))
    g_got = jax.grad(lambda *a: jnp.sum(f(*a).astype(jnp.float32) * cot),
                     argnums=tuple(range(n)))(*args)
    for name, a, b in zip(NAMES, g_got, g_ref):
        assert a.dtype == b.dtype, name
        assert a.shape == b.shape, name
        assert np.isfinite(np.asarray(a)).all(), name
    assert _rel(g_got[2], g_ref[2]) < 5e-2, "dV"
    assert _rel(g_got[0], g_ref[0]) < 5e-2, "dQ"
    assert _rel(g_got[1], g_ref[1]) < 5e-2, "dK"


@pytest.mark.parametrize("nc,blk", [(5, 16), (37, 8), (64, 16), (70, 16)])
def test_blocked_state_passing_matches_dense(nc, blk):
    """The two-level scan reassociates the dense contraction; it must agree
    to rounding, including when nc is not a multiple of the block."""
    from base_model.models.mamba3_pallas import state_passing_blocked, state_passing_jax
    nh, P, N = 3, 16, 16
    k = jax.random.split(jax.random.PRNGKey(nc), 2)
    states = jax.random.normal(k[0], (nc, nh, P, N))
    A_end = -jax.random.uniform(k[1], (nh, nc), minval=0.05, maxval=3.0)
    prec = jax.lax.Precision.HIGHEST
    dense = state_passing_jax(states, A_end, prec)
    blocked = state_passing_blocked(states, A_end, prec, block=blk)
    assert blocked.shape == dense.shape
    assert _rel(blocked, dense) < 1e-5
    assert np.allclose(np.asarray(blocked[0]), 0.0)
    cot = jax.random.normal(jax.random.PRNGKey(7), dense.shape)
    g_d = jax.grad(lambda s, a: jnp.sum(state_passing_jax(s, a, prec) * cot), argnums=(0, 1))(states, A_end)
    g_b = jax.grad(lambda s, a: jnp.sum(state_passing_blocked(s, a, prec, block=blk) * cot), argnums=(0, 1))(states, A_end)
    assert _rel(g_b[0], g_d[0]) < 1e-5 and _rel(g_b[1], g_d[1]) < 1e-5


def test_state_passing_knob_selects_the_blocked_form(monkeypatch):
    from base_model.models.mamba3_pallas import state_passing_mode
    monkeypatch.delenv("MAMBA3_STATE_PASSING", raising=False)
    assert state_passing_mode() == "blocked"
    monkeypatch.setenv("MAMBA3_STATE_PASSING", "dense")
    assert state_passing_mode() == "dense"
    monkeypatch.setenv("MAMBA3_STATE_PASSING", "blocked")
    L, nh, N, R, P, CS = 200, 2, 64, 16, 32, 16          # 13 chunks: not a multiple of 16
    args = _inputs(L, nh, N, R, P, seed=8)
    ref = reference(*args, chunk_size=CS)
    got = mamba3_ssd_chunked_pallas(*args, chunk_size=CS, compute_dtype=jnp.float32, interpret=True)
    assert _rel(got, ref) < 1e-5
    monkeypatch.setenv("MAMBA3_STATE_PASSING", "chunked")
    with pytest.raises(ValueError, match="MAMBA3_STATE_PASSING"):
        state_passing_mode()


def test_unsupported_shapes_are_refused_by_the_entry_point():
    L, nh, N, R, P, CS = 64, 2, 64, 16, 16, 16
    args = list(_inputs(L, nh, N, R, P, seed=9))
    args[6] = args[6][..., :8]                          # R = 8: below the tile
    with pytest.raises(ValueError, match="powers of two"):
        mamba3_ssd_chunked_pallas(*args, chunk_size=CS, compute_dtype=jnp.float32, interpret=True)
    args = list(_inputs(L, nh, N, R, P, seed=9))
    args[0] = jnp.repeat(args[0][:, None, :], nh, axis=1)   # per-head Q: not the fused contract
    with pytest.raises(ValueError, match="shared over heads"):
        mamba3_ssd_chunked_pallas(*args, chunk_size=CS, compute_dtype=jnp.float32, interpret=True)
