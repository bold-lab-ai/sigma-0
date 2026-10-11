"""Fused Pallas kernels for the Mamba3 SISO chunked SSD (phases 1-7).

Why this exists. The XLA reference in mamba3_jax.py expresses the SSD as
multi-operand einsums over per-head tensors. XLA turns each into pairwise dots
with relayout copies and materialises every intermediate in HBM: the
(nheads, nc, CS, CS) decay masks, the (nc, nheads, headdim, d_state) chunk
states, their gradients, and in phases 1-2 the per-head rotated query and
scaled key, (L, nheads, d_state) float32 each, built from a B and a C that
are the same for every head. For the 75m preset at 13,000 tokens that is
hundreds of megabytes per layer per sequence, read and written several times
in the forward and again in the backward. The step is bound by that traffic,
not by arithmetic (PR #32, jobs 6224012 and 6287769; the kernel profile of
job 6395176 put 40% of the layer in XLA relayout and reduce fusions).

What the kernels do instead. Three stages, one program per (chunk, head):

  phase1      reads the CS x d_state tile of the shared B and C once, the
              head's biases, cos and sin of its cumulative rotation angles
              for the chunk and its trapezoidal scale, and writes the head's
              rotated query and scaled key tiles in the tile dtype plus the
              QK-dot skip term. Elementwise; its backward applies the
              transposed rotation, bias and scale in the same way and writes
              the per-head gradient of the shared B and C in the tile dtype
              (summed over heads by XLA), the bias gradients, the angle and
              scale gradients. Nothing per head in float32 ever exists.
  chunk_state the per-chunk state contribution (phase 4).
  chunk_scan  the masked intra-chunk product, the state-to-output term, the
              feedthrough and the SiLU gate (phases 3, 6, 7).

The SSD kernels keep the CS x CS decay mask and every product in registers
and write the CS x headdim output once; their backward recomputes the forward
tile instead of reading saved intermediates (the FlashAttention recipe), so
nothing of size CS x CS ever touches HBM. The chunk_scan backward is three
kernels (gate and state, value, query and key), each recomputing the few
products it needs: one kernel holding every tile of the backward ran at 32
registers per thread and 6% occupancy, 2.2 ms per layer-step; the three
together take 1.0 ms (jobs 6422282, 6425384). Phase 1 is a separate kernel on
purpose: folding it into the SSD kernels (job 6416484) made the compiler cap
the scan kernels at 32 registers per thread and spill to local memory (job
6421949: 3.8e8 local loads in the forward, occupancy 6-12%), and every kernel
ran 3-4x slower. Left in JAX: the cumulative rotation angles (a cumsum over
the sequence), the trapezoidal weights, the decay cumsum and phase 5, the
cross-chunk state passing, a small two-level scan that autodiff handles.

The decomposition, per chunk c and head h, with the shared C, B (CS, N), the
head's biases qb, kb (N), angles theta (CS, R), weights gamma, scale (CS),
feedthrough D, V, Z (CS, P), a = ADT (CS) and cum = cumsum(a):

    Q            = rope(C + qb, theta)                            (phase 1)
    K            = rope(B + kb, theta) * scale
    QK_dot       = sum_n (C + qb)(B + kb) * gamma
    L[l, s]      = exp(cum[l] - cum[s])  for l > s, else 0        (phase 3)
    Y_diag       = (Q K^T * L) V
    states       = (V * exp(cum[-1] - cum))^T K       -> (P, N)   (phase 4)
    state_in     = state passing over chunks           (phase 5, JAX)
    Y_off        = exp(cum)[:, None] * (Q state_in^T)              (phase 6)
    Y            = (Y_diag + Y_off + (D + QK_dot) V) * silu(Z)     (phase 7)

The rotation acts on interleaved pairs (2j, 2j+1) of the first 2R state
dims. The phase-1 kernels address the even and the odd members of the pairs
with stride-2 slices of the tile, so the rotation and its transpose are
elementwise between two (CS, R) tiles, exact, and the layout stays the
reference's.

Selection. Mamba3SSM reads MAMBA3_SSD_IMPL: "xla" (default, the reference)
or "pallas". Contraction precision follows the existing knob
MAMBA3_CONTRACTION_PRECISION ("highest" keeps full f32, "default" allows
TF32), and MAMBA3_PALLAS_DTYPE ("float32" or "bfloat16") sets the dtype the
tensor cores see; accumulation is always f32. Off GPU the kernels run in
Pallas interpret mode, which is what the CPU tests use.

Tile constraint. Triton needs power-of-two tile sizes, so chunk_size,
d_state, headdim, the number of rotation angles R and the unrotated rest
d_state - 2R (if any) must be powers of two and at least 16. Shapes that do
not qualify (the small book pre-layers, a d_state below 64 at rope_fraction
0.5) fall back to the reference.
"""
from __future__ import annotations

import functools
import os
import warnings

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from base_model.models.mamba3_jax import (
    angle_dt_cumsum,
    mamba3_contraction_precision,
    mamba3_contraction_precision_mode,
    segsum,
    trapezoid_gamma_scale,
)

_MIN_TILE = 16


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

def ssd_impl_mode() -> str:
    """Which SSD implementation Mamba3SSM uses: xla or pallas."""
    mode = os.environ.get("MAMBA3_SSD_IMPL") or "xla"
    if mode not in {"xla", "pallas"}:
        raise ValueError("MAMBA3_SSD_IMPL must be xla or pallas")
    return mode


def pallas_compute_dtype(default=jnp.float32):
    """dtype handed to the tensor cores inside the kernels (accumulation f32).

    MAMBA3_PALLAS_DTYPE overrides; unset, the layer's compute dtype (--dtype)
    decides, so a bf16 layer gets bf16 tensor-core operands without a second
    knob to remember.
    """
    name = os.environ.get("MAMBA3_PALLAS_DTYPE") or jnp.dtype(default).name
    if name not in {"float32", "bfloat16"}:
        raise ValueError("MAMBA3_PALLAS_DTYPE must be float32 or bfloat16")
    return jnp.dtype(name)


def _dot_precision():
    # "highest" keeps f32 contractions exact (no TF32); "default" lets Triton
    # use TF32 for f32 operands, matching the reference's semantics.
    if mamba3_contraction_precision_mode() == "highest":
        return lax.Precision.HIGHEST
    return lax.Precision.DEFAULT


def _pow2_tile(n: int) -> bool:
    return n >= _MIN_TILE and (n & (n - 1)) == 0


def pallas_tiles_supported(chunk_size: int, d_state: int, headdim: int,
                           num_rope_angles: int) -> bool:
    """Triton tiles must be powers of two and large enough for tensor cores:
    the chunk, the state, the head, the rotation angles and the unrotated rest."""
    if not (_pow2_tile(chunk_size) and _pow2_tile(d_state) and _pow2_tile(headdim)):
        return False
    R = int(num_rope_angles)
    if not (_pow2_tile(R) and 2 * R <= d_state):
        return False
    rest = d_state - 2 * R
    return rest == 0 or _pow2_tile(rest)


def _interpret_default() -> bool:
    # Compiled Triton kernels exist for GPU only; everywhere else the same
    # kernels run through the Pallas interpreter (correctness, not speed).
    return jax.default_backend() != "gpu"


# ---------------------------------------------------------------------------
# Phase 1 kernels. Tiles: C, B (CS, N) shared over heads; biases (N,); cos, sin
# (CS, R); scale (CS,). The even and odd members of the rotated pairs are the
# stride-2 slices [0:2R:2] and [1:2R:2]; the rest [2R:N] passes through.
# ---------------------------------------------------------------------------

def _pairs(ref, R, N, row=slice(None)):
    """(even, odd, rest) float32 views of a tile whose last dim is the state;
    rest is None when the rotation covers every state dim."""
    idx = (lambda s: (row, s)) if ref.ndim == 2 else (lambda s: (s,))
    e = ref[idx(pl.ds(0, R, stride=2))].astype(jnp.float32)
    o = ref[idx(pl.ds(1, R, stride=2))].astype(jnp.float32)
    r = ref[idx(pl.ds(2 * R, N - 2 * R))].astype(jnp.float32) if N > 2 * R else None
    return e, o, r


def _store_pairs(ref, R, N, e, o, r):
    ref[:, pl.ds(0, R, stride=2)] = e.astype(ref.dtype)
    ref[:, pl.ds(1, R, stride=2)] = o.astype(ref.dtype)
    if r is not None:
        ref[:, pl.ds(2 * R, N - 2 * R)] = r.astype(ref.dtype)


def _store_pairs_1d(ref, R, N, e, o, r):
    ref[pl.ds(0, R, stride=2)] = e.astype(ref.dtype)
    ref[pl.ds(1, R, stride=2)] = o.astype(ref.dtype)
    if r is not None:
        ref[pl.ds(2 * R, N - 2 * R)] = r.astype(ref.dtype)


def _biased(tile_ref, bias_ref, R, N):
    """(even, odd, rest) of tile + bias, float32."""
    te, to, tr = _pairs(tile_ref, R, N)
    be, bo, br = _pairs(bias_ref, R, N)
    return te + be[None, :], to + bo[None, :], (None if tr is None else tr + br[None, :])


def _rowsum_pairs(ae, ao, ar, be, bo, br):
    """sum over the state dim of a * b -> (CS,)."""
    s = jnp.sum(ae * be + ao * bo, axis=1)
    return s if ar is None else s + jnp.sum(ar * br, axis=1)


def _phase1_fwd_kernel(c_ref, b_ref, qbias_ref, kbias_ref, cos_ref, sin_ref, scale_ref,
                       q_ref, k_ref, qk_ref, *, R, N):
    cs, sn = cos_ref[...], sin_ref[...]                    # (CS, R)
    qe, qo, qr = _biased(c_ref, qbias_ref, R, N)           # C + qb
    ke, ko, kr = _biased(b_ref, kbias_ref, R, N)           # B + kb
    qk_ref[...] = _rowsum_pairs(qe, qo, qr, ke, ko, kr).astype(qk_ref.dtype)
    # rope: even' = even cos - odd sin, odd' = even sin + odd cos (mamba3_jax.apply_rope)
    _store_pairs(q_ref, R, N, qe * cs - qo * sn, qe * sn + qo * cs, qr)
    scl = scale_ref[...][:, None]                           # (CS, 1)
    _store_pairs(k_ref, R, N, (ke * cs - ko * sn) * scl, (ke * sn + ko * cs) * scl,
                 None if kr is None else kr * scl)


def _phase1_bwd_kernel(c_ref, b_ref, qbias_ref, kbias_ref, cos_ref, sin_ref, scale_ref,
                       dq_ref, dk_ref, dqk_ref,
                       dc_ref, db_ref, dqbias_ref, dkbias_ref, dth_ref, dscale_ref, *, R, N):
    cs, sn = cos_ref[...], sin_ref[...]
    qe, qo, qr = _biased(c_ref, qbias_ref, R, N)
    ke, ko, kr = _biased(b_ref, kbias_ref, R, N)
    scl = scale_ref[...][:, None]
    de, do, dr = _pairs(dq_ref, R, N)                       # cotangent of the rotated query
    ge, go, gr = _pairs(dk_ref, R, N)                       # cotangent of the scaled key
    dqk = dqk_ref[...][:, None]                             # cotangent of the QK-dot skip (CS, 1)

    # K = rope(B + kb) * scale: dscale = sum_n dK * rope(B + kb); d rope = dK * scale.
    kre, kro = ke * cs - ko * sn, ke * sn + ko * cs
    dscale = jnp.sum(ge * kre + go * kro, axis=1)
    if kr is not None:
        dscale = dscale + jnp.sum(gr * kr, axis=1)
    dscale_ref[...] = dscale.astype(dscale_ref.dtype)
    ge, go = ge * scl, go * scl
    gr = None if gr is None else gr * scl

    # Transposed rotation: even = d_even cos + d_odd sin, odd = -d_even sin + d_odd cos.
    dqe, dqo = de * cs + do * sn, -de * sn + do * cs
    dke, dko = ge * cs + go * sn, -ge * sn + go * cs
    # d theta: d(even')/dtheta = -even sin - odd cos, d(odd')/dtheta = even cos - odd sin.
    dth = (-sn * (de * qe + do * qo) + cs * (do * qe - de * qo)
           - sn * (ge * ke + go * ko) + cs * (go * ke - ge * ko))
    dth_ref[...] = dth.astype(dth_ref.dtype)
    # QK_dot = sum_n (C + qb)(B + kb): the skip term's cotangent flows to both.
    dqe, dqo = dqe + dqk * ke, dqo + dqk * ko
    dke, dko = dke + dqk * qe, dko + dqk * qo
    dqr = None if dr is None else dr + dqk * kr
    dkr = None if gr is None else gr + dqk * qr

    _store_pairs(dc_ref, R, N, dqe, dqo, dqr)
    _store_pairs(db_ref, R, N, dke, dko, dkr)
    _store_pairs_1d(dqbias_ref, R, N, jnp.sum(dqe, axis=0), jnp.sum(dqo, axis=0),
                    None if dqr is None else jnp.sum(dqr, axis=0))
    _store_pairs_1d(dkbias_ref, R, N, jnp.sum(dke, axis=0), jnp.sum(dko, axis=0),
                    None if dkr is None else jnp.sum(dkr, axis=0))


# ---------------------------------------------------------------------------
# SSD kernel bodies. All refs are already sliced to one (chunk, head) tile.
# ---------------------------------------------------------------------------

def _causal_decay(cum):
    """L[l, s] = exp(cum[l] - cum[s]) strictly below the diagonal, else 0."""
    CS = cum.shape[0]
    l_idx = lax.broadcasted_iota(jnp.int32, (CS, CS), 0)
    s_idx = lax.broadcasted_iota(jnp.int32, (CS, CS), 1)
    causal = l_idx > s_idx
    # Mask before the exp so above-diagonal entries never overflow.
    diff = jnp.where(causal, cum[:, None] - cum[None, :], 0.0)
    return jnp.where(causal, jnp.exp(diff), 0.0)


def _last(cum):
    """cum[CS - 1] without a data-dependent index (Triton has none)."""
    CS = cum.shape[0]
    idx = lax.broadcasted_iota(jnp.int32, (CS,), 0)
    return jnp.sum(jnp.where(idx == CS - 1, cum, 0.0))


def _f32(ref):
    """Load a tile as float32. Tiles may be stored in bf16 (io_dtype); the
    Pallas Triton lowering has no bf16 x bf16 elementwise ops, and the dots
    cast their operands to the compute dtype themselves."""
    return ref[...].astype(jnp.float32)


def _chunk_state_fwd_kernel(k_ref, v_ref, cum_ref, states_ref, *, cdt, prec):
    k = _f32(k_ref)                      # (CS, N)
    v = _f32(v_ref)                      # (CS, P)
    cum = cum_ref[...]                   # (CS,)
    ds = jnp.exp(_last(cum) - cum)       # decay from each position to chunk end
    v_s = (v * ds[:, None]).astype(cdt)
    states = pl.dot(v_s, k.astype(cdt), trans_a=True, precision=prec)  # (P, N)
    states_ref[...] = states.astype(states_ref.dtype)


def _chunk_state_bwd_kernel(k_ref, v_ref, cum_ref, dst_ref,
                            dk_ref, dv_ref, dcum_ref, *, cdt, prec):
    k = _f32(k_ref)
    v = _f32(v_ref)
    cum = cum_ref[...]
    dst = _f32(dst_ref)                  # (P, N) gradient w.r.t. states
    ds = jnp.exp(_last(cum) - cum)
    vd = pl.dot(v.astype(cdt), dst.astype(cdt), precision=prec)               # (CS, N)
    kd = pl.dot(k.astype(cdt), dst.astype(cdt), trans_b=True, precision=prec) # (CS, P)
    dk_ref[...] = (ds[:, None] * vd).astype(dk_ref.dtype)
    dv_ref[...] = (ds[:, None] * kd).astype(dv_ref.dtype)
    # states = sum_s ds[s] V[s] K[s]^T, ds[s] = exp(cum[-1] - cum[s]).
    dds = jnp.sum(vd * k, axis=1)        # (CS,)
    CS = cum.shape[0]
    idx = lax.broadcasted_iota(jnp.int32, (CS,), 0)
    dcum = -dds * ds + jnp.where(idx == CS - 1, jnp.sum(dds * ds), 0.0)
    dcum_ref[...] = dcum.astype(dcum_ref.dtype)


def _chunk_scan_fwd_kernel(q_ref, k_ref, v_ref, z_ref, sin_ref, cum_ref, dq_ref,
                           y_ref, *, cdt, prec):
    q = _f32(q_ref)                      # (CS, N)
    k = _f32(k_ref)                      # (CS, N)
    v = _f32(v_ref)                      # (CS, P)
    z = _f32(z_ref)                      # (CS, P)
    s_in = _f32(sin_ref)                 # (P, N) state entering this chunk
    cum = cum_ref[...]                   # (CS,)
    dq = dq_ref[...]                     # (CS,)  D + QK_dot * gamma
    Lm = _causal_decay(cum)
    S = pl.dot(q.astype(cdt), k.astype(cdt), trans_b=True, precision=prec)      # (CS, CS)
    y_diag = pl.dot((S * Lm).astype(cdt), v.astype(cdt), precision=prec)        # (CS, P)
    qs = pl.dot(q.astype(cdt), s_in.astype(cdt), trans_b=True, precision=prec)  # (CS, P)
    y = y_diag + jnp.exp(cum)[:, None] * qs + dq[:, None] * v
    y = y * (z * jax.nn.sigmoid(z))
    y_ref[...] = y.astype(y_ref.dtype)


def _chunk_scan_bwd_gate_kernel(q_ref, k_ref, v_ref, z_ref, sin_ref, cum_ref, dq_ref, dy_ref,
                                dz_ref, ddq_ref, dsin_ref, dcum_ref, *, cdt, prec):
    """Backward of chunk_scan, part 1: the gate, the feedthrough weight, the
    incoming state, and the decay's share through the cross-chunk term."""
    q = _f32(q_ref)
    k = _f32(k_ref)
    v = _f32(v_ref)
    z = _f32(z_ref)
    s_in = _f32(sin_ref)
    cum = cum_ref[...]
    dq = dq_ref[...]
    dy = _f32(dy_ref)

    # Recompute the pre-gate output of this tile (cheaper than storing it).
    Lm = _causal_decay(cum)
    S = pl.dot(q.astype(cdt), k.astype(cdt), trans_b=True, precision=prec)      # (CS, CS)
    y_diag = pl.dot((S * Lm).astype(cdt), v.astype(cdt), precision=prec)        # (CS, P)
    qs = pl.dot(q.astype(cdt), s_in.astype(cdt), trans_b=True, precision=prec)  # (CS, P)
    e = jnp.exp(cum)
    yp = y_diag + e[:, None] * qs + dq[:, None] * v

    # Gate: y = yp * z * sigmoid(z).
    sig = jax.nn.sigmoid(z)
    silu = z * sig
    dz_ref[...] = (dy * yp * (sig * (1.0 + z * (1.0 - sig)))).astype(dz_ref.dtype)
    g = dy * silu                                                              # d/d yp

    # Feedthrough (D + QK_dot) * V, and the cross-chunk term exp(cum) * (Q state_in^T).
    ddq_ref[...] = jnp.sum(g * v, axis=1).astype(ddq_ref.dtype)
    ge = g * e[:, None]
    dsin_ref[...] = pl.dot(ge.astype(cdt), q.astype(cdt), trans_a=True,
                           precision=prec).astype(dsin_ref.dtype)             # (P, N)
    dcum_ref[...] = (jnp.sum(g * qs, axis=1) * e).astype(dcum_ref.dtype)


def _chunk_scan_bwd_value_kernel(q_ref, k_ref, v_ref, z_ref, cum_ref, dq_ref, dy_ref,
                                 dv_ref, dcum_ref, *, cdt, prec):
    """Backward of chunk_scan, part 2: the value, and the decay's share through
    the intra-chunk mask."""
    q = _f32(q_ref)
    k = _f32(k_ref)
    v = _f32(v_ref)
    z = _f32(z_ref)
    cum = cum_ref[...]
    dq = dq_ref[...]
    dy = _f32(dy_ref)

    Lm = _causal_decay(cum)
    S = pl.dot(q.astype(cdt), k.astype(cdt), trans_b=True, precision=prec)      # (CS, CS)
    M = S * Lm
    g = dy * (z * jax.nn.sigmoid(z))                                           # d/d yp
    # Feedthrough plus the intra-chunk term (Q K^T * L) V.
    dv = dq[:, None] * g + pl.dot(M.astype(cdt), g.astype(cdt), trans_a=True, precision=prec)
    dv_ref[...] = dv.astype(dv_ref.dtype)
    dM = pl.dot(g.astype(cdt), v.astype(cdt), trans_b=True, precision=prec)    # (CS, CS)
    t = dM * S * Lm                      # dL * L, zero above the diagonal
    dcum_ref[...] = (jnp.sum(t, axis=1) - jnp.sum(t, axis=0)).astype(dcum_ref.dtype)


def _chunk_scan_bwd_qk_kernel(q_ref, k_ref, v_ref, z_ref, sin_ref, cum_ref, dy_ref,
                              dq_out_ref, dk_ref, *, cdt, prec):
    """Backward of chunk_scan, part 3: the query and the key."""
    q = _f32(q_ref)
    k = _f32(k_ref)
    v = _f32(v_ref)
    z = _f32(z_ref)
    s_in = _f32(sin_ref)
    cum = cum_ref[...]
    dy = _f32(dy_ref)

    Lm = _causal_decay(cum)
    e = jnp.exp(cum)
    g = dy * (z * jax.nn.sigmoid(z))                                           # d/d yp
    ge = g * e[:, None]
    dM = pl.dot(g.astype(cdt), v.astype(cdt), trans_b=True, precision=prec)    # (CS, CS)
    dS = dM * Lm
    # Cross-chunk term exp(cum) * (Q state_in^T) and intra-chunk term (Q K^T * L) V.
    dq_out = (pl.dot(ge.astype(cdt), s_in.astype(cdt), precision=prec)
              + pl.dot(dS.astype(cdt), k.astype(cdt), precision=prec))           # (CS, N)
    dq_out_ref[...] = dq_out.astype(dq_out_ref.dtype)
    dk_ref[...] = pl.dot(dS.astype(cdt), q.astype(cdt), trans_a=True,
                         precision=prec).astype(dk_ref.dtype)


# ---------------------------------------------------------------------------
# pallas_call wrappers. Layouts:
#   Cp, Bp: (L_pad, N) shared over heads   qbias, kbias: (nh, N)   ang, cos, sin: (L_pad, nh, R)
#   scale: (nh, L_pad)                     Q_c, K_c: (nc, CS, nh, N)   V_c, Z_c, Y: (nc, CS, nh, P)
#   A_cumsum: (nh, nc, CS)   dq_c, qk_c: (nc, CS, nh)   states, state_in: (nc, nh, P, N)
#   per-head gradient partials of Cp, Bp: (L_pad, nh, N); of the biases: (nc, nh, N)
# ---------------------------------------------------------------------------

def _spec_ln(CS, N):                # (L_pad, N) -> tile (CS, N), same for every head
    return pl.BlockSpec((CS, N), lambda c, h: (c, 0))


def _spec_lhn(CS, N):               # (L_pad, nh, N) -> tile (CS, N)
    return pl.BlockSpec((CS, None, N), lambda c, h: (c, h, 0))


def _spec_hn(N):                    # (nh, N) -> tile (N,)
    return pl.BlockSpec((None, N), lambda c, h: (h, 0))


def _spec_chn(N):                   # (nc, nh, N) -> tile (N,)
    return pl.BlockSpec((None, None, N), lambda c, h: (c, h, 0))


def _spec_lhr(CS, R):               # (L_pad, nh, R) -> tile (CS, R)
    return pl.BlockSpec((CS, None, R), lambda c, h: (c, h, 0))


def _spec_hl(CS):                   # (nh, L_pad) -> tile (CS,)
    return pl.BlockSpec((None, CS), lambda c, h: (h, c))


def _spec_lcn(CS, last):            # (nc, CS, nh, last) -> tile (CS, last)
    return pl.BlockSpec((None, CS, None, last), lambda c, h: (c, 0, h, 0))


def _spec_hcl(CS):                  # (nh, nc, CS) -> tile (CS,)
    return pl.BlockSpec((None, None, CS), lambda c, h: (h, c, 0))


def _spec_clh(CS):                  # (nc, CS, nh) -> tile (CS,)
    return pl.BlockSpec((None, CS, None), lambda c, h: (c, 0, h))


def _spec_state(P, N):              # (nc, nh, P, N) -> tile (P, N)
    return pl.BlockSpec((None, None, P, N), lambda c, h: (c, h, 0, 0))


def _call(kernel, *, grid, in_specs, out_specs, out_shape, interpret, cfg):
    compiler_params = None
    if not interpret:
        from jax.experimental.pallas import triton as plgpu
        compiler_params = plgpu.CompilerParams(num_warps=cfg["num_warps"],
                                               num_stages=cfg["num_stages"])
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=compiler_params,
    )


@functools.lru_cache(maxsize=None)
def _build(cdt_name: str, prec_name: str, interpret: bool, num_warps: int, num_stages: int):
    """Build the three custom_vjp functions (phase1, chunk_state, chunk_scan)
    for one static configuration."""
    cdt = jnp.dtype(cdt_name)
    prec = getattr(lax.Precision, prec_name)
    cfg = {"num_warps": num_warps, "num_stages": num_stages}
    kw = dict(cdt=cdt, prec=prec)
    f32 = jnp.float32
    SDS = jax.ShapeDtypeStruct

    # ---- phase 1: shared B, C -> per-head rotated query, scaled key, QK-dot ----
    def _p1_specs(CS, N, R):
        return [_spec_ln(CS, N), _spec_ln(CS, N), _spec_hn(N), _spec_hn(N), _spec_lhr(CS, R),
                _spec_lhr(CS, R), _spec_hl(CS)]

    @functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
    def phase1(Cp, Bp, qbias, kbias, ang, scale, CS, io_dtype_name):
        return _phase1_run(Cp, Bp, qbias, kbias, ang, scale, CS, io_dtype_name)

    def _phase1_run(Cp, Bp, qbias, kbias, ang, scale, CS, io_dtype_name):
        L_pad, N = Cp.shape
        nh, R = ang.shape[1], ang.shape[2]
        nc = L_pad // CS
        io = jnp.dtype(io_dtype_name)
        cs, sn = jnp.cos(ang), jnp.sin(ang)                  # (L_pad, nh, R), XLA
        f = _call(
            functools.partial(_phase1_fwd_kernel, R=R, N=N),
            grid=(nc, nh),
            in_specs=_p1_specs(CS, N, R),
            out_specs=[_spec_lcn(CS, N), _spec_lcn(CS, N), _spec_clh(CS)],
            out_shape=[SDS((nc, CS, nh, N), io), SDS((nc, CS, nh, N), io), SDS((nc, CS, nh), f32)],
            interpret=interpret, cfg=cfg)
        return f(Cp, Bp, qbias, kbias, cs, sn, scale)

    def phase1_fwd_rule(Cp, Bp, qbias, kbias, ang, scale, CS, io_dtype_name):
        return (_phase1_run(Cp, Bp, qbias, kbias, ang, scale, CS, io_dtype_name),
                (Cp, Bp, qbias, kbias, ang, scale))

    def phase1_bwd_rule(CS, io_dtype_name, res, cot):
        Cp, Bp, qbias, kbias, ang, scale = res
        dQ_c, dK_c, dqk_c = cot
        L_pad, N = Cp.shape
        nh, R = ang.shape[1], ang.shape[2]
        nc = L_pad // CS
        io = jnp.dtype(io_dtype_name)
        cs, sn = jnp.cos(ang), jnp.sin(ang)
        f = _call(
            functools.partial(_phase1_bwd_kernel, R=R, N=N),
            grid=(nc, nh),
            in_specs=_p1_specs(CS, N, R) + [_spec_lcn(CS, N), _spec_lcn(CS, N), _spec_clh(CS)],
            out_specs=[_spec_lhn(CS, N), _spec_lhn(CS, N), _spec_chn(N), _spec_chn(N),
                       _spec_lhr(CS, R), _spec_hl(CS)],
            out_shape=[SDS((L_pad, nh, N), io),      # per-head partial of dCp
                       SDS((L_pad, nh, N), io),      # per-head partial of dBp
                       SDS((nc, nh, N), f32),        # per-chunk partial of dqbias
                       SDS((nc, nh, N), f32),        # per-chunk partial of dkbias
                       SDS((L_pad, nh, R), f32),     # dang
                       SDS((nh, L_pad), f32)],       # dscale
            interpret=interpret, cfg=cfg)
        dCh, dBh, dqbias_c, dkbias_c, dang, dscale = f(
            Cp, Bp, qbias, kbias, cs, sn, scale,
            dQ_c.astype(io), dK_c.astype(io), dqk_c.astype(f32))
        return (jnp.sum(dCh.astype(f32), axis=1), jnp.sum(dBh.astype(f32), axis=1),
                jnp.sum(dqbias_c, axis=0), jnp.sum(dkbias_c, axis=0), dang, dscale)

    phase1.defvjp(phase1_fwd_rule, phase1_bwd_rule)

    # ---- phase 4: chunk states --------------------------------------------
    @jax.custom_vjp
    def chunk_state(K_c, V_c, A_cumsum):
        return _chunk_state_fwd(K_c, V_c, A_cumsum)

    def _chunk_state_fwd(K_c, V_c, A_cumsum):
        nc, CS, nh, N = K_c.shape
        P = V_c.shape[-1]
        f = _call(
            functools.partial(_chunk_state_fwd_kernel, **kw),
            grid=(nc, nh),
            in_specs=[_spec_lcn(CS, N), _spec_lcn(CS, P), _spec_hcl(CS)],
            out_specs=_spec_state(P, N),
            out_shape=jax.ShapeDtypeStruct((nc, nh, P, N), jnp.float32),
            interpret=interpret, cfg=cfg)
        return f(K_c, V_c, A_cumsum)

    def chunk_state_fwd_rule(K_c, V_c, A_cumsum):
        return _chunk_state_fwd(K_c, V_c, A_cumsum), (K_c, V_c, A_cumsum)

    def chunk_state_bwd_rule(res, dstates):
        K_c, V_c, A_cumsum = res
        nc, CS, nh, N = K_c.shape
        P = V_c.shape[-1]
        f = _call(
            functools.partial(_chunk_state_bwd_kernel, **kw),
            grid=(nc, nh),
            in_specs=[_spec_lcn(CS, N), _spec_lcn(CS, P), _spec_hcl(CS), _spec_state(P, N)],
            out_specs=[_spec_lcn(CS, N), _spec_lcn(CS, P), _spec_hcl(CS)],
            out_shape=[jax.ShapeDtypeStruct(K_c.shape, K_c.dtype),
                       jax.ShapeDtypeStruct(V_c.shape, V_c.dtype),
                       jax.ShapeDtypeStruct(A_cumsum.shape, A_cumsum.dtype)],
            interpret=interpret, cfg=cfg)
        dK, dV, dcum = f(K_c, V_c, A_cumsum, dstates.astype(jnp.float32))
        return dK, dV, dcum

    chunk_state.defvjp(chunk_state_fwd_rule, chunk_state_bwd_rule)

    # ---- phases 3 + 6 + 7: chunk scan -------------------------------------
    @jax.custom_vjp
    def chunk_scan(Q_c, K_c, V_c, Z_c, state_in, A_cumsum, dq_c):
        return _chunk_scan_fwd(Q_c, K_c, V_c, Z_c, state_in, A_cumsum, dq_c)

    def _chunk_scan_fwd(Q_c, K_c, V_c, Z_c, state_in, A_cumsum, dq_c):
        nc, CS, nh, N = Q_c.shape
        P = V_c.shape[-1]
        f = _call(
            functools.partial(_chunk_scan_fwd_kernel, **kw),
            grid=(nc, nh),
            in_specs=[_spec_lcn(CS, N), _spec_lcn(CS, N), _spec_lcn(CS, P), _spec_lcn(CS, P),
                      _spec_state(P, N), _spec_hcl(CS), _spec_clh(CS)],
            out_specs=_spec_lcn(CS, P),
            out_shape=jax.ShapeDtypeStruct(V_c.shape, V_c.dtype),
            interpret=interpret, cfg=cfg)
        return f(Q_c, K_c, V_c, Z_c, state_in.astype(jnp.float32), A_cumsum, dq_c)

    def chunk_scan_fwd_rule(Q_c, K_c, V_c, Z_c, state_in, A_cumsum, dq_c):
        y = _chunk_scan_fwd(Q_c, K_c, V_c, Z_c, state_in, A_cumsum, dq_c)
        return y, (Q_c, K_c, V_c, Z_c, state_in, A_cumsum, dq_c)

    def chunk_scan_bwd_rule(res, dY):
        # Three kernels instead of one. The single kernel holding every tile of
        # the backward ran 2.2 ms per layer-step at 32 registers per thread and
        # 6% occupancy (job 6422282); the three parts together run 1.0 ms
        # (job 6425384: 0.44 + 0.25 + 0.28). Nsight Compute shows they still
        # compile to 32 registers with local-memory traffic, so the gain is
        # three short independent dependency chains and three times the
        # programs in flight, not fewer spills. Each part recomputes the few
        # products it needs; the extra reads of the inputs are small.
        Q_c, K_c, V_c, Z_c, state_in, A_cumsum, dq_c = res
        nc, CS, nh, N = Q_c.shape
        P = V_c.shape[-1]
        sin_f32 = state_in.astype(jnp.float32)
        dY = dY.astype(V_c.dtype)
        gate = _call(
            functools.partial(_chunk_scan_bwd_gate_kernel, **kw),
            grid=(nc, nh),
            in_specs=[_spec_lcn(CS, N), _spec_lcn(CS, N), _spec_lcn(CS, P), _spec_lcn(CS, P),
                      _spec_state(P, N), _spec_hcl(CS), _spec_clh(CS), _spec_lcn(CS, P)],
            out_specs=[_spec_lcn(CS, P), _spec_clh(CS), _spec_state(P, N), _spec_hcl(CS)],
            out_shape=[jax.ShapeDtypeStruct(Z_c.shape, Z_c.dtype),
                       jax.ShapeDtypeStruct(dq_c.shape, dq_c.dtype),
                       jax.ShapeDtypeStruct(state_in.shape, jnp.float32),
                       jax.ShapeDtypeStruct(A_cumsum.shape, A_cumsum.dtype)],
            interpret=interpret, cfg=cfg)
        value = _call(
            functools.partial(_chunk_scan_bwd_value_kernel, **kw),
            grid=(nc, nh),
            in_specs=[_spec_lcn(CS, N), _spec_lcn(CS, N), _spec_lcn(CS, P), _spec_lcn(CS, P),
                      _spec_hcl(CS), _spec_clh(CS), _spec_lcn(CS, P)],
            out_specs=[_spec_lcn(CS, P), _spec_hcl(CS)],
            out_shape=[jax.ShapeDtypeStruct(V_c.shape, V_c.dtype),
                       jax.ShapeDtypeStruct(A_cumsum.shape, A_cumsum.dtype)],
            interpret=interpret, cfg=cfg)
        qk = _call(
            functools.partial(_chunk_scan_bwd_qk_kernel, **kw),
            grid=(nc, nh),
            in_specs=[_spec_lcn(CS, N), _spec_lcn(CS, N), _spec_lcn(CS, P), _spec_lcn(CS, P),
                      _spec_state(P, N), _spec_hcl(CS), _spec_lcn(CS, P)],
            out_specs=[_spec_lcn(CS, N), _spec_lcn(CS, N)],
            out_shape=[jax.ShapeDtypeStruct(Q_c.shape, Q_c.dtype),
                       jax.ShapeDtypeStruct(K_c.shape, K_c.dtype)],
            interpret=interpret, cfg=cfg)
        dZ, ddq, dsin, dcum_a = gate(Q_c, K_c, V_c, Z_c, sin_f32, A_cumsum, dq_c, dY)
        dV, dcum_b = value(Q_c, K_c, V_c, Z_c, A_cumsum, dq_c, dY)
        dQ, dK = qk(Q_c, K_c, V_c, Z_c, sin_f32, A_cumsum, dY)
        return dQ, dK, dV, dZ, dsin.astype(state_in.dtype), dcum_a + dcum_b, ddq

    chunk_scan.defvjp(chunk_scan_fwd_rule, chunk_scan_bwd_rule)

    return phase1, chunk_state, chunk_scan


# ---------------------------------------------------------------------------
# Phase 5 in JAX: identical to the reference, so autodiff composes.
# ---------------------------------------------------------------------------

def state_passing_jax(states, A_end, precision):
    """State entering each chunk from the per-chunk states.

    states: (nc, nh, P, N) contribution of each chunk; A_end: (nh, nc) total
    log-decay of each chunk. Returns (nc, nh, P, N) with row 0 all zeros.
    Same dense segsum form as the reference (mamba3_jax._phases_4_5_6_jax).
    """
    nc, nh, P, N = states.shape
    initial = jnp.zeros((1, nh, P, N), dtype=states.dtype)
    st = jnp.concatenate([initial, states], axis=0)              # (nc+1, ...)
    A_end_padded = jnp.pad(A_end, ((0, 0), (1, 0)))              # (nh, nc+1)
    decay_chunk = jnp.exp(segsum(A_end_padded))                  # (nh, nc+1, nc+1)
    new_states = jnp.einsum('hzc,chpn->zhpn', decay_chunk, st, precision=precision)
    return new_states[:-1]


def state_passing_blocked(states, A_end, precision, block=16):
    """Same result as state_passing_jax, as a two-level scan.

    The dense form contracts an (nc+1) x (nc+1) decay matrix against every
    chunk state: O(nc^2) work, 22 GFLOP per layer at nc=204, for what is a
    linear recurrence. Here chunks are grouped in blocks of `block`: a dense
    block x block contraction inside each block, one small dense contraction
    over the nb block totals for the carry entering each block, and one
    broadcast-multiply-add to combine. O(nc x block) work, every exponent a
    sum of negative log-decays, so it is as stable as the dense form.

    states: (nc, nh, P, N); A_end: (nh, nc) total log-decay per chunk.
    Returns (nc, nh, P, N): the state entering each chunk (row 0 is zero).
    """
    nc, nh, P, N = states.shape
    blk = min(block, nc)
    nb = -(-nc // blk)
    pad = nb * blk - nc
    if pad:
        # Padding chunks come after the real ones with zero state and zero
        # decay; they receive carries but contribute nothing, and are trimmed.
        states = jnp.pad(states, ((0, pad), (0, 0), (0, 0), (0, 0)))
        A_end = jnp.pad(A_end, ((0, 0), (0, pad)))
    st = states.reshape(nb, blk, nh, P, N)                       # (nb, blk, nh, P, N)
    a = A_end.reshape(nh, nb, blk)                               # (nh, nb, blk)
    cum_incl = jnp.cumsum(a, axis=-1)                            # decay from block start to end of chunk j
    cum_excl = cum_incl - a                                      # decay from block start to start of chunk z
    tot = cum_incl[:, :, -1]                                     # (nh, nb) whole-block decay

    # Inside a block: state entering z from chunks j < z of the same block.
    diff = cum_excl[:, :, :, None] - cum_incl[:, :, None, :]     # (nh, nb, z, j): exponent for j < z
    strict = jnp.tril(jnp.ones((blk, blk), dtype=bool), k=-1)
    decay_local = jnp.where(strict[None, None], jnp.exp(jnp.where(strict[None, None], diff, 0.0)), 0.0)
    local_in = jnp.einsum('hbzj,bjhpn->bzhpn', decay_local, st, precision=precision)

    # Block outputs: state at the end of each block from its own chunks.
    decay_out = jnp.exp(tot[:, :, None] - cum_incl)             # (nh, nb, j)
    block_out = jnp.einsum('hbj,bjhpn->bhpn', decay_out, st, precision=precision)

    # Carry entering each block: the dense form over nb blocks (nb is small).
    zero = jnp.zeros((1, nh, P, N), dtype=states.dtype)
    bo = jnp.concatenate([zero, block_out], axis=0)              # (nb+1, nh, P, N)
    tot_pad = jnp.pad(tot, ((0, 0), (1, 0)))                     # (nh, nb+1)
    decay_blocks = jnp.exp(segsum(tot_pad))                      # (nh, nb+1, nb+1)
    carry = jnp.einsum('hbc,chpn->bhpn', decay_blocks, bo, precision=precision)[:-1]  # (nb, nh, P, N)

    # Combine: carry decays from block start to the start of chunk z.
    state_in = local_in + jnp.exp(cum_excl).transpose(1, 2, 0)[:, :, :, None, None] * carry[:, None]
    return state_in.reshape(nb * blk, nh, P, N)[:nc]


def state_passing_mode() -> str:
    """blocked by default: same numbers to 1e-5, 0.68 vs 1.08 ms per layer
    forward at nc=204 (job 6394522), and a 6% step gain under HIGHEST."""
    mode = os.environ.get("MAMBA3_STATE_PASSING") or "blocked"
    if mode not in {"dense", "blocked"}:
        raise ValueError("MAMBA3_STATE_PASSING must be dense or blocked")
    return mode


# ---------------------------------------------------------------------------
# Public entry point: the reference's semantics with B and C shared over heads.
# ---------------------------------------------------------------------------

def prepare_fused_inputs(Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z,
                         chunk_size, io_dtype):
    """The JAX side of the fused path: cumulative rotation angles, trapezoidal
    weights, padding to whole chunks and the chunked layouts. Everything here
    is a few megabytes per layer; the per-head tensors of phase 1 are never
    formed by XLA.

    Q, K: (L, N) shared over heads; angles: (L, R), (L, 1, R) or (L, nh, R);
    the rest as in mamba3_ssd_chunked_jax. Returns a dict of kernel inputs
    plus L and L_pad.
    """
    f32 = jnp.float32
    if Q.ndim != 2 or K.ndim != 2:
        raise ValueError("the fused path takes Q and K as (L, d_state), shared over heads")
    L, N = Q.shape
    nh, P = V.shape[1], V.shape[2]
    CS = int(chunk_size)
    if angles.ndim == 2:
        angles = angles[:, None, :]
    R = angles.shape[-1]
    if not pallas_tiles_supported(CS, N, P, R):
        raise ValueError(
            f"tiles chunk_size={CS}, d_state={N}, headdim={P}, num_rope_angles={R} are not "
            f"powers of two >= {_MIN_TILE}; use the XLA reference for this shape")

    angles_cumsum = angle_dt_cumsum(angles.astype(f32), DT)     # (L, nh, R), broadcasts over heads
    gamma, scale = trapezoid_gamma_scale(DT, Trap)               # (nh, L)

    pad_len = (CS - L % CS) % CS
    L_pad = L + pad_len
    nc = L_pad // CS

    def pad_rows(x):
        return jnp.pad(x, [(0, pad_len)] + [(0, 0)] * (x.ndim - 1)) if pad_len else x

    def pad_cols(x):
        return jnp.pad(x, ((0, 0), (0, pad_len))) if pad_len else x

    io_dtype = jnp.dtype(io_dtype)
    prep = dict(
        Cp=pad_rows(Q.astype(f32)),                              # (L_pad, N)
        Bp=pad_rows(K.astype(f32)),
        qbias=Q_bias.astype(f32),                                # (nh, N)
        kbias=K_bias.astype(f32),
        ang=pad_rows(angles_cumsum),                             # (L_pad, nh, R)
        gamma_c=pad_cols(gamma).T.reshape(nc, CS, nh),           # (nc, CS, nh); 0 in the padding
        scale=pad_cols(scale),                                   # (nh, L_pad)
        D=D.astype(f32),
        V_c=pad_rows(V).reshape(nc, CS, nh, P).astype(io_dtype),
        Z_c=pad_rows(Z).reshape(nc, CS, nh, P).astype(io_dtype),
        A_cumsum=jnp.cumsum(pad_cols(ADT).reshape(nh, nc, CS), axis=-1),   # (nh, nc, CS)
    )
    return prep, L, L_pad


def run_fused_kernels(prep, L, L_pad, *, kernels, precision, chunk_size, io_dtype):
    """phase1 -> chunk_state -> state passing (JAX) -> chunk_scan; returns (L, nh, P)."""
    phase1, chunk_state, chunk_scan = kernels
    p = prep
    nh, P = p["V_c"].shape[2], p["V_c"].shape[3]
    Q_c, K_c, qk_c = phase1(p["Cp"], p["Bp"], p["qbias"], p["kbias"], p["ang"], p["scale"],
                            int(chunk_size), jnp.dtype(io_dtype).name)
    states = chunk_state(K_c, p["V_c"], p["A_cumsum"])                       # (nc, nh, P, N)
    passing = state_passing_blocked if state_passing_mode() == "blocked" else state_passing_jax
    state_in = passing(states, p["A_cumsum"][:, :, -1], precision)
    dq_c = p["D"][None, None, :] + qk_c * p["gamma_c"]                       # (nc, CS, nh)
    Y = chunk_scan(Q_c, K_c, p["V_c"], p["Z_c"], state_in, p["A_cumsum"], dq_c)  # (nc, CS, nh, P)
    return Y.reshape(L_pad, nh, P)[:L]


def mamba3_ssd_chunked_pallas(
    Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z,
    chunk_size=64,
    *,
    compute_dtype=None,
    io_dtype=jnp.float32,
    layer_dtype=jnp.float32,
    interpret=None,
    num_warps=4,
    num_stages=2,
):
    """Mamba3 SISO chunked forward with fused Pallas kernels.

    Same result as mamba3_jax.mamba3_ssd_chunked_jax called with Q, K and
    angles repeated over heads. Here Q, K are (L, N), shared over heads (the
    model's B and C after BCNorm), angles (L, R) or (L, 1, R) or (L, nh, R);
    V, Z (L, nh, P); ADT, DT, Trap (nh, L); Q_bias, K_bias (nh, N); D (nh,).
    Returns (L, nh, P). Differentiable through jax.grad (custom VJPs inside).

    compute_dtype: operand dtype inside the SSD kernels (default: the knob,
    then layer_dtype). io_dtype: dtype of the per-head tiles the kernels
    exchange (rotated Q, scaled K, V, Z, Y and the gradient partials); bf16
    halves that traffic. The shared Q, K, the decay math and the state
    passing stay float32 regardless.
    """
    if compute_dtype is None:
        compute_dtype = pallas_compute_dtype(layer_dtype)
    if interpret is None:
        interpret = _interpret_default()
    prec = _dot_precision()
    kernels = _build(jnp.dtype(compute_dtype).name, prec.name, bool(interpret),
                     int(num_warps), int(num_stages))
    prep, L, L_pad = prepare_fused_inputs(Q, K, V, ADT, DT, Trap, angles, Q_bias, K_bias, D, Z,
                                          chunk_size, io_dtype)
    return run_fused_kernels(prep, L, L_pad, kernels=kernels,
                             precision=mamba3_contraction_precision(),
                             chunk_size=chunk_size, io_dtype=io_dtype)


_warned_fallback = set()


def warn_fallback_once(chunk_size, d_state, headdim, num_rope_angles):
    key = (chunk_size, d_state, headdim, num_rope_angles)
    if key not in _warned_fallback:
        _warned_fallback.add(key)
        warnings.warn(
            f"MAMBA3_SSD_IMPL=pallas: tiles chunk_size={chunk_size}, d_state={d_state}, "
            f"headdim={headdim}, num_rope_angles={num_rope_angles} are not powers of two "
            f">= {_MIN_TILE} (chunk, state, head, rotation angles and the unrotated rest); "
            "this layer uses the XLA reference instead.", RuntimeWarning, stacklevel=3)
