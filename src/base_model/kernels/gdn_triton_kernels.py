"""Fused WY correction Triton kernel for GDN chunkwise path.

Fuses L construction + hierarchical block-inverse + application into a single
kernel launch, keeping the 64x64 W matrix in SRAM throughout.

Forward: Triton kernel (fast — single launch, no global memory round-trips)
Backward: Pure JAX solve_triangular (correct, tested, not on critical path)

Usage from gdn.py:
    from base_model.kernels.gdn_triton_kernels import wy_correction_fused
    v_corrected, k_cumdecay = wy_correction_fused(
        k_beta, k, v_beta, decay_mask_4d, k_with_decay, L)
"""
import jax
import jax.numpy as jnp

try:
    import triton
    import triton.language as tl
    import jax_triton as jt
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel: fused WY solve + application
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _wy_solve_apply_kernel(
        # Inputs: L_mat (nc*nh, C, C), rhs (nc*nh, C, D)
        L_ptr, rhs_ptr,
        # Output: result (nc*nh, C, D)
        out_ptr,
        # Dimensions
        stride_batch_L,   # nc*nh stride for L (= C*C)
        stride_row_L,     # row stride for L (= C)
        stride_batch_rhs, # nc*nh stride for rhs (= C*D)
        stride_row_rhs,   # row stride for rhs (= D)
        stride_batch_out, # nc*nh stride for out (= C*D)
        stride_row_out,   # row stride for out (= D)
        # Constexprs
        C: tl.constexpr,       # chunk size (64)
        D: tl.constexpr,       # rhs columns (total, e.g. 256 for hvd)
        BD: tl.constexpr,      # tile size for D dimension (64)
        BC: tl.constexpr,      # sub-block size for hierarchical inverse (16)
    ):
        """Solve (I - L) @ out = rhs and store out.

        L is strictly lower triangular (nc*nh, C, C).
        Uses hierarchical 4x16 block-inverse (FLA pattern) then tiled matmul.

        Grid: (num_d_tiles, nc * nh)
          program_id(0) = d_tile index (which BD-wide slice of D)
          program_id(1) = batch index (which chunk*head)
        """
        i_d = tl.program_id(0)   # which D-tile
        i_b = tl.program_id(1)   # which batch (chunk*head)

        # ------------------------------------------------------------------
        # Phase 1: Load L matrix (C x C) into SRAM and negate it
        # Only the first D-tile does the inverse; others reuse via reload.
        # Actually, each program needs W independently since no cross-SM sharing.
        # ------------------------------------------------------------------
        # Load L block: (C, C) — L is strictly lower triangular
        L_base = L_ptr + i_b * stride_batch_L
        offs_r = tl.arange(0, C)  # [C]
        offs_c = tl.arange(0, C)  # [C]
        # L[r, c] at L_base + r * stride_row_L + c
        L_block = tl.load(L_base + offs_r[:, None] * stride_row_L + offs_c[None, :])  # (C, C) f32

        # A = -L (strictly lower triangular, negated for (I-L)^{-1} = (I + A + A^2 + ...))
        # We compute W = (I - L)^{-1} via hierarchical block forward substitution
        # Following FLA: split 64x64 into 4 diagonal 16x16 blocks

        # ------------------------------------------------------------------
        # Phase 2: Hierarchical 4x16 block-inverse
        # ------------------------------------------------------------------
        o_16 = tl.arange(0, 16)  # BC=16

        # Extract diagonal blocks and invert each via forward substitution
        # Block (i,j) lives at L[i*16:(i+1)*16, j*16:(j+1)*16]
        # A_ii = -L[i*16:(i+1)*16, i*16:(i+1)*16] (negated, strictly lower)

        # --- Block 11 (rows 0:16, cols 0:16) ---
        b_A11 = tl.load(L_base + (offs_r[:, None] % 16 + 0) * stride_row_L +
                         (offs_c[None, :] % 16 + 0),
                         mask=(offs_r[:, None] < 16) & (offs_c[None, :] < 16), other=0.0)
        # Extract just the 16x16 sub-block
        b_A11 = tl.reshape(b_A11, [C, C])
        # Reload properly: use 16-range indices
        b_A11 = tl.load(L_base + o_16[:, None] * stride_row_L + o_16[None, :])
        m_lower = o_16[:, None] > o_16[None, :]
        m_I = o_16[:, None] == o_16[None, :]
        b_A11 = -tl.where(m_lower, b_A11, 0.0)

        for i in range(2, 16):
            b_a = -tl.load(L_base + i * stride_row_L + o_16)
            b_a = b_a + tl.sum(b_a[:, None] * b_A11, 0)
            b_A11 = tl.where((o_16 == i)[:, None], b_a, b_A11)
        b_A11 = b_A11 + m_I

        # --- Block 22 (rows 16:32, cols 16:32) ---
        b_A22 = tl.load(L_base + (o_16[:, None] + 16) * stride_row_L + (o_16[None, :] + 16))
        b_A22 = -tl.where(m_lower, b_A22, 0.0)
        for i in range(2, 16):
            b_a = -tl.load(L_base + (i + 16) * stride_row_L + o_16 + 16)
            b_a = b_a + tl.sum(b_a[:, None] * b_A22, 0)
            b_A22 = tl.where((o_16 == i)[:, None], b_a, b_A22)
        b_A22 = b_A22 + m_I

        # --- Block 33 (rows 32:48, cols 32:48) ---
        b_A33 = tl.load(L_base + (o_16[:, None] + 32) * stride_row_L + (o_16[None, :] + 32))
        b_A33 = -tl.where(m_lower, b_A33, 0.0)
        for i in range(2, 16):
            b_a = -tl.load(L_base + (i + 32) * stride_row_L + o_16 + 32)
            b_a = b_a + tl.sum(b_a[:, None] * b_A33, 0)
            b_A33 = tl.where((o_16 == i)[:, None], b_a, b_A33)
        b_A33 = b_A33 + m_I

        # --- Block 44 (rows 48:64, cols 48:64) ---
        b_A44 = tl.load(L_base + (o_16[:, None] + 48) * stride_row_L + (o_16[None, :] + 48))
        b_A44 = -tl.where(m_lower, b_A44, 0.0)
        for i in range(2, 16):
            b_a = -tl.load(L_base + (i + 48) * stride_row_L + o_16 + 48)
            b_a = b_a + tl.sum(b_a[:, None] * b_A44, 0)
            b_A44 = tl.where((o_16 == i)[:, None], b_a, b_A44)
        b_A44 = b_A44 + m_I

        # --- Off-diagonal raw blocks ---
        b_L21 = tl.load(L_base + (o_16[:, None] + 16) * stride_row_L + o_16[None, :])
        b_L31 = tl.load(L_base + (o_16[:, None] + 32) * stride_row_L + o_16[None, :])
        b_L32 = tl.load(L_base + (o_16[:, None] + 32) * stride_row_L + (o_16[None, :] + 16))
        b_L41 = tl.load(L_base + (o_16[:, None] + 48) * stride_row_L + o_16[None, :])
        b_L42 = tl.load(L_base + (o_16[:, None] + 48) * stride_row_L + (o_16[None, :] + 16))
        b_L43 = tl.load(L_base + (o_16[:, None] + 48) * stride_row_L + (o_16[None, :] + 32))

        # --- Merge: compute off-diagonal inverse blocks ---
        # W_21 = -A22_inv @ L_21 @ A11_inv
        b_W21 = -tl.dot(tl.dot(b_A22, b_L21), b_A11)
        # W_32 = -A33_inv @ L_32 @ A22_inv
        b_W32 = -tl.dot(tl.dot(b_A33, b_L32), b_A22)
        # W_43 = -A44_inv @ L_43 @ A33_inv
        b_W43 = -tl.dot(tl.dot(b_A44, b_L43), b_A33)

        # W_31 = -A33_inv @ (L_31 @ A11_inv + L_32 @ W_21)
        b_W31 = -tl.dot(b_A33, tl.dot(b_L31, b_A11) + tl.dot(b_L32, b_W21))
        # W_42 = -A44_inv @ (L_42 @ A22_inv + L_43 @ W_32)
        b_W42 = -tl.dot(b_A44, tl.dot(b_L42, b_A22) + tl.dot(b_L43, b_W32))
        # W_41 = -A44_inv @ (L_41 @ A11_inv + L_42 @ W_21 + L_43 @ W_31)
        b_W41 = -tl.dot(b_A44, tl.dot(b_L41, b_A11) + tl.dot(b_L42, b_W21) + tl.dot(b_L43, b_W31))

        # ------------------------------------------------------------------
        # Phase 3: Apply W to rhs tile — W @ rhs[:, i_d*BD:(i_d+1)*BD]
        # W is stored as 4x4 grid of 16x16 blocks (10 non-zero blocks)
        # ------------------------------------------------------------------
        rhs_base = rhs_ptr + i_b * stride_batch_rhs
        out_base = out_ptr + i_b * stride_batch_out
        d_offs = i_d * BD + tl.arange(0, BD)  # [BD]
        d_mask = d_offs < D

        # Load rhs tiles: (16, BD) each
        rhs1 = tl.load(rhs_base + o_16[:, None] * stride_row_rhs + d_offs[None, :],
                        mask=d_mask[None, :], other=0.0)
        rhs2 = tl.load(rhs_base + (o_16[:, None] + 16) * stride_row_rhs + d_offs[None, :],
                        mask=d_mask[None, :], other=0.0)
        rhs3 = tl.load(rhs_base + (o_16[:, None] + 32) * stride_row_rhs + d_offs[None, :],
                        mask=d_mask[None, :], other=0.0)
        rhs4 = tl.load(rhs_base + (o_16[:, None] + 48) * stride_row_rhs + d_offs[None, :],
                        mask=d_mask[None, :], other=0.0)

        # out1 = W11 @ rhs1  (only diagonal for row 0)
        out1 = tl.dot(b_A11, rhs1)

        # out2 = W21 @ rhs1 + W22 @ rhs2
        out2 = tl.dot(b_W21, rhs1) + tl.dot(b_A22, rhs2)

        # out3 = W31 @ rhs1 + W32 @ rhs2 + W33 @ rhs3
        out3 = tl.dot(b_W31, rhs1) + tl.dot(b_W32, rhs2) + tl.dot(b_A33, rhs3)

        # out4 = W41 @ rhs1 + W42 @ rhs2 + W43 @ rhs3 + W44 @ rhs4
        out4 = tl.dot(b_W41, rhs1) + tl.dot(b_W42, rhs2) + tl.dot(b_W43, rhs3) + tl.dot(b_A44, rhs4)

        # Store
        tl.store(out_base + o_16[:, None] * stride_row_out + d_offs[None, :],
                 out1, mask=d_mask[None, :])
        tl.store(out_base + (o_16[:, None] + 16) * stride_row_out + d_offs[None, :],
                 out2, mask=d_mask[None, :])
        tl.store(out_base + (o_16[:, None] + 32) * stride_row_out + d_offs[None, :],
                 out3, mask=d_mask[None, :])
        tl.store(out_base + (o_16[:, None] + 48) * stride_row_out + d_offs[None, :],
                 out4, mask=d_mask[None, :])


# ---------------------------------------------------------------------------
# JAX wrapper: call Triton kernel via jax_triton.triton_call
# ---------------------------------------------------------------------------
def _wy_solve_apply_triton(L_mat, rhs):
    """Solve (I - L) @ out = rhs using fused Triton kernel.

    Args:
        L_mat: (nc*nh, C, C) float32 — strictly lower triangular L matrix
        rhs:   (nc*nh, C, D) float32 — right-hand side

    Returns:
        out:   (nc*nh, C, D) float32 — solution
    """
    batch, C, D = rhs.shape
    BD = 64  # tile size for D dimension

    num_d_tiles = (D + BD - 1) // BD
    grid = (num_d_tiles, batch)

    out = jt.triton_call(
        L_mat, rhs,
        kernel=_wy_solve_apply_kernel,
        out_shape=jax.ShapeDtypeStruct(rhs.shape, jnp.float32),
        grid=grid,
        num_warps=4,
        # Strides (as scalar args, matching kernel parameter order)
        stride_batch_L=C * C,
        stride_row_L=C,
        stride_batch_rhs=C * D,
        stride_row_rhs=D,
        stride_batch_out=C * D,
        stride_row_out=D,
        # Constexprs
        C=C,
        D=D,
        BD=BD,
        BC=16,
    )
    return out


# ---------------------------------------------------------------------------
# custom_vjp wrapper: Triton forward, JAX backward
# ---------------------------------------------------------------------------
@jax.custom_vjp
def wy_correction_fused(k_beta, k, v_beta, decay_mask_4d, k_with_decay, L):
    """Fused WY correction: compute v_corrected and k_cumdecay.

    Forward uses Triton kernel for speed.
    Backward uses JAX solve_triangular for correct autodiff.

    Args:
        k_beta:        (nc, C, nh, hd) — beta-scaled keys
        k:             (nc, C, nh, hd) — keys
        v_beta:        (nc, C, nh, hvd) — beta-scaled values
        decay_mask_4d: (nc, C, C, nh) — causal decay mask
        k_with_decay:  (nc, C, nh, hd) — k_beta * exp(decay_cum)
        L:             (nc, C, C, nh) — strictly lower triangular L matrix

    Returns:
        v_corrected: (nc, C, nh, hvd)
        k_cumdecay:  (nc, C, nh, hd)
    """
    nc, C, nh, hd = k_beta.shape
    hvd = v_beta.shape[-1]

    # Reshape L to (nc*nh, C, C) for Triton
    L_mat = L.transpose(0, 3, 1, 2).reshape(nc * nh, C, C)
    I_minus_L = jnp.eye(C, dtype=jnp.float32)[None, :, :] - L_mat

    # Reshape rhs to (nc*nh, C, D)
    vb_mat = v_beta.transpose(0, 2, 1, 3).reshape(nc * nh, C, hvd)
    kwd_mat = k_with_decay.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)

    # Triton solve
    v_corr = _wy_solve_apply_triton(I_minus_L, vb_mat)
    k_cum = _wy_solve_apply_triton(I_minus_L, kwd_mat)

    # Reshape back
    v_corrected = v_corr.reshape(nc, nh, C, hvd).transpose(0, 2, 1, 3)
    k_cumdecay = k_cum.reshape(nc, nh, C, hd).transpose(0, 2, 1, 3)

    return v_corrected, k_cumdecay


def _wy_fwd(k_beta, k, v_beta, decay_mask_4d, k_with_decay, L):
    result = wy_correction_fused(k_beta, k, v_beta, decay_mask_4d, k_with_decay, L)
    # Save inputs for backward (JAX solve_triangular path)
    return result, (k_beta, k, v_beta, decay_mask_4d, k_with_decay, L)


def _wy_bwd(residuals, g):
    """Backward pass: use JAX solve_triangular for correct gradients.

    We re-derive (I-L)^{-1} in pure JAX and let autodiff handle the rest.
    This is not on the critical path (backward is already ~1x forward cost
    with or without Triton, dominated by the scan backward).
    """
    k_beta, k, v_beta, decay_mask_4d, k_with_decay, L = residuals
    nc, C, nh, hd = k_beta.shape
    hvd = v_beta.shape[-1]

    # Reconstruct I - L
    L_mat = L.transpose(0, 3, 1, 2).reshape(nc * nh, C, C)
    I_minus_L = jnp.eye(C, dtype=jnp.float32)[None, :, :] - L_mat

    # v_corrected via solve_triangular (differentiable)
    vb_mat = v_beta.transpose(0, 2, 1, 3).reshape(nc * nh, C, hvd)
    v_corr = jax.scipy.linalg.solve_triangular(I_minus_L, vb_mat, lower=True)
    v_corrected = v_corr.reshape(nc, nh, C, hvd).transpose(0, 2, 1, 3)

    # k_cumdecay via solve_triangular (differentiable)
    kwd_mat = k_with_decay.transpose(0, 2, 1, 3).reshape(nc * nh, C, hd)
    k_cum = jax.scipy.linalg.solve_triangular(I_minus_L, kwd_mat, lower=True)
    k_cumdecay = k_cum.reshape(nc, nh, C, hd).transpose(0, 2, 1, 3)

    # Now compute VJP through the JAX ops
    g_v, g_k = g
    primals = (k_beta, k, v_beta, decay_mask_4d, k_with_decay, L)

    # Use JAX's autodiff through solve_triangular
    def _jax_fwd(k_beta_, k_, v_beta_, decay_mask_4d_, k_with_decay_, L_):
        nc_, C_, nh_, hd_ = k_beta_.shape
        hvd_ = v_beta_.shape[-1]
        L_mat_ = L_.transpose(0, 3, 1, 2).reshape(nc_ * nh_, C_, C_)
        I_minus_L_ = jnp.eye(C_, dtype=jnp.float32)[None, :, :] - L_mat_
        vb_ = v_beta_.transpose(0, 2, 1, 3).reshape(nc_ * nh_, C_, hvd_)
        v_c = jax.scipy.linalg.solve_triangular(I_minus_L_, vb_, lower=True)
        v_c = v_c.reshape(nc_, nh_, C_, hvd_).transpose(0, 2, 1, 3)
        kwd_ = k_with_decay_.transpose(0, 2, 1, 3).reshape(nc_ * nh_, C_, hd_)
        k_c = jax.scipy.linalg.solve_triangular(I_minus_L_, kwd_, lower=True)
        k_c = k_c.reshape(nc_, nh_, C_, hd_).transpose(0, 2, 1, 3)
        return v_c, k_c

    _, vjp_fn = jax.vjp(_jax_fwd, *primals)
    return vjp_fn((g_v, g_k))


wy_correction_fused.defvjp(_wy_fwd, _wy_bwd)
