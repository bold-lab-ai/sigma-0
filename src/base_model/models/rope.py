"""Rotary Position Embedding (RoPE) for JAX.

Standalone implementation compatible with bf16 compute dtype.
Applied to Q, K after projection (not to input embeddings like sinusoidal PE).

Core idea: encode position by rotating pairs of dimensions in Q/K by
position-dependent angles. This makes attention logits depend only on
relative position (q_m · k_n) = f(x_m, x_n, m-n), enabling length
generalization and efficient KV caching.

Formula:
    theta_i = pos / base^(2i/dim)   for dimension pair i
    q_rotated = q * cos(theta) + rotate_half(q) * sin(theta)

References:
    - RoFormer (Su et al., 2021): https://arxiv.org/abs/2104.09864
    - NTK-aware scaling (Reddit/bloc97): interpolate base frequency for
      context extension without fine-tuning
"""

import jax.numpy as jnp


def rotary_embedding(dim: int, max_seq_len: int,
                     base: float = 10000.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute sin/cos tables for RoPE.

    Args:
        dim: head dimension (must be even — pairs of dims are rotated)
        max_seq_len: maximum sequence length to precompute for
        base: frequency base (10000 standard, increase for NTK scaling)

    Returns:
        (cos_table, sin_table): each (max_seq_len, dim), float32
    """
    assert dim % 2 == 0, f"RoPE requires even dim, got {dim}"
    half_dim = dim // 2

    # Inverse frequencies: theta_i = 1 / base^(2i/dim) for i in [0, dim/2)
    freq_seq = jnp.arange(half_dim, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))   # (dim/2,)

    # Outer product: positions x frequencies -> angles
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)  # (max_seq_len,)
    angles = jnp.outer(positions, inv_freq)                  # (max_seq_len, dim/2)

    # Duplicate to full dim: [theta_0, theta_1, ..., theta_0, theta_1, ...]
    # so cos/sin tables are (max_seq_len, dim) matching the split-half layout
    cos_table = jnp.cos(angles)  # (max_seq_len, dim/2)
    sin_table = jnp.sin(angles)  # (max_seq_len, dim/2)

    return cos_table, sin_table


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Split x into two halves and swap with sign flip.

    x = [x1, x2] -> [-x2, x1]

    This implements the rotation matrix multiplication efficiently:
        R(theta) @ x = x * cos(theta) + rotate_half(x) * sin(theta)
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(q: jnp.ndarray, k: jnp.ndarray,
               cos: jnp.ndarray, sin: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE to Q and K using precomputed cos/sin tables.

    Args:
        q: (..., L, n_heads, head_dim) or (L, n_heads, head_dim)
        k: same shape as q
        cos: (L, head_dim/2) — sliced from cos_table at the right positions
        sin: (L, head_dim/2) — sliced from sin_table at the right positions

    Returns:
        (q_rotated, k_rotated): same shapes as inputs, same dtype
    """
    orig_dtype = q.dtype

    # Upcast for precision (RoPE rotation is sensitive to FP error)
    q_f = q.astype(jnp.float32)
    k_f = k.astype(jnp.float32)

    # cos/sin are (L, dim/2) — broadcast to (..., L, 1, dim/2) for multi-head
    cos_expanded = cos[:, None, :]   # (L, 1, dim/2)
    sin_expanded = sin[:, None, :]   # (L, 1, dim/2)

    # Split q, k into halves
    q1, q2 = q_f[..., :q_f.shape[-1] // 2], q_f[..., q_f.shape[-1] // 2:]
    k1, k2 = k_f[..., :k_f.shape[-1] // 2], k_f[..., k_f.shape[-1] // 2:]

    # Apply rotation: [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    q_rot = jnp.concatenate([
        q1 * cos_expanded - q2 * sin_expanded,
        q2 * cos_expanded + q1 * sin_expanded,
    ], axis=-1)

    k_rot = jnp.concatenate([
        k1 * cos_expanded - k2 * sin_expanded,
        k2 * cos_expanded + k1 * sin_expanded,
    ], axis=-1)

    return q_rot.astype(orig_dtype), k_rot.astype(orig_dtype)


def apply_rope_at_positions(q: jnp.ndarray, k: jnp.ndarray,
                            positions: jnp.ndarray,
                            cos_table: jnp.ndarray,
                            sin_table: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE at arbitrary integer positions (for KV cache inference).

    Args:
        q: (L, n_heads, head_dim)
        k: (L, n_heads, head_dim)
        positions: (L,) int32 — absolute position of each token
        cos_table: (max_seq_len, head_dim/2) — precomputed
        sin_table: (max_seq_len, head_dim/2) — precomputed

    Returns:
        (q_rotated, k_rotated)
    """
    cos = cos_table[positions]  # (L, dim/2)
    sin = sin_table[positions]  # (L, dim/2)
    return apply_rope(q, k, cos, sin)


def rotary_embedding_ntk(dim: int, max_seq_len: int,
                         base: float = 10000.0,
                         scale_factor: float = 1.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """NTK-aware RoPE scaling for context extension.

    Scales the base frequency so that the same position range covers a longer
    context without fine-tuning. When scale_factor=1.0, identical to standard RoPE.

    NTK scaling adjusts the base: base_scaled = base * scale_factor^(dim / (dim-2))

    This distributes the scaling across all frequency bands (unlike linear
    interpolation which compresses all frequencies equally), preserving
    high-frequency local patterns while extending low-frequency global range.

    Args:
        dim: head dimension
        max_seq_len: maximum positions to precompute
        base: original frequency base
        scale_factor: context extension ratio (e.g., 2.0 for 2x context)

    Returns:
        (cos_table, sin_table): each (max_seq_len, dim/2)
    """
    if scale_factor == 1.0:
        return rotary_embedding(dim, max_seq_len, base)

    # NTK-aware base scaling
    base_scaled = base * (scale_factor ** (dim / (dim - 2)))
    return rotary_embedding(dim, max_seq_len, base=base_scaled)
