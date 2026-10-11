"""NSA (Native Sparse Attention) block as drop-in replacement for S5SSM.

Three-branch attention:
  1. Compressed: Mean-pooled K,V → global coarse view of full history
  2. Selected: Top-k blocks from compressed importance → fine sparse view
  3. Sliding Window: Local causal window → fine local context

Combined via learned sigmoid gates: o = g_cmp·o_cmp + g_slc·o_slc + g_swa·o_swa

Reference: the AlphaTrade/LOBS5 tree, refs/nsa/native_sparse_attention/ops/naive.py

LOB-specific design:
  - block_size=26 (1 LOB message = 26 tokens)
  - block_counts=16 (S=16 selected blocks = 416 tokens fine context)
  - window_size=512 (~21 messages local context)
  - MHA (not GQA)
  - PE: RoPE (default). DEPRECATED sinusoidal kept for old checkpoint compat only.
"""
from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from base_model.models.transformer import sinusoidal_positional_encoding, positional_encoding_at_positions  # DEPRECATED: sinusoidal PE kept only for old checkpoint backward compat
from base_model.models.rope import rotary_embedding, apply_rope, apply_rope_at_positions, rotary_embedding_ntk


def flash_causal_attention(q, k, v):
    """Causal attention via cuDNN flash kernel — O(L) memory per head.

    cuDNN requires bf16/fp16 — inputs are cast automatically and output
    is cast back to the original dtype.

    Falls back to XLA when head_dim is incompatible with cuDNN
    (must be <= 256 and multiple of 8, e.g. book encoder with head_dim=503).

    Args:
        q, k, v: (L, n_heads, head_dim) — scale applied internally (1/sqrt(d))

    Returns:
        output: (L, n_heads, head_dim)
    """
    head_dim = q.shape[-1]
    orig_dtype = q.dtype
    if head_dim <= 256 and head_dim % 8 == 0:
        # cuDNN flash attention — O(L) memory, bf16 internal precision.
        # NOTE: cuDNN uses online softmax which differs numerically from
        # explicit softmax used in inference. This causes ~0.5 KL divergence
        # after 100 messages of cached inference. Acceptable trade-off for
        # 2x faster training. For future work, consider XLA implementation
        # to eliminate this divergence.
        q_f = q.astype(jnp.bfloat16)
        k_f = k.astype(jnp.bfloat16)
        v_f = v.astype(jnp.bfloat16)
        out = jax.nn.dot_product_attention(
            q_f, k_f, v_f, is_causal=True, implementation='cudnn')
        return out.astype(orig_dtype)
    else:
        return jax.nn.dot_product_attention(
            q, k, v, is_causal=True, implementation='xla')


def sliding_window_attention(q, k, v, window_size, scale, block_q=26):
    """O(L·W) memory sliding window causal attention via scan over query blocks.

    Each scan step processes block_q queries against a window of W + block_q - 1
    keys. Memory per step: O(block_q × W × n_heads) — never materializes L×L.

    Args:
        q: (L, n_heads, head_dim) query tensor
        k: (L, n_heads, head_dim) key tensor
        v: (L, n_heads, head_dim) value tensor
        window_size: int, sliding window size W
        scale: float, attention scale (1/sqrt(head_dim))
        block_q: int, queries per scan step (must divide L)

    Returns:
        output: (L, n_heads, head_dim)
    """
    L, n_heads, head_dim = q.shape
    W = window_size
    n_blocks = L // block_q

    # Pad K, V with W-1 zeros at the start for uniform windowing
    k_pad = jnp.pad(k, ((W - 1, 0), (0, 0), (0, 0)))
    v_pad = jnp.pad(v, ((W - 1, 0), (0, 0), (0, 0)))

    kv_win_size = W + block_q - 1

    def process_block(carry, block_idx):
        q_start = block_idx * block_q

        q_chunk = jax.lax.dynamic_slice(
            q, (q_start, 0, 0), (block_q, n_heads, head_dim))
        k_chunk = jax.lax.dynamic_slice(
            k_pad, (q_start, 0, 0), (kv_win_size, n_heads, head_dim))
        v_chunk = jax.lax.dynamic_slice(
            v_pad, (q_start, 0, 0), (kv_win_size, n_heads, head_dim))

        # Attention scores: (block_q, n_heads, kv_win_size)
        scores = jnp.einsum('qhd,khd->qhk', q_chunk * scale, k_chunk)

        # Causal + validity + window mask
        # k position j in padded array → original position (q_start + j) - (W - 1)
        q_pos = q_start + jnp.arange(block_q)
        k_orig_pos = q_start + jnp.arange(kv_win_size) - (W - 1)

        causal = k_orig_pos[None, :] <= q_pos[:, None]       # key ≤ query
        valid = k_orig_pos[None, :] >= 0                       # key ≥ 0
        in_window = (q_pos[:, None] - k_orig_pos[None, :]) < W  # within window
        mask = causal & valid & in_window

        scores = jnp.where(
            mask[:, None, :], scores,
            jnp.finfo(jnp.float32).min)
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        attn = attn.astype(q.dtype)

        out = jnp.einsum('qhk,khd->qhd', attn, v_chunk)
        return carry, out

    _, outputs = jax.lax.scan(process_block, None, jnp.arange(n_blocks))
    return outputs.reshape(L, n_heads, head_dim)


def flash_causal_attention_cached(q, k, v, mask, pos):
    """Causal attention matching training's cuDNN kernel, for KV cache inference.

    cuDNN requires Q and K to have compatible seq lengths for bias.
    Strategy: pad Q with zeros to length L_kv, run full causal attention,
    extract outputs at positions pos..pos+L_q-1.

    Args:
        q: (L_q, n_heads, head_dim) — current query tokens
        k: (L_kv, n_heads, head_dim) — full KV cache
        v: (L_kv, n_heads, head_dim) — full KV cache
        mask: (L_q, L_kv) bool — True where attention is allowed
        pos: int — starting position of Q in the sequence

    Returns:
        output: (L_q, n_heads, head_dim)
    """
    L_q, n_heads, head_dim = q.shape
    L_kv = k.shape[0]
    orig_dtype = q.dtype

    if head_dim <= 256 and head_dim % 8 == 0:
        # Pad Q to L_kv so cuDNN sees matching seq lengths.
        # is_causal=True generates the correct lower-triangular mask:
        # Q at position pos+i attends to K at positions 0..pos+i.
        # Padded Q positions (0..pos-1) produce garbage but we discard them.
        q_padded = jnp.zeros((L_kv, n_heads, head_dim), dtype=jnp.bfloat16)
        q_padded = jax.lax.dynamic_update_slice(
            q_padded, q.astype(jnp.bfloat16), (pos, 0, 0))

        k_f = k.astype(jnp.bfloat16)
        v_f = v.astype(jnp.bfloat16)

        out = jax.nn.dot_product_attention(
            q_padded, k_f, v_f, is_causal=True, implementation='cudnn')

        # Extract outputs at the queried positions only
        out_slice = jax.lax.dynamic_slice(
            out, (pos, 0, 0), (L_q, n_heads, head_dim))
        return out_slice.astype(orig_dtype)
    else:
        # XLA fallback: explicit mask-based attention
        scores = jnp.einsum('qhd,khd->qhk', q / jnp.sqrt(head_dim).astype(q.dtype), k)
        scores = jnp.where(mask[:, None, :], scores, jnp.finfo(jnp.float32).min)
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
        return jnp.einsum('qhk,khd->qhd', attn, v)


class NSABlock(nn.Module):
    """Three-branch Native Sparse Attention block.

    Drop-in replacement for S5SSM / TransformerBlock.
    Same interface: __call__(input_sequence: (L, H)) -> (L, H).

    Position encoding:
        pe_type='rope': RoPE applied to Q, K after projection (RECOMMENDED)
        pe_type='sinusoidal': DEPRECATED — additive sinusoidal PE on input (old checkpoint compat only)
    """
    H: int              # d_model
    n_heads: int         # number of attention heads
    d_ff: int            # FFN intermediate dim (default 4*H)
    block_size: int = 26      # tokens per compressed block (1 LOB message)
    block_counts: int = 16    # S: number of selected blocks for Branch 2
    window_size: int = 512    # sliding window size for Branch 3
    max_cache_len: int = 26000
    pe_type: str = 'rope'   # 'rope' (default) or 'sinusoidal' (DEPRECATED, old checkpoint compat only)
    rope_base: float = 10000.0    # RoPE frequency base
    dropout: float = 0.0
    training: bool = True
    dtype: jnp.dtype = jnp.float32
    step_rescale: float = 1.0  # ignored — S5 compat

    def setup(self):
        # Adjust n_heads if H is not divisible (e.g., book encoder with H=d_book=503)
        nh = self.n_heads
        while nh > 1 and self.H % nh != 0:
            nh -= 1
        self._nh = nh
        self._hd = self.H // nh

        # RoPE requires even head_dim for rotation pairs. If head_dim is odd
        # (e.g., book encoder H=503 → hd=503), fall back to sinusoidal PE.
        # DEPRECATED: sinusoidal fallback only for book encoder with odd head_dim
        if self.pe_type == 'rope' and self._hd % 2 != 0:
            self._effective_pe = 'sinusoidal'  # DEPRECATED: sinusoidal fallback for odd head_dim
        else:
            self._effective_pe = self.pe_type

        # QKV + output projections
        self.q_proj = nn.Dense(self.H, dtype=self.dtype, use_bias=False)
        self.k_proj = nn.Dense(self.H, dtype=self.dtype, use_bias=False)
        self.v_proj = nn.Dense(self.H, dtype=self.dtype, use_bias=False)
        self.o_proj = nn.Dense(self.H, dtype=self.dtype, use_bias=False)

        # Gate projection: 3 sigmoid gates per head (compressed, selected, swa)
        self.g_proj = nn.Dense(self._nh * 3, dtype=self.dtype, use_bias=False)

        # Pre-LN
        self.norm1 = nn.LayerNorm(dtype=self.dtype)
        self.norm2 = nn.LayerNorm(dtype=self.dtype)

        # FFN
        self.ff = nn.Sequential([
            nn.Dense(self.d_ff, dtype=self.dtype),
            nn.gelu,
            nn.Dense(self.H, dtype=self.dtype),
        ])

        self.drop = nn.Dropout(self.dropout, deterministic=not self.training)

    # ------------------------------------------------------------------
    # Branch 1: Compressed Attention (global coarse)
    # ------------------------------------------------------------------
    def _compressed_attention(self, q, k, v, scale, S):
        """Mean-pool K,V into block summaries, full causal attention Q vs summaries.

        Also extracts block importance scores for Branch 2's top-k selection.

        Returns:
            o_cmp: (L, n_heads, head_dim) compressed attention output
            top_indices: (n_msgs, S) selected block indices per message
        """
        L, n_heads, hd = q.shape
        BS = self.block_size
        n_blocks = L // BS

        # Mean-pool K, V per block → (n_blocks, n_heads, hd)
        k_cmp = k.reshape(n_blocks, BS, n_heads, hd).mean(axis=1)
        v_cmp = v.reshape(n_blocks, BS, n_heads, hd).mean(axis=1)

        # Attention scores: (L, n_heads, n_blocks)
        scores = jnp.einsum('lhd,nhd->lhn', q * scale, k_cmp)

        # Causal mask: token t sees block c iff c <= floor((t - BS + 1) / BS)
        # (block c is complete once all BS tokens are past)
        t_idx = jnp.arange(L)
        c_idx = jnp.arange(n_blocks)
        num_visible = jnp.maximum(0, (t_idx - BS + 1) // BS + 1)  # (L,)
        causal_mask = c_idx[None, :] < num_visible[:, None]        # (L, n_blocks)

        scores = jnp.where(
            causal_mask[:, None, :], scores,
            jnp.finfo(jnp.float32).min)

        has_visible = num_visible > 0  # (L,)
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        attn = jnp.where(has_visible[:, None, None], attn, 0.0)
        attn = attn.astype(q.dtype)

        # Compressed output
        o_cmp = jnp.einsum('lhn,nhd->lhd', attn, v_cmp)

        # --- Block importance for top-k selection ---
        # Sum attention across heads → per-token importance per block
        importance = attn.astype(jnp.float32).sum(axis=1)  # (L, n_blocks)

        # Force current block to importance 1.0 (always selected)
        current_block = t_idx // BS
        local_mask = c_idx[None, :] == current_block[:, None]
        importance = jnp.where(local_mask, 1.0, importance)

        # Mask future/invalid blocks
        importance = jnp.where(causal_mask, importance, -1e9)

        # Per-message: average importance across tokens in each message
        importance_msgs = importance.reshape(n_blocks, BS, n_blocks).mean(axis=1)

        # Top-S per message
        _, top_indices = jax.lax.top_k(importance_msgs, S)  # (n_msgs, S)

        return o_cmp, top_indices

    # ------------------------------------------------------------------
    # Branch 2: Selected Attention (fine sparse)
    # ------------------------------------------------------------------
    def _selected_attention(self, q, k, v, top_indices, scale, S):
        """Gather K,V from top-S blocks, per-message causal attention.

        Args:
            q: (L, n_heads, hd)
            k: (L, n_heads, hd)
            v: (L, n_heads, hd)
            top_indices: (n_msgs, S) block indices selected per message
            scale: float
            S: int, number of selected blocks

        Returns:
            o_slc: (L, n_heads, hd)
        """
        L, n_heads, hd = q.shape
        BS = self.block_size
        n_msgs = L // BS

        # Reshape into messages
        q_msgs = q.reshape(n_msgs, BS, n_heads, hd)
        k_blocks = k.reshape(n_msgs, BS, n_heads, hd)
        v_blocks = v.reshape(n_msgs, BS, n_heads, hd)

        # Clamp indices to valid range
        safe_indices = jnp.clip(top_indices, 0, n_msgs - 1)  # (n_msgs, S)

        # Gather K, V from selected blocks → (n_msgs, S, BS, n_heads, hd)
        k_sel = k_blocks[safe_indices].reshape(n_msgs, S * BS, n_heads, hd)
        v_sel = v_blocks[safe_indices].reshape(n_msgs, S * BS, n_heads, hd)

        # Attention scores: (n_msgs, BS, n_heads, S*BS)
        scores = jnp.einsum('mbhd,mshd->mbhs', q_msgs * scale, k_sel)

        # Causal mask: key position ≤ query position
        q_pos = jnp.arange(n_msgs)[:, None] * BS + jnp.arange(BS)[None, :]  # (n_msgs, BS)
        k_offset = jnp.arange(S * BS)
        k_block_origins = safe_indices[:, k_offset // BS]  # (n_msgs, S*BS)
        k_pos = k_block_origins * BS + k_offset[None, :] % BS  # (n_msgs, S*BS)

        causal = k_pos[:, None, :] <= q_pos[:, :, None]  # (n_msgs, BS, S*BS)

        scores = jnp.where(
            causal[:, :, None, :], scores,
            jnp.finfo(jnp.float32).min)

        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)

        # Zero out for queries with no valid keys
        has_valid = causal.any(axis=-1)  # (n_msgs, BS)
        attn = jnp.where(has_valid[:, :, None, None], attn, 0.0)
        attn = attn.astype(q.dtype)

        o_slc = jnp.einsum('mbhs,mshd->mbhd', attn, v_sel)
        return o_slc.reshape(L, n_heads, hd)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------
    def __call__(self, input_sequence):
        """Forward pass. Input: (L, H), Output: (L, H)."""
        L = input_sequence.shape[0]
        nh = self._nh
        hd = self._hd
        scale = jnp.float32(hd) ** -0.5

        if self._effective_pe == 'rope':
            # RoPE: no input PE — position is injected into Q, K after projection
            x = input_sequence.astype(self.dtype)
        else:
            # DEPRECATED: sinusoidal PE — additive on input (old checkpoint compat only)
            pe = sinusoidal_positional_encoding(L, self.H)  # DEPRECATED: sinusoidal PE
            x = input_sequence + pe  # DEPRECATED: sinusoidal PE additive
            x = x.astype(self.dtype)  # DEPRECATED: sinusoidal PE cast

        # Pre-norm
        h = self.norm1(x)

        # QKV projections → (L, n_heads, head_dim)
        q = self.q_proj(h).reshape(L, nh, hd)
        k = self.k_proj(h).reshape(L, nh, hd)
        v = self.v_proj(h).reshape(L, nh, hd)

        if self._effective_pe == 'rope':
            # Apply RoPE to Q, K (not V) — positions 0..L-1
            cos_table, sin_table = rotary_embedding(hd, L, base=self.rope_base)
            q, k = apply_rope(q, k, cos_table, sin_table)

        # Gates: 3 sigmoid gates per head
        gates = jax.nn.sigmoid(self.g_proj(h)).reshape(L, nh, 3)
        g_cmp = gates[..., 0:1]  # (L, nh, 1)
        g_slc = gates[..., 1:2]
        g_swa = gates[..., 2:3]

        # Effective block_counts (can't select more blocks than exist)
        BS = self.block_size
        n_blocks = L // BS
        S = min(self.block_counts, n_blocks)

        # Branch 1: Compressed (global coarse)
        o_cmp, top_indices = self._compressed_attention(q, k, v, scale, S)

        # Branch 2: Selected (fine sparse)
        o_slc = self._selected_attention(q, k, v, top_indices, scale, S)

        # Branch 3: Causal attention via cuDNN flash — O(L) memory
        # Uses full causal (not windowed) for memory efficiency.
        # Gates learn to weight local vs global context.
        o_swa = flash_causal_attention(q, k, v)

        # Combine via gates
        o = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa  # (L, nh, hd)

        # Output projection + residual
        o = self.o_proj(o.reshape(L, self.H))
        o = self.drop(o)
        x = x + o

        # Pre-norm FFN + residual
        h = self.norm2(x)
        h = self.ff(h)
        h = self.drop(h)
        x = x + h

        return x

    # ------------------------------------------------------------------
    # Prefill: training-identical forward + KV cache construction
    # ------------------------------------------------------------------
    def prefill(self, input_sequence, max_cache_len):
        """Same computation as __call__ (flash attention), but also builds
        the KV cache for subsequent __call_inference__ decode steps.

        Args:
            input_sequence: (L, H) — conditioning tokens
            max_cache_len: int — total cache size (cond + gen tokens)

        Returns:
            (cache, output) where cache = (k_cache, v_cache, k_cmp, v_cmp, pos)
        """
        L = input_sequence.shape[0]
        nh = self._nh
        hd = self._hd
        BS = self.block_size

        if self._effective_pe == 'rope':
            # RoPE: no input PE
            x = input_sequence.astype(self.dtype)
        else:
            # DEPRECATED: sinusoidal PE with Position Interpolation (old checkpoint compat only)
            # DEPRECATED: Scale positions by train_max / total_max so all positions stay within
            # DEPRECATED: the training range [0, train_max). Both prefill and decode must use
            # DEPRECATED: the SAME scale factor for position continuity.
            train_max_pos = 13000  # DEPRECATED: sinusoidal PE — msg_seq_len * block_size
            pi_scale = train_max_pos / max(max_cache_len, train_max_pos)  # DEPRECATED: sinusoidal PI scale
            positions = jnp.arange(L).astype(jnp.float32) * pi_scale  # DEPRECATED: sinusoidal PI positions
            pe = positional_encoding_at_positions(positions, self.H)  # DEPRECATED: sinusoidal PE lookup
            x = input_sequence + pe  # DEPRECATED: sinusoidal PE additive
            x = x.astype(self.dtype)  # DEPRECATED: sinusoidal PE cast

        h = self.norm1(x)

        q = self.q_proj(h).reshape(L, nh, hd)
        k = self.k_proj(h).reshape(L, nh, hd)
        v = self.v_proj(h).reshape(L, nh, hd)

        if self._effective_pe == 'rope':
            # RoPE on Q, K at positions 0..L-1
            # For inference context extension: use NTK scaling if cache exceeds
            # training range, otherwise standard RoPE
            train_max_pos = 13000
            if max_cache_len > train_max_pos:
                scale_factor = max_cache_len / train_max_pos
                cos_table, sin_table = rotary_embedding_ntk(
                    hd, max_cache_len, base=self.rope_base,
                    scale_factor=scale_factor)
            else:
                cos_table, sin_table = rotary_embedding(
                    hd, L, base=self.rope_base)
            positions = jnp.arange(L, dtype=jnp.int32)
            q, k = apply_rope_at_positions(q, k, positions, cos_table, sin_table)

        scale = jnp.float32(hd) ** -0.5
        gates = jax.nn.sigmoid(self.g_proj(h)).reshape(L, nh, 3)
        g_cmp = gates[..., 0:1]
        g_slc = gates[..., 1:2]
        g_swa = gates[..., 2:3]

        n_blocks = L // BS
        S = min(self.block_counts, n_blocks)
        o_cmp, top_indices = self._compressed_attention(q, k, v, scale, S)
        o_slc = self._selected_attention(q, k, v, top_indices, scale, S)
        o_swa = flash_causal_attention(q, k, v)

        o = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
        o = self.o_proj(o.reshape(L, self.H))
        o = self.drop(o)
        x = x + o

        h = self.norm2(x)
        h = self.ff(h)
        h = self.drop(h)
        x = x + h

        # === Build KV cache from this forward pass ===
        # Leading dim=1 matches initialize_cache format (placeholder for vmap)
        k_cache = jnp.zeros((1, nh, max_cache_len, hd), dtype=self.dtype)
        v_cache = jnp.zeros((1, nh, max_cache_len, hd), dtype=self.dtype)
        k_t = jnp.transpose(k, (1, 0, 2))  # (nh, L, hd)
        v_t = jnp.transpose(v, (1, 0, 2))
        k_cache = k_cache.at[0, :, :L, :].set(k_t)
        v_cache = v_cache.at[0, :, :L, :].set(v_t)

        # Compressed cache
        max_blocks = max_cache_len // BS
        k_cmp = jnp.zeros((1, nh, max_blocks, hd), dtype=self.dtype)
        v_cmp = jnp.zeros((1, nh, max_blocks, hd), dtype=self.dtype)

        n_complete = L // BS
        if n_complete > 0:
            k_blocks = k.reshape(n_complete, BS, nh, hd).mean(axis=1)
            v_blocks = v.reshape(n_complete, BS, nh, hd).mean(axis=1)
            k_cmp = k_cmp.at[0, :, :n_complete, :].set(jnp.transpose(k_blocks, (1, 0, 2)))
            v_cmp = v_cmp.at[0, :, :n_complete, :].set(jnp.transpose(v_blocks, (1, 0, 2)))

        pos = jnp.zeros((1,), dtype=jnp.int32) + L
        cache = (k_cache, v_cache, k_cmp, v_cmp, pos)

        return cache, x

    # ------------------------------------------------------------------
    # Autoregressive inference with KV cache
    # ------------------------------------------------------------------
    def __call_inference__(self, hidden, input_sequence, resets):
        """Autoregressive inference with KV + compressed cache.

        hidden = (k_cache, v_cache, k_cmp_cache, v_cmp_cache, pos)
        Shapes carry a leading-1 placeholder dim from initialize_cache.
        """
        k_cache, v_cache, k_cmp_cache, v_cmp_cache, pos = hidden

        # Squeeze leading-1 from initialize_cache
        k_cache = jnp.squeeze(k_cache, axis=0) if k_cache.ndim == 4 else k_cache
        v_cache = jnp.squeeze(v_cache, axis=0) if v_cache.ndim == 4 else v_cache
        k_cmp_cache = jnp.squeeze(k_cmp_cache, axis=0) if k_cmp_cache.ndim == 4 else k_cmp_cache
        v_cmp_cache = jnp.squeeze(v_cmp_cache, axis=0) if v_cmp_cache.ndim == 4 else v_cmp_cache
        pos = jnp.squeeze(pos) if pos.ndim >= 1 else pos

        L = input_sequence.shape[0]
        nh = self._nh
        hd = self._hd
        scale = jnp.float32(hd) ** -0.5

        BS = self.block_size
        max_len = k_cache.shape[1]
        max_blocks = k_cmp_cache.shape[1]
        S = min(self.block_counts, max_blocks)

        if self._effective_pe == 'rope':
            # RoPE: no input PE — position injected into Q, K below
            x = input_sequence.astype(self.dtype)
        else:
            # DEPRECATED: sinusoidal PE with Position Interpolation (old checkpoint compat only)
            # DEPRECATED: Training uses msg_seq_len=500 = 13000 tokens. Positions beyond 13000
            # DEPRECATED: are OOD. PI scales positions to stay within [0, train_max).
            train_max_pos = 13000  # DEPRECATED: sinusoidal PE — msg_seq_len * block_size
            positions = pos + jnp.arange(L)  # DEPRECATED: sinusoidal PE absolute positions
            # DEPRECATED: Scale if we'd exceed training range
            total_max_pos = jnp.maximum(max_len, train_max_pos)  # DEPRECATED: sinusoidal PI denominator
            pi_scale = train_max_pos / total_max_pos  # DEPRECATED: sinusoidal PI scale factor
            positions_scaled = (positions * pi_scale).astype(jnp.float32)  # DEPRECATED: sinusoidal PI scaled positions
            pe = positional_encoding_at_positions(positions_scaled, self.H)  # DEPRECATED: sinusoidal PE lookup
            x = (input_sequence + pe).astype(self.dtype)  # DEPRECATED: sinusoidal PE additive

        # Pre-norm
        h = self.norm1(x)

        # QKV
        q = self.q_proj(h).reshape(L, nh, hd)
        k_new = self.k_proj(h).reshape(L, nh, hd)
        v_new = self.v_proj(h).reshape(L, nh, hd)

        if self._effective_pe == 'rope':
            # Apply RoPE to Q and K_new at absolute positions pos..pos+L-1.
            # K is RoPE'd BEFORE writing to cache so cached K already has
            # positional information baked in.
            train_max_pos = 13000
            if max_len > train_max_pos:
                scale_factor = max_len / train_max_pos
                cos_table, sin_table = rotary_embedding_ntk(
                    hd, max_len, base=self.rope_base,
                    scale_factor=scale_factor)
            else:
                cos_table, sin_table = rotary_embedding(
                    hd, max_len, base=self.rope_base)
            rope_positions = (pos + jnp.arange(L)).astype(jnp.int32)
            q, k_new = apply_rope_at_positions(
                q, k_new, rope_positions, cos_table, sin_table)

        # Write K, V into main cache
        k_new_t = jnp.transpose(k_new, (1, 0, 2))  # (nh, L, hd)
        v_new_t = jnp.transpose(v_new, (1, 0, 2))
        k_cache = jax.lax.dynamic_update_slice(k_cache, k_new_t, (0, pos, 0))
        v_cache = jax.lax.dynamic_update_slice(v_cache, v_new_t, (0, pos, 0))

        valid_len = pos + L

        # Number of PREVIOUSLY completed blocks (before this call's tokens).
        # Used by Branch 1 as scalar mask — matches training's causal semantics
        # because the compressed cache hasn't been updated yet at this point.
        n_complete_before = pos // BS

        # Gates
        gates = jax.nn.sigmoid(self.g_proj(h)).reshape(L, nh, 3)
        g_cmp = gates[..., 0:1]
        g_slc = gates[..., 1:2]
        g_swa = gates[..., 2:3]

        # Q in (nh, L, hd) for cache-based attention
        q_t = jnp.transpose(q, (1, 0, 2))
        query_positions = pos + jnp.arange(L)

        # === Branch 3: Explicit causal attention over KV cache ===
        # Uses XLA einsum (not cuDNN) for L=1 decode — 2.6x faster than cuDNN
        # with custom bias because XLA fuses the small matmul trivially while
        # cuDNN has per-call kernel launch overhead (~10ms × 13000 calls = 120s).
        # Prefill already handles conditioning with training-identical flash attention.
        # For L=1 decode, explicit and cuDNN softmax produce identical results.
        k_all = k_cache  # (nh, max_len, hd) — native layout, no transpose
        v_all = v_cache

        swa_scores = jnp.einsum('hqd,hkd->hqk', q_t * scale, k_all)
        all_positions = jnp.arange(max_len)
        swa_mask = ((all_positions[None, :] < valid_len) &
                    (all_positions[None, :] <= query_positions[:, None]))
        swa_scores = jnp.where(
            swa_mask[None, :, :], swa_scores, jnp.finfo(jnp.float32).min)
        swa_attn = jax.nn.softmax(swa_scores.astype(jnp.float32), axis=-1)
        swa_attn = swa_attn.astype(self.dtype)
        o_swa = jnp.einsum('hqk,hkd->hqd', swa_attn, v_all)
        o_swa = jnp.transpose(o_swa, (1, 0, 2))

        # === Branch 1: Compressed ===
        # Use n_complete_before (NOT n_complete_now) — the compressed cache
        # hasn't been updated yet, so only previously completed blocks are valid.
        # This matches training's causal mask where tokens in the current block
        # cannot see the current block's compressed representation.
        cmp_scores = jnp.einsum('lhd,hnd->lhn', q * scale, k_cmp_cache)
        n_visible_cmp = n_complete_before
        cmp_mask = jnp.arange(max_blocks)[None, :] < n_visible_cmp
        cmp_scores = jnp.where(
            cmp_mask[:, None, :], cmp_scores, jnp.finfo(jnp.float32).min)
        has_cmp = n_visible_cmp > 0
        cmp_attn = jax.nn.softmax(cmp_scores.astype(jnp.float32), axis=-1)
        cmp_attn = jnp.where(has_cmp, cmp_attn, 0.0).astype(self.dtype)
        o_cmp = jnp.einsum('lhn,hnd->lhd', cmp_attn, v_cmp_cache)

        # === Branch 2: Selected ===
        importance = cmp_attn.astype(jnp.float32).sum(axis=1)  # (L, max_blocks)
        current_block = query_positions // BS
        local_mask = jnp.arange(max_blocks)[None, :] == current_block[:, None]
        importance = jnp.where(local_mask, 1.0, importance)
        importance = jnp.where(cmp_mask, importance, -1e9)

        _, top_idx = jax.lax.top_k(importance, S)  # (L, S)

        # Expand block indices to token indices
        token_offsets = jnp.arange(BS)[None, None, :]       # (1, 1, BS)
        token_indices = top_idx[:, :, None] * BS + token_offsets  # (L, S, BS)
        token_indices = jnp.clip(token_indices.reshape(L, S * BS), 0, max_len - 1)

        # Gather K, V from main cache
        k_sel = k_cache[:, token_indices.reshape(-1), :].reshape(nh, L, S * BS, hd)
        v_sel = v_cache[:, token_indices.reshape(-1), :].reshape(nh, L, S * BS, hd)

        sel_scores = jnp.einsum('hld,hlsd->hls', q_t * scale, k_sel)
        sel_positions = token_indices  # (L, S*BS)
        sel_mask = ((sel_positions < valid_len) &
                    (sel_positions <= query_positions[:, None]))
        sel_scores = jnp.where(
            sel_mask[None, :, :], sel_scores, jnp.finfo(jnp.float32).min)
        has_sel = sel_mask.any(axis=-1)  # (L,)
        sel_attn = jax.nn.softmax(sel_scores.astype(jnp.float32), axis=-1)
        sel_attn = jnp.where(has_sel[None, :, None], sel_attn, 0.0).astype(self.dtype)
        o_slc = jnp.einsum('hls,hlsd->hld', sel_attn, v_sel)
        o_slc = jnp.transpose(o_slc, (1, 0, 2))

        # Combine
        o = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
        o = self.o_proj(o.reshape(L, self.H))
        o = self.drop(o)
        x = x + o

        # FFN
        h = self.norm2(x)
        h = self.ff(h)
        h = self.drop(h)
        x = x + h

        # Update compressed cache AFTER attention (not before).
        # This ensures Branch 1 only sees previously completed blocks,
        # matching training's causal semantics.
        n_complete_now = valid_len // BS
        new_block_idx = n_complete_now - 1
        block_just_completed = (valid_len % BS == 0) & (valid_len > 0)

        block_start = jnp.maximum(0, new_block_idx) * BS
        k_block = jax.lax.dynamic_slice(k_cache, (0, block_start, 0), (nh, BS, hd))
        v_block = jax.lax.dynamic_slice(v_cache, (0, block_start, 0), (nh, BS, hd))
        k_block_mean = k_block.mean(axis=1, keepdims=True)
        v_block_mean = v_block.mean(axis=1, keepdims=True)

        safe_block_idx = jnp.maximum(0, new_block_idx)
        k_cmp_new = jax.lax.dynamic_update_slice(
            k_cmp_cache, k_block_mean, (0, safe_block_idx, 0))
        v_cmp_new = jax.lax.dynamic_update_slice(
            v_cmp_cache, v_block_mean, (0, safe_block_idx, 0))

        k_cmp_cache = jnp.where(block_just_completed, k_cmp_new, k_cmp_cache)
        v_cmp_cache = jnp.where(block_just_completed, v_cmp_new, v_cmp_cache)

        new_hidden = (k_cache, v_cache, k_cmp_cache, v_cmp_cache, pos + L)
        return new_hidden, x

    @staticmethod
    def initialize_cache(batch_size, n_heads, head_dim, max_cache_len,
                         block_size, dtype=jnp.float32):
        """Create empty KV + compressed cache for one NSA layer.

        Returns 4D tensors with leading dim=1 (placeholder for inner vmap).
        """
        k = jnp.zeros((1, n_heads, max_cache_len, head_dim), dtype=dtype)
        v = jnp.zeros((1, n_heads, max_cache_len, head_dim), dtype=dtype)
        max_blocks = max_cache_len // block_size
        k_cmp = jnp.zeros((1, n_heads, max_blocks, head_dim), dtype=dtype)
        v_cmp = jnp.zeros((1, n_heads, max_blocks, head_dim), dtype=dtype)
        idx = jnp.zeros((1,), dtype=jnp.int32)
        return (k, v, k_cmp, v_cmp, idx)


# Flag for SequenceLayer to detect and bypass S5-specific wrapping
NSABlock.is_transformer = True


def init_NSABlock(H, n_heads, d_ff=0, block_size=26, block_counts=16,
                  window_size=512, dropout=0.0, dtype=jnp.float32,
                  remat=False, max_cache_len=26000,
                  pe_type='rope', rope_base=10000.0, **kwargs):
    """Factory matching init_S5SSM / init_TransformerBlock interface.

    Accepts and ignores S5-specific kwargs (Lambda_re_init, V, etc.)
    via **kwargs for compatibility.

    Args:
        pe_type: 'rope' (default, RECOMMENDED) or 'sinusoidal' (DEPRECATED, old checkpoint compat only).
                 RoPE rotates Q, K after projection. DEPRECATED sinusoidal adds PE to input.
        rope_base: RoPE frequency base (default 10000). Only used when pe_type='rope'.

    Returns:
        partial(NSABlock, ...) — same pattern as init_S5SSM
    """
    if d_ff <= 0:
        d_ff = 4 * H

    block_cls = NSABlock
    if remat:
        block_cls = nn.remat(NSABlock)

    return partial(
        block_cls,
        H=H,
        n_heads=n_heads,
        d_ff=d_ff,
        block_size=block_size,
        block_counts=block_counts,
        window_size=window_size,
        max_cache_len=max_cache_len,
        pe_type=pe_type,
        rope_base=rope_base,
        dropout=dropout,
        dtype=dtype,
    )
