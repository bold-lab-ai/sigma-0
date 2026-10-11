"""Sequence-parallel SSD phases 4-5-6 for Mamba3.

Why this exists. Training memory for the recurrent trunk grows at 35.4 GB per
thousand messages per GPU, which caps a single GH200 at roughly 2,000 messages
and puts 4,000 at 134 to 142 GB against 85.5 available. Splitting the sequence
across the four GPUs on a node divides that by four. Combined with two sequences
per group it reaches a 4,000-message context at the same per-GPU memory, the
same tokens per step and the same step count as the 500-message configuration
in use today, so the context grows eightfold and nothing else moves.

What makes it cheap here. Every other operation in this model is pointwise along
the sequence: embedding, RMSNorm, the dense projections, half_glu1, the decoder.
The SSD scan is the only operator that couples positions, so it is the only one
that needs to know the sequence is split.

The decomposition. Phase 5 in mamba3_jax is not the sequential scan its comment
claims. It materialises the full (nc+1, nc+1) inter-chunk decay matrix and
contracts it against the per-chunk states in one einsum, which is O(nc^2). That
is what makes the split profitable twice over: with D devices each one runs
(nc/D)^2 instead of nc^2, so the total inter-chunk work falls by a factor of D
as well as the memory.

The recurrence over chunks is affine, state_out = exp(A) * state_in + local, so
it is associative and the device boundaries compose exactly the same way the
chunk boundaries do. The cross-device step is therefore the identical all-pairs
formula applied one level up, over D elements instead of nc.

The implementation is smaller than the explanation because of one accident of
the reference code: it computes new_states with nc+1 entries and then throws the
last one away. That discarded entry is precisely the state leaving the last
chunk, which is exactly what a device has to hand to its successor. Keeping it
is the whole of phase 5's local half.

Numerics. Under the mask, decay entries are exp(cumA[z] - cumA[c]) for c <= z
and zero above, and the split reassociates that sum without changing any term,
so the sharded result matches the unsharded one to floating-point reassociation
only. test_mamba3_seqpar_parity.py asserts that against the reference.
"""
import jax
import jax.numpy as jnp

from base_model.models.mamba3_jax import mamba3_contraction_precision, segsum


def _local_states(Q_c, K_c, V_c, A_cumsum, precision):
    """Phase 4 plus the local half of phase 5.

    Returns (states_in, state_out) where states_in[z] is the state entering
    local chunk z counting only this device's chunks, and state_out is the
    state leaving this device's last chunk, again counting only its own
    chunks. state_out is the entry the reference discards as new_states[-1].
    """
    decay_states = jnp.exp(A_cumsum[:, :, -1:] - A_cumsum)      # (nh, nc, CS)
    states = jnp.einsum('cshp,hcs,cshn->chpn',
                        V_c, decay_states, K_c, precision=precision)

    initial_state = jnp.zeros((1,) + states.shape[1:], dtype=states.dtype)
    states = jnp.concatenate([initial_state, states], axis=0)   # (nc+1, ...)
    A_end = A_cumsum[:, :, -1]                                  # (nh, nc)
    A_end_padded = jnp.pad(A_end, ((0, 0), (1, 0)))             # (nh, nc+1)
    decay_chunk = jnp.exp(segsum(A_end_padded))                 # (nh, nc+1, nc+1)
    new_states = jnp.einsum('hzc,chpn->zhpn',
                            decay_chunk, states, precision=precision)
    return new_states[:-1], new_states[-1]


def _carry_in(state_out, A_cumsum, axis_name, precision):
    """The state entering this device, given every device's local exit state.

    Identical algebra to phase 5, one level up: D elements instead of nc, and
    the per-device total log-decay in place of the per-chunk one. D is 4, so
    gathering all of them and running the (D+1, D+1) contraction redundantly on
    every device costs less than arranging a shifted exchange would.
    """
    T = A_cumsum[:, :, -1].sum(axis=-1)                         # (nh,)
    T_all = jax.lax.all_gather(T, axis_name)                    # (D, nh)
    S_all = jax.lax.all_gather(state_out, axis_name)            # (D, nh, hd, N)

    S_padded = jnp.concatenate(
        [jnp.zeros((1,) + S_all.shape[1:], dtype=S_all.dtype), S_all], axis=0)
    T_padded = jnp.pad(T_all.T, ((0, 0), (1, 0)))               # (nh, D+1)
    decay_dev = jnp.exp(segsum(T_padded))                       # (nh, D+1, D+1)
    H_all = jnp.einsum('hde,ehpn->dhpn',
                       decay_dev, S_padded, precision=precision)
    return H_all[jax.lax.axis_index(axis_name)]                 # (nh, hd, N)


def phases_4_5_6_seqpar(Q_c, K_c, V_c, ADT_c, A_cumsum, *, axis_name):
    """Drop-in for _phases_4_5_6_jax when the chunk axis is sharded.

    Same signature and same return as the reference apart from axis_name, so it
    can be passed straight to mamba3_ssd_chunked_jax as phases_4_5_6_fn. Must be
    called inside a shard_map over axis_name with the chunk axis of every
    chunked input mapped to it.
    """
    precision = mamba3_contraction_precision()
    states_in, state_out = _local_states(Q_c, K_c, V_c, A_cumsum, precision)

    carry = _carry_in(state_out, A_cumsum, axis_name, precision)

    # Decay from this device's start to the start of each of its chunks. Same
    # padded prefix the local pass uses, so chunk 0 sees the carry undecayed.
    A_end = A_cumsum[:, :, -1]                                  # (nh, nc)
    cum_to_chunk = jnp.pad(jnp.cumsum(A_end, axis=-1)[:, :-1], ((0, 0), (1, 0)))
    states_in = states_in + jnp.exp(cum_to_chunk)[..., None, None].transpose(
        1, 0, 2, 3) * carry[None]

    state_decay_out = jnp.exp(A_cumsum)                         # (nh, nc, CS)
    return jnp.einsum('clhn,chpn,hcl->clhp',
                      Q_c, states_in, state_decay_out, precision=precision)
