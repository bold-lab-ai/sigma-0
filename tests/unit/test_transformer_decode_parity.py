"""The training forward and the KV-cache decode must compute the same function.

A hybrid trains through TransformerBlock.__call__ and generates through
prefill + __call_inference__. Nothing forces those three to agree, and when they
disagree the model still trains fine, still reports a good held-out CE, and only
its generated sequences are wrong. That failure is invisible to every gate we
had: the loss uses the training path, so it stays clean while generation rots.

This is not hypothetical. use_positional_encoding was added to __call__ and not
to the other two, so a NoPE hybrid trained without positional encoding and then
generated with one injected into every query and key. It cost 11 points of
reference recall at 32,000 steps and read, at first, like an architectural
result.

The test is the obvious one and should have existed from the first commit: run
both paths over identical real inputs and require the same numbers.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from base_model.models.transformer import init_TransformerBlock

H, N_HEADS, D_FF, L_COND, N_STEP = 64, 4, 128, 24, 6

# The shape actually trained, so the head-count fallback loop in prefill and
# __call_inference__ is exercised on the same divisibility it sees in production.
PROD_H, PROD_HEADS, PROD_D_FF = 640, 10, 1280


def _paths(use_pe, seed=0, H=H, n_heads=N_HEADS, d_ff=D_FF):
    """Returns (parallel_out, decoded_out) over the same input sequence."""
    block = init_TransformerBlock(
        H=H, n_heads=n_heads, d_ff=d_ff, dropout=0.0, training=False,
        use_flash=False, use_positional_encoding=use_pe,
        max_cache_len=L_COND + N_STEP + 4,
    )()
    L = L_COND + N_STEP
    x = jax.random.normal(jax.random.PRNGKey(seed), (L, H), dtype=jnp.float32)
    params = block.init(jax.random.PRNGKey(1), x)

    parallel = block.apply(params, x)

    def prefill(p, seq):
        return block.apply(p, seq, L_COND + N_STEP + 4, method=block.prefill)

    def step(p, hidden, tok):
        return block.apply(p, hidden, tok, None, method=block.__call_inference__)

    hidden, _ = prefill(params, x[:L_COND])
    outs = []
    for t in range(L_COND, L):
        hidden, y = step(params, hidden, x[t:t + 1])
        outs.append(y[0])
    return np.asarray(parallel[L_COND:]), np.asarray(jnp.stack(outs))


@pytest.mark.parametrize("use_pe", [True, False])
def test_decode_matches_training_forward(use_pe):
    """Token-by-token decoding must reproduce the parallel forward pass.

    Both settings are checked. use_pe=True was already consistent, so it is the
    control: it shows the test is measuring path agreement and not merely
    detecting that PE exists.
    """
    par, dec = _paths(use_pe)
    err = np.abs(par - dec).max()
    assert err < 2e-4, (
        f"use_positional_encoding={use_pe}: decode differs from training "
        f"forward by {err:.3e}. The generated distribution is not the one the "
        f"loss was computed against."
    )


def test_decode_parity_at_production_width():
    """Same check at the width the hybrid actually trains at, NoPE as deployed."""
    par, dec = _paths(False, seed=3, H=PROD_H, n_heads=PROD_HEADS, d_ff=PROD_D_FF)
    err = np.abs(par - dec).max()
    assert err < 2e-4, f"d_model={PROD_H}: decode differs by {err:.3e}"


def test_prefill_output_matches_call():
    """prefill returns the forward output and a cache; they must share an input.

    The regression was here: the returned output came from self(seq) with PE
    disabled while the cached keys were rebuilt from seq + PE, so even the
    conditioning window's cache disagreed with the activations above it.
    """
    block = init_TransformerBlock(
        H=H, n_heads=N_HEADS, d_ff=D_FF, dropout=0.0, training=False,
        use_flash=False, use_positional_encoding=False,
        max_cache_len=L_COND + 4,
    )()
    x = jax.random.normal(jax.random.PRNGKey(2), (L_COND, H), dtype=jnp.float32)
    params = block.init(jax.random.PRNGKey(1), x)
    (_, _, pos), out = block.apply(params, x, L_COND + 4, method=block.prefill)
    assert int(pos[0]) == L_COND
    np.testing.assert_allclose(np.asarray(out), np.asarray(block.apply(params, x)),
                               rtol=0, atol=1e-6)
