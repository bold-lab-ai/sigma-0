import jax
import jax.numpy as jnp
import pytest

from base_model.models.transformer import _pallas_attention_fn


def _xla_attention(query, key, value):
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        implementation="xla",
    )


@pytest.mark.parametrize("seq_len", (7, 128, 130))
def test_pallas_attention_forward_matches_xla_with_padding(seq_len):
    keys = jax.random.split(jax.random.key(seq_len), 3)
    shape = (2, seq_len, 2, 16)
    query, key, value = (
        jax.random.normal(rng, shape, dtype=jnp.float32) for rng in keys
    )

    expected = _xla_attention(query, key, value)
    actual = _pallas_attention_fn(query, key, value, interpret=True)

    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_pallas_attention_backward_matches_xla():
    keys = jax.random.split(jax.random.key(2026), 4)
    shape = (1, 130, 2, 16)
    query, key, value = (
        jax.random.normal(rng, shape, dtype=jnp.float32) for rng in keys[:3]
    )
    cotangent = jax.random.normal(keys[3], shape, dtype=jnp.float32)

    def scalar_loss(attention_fn, q, k, v):
        return jnp.sum(attention_fn(q, k, v) * cotangent)

    expected = jax.grad(
        lambda q, k, v: scalar_loss(_xla_attention, q, k, v),
        argnums=(0, 1, 2),
    )(query, key, value)
    actual = jax.grad(
        lambda q, k, v: scalar_loss(
            lambda q_, k_, v_: _pallas_attention_fn(
                q_, k_, v_, interpret=True
            ),
            q,
            k,
            v,
        ),
        argnums=(0, 1, 2),
    )(query, key, value)

    for actual_grad, expected_grad in zip(actual, expected):
        assert actual_grad.shape == expected_grad.shape
        assert jnp.allclose(actual_grad, expected_grad, rtol=8e-4, atol=8e-4)
