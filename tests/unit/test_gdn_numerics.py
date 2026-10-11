import jax
import jax.numpy as jnp
import numpy as np
import pytest

from base_model.models.gdn import GDNSSM, _chunkwise_gdn, _l2_normalize


def _all_finite(tree) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _recurrent_gdn(q, k, v, beta, alpha_log):
    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    beta = beta.reshape(-1, *beta.shape[2:])
    alpha_log = alpha_log.reshape(-1, *alpha_log.shape[2:])
    num_heads, head_dim = q.shape[1:]
    head_v_dim = v.shape[-1]

    def step(state, values):
        q_t, k_t, v_t, beta_t, alpha_log_t = values
        state = state * jnp.exp(alpha_log_t)[:, None, :]
        delta = v_t - jnp.einsum("nvk,nk->nv", state, k_t)
        state = state + jnp.einsum(
            "nv,nk->nvk", beta_t[:, None] * delta, k_t
        )
        output = jnp.einsum("nvk,nk->nv", state, q_t)
        return state, output

    _, output = jax.lax.scan(
        step,
        jnp.zeros((num_heads, head_v_dim, head_dim), dtype=q.dtype),
        (q, k, v, beta, alpha_log),
    )
    return output


def test_l2_normalize_has_finite_zero_vector_backward() -> None:
    x = jnp.zeros((8,), dtype=jnp.float32)
    output = _l2_normalize(x)
    grad = jax.grad(lambda value: jnp.sum(_l2_normalize(value)))(x)

    assert _all_finite((output, grad))
    np.testing.assert_array_equal(np.asarray(output), np.zeros(8, dtype=np.float32))


@pytest.mark.parametrize("use_kda", [False, True])
def test_chunkwise_forward_and_backward_match_recurrent_reference(use_kda: bool) -> None:
    num_chunks, chunk_size = 2, 4
    num_heads, head_dim, head_v_dim = 1, 4, 8
    keys = jax.random.split(jax.random.PRNGKey(9 if use_kda else 4), 5)
    q = _l2_normalize(
        jax.random.normal(keys[0], (num_chunks, chunk_size, num_heads, head_dim))
    ) * head_dim ** -0.5
    k = _l2_normalize(
        jax.random.normal(keys[1], (num_chunks, chunk_size, num_heads, head_dim))
    )
    v = jax.random.normal(
        keys[2], (num_chunks, chunk_size, num_heads, head_v_dim)
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[3], (num_chunks, chunk_size, num_heads))
    )
    alpha_dim = head_dim if use_kda else 1
    alpha_log = jax.nn.log_sigmoid(
        jax.random.normal(
            keys[4], (num_chunks, chunk_size, num_heads, alpha_dim)
        )
    )
    inputs = (q, k, v, beta, alpha_log)

    def chunk_loss(*values):
        output = _chunkwise_gdn(
            *values,
            nh=num_heads,
            hd=head_dim,
            hvd=head_v_dim,
            C=chunk_size,
            num_chunks=num_chunks,
        )
        return jnp.mean(jnp.square(output)), output

    def recurrent_loss(*values):
        output = _recurrent_gdn(*values)
        return jnp.mean(jnp.square(output)), output

    (chunk_value, chunk_output), chunk_grads = jax.value_and_grad(
        chunk_loss, argnums=range(5), has_aux=True
    )(*inputs)
    (recurrent_value, recurrent_output), recurrent_grads = jax.value_and_grad(
        recurrent_loss, argnums=range(5), has_aux=True
    )(*inputs)

    assert _all_finite((chunk_value, chunk_output, chunk_grads))
    np.testing.assert_allclose(chunk_value, recurrent_value, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        chunk_output.reshape(recurrent_output.shape),
        recurrent_output,
        rtol=2e-5,
        atol=2e-6,
    )
    for chunk_grad, recurrent_grad in zip(chunk_grads, recurrent_grads):
        np.testing.assert_allclose(
            chunk_grad, recurrent_grad, rtol=2e-4, atol=2e-5
        )


@pytest.mark.parametrize("use_kda", [False, True])
def test_module_backward_survives_zero_input_rows(use_kda: bool) -> None:
    model = GDNSSM(
        H=4,
        num_heads=1,
        head_dim=4,
        expand_v=1,
        chunk_size=2,
        use_conv=False,
        use_kda=use_kda,
    )
    key = jax.random.PRNGKey(17)
    inputs = jax.random.normal(key, (4, 4)).at[:2].set(0.0)
    params = model.init(key, inputs)["params"]

    def loss_fn(candidate_params):
        output = model.apply({"params": candidate_params}, inputs)
        return jnp.mean(jnp.square(output))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert _all_finite((loss, grads))
