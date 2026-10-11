from __future__ import annotations

import os

try:
    _cpus = sorted(os.sched_getaffinity(0))
    if len(_cpus) > 4:
        os.sched_setaffinity(0, set(_cpus[:4]))
except (AttributeError, OSError):
    pass

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("chex")

from matching_engine.jaxob.JaxOrderBookArrays import (
    get_order_by_time,
    get_order_by_time_and_price,
)
from matching_engine.jaxob.jaxob_constants import NEGATIVE_RETURN_ID


SIDE = jnp.array(
    [
        [100, 10, 1, 11, 34_200, 100],
        [200, 20, 2, 22, 34_200, 500_000_000],
        [-1, -1, -1, -1, -1, -1],
    ],
    dtype=jnp.int32,
)


def test_time_lookup_requires_an_exact_timestamp() -> None:
    exact = np.asarray(get_order_by_time(SIDE, 34_200, 100))
    near = np.asarray(get_order_by_time(SIDE, 34_200, 101))

    np.testing.assert_array_equal(exact, np.asarray(SIDE[0]))
    np.testing.assert_array_equal(
        near,
        np.full(6, NEGATIVE_RETURN_ID, dtype=np.int32),
    )


def test_price_lookup_rejects_a_near_timestamp() -> None:
    exact = np.asarray(
        get_order_by_time_and_price(SIDE, 34_200, 500_000_000, 200)
    )
    near = np.asarray(
        get_order_by_time_and_price(SIDE, 34_200, 500_000_001, 200)
    )

    np.testing.assert_array_equal(exact, np.asarray(SIDE[1]))
    np.testing.assert_array_equal(
        near,
        np.full(6, NEGATIVE_RETURN_ID, dtype=np.int32),
    )


def test_price_lookup_falls_back_only_to_the_same_exact_timestamp() -> None:
    result = np.asarray(
        get_order_by_time_and_price(SIDE, 34_200, 100, 999)
    )

    np.testing.assert_array_equal(result, np.asarray(SIDE[0]))
