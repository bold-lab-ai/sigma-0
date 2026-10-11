"""SINGLE_NODE_SHARD_MAP: one node can run the multi-node shard_map path.

Unset, a single process with --hierarchical still gets the historical 1D
'data' mesh (nothing changes for smokes and the zoo). With the switch, the
mesh is the (1, N) ('nodes', 'gpus') mesh the hierarchical train step wants,
and the batch axis spans both mesh axes exactly as on many nodes.
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                   # noqa: E402
import pytest                                                # noqa: E402

from base_model.sharding.sharding_utils import (            # noqa: E402
    _get_batch_axis,
    create_simple_mesh,
)

pytestmark = pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 host devices")


def test_default_single_node_hierarchical_is_still_the_1d_data_mesh():
    mesh = create_simple_mesh(4, hierarchical=True)
    assert mesh.axis_names == ("data",)
    assert mesh.devices.shape == (4,)
    assert _get_batch_axis(mesh) == "data"


def test_switch_builds_the_one_by_n_hierarchical_mesh():
    mesh = create_simple_mesh(4, hierarchical=True, allow_single_node_hierarchical=True)
    assert mesh.axis_names == ("nodes", "gpus")
    assert mesh.devices.shape == (1, 4)
    assert _get_batch_axis(mesh) == ("nodes", "gpus")


def test_switch_is_inert_without_hierarchical_or_with_tp():
    assert create_simple_mesh(4, hierarchical=False, allow_single_node_hierarchical=True).axis_names == ("data",)
    assert create_simple_mesh(4, hierarchical=True, tp_size=4, allow_single_node_hierarchical=True).axis_names == ("tp",)
