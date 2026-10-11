from argparse import Namespace

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="full backbone compilation is validated in the GPU smoke matrix",
)

from base_model.models.layers import SequenceLayer
from base_model.models.registry import available_architectures, build_backbone


def _args(architecture: str) -> Namespace:
    return Namespace(
        architecture=architecture,
        ssm_type=None,
        model_type=None,
        d_model=64,
        p_dropout=0.0,
        ssm_size_base=64,
        blocks=4,
        conj_sym=True,
        C_init="trunc_standard_normal",
        discretization="zoh",
        dt_min=0.001,
        dt_max=0.1,
        clip_eigs=True,
        bidirectional=False,
        n_heads=4,
        d_ff=128,
        dtype="float32",
        remat=False,
        use_flash=False,
        attention_max_cache_len=104,
        block_size=26,
        block_counts=2,
        window_size=26,
        pe_type="rope",
        rope_base=10000.0,
        mamba3_d_state=16,
        mamba3_expand=2,
        mamba3_headdim=16,
        mamba3_chunk_size=16,
        mamba3_rope_fraction=0.5,
        mamba3_use_triton=False,
        mamba3_use_cuda=False,
        tp_size=1,
        gdn_head_dim=16,
        gdn_num_heads=4,
        gdn_expand_v=2,
        gdn_chunk_size=16,
        gdn_use_conv=True,
    )


@pytest.mark.parametrize("architecture", available_architectures())
def test_tiny_backbone_forward_is_finite(architecture: str) -> None:
    built = build_backbone(_args(architecture))
    layer = SequenceLayer(
        ssm=built.layer_factory,
        dropout=0.0,
        d_model=64,
        training=False,
        prenorm=True,
    )
    inputs = jnp.ones((52, 64), dtype=jnp.float32)
    variables = layer.init(jax.random.key(7), inputs)
    outputs = layer.apply(variables, inputs)
    assert outputs.shape == inputs.shape
    assert bool(jnp.all(jnp.isfinite(outputs)))


def test_nsa_blocks_are_message_aligned() -> None:
    args = _args("nsa")
    args.block_size = 24
    with pytest.raises(ValueError, match="26-token message length"):
        build_backbone(args)
