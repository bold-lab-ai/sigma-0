from types import SimpleNamespace

import pytest

from base_model.models import (
    DEFAULT_ARCHITECTURE,
    attention_config_for_width,
    available_architectures,
    recurrent_carry_for_width,
    resolve_architecture,
)
from base_model.models.registry import BackboneBuild, BackboneDefinition


def test_mamba3_is_the_default_and_mamba2_is_excluded() -> None:
    assert DEFAULT_ARCHITECTURE == "mamba3"
    assert resolve_architecture() == "mamba3"
    assert available_architectures() == (
        "mamba3",
        "s5",
        "gdn",
        "kda",
        "transformer",
        "nsa",
        "hybrid_mamba3",
    )
    with pytest.raises(ValueError, match="intentionally not part"):
        resolve_architecture("mamba2")
    with pytest.raises(ValueError, match="intentionally not part"):
        resolve_architecture(model_type="mamba2")
    with pytest.raises(ValueError, match="unknown architecture"):
        resolve_architecture(model_type="not-a-model")


def test_legacy_dual_selectors_resolve_without_silent_fallback() -> None:
    assert resolve_architecture(ssm_type="gdn", model_type="s5") == "gdn"
    assert resolve_architecture(ssm_type="s5", model_type="transformer") == "transformer"
    assert resolve_architecture(ssm_type="s5", model_type="nsa") == "nsa"
    assert resolve_architecture(ssm_type="transformer", model_type="transformer") == "transformer"
    assert resolve_architecture(ssm_type="nsa", model_type="nsa") == "nsa"
    with pytest.raises(ValueError, match="conflicting legacy selectors"):
        resolve_architecture(ssm_type="gdn", model_type="transformer")


def test_canonical_and_legacy_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting architecture selectors"):
        resolve_architecture("mamba3", ssm_type="gdn")


def test_book_attention_cache_matches_effective_head_count() -> None:
    backbone = BackboneBuild(
        BackboneDefinition("nsa", "internal", True),
        lambda: None,
        {},
        {
            "model_type": "nsa",
            "n_heads": 4,
            "head_dim": 16,
            "max_cache_len": 416,
            "block_size": 26,
        },
    )
    config = attention_config_for_width(backbone, 503, max_cache_len=520)
    assert config["n_heads"] == 1
    assert config["head_dim"] == 503
    assert config["max_cache_len"] == 520
    assert config["block_size"] == 26


def test_recurrent_book_carries_match_effective_layer_geometry() -> None:
    mamba = recurrent_carry_for_width(
        {
            "ssm_type": "mamba3",
            "n_heads": 8,
            "headdim": 16,
            "d_state": 16,
            "num_rope_angles": 4,
            "expand": 2,
            "tp_size": 1,
        },
        41,
    )
    assert (mamba["n_heads"], mamba["headdim"]) == (5, 16)

    gdn = recurrent_carry_for_width(
        {
            "ssm_type": "gdn",
            "num_heads": 4,
            "head_dim": 16,
            "head_v_dim": 32,
            "expand_v": 2,
            "use_conv": True,
        },
        10,
    )
    assert (gdn["num_heads"], gdn["head_dim"], gdn["head_v_dim"]) == (1, 10, 20)


def _tiny_args(architecture: str, dtype: str) -> SimpleNamespace:
    return SimpleNamespace(
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
        dtype=dtype,
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


@pytest.mark.parametrize("architecture", ["s5", "gdn", "kda"])
def test_dtype_flag_is_refused_where_nothing_reads_it(architecture: str) -> None:
    """--dtype used to be accepted, forwarded and logged for these
    architectures while no builder read it, so a run tagged bfloat16 trained
    in float32. Until an architecture honours the flag, asking for a
    non-float32 dtype is an error, not a silent no-op."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    from base_model.models.registry import build_backbone

    build_backbone(_tiny_args(architecture, "float32"))  # the historical value builds
    with pytest.raises(ValueError, match="--dtype=bfloat16 has no effect"):
        build_backbone(_tiny_args(architecture, "bfloat16"))


@pytest.mark.parametrize("architecture", ["transformer", "nsa"])
def test_dtype_flag_reaches_the_attention_builders(architecture: str) -> None:
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    from base_model.models.registry import build_backbone

    built = build_backbone(_tiny_args(architecture, "bfloat16"))
    assert built.attention_config["dtype"] == jnp.bfloat16


def test_dtype_flag_reaches_mamba3_and_the_hybrid_mixer() -> None:
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    from base_model.models.registry import build_backbone

    built = build_backbone(_tiny_args("mamba3", "bfloat16"))
    assert built.layer_factory.keywords["dtype"] == jnp.bfloat16
    assert build_backbone(_tiny_args("mamba3", "float32")).layer_factory.keywords["dtype"] == jnp.float32

    args = _tiny_args("hybrid_mamba3", "bfloat16")
    args.n_layers = 4
    args.hybrid_attn_heads = 4
    args.hybrid_attn_flash = False
    hybrid = build_backbone(args)
    assert hybrid.layer_factory.keywords["dtype"] == jnp.bfloat16
    assert hybrid.attn_layer_factory.keywords["dtype"] == jnp.bfloat16
