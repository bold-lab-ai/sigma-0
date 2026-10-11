from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from base_model.models.gdn import GDNSSM
from base_model.models.lob_seq_model import (
    BatchPaddedLobPredModel,
    LobBookModel,
    PaddedLobPredModel,
)
from base_model.models.layers import SequenceLayer
from base_model.models.mamba3 import Mamba3SSM, recurrent_angle_mode, recurrent_diagonal_mode
from base_model.models.mamba3_jax import mamba3_contraction_precision_mode
from base_model.models.nsa import NSABlock, init_NSABlock
from base_model.models.ssm import S5SSM
from base_model.models.transformer import TransformerBlock


@pytest.mark.parametrize(
    ("backbone", "decode_method"),
    (
        (Mamba3SSM, "__call_rnn__"),
        (GDNSSM, "__call_rnn__"),
        (S5SSM, "__call_rnn__"),
        (TransformerBlock, "__call_inference__"),
        (NSABlock, "__call_inference__"),
    ),
)
def test_archived_backbones_expose_a_decode_method(backbone, decode_method) -> None:
    assert callable(getattr(backbone, "__call__"))
    assert callable(getattr(backbone, decode_method))


def test_attention_backbones_expose_prefill() -> None:
    assert callable(TransformerBlock.prefill)
    assert callable(NSABlock.prefill)


def test_sequence_adapter_supports_both_historical_decode_spellings() -> None:
    assert callable(SequenceLayer.__call_rnn__)
    assert callable(SequenceLayer.__call_inference__)


def test_padded_training_model_exposes_legacy_autoregressive_entrypoint() -> None:
    assert callable(PaddedLobPredModel.__call_ar__)
    assert callable(BatchPaddedLobPredModel.__call_ar__)


def test_nsa_factory_is_26_token_aligned_by_default() -> None:
    factory = init_NSABlock(H=64, n_heads=4)
    assert isinstance(factory, partial)
    assert factory.keywords["block_size"] == 26
    assert factory.keywords["max_cache_len"] == 26000


def test_mamba3_book_carry_uses_book_width_geometry() -> None:
    pre, post = LobBookModel.initialize_carry(
        1,
        64,
        1,
        1,
        ssm_type="mamba3",
        d_book=503,
        n_heads=8,
        headdim=16,
        d_state=16,
        num_rope_angles=4,
        expand=2,
        tp_size=1,
        requested_headdim=16,
    )
    assert pre[0][0].shape == (1, 1, 62, 4)
    assert post[0][0].shape == (1, 1, 8, 4)


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_mamba3_parallel_and_recurrent_paths_are_equivalent(seed: int) -> None:
    """Inference must implement the same causal operator used for training."""
    model = Mamba3SSM(
        H=8,
        d_state=4,
        expand=2,
        headdim=4,
        chunk_size=4,
        rope_fraction=0.5,
    )
    input_key, parameter_key = jax.random.split(jax.random.key(seed))
    inputs = jax.random.normal(input_key, (13, 8))
    variables = model.init(parameter_key, inputs)

    parallel = model.apply(variables, inputs)
    hidden = (
        jnp.zeros((1, 4, 1), dtype=jnp.float32),
        jnp.zeros((1, 4, 4, 4), dtype=jnp.float32),
        jnp.zeros((1, 1, 4, 4), dtype=jnp.float32),
        jnp.zeros((1, 4, 4), dtype=jnp.float32),
    )
    _, recurrent = model.apply(
        variables,
        hidden,
        inputs,
        jnp.zeros((inputs.shape[0],), dtype=bool),
        method=model.__call_rnn__,
    )

    np.testing.assert_allclose(recurrent, parallel, rtol=2e-5, atol=1e-6)


def test_mamba3_recurrent_diagonal_mode_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("MAMBA3_RECURRENT_DIAGONAL_MODE", raising=False)
    assert recurrent_diagonal_mode() == "current"

    monkeypatch.setenv("MAMBA3_RECURRENT_DIAGONAL_MODE", "legacy_double")
    assert recurrent_diagonal_mode() == "legacy_double"

    monkeypatch.setenv("MAMBA3_RECURRENT_DIAGONAL_MODE", "invalid")
    with pytest.raises(ValueError, match="MAMBA3_RECURRENT_DIAGONAL_MODE"):
        recurrent_diagonal_mode()


def test_mamba3_recurrent_angle_mode_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("MAMBA3_RECURRENT_ANGLE_MODE", raising=False)
    assert recurrent_angle_mode() == "wrapped"

    monkeypatch.setenv("MAMBA3_RECURRENT_ANGLE_MODE", "legacy_unbounded")
    assert recurrent_angle_mode() == "legacy_unbounded"

    monkeypatch.setenv("MAMBA3_RECURRENT_ANGLE_MODE", "invalid")
    with pytest.raises(ValueError, match="MAMBA3_RECURRENT_ANGLE_MODE"):
        recurrent_angle_mode()


def test_mamba3_contraction_precision_mode_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("MAMBA3_CONTRACTION_PRECISION", raising=False)
    assert mamba3_contraction_precision_mode() == "highest"

    monkeypatch.setenv("MAMBA3_CONTRACTION_PRECISION", "default")
    assert mamba3_contraction_precision_mode() == "default"

    monkeypatch.setenv("MAMBA3_CONTRACTION_PRECISION", "invalid")
    with pytest.raises(ValueError, match="MAMBA3_CONTRACTION_PRECISION"):
        mamba3_contraction_precision_mode()


@pytest.mark.parametrize("model_type", ("transformer", "nsa"))
def test_attention_book_carry_uses_book_head_geometry(model_type: str) -> None:
    base = {
        "model_type": model_type,
        "n_heads": 4,
        "head_dim": 16,
        "max_cache_len": 416,
    }
    book = {**base, "n_heads": 1, "head_dim": 503}
    if model_type == "nsa":
        base["block_size"] = 26
        book["block_size"] = 26
    pre, post = LobBookModel.initialize_carry(
        1,
        64,
        1,
        1,
        is_transformer=True,
        transformer_config=base,
        transformer_config_book=book,
        ssm_type=model_type,
    )
    assert pre[0][0].shape == (1, 1, 416, 503)
    assert post[0][0].shape == (1, 4, 416, 16)
