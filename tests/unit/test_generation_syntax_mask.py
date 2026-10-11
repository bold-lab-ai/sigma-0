import numpy as np

from base_model.inference import validation as valh
from base_model.encoding.encoding import Vocab


def test_mamba3_mask_blocks_start_in_generated_time_positions() -> None:
    """The recurrent/Mamba3 path must not sample START inside time fields."""
    vocab = Vocab()
    mask = np.asarray(
        valh.syntax_validation_matrix(vocab, block_start_tok=False)
    )

    assert not mask[:, vocab.START_TOK].any()
    for field in ("delta_t_s", "delta_t_ns", "time_s", "time_ns"):
        start, end = valh.get_idx_from_field(field)
        assert not mask[start:end, vocab.START_TOK].any()
        assert mask[start:end].any(axis=1).all()


def test_legacy_recurrent_mask_can_reproduce_unblocked_start(
    monkeypatch,
) -> None:
    vocab = Vocab()
    monkeypatch.setenv("MAMBA3_START_TOKEN_MODE", "legacy_recurrent")

    recurrent_mask = np.asarray(
        valh.syntax_validation_matrix(vocab, block_start_tok=False)
    )
    attention_mask = np.asarray(
        valh.syntax_validation_matrix(vocab, block_start_tok=True)
    )

    assert recurrent_mask[:, vocab.START_TOK].any()
    assert not attention_mask[:, vocab.START_TOK].any()


def test_invalid_start_token_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("MAMBA3_START_TOKEN_MODE", "invalid")

    try:
        valh.syntax_validation_matrix(Vocab(), block_start_tok=False)
    except ValueError as exc:
        assert "MAMBA3_START_TOKEN_MODE" in str(exc)
    else:
        raise AssertionError("invalid START-token mode was accepted")
