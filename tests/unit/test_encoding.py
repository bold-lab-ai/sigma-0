from base_model.encoding import DEFAULT_TOKEN_MODE, SUPPORTED_TOKEN_MODES, supported_token_modes
from base_model.encoding.encoding import Message_Tokenizer
from base_model.training.train_helpers import TIME_END_I, TIME_START_I


def test_default_token_mode_is_26tok() -> None:
    assert DEFAULT_TOKEN_MODE == "26tok"
    assert supported_token_modes() == SUPPORTED_TOKEN_MODES
    assert "26tok" in supported_token_modes()


def test_training_time_mask_uses_canonical_26tok_bounds() -> None:
    assert Message_Tokenizer.MSG_LEN == 26
    assert (TIME_START_I, TIME_END_I) == (11, 15)
    assert (TIME_START_I, TIME_END_I) == (
        Message_Tokenizer.TIME_START_I,
        Message_Tokenizer.TIME_END_I,
    )
