"""The token-mode selector actually selects, including for names already imported.

These tests need the encodings themselves, so they need jax and are skipped where
it is absent -- unlike the message-perplexity tests, which are deliberately
dependency-free. What is checked here cannot be checked without importing: that a
name bound by `from lob.encoding import X` before the switch still sees the switch
afterwards. That is the whole point of the proxy, and it is exactly what a
module-global rebind would fail to do.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

jax = pytest.importorskip("jax", reason="the encodings import jax")

from base_model.encoding.encoding import (  # noqa: E402
    DEFAULT_TOKEN_MODE, SUPPORTED_TOKEN_MODES, Message_Tokenizer, Vocab,
    get_token_mode, set_token_mode,
)

EXPECTED_MSG_LEN = {"22tok": 22, "23tok": 23, "24tok": 24, "26tok": 26}


@pytest.fixture(autouse=True)
def _restore_default():
    yield
    set_token_mode(DEFAULT_TOKEN_MODE)


def test_default_is_26tok_so_nothing_changes_for_callers_that_never_switch():
    assert get_token_mode() == "26tok"
    assert int(Message_Tokenizer.MSG_LEN) == 26


@pytest.mark.parametrize("mode", SUPPORTED_TOKEN_MODES)
def test_every_supported_mode_selects_its_own_tokenizer(mode):
    set_token_mode(mode)
    assert int(Message_Tokenizer.MSG_LEN) == EXPECTED_MSG_LEN[mode]
    assert int(sum(Message_Tokenizer.TOK_LENS)) == EXPECTED_MSG_LEN[mode]
    assert len(Message_Tokenizer.FIELDS) == len(Message_Tokenizer.TOK_LENS)


def test_a_name_imported_before_the_switch_still_follows_it():
    """The reason this is a proxy and not a module-global rebind."""
    from base_model.encoding.encoding import Message_Tokenizer as bound_early
    set_token_mode("26tok")
    assert int(bound_early.MSG_LEN) == 26
    set_token_mode("24tok")
    assert int(bound_early.MSG_LEN) == 24, \
        "a name bound before the switch did not follow it; the proxy is not working"


def test_downstream_modules_follow_too():
    """train_helpers and the inference modules read these lazily now."""
    import base_model.training.train_helpers as th
    import base_model.inference.generate as inf
    set_token_mode("26tok")
    assert (th.TIME_START_I, th.TIME_END_I) == (11, 15)
    assert inf.REF_LEN == 26 - 16
    set_token_mode("24tok")
    assert (th.TIME_START_I, th.TIME_END_I) == (10, 14), \
        "train_helpers kept the 26-token time columns after switching to 24tok"
    assert inf.REF_LEN == 24 - 15


def test_instantiation_goes_to_the_active_implementation():
    set_token_mode("24tok")
    assert type(Vocab()).__module__.endswith("encoding_24tok")
    set_token_mode("26tok")
    assert type(Vocab()).__module__.endswith("encoding_26tok")


def test_an_unsupported_mode_is_refused_and_leaves_the_state_alone():
    set_token_mode("26tok")
    with pytest.raises(ValueError):
        set_token_mode("99tok")
    assert get_token_mode() == "26tok"


def test_a_constant_one_encoding_lacks_names_the_encoding_not_the_proxy():
    """22tok defines no TIME_START_I; the error should say so."""
    set_token_mode("22tok")
    with pytest.raises(AttributeError, match="22tok"):
        Message_Tokenizer.TIME_START_I
