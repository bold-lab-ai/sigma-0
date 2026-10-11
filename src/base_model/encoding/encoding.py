"""Runtime selector for the message encoding.

Until now this module *was* the 26-token tokenizer, byte-identical to
``encoding_26tok.py``, and the earlier ``TOKEN_MODE`` environment switch had been
removed because it routed every value to the same code -- a switch that lied. The
honest fix is not to re-add a lying switch but to make the selection real, which
is what this module now does.

Ten modules bind these names at import time (``from lob.encoding import
Message_Tokenizer``), so rebinding a module global after the fact would not reach
them. ``Message_Tokenizer`` and ``Vocab`` are therefore **proxies**: attribute
access and instantiation forward to whichever implementation is active, so a name
imported before the switch still sees the switch. Everything in this repo uses
these classes through class attributes (``MSG_LEN``, ``FIELDS``, ``TOK_LENS`` and
friends) plus two instantiations, all of which the proxy covers.

    from base_model.encoding.encoding import Message_Tokenizer, set_token_mode
    Message_Tokenizer.MSG_LEN        # 26
    set_token_mode("24tok")
    Message_Tokenizer.MSG_LEN        # 24, in every module that imported the name

The default is 26tok, so nothing changes for code that never calls
``set_token_mode``. Switching is a global, process-wide act: call it once, before
building a dataset or a model, and never mid-run -- a tokenizer change part-way
through would silently reinterpret data already in flight.
"""

from __future__ import annotations

import importlib
from typing import Any

SUPPORTED_TOKEN_MODES = ("22tok", "23tok", "24tok", "26tok")
DEFAULT_TOKEN_MODE = "26tok"

_active_mode = DEFAULT_TOKEN_MODE
_modules: dict[str, Any] = {}


def _module(mode: str):
    if mode not in SUPPORTED_TOKEN_MODES:
        raise ValueError(
            f"token_mode {mode!r} is not one of {SUPPORTED_TOKEN_MODES}")
    if mode not in _modules:
        _modules[mode] = importlib.import_module(f"base_model.encoding.encoding_{mode}")
    return _modules[mode]


def get_token_mode() -> str:
    return _active_mode


def supported_token_modes() -> tuple[str, ...]:
    """Return the encodings the runtime selector can switch to."""
    return SUPPORTED_TOKEN_MODES


def set_token_mode(mode: str) -> None:
    """Point every proxy at `mode`. Process-wide; call before building anything."""
    global _active_mode
    _module(mode)          # import first, so a bad name cannot leave a half state
    _active_mode = mode


class _Proxy(type):
    """Forward class attribute access and construction to the active module."""

    _target: str = ""

    def __getattr__(cls, name: str) -> Any:
        impl = getattr(_module(_active_mode), cls._target)
        try:
            return getattr(impl, name)
        except AttributeError:
            # Not every encoding defines every constant: 22tok has no
            # TIME_START_I/TIME_END_I, though its own comments record that they
            # would be 9 and 13. Say which mode is missing what, so the gap reads
            # as a gap rather than as a broken proxy.
            raise AttributeError(
                f"{cls._target} of {_active_mode} has no {name!r}; that encoding "
                "does not define it (26tok, 24tok and 23tok do)") from None

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return getattr(_module(_active_mode), cls._target)(*args, **kwargs)

    def __repr__(cls) -> str:
        return f"<{cls._target} of {_active_mode}>"


class Message_Tokenizer(metaclass=_Proxy):
    _target = "Message_Tokenizer"


class Vocab(metaclass=_Proxy):
    _target = "Vocab"


def encode_msgs(*args: Any, **kwargs: Any) -> Any:
    return _module(_active_mode).encode_msgs(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Anything else this module used to expose comes from the active encoding.

    Dunder probes (``__path__``, ``__wrapped__``, ...) come from the import
    machinery and introspection, not from callers; answering them would import
    the active tokenizer, and its pandas dependency, as a side effect of
    ``import base_model.encoding``.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return getattr(_module(_active_mode), name)
