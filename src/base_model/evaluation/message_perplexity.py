"""Message-level perplexity: the one loss number that survives a change of tokenizer.

Why this exists
---------------
Everything in this repo reports **per-token** cross-entropy, and per-token CE cannot
be compared across tokenizations: 22/23/24/26-token encodings cut the same LOBSTER
message into different numbers of tokens, so the denominator differs and 0.60 nats
per token under one encoding is not 0.60 under another.

Per **message** it is comparable, and exactly so rather than approximately. By the
chain rule,

    -log P(message m | messages 1..m-1) = sum over the tokens of message m
                                          of -log P(token_i | everything before it),

and for a lossless bijective encoding that sum is a property of the message and the
model, not of the encoding. The 22/23/24/26-token encodings are lossless. (The 1tok
VQ encoding is **not**: -log P(token) omits the quantisation error, so it is a lower
bound on the message NLL and must not be ranked alongside the others.)

The two variants
----------------
``full``
    Every token of the message. This is -log P(entire message | past).

``main`` (the headline)
    The message's own new content, excluding the reference fields that point at a
    previously seen order. Concretely: event type, direction, price, size, and the
    inter-arrival time.

Two things about ``main`` are easy to get wrong, so they are spelled out here.

**It already is the absolute price and the absolute timestamp.** The encodings store
a *relative* price and a *delta* time, but both references are fixed by the past:
the price is quoted against the previous mid-price rounded to the tick
(``encoding_22tok.py``, ``p_ref = ((bid + ask) / 2).shift()``), and the timestamp is
the previous message's timestamp plus ``delta_t``. Conditioned on the past those maps
are bijections, so scoring the relative representation *is* scoring the absolute
field. No change to any encoding is needed to report "absolute price" and "absolute
time".

**Each field is paid for exactly once.** The encodings carry the message time twice,
as ``delta_t_*`` and again as absolute ``time_s``/``time_ns``, the second being a
deterministic function of the first and the past. Scoring both double-counts the time;
scoring neither throws the time away. ``main`` therefore scores ``delta_t_*`` and
excludes the absolute-time duplicate. Which half to keep is not arbitrary: every
26-token training run to date sets ``ignore_times=True``, so the absolute-time output
heads were never trained, and charging a model for them measures an artefact of the
training setup rather than the model.

The leftover is not discarded, it is reported: ``full = main + absolute_time +
reference``, an exact additive decomposition, and the per-field nats are kept so any
other subset can be read off later without re-running anything.

Units
-----
``nats_per_message`` is the primitive. ``bits_per_message`` is it over ln 2, and
``perplexity`` is its exponential -- the effective number of equally likely messages.
Message perplexity is a large number (a LOB message carries tens of bits), so the
bits form is usually the readable one; all three are reported because they are the
same measurement.

This module is deliberately dependency-free: it aggregates a per-field table, so the
notebook, the tests and the CI guard can all import it without JAX or a checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "REFERENCE_SUFFIX",
    "ABSOLUTE_TIME_FIELDS",
    "ROLES",
    "classify_field",
    "field_widths_from_tokenizer",
    "MessagePerplexity",
    "message_perplexity",
    "message_perplexity_from_per_field",
]

# A reference field points at an order that already exists in the book. Every
# encoding in this repo marks them with this suffix, and computes its own
# NEW_MSG_LEN the same way, so the suffix is the encodings' own convention
# rather than a rule invented here.
REFERENCE_SUFFIX = "_ref"

# The absolute timestamp, carried alongside delta_t and determined by it.
ABSOLUTE_TIME_FIELDS = frozenset({"time_s", "time_ns"})

ROLES = ("main", "absolute_time", "reference")


def classify_field(name: str) -> str:
    """Return which of ROLES a field belongs to.

    Derived from the field name because that is what every encoding module already
    keys on; nothing here hardcodes a token position, so a new encoding with the
    same field vocabulary works without touching this file.
    """
    if name.endswith(REFERENCE_SUFFIX):
        return "reference"
    if name in ABSOLUTE_TIME_FIELDS:
        return "absolute_time"
    return "main"


def field_widths_from_tokenizer(tokenizer: Any) -> dict[str, int]:
    """Map field name -> number of tokens, read off a Message_Tokenizer class.

    Uses the tokenizer's own FIELDS and TOK_LENS so 22/23/24/26tok all work, and so
    a mismatch between this module and an encoding shows up as a KeyError here
    rather than as a quietly wrong number downstream.
    """
    fields = tuple(tokenizer.FIELDS)
    widths = [int(w) for w in tokenizer.TOK_LENS]
    if len(fields) != len(widths):
        raise ValueError(
            f"tokenizer has {len(fields)} fields but {len(widths)} token widths")
    return dict(zip(fields, widths))


@dataclass(frozen=True)
class MessagePerplexity:
    """The two variants plus the decomposition that connects them."""

    token_mode: str
    n_messages: int | None
    # nats per message, by role; components sum to `full`
    components: dict[str, float]
    per_field_nats: dict[str, float]
    scored_fields: dict[str, tuple[str, ...]] = _dc_field(default_factory=dict)
    n_tokens: dict[str, int] = _dc_field(default_factory=dict)
    lossless: bool = True
    # Fields the encoding defines that the table did not carry. Under
    # ignore_times the absolute-time fields are absent from the loss entirely,
    # and then `full` is "every field that was scored", not "the whole message".
    # Recording this is the difference between a caveat and a wrong number.
    missing_fields: tuple[str, ...] = ()

    @property
    def main_nats(self) -> float:
        return self.components["main"]

    @property
    def full_nats(self) -> float:
        return sum(self.components[r] for r in ROLES)

    def as_dict(self) -> dict[str, Any]:
        def view(nats: float, role_key: str) -> dict[str, Any]:
            return {
                "nats_per_message": nats,
                "bits_per_message": nats / math.log(2.0),
                # exp() of a whole message's NLL overflows float for large messages;
                # report it as inf rather than raising, the bits form is the usable one
                "perplexity": math.exp(nats) if nats < 700.0 else math.inf,
                "n_tokens": self.n_tokens.get(role_key, 0),
                "fields": list(self.scored_fields.get(role_key, ())),
            }

        return {
            "token_mode": self.token_mode,
            "n_messages": self.n_messages,
            "lossless_encoding": self.lossless,
            "missing_fields": list(self.missing_fields),
            "full_covers_whole_message": not self.missing_fields,
            "variants": {
                "main": view(self.main_nats, "main"),
                "full": view(self.full_nats, "full"),
            },
            "components_nats_per_message": {
                role: self.components[role] for role in ROLES
            },
            "per_field_nats_per_message": dict(self.per_field_nats),
        }


def message_perplexity(
    per_field_nats: Mapping[str, float],
    *,
    token_mode: str,
    field_widths: Mapping[str, int] | None = None,
    n_messages: int | None = None,
    lossless: bool = True,
    expected_fields: Sequence[str] | None = None,
) -> MessagePerplexity:
    """Aggregate a per-field nats-per-message table into the two variants.

    `per_field_nats` maps a field name to the total nats that field costs **per
    message** (i.e. already multiplied by the field's token width, not a per-token
    mean). `field_widths` is optional and only used to report token counts.

    Pass `expected_fields` (the encoding's full FIELDS tuple) to have absent fields
    recorded. This matters in practice: every 26-token run sets ignore_times, which
    removes the absolute-time fields from the loss, and without this the `full`
    number silently stops being the whole message.
    """
    components = {role: 0.0 for role in ROLES}
    scored: dict[str, list[str]] = {role: [] for role in ROLES}
    n_tokens = {role: 0 for role in ROLES}

    for name, nats in per_field_nats.items():
        role = classify_field(name)
        components[role] += float(nats)
        scored[role].append(name)
        if field_widths is not None:
            if name not in field_widths:
                raise KeyError(
                    f"field {name!r} has no width in field_widths; the per-field "
                    "table and the tokenizer disagree")
            n_tokens[role] += int(field_widths[name])

    missing: tuple[str, ...] = ()
    if expected_fields is not None:
        missing = tuple(f for f in expected_fields if f not in per_field_nats)

    scored_fields = {role: tuple(scored[role]) for role in ROLES}
    scored_fields["full"] = tuple(
        name for role in ROLES for name in scored[role])
    n_tokens["full"] = sum(n_tokens[role] for role in ROLES)

    return MessagePerplexity(
        token_mode=token_mode,
        n_messages=n_messages,
        components=components,
        per_field_nats={k: float(v) for k, v in per_field_nats.items()},
        scored_fields=scored_fields,
        n_tokens=n_tokens,
        lossless=lossless,
        missing_fields=missing,
    )


def message_perplexity_from_per_field(
    per_field_rows: Iterable[Mapping[str, Any]],
    *,
    token_mode: str,
    n_messages: int | None = None,
    lossless: bool = True,
    expected_fields: Sequence[str] | None = None,
) -> MessagePerplexity:
    """Convenience wrapper for the `per_field` block of a ce-by-position record.

    Those rows carry a per-token mean `ce` and a `width`, so the nats a field costs
    per message is `ce * width` -- the same product whose sum the record already
    reports as `aggregate.ce_per_message_nats`.
    """
    per_field_nats: dict[str, float] = {}
    widths: dict[str, int] = {}
    for row in per_field_rows:
        name = str(row["field"])
        width = int(row["width"])
        per_field_nats[name] = float(row["ce"]) * width
        widths[name] = width
    return message_perplexity(
        per_field_nats,
        token_mode=token_mode,
        field_widths=widths,
        n_messages=n_messages,
        lossless=lossless,
        expected_fields=expected_fields,
    )
