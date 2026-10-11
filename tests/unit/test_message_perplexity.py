"""Message-level perplexity: the maths, and the invariance that is the whole point.

`lob.message_perplexity` is deliberately dependency-free, so it is imported here
directly. The encodings are **not** imported -- that would pull in jax, which the CI
runner does not have -- their FIELDS and TOK_LENS are read out of the source with
`ast` instead. That keeps the drift check real: change an encoding's field list and
these tests see it.
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from base_model.evaluation.message_perplexity import (  # noqa: E402
    ROLES,
    classify_field,
    message_perplexity,
    message_perplexity_from_per_field,
)

ENCODINGS = ("22tok", "23tok", "24tok", "26tok")
ARTIFACTS = REPO / "docs" / "experiments" / "trainloss_profiling_20260826" / "data"

# Token counts each role must come to, per encoding. Written out rather than
# recomputed so that an encoding change has to be acknowledged here.
EXPECTED_WIDTHS = {
    "22tok": {"main": 9, "absolute_time": 5, "reference": 8},
    "23tok": {"main": 9, "absolute_time": 5, "reference": 9},
    "24tok": {"main": 10, "absolute_time": 5, "reference": 9},
    "26tok": {"main": 11, "absolute_time": 5, "reference": 10},
}


def _read_class_attrs(mode: str) -> dict[str, list]:
    """Pull FIELDS and TOK_LENS out of an encoding module without importing it."""
    src = (REPO / "src" / "base_model" / "encoding" / f"encoding_{mode}.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: dict[str, list] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "Message_Tokenizer"):
            continue
        for stmt in node.body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == "FIELDS":
                out["FIELDS"] = list(ast.literal_eval(stmt.value))
            elif target.id == "TOK_LENS":
                # TOK_LENS = np.array((...)); the tuple is the sole call argument
                value = stmt.value
                if isinstance(value, ast.Call) and value.args:
                    value = value.args[0]
                out["TOK_LENS"] = [int(v) for v in ast.literal_eval(value)]
    missing = {"FIELDS", "TOK_LENS"} - set(out)
    if missing:
        raise AssertionError(f"encoding_{mode}.py: could not read {sorted(missing)}")
    return out


def _widths(mode: str) -> dict[str, int]:
    attrs = _read_class_attrs(mode)
    fields, lens = attrs["FIELDS"], attrs["TOK_LENS"]
    assert len(fields) == len(lens), f"encoding_{mode}: FIELDS/TOK_LENS length mismatch"
    return dict(zip(fields, lens))


# --------------------------------------------------------------------------
# the partition


@pytest.mark.parametrize("mode", ENCODINGS)
def test_role_partition_covers_the_message_exactly(mode):
    widths = _widths(mode)
    got = {role: 0 for role in ROLES}
    for name, w in widths.items():
        got[classify_field(name)] += w
    assert got == EXPECTED_WIDTHS[mode], f"{mode}: {got}"
    assert sum(got.values()) == int(mode.removesuffix("tok")), \
        f"{mode}: roles cover {sum(got.values())} tokens, not {mode}"


@pytest.mark.parametrize("mode", ENCODINGS)
def test_every_encoding_has_exactly_four_reference_fields(mode):
    refs = [f for f in _widths(mode) if classify_field(f) == "reference"]
    assert len(refs) == 4, f"{mode}: {refs}"


# --------------------------------------------------------------------------
# the invariance that justifies the metric


def _synthetic_table(mode: str) -> dict[str, float]:
    """Spread one fixed semantic budget across whatever fields `mode` happens to use.

    The budget is the same message content every time; only the encoding's way of
    slicing it changes. A tokenization-invariant metric must return the same numbers.
    """
    budget = {
        "event": 1.10,       # event type and side, however the encoding splits them
        "price": 2.25,
        "size": 1.04,
        "delta_t": 9.72,
        "abs_time": 0.47,
        "reference": 6.27,
    }
    fields = list(_widths(mode))
    table: dict[str, float] = {}

    def spread(total: float, names: list[str]) -> None:
        assert names, "no field to carry this part of the budget"
        # uneven on purpose: an implementation that only sums correctly for equal
        # splits would pass an even split by accident
        weights = [i + 1 for i in range(len(names))]
        scale = total / sum(weights)
        for name, w in zip(names, weights):
            table[name] = w * scale

    spread(budget["event"], [f for f in fields if f in ("event_type", "direction", "event_dir")])
    spread(budget["price"], ["price"])
    spread(budget["size"], ["size"])
    spread(budget["delta_t"], [f for f in fields if f.startswith("delta_t")])
    spread(budget["abs_time"], [f for f in fields if f in ("time_s", "time_ns")])
    spread(budget["reference"], [f for f in fields if f.endswith("_ref")])
    assert set(table) == set(fields), f"{mode}: uncovered fields {set(fields) ^ set(table)}"
    return table


def test_the_metric_is_invariant_across_tokenizations():
    results = {
        mode: message_perplexity(
            _synthetic_table(mode), token_mode=mode, field_widths=_widths(mode))
        for mode in ENCODINGS
    }
    mains = {m: r.main_nats for m, r in results.items()}
    fulls = {m: r.full_nats for m, r in results.items()}
    reference = results["26tok"]
    for mode, r in results.items():
        assert math.isclose(r.main_nats, reference.main_nats, rel_tol=1e-12), \
            f"main differs across encodings: {mains}"
        assert math.isclose(r.full_nats, reference.full_nats, rel_tol=1e-12), \
            f"full differs across encodings: {fulls}"
    # and the token counts genuinely do differ, so the invariance is not vacuous
    assert len({r.n_tokens["full"] for r in results.values()}) == len(ENCODINGS)


# --------------------------------------------------------------------------
# the arithmetic


def test_components_sum_to_full_and_full_dominates_main():
    for mode in ENCODINGS:
        r = message_perplexity(_synthetic_table(mode), token_mode=mode)
        assert math.isclose(sum(r.components[k] for k in ROLES), r.full_nats, rel_tol=1e-12)
        assert r.full_nats >= r.main_nats
        d = r.as_dict()
        assert math.isclose(
            d["variants"]["full"]["bits_per_message"],
            d["variants"]["full"]["nats_per_message"] / math.log(2), rel_tol=1e-12)


def test_scored_fields_are_recorded_so_two_numbers_can_be_checked_for_comparability():
    r = message_perplexity(_synthetic_table("26tok"), token_mode="26tok",
                           field_widths=_widths("26tok"))
    d = r.as_dict()
    assert d["variants"]["main"]["fields"], "main variant must name what it scored"
    assert not any(f.endswith("_ref") for f in d["variants"]["main"]["fields"])
    assert "time_s" not in d["variants"]["main"]["fields"]
    assert set(d["variants"]["full"]["fields"]) == set(_widths("26tok"))


def test_width_disagreement_is_an_error_not_a_silent_wrong_number():
    table = _synthetic_table("26tok")
    widths = _widths("26tok")
    widths.pop("price")
    with pytest.raises(KeyError):
        message_perplexity(table, token_mode="26tok", field_widths=widths)


# --------------------------------------------------------------------------
# against the committed evidence


@pytest.mark.parametrize("label", ("heldout", "train_period"))
def test_committed_artifact_reproduces_its_own_aggregate(label):
    paths = sorted(ARTIFACTS.glob(f"*_{label}.json"))
    if not paths:
        pytest.skip(f"no committed {label} ce-by-position record")
    record = json.loads(paths[0].read_text())
    r = message_perplexity_from_per_field(
        record["per_field"], token_mode="26tok",
        n_messages=record["slice"]["messages"])
    # the record sums per-position means; this sums per-field means times widths.
    # Two paths over float32 evidence, so compare relatively.
    assert math.isclose(r.full_nats, record["aggregate"]["ce_per_message_nats"],
                        rel_tol=1e-6)
    assert r.main_nats < r.full_nats
    assert r.components["reference"] > 0.0


def test_ignore_times_is_recorded_not_silently_absorbed():
    """Under ignore_times the absolute-time fields never enter the loss, so `full`
    stops being the whole message. That has to show up in the record."""
    widths = _widths("26tok")
    table = {k: v for k, v in _synthetic_table("26tok").items()
             if k not in ("time_s", "time_ns")}
    r = message_perplexity(table, token_mode="26tok", field_widths=widths,
                           expected_fields=list(widths))
    d = r.as_dict()
    assert set(d["missing_fields"]) == {"time_s", "time_ns"}
    assert d["full_covers_whole_message"] is False
    assert r.components["absolute_time"] == 0.0
    # main is unaffected: it never scored those fields in the first place
    full_table = message_perplexity(_synthetic_table("26tok"), token_mode="26tok")
    assert math.isclose(r.main_nats, full_table.main_nats, rel_tol=1e-12)


def test_complete_table_reports_no_missing_fields():
    widths = _widths("26tok")
    r = message_perplexity(_synthetic_table("26tok"), token_mode="26tok",
                           field_widths=widths, expected_fields=list(widths))
    assert r.as_dict()["missing_fields"] == []
    assert r.as_dict()["full_covers_whole_message"] is True
