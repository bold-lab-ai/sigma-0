"""The generator's vmap axes must agree with the generator's signature.

`jax.vmap` takes one `in_axes` entry per positional argument. Nothing in the
module raises when they drift apart: the mismatch surfaces at the first call
inside an allocation, as a ValueError about tuple lengths, after the model and
the data have been loaded. Adding an argument to `generate` and forgetting the
axis is exactly the mistake `do(Q)` injection could have introduced, so the
invariant is pinned here.

Parsed with `ast` rather than imported: the property is about the source, and
this way the test runs on a login node with no jax, no GPU and no checkpoint.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "base_model" / "inference" / "generate.py"
TREE = ast.parse(SRC.read_text())


def _func(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {SRC}")


def _vmap_in_axes(assigned_to: str) -> tuple[int, str]:
    """(len(in_axes), vmapped function name) for `<assigned_to> = jax.jit(jax.vmap(...))`."""
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == assigned_to for t in node.targets):
            continue
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (isinstance(func, ast.Attribute) and func.attr == "vmap"):
                continue
            fn_name = call.args[0].id
            for kw in call.keywords:
                if kw.arg == "in_axes":
                    return len(kw.value.elts), fn_name
            raise AssertionError(f"{assigned_to}: vmap without in_axes")
    raise AssertionError(f"{assigned_to} not found in {SRC}")


@pytest.mark.parametrize(
    "batched", ["generate_batched", "generate_batched_1tok"]
)
def test_vmap_axes_match_signature(batched: str) -> None:
    n_axes, fn_name = _vmap_in_axes(batched)
    n_params = len(_func(fn_name).args.args)
    assert n_axes == n_params, (
        f"{batched}: in_axes has {n_axes} entries but {fn_name} takes {n_params} "
        "positional arguments; vmap requires one entry per argument"
    )


def test_injection_is_off_by_default() -> None:
    """Every injection knob defaults to no injection."""
    gen = _func("generate")
    names = [a.arg for a in gen.args.args]
    assert names[-2:] == ["inject_flag_seq", "inject_msg_seq"]
    # the two injection slots are the last two defaults, both None
    for default in gen.args.defaults[-2:]:
        assert isinstance(default, ast.Constant) and default.value is None

    sample_new = _func("sample_new")
    kwargs = dict(
        zip(
            [a.arg for a in sample_new.args.args][-len(sample_new.args.defaults):],
            sample_new.args.defaults,
        )
    )
    for knob in (
        "inject_step",
        "inject_event_type",
        "inject_side",
        "inject_qty",
        "inject_price_offset_ticks",
    ):
        assert knob in kwargs, f"sample_new lost the {knob} argument"
    assert isinstance(kwargs["inject_step"], ast.Constant)
    assert kwargs["inject_step"].value is None, "injection must default to off"


def test_1tok_generator_has_no_injection_slots() -> None:
    """The 1tok path has no injection point; its signature must not pretend to."""
    names = [a.arg for a in _func("generate_1tok").args.args]
    assert not [n for n in names if n.startswith("inject")]
