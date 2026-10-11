"""The token mode has to reach the thing that builds the tensors.

This is not about the selector working -- `test_token_mode_selector.py` covers
that -- but about the five separate places along the launch path that pinned 26
tokens independently of it. Four of them failed silently: the batch script
printed `token_mode: 24tok` while `node_wrapper.sh` passed `--token_mode=26tok`
literally, the dataset fell back to its own default, the model was built to match
the dataset, and a 61-step run finished and wrote a checkpoint recording 26tok.
The only thing that gave it away was the tokens-per-step arithmetic.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def _text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_the_node_wrapper_forwards_the_mode_it_was_given():
    """It used to pass the literal string, which is how a 24tok job trained 26."""
    line = [l for l in _text("run/base_model/node_wrapper.sh").splitlines()
            if "--token_mode" in l]
    assert line, "node_wrapper.sh no longer passes --token_mode at all"
    assert "TOKEN_MODE" in line[0], (
        f"node_wrapper.sh pins the mode instead of forwarding it: {line[0].strip()}")


def test_the_batch_script_forwards_the_mode_to_the_step():
    """The export was honoured but the srun line overrode it, which is worse:
    the script printed the requested mode and ran another one."""
    text = _text("run/base_model/train_full_autoreg.batch")
    assert 'TOKEN_MODE="26tok"' not in text, (
        "the batch script hardcodes TOKEN_MODE on the line that passes it to srun; "
        "it will print the requested mode and run 26tok")


def test_no_argparse_choice_list_admits_only_one_mode():
    for rel in ("run/base_model/runtime/train.py", "run/base_model/runtime/inference.py"):
        text = _text(rel)
        for match in re.finditer(r"--token_mode.*?\)", text, re.S):
            block = match.group(0)
            assert "choices=['26tok']" not in block and 'choices=["26tok"]' not in block, (
                f"{rel} accepts only 26tok, so any other mode dies at the parser")


def test_the_dataset_selects_the_encoding_it_was_told_to_use():
    """Storing token_mode on the object is not selecting it; the module-level
    import gives whichever encoding is active when the module loads."""
    text = _text("src/base_model/data/lobster_dataloader.py")
    tree = ast.parse(text)
    setup = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "setup"), None)
    assert setup is not None, "LOBSTER dataset has no setup()"
    body = ast.dump(setup)
    assert "set_token_mode" in body, (
        "setup() does not select the encoding, so the dataset tokenises with "
        "whatever mode happened to be active")


def test_training_asserts_the_dataset_agrees_with_the_requested_mode():
    """The check that would have caught all four of the above in one line."""
    text = _text("src/base_model/training/train.py")
    assert "_tokens_per_message" in text and "did not reach the tokenizer" in text, (
        "train.py does not verify that the dataset produced the requested number "
        "of tokens per message; a mode that never arrives will train silently")


@pytest.mark.parametrize("mode,width", [("22tok", 22), ("23tok", 23),
                                        ("24tok", 24), ("26tok", 26)])
def test_the_batch_script_knows_every_supported_width(mode, width):
    text = _text("run/base_model/train_full_autoreg.batch")
    assert f"{mode}) TOKENS_PER_MSG={width}" in text, (
        f"the batch script cannot size {mode}, so its context arithmetic is wrong")
