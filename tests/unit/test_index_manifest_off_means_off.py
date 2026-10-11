"""The sample-index manifest must not grow fields for features that are off.

``sample_indices_rank0.json`` is hashed into the do(Q) injection arm digest
(``lobpipeline/injection_verification.py``, ``list_rel_files``). A field added
unconditionally therefore changes the digest of every rollout, including ones
that never asked for the feature, and breaks a byte-identity claim that was
measured on one specific node and cannot be re-measured anywhere else
(defect D-X7).

The failure is quiet in the worst way: generation is unaffected, results look
normal, and only the injection audit notices -- by which point the natural
reading is "the audit is being fussy" rather than "the manifest changed".

This test pins the manifest's shape when nothing optional is requested. Adding a
field here is not forbidden; adding it *unconditionally* is. A new optional
feature attaches its fields inside its own branch, the way --window_starts_file
and the --inject_* flags do.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
INFERENCE = REPO / "run" / "base_model" / "runtime" / "inference.py"

# What a run writes when it asks for nothing optional. Order-independent.
BASELINE_KEYS = {
    "selection_seed",
    "sample_indices_file",
    "dataset_length",
    "requested_sequences",
    "world_size",
    "rank",
    "all_indices_sha256",
    "all_indices",
    "rank_indices",
}


def _manifest_block() -> str:
    """The manifest construction, lifted out of a module that needs jax to import."""
    src = INFERENCE.read_text()
    try:
        i = src.index("    index_manifest = {")
        j = src.index("    index_path = ", i)
    except ValueError:  # pragma: no cover - only if the block is renamed
        pytest.fail("could not locate the index_manifest block in inference.py")
    return textwrap.dedent(src[i:j])


class _Args:
    sample_indices_file = None
    window_starts_file = None


def _build(explicit_windows):
    ns = {
        "run_args": _Args,
        "explicit_windows": explicit_windows,
        "hashlib": hashlib,
        "json": json,
        "ds": list(range(1234)),
        "n_total": 4,
        "world_size": 1,
        "rank": 0,
        "all_indices": [0, 1, 2, 3],
        "rank_indices": [0, 1, 2, 3],
        "index_bytes": json.dumps([0, 1, 2, 3], separators=(",", ":")).encode(),
    }
    exec(_manifest_block(), ns)  # noqa: S102 - executing repo source, by design
    return ns["index_manifest"]


def test_manifest_gains_nothing_when_no_optional_feature_is_requested():
    assert set(_build(None)) == BASELINE_KEYS


def test_manifest_is_stable_byte_for_byte():
    """Key set alone is not enough: a changed value moves the digest too."""
    m = _build(None)
    payload = json.dumps(m, sort_keys=True, separators=(",", ":")).encode()
    # Recomputed from the same fixture inputs, so this pins values as well as keys.
    assert hashlib.sha256(payload).hexdigest() == (
        "73b6afe44555aa3b742b6a39f1951db4ee9e54a0efd1458bad642d8fe08c63b5")


def test_event_anchored_fields_appear_only_when_the_feature_is_on():
    windows = np.asarray([[0, 10], [1, 20], [0, 30], [1, 40]], dtype=np.int64)
    on = _build(windows)
    assert set(on) - BASELINE_KEYS == {"window_starts_file", "event_anchored", "anchor_windows"}
    assert on["event_anchored"] is True
    assert on["anchor_windows"] == windows[[0, 1, 2, 3]].tolist()


def test_selection_seed_is_unchanged_for_a_plain_run():
    """The seed is part of the manifest, so widening its condition must not move it."""
    assert _build(None)["selection_seed"] == 42
