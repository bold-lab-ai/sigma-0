"""Resolve a blessed checkpoint by name, under this repo's pin contract.

Models are treated the way data already is: `models/` is a symlink to the
shared store (like `data/SP500-2022-2025`), and the table that says what is in
it lives in the repository (like `tools/data_symlinks.tsv`). The table here is
`model_store_pins.json`, one entry per checkpoint::

    from lobpipeline.model_store import resolve
    m = resolve("m3-megacap8-8p2m-msgonly")
    m["path"], m["step"]      # -> models/<name>/checkpoint, 40760
    m["use_book_data"]        # -> False; this one is message-only

Two properties, both deliberate, both matching
`post_training/heuristic_learning/checkpoints.py`:

* **The step is pinned, never discovered.** Nothing here reads a checkpoint
  directory, and no entry means "newest". A later step is not a better step —
  it is a different model, and a number measured on one does not transfer.
* **The pin carries a SHA-256** of that step's `metadata/metadata`, so
  `verify()` can say whether the thing on disk is still the thing the entry
  describes. The same contract, extended to the checkpoints that were only
  reachable by knowing a path.

What the entries buy over a path: the paths live in a rescue mirror with
unguessable directory names, the blessed step is not the last one on disk for
several runs, and the fields record the traps — `use_book_data` (message-only
vs book+messages at the same size), `mamba3_legacy_norm` ("verify" means
pre-2026-04-22 weights: check restore parity before trusting generation), and
the two distinct 78M models at step 28730 (`m3-goog-78m-u6gb` is GOOG-only,
`m3-sp500-78m-s42` is the SP500 ladder point).

Stdlib only, so this imports on a login node without jax.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ModelStoreError(RuntimeError):
    """Raised when the store, the pin table, or a requested model is unusable."""


def _repo_root() -> Path:
    # src/lobpipeline/model_store.py -> src/lobpipeline -> src -> repo
    return Path(__file__).resolve().parent.parent.parent


def pins_path() -> Path:
    return Path(__file__).resolve().parent / "model_store_pins.json"


def store_path(repo_root: str | Path | None = None) -> Path:
    """The `models/` symlink: the payload side, absent in a clone off-cluster."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    return root / "models"


def load_pins() -> dict[str, dict[str, Any]]:
    path = pins_path()
    if not path.is_file():
        raise ModelStoreError(f"pin table not found at {path}")
    models = json.loads(path.read_text()).get("models")
    if not isinstance(models, dict) or not models:
        raise ModelStoreError(f"pin table at {path} lists no models")
    return models


def list_models() -> list[str]:
    return sorted(load_pins())


def resolve(name: str, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return the pin for `name` with `path` filled in from the store.

    `path` is the orbax manager root; the checkpoint to load is
    `path/<step>`. Raises if the name is unknown, or if the store is not
    reachable from this account.
    """
    pins = load_pins()
    if name not in pins:
        near = sorted(m for m in pins if name.lower() in m.lower())
        hint = f" (close: {', '.join(near[:5])})" if near else ""
        raise ModelStoreError(f"unknown model {name!r}{hint}; see list_models()")

    entry = dict(pins[name])
    store = store_path(repo_root)
    checkpoint = store / name / "checkpoint"
    if not checkpoint.exists():
        raise ModelStoreError(
            f"{name} is pinned but its payload is not reachable at {checkpoint} "
            "— is the models/ symlink present and the shared store readable "
            "from this account?"
        )
    entry["name"] = name
    entry["path"] = str(checkpoint)
    return entry


def metadata_path(name: str, repo_root: str | Path | None = None) -> Path:
    entry = resolve(name, repo_root)
    return Path(entry["path"]) / str(entry["step"]) / "metadata" / "metadata"


def verify(name: str, repo_root: str | Path | None = None) -> bool:
    """True iff the pinned step's metadata still hashes to the recorded digest.

    Reads one file at a known path. It does not enumerate anything.
    """
    entry = resolve(name, repo_root)
    meta = metadata_path(name, repo_root)
    if not meta.is_file():
        raise ModelStoreError(f"{name}: pinned step {entry['step']} has no metadata at {meta}")
    return hashlib.sha256(meta.read_bytes()).hexdigest() == entry["metadata_sha256"]


def legacy_norm_required(name: str) -> bool:
    """True iff this checkpoint's weights were trained under the legacy epsilon.

    `flax.linen.RMSNorm`'s first positional argument is `epsilon`, not the
    feature count as in PyTorch, so Mamba3 built before 2026-04-22 trained with
    epsilon equal to `d_state` or `d_inner` and its weights only mean what they
    meant under that normalizer. Evaluating them at the corrected `1e-6` leaves
    every normalized tensor up to `sqrt(d)` too large.

    Reads the pin only; it never opens the checkpoint. Unknown names raise
    rather than defaulting, because defaulting is exactly how the model-zoo
    benchmark came to score two checkpoints under a normalizer they had never
    seen.
    """
    pins = load_pins()
    if name not in pins:
        raise ModelStoreError(f"unknown model {name!r}; see list_models()")
    flag = pins[name].get("mamba3_legacy_norm")
    # "verify" is the pins file's word for pre-2026-04-22 weights.
    return flag is True or flag == "verify"


def inference_env(name: str) -> dict[str, str]:
    """The environment a run of this checkpoint needs, as a dict of strings.

    Today that is the RMSNorm epsilon regime; `run/base_model/runtime/
    inference.py` honours `MAMBA3_LEGACY_NORM` and does nothing when it is
    absent, so an empty value here is the correct instruction for a checkpoint
    trained after the fix.
    """
    return {"MAMBA3_LEGACY_NORM": "1" if legacy_norm_required(name) else ""}


def _cli() -> int:
    """`python -m lobpipeline.model_store env <name>` for shell callers.

    Batch scripts cannot import this module's caller stack, so they ask for the
    assignments and `eval` them. Printing `NAME=` for the fixed-epsilon case is
    deliberate: it unsets a value inherited from a previous model in the same
    shell.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="lobpipeline.model_store")
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("env", help="print shell assignments for one model")
    env.add_argument("name")
    sub.add_parser("list", help="list pinned model names")
    args = parser.parse_args()

    if args.command == "list":
        for name in list_models():
            print(name)
        return 0
    for key, value in inference_env(args.name).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
