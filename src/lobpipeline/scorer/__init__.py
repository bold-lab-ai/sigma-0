"""The LOB-Bench scorer sigma-0 runs, and the record of how it was merged.

sigma-0's scorer used to live only on Lustre, and `lob_bench/scoring.py` recorded
that checkout's Git revision in every summary.  Recording a revision is only
worth anything if the revision resolves, and this one does not: the checkout sits
on `try-merge-daily`, a branch with no upstream, so the hash in every summary
written so far names a scorer to one filesystem and to nothing else.

Note that two directories in this repository are called ``lob_bench``.  The one
beside this file is the merged scorer and is never imported: it is run as a
program, and it keeps that name so that ``<benchmark root>/lob_bench/run_bench.py``
resolves for this tree exactly as it does for the external checkout.  The other,
``lobpipeline/lob_bench``, is the adapter that builds that command line.

What is committed here is not a copy of either side.  It is the three-way merge
of that checkout with Sat-A/lob_bench against their common ancestor on
`peernagy/lob_bench`, recorded in ``merge.json``: base, both sides, what the
merge brought, what was resolved to which side, and what was deliberately not
taken.  Every production module is byte-identical to the checkout's, so the
merge added tests, scripts and documentation without moving a scored number.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCORER_ROOT = Path(__file__).resolve().parent
MERGE_RECORD_PATH = SCORER_ROOT / "merge.json"

# Directories and suffixes Python and pytest create inside any tree they touch.
# They are excluded because the hash has to be a property of the source, not of
# whether anything has imported it yet: importing the package writes .pyc files
# beside it, and a hash that counted those would turn "somebody ran the tests"
# into "this is no longer the merge it claims to be".
_GENERATED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ipynb_checkpoints"})
_GENERATED_SUFFIXES = frozenset({".pyc", ".pyo"})


def read_merge_record(name: str = "lob_bench") -> dict[str, Any]:
    """Return one merged tree's record, or say which names exist."""
    payload = json.loads(MERGE_RECORD_PATH.read_text(encoding="utf-8"))
    entries = {key: value for key, value in payload.items() if key != "contract"}
    try:
        return entries[name]
    except KeyError:
        raise KeyError(
            f"no merged tree named {name!r}; recorded trees: "
            f"{', '.join(sorted(entries))}"
        ) from None


def source_files(root: str | Path) -> list[Path]:
    """Return a tree's source files, in a stable order, without build output."""
    base = Path(root).resolve()
    found = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if _GENERATED_DIRS.intersection(relative.parts):
            continue
        if path.suffix in _GENERATED_SUFFIXES:
            continue
        found.append(path)
    return found


def tree_sha256(root: str | Path) -> str:
    """Hash a directory's source contents and layout, independent of the host.

    Paths are hashed alongside their bytes, so a file that moved is a different
    tree even when every byte is still present.
    """
    base = Path(root).resolve()
    digest = hashlib.sha256()
    for path in source_files(base):
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def merged_tree(name: str = "lob_bench") -> Path:
    """Return the merged tree's path, checked against its record.

    The check runs on every call rather than on demand: a caller about to hand
    this path to a scorer is exactly the caller who must not be handed a tree
    that has been edited since the merge was recorded.
    """
    record = read_merge_record(name)
    tree = SCORER_ROOT / record["path"]
    if not tree.is_dir():
        raise NotADirectoryError(f"merged tree missing: {tree}")

    count = len(source_files(tree))
    if count != record["file_count"]:
        raise ValueError(
            f"{tree}: recorded at {record['file_count']} files, found {count}"
        )
    actual = tree_sha256(tree)
    if actual != record["tree_sha256"]:
        raise ValueError(
            f"{tree}: recorded tree_sha256 {record['tree_sha256']}, found {actual}; "
            f"this is no longer the merge of {record['ours']['commit'][:8]} "
            f"and {record['theirs']['commit'][:8]}"
        )
    return tree


def merged_benchmark_root(name: str = "lob_bench") -> Path:
    """Return the ``--benchmark-root`` that drives the merged scorer.

    The scorer is addressed as ``<root>/lob_bench/run_bench.py``, so the root is
    the directory holding the merged tree rather than the tree itself.
    """
    merged_tree(name)
    return SCORER_ROOT


def recorded_revision(root: str | Path) -> str | None:
    """Return the identity of a merged scorer root, if it is one.

    A merge has two parents and no single upstream commit, so the honest
    identifier is the hash of the tree itself.  Unlike a checkout's ``HEAD`` it
    can be recomputed anywhere from the files it names, which is the property the
    Lustre checkout's revision does not have.
    """
    record_path = Path(root).expanduser().resolve() / MERGE_RECORD_PATH.name
    if not record_path.is_file():
        return None
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        if key == "contract" or not isinstance(value, dict):
            continue
        if "tree_sha256" in value:
            return str(value["tree_sha256"])
    return None


__all__ = [
    "MERGE_RECORD_PATH",
    "SCORER_ROOT",
    "merged_benchmark_root",
    "merged_tree",
    "read_merge_record",
    "recorded_revision",
    "source_files",
    "tree_sha256",
]
