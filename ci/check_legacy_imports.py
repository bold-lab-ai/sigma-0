"""Refuse new imports from the retired ``s5``, ``lob`` and root-module locations.

The model stack lives in ``base_model``. The ``s5`` and ``lob`` packages and the
root modules ``preproc``, ``utils`` and ``mamba3_legacy_norm`` no longer exist;
an import of one of them is a stale rebase, not a choice. Standard library
only; runs in seconds.

Usage:
    python ci/check_legacy_imports.py              # scan src/ run/ tests/ tools/ ci/
    python ci/check_legacy_imports.py --self-test  # prove it catches a planted import
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "run", "tests", "tools", "ci")
LEGACY_ROOTS = ("s5", "lob", "preproc", "utils", "mamba3_legacy_norm", "m3_kernels", "nsa_kernels")

# Evidence-locked files: their bytes are hashed by ci/evidence/*.json, so their
# imports change only together with a re-record on the cluster (PR 4).
ALLOWED = {
    # this checker names the roots it forbids
    "ci/check_legacy_imports.py",
}

_ROOTS = "|".join(LEGACY_ROOTS)
PATTERN = re.compile(rf"^\s*(?:from\s+(?:{_ROOTS})(?:\.[\w.]*)?\s+import\b|import\s+(?:{_ROOTS})(?:\.[\w.]*)?(?:\s+as\s+\w+)?\s*(?:#.*)?$)")


def scan(root: Path, allowed: set[str]) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(root).as_posix()
            if rel in allowed:
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if PATTERN.match(line):
                    hits.append((rel, n, line.strip()))
    return hits


def self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src" / "pkg").mkdir(parents=True)
        (root / "src" / "pkg" / "clean.py").write_text("from base_model.encoding import Vocab\nimport base_model.models.mamba3 as m\n")
        (root / "src" / "pkg" / "dirty.py").write_text("import numpy\nfrom lob.encoding import Vocab\n    import s5.mamba3 as m\nimport preproc\nfrom utils import debug\n")
        (root / "src" / "pkg" / "allowed.py").write_text("from lob import train_helpers\n")
        hits = scan(root, {"src/pkg/allowed.py"})
        want = {("src/pkg/dirty.py", 2), ("src/pkg/dirty.py", 3), ("src/pkg/dirty.py", 4), ("src/pkg/dirty.py", 5)}
        got = {(h[0], h[1]) for h in hits}
        ok = got == want
        print(f"  {'ok' if ok else 'WRONG'}  planted imports caught: {sorted(got)}")
        # a name that merely starts with a legacy root must not trip it
        (root / "src" / "pkg" / "near.py").write_text("import lobpipeline.model_store\nfrom utilsx import y\nimport s5x\n")
        near = [h for h in scan(root, set()) if h[0].endswith("near.py")]
        ok2 = not near
        print(f"  {'ok' if ok2 else 'WRONG'}  prefixes lobpipeline/utilsx/s5x not flagged")
    if ok and ok2:
        print("self-test passed: the guard catches planted legacy imports and nothing else")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    hits = scan(REPO, ALLOWED)
    for rel, n, line in hits:
        print(f"  {rel}:{n}: {line}")
    if hits:
        print(f"{len(hits)} legacy import(s). Import from base_model instead (docs/architecture.md has the map).")
        return 1
    print(f"no legacy imports outside the {len(ALLOWED)} allowed files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
