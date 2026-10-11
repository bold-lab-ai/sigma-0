#!/usr/bin/env python3
"""Guards for the per-position CE decomposition.

Every curve this PR produces is labelled by two constants in the encoding
(FIELDS and TOK_LENS) and by code that slices with them. None of that is
covered by a type system or an existing test, and all of it can drift
silently: an encoding change relabels every curve without breaking anything,
and a private copy of the widths table is how relabeling sneaks in. Each
guard names the silent failure it exists to catch.

Standard library only; sources are parsed with ast or read as text, never
imported (the modules pull in jax/numpy, absent on this runner, and import
executes module code).

--self-test corrupts a copy of each guarded input and requires the matching
guard to fail, so a guard that has quietly stopped checking anything cannot
stay green.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ENCODING_MASTER = "src/base_model/encoding/encoding_26tok.py"
ENCODING_ACTIVE = "src/base_model/encoding/encoding.py"
OFFLINE = "run/base_model/evaluate_ce_by_position.py"
TRAIN = "src/base_model/training/train.py"
HELPERS = "src/base_model/training/train_helpers.py"
EVIDENCE = "ci/evidence/profiling.json"
NOTEBOOK = "docs/experiments/trainloss_profiling_20260826/entropy_floor_analysis.ipynb"
NOTEBOOK_DATA = "docs/experiments/trainloss_profiling_20260826/data"
ARTIFACT_DIR = NOTEBOOK_DATA  # the five ce-by-position records sit next to the CSVs

GUARDED_FILES = (ENCODING_MASTER, ENCODING_ACTIVE, OFFLINE, TRAIN, HELPERS)

# The widths tuple as source text: its literal presence anywhere outside the
# encoding module is a private fork of the table.
WIDTHS_LITERAL = "1, 1, 3, 2, 1, 3, 2, 3, 3, 2, 2, 3"


def _text(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def guard_encoding_table(root: Path) -> list[str]:
    """FIELDS/TOK_LENS agree with each other, sum to 26, and the active copy
    of the encoding is byte-identical to the master.

    Hole plugged: a changed width silently re-buckets positions into the
    wrong fields; every ce_field curve keeps its name and changes meaning.
    A diverged encoding.py does the same by making the imported module and
    the audited module different files.
    """
    problems: list[str] = []
    tree = ast.parse(_text(root, ENCODING_MASTER))
    fields = tok_lens = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "FIELDS":
                try:
                    fields = ast.literal_eval(node.value)
                except ValueError:
                    problems.append(f"{ENCODING_MASTER}: FIELDS is not a literal")
            if name == "TOK_LENS":
                # TOK_LENS = np.array((...)) -- take the call's first argument
                value = node.value
                if isinstance(value, ast.Call) and value.args:
                    value = value.args[0]
                try:
                    tok_lens = ast.literal_eval(value)
                except ValueError:
                    problems.append(f"{ENCODING_MASTER}: TOK_LENS is not a literal")
    if fields is None or tok_lens is None:
        problems.append(f"{ENCODING_MASTER}: FIELDS or TOK_LENS assignment not found")
        return problems
    if len(fields) != len(tok_lens):
        problems.append(f"{ENCODING_MASTER}: {len(fields)} fields but "
                        f"{len(tok_lens)} widths")
    if sum(tok_lens) != 26:
        problems.append(f"{ENCODING_MASTER}: TOK_LENS sums to {sum(tok_lens)}, "
                        f"not 26; every per-position curve is relabelled")
    # This used to require encoding.py to be byte-identical to encoding_26tok.py.
    # encoding.py is a runtime selector now, so identity is the wrong test; what
    # the check was protecting -- that the imported encoding is the audited one --
    # becomes "the selector defaults to 26tok and lists it as supported".
    active = _text(root, ENCODING_ACTIVE)
    if 'DEFAULT_TOKEN_MODE = "26tok"' not in active:
        problems.append(f"{ENCODING_ACTIVE}: does not default to 26tok, so the "
                        f"imported encoding is no longer the audited one")
    if '"26tok"' not in active or "SUPPORTED_TOKEN_MODES" not in active:
        problems.append(f"{ENCODING_ACTIVE}: no SUPPORTED_TOKEN_MODES listing 26tok")
    if not problems:
        print(f"[guard] {ENCODING_MASTER}: {len(fields)} fields, widths sum "
              f"{sum(tok_lens)}; {ENCODING_ACTIVE} selects it by default")
    return problems


def guard_offline_uses_shared_constants(root: Path) -> list[str]:
    """The offline script slices with Message_Tokenizer constants.

    Hole plugged: a hardcoded widths tuple in the script keeps working after
    the encoding changes, silently mislabelling every JSON it writes.
    """
    problems: list[str] = []
    text = _text(root, OFFLINE)
    for needle in ("Message_Tokenizer.TOK_LENS", "Message_Tokenizer.FIELDS"):
        if needle not in text:
            problems.append(f"{OFFLINE}: no longer reads {needle}")
    if WIDTHS_LITERAL in text:
        problems.append(f"{OFFLINE}: carries a private copy of the widths table")
    if not problems:
        print(f"[guard] {OFFLINE}: slices via Message_Tokenizer constants, no fork")
    return problems


def guard_training_wiring(root: Path) -> list[str]:
    """The in-training decomposition is wired and also uses the shared table.

    Hole plugged: LOG_CE_BY_POSITION printing nothing is indistinguishable
    from a healthy default-off run; the env flag, the kwarg hand-off and the
    ce_field keys must all exist for a curve to ever appear.
    """
    problems: list[str] = []
    train_text = _text(root, TRAIN)
    helpers_text = _text(root, HELPERS)
    if "_LOG_CE_BY_POSITION" not in helpers_text:
        problems.append(f"{HELPERS}: LOG_CE_BY_POSITION flag is gone")
    if '"ce_by_pos"' not in helpers_text:
        problems.append(f"{HELPERS}: ce vector no longer handed to the callback")
    for needle in ("ce_field/", "ce_pos/", "Message_Tokenizer.TOK_LENS"):
        if needle not in train_text:
            problems.append(f"{TRAIN}: {needle} missing from the wandb callback")
    if WIDTHS_LITERAL in train_text:
        problems.append(f"{TRAIN}: carries a private copy of the widths table")
    if not problems:
        print(f"[guard] {TRAIN}+{HELPERS}: flag, hand-off and ce_field/ce_pos keys present")
    return problems


def guard_evidence(root: Path) -> list[str]:
    """Recorded evidence still describes the current sources.

    Hole plugged: numbers recorded on the cluster keep persuading after the
    code that produced them changes; a stale hash must fail, not persuade.
    Absence is allowed (and printed) because evidence lands only after the
    first cluster run.
    """
    problems: list[str] = []
    path = root / EVIDENCE
    if not path.is_file():
        print(f"[guard] {EVIDENCE}: not recorded yet (cluster evidence pending)")
        return problems
    doc = json.loads(path.read_text(encoding="utf-8"))
    sources = doc.get("sources") or {}
    if not sources:
        problems.append(f"{EVIDENCE}: declares no source files, so nothing "
                        f"stops it from going stale")
    for rel, recorded in sorted(sources.items()):
        target = root / rel
        if not target.is_file():
            problems.append(f"{EVIDENCE}: source {rel} does not exist")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != recorded:
            problems.append(f"{EVIDENCE}: {rel} changed since recording "
                            f"({actual[:12]} != {recorded[:12]})")
    if not problems:
        print(f"[guard] {EVIDENCE}: {len(sources)} source hashes hold")
    return problems


def guard_notebook(root: Path) -> list[str]:
    """The analysis notebook is committed executed, and reads only committed inputs.

    Hole plugged: GitHub renders .ipynb inline, so this notebook is a document
    rather than a file to download -- but only while its outputs travel with it.
    Stripping them changes nothing a reviewer can see except base64 churn, and the
    page then arrives as a list of empty headings. That failure cannot be caught by
    reading the diff, so it is caught here.

    The second half is provenance: the notebook derives its numbers from the two
    decomposition artifacts and the three inventory tables in data/. If it grew a
    path to somewhere untracked, its figures would stop being reproducible from a
    checkout without anyone noticing.
    """
    problems: list[str] = []
    path = root / NOTEBOOK
    if not path.is_file():
        problems.append(f"{NOTEBOOK}: missing")
        return problems

    doc = json.loads(path.read_text(encoding="utf-8"))
    cells = doc.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code" and c.get("source")]
    empty = [i for i, c in enumerate(cells)
             if c.get("cell_type") == "code" and c.get("source") and not c.get("outputs")]
    if empty:
        problems.append(f"{NOTEBOOK}: cells {empty} carry no output, so the notebook "
                        f"was committed unexecuted and renders as empty headings")
    raised = [(i, o.get("ename")) for i, c in enumerate(cells)
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if raised:
        problems.append(f"{NOTEBOOK}: cells raised: {raised}")

    figures = sum(1 for c in cells for o in c.get("outputs", [])
                  if o.get("output_type") == "display_data"
                  and "image/png" in o.get("data", {}))
    if figures < 8:
        problems.append(f"{NOTEBOOK}: {figures} figures in the committed outputs, expected 8")
    tables = sum(1 for c in cells for o in c.get("outputs", [])
                 if o.get("output_type") == "execute_result")
    if tables < 2:
        problems.append(f"{NOTEBOOK}: {tables} tables in the committed outputs, expected 2")

    source = "".join("".join(c.get("source", [])) for c in code_cells)
    # The notebook builds filenames from a label, so match the labels rather than the
    # full names: a guard that looks for a string the source never spells is a guard
    # that fails for the wrong reason.
    for label in ("train_period_v2", "valset_2025", "heldout_v2", "valset_2022",
                  "valset_2023", "valset_2024", "valset_2025h1", "valset_2025h2",
                  "valset_all"):
        if label not in source:
            problems.append(f"{NOTEBOOK}: never reads the {label} slice it claims to")
        if not (root / ARTIFACT_DIR).joinpath(
                f"j5749206_9o8um51n_5749206_{label}.json").is_file():
            problems.append(f"{NOTEBOOK}: the {label} artifact is not committed")
    for name in ("ruler_by_size.csv", "axis_fits.csv", "validation_sets_inventory.csv"):
        if name not in source:
            continue
        if not (root / NOTEBOOK_DATA / name).is_file():
            problems.append(f"{NOTEBOOK}: reads {name}, which is not committed under "
                            f"{NOTEBOOK_DATA}/")

    if not problems:
        print(f"[guard] {NOTEBOOK}: executed, {figures} figures, {tables} tables, "
              f"inputs all committed")
    return problems


GUARDS = (
    guard_encoding_table,
    guard_offline_uses_shared_constants,
    guard_training_wiring,
    guard_evidence,
    guard_notebook,
)


def run_guards(root: Path) -> int:
    problems: list[str] = []
    for guard in GUARDS:
        problems.extend(guard(root))
    if problems:
        print("\nFAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll ce-decomposition guards hold.")
    return 0


def _copy_tree(root: Path, troot: Path) -> None:
    for f in GUARDED_FILES:
        dest = troot / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / f, dest)
    for rel in (EVIDENCE, NOTEBOOK):
        src = root / rel
        if src.is_file():
            dest = troot / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    src_data = root / NOTEBOOK_DATA
    if src_data.is_dir():
        shutil.copytree(src_data, troot / NOTEBOOK_DATA, dirs_exist_ok=True)


def self_test(root: Path) -> int:
    failures: list[str] = []
    covered: set[str] = set()

    def expect(name: str, guard, mutate) -> None:
        covered.add(guard.__name__)
        with tempfile.TemporaryDirectory() as td:
            troot = Path(td)
            _copy_tree(root, troot)
            mutate(troot)
            if guard(troot):
                print(f"caught: {name}")
            else:
                failures.append(f"{name}: {guard.__name__} stayed green on a "
                                f"corrupted tree")

    def replace(rel: str, old: str, new: str):
        def _mut(troot: Path) -> None:
            target = troot / rel
            text = target.read_text(encoding="utf-8")
            if old not in text:
                raise SystemExit(f"self-test anchor {old!r} not found in {rel}; "
                                 f"the self-test itself is stale")
            target.write_text(text.replace(old, new), encoding="utf-8")
        return _mut

    expect("width changed", guard_encoding_table,
           replace(ENCODING_MASTER, WIDTHS_LITERAL,
                   "1, 1, 3, 2, 1, 3, 2, 3, 3, 2, 2, 4"))
    expect("offline table forked", guard_offline_uses_shared_constants,
           replace(OFFLINE, "Message_Tokenizer.TOK_LENS",
                   f"({WIDTHS_LITERAL})"))
    expect("wandb keys renamed", guard_training_wiring,
           replace(TRAIN, "ce_field/", "ce_fld/"))

    def stale_evidence(troot: Path) -> None:
        dest = troot / EVIDENCE
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            "kind": "profiling",
            "sources": {ENCODING_MASTER: "0" * 64},
        }), encoding="utf-8")
    expect("evidence gone stale", guard_evidence, stale_evidence)

    def strip_outputs(troot: Path) -> None:
        # The realistic corruption: `nbstrip`-style cleaning, or a merge that drops
        # outputs. Nothing else about the file changes, which is exactly why review
        # cannot see it.
        target = troot / NOTEBOOK
        doc = json.loads(target.read_text(encoding="utf-8"))
        for cell in doc["cells"]:
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
        target.write_text(json.dumps(doc), encoding="utf-8")
    expect("notebook outputs stripped", guard_notebook, strip_outputs)

    def drop_input(troot: Path) -> None:
        (troot / NOTEBOOK_DATA / "ruler_by_size.csv").unlink()
    expect("notebook input uncommitted", guard_notebook, drop_input)

    # The invariant worth holding is coverage, not a case count: a guard added
    # without a corruption case is a guard nobody has seen fail, and a count
    # assertion would go red every time a case is added instead.
    uncovered = sorted(g.__name__ for g in GUARDS if g.__name__ not in covered)
    for name in uncovered:
        failures.append(f"{name}: no self-test case corrupts its input, so nothing "
                        f"has ever seen it fail")
    print(f"\nself-test coverage: {len(covered)}/{len(GUARDS)} guards, "
          f"{', '.join(sorted(covered))}")

    if failures:
        print("\nSELF-TEST FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Self-test: every guard fails when its input is corrupted.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test(args.root)
    return run_guards(args.root)


if __name__ == "__main__":
    sys.exit(main())
