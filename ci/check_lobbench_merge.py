#!/usr/bin/env python3
"""Audit the merged LOB-Bench scorer and the modes the adapter drives.

Standard library only, and nothing here imports the scorer: importing pulls in
pandas, scipy, statsmodels and jax, none of which belong on a bare runner. The
driver is parsed, the tree is hashed, and the two test files are read as text.

Five guards, each closing a hole the merge leaves open:

  tree hash     merge.json says the tree is a specific three-way merge. Edit a
                file under scorer/lob_bench and every test that reads the tree
                still passes while the recorded provenance describes something
                else. Re-hashing here makes an edit a failure.

  WS-21 names   Theirs defines four *_levels scores; ours replaced them with
                *_ticks after those four drifted about sixteen times as much as
                the rest. A later merge, or a hand edit, that lets *_levels back
                in would keep publishing a mean under the name WS-21 while
                averaging different quantities. The declared names are asserted
                against the pinned list.

  mode configs  --mode cond/context/time_lagged resolve by finding a named dict
                in the driver's source. Rename or drop one upstream and the mode
                stops existing, with no test in this repo touching it. Each
                mode's config is required to be present and non-empty.

  test labels   test_contextual_scoring.py is carried in as documentation, not
                as a test: it has no asserts. merge.json says so. If someone adds
                asserts, the note is wrong and must be rewritten -- and if
                someone strips the asserts out of test_parallel.py, the one real
                check theirs brought becomes decorative. Both are asserted.

  self test     Every guard above is a claim this file makes about other files.
                --self-test corrupts a copy of each input and requires the guard
                to fail, so a guard that has quietly stopped checking anything
                cannot stay green.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCORER = REPO / "src" / "lobpipeline" / "scorer"
ADAPTER = REPO / "src" / "lobpipeline" / "lob_bench" / "scoring.py"
NOTEBOOKS = (
    REPO / "docs" / "experiments" / "model_zoo" / "lobbench_divergence.ipynb",
    REPO / "docs" / "experiments" / "model_zoo" / "zoo_benchmark.ipynb",
    REPO / "docs" / "experiments" / "model_zoo" / "rmsnorm_epsilon.ipynb",
)
NOTEBOOK = NOTEBOOKS[0]
# The notebook is committed with its outputs so it renders inline on GitHub.
# That only helps below the size where GitHub stops rendering; the baseline
# that does render in this repository is ~592 KB.
NOTEBOOK_MAX_BYTES = 900_000

GENERATED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ipynb_checkpoints"})
GENERATED_SUFFIXES = frozenset({".pyc", ".pyo"})

# Which module-level dict in run_bench.py each mode reads its score names from.
# Kept here rather than imported so this file needs nothing but the standard
# library; test_merged_scorer.py asserts the two agree.
MODE_CONFIGS = {
    "uncond": "DEFAULT_SCORING_CONFIG",
    "cond": "DEFAULT_SCORING_CONFIG_COND",
    "context": "DEFAULT_SCORING_CONFIG_CONTEXT",
    "time_lagged": "DEFAULT_SCORING_CONFIG_TIME_LAGGED",
}


class Failed(Exception):
    """A guard did not hold."""


def _source_files(root: Path) -> list[Path]:
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if GENERATED_DIRS.intersection(rel.parts) or path.suffix in GENERATED_SUFFIXES:
            continue
        found.append(path)
    return found


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _source_files(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _dict_keys(source: str, name: str, where: Path) -> list[str]:
    for node in ast.parse(source, filename=str(where)).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        keys = []
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise Failed(f"{where}: {name} has a non-literal key")
            keys.append(key.value)
        return keys
    raise Failed(f"{where}: no module-level {name}")


def _pinned_features(where: Path) -> tuple[str, ...]:
    return tuple(_dict_keys_or_tuple(where.read_text(encoding="utf-8"), "LOBBENCH_FEATURES", where))


def _dict_keys_or_tuple(source: str, name: str, where: Path) -> list[str]:
    for node in ast.parse(source, filename=str(where)).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise Failed(f"{where}: no module-level {name}")


def guard_tree_hash(scorer: Path) -> list[str]:
    record = json.loads((scorer / "merge.json").read_text(encoding="utf-8"))["lob_bench"]
    for side in ("base", "ours", "theirs"):
        commit = record[side]["commit"]
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise Failed(f"merge.json {side}.commit is not a full sha: {commit!r}")
    tree = scorer / record["path"]
    count = len(_source_files(tree))
    if count != record["file_count"]:
        raise Failed(f"{tree}: recorded {record['file_count']} files, found {count}")
    actual = _tree_sha256(tree)
    if actual != record["tree_sha256"]:
        raise Failed(
            f"{tree}: recorded {record['tree_sha256'][:16]}..., found {actual[:16]}...; "
            "the tree is no longer the merge merge.json describes"
        )
    return [
        f"merge: {record['ours']['commit'][:8]} + {record['theirs']['commit'][:8]} "
        f"over {record['base']['commit'][:8]}",
        f"tree:  {count} files, sha256 {actual[:16]}...",
    ]


def guard_ws21_names(scorer: Path, adapter: Path) -> list[str]:
    record = json.loads((scorer / "merge.json").read_text(encoding="utf-8"))["lob_bench"]
    driver = scorer / record["path"] / "run_bench.py"
    declared = _dict_keys(driver.read_text(encoding="utf-8"), MODE_CONFIGS["uncond"], driver)
    pinned = _pinned_features(adapter)
    levels = [n for n in declared if n.endswith("_levels")]
    if levels:
        raise Failed(
            f"the merged driver declares {levels}; those four were replaced by *_ticks "
            "because they drifted about sixteen times as much as the rest, and a mean "
            "over them is not WS-21"
        )
    if tuple(declared) != pinned:
        raise Failed(
            f"driver declares {declared}, adapter pins {list(pinned)}; WS-21 is a mean "
            "over exactly the pinned names"
        )
    return [f"WS-21: {len(declared)} names, no *_levels, driver and adapter agree"]


def guard_mode_configs(scorer: Path) -> list[str]:
    record = json.loads((scorer / "merge.json").read_text(encoding="utf-8"))["lob_bench"]
    driver = scorer / record["path"] / "run_bench.py"
    source = driver.read_text(encoding="utf-8")
    lines = []
    for mode, config in sorted(MODE_CONFIGS.items()):
        keys = _dict_keys(source, config, driver)
        if not keys:
            raise Failed(f"{config} is empty; mode {mode!r} would score nothing")
        lines.append(f"--mode {mode:<12} {len(keys):>2} score(s) from {config}")
    return lines


def guard_test_labels(scorer: Path) -> list[str]:
    record = json.loads((scorer / "merge.json").read_text(encoding="utf-8"))["lob_bench"]
    tests = scorer / record["path"] / "tests"

    demo = tests / "test_contextual_scoring.py"
    body = demo.read_text(encoding="utf-8")
    if "assert " in body:
        raise Failed(
            f"{demo.name} now contains asserts, so merge.json's note calling it a "
            "demonstration with no failure path is out of date"
        )

    real = tests / "test_parallel.py"
    text = real.read_text(encoding="utf-8")
    n = text.count("assert ")
    if n < 3 or "allclose" not in text:
        raise Failed(
            f"{real.name} has {n} asserts and "
            f"{'no' if 'allclose' not in text else 'an'} allclose; it is the only check "
            "theirs brought that can fail, and it must keep comparing the fast paths "
            "against the reference"
        )
    return [
        f"{demo.name}: 0 asserts, labelled as documentation in merge.json",
        f"{real.name}: {n} asserts, compares fast paths against the reference",
    ]


def guard_lag_offset(scorer: Path) -> list[str]:
    """The lag must offset the two score arrays, not truncate both merged frames.

    The tree hash alone cannot hold this. merge.json's `refresh` recipe says to
    re-derive the tree from upstream and re-apply the substitutions; someone
    following it would drop this patch, recompute the hash, and the pin would
    agree with a tree that measures a contemporaneous conditional again. The
    property has to be asserted directly.
    """
    path = scorer / "lob_bench" / "scoring.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    slicers, offsets = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_align_and_slice":
            if ".iloc[lag:]" in ast.unparse(node):
                raise Failed(
                    f"{path.name}: _align_and_slice still drops the first lag rows of "
                    "both frames. Slicing both by the same amount truncates; it does "
                    "not offset, and --time_lagged degenerates to --conditional."
                )
            slicers.append(node)
        elif node.name == "_align_scores":
            body = ast.unparse(node)
            if "[:-lag]" not in body or "[lag:]" not in body:
                raise Failed(
                    f"{path.name}: _align_scores does not return an offset pair. "
                    "Expected the conditioning array at [:-lag] against the "
                    "evaluated array at [lag:]."
                )
            offsets.append(node)
    if len(slicers) != 2 or len(offsets) != 2:
        raise Failed(
            f"{path.name}: expected two copies of each helper (the module-level path "
            f"and the one inside merge_and_score_lagged); found {len(slicers)} and "
            f"{len(offsets)}. A copy that drifts is a copy that stops being checked."
        )
    return [
        "time_lagged: both copies offset sl[:-lag] against se[lag:], neither truncates",
    ]


def guard_notebook_renders(_scorer: Path) -> list[str]:
    """The committed notebook must be readable without running it.

    Stripping the outputs, or an execution that stopped at an error, leaves a
    file that still parses and still diffs cleanly -- a human reviewer sees a
    wall of base64 either way. Size is checked for the same reason: a notebook
    too large for GitHub to render is committed, correct, and unread.
    """
    lines = []
    for notebook in NOTEBOOKS if NOTEBOOK is NOTEBOOKS[0] else (NOTEBOOK,):
        lines.append(_check_one_notebook(notebook))
    return lines


def _check_one_notebook(NOTEBOOK: Path) -> str:
    if not NOTEBOOK.is_file():
        raise Failed(f"{NOTEBOOK}: notebook is not committed")
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    code = [c for c in cells if c.get("cell_type") == "code"]
    if not code:
        raise Failed(f"{NOTEBOOK}: no code cells")
    bare = [i for i, c in enumerate(code) if not c.get("outputs")]
    if bare:
        raise Failed(
            f"{NOTEBOOK}: code cell(s) {bare} carry no output; the notebook was "
            "committed unexecuted or with its outputs stripped"
        )
    errors = [
        o.get("ename")
        for c in code
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    if errors:
        raise Failed(f"{NOTEBOOK}: execution left {len(errors)} error output(s): {errors}")
    figures = sum(
        1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {})
    )
    if figures < 7:
        raise Failed(f"{NOTEBOOK}: {figures} figures, expected at least 7")
    size = NOTEBOOK.stat().st_size
    if size > NOTEBOOK_MAX_BYTES:
        raise Failed(
            f"{NOTEBOOK}: {size} bytes exceeds {NOTEBOOK_MAX_BYTES}; GitHub will "
            "not render it inline and committing the outputs buys nothing"
        )
    return (f"{NOTEBOOK.name}: {len(code)} code cells all with output, "
            f"{figures} figures, 0 errors, {size // 1024} KB")


GUARDS = (
    ("merged tree hashes to its record", lambda s: guard_tree_hash(s)),
    ("WS-21 names, no *_levels", lambda s: guard_ws21_names(s, ADAPTER)),
    ("every mode's config resolves", lambda s: guard_mode_configs(s)),
    ("time_lagged offsets rather than truncates", lambda s: guard_lag_offset(s)),
    ("the two merged test files are what merge.json says", lambda s: guard_test_labels(s)),
    ("the committed notebook still renders", guard_notebook_renders),
)


def run(scorer: Path) -> int:
    failures = 0
    for title, guard in GUARDS:
        try:
            for line in guard(scorer):
                print(f"  {line}")
            print(f"ok   {title}")
        except Failed as exc:
            print(f"FAIL {title}\n     {exc}")
            failures += 1
        except Exception as exc:  # a guard that cannot run is a guard that failed
            print(f"FAIL {title}\n     {type(exc).__name__}: {exc}")
            failures += 1
    return failures


# Each corruption is a thing that could plausibly happen: an edit to the merged
# tree, a *_levels score coming back, a mode's config disappearing, the real test
# being stripped. The guard that is supposed to notice must return non-zero.
# Each entry names the guard that must be the one to notice.  Checking only the
# exit code is not enough: every edit inside the merged tree also moves the tree
# hash, so the hash guard answers for all of them and a content guard that has
# stopped working still looks green.
CORRUPTIONS = {
    "an edited file in the merged tree": ("merged tree hashes to its record", lambda s: (s / "lob_bench" / "metrics.py").write_text(
        (s / "lob_bench" / "metrics.py").read_text(encoding="utf-8") + "\n# edited\n",
        encoding="utf-8",
    )),
    "a *_levels score back in the driver": ("WS-21 names, no *_levels", lambda s: _patch(
        s / "lob_bench" / "run_bench.py", '"spread": {', '"limit_ask_order_levels": {'
    )),
    "a mode's config renamed away": ("every mode's config resolves", lambda s: _patch(
        s / "lob_bench" / "run_bench.py",
        "DEFAULT_SCORING_CONFIG_CONTEXT = {",
        "DEFAULT_SCORING_CONFIG_CONTEXT_OLD = {",
    )),
    "the lag offset reverted to a truncation": ("time_lagged offsets rather than truncates", lambda s: _patch(
        s / "lob_bench" / "scoring.py",
        "return sl[:-lag], se[lag:]",
        "return sl, se",
    )),
    "the asserts stripped from test_parallel.py": ("the two merged test files are what merge.json says", lambda s: _patch_all(
        s / "lob_bench" / "tests" / "test_parallel.py", "assert ", "pass  # "
    )),
}

# The notebook is not inside the scorer tree, so its corruptions are applied to
# a copy of the repository's own file rather than to the copied scorer.
NOTEBOOK_CORRUPTIONS = {
    "the notebook's outputs stripped": lambda payload: [
        c.update({"outputs": []}) for c in payload["cells"] if c.get("cell_type") == "code"
    ],
    "an error output left in the notebook": lambda payload: next(
        c["outputs"].append({"output_type": "error", "ename": "ValueError",
                             "evalue": "x", "traceback": []})
        for c in payload["cells"] if c.get("cell_type") == "code" and c.get("outputs")
    ),
}


def _patch(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise Failed(f"self-test cannot corrupt {path.name}: {old!r} not found")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _patch_all(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise Failed(f"self-test cannot corrupt {path.name}: {old!r} not found")
    path.write_text(text.replace(old, new), encoding="utf-8")


def self_test() -> int:
    print("self-test: each corruption must be caught by its own guard")
    failures = 0
    for name, (guard_title, corrupt) in CORRUPTIONS.items():
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "scorer"
            shutil.copytree(SCORER, copy)
            corrupt(copy)
            result = subprocess.run(
                [sys.executable, __file__, "--scorer", str(copy)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  NOT CAUGHT  {name}")
                failures += 1
            elif f"FAIL {guard_title}" not in result.stdout:
                # Something failed, but not the guard whose job this is.  Almost
                # always the tree hash standing in for a content guard that has
                # quietly stopped working.
                print(f"  WRONG GUARD {name}\n              expected {guard_title!r}")
                failures += 1
            else:
                print(f"  caught      {name}")

    for name, corrupt in NOTEBOOK_CORRUPTIONS.items():
        with tempfile.TemporaryDirectory() as tmp:
            payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
            corrupt(payload)
            broken = Path(tmp) / "lobbench_divergence.ipynb"
            broken.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, __file__, "--notebook", str(broken)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 or "FAIL the committed notebook still renders" not in result.stdout:
                print(f"  NOT CAUGHT  {name}")
                failures += 1
            else:
                print(f"  caught      {name}")

    if failures:
        print(f"{failures} corruption(s) went unnoticed; a guard has stopped checking")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", default=str(SCORER), help=argparse.SUPPRESS)
    parser.add_argument("--notebook", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="corrupt a copy of each input and require the matching guard to fail",
    )
    args = parser.parse_args()
    if args.notebook:
        global NOTEBOOK
        NOTEBOOK = Path(args.notebook)
    if args.self_test:
        return 1 if self_test() else 0
    return 1 if run(Path(args.scorer)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
