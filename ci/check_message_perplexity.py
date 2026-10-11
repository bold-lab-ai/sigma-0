"""Guard the message-perplexity metric. Standard library only, nothing imported.

The unit tests cover the arithmetic and the cross-tokenization invariance. What is
left for CI is everything that can rot *between* files:

1. The committed report is internally consistent: both variants present, the role
   components add up to `full`, `full >= main`, and the `main` variant names no
   reference field and no absolute-time field. A metric whose two numbers cannot be
   checked against each other is not a metric.
2. The report's totals still match the ce-by-position records they were derived
   from. The evaluator sums per-position means, the reporter sums per-field means
   times widths; if those two ever stop agreeing, one of them has changed meaning.
   And the covering direction: every record in the tree is *in* the report. Check 2
   read report -> tree, which cannot see a record nobody derived; that is how a
   one-level `*.json` in the regeneration command survived three commits that put
   records in subdirectories.
3. `EXPECTED_FIELDS` in the reporter still equals the encoding's own FIELDS, read
   out of the source with `ast`. This is the drift check: a duplicated field list is
   fine only for as long as something fails when it goes stale.
4. The role partition (main / absolute_time / reference) covers every encoding
   exactly, with the token counts the metric claims.
5. The notebook is committed executed, with an output on every code cell and no
   error outputs -- the check a human reviewer cannot do on a base64 diff.

Usage:
    python ci/check_message_perplexity.py               # audit the repo
    python ci/check_message_perplexity.py --self-test   # prove the guard bites
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

METRIC = "src/base_model/evaluation/message_perplexity.py"
REPORTER = "run/base_model/message_perplexity_report.py"
REPORT = "artifacts/message_perplexity/report.json"
RECORDS = "docs/experiments/trainloss_profiling_20260826/data"
NOTEBOOK = "docs/experiments/message_perplexity/message_perplexity.ipynb"
ENCODINGS = ("22tok", "23tok", "24tok", "26tok")

# What the metric claims each role costs, per encoding. Duplicated from the
# encodings on purpose and checked against them below.
EXPECTED_WIDTHS = {
    "22tok": {"main": 9, "absolute_time": 5, "reference": 8},
    "23tok": {"main": 9, "absolute_time": 5, "reference": 9},
    "24tok": {"main": 10, "absolute_time": 5, "reference": 9},
    "26tok": {"main": 11, "absolute_time": 5, "reference": 10},
}
ABSOLUTE_TIME_FIELDS = {"time_s", "time_ns"}
REL_TOL = 1e-6


def _role(name: str) -> str:
    if name.endswith("_ref"):
        return "reference"
    if name in ABSOLUTE_TIME_FIELDS:
        return "absolute_time"
    return "main"


def _encoding_fields(root: Path, mode: str) -> dict[str, int] | None:
    """FIELDS -> token width, read from the encoding source without importing it."""
    path = root / "src" / "base_model" / "encoding" / f"encoding_{mode}.py"
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fields = widths = None
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
                fields = list(ast.literal_eval(stmt.value))
            elif target.id == "TOK_LENS":
                value = stmt.value
                if isinstance(value, ast.Call) and value.args:
                    value = value.args[0]
                widths = [int(v) for v in ast.literal_eval(value)]
    if fields is None or widths is None or len(fields) != len(widths):
        return None
    return dict(zip(fields, widths))


def _reporter_expected_fields(root: Path) -> list[str] | None:
    path = root / REPORTER
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "EXPECTED_FIELDS":
                return list(ast.literal_eval(node.value))
    return None


# --------------------------------------------------------------------------


def guard_partition(root: Path) -> list[str]:
    """The role partition covers every encoding exactly."""
    problems: list[str] = []
    seen = 0
    for mode in ENCODINGS:
        widths = _encoding_fields(root, mode)
        if widths is None:
            problems.append(f"encoding_{mode}.py: could not read FIELDS/TOK_LENS")
            continue
        seen += 1
        got = {"main": 0, "absolute_time": 0, "reference": 0}
        for name, w in widths.items():
            got[_role(name)] += w
        if got != EXPECTED_WIDTHS[mode]:
            problems.append(
                f"encoding_{mode}: role widths are {got}, the metric claims "
                f"{EXPECTED_WIDTHS[mode]}")
        total = sum(got.values())
        if total != int(mode.removesuffix("tok")):
            problems.append(
                f"encoding_{mode}: roles cover {total} tokens, not {mode}")
        refs = [f for f in widths if _role(f) == "reference"]
        if len(refs) != 4:
            problems.append(f"encoding_{mode}: {len(refs)} reference fields, expected 4")
    if not problems:
        print(f"[guard] role partition covers {seen} encodings exactly")
    return problems


def guard_reporter_field_list(root: Path) -> list[str]:
    """The reporter's duplicated field list still matches the 26tok encoding."""
    problems: list[str] = []
    expected = _reporter_expected_fields(root)
    if expected is None:
        return [f"{REPORTER}: EXPECTED_FIELDS not found"]
    widths = _encoding_fields(root, "26tok")
    if widths is None:
        return ["encoding_26tok.py: could not read FIELDS"]
    if expected != list(widths):
        problems.append(
            f"{REPORTER}: EXPECTED_FIELDS has drifted from encoding_26tok.FIELDS "
            f"({expected} != {list(widths)})")
    if not problems:
        print(f"[guard] {REPORTER}: field list matches encoding_26tok ({len(expected)} fields)")
    return problems


def guard_report(root: Path) -> list[str]:
    """The committed report is internally consistent and matches its sources."""
    problems: list[str] = []
    path = root / REPORT
    if not path.is_file():
        print(f"[guard] {REPORT}: not derived yet")
        return problems
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{REPORT} does not parse: {exc}"]

    entries = doc.get("entries") or []
    if not entries:
        return [f"{REPORT}: no entries, so it asserts nothing"]

    for entry in entries:
        tag = entry.get("slice_label") or entry.get("source_record") or "?"
        variants = entry.get("variants") or {}
        for name in ("main", "full"):
            if name not in variants:
                problems.append(f"{REPORT}[{tag}]: variant {name!r} missing")
        if problems:
            continue
        main_v, full_v = variants["main"], variants["full"]

        if full_v["nats_per_message"] < main_v["nats_per_message"]:
            problems.append(
                f"{REPORT}[{tag}]: full ({full_v['nats_per_message']:.4f}) is below "
                f"main ({main_v['nats_per_message']:.4f}); scoring more fields cannot "
                "lower the NLL")

        comps = entry.get("components_nats_per_message") or {}
        total = sum(comps.values())
        if not math.isclose(total, full_v["nats_per_message"], rel_tol=REL_TOL):
            problems.append(
                f"{REPORT}[{tag}]: components sum to {total:.6f} but full is "
                f"{full_v['nats_per_message']:.6f}")
        if not math.isclose(comps.get("main", float("nan")),
                            main_v["nats_per_message"], rel_tol=REL_TOL):
            problems.append(f"{REPORT}[{tag}]: main component and main variant disagree")

        for name, nats in (("main", main_v), ("full", full_v)):
            bits = nats["bits_per_message"]
            if not math.isclose(bits, nats["nats_per_message"] / math.log(2), rel_tol=REL_TOL):
                problems.append(f"{REPORT}[{tag}]: {name} bits are not nats / ln 2")

        fields = main_v.get("fields") or []
        if not fields:
            problems.append(f"{REPORT}[{tag}]: main variant does not say what it scored")
        for name in fields:
            if _role(name) != "main":
                problems.append(
                    f"{REPORT}[{tag}]: main variant scored {name!r}, which is "
                    f"{_role(name)}")

        per_field = entry.get("per_field_nats_per_message") or {}
        if per_field:
            field_total = sum(per_field.values())
            if not math.isclose(field_total, full_v["nats_per_message"], rel_tol=REL_TOL):
                problems.append(
                    f"{REPORT}[{tag}]: per-field nats sum to {field_total:.6f}, "
                    f"full says {full_v['nats_per_message']:.6f}")

        cross = entry.get("cross_check")
        if cross is not None and cross.get("relative_difference", 0.0) > 1e-5:
            problems.append(
                f"{REPORT}[{tag}]: derived total drifted from the source record by "
                f"{cross['relative_difference']:.2e}")

        source = entry.get("source_record")
        if source and not (root / source).is_file():
            problems.append(f"{REPORT}[{tag}]: source record {source} is not in the tree")

    if not problems:
        print(f"[guard] {REPORT}: {len(entries)} entries consistent "
              "(components add up, full >= main, sources present)")
    return problems


def guard_report_covers_every_record(root: Path) -> list[str]:
    """Every record in the tree is in the report, not just every entry in the tree.

    `guard_report` walks report -> tree: each entry names a file that exists. That
    direction is blind to omission, and omission is the failure that actually
    happened -- records landed in `pair/` and `xtok/` while the regeneration command
    still globbed one level deep, so the report was a strict subset of its own
    inputs and every per-entry check still passed. Walking tree -> report is what
    makes "derived from the committed records" mean all of them.
    """
    records = root / RECORDS
    report = root / REPORT
    if not records.is_dir():
        print(f"[guard] {RECORDS}: no record tree here")
        return []
    if not report.is_file():
        return []  # guard_report already says the report is not derived yet
    try:
        doc = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []  # guard_report already reports the parse failure

    on_disk = {p.relative_to(root).as_posix() for p in records.rglob("*.json")}
    derived = {e.get("source_record") for e in doc.get("entries") or []}
    missing = sorted(on_disk - derived)
    if missing:
        shown = ", ".join(missing[:4]) + (", ..." if len(missing) > 4 else "")
        return [f"{REPORT}: {len(missing)} of {len(on_disk)} records under {RECORDS} "
                f"are not in the report ({shown}); regenerate it with "
                f"--records {RECORDS}, which walks the tree"]
    print(f"[guard] {RECORDS}: all {len(on_disk)} records are in {REPORT}")
    return []


def guard_notebook(root: Path) -> list[str]:
    """The notebook is committed executed, with no error outputs."""
    problems: list[str] = []
    path = root / NOTEBOOK
    if not path.is_file():
        print(f"[guard] {NOTEBOOK}: not written yet")
        return problems
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{NOTEBOOK} does not parse: {exc}"]
    empty = errored = figures = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if not "".join(cell.get("source", [])).strip():
            continue
        outputs = cell.get("outputs", [])
        if not outputs:
            empty += 1
        for out in outputs:
            if out.get("output_type") == "error":
                errored += 1
            if "image/png" in (out.get("data") or {}):
                figures += 1
    if empty:
        problems.append(f"{NOTEBOOK}: {empty} code cell(s) carry no output")
    if errored:
        problems.append(f"{NOTEBOOK}: {errored} code cell(s) ended in an error")
    if not problems:
        print(f"[guard] {NOTEBOOK}: executed, {figures} figures, no error outputs")
    return problems


def guard_tracked(root: Path, tracked: set[str] | None = None) -> list[str]:
    """The metric, its reporter and its evidence are in version control."""
    if tracked is None:
        try:
            out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                                 capture_output=True, text=True, check=True).stdout
            tracked = {p for p in out.split("\0") if p}
        except Exception:
            return []
    if not tracked:
        return []
    problems = []
    for rel in (METRIC, REPORTER, REPORT, NOTEBOOK):
        if (root / rel).is_file() and rel not in tracked:
            problems.append(f"{rel} exists but is not tracked by git")
    if not problems:
        print("[guard] metric, reporter and evidence are tracked")
    return problems


GUARDS = (
    guard_partition,
    guard_reporter_field_list,
    guard_report,
    guard_report_covers_every_record,
    guard_notebook,
    guard_tracked,
)


def run_guards(root: Path) -> int:
    problems: list[str] = []
    for guard in GUARDS:
        problems.extend(guard(root))
    if problems:
        print(f"\ncheck_message_perplexity: {len(problems)} problem(s)")
        for problem in problems:
            print(" -", problem)
        return 1
    print("\nAll message-perplexity guards hold.")
    return 0


# --------------------------------------------------------------------------


def _good_report() -> dict:
    per_field = {"event_type": 0.72, "direction": 0.39, "price": 2.25, "size": 1.04,
                 "delta_t_s": 0.0, "delta_t_ns": 9.72, "time_s": 0.03, "time_ns": 0.44,
                 "price_ref": 0.52, "size_ref": 0.02, "time_s_ref": 0.37,
                 "time_ns_ref": 5.35}
    main = sum(v for k, v in per_field.items() if _role(k) == "main")
    abst = sum(v for k, v in per_field.items() if _role(k) == "absolute_time")
    ref = sum(v for k, v in per_field.items() if _role(k) == "reference")
    full = main + abst + ref
    ln2 = math.log(2)
    return {
        "schema_version": 1,
        "metric": "message_perplexity",
        "entries": [{
            "slice_label": "heldout",
            "source_record": None,
            "variants": {
                "main": {"nats_per_message": main, "bits_per_message": main / ln2,
                         "fields": [k for k in per_field if _role(k) == "main"]},
                "full": {"nats_per_message": full, "bits_per_message": full / ln2,
                         "fields": list(per_field)},
            },
            "components_nats_per_message": {
                "main": main, "absolute_time": abst, "reference": ref},
            "per_field_nats_per_message": per_field,
        }],
    }


def self_test() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = root / REPORT
        report.parent.mkdir(parents=True)

        def audit(doc) -> list[str]:
            report.write_text(json.dumps(doc), encoding="utf-8")
            return guard_report(root)

        if audit(_good_report()):
            failures.append("a clean report should pass: " + str(guard_report(root)))

        bad = _good_report()
        bad["entries"][0]["variants"]["full"]["nats_per_message"] = 1.0
        if not any("cannot lower the NLL" in p for p in audit(bad)):
            failures.append("full below main not caught")

        bad = _good_report()
        bad["entries"][0]["components_nats_per_message"]["reference"] += 3.0
        if not any("components sum to" in p for p in audit(bad)):
            failures.append("components not adding up not caught")

        bad = _good_report()
        bad["entries"][0]["variants"]["main"]["fields"].append("price_ref")
        if not any("which is reference" in p for p in audit(bad)):
            failures.append("a reference field inside main not caught")

        bad = _good_report()
        bad["entries"][0]["variants"]["main"]["bits_per_message"] *= 1.5
        if not any("bits are not nats" in p for p in audit(bad)):
            failures.append("bits/nats inconsistency not caught")

        bad = _good_report()
        bad["entries"][0]["per_field_nats_per_message"]["price"] += 2.0
        if not any("per-field nats sum to" in p for p in audit(bad)):
            failures.append("per-field table not matching the total not caught")

        bad = _good_report()
        del bad["entries"][0]["variants"]["main"]
        if not any("variant 'main' missing" in p for p in audit(bad)):
            failures.append("missing variant not caught")

        bad = _good_report()
        bad["entries"] = []
        if not any("no entries" in p for p in audit(bad)):
            failures.append("empty report not caught")

        # the covering guard must notice a record nobody derived -- the failure
        # that a one-level glob in the regeneration command produces
        records = root / RECORDS
        records.mkdir(parents=True)
        (records / "top.json").write_text("{}", encoding="utf-8")
        (records / "sub").mkdir()
        (records / "sub" / "nested.json").write_text("{}", encoding="utf-8")

        covered = _good_report()
        covered["entries"] = [dict(covered["entries"][0], source_record=src)
                              for src in (f"{RECORDS}/top.json",
                                          f"{RECORDS}/sub/nested.json")]
        report.write_text(json.dumps(covered), encoding="utf-8")
        if guard_report_covers_every_record(root):
            failures.append("a report covering every record should pass")

        covered["entries"] = covered["entries"][:1]
        report.write_text(json.dumps(covered), encoding="utf-8")
        if not any("are not in the report" in p
                   for p in guard_report_covers_every_record(root)):
            failures.append("a record missing from the report not caught")

        # the partition guard must reject an encoding whose widths moved
        enc = root / "src" / "base_model" / "encoding"
        enc.mkdir(parents=True)
        (enc / "encoding_26tok.py").write_text(
            "import numpy as np\n"
            "class Message_Tokenizer:\n"
            "    FIELDS = ('event_type', 'direction', 'price', 'size', 'delta_t_s',\n"
            "              'delta_t_ns', 'time_s', 'time_ns', 'price_ref', 'size_ref',\n"
            "              'time_s_ref', 'time_ns_ref')\n"
            "    TOK_LENS = np.array((1, 1, 9, 2, 1, 3, 2, 3, 3, 2, 2, 3))\n",
            encoding="utf-8")
        if not any("role widths are" in p for p in guard_partition(root)):
            failures.append("a changed encoding width not caught")

        # a notebook committed without outputs must fail
        nb = root / NOTEBOOK
        nb.parent.mkdir(parents=True)
        nb.write_text(json.dumps({"cells": [
            {"cell_type": "code", "source": ["print(1)"], "outputs": []}]}),
            encoding="utf-8")
        if not any("carry no output" in p for p in guard_notebook(root)):
            failures.append("a stripped notebook not caught")

    if failures:
        print("SELF-TEST FAIL")
        for failure in failures:
            print(" -", failure)
        return 1
    print("SELF-TEST OK: 10 bite cases behave")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    return self_test() if args.self_test else run_guards(REPO)


if __name__ == "__main__":
    sys.exit(main())
