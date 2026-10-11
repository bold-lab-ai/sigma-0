"""Assert on evidence recorded from the cluster. Standard library only.

Why this exists
---------------
The things worth testing here cannot run on a GitHub runner: they need GH200s,
Lustre, frozen checkpoints and real market data. Installing dependencies until
the runner is *almost* the cluster would still only exercise the portable
periphery, and would leave the execution logic that actually matters -- matching,
fill attribution, replay fidelity -- untested. That is the wrong thing to
automate.

So the split is: **the cluster produces evidence, CI audits it.** A run on the
cluster writes `ci/evidence/simulator.json`; this checker reads it and asserts
the claims still hold. It imports nothing outside the standard library and
finishes in well under a second.

The part that makes this honest
-------------------------------
Recorded evidence is worthless if it can go stale silently. So the recorder
stores a SHA-256 of every source file the evidence depends on, and this checker
recomputes them. Change `mm_sim.py` without re-running on the cluster and CI
fails with the exact file that drifted -- which is the whole point, because the
failure mode being prevented is "the numbers in the file describe code that no
longer exists".

Usage:
    python ci/check_evidence.py                 # audit all evidence files
    python ci/check_evidence.py --file <path>   # audit one
    python ci/check_evidence.py --self-test     # prove the audit can fail
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO / "ci" / "evidence"

# Comparison operators an assertion may use, by name, so the evidence file
# stays declarative data rather than executable code.
OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


class EvidenceError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dig(obj: Any, dotted: str) -> Any:
    """Fetch `a.b.c` out of nested dicts, failing loudly on a missing key."""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise EvidenceError(f"metric {dotted!r} is not present in the evidence")
        cur = cur[part]
    return cur


def check_sources(doc: Dict[str, Any]) -> List[str]:
    """Recompute the hashes of every file the evidence claims to describe."""
    problems: List[str] = []
    sources = doc.get("sources") or {}
    if not sources:
        problems.append(
            "evidence declares no source files, so nothing stops it from going "
            "stale; the recorder must list what it exercised"
        )
    for rel, recorded in sorted(sources.items()):
        path = REPO / rel
        if not path.is_file():
            problems.append(f"{rel}: recorded in the evidence but missing from the tree")
            continue
        actual = sha256_of(path)
        if actual != recorded:
            problems.append(
                f"{rel}: changed since the evidence was recorded\n"
                f"      recorded {recorded[:16]}…  now {actual[:16]}…\n"
                f"      re-run `ci/record_evidence.py` on the cluster and commit the result"
            )
    return problems


def check_assertions(doc: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Evaluate every declared assertion against the recorded metrics."""
    problems: List[str] = []
    passed: List[str] = []
    metrics = doc.get("metrics") or {}
    assertions = doc.get("assertions") or []
    if not assertions:
        problems.append("evidence declares no assertions, so it asserts nothing")

    for a in assertions:
        name = a.get("name", "<unnamed>")
        try:
            metric, op_name, want = a["metric"], a["op"], a["value"]
        except KeyError as exc:
            problems.append(f"{name}: malformed assertion, missing {exc}")
            continue
        op = OPS.get(op_name)
        if op is None:
            problems.append(f"{name}: unknown operator {op_name!r}")
            continue
        try:
            got = dig(metrics, metric)
        except EvidenceError as exc:
            problems.append(f"{name}: {exc}")
            continue
        if op(got, want):
            passed.append(f"{name}: {metric} = {got} {op_name} {want}")
        else:
            why = a.get("why", "")
            problems.append(
                f"{name}: {metric} = {got}, required {op_name} {want}"
                + (f"\n      {why}" if why else "")
            )
    return problems, passed


def audit(path: Path) -> Tuple[List[str], List[str]]:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path.name}: not valid JSON -- {exc}"], []
    problems = check_sources(doc)
    more, passed = check_assertions(doc)
    return problems + more, passed


def self_test() -> int:
    """Prove the audit can fail. Synthetic evidence only; ci/evidence/ is not read.

    The audit's whole value is that a drifted source or a broken assertion goes red, so
    each is planted in a temp file and must be caught. Without this, a green audit is
    indistinguishable from an audit that stopped checking anything.
    """
    anchor = "ci/check_evidence.py"  # a tracked file whose current hash is at hand
    real = sha256_of(REPO / anchor)
    ok_doc = {
        "sources": {anchor: real},
        "metrics": {"m": {"x": 1}},
        "assertions": [{"name": "x holds", "metric": "m.x", "op": "==", "value": 1}],
    }
    cases = [
        ("well-formed evidence passes", ok_doc, False),
        ("a drifted source hash is caught",
         {**ok_doc, "sources": {anchor: "0" * 64}}, True),
        ("a failing assertion is caught",
         {**ok_doc, "assertions": [
             {"name": "x is two", "metric": "m.x", "op": "==", "value": 2}]}, True),
        ("an absent metric is caught",
         {**ok_doc, "assertions": [
             {"name": "gone", "metric": "m.y", "op": "==", "value": 1}]}, True),
        ("evidence that asserts nothing is caught", {**ok_doc, "assertions": []}, True),
        ("evidence that hashes nothing is caught", {**ok_doc, "sources": {}}, True),
    ]
    wrong = 0
    with tempfile.TemporaryDirectory() as td:
        for name, doc, want_fail in cases:
            probe = Path(td) / "probe.json"
            probe.write_text(json.dumps(doc))
            problems, _ = audit(probe)
            failed = bool(problems)
            state = "caught" if failed else "clean"
            good = failed == want_fail
            wrong += 0 if good else 1
            print(f"  {state:6s} {('ok' if good else 'WRONG'):5s}  {name}")
    if wrong:
        print(f"{wrong} self-test probe(s) failed", file=sys.stderr)
        return 1
    print("self-test passed: drift, broken, absent and missing assertions all go red")
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", action="append", help="audit only this evidence file")
    p.add_argument("--self-test", action="store_true",
                   help="prove the audit can fail, against synthetic evidence")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()

    files = ([Path(f).resolve() for f in args.file]
             if args.file else sorted(EVIDENCE_DIR.glob("*.json")))
    if not files:
        print(f"no evidence files under {EVIDENCE_DIR.relative_to(REPO)}", file=sys.stderr)
        return 1

    failed = 0
    for path in files:
        doc_problems, doc_passed = audit(path)
        rel = path.relative_to(REPO)
        head = json.loads(path.read_text()) if path.is_file() else {}
        recorded_on = head.get("recorded", {}).get("utc", "?")
        commit = head.get("recorded", {}).get("commit", "?")[:8]
        print(f"\n=== {rel}  (recorded {recorded_on} at {commit}) ===")
        for line in doc_passed:
            print(f"  ok    {line}")
        for line in doc_problems:
            print(f"  FAIL  {line}")
        if doc_problems:
            failed += 1

    print()
    if failed:
        print(f"{failed} evidence file(s) failed the audit")
        return 1
    print(f"all {len(files)} evidence file(s) pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
