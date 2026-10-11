"""Break the return bench on purpose and require the tests to notice. Stdlib only.

Why this exists
---------------
`ci/evidence/return_bench.json` carries both the measured numbers and the
assertions they must satisfy, and one run wrote both. Auditing it proves the
numbers have not drifted from the code that produced them -- worth having, and it
has caught real drift -- but a wrong implementation would have recorded a wrong
number, and the assertion could have been written to accommodate it. Reading a
file and comparing it against conditions stored in the same file cannot detect
that.

Tests are better, because they execute the code. But a test suite that only ever
sees correct input cannot tell a working implementation from one that returns the
right answer for the wrong reason. "All tests pass" is evidence about the tests
only if the tests would have failed otherwise.

So this makes that last step explicit. It plants specific defects in the metric,
runs the suite against each planted copy, and requires each one to fail. A defect
that survives means the suite does not cover it, and the corresponding "all
green" was not saying what it appeared to say.

Each mutation is a real mistake, not a syntactic scramble:

  no_coverage_check    drop the explicit end-of-sequence test, so searchsorted
                       silently returns the final row and a long horizon becomes
                       "the return at the end of the sequence". On real data this
                       corrupts 42% of 500 ms samples while raising nothing.

  searchsorted_left    right -> left, which skips a message sharing the target
                       timestamp. Real order book data has many messages inside
                       one microsecond, so this changes answers without ever
                       looking wrong.

  reads_the_future     drop the -1, so the row *after* the horizon is read. A
                       look-ahead that makes results better, not worse, which is
                       the direction nobody questions.

  no_anchor_check      stop requiring the last conditioning message to be a
                       trade, which turns an event-anchored measurement back into
                       an ordinary windowed one while every number stays
                       plausible.

  identity_shortcut    perturb the identity path by 1e-4, so the check that is
                       supposed to return exactly 1.0 returns nearly 1.0.

Run:  python ci/mutation_check.py [-v]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Test files each mutant copy is run against. They must be enough to cover every
# mutation; the run below reports it when they are not.
TEST_FILES = (
    "tests/unit/test_event_bench_end_to_end.py",
    "tests/unit/test_return_bench_event.py",
)

TARGETS = (
    "src/lobpipeline/return_bench/event_anchored.py",
    "src/lobpipeline/return_bench/run_event_bench.py",
)

MUTATIONS = {
    "no_coverage_check": (
        "    if times.size == 0 or times[-1] < t_target:",
        "    if times.size == 0 or False:",
    ),
    "searchsorted_left": (
        'side="right")) - 1',
        'side="left")) - 1',
    ),
    "reads_the_future": (
        'j = int(np.searchsorted(times, t_target, side="right")) - 1',
        'j = int(np.searchsorted(times, t_target, side="right"))',
    ),
    "no_anchor_check": (
        "if require_trade and etype != EVENT_TYPE_TRADE:",
        "if False and etype != EVENT_TYPE_TRADE:",
    ),
    "identity_shortcut": (
        "            if identity:\n                gen_t, gen_m = real_t, real_m",
        "            if identity:\n                gen_t, gen_m = real_t, real_m * 1.0001",
    ),
}

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def build_tree(root: Path, name: str, old, new) -> Path:
    """A self-contained copy of src/ and the tests, with one defect planted.

    The layout matters: tests/conftest.py resolves ROOT/src from its own location,
    so the tests must live inside the copied tree. Run them from the real tests/
    directory instead and conftest puts the real src/ first on sys.path, every
    mutant is silently ignored, and all of them "survive" -- which is how this
    check first appeared to prove the opposite of the truth.
    """
    d = root / name
    shutil.copytree(REPO / "src", d / "src", ignore=IGNORE)
    (d / "tests").mkdir(parents=True)
    shutil.copy(REPO / "tests" / "conftest.py", d / "tests" / "conftest.py")
    for i, rel in enumerate(TEST_FILES):
        shutil.copy(REPO / rel, d / "tests" / f"t{i}.py")

    if old is None:
        return d
    planted = 0
    for rel in TARGETS:
        p = d / rel
        s = p.read_text()
        if old in s:
            p.write_text(s.replace(old, new, 1))
            planted += 1
    if not planted:
        raise SystemExit(
            f"mutation {name!r} matched nothing in {TARGETS}. It would be recorded as "
            f"'survived' while never having been applied — fix the pattern, do not "
            f"weaken the check.")
    return d


def run_tests(d: Path, verbose: bool) -> tuple[bool, str]:
    files = [f"tests/t{i}.py" for i in range(len(TEST_FILES))]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider"],
        cwd=d, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(no output)"
    if verbose:
        print(proc.stdout[-2000:])
    return proc.returncode == 0, tail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="mutation_check_") as tmp:
        root = Path(tmp)

        # Control first: if the unmutated copy fails, every "killed" below would be
        # meaningless because the suite is red for an unrelated reason.
        passed, tail = run_tests(build_tree(root, "control", None, None), args.verbose)
        print(f"{'control (unmutated)':24s} {'ok' if passed else 'BROKEN':9s} {tail}")
        if not passed:
            print("\nThe suite does not pass on unmutated source, so nothing below can be "
                  "interpreted. Fix that first.", file=sys.stderr)
            return 2

        survivors = []
        for name, (old, new) in MUTATIONS.items():
            passed, tail = run_tests(build_tree(root, name, old, new), args.verbose)
            verdict = "SURVIVED" if passed else "killed"
            if passed:
                survivors.append(name)
            print(f"{name:24s} {verdict:9s} {tail}")

    print()
    if survivors:
        print(f"{len(survivors)} mutation(s) survived: {survivors}", file=sys.stderr)
        print("The suite passes on code containing these defects, so a green run does "
              "not rule them out. Add a case that fails, rather than removing the "
              "mutation.", file=sys.stderr)
        return 1
    print(f"all {len(MUTATIONS)} planted defects were caught by the suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
