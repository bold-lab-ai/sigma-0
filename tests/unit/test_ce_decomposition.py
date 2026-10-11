"""The CE-decomposition labelling contracts hold, and their guards can fail.

Both tests shell out to ci/check_ce_decomposition.py instead of importing the
audited modules: importing pulls in jax (absent on CI runners), and the
checker is the single source of truth for what "consistent" means, so the
test suite and the workflow cannot drift apart.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "ci" / "check_ce_decomposition.py"

# Read the guard tuple out of the checker's source rather than importing it: the
# module sits outside the package path and importing it here would couple the test
# suite to the checker's imports.
GUARDS = re.findall(r"^\s+(guard_\w+),$",
                    CHECKER.read_text(encoding="utf-8"), flags=re.MULTILINE)
assert GUARDS, "could not read the GUARDS tuple out of the checker"


def test_decomposition_guards_hold():
    proc = subprocess.run([sys.executable, str(CHECKER)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_guards_catch_their_corruptions():
    proc = subprocess.run([sys.executable, str(CHECKER), "--self-test"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Coverage, not a count: the number of corruption cases is free to grow, but a
    # guard with no case behind it is one nobody has ever seen fail. The previous
    # spelling asserted `== 4` and went red the first time a case was added.
    assert f"{len(GUARDS)}/{len(GUARDS)} guards" in proc.stdout, proc.stdout
    assert proc.stdout.count("caught: ") >= len(GUARDS), proc.stdout
