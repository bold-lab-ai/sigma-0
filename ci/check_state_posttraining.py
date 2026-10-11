#!/usr/bin/env python3
"""Guard the state-model post-training package's own claims.

Standard library + ast only: the runner has no torch and importing the package
would execute it.  Every gate documents the hole it closes — what would go
silently wrong if someone edited the code without tripping any other test.

Gates:

  default-clock   The package's core promise is a 1 ms wall-clock frame grid;
                  every "H frames = H milliseconds" statement downstream leans
                  on DEFAULT_INTERVAL_NS.  Someone tuning an experiment can
                  change the default and nothing else fails — horizons quietly
                  stop meaning what the docs say.  Pinned here.

  four-methods    The PR ships ReFL, Flow-GRPO, AWM, RAM and DMD(+DMDR).
                  Deleting or renaming one file (and its export) leaves an
                  importable package with a silently smaller method set.  The
                  class definitions and the __init__ exports are asserted.

  grpo-noise      Flow-GRPO's constructor refuses a deterministic sampler:
                  with noise_scale = 0 the group collapses onto one path, the
                  advantage has no variance, and training becomes a silent
                  no-op that still prints losses.  Removing that raise is a
                  one-line edit no test notices from the outside.

  loss-mask       The pretraining loss is masked to generated frames.  The
                  sigma-flow ancestor's top defect was exactly this mask
                  missing — half of every loss tensor was an analytically
                  solvable filler term.  Dropping the frame_mask keyword from
                  the fm_loss call reintroduces it without breaking anything.

  param-account   The backbone claims parity with BookDiT's published account
                  (953,769 parameters at width 128 / depth 4 / 41 channels)
                  and a unit test asserts the exact number.  Deleting that
                  test is how a drifted port hides; its presence is asserted.

  validity-test   The chart's whole point is that ANY tensor decodes to a
                  valid book; the unit test that decodes pure noise and checks
                  spread/ladder is the executable form of that claim.  Its
                  presence is asserted here so it cannot be quietly dropped.

Run with --self-test to prove each gate fails on a broken copy.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TRAINERS = {
    "refl.py": ["ReFLTrainer"],
    "flow_grpo.py": ["FlowGRPOTrainer"],
    "awm.py": ["AWMTrainer"],
    "ram.py": ["RAMTrainer"],
    "dmd.py": ["DMDTrainer", "DMDRTrainer"],
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def gate_default_clock(root: Path) -> list:
    errors = []
    tree = _parse(root / "src/state/frames.py")
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "DEFAULT_INTERVAL_NS":
                    if isinstance(node.value, ast.Constant):
                        found = node.value.value
    if found != 1_000_000:
        errors.append(f"DEFAULT_INTERVAL_NS is {found!r}, expected 1_000_000 (1 ms)")
    else:
        print(f"  default-clock: DEFAULT_INTERVAL_NS = {found} (1 ms per frame)")
    return errors


def gate_four_methods(root: Path) -> list:
    errors = []
    pt = root / "src/state/posttraining"
    for fname, classes in TRAINERS.items():
        path = pt / fname
        if not path.exists():
            errors.append(f"missing {path.relative_to(root)}")
            continue
        defined = {n.name for n in ast.walk(_parse(path)) if isinstance(n, ast.ClassDef)}
        for cls in classes:
            if cls not in defined:
                errors.append(f"{fname} no longer defines {cls}")
    init_names = set()
    for node in ast.walk(_parse(pt / "__init__.py")):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    init_names = {
                        e.value for e in node.value.elts if isinstance(e, ast.Constant)
                    }
    expected = {c for classes in TRAINERS.values() for c in classes}
    missing = expected - init_names
    if missing:
        errors.append(f"__init__.__all__ dropped {sorted(missing)}")
    if not errors:
        print(f"  four-methods: {len(expected)} trainers defined and exported")
    return errors


def gate_grpo_noise(root: Path) -> list:
    tree = _parse(root / "src/state/posttraining/flow_grpo.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.If) and "noise_scale" in ast.dump(stmt.test):
                    if any(isinstance(s, ast.Raise) for s in ast.walk(stmt)):
                        print("  grpo-noise: constructor still raises on a deterministic sampler")
                        return []
    return ["FlowGRPOTrainer.__init__ no longer rejects noise_scale <= 0"]


def gate_loss_mask(root: Path) -> list:
    tree = _parse(root / "src/state/train.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "pretrain_step":
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    name = getattr(call.func, "id", getattr(call.func, "attr", ""))
                    if name == "fm_loss":
                        kw = {k.arg for k in call.keywords}
                        if "frame_mask" in kw:
                            print("  loss-mask: pretrain_step passes frame_mask to fm_loss")
                            return []
                        return ["pretrain_step calls fm_loss WITHOUT frame_mask (context frames back in the loss)"]
    return ["pretrain_step or its fm_loss call not found in src/state/train.py"]


def _test_asserts(root: Path, test_name: str, constant=None) -> list:
    tree = _parse(root / "tests/unit/test_state_core.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            if constant is None:
                return []
            for c in ast.walk(node):
                if isinstance(c, ast.Constant) and c.value == constant:
                    return []
            return [f"{test_name} no longer pins the constant {constant}"]
    return [f"unit test {test_name} is gone"]


def gate_param_account(root: Path) -> list:
    errors = _test_asserts(
        root, "test_framedit_parameter_account_matches_bookdit_champion", 953_769
    )
    if not errors:
        print("  param-account: 953,769-parameter parity test present")
    return errors


def gate_validity_test(root: Path) -> list:
    errors = _test_asserts(root, "test_chart_decode_of_noise_is_a_valid_book")
    if not errors:
        print("  validity-test: decode-noise-is-valid test present")
    return errors


GATES = [
    ("default-clock", gate_default_clock),
    ("four-methods", gate_four_methods),
    ("grpo-noise", gate_grpo_noise),
    ("loss-mask", gate_loss_mask),
    ("param-account", gate_param_account),
    ("validity-test", gate_validity_test),
]


def run_all(root: Path) -> int:
    failures = []
    for name, gate in GATES:
        try:
            errs = gate(root)
        except Exception as exc:  # a gate that cannot run has failed
            errs = [f"gate crashed: {exc!r}"]
        for e in errs:
            print(f"  FAIL [{name}] {e}")
        failures.extend(errs)
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print("all gates passed")
    return 0


# ------------------------------------------------------------------ self-test


def _broken_copy(edits: list) -> Path:
    """Copy the audited files into a temp root and apply (path, old, new) edits."""
    tmp = Path(tempfile.mkdtemp(prefix="state_check_selftest_"))
    for rel in [
        "src/state/frames.py",
        "src/state/train.py",
        "src/state/posttraining",
        "tests/unit/test_state_core.py",
    ]:
        src, dst = REPO / rel, tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
    for rel, old, new in edits:
        path = tmp / rel
        text = path.read_text()
        if old not in text:
            raise AssertionError(f"self-test edit target not found in {rel}: {old!r}")
        path.write_text(text.replace(old, new))
    return tmp


def self_test() -> int:
    breaks = [
        ("default-clock", [("src/state/frames.py", "1_000_000  # 1 ms", "2_000_000  # 1 ms")]),
        ("four-methods", [("src/state/posttraining/awm.py", "class AWMTrainer", "class AWMTrainerRenamed")]),
        ("grpo-noise", [("src/state/posttraining/flow_grpo.py", "if flow.noise_scale <= 0.0:", "if False:")]),
        ("loss-mask", [("src/state/train.py", ", frame_mask=1.0 - obs", "")]),
        ("param-account", [("tests/unit/test_state_core.py", "953_769", "953_770")]),
        ("validity-test", [
            ("tests/unit/test_state_core.py",
             "def test_chart_decode_of_noise_is_a_valid_book",
             "def disabled_chart_decode_of_noise_is_a_valid_book"),
        ]),
    ]
    print("self-test: each gate must fail on its broken copy")
    ok = True
    for gate_name, edits in breaks:
        tmp = _broken_copy(edits)
        try:
            gate = dict(GATES)[gate_name]
            errs = []
            try:
                errs = gate(tmp)
            except Exception as exc:
                errs = [repr(exc)]
            if errs:
                print(f"  caught {gate_name}: {errs[0]}")
            else:
                print(f"  MISSED {gate_name}: the broken copy passed")
                ok = False
        finally:
            shutil.rmtree(tmp)
    # And the pristine tree must pass every gate.
    print("self-test: the pristine tree must pass")
    if run_all(REPO) != 0:
        ok = False
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    return run_all(REPO)


if __name__ == "__main__":
    sys.exit(main())
