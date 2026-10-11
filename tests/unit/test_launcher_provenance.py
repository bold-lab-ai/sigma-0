"""Regression pins for the execution class, D-X1 through D-X6.

Every defect in class X shares a shape: the artifact and the code that produced
it come apart, and nothing in the pipeline notices. These tests are the
noticing. They are stdlib only and touch no accelerator, so they run in CI.

See docs/fidelity_defect_taxonomy.md for what each ID means.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENVDIR = ROOT / "run" / "environment"
PREFLIGHT = ENVDIR / "preflight.sh"

BUILD = ENVDIR / "build_episodes.sbatch"
MULTINODE = ENVDIR / "run_episodes_multinode.sh"
CLAUDE_PREFLIGHT = ENVDIR / "preflight_claude_p.sbatch"
ATTACH_RUN = ROOT / "tools" / "dev" / "attach_run.sh"

LAUNCHERS = [BUILD, MULTINODE, CLAUDE_PREFLIGHT]
SBATCHES = [BUILD, CLAUDE_PREFLIGHT]

NEUTRAL_LOG_ROOT = "/lus/lfs1aip2/projects/public/u6gb/slurm_logs"
# The worktree that produced tasks/agentic_mm_20260731T164839Z, cut before the
# fixes those episodes were assumed to contain.
STALE_WORKTREE = "agentic-mm-runtime-20260731"


# ----------------------------------------------------------------- D-X2 ----
@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_no_launcher_hardcodes_the_stale_worktree(path):
    """A default WORKDIR is not a convenience, it is a silent failure mode:
    new data produced by old code, with the artifact's mtime falsely reading
    as freshness."""
    hits = [i + 1 for i, line in enumerate(path.read_text().splitlines())
            if STALE_WORKTREE in line]
    assert not hits, f"{path.name} still references {STALE_WORKTREE} at lines {hits}"


@pytest.mark.parametrize("path", [BUILD, MULTINODE], ids=lambda p: p.name)
def test_workdir_is_required_not_defaulted(path):
    """`${WORKDIR:?...}` aborts when unset; `${WORKDIR:-...}` substitutes a
    default. Only the first makes the choice of code explicit."""
    text = path.read_text()
    assert re.search(r'WORKDIR="\$\{WORKDIR:\?', text), \
        f"{path.name} must require WORKDIR with :? rather than defaulting it"
    assert not re.search(r'WORKDIR="\$\{WORKDIR:-', text), \
        f"{path.name} still provides a default WORKDIR"


# ----------------------------------------------------------------- D-X3 ----
@pytest.mark.parametrize("path", SBATCHES, ids=lambda p: p.name)
def test_sbatch_output_goes_to_the_neutral_log_root(path):
    """Slurm reads #SBATCH directives before any shell expansion, so an output
    path cannot follow $WORKDIR. Keying on the job id instead is what keeps the
    log, the per-rank logs and the manifest joinable."""
    outs = [l for l in path.read_text().splitlines() if l.startswith("#SBATCH --output")]
    assert outs, f"{path.name} declares no --output"
    for line in outs:
        assert NEUTRAL_LOG_ROOT in line, f"{path.name}: {line}"


@pytest.mark.parametrize("path", [BUILD, MULTINODE], ids=lambda p: p.name)
def test_log_dir_is_not_inside_a_worktree(path):
    text = path.read_text()
    assert not re.search(r'LOG_DIR="\$WORKDIR', text), \
        f"{path.name} writes logs inside the worktree; use the neutral root"


# ----------------------------------------------------------------- D-X6 ----
def test_build_runs_preflight_before_spending_the_allocation():
    lines = BUILD.read_text().splitlines()

    def first(pred):
        return next((i for i, l in enumerate(lines) if pred(l)), None)

    gate = first(lambda l: "preflight_or_die" in l)
    assert gate is not None, "build_episodes.sbatch never calls preflight_or_die"
    assert first(lambda l: "source" in l and "preflight.sh" in l) is not None

    # Anything that costs real time must come after the gate.
    expensive = first(lambda l: l.strip().startswith('"$PYTHON"')
                      or "module load" in l or "run_generation " in l)
    assert expensive is not None and gate < expensive, (
        f"preflight at line {gate + 1} does not precede the first expensive "
        f"step at line {(expensive or 0) + 1}"
    )


# ----------------------------------------------------------------- D-X2 ----
def test_manifest_records_the_code_that_produced_the_data():
    """An artifact that cannot name its own commit can only offer its mtime,
    and mtime is not freshness."""
    text = BUILD.read_text()
    for key in ('"workdir"', '"git_sha"', '"git_dirty_src_run"'):
        assert key in text, f"manifest does not record {key}"


# ----------------------------------------------------------------- D-X1 ----
def _run_shadow_check(cmd: str, path_prefix: str) -> int:
    """Call preflight_check_path_shadow with a controlled PATH."""
    script = f'source "{PREFLIGHT}"; preflight_check_path_shadow "{cmd}"'
    env = dict(os.environ, PATH=f"{path_prefix}:{os.environ.get('PATH', '')}")
    return subprocess.run(["bash", "-c", script], env=env,
                          capture_output=True, text=True).returncode


def test_path_shadow_check_rejects_a_non_executable_first_match():
    """The exact fault that killed step 5836919.420: the shell skips PATH
    candidates without the execute bit, Slurm's launcher does not, so a
    sourceable fragment named like a real binary wins the resolution and the
    step dies with EACCES two seconds in."""
    with tempfile.TemporaryDirectory() as d:
        victim = Path(d) / "zzprobe"
        victim.write_text("#!/bin/sh\n# sourceable fragment, not a program\n")
        victim.chmod(0o664)
        assert _run_shadow_check("zzprobe", d) == 1, "non-executable first match accepted"

        victim.chmod(victim.stat().st_mode | stat.S_IXUSR)
        assert _run_shadow_check("zzprobe", d) == 0, "executable first match rejected"


@pytest.mark.parametrize("path", LAUNCHERS + [ATTACH_RUN],
                         ids=lambda p: p.name)
def test_no_launcher_hands_a_bare_command_name_to_srun(path):
    """D-X1 at the source rather than at the detector. run_episodes_multinode.sh
    used `srun ... env VAR=... bash script`, which is the exact shape that
    resolves to the wrong file. Variables belong in --export and the command
    must be an absolute interpreter."""
    text = path.read_text()
    # Join continuation lines so a command split across backslashes is seen whole.
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    for line in joined.splitlines():
        s = line.strip()
        if not s.startswith("srun ") and " srun " not in f" {s}":
            continue
        if s.lstrip().startswith("#"):
            continue
        # Collapse quoted regions before splitting. An option value may hold a
        # quoted space (--export="A=1,B=x y"), and shell parameter expansions
        # such as ${A[$i]# } defeat shlex, so neither naive splitting nor a
        # shell lexer finds the command position reliably. Blanking the quoted
        # spans first makes plain whitespace splitting exact for this purpose.
        flat = re.sub(r"'[^']*'", "'Q'", re.sub(r'"[^"]*"', '"Q"', s))
        toks = flat.split()
        if "srun" not in toks:
            continue
        idx = toks.index("srun") + 1
        while idx < len(toks) and toks[idx].startswith("-"):
            idx += 1
        if idx >= len(toks):
            continue
        cmd = toks[idx]
        assert cmd != "env", f"{path.name}: srun's first word is a bare `env`: {s[:120]}"
        assert cmd.startswith("/") or cmd.startswith("$") or cmd.startswith('"'), (
            f"{path.name}: srun's command should be an absolute path, got {cmd!r}"
        )


def test_path_shadow_check_rejects_a_directory():
    """execve() on a directory fails too, and a directory named like a command
    is easier to create by accident than a non-executable file."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "zzprobe").mkdir()
        assert _run_shadow_check("zzprobe", d) == 1


def test_path_shadow_check_reports_a_command_that_is_absent():
    """A name that resolves nowhere must fail loudly, not pass by default."""
    with tempfile.TemporaryDirectory() as d:
        assert _run_shadow_check("zznotacommand", d) == 1


# ----------------------------------------------------------------- gate ----
def test_workdir_gate_rejects_empty_missing_and_mismatched():
    def gate(wd: str, expect: str | None = None) -> int:
        env = dict(os.environ)
        if expect:
            env["EXPECT_SHA"] = expect
        else:
            env.pop("EXPECT_SHA", None)
        script = f'source "{PREFLIGHT}"; preflight_check_workdir "{wd}"'
        return subprocess.run(["bash", "-c", script], env=env,
                              capture_output=True, text=True).returncode

    assert gate("") == 1, "empty WORKDIR accepted"
    assert gate("/nonexistent-zz") == 1, "missing WORKDIR accepted"
    assert gate(str(ROOT)) == 0, "a real worktree was rejected"
    assert gate(str(ROOT), expect="deadbeefdead") == 1, "EXPECT_SHA mismatch accepted"

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert gate(str(ROOT), expect=head[:7]) == 0, "matching EXPECT_SHA rejected"


def test_preflight_is_executable():
    assert os.access(PREFLIGHT, os.X_OK), "preflight.sh is not executable"


# ----------------------------------------------------------------- D-T3 ----
def test_manifest_declares_the_engine_configuration():
    """Without it a replay must guess, and the defaults are not what the
    generator runs: book_depth 10 against 500. Guessing wrong produces a
    different book from the same messages, which is class T in one line."""
    text = BUILD.read_text()
    assert '"engine"' in text, "the manifest declares no engine configuration"
    for key in ('"wide_levels"', '"book_depth"', '"nOrders"', '"tick_size"', '"init_id"'):
        assert key in text, f"manifest engine block does not record {key}"


def test_the_declared_depth_is_the_depth_that_is_used():
    """A manifest that can disagree with the run is worse than no manifest.
    wide_levels is one variable feeding the mount, the generation flag and the
    record, so the three cannot drift apart."""
    text = BUILD.read_text()
    assert re.search(r'WIDE_LEVELS="\$\{WIDE_LEVELS:-\d+\}"', text), \
        "wide_levels must be a variable, not repeated literals"
    assert "--wide_levels=\"$WIDE_LEVELS\"" in text, "generation must use the variable"
    assert '"$WIDE_LEVELS"' in text, "the wide-book mount must use the variable"
    # The literal 500 must not survive anywhere that decides a width.
    for line in text.splitlines():
        if "wide_levels=500" in line.replace(" ", "") or "wide_book" in line and " 500" in line:
            raise AssertionError(f"hardcoded width survives: {line.strip()}")


def test_derived_engine_fields_match_how_the_generator_derives_them():
    """sample_new uses nOrders = max(100, wide_levels*2 + 50) and
    book_depth = wide_levels. The record states the same formula rather than a
    number someone typed."""
    text = BUILD.read_text()
    assert '"book_depth": ${WIDE_LEVELS}' in text
    assert '"nOrders": max(100, ${WIDE_LEVELS} * 2 + 50)' in text


# ----------------------------------------------------------------- D-X7 ----
def test_manifest_records_the_node():
    """Generation is deterministic within a node and not across nodes, so the
    node is part of the configuration rather than incidental metadata. A record
    that does not name it cannot be regenerated, which is the same shape as
    D-T3: an incomplete record is an unreproducible artifact."""
    assert '"node"' in BUILD.read_text(), "the manifest does not record the node"
