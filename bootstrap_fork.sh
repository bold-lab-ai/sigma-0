#!/usr/bin/env bash
# ============================================================================
# bootstrap_fork.sh — one-command setup for a fresh sigma-0 clone or fork
# ============================================================================
# Run once from the repository root after cloning:
#
#     bash bootstrap_fork.sh
#
# Idempotent: safe to re-run after pulling.
#
# It exists because the steps below are the ones that fail *later*, expensively:
# a missing data symlink surfaces as a dataloader error inside an allocation, a
# missing log directory makes sbatch reject the job after you queued it, and a
# broken import surfaces on the compute node rather than here. All of it is
# checkable in a couple of seconds on a login node.
#
#   1. umask 002 (shared-filesystem group convention)
#   2. git submodules (no-op today — nothing is a submodule)
#   3. data/ symlinks from tools/data_symlinks.tsv (existing links untouched;
#      unreadable targets are skipped with a warning, never guessed)
#   4. log dir creation (sbatch --output dirs must exist BEFORE submission)
#   5. login-node gates: core-tier import check + launcher syntax checks
#   6. prints the next steps
#
# Lustre note: step 3 probes each target with a single exact-path test. It never
# lists or walks anything under /lus.
# ============================================================================
set -euo pipefail
umask 002
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$ROOT"

fail=0
note() { printf '  %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }

step "1/6 group permissions"
note "umask set to 002 for this run; setup_env.sh sets it per shell."

step "2/6 git submodules"
if [ -f .gitmodules ]; then
    git submodule update --init --recursive
    note "submodules initialized."
else
    note "no submodules — nothing to do."
fi

step "3/6 data symlinks (tools/data_symlinks.tsv)"
if [ -f tools/data_symlinks.tsv ]; then
    mkdir -p data && chmod g+rwX data
    while IFS=$'\t' read -r name target _note; do
        case "$name" in ''|\#*) continue ;; esac
        link="data/$name"
        if [ -L "$link" ] || [ -e "$link" ]; then
            note "exists:   $link"
            continue
        fi
        # Cheap exact-path probe only — never scan /lus targets. A directory
        # needs +x as well as +r: readable-but-not-traversable is a real mode
        # on cross-account shared paths, and it surfaces later as a loader
        # error on a compute node rather than here.
        if [ -r "$target" ] && { [ ! -d "$target" ] || [ -x "$target" ]; }; then
            ln -sfn "$target" "$link"
            note "created:  $link -> $target"
        else
            note "SKIPPED (target not readable from this account): $link -> $target"
            fail=1
        fi
    done < tools/data_symlinks.tsv
else
    note "tools/data_symlinks.tsv missing — skipping."
fi

step "4/6 log directory"
mkdir -p logs && chmod g+rwX logs
note "logs/ ready (train_full_autoreg.batch writes --output there)."

step "5/6 login-node gates"
PYTHONPATH="$ROOT/src" python3 run/migrate/check_imports.py --tiers core
bash -n run/base_model/train_full_autoreg.batch run/base_model/node_wrapper.sh
bash -n run/environment/preflight.sh setup_env.sh
note "core imports + launcher syntax OK."

step "6/6 next steps"
cat <<'EOS'
  source setup_env.sh
  # full gates under the training conda python (docs/environments.md, section 1):
  #   PYTHONPATH=src <conda-python> run/migrate/check_imports.py
  #   PYTHONPATH=src <conda-python> -m pytest tests -q --ignore=tests/integration
  # scoring venv (LOB-Bench), once per clone:
  #   bash envs/lobbench_venv_setup.sh <BASE_PY>
  # first training smoke:
  #   sbatch run/base_model/smoke_train.sbatch
  Docs: docs/README.md   Conventions: docs/conventions.md
EOS

if [ "$fail" -ne 0 ]; then
    echo "NOTE: some data symlinks were skipped (unreadable targets) — see above." >&2
fi
