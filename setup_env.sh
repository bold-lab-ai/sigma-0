#!/usr/bin/env bash
# Per-shell environment for sigma-0.
#
# Source this file (do not execute it):
#
#     source setup_env.sh
#
# Idempotent — safe to source repeatedly (PYTHONPATH is only prepended once).
#
# It sets only what every shell in this repo needs. It deliberately does NOT
# activate a conda environment: which family you need is decided by the orbax
# version that wrote the checkpoint you are loading, not by preference, and the
# activation pattern (by prefix, never `conda activate`) is documented in
# docs/environments.md, section 1.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# First-party packages live under src/ (repo convention: PYTHONPATH=src, the
# same root tests/conftest.py inserts).
case ":${PYTHONPATH:-}:" in
    *":$ROOT/src:"*) : ;;  # already on PYTHONPATH — do nothing
    *) export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# Group-writable outputs on the shared filesystem (group brics.u6gb): dirs get
# g+rwx, files g+rw. Anything this repo writes to /lus must stay readable by
# the rest of the team.
umask 002

# NOT set here, on purpose: MAMBA3_LEGACY_NORM. Unset means fixed-norm weights,
# which is right for every checkpoint trained after 2026-04-22. Export it to 1
# only in the shell that loads a pre-2026-04-22 Mamba3 checkpoint — see
# docs/checkpoints.md and src/base_model/models/legacy_norm.py.
