#!/bin/bash
# Cheap gate, run before anything expensive. Target: under two seconds, no GPU,
# no Python import, no filesystem walk.
#
# It exists because of D-X6: an allocation was consumed and a step died two
# seconds in, on a fault that a one-second probe would have caught. Anything
# that can be checked without touching the accelerator belongs here.
#
# Usage:
#   source run/environment/preflight.sh && preflight_or_die "$WORKDIR" python3 rsync
#   run/environment/preflight.sh <workdir> [cmd ...]      # standalone

# --- D-X1 -------------------------------------------------------------------
# The shell skips PATH candidates without the execute bit; Slurm's task
# launcher does not. It resolves a bare command name to the first *name* match
# and calls execve() on it, so a non-executable file earlier in PATH shadows
# the real binary and the step dies with EACCES.
#
# This actually happened: $HOME/.local/bin/env is a sourceable fragment written
# by the uv installer, mode 0664, and ~/.local/bin precedes /usr/bin. `which -a
# env` reported /usr/bin/env, srun disagreed, and step 5836919.420 exited 13:0
# after two seconds with JobName=env.
preflight_check_path_shadow() {
    local cmd="$1" d first="" rc=0
    local -a dirs
    IFS=: read -ra dirs <<< "$PATH"
    for d in "${dirs[@]}"; do
        [ -n "$d" ] || d="."
        if [ -e "$d/$cmd" ] || [ -L "$d/$cmd" ]; then first="$d/$cmd"; break; fi
    done
    if [ -z "$first" ]; then
        echo "[preflight] FAIL: '$cmd' is not on PATH at all" >&2
        return 1
    fi
    if [ -d "$first" ]; then
        echo "[preflight] FAIL: '$cmd' resolves to a DIRECTORY $first" >&2
        echo "[preflight]       srun would execve() it and die with EACCES/EISDIR" >&2
        rc=1
    elif [ ! -x "$first" ]; then
        echo "[preflight] FAIL: '$cmd' resolves to $first which is NOT executable" >&2
        echo "[preflight]       mode=$(stat -c '%A' "$first" 2>/dev/null)" >&2
        echo "[preflight]       the shell skips it, srun does not. Fix by removing it from" >&2
        echo "[preflight]       PATH, or call the command by absolute path." >&2
        rc=1
    else
        echo "[preflight] ok: $cmd -> $first"
    fi
    return $rc
}

# --- D-X2 -------------------------------------------------------------------
# The worktree must be named, must be a git worktree, and its HEAD is recorded
# so the artifact can later be joined back to the code that produced it. A
# directory that merely exists is not evidence of anything.
preflight_check_workdir() {
    local wd="${1:-}"
    if [ -z "$wd" ]; then
        echo "[preflight] FAIL: WORKDIR is empty. It must be set explicitly;" >&2
        echo "[preflight]       there is deliberately no default, because a stale default" >&2
        echo "[preflight]       silently produces new data from old code." >&2
        return 1
    fi
    if [ ! -d "$wd" ]; then
        echo "[preflight] FAIL: WORKDIR does not exist: $wd" >&2
        return 1
    fi
    if ! git -C "$wd" rev-parse --git-dir >/dev/null 2>&1; then
        echo "[preflight] FAIL: WORKDIR is not a git worktree: $wd" >&2
        return 1
    fi
    PREFLIGHT_GIT_SHA="$(git -C "$wd" rev-parse HEAD)"
    # Bounded on purpose: walking the whole tree on Lustre is expensive, and
    # only these two subtrees can change what the job computes.
    if git -C "$wd" diff --quiet HEAD -- src run 2>/dev/null; then
        PREFLIGHT_GIT_DIRTY=0
    else
        PREFLIGHT_GIT_DIRTY=1
    fi
    export PREFLIGHT_GIT_SHA PREFLIGHT_GIT_DIRTY
    echo "[preflight] ok: workdir=$wd sha=${PREFLIGHT_GIT_SHA:0:12} dirty(src,run)=$PREFLIGHT_GIT_DIRTY"

    # The submitter may assert which code this run is meant to be. When they do,
    # a mismatch is a hard stop rather than a surprise discovered weeks later.
    if [ -n "${EXPECT_SHA:-}" ]; then
        case "$PREFLIGHT_GIT_SHA" in
            "$EXPECT_SHA"*) echo "[preflight] ok: HEAD matches EXPECT_SHA=$EXPECT_SHA" ;;
            *) echo "[preflight] FAIL: EXPECT_SHA=$EXPECT_SHA but HEAD=$PREFLIGHT_GIT_SHA" >&2
               return 1 ;;
        esac
    fi
    return 0
}

# Advisory, never fatal. The gate must assert what this run actually depends
# on; a gate that fails on a hazard the run cannot trip teaches people to
# bypass it, which is worse than no gate. Reintroducing a dependency on a
# shadowed name is prevented by the launcher assertion in
# tests/unit/test_launcher_provenance.py, not by failing here.
PREFLIGHT_ADVISE="${PREFLIGHT_ADVISE:-env}"

preflight_advise_path_shadows() {
    local c
    for c in $PREFLIGHT_ADVISE; do
        preflight_check_path_shadow "$c" >/dev/null 2>&1 || \
            echo "[preflight] note: '$c' is shadowed by a non-executable file on PATH." \
                 "Harmless while nothing hands it to srun; see D-X1."
    done
}

preflight_or_die() {
    local wd="$1"; shift
    local rc=0
    preflight_check_workdir "$wd" || rc=1
    local c
    for c in "$@"; do
        preflight_check_path_shadow "$c" || rc=1
    done
    preflight_advise_path_shadows
    if [ "$rc" -ne 0 ]; then
        echo "[preflight] ABORT before consuming the allocation" >&2
        return 1
    fi
    echo "[preflight] all checks passed"
    return 0
}

# Standalone invocation.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -uo pipefail
    WD="${1:-${WORKDIR:-}}"
    shift 2>/dev/null || true
    if [ "$#" -eq 0 ]; then set -- python3 rsync; fi
    preflight_or_die "$WD" "$@" || exit 1
fi
