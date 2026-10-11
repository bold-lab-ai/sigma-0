#!/bin/bash
# Attach a workload to an allocation that is already RUNNING, correctly.
#
# Two defects are fixed by construction here rather than by remembering not to
# trip on them.
#
# D-X4: a `tmux new-session -d` started from inside an srun step is a child of
# that step and is killed when the step ends. The fix is not "avoid tmux", it
# is to remove the indirection: the payload script IS the srun step, so there
# is no second process to be reaped.
#
# D-X1: never hand a bare command name to srun as the first word. Slurm
# resolves it through PATH without testing the execute bit, so a non-executable
# file of the same name earlier in PATH wins and the step dies with EACCES.
# This launcher always execs an absolute /bin/bash and passes environment
# through --export, never through a leading `env`.
#
# Usage:
#   attach_run.sh <jobid> <node> <script> [VAR=VAL ...]
#
# Example:
#   tools/dev/attach_run.sh 5836919 nid010488 run/environment/build_episodes.sbatch \
#       WORKDIR=/lus/.../sigma-0-worktrees/regen-fixed-20260801 STOCK=GS

set -uo pipefail

JOBID="${1:?usage: $0 <jobid> <node> <script> [VAR=VAL ...]}"
NODE="${2:?missing node}"
SCRIPT="${3:?missing script}"
shift 3

LOG_ROOT="${SIGMA_LOG_ROOT:-/lus/lfs1aip2/projects/public/u6gb/slurm_logs}"
mkdir -p "$LOG_ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_ROOT/attach_${JOBID}_${NODE}_${STAMP}.out"

[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT" >&2; exit 1; }

# Collect VAR=VAL pairs for --export, and pull WORKDIR out for the preflight.
EXPORTS="ALL"
WD=""
for kv in "$@"; do
    case "$kv" in
        *=*) EXPORTS="$EXPORTS,$kv"
             case "$kv" in WORKDIR=*) WD="${kv#WORKDIR=}" ;; esac ;;
        *) echo "not a VAR=VAL pair: $kv" >&2; exit 1 ;;
    esac
done

# --- gate on the login node, before touching the allocation -----------------
# This script lives in tools/dev/; the preflight stays with the launchers.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PREFLIGHT="$REPO_ROOT/run/environment/preflight.sh"
if [ -f "$PREFLIGHT" ] && [ -n "$WD" ]; then
    # shellcheck source=/dev/null
    source "$PREFLIGHT"
    preflight_or_die "$WD" python3 rsync || exit 1
fi

# Inherited SLURM_* would make srun think it is a subtask of this shell's own
# allocation. Strip them so --jobid means what it says.
for v in $(env | sed -n 's/^\(SLURM_[A-Z_]*\)=.*/\1/p'); do unset "$v"; done

echo "attaching to job=$JOBID node=$NODE"
echo "log: $LOG"

# No `env` as the first word; an absolute interpreter; the payload is the step.
srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --nodelist="$NODE" \
     --export="$EXPORTS" --cpu-bind=none \
     /bin/bash "$SCRIPT" > "$LOG" 2>&1 &
SRUN_PID=$!
echo "srun pid=$SRUN_PID"
echo "$LOG"
wait "$SRUN_PID"
RC=$?
echo "srun exited rc=$RC"
exit "$RC"
