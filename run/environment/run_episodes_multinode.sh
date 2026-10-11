#!/bin/bash
# Spread Stage A generation across several nodes of already-running allocations.
#
# Why this is worth doing, and what it should buy
# -----------------------------------------------
# The stages are independent: one background-flow generation plus J signal
# rollouts, all over the same conditioning windows (index selection inside
# runtime/inference.py is pinned to seed 42 regardless of --seed, so the
# windows match without coordination). They therefore parallelise across nodes
# with no communication at all.
#
# The gain to spend this on is statistical power, not wall clock. Eight windows
# cannot clear a 2-sigma threshold no matter how fast they are produced. Five
# nodes let the same wall clock buy five times the windows, which is the
# difference between a pipeline smoke and an evaluation whose numbers mean
# something. Scale N_SEQUENCES with the node count rather than leaving it at 8.
#
# Usage:
#   run_episodes_multinode.sh "<jobid:node> <jobid:node> ..." [N_SEQUENCES] [STOCK]
#
# Example (4 nodes from one allocation plus 1 from another, 40 windows):
#   run_episodes_multinode.sh \
#     "5836919:nid010138 5836919:nid010414 5836919:nid010488 5836919:nid010873 5827830:nid010937" \
#     40 GS

set -uo pipefail

# No default (D-X2): a stale default silently runs new data through old code.
WORKDIR="${WORKDIR:?WORKDIR must be set explicitly, e.g. WORKDIR=/lus/.../sigma-0-worktrees/<name> $0 ...}"
TARGETS="${1:?usage: $0 \"<jobid:node> ...\" [N_SEQUENCES] [STOCK]}"
N_SEQUENCES="${2:-40}"
STOCK="${3:-GS}"

BATCH_SIZE="${BATCH_SIZE:-2}"
WORLD_SIZE=4
BG_SEED="${BG_SEED:-2026}"
ALL_SIGNAL_SEEDS="${ALL_SIGNAL_SEEDS:-3001 3002 3003 3004 3005 3006 3007 3008}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TASK_ROOT="/lus/lfs1aip2/projects/public/u6gb/tasks/agentic_mm_${STAMP}"
LOG_DIR="${SIGMA_LOG_ROOT:-/lus/lfs1aip2/projects/public/u6gb/slurm_logs}"
mkdir -p "$LOG_DIR" "$TASK_ROOT"

# Inherited SLURM_* would make srun believe it is confined to whatever
# allocation this shell already belongs to, and it would refuse the attach with
# "Only allocated 1 nodes asked for N". --jobid does not override them.
strip_slurm() { for v in $(compgen -v SLURM_); do unset "$v"; done; }

read -ra NODES <<<"$TARGETS"
N_NODES=${#NODES[@]}
echo "[multinode] stamp=$STAMP  nodes=$N_NODES  stock=$STOCK  windows=$N_SEQUENCES"
echo "[multinode] task_root=$TASK_ROOT"

if [ $(( N_SEQUENCES % (WORLD_SIZE * BATCH_SIZE) )) -ne 0 ]; then
    echo "[multinode] FATAL: N_SEQUENCES=$N_SEQUENCES must divide by world_size*batch=$((WORLD_SIZE*BATCH_SIZE))" >&2
    exit 2
fi

# --- gate: every target node must actually have idle GPUs -------------------
# An allocation showing RUNNING says nothing about whether its GPUs are busy;
# only compute processes do. A pre-start check is also not a lease, so this is
# re-checked immediately before launching rather than trusted from earlier.
echo "[multinode] --- GPU gate ---"
BUSY=0
for t in "${NODES[@]}"; do
    JID="${t%%:*}"; NODE="${t##*:}"
    OUT=$( strip_slurm; timeout 90 srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 \
             --nodelist="$NODE" --cpu-bind=none \
             /bin/bash -c 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l' 2>/dev/null | tail -1 )
    OUT="${OUT:-unknown}"
    if [ "$OUT" = "0" ]; then
        echo "  $NODE (job $JID): idle"
    else
        echo "  $NODE (job $JID): BUSY ($OUT compute processes) -- will not launch here"
        BUSY=$((BUSY + 1))
    fi
done
if [ "$BUSY" -gt 0 ]; then
    echo "[multinode] FATAL: $BUSY of $N_NODES targets are busy. Someone else is using them; not competing." >&2
    exit 3
fi

# --- assign stages to nodes -------------------------------------------------
# Node 0 takes the background flow (and writes the manifest); the signal seeds
# are dealt round-robin across the remaining nodes.
declare -a SEEDS_FOR
for ((i = 0; i < N_NODES; i++)); do SEEDS_FOR[$i]=""; done
i=0
SIGNAL_NODES=$(( N_NODES > 1 ? N_NODES - 1 : 1 ))
for s in $ALL_SIGNAL_SEEDS; do
    idx=$(( N_NODES > 1 ? 1 + (i % SIGNAL_NODES) : 0 ))
    SEEDS_FOR[$idx]="${SEEDS_FOR[$idx]} $s"
    i=$((i + 1))
done

echo "[multinode] --- stage assignment ---"
for ((i = 0; i < N_NODES; i++)); do
    NODE="${NODES[$i]##*:}"
    BG=$([ "$i" -eq 0 ] && echo "background" || echo "-")
    echo "  $NODE: bg=$BG seeds=[${SEEDS_FOR[$i]# }]"
done

# --- launch -----------------------------------------------------------------
PIDS=()
for ((i = 0; i < N_NODES; i++)); do
    JID="${NODES[$i]%%:*}"; NODE="${NODES[$i]##*:}"
    RUN_BG=$([ "$i" -eq 0 ] && echo 1 || echo 0)
    WRITE_MF=$([ "$i" -eq 0 ] && echo 1 || echo 0)
    OUT="$LOG_DIR/multinode_${STAMP}_${NODE}.out"
    (
        strip_slurm
        # Variables go through --export, and the command is an absolute
        # interpreter. Never `env VAR=... cmd` here (D-X1): srun resolves its
        # first word through PATH without testing the execute bit, so
        # ~/.local/bin/env -- a sourceable fragment, mode 0664 -- wins the
        # resolution and the step dies with EACCES two seconds in.
        srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$NODE" \
             --gpus-per-node=4 --cpu-bind=none \
             --export="ALL,WORKDIR=$WORKDIR,STAMP=$STAMP,TASK_ROOT=$TASK_ROOT,SHARD=$NODE,STOCK=$STOCK,N_SEQUENCES=$N_SEQUENCES,BATCH_SIZE=$BATCH_SIZE,BG_SEED=$BG_SEED,RUN_BACKGROUND=$RUN_BG,WRITE_MANIFEST=$WRITE_MF,SIGNAL_SEEDS=${SEEDS_FOR[$i]# }" \
             /bin/bash "$WORKDIR/run/environment/build_episodes.sbatch"
        echo "[multinode] $NODE srun exit=$?"
    ) > "$OUT" 2>&1 &
    PIDS+=("$!")
    echo "[multinode] launched $NODE -> $OUT"
done

echo "[multinode] waiting for $N_NODES shards..."
RC=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[multinode] shard ${NODES[$i]##*:} FAILED" >&2
        RC=1
    fi
done

echo "[multinode] all shards finished (rc=$RC)"
echo "[multinode] task_root=$TASK_ROOT"
exit "$RC"
