#!/bin/bash
# Add index of every stock-day of one month (CPU only): the first event-1 row of
# each order id, read once per day instead of once per decision. See
# build_add_index.py. Resubmit after a timeout; finished days are kept.
#
#   ADD_DIR=<dir outside the repo> sbatch --nodes=1 --output=<dir>/%x_%j.out \
#       run/post_training/policy_bench/build_add_index.sbatch
#
#SBATCH --job-name=policy-bench-adds
#SBATCH --account=brics.u6oz
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=04:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ADD_DIR="${ADD_DIR:?ADD_DIR is required}"
PYTHON="${PYTHON:-/opt/cray/pe/python/3.11.7/bin/python3}"
# The simulator's shards; the bench's rollouts start from these days.
SHARD_DIR="${SHARD_DIR:-/projects/public/u6oz/quant/lob_preproc_sp500_squashfs_v2_20261013}"
MANIFEST="${MANIFEST:-/projects/public/u6oz/quant/dataset_q4_2025_jan2026_v3/test.json}"
MONTH="${MONTH:-2026-01}"
N_MOUNTS="${N_MOUNTS:-8}"
WORKERS="${WORKERS:-32}"

MNT="/tmp/policy_bench_adds_${SLURM_JOB_ID}"
cleanup() {
  for k in $(seq "$N_MOUNTS"); do
    fusermount -u "$MNT/$k/$MONTH" 2>/dev/null || fusermount3 -u "$MNT/$k/$MONTH" 2>/dev/null || true
    rmdir "$MNT/$k/$MONTH" "$MNT/$k" 2>/dev/null || true
  done
  rmdir "$MNT" 2>/dev/null || true    # never rm -r: a mount that failed to detach is the read-only shard
}
trap cleanup EXIT
mount_shard() {   # <mount point>
  mkdir -p "$1"
  for sq in /usr/lib/apptainer/bin/squashfuse_ll /usr/bin/squashfuse_ll; do
    [ -x "$sq" ] || continue
    "$sq" -o ro "$SHARD_DIR/shard_$MONTH.squashfs" "$1" 2>/dev/null && [ -n "$(ls -A "$1" 2>/dev/null)" ] && return 0
    fusermount -u "$1" 2>/dev/null || true
  done
  echo "mount failed on $(hostname): $1" >&2; exit 3
}
roots=()
for k in $(seq "$N_MOUNTS"); do mount_shard "$MNT/$k/$MONTH"; roots+=("$MNT/$k"); done
mkdir -p "$ADD_DIR"

cd "$WORKDIR"
source setup_env.sh
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) host=$(hostname)"

"$PYTHON" run/post_training/policy_bench/build_add_index.py \
  --roots "$(IFS=,; echo "${roots[*]}")" --manifest "$MANIFEST" --month "$MONTH" \
  --out_dir "$ADD_DIR" --workers "$WORKERS"
