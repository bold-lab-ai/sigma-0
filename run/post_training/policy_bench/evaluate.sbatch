#!/bin/bash
# Evaluate one policy on the policy bench (one GPU). See evaluate.py.
# Pass the policy and the output with the arguments of evaluate.py.
#
#   PYTHON=<env with torch>/bin/python sbatch --nodes=1 --output=<dir>/%x_%j.out \
#       run/post_training/policy_bench/evaluate.sbatch --policy <module:Class> --kwargs <json> --out <file.json>
#
#SBATCH --job-name=policy-bench-eval
#SBATCH --account=brics.u6oz
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=04:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${PYTHON:?PYTHON (an env with torch) is required}"
CONFIG="${CONFIG:-configs/policy_bench/v1.yaml}"
Q=/projects/public/u6oz/quant
SHARD_DIR="${SHARD_DIR:-$Q/lob_preproc_sp500_squashfs_v2_20261013}"   # the config's shards
MONTH="${MONTH:-2026-01}"

MNT="/tmp/policy_bench_eval_${SLURM_JOB_ID}"
cleanup() { fusermount -u "$MNT/$MONTH" 2>/dev/null || fusermount3 -u "$MNT/$MONTH" 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$MNT/$MONTH"
ok=0
for sq in /usr/lib/apptainer/bin/squashfuse_ll /usr/bin/squashfuse_ll; do
  [ -x "$sq" ] || continue
  "$sq" -o ro "$SHARD_DIR/shard_$MONTH.squashfs" "$MNT/$MONTH" 2>/dev/null \
    && [ -n "$(ls -A "$MNT/$MONTH" 2>/dev/null)" ] && { ok=1; break; }
  fusermount -u "$MNT/$MONTH" 2>/dev/null || true
done
[ "$ok" = 1 ] || { echo "mount failed on $(hostname): $MONTH" >&2; exit 3; }

cd "$WORKDIR"
source setup_env.sh
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) host=$(hostname)"

"$PYTHON" run/post_training/policy_bench/evaluate.py --config "$CONFIG" --shard_root "$MNT" "$@"
