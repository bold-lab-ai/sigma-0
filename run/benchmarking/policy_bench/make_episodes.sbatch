#!/bin/bash
# Draw an episode set of one split (CPU; mounts the month's shard). See make_episodes.py.
#
#   PYTHON=<env>/bin/python MONTH=2026-01 sbatch --nodes=1 --output=<dir>/%x_%j.out \
#       run/benchmarking/policy_bench/make_episodes.sbatch --manifest <manifest> --split dev \
#       --n 200 --seed <seed> --out <file.json>
#
#SBATCH --job-name=policy-bench-episodes
#SBATCH --account=brics.u6oz
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
echo "start host=$(hostname) job=${SLURM_JOB_ID:-} $(date -Is)"

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${PYTHON:?PYTHON (an env with torch) is required}"
Q=/projects/public/u6oz/quant
SHARD_DIR="${SHARD_DIR:-$Q/lob_preproc_sp500_squashfs_v2_20261013}"   # the config's shards
MONTH="${MONTH:?MONTH (YYYY-MM, the shard to mount) is required}"

MNT="/tmp/policy_bench_episodes_${SLURM_JOB_ID}"
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

# Fail fast when WORKDIR or PYTHON cannot be read: job 7860323 hung 2 h 21 min before its
# first line on a node whose /home had stopped answering.
timeout -s KILL 120 bash -c 'cat "$1/setup_env.sh" >/dev/null && git -C "$1" rev-parse HEAD >/dev/null && "$2" -c 0' \
  _ "$WORKDIR" "$PYTHON" || { echo "WORKDIR or PYTHON not readable on $(hostname) within 120 s" >&2; exit 4; }
cd "$WORKDIR"
source setup_env.sh
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) host=$(hostname)"

"$PYTHON" run/benchmarking/policy_bench/make_episodes.py --shard_root "$MNT" --month "$MONTH" "$@"
