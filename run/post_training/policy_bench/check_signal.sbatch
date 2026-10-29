#!/bin/bash
# G4 of the policy bench on one GPU: the return signal's test Pearson and our
# processed rows. See check_signal.py for the checks.
#
#   OUT_DIR=<dir outside the repo> PYTHON=<env with torch>/bin/python \
#       sbatch --nodes=1 --output=<dir>/%x_%j.out run/post_training/policy_bench/check_signal.sbatch
#
#SBATCH --job-name=policy-bench-g4
#SBATCH --account=brics.u6oz
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=02:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
PYTHON="${PYTHON:?PYTHON (an env with torch) is required}"
Q=/projects/public/u6oz/quant
# The signal was trained and scored on the bookfix shards with the v2 manifest;
# the simulator's data are the v2_20261013 shards with the v3 manifest.
SIGNAL_SHARDS="${SIGNAL_SHARDS:-$Q/lob_preproc_sp500_squashfs_bookfix20261012}"
SIGNAL_MANIFEST="${SIGNAL_MANIFEST:-$Q/dataset_q4_2025_jan2026_v2/test.json}"
SIM_SHARDS="${SIM_SHARDS:-$Q/lob_preproc_sp500_squashfs_v2_20261013}"
SIM_MANIFEST="${SIM_MANIFEST:-$Q/dataset_q4_2025_jan2026_v3/test.json}"
CKPT="${CKPT:-/projects/public/u6oz/model_zoo/nano-sigma/returnft_lobert_pass1/ckpt.pt}"
MONTH="${MONTH:-2026-01}"

MNT="/tmp/policy_bench_g4_${SLURM_JOB_ID}"
cleanup() {
  for k in sig sim; do fusermount -u "$MNT/$k/$MONTH" 2>/dev/null || fusermount3 -u "$MNT/$k/$MONTH" 2>/dev/null || true; done
  rm -rf "$MNT"
}
trap cleanup EXIT
mount_shard() {   # <shard dir> <mount point>
  mkdir -p "$2"
  for sq in /usr/lib/apptainer/bin/squashfuse_ll /usr/bin/squashfuse_ll; do
    [ -x "$sq" ] || continue
    "$sq" -o ro "$1/shard_$MONTH.squashfs" "$2" 2>/dev/null && [ -n "$(ls -A "$2" 2>/dev/null)" ] && return 0
    fusermount -u "$2" 2>/dev/null || true
  done
  echo "mount failed on $(hostname): $1" >&2; exit 3
}
mount_shard "$SIGNAL_SHARDS" "$MNT/sig/$MONTH"
mount_shard "$SIM_SHARDS" "$MNT/sim/$MONTH"
mkdir -p "$OUT_DIR"

cd "$WORKDIR"
source setup_env.sh
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) host=$(hostname)"

"$PYTHON" run/post_training/policy_bench/check_signal.py \
  --signal_root "$MNT/sig" --signal_manifest "$SIGNAL_MANIFEST" \
  --sim_root "$MNT/sim" --sim_manifest "$SIM_MANIFEST" --ckpt "$CKPT" \
  --workers "$((SLURM_CPUS_PER_TASK - 8))" --out "$OUT_DIR/g4_${SLURM_JOB_ID}.json" "$@"
chmod g+rw "$OUT_DIR/g4_${SLURM_JOB_ID}.json"
