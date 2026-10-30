#!/bin/bash
# G5 of the policy bench on one GPU: ReplayMarket + the sign rule against the
# PnL bench. See check_replay.py for the check.
#
#   OUT_DIR=<dir outside the repo> ADD_DIR=<build_add_index.sbatch ADD_DIR> PYTHON=<env with torch>/bin/python \
#       sbatch --nodes=1 --output=<dir>/%x_%j.out run/post_training/policy_bench/check_replay.sbatch
#
#SBATCH --job-name=policy-bench-g5
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
ADD_DIR="${ADD_DIR:?ADD_DIR (build_add_index.sbatch, sim shards) is required}"
Q=/projects/public/u6oz/quant
SHARD_DIR="${SHARD_DIR:-$Q/lob_preproc_sp500_squashfs_v2_20261013}"
SIGNAL_MANIFEST="${SIGNAL_MANIFEST:-$Q/dataset_q4_2025_jan2026_v2/test.json}"
SIM_MANIFEST="${SIM_MANIFEST:-$Q/dataset_q4_2025_jan2026_v3/test.json}"
CKPT="${CKPT:-/projects/public/u6oz/model_zoo/nano-sigma/returnft_lobert_pass1/ckpt.pt}"
MONTH="${MONTH:-2026-01}"
WORKERS="${WORKERS:-32}"

MNT="/tmp/policy_bench_g5_${SLURM_JOB_ID}"
cleanup() { fusermount -u "$MNT/$MONTH" 2>/dev/null || fusermount3 -u "$MNT/$MONTH" 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$MNT/$MONTH" "$OUT_DIR"
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

"$PYTHON" run/post_training/policy_bench/check_replay.py \
  --shard_root "$MNT" --signal_manifest "$SIGNAL_MANIFEST" --sim_manifest "$SIM_MANIFEST" \
  --add_dir "$ADD_DIR" --ckpt "$CKPT" --workers "$WORKERS" --out "$OUT_DIR/g5_${SLURM_JOB_ID}.json" "$@"
chmod g+rw "$OUT_DIR"/g5_"${SLURM_JOB_ID}"*
