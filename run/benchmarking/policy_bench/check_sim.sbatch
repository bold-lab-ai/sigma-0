#!/bin/bash
# G6, G7 and the signal half of G8 on one GPU: SimMarket with newtok_a_r1.
# See check_sim.py for the checks.
#
#   OUT_DIR=<dir outside the repo> ADD_DIR=<build_add_index.sbatch ADD_DIR> PYTHON=<env with torch>/bin/python \
#       sbatch --nodes=1 --output=<dir>/%x_%j.out run/benchmarking/policy_bench/check_sim.sbatch
#
#SBATCH --job-name=policy-bench-sim
#SBATCH --account=brics.u6oz
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=03:00:00

set -euo pipefail
echo "start host=$(hostname) job=${SLURM_JOB_ID:-} $(date -Is)"

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
PYTHON="${PYTHON:?PYTHON (an env with torch) is required}"
ADD_DIR="${ADD_DIR:?ADD_DIR (build_add_index.sbatch, sim shards) is required}"
Q=/projects/public/u6oz/quant
SHARD_DIR="${SHARD_DIR:-$Q/lob_preproc_sp500_squashfs_v2_20261013}"
MANIFEST="${MANIFEST:-$Q/dataset_q4_2025_jan2026_v3/test.json}"
Z=/projects/public/u6oz/model_zoo/nano-sigma
SIM_CKPT="${SIM_CKPT:-$Z/newtok_a_r1/ckpt.pt}"
SIGNAL_CKPT="${SIGNAL_CKPT:-$Z/returnft_lobert_pass1/ckpt.pt}"
MONTH="${MONTH:-2026-01}"

MNT="/tmp/policy_bench_sim_${SLURM_JOB_ID}"
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

# Fail fast when WORKDIR or PYTHON cannot be read: job 7860323 hung 2 h 21 min before its
# first line on a node whose /home had stopped answering.
timeout -s KILL 120 bash -c 'cat "$1/setup_env.sh" >/dev/null && git -C "$1" rev-parse HEAD >/dev/null && "$2" -c 0' \
  _ "$WORKDIR" "$PYTHON" || { echo "WORKDIR or PYTHON not readable on $(hostname) within 120 s" >&2; exit 4; }
cd "$WORKDIR"
source setup_env.sh
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) host=$(hostname)"

"$PYTHON" run/benchmarking/policy_bench/check_sim.py \
  --shard_root "$MNT" --manifest "$MANIFEST" --add_dir "$ADD_DIR" \
  --sim_ckpt "$SIM_CKPT" --signal_ckpt "$SIGNAL_CKPT" --out "$OUT_DIR/sim_${SLURM_JOB_ID}.json" "$@"
chmod g+rw "$OUT_DIR"/sim_"${SLURM_JOB_ID}"*
