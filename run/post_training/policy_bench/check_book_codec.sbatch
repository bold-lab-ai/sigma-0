#!/bin/bash
# G1 + G2 of the policy bench on real stock-days (CPU only): the L3 book, the
# newtok codec and the legality rules. See check_book_codec.py for the checks.
#
#   OUT_DIR=<dir outside the repo> sbatch --nodes=1 --output=<dir>/%x_%j.out \
#       run/post_training/policy_bench/check_book_codec.sbatch
#
#SBATCH --job-name=policy-bench-g1g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=02:00:00

set -euo pipefail
echo "start host=$(hostname) job=${SLURM_JOB_ID:-} $(date -Is)"

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
PYTHON="${PYTHON:-/opt/cray/pe/python/3.11.7/bin/python3}"
# The shards newtok_a_r1 was trained on, with the v3 manifest that adds the full-book files.
SHARD_DIR="${SHARD_DIR:-/projects/public/u6oz/quant/lob_preproc_sp500_squashfs_v2_20261013}"
MANIFEST="${MANIFEST:-/projects/public/u6oz/quant/dataset_q4_2025_jan2026_v3/test.json}"
MONTH="${MONTH:-2026-01}"
N_DAYS="${N_DAYS:-40}"
SEED="${SEED:-0}"

MNT="/tmp/policy_bench_g1g2_${SLURM_JOB_ID}"
cleanup() { fusermount -u "$MNT/$MONTH" 2>/dev/null || fusermount3 -u "$MNT/$MONTH" 2>/dev/null || true; rm -rf "$MNT"; }
trap cleanup EXIT
mkdir -p "$MNT/$MONTH" "$OUT_DIR"
# Some nodes lack the apptainer squashfuse and some system builds lack lz4: try both.
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

"$PYTHON" run/post_training/policy_bench/check_book_codec.py \
  --shard_root "$MNT" --manifest "$MANIFEST" --n_days "$N_DAYS" --seed "$SEED" \
  --workers "${SLURM_CPUS_PER_TASK:-72}" --out "$OUT_DIR/g1g2_${SLURM_JOB_ID}.json"
chmod g+rw "$OUT_DIR/g1g2_${SLURM_JOB_ID}.json"
