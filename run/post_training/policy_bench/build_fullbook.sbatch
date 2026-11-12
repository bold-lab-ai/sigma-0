#!/bin/bash
# Full-book files from the raw L0 messages (CPU only). See build_fullbook.py.
# Pass the manifest, month and outputs with the arguments of build_fullbook.py.
#
#   PYTHON=<env with numpy and pandas>/bin/python sbatch --nodes=1 --output=<dir>/%x_%j.out \
#       run/post_training/policy_bench/build_fullbook.sbatch --manifest <file.json> --month 2026-01 \
#       --compare_dir <quant>/lob_fullbook_q4_2025_jan2026 --report <file.jsonl>
#
#SBATCH --job-name=policy-bench-fullbook
#SBATCH --account=brics.u6oz
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00

set -euo pipefail
echo "start host=$(hostname) job=${SLURM_JOB_ID:-} $(date -Is)"

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${PYTHON:?PYTHON (an env with numpy and pandas) is required}"
Q=/projects/public/u6oz/quant
SHARD_DIR="${SHARD_DIR:-$Q/lob_preproc_sp500_squashfs_v2_20261013}"
LOBSTER="${LOBSTER:-$Q/lobster}"
MONTH="${MONTH:-2026-01}"

MNT="/tmp/policy_bench_fullbook_${SLURM_JOB_ID}"
export TMPDIR="/tmp/policy_bench_fullbook_tmp_${SLURM_JOB_ID}"
cleanup() { fusermount -u "$MNT/$MONTH" 2>/dev/null || fusermount3 -u "$MNT/$MONTH" 2>/dev/null || true; rm -rf "$TMPDIR"; }
trap cleanup EXIT
mkdir -p "$MNT/$MONTH" "$TMPDIR"
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

"$PYTHON" run/post_training/policy_bench/build_fullbook.py --shard_root "$MNT" --lobster "$LOBSTER" \
  --month "$MONTH" --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-32}}" "$@"
