#!/bin/bash
# CPU-only validation of all 47 training shards plus one real 26-token batch.
#SBATCH --job-name=sigma0-zoo-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00:30:00
#SBATCH --output=model_zoo_data_%j.out
#SBATCH --error=model_zoo_data_%j.err

set -euo pipefail

export WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CURRENT_COMMIT=$(git -C "$WORKDIR" rev-parse HEAD)
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]; then
    echo "Refusing code drift: expected $EXPECTED_COMMIT, found $CURRENT_COMMIT" >&2
    exit 2
fi

SIGMA0_JOB_TMPDIR="${SIGMA0_JOB_TMPDIR:-/tmp/${USER:-${LOGNAME:-unknown}}/sigma0/${SLURM_JOB_ID:-manual}_${SLURM_PROCID:-0}}"
mkdir -p "$SIGMA0_JOB_TMPDIR"
[ -d "$SIGMA0_JOB_TMPDIR" ] && [ -w "$SIGMA0_JOB_TMPDIR" ] || {
    echo "Job-local temporary directory is unavailable: $SIGMA0_JOB_TMPDIR" >&2
    exit 2
}
export TMPDIR="$SIGMA0_JOB_TMPDIR"
MOUNT_ROOT="$TMPDIR/data_mounts"
SQUASHFS_DIR="$WORKDIR/data/SP500-2022-2025"
mkdir -p "$MOUNT_ROOT"

cleanup() {
    for mount_dir in "$MOUNT_ROOT"/*/; do
        [ -d "$mount_dir" ] || continue
        if mountpoint -q "$mount_dir"; then
            fusermount -u "$mount_dir" || true
        fi
    done
}
trap cleanup EXIT

months=()
for year in 2022 2023 2024; do
    for month in {01..12}; do
        months+=("$year-$month")
    done
done
for month in {01..11}; do
    months+=("2025-$month")
done

DATA_ROOT=""
for year_month in "${months[@]}"; do
    shard="$SQUASHFS_DIR/shard_${year_month}.squashfs"
    mount_dir="$MOUNT_ROOT/$year_month"
    [ -f "$shard" ] || { echo "Missing shard: $shard" >&2; exit 2; }
    mkdir -p "$mount_dir"
    squashfuse "$shard" "$mount_dir"
    [ -f "$mount_dir/index.json" ] || { echo "Missing mounted index: $year_month" >&2; exit 2; }
    DATA_ROOT="${DATA_ROOT:+$DATA_ROOT,}$mount_dir"
done

export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$WORKDIR/src:$WORKDIR:${PYTHONPATH:-}"
PYTHON=/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python

echo "[data-smoke] commit=$CURRENT_COMMIT shards=${#months[@]}"
"$PYTHON" -u "$WORKDIR/run/base_model/check_model_zoo_data.py" --data-root "$DATA_ROOT"
