#!/bin/bash
#SBATCH --job-name=ce-decomp
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ce_decomp_%j.out

# Per-position CE decomposition of one checkpoint on two slices: an
# in-training-range month and a held-out month. Single GPU, two squashfuse
# mounts, two JSONs into artifacts/ce_by_position/. The mount block mirrors
# node_wrapper.sh's multi-shard mode (per-node mounts under $TMPDIR; block
# reads of one pre-striped file instead of 488 ticker-dir stats).

set -euo pipefail

QUANT_ROOT=${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}
export CONDA_PREFIX=$QUANT_ROOT/miniforge3
export PATH=$CONDA_PREFIX/bin:$PATH

REPO=${REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"

CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the checkpoint run directory}
CKPT_STEP=${CKPT_STEP:-latest}
TICKERS=${TICKERS:-GOOG}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_BATCHES=${MAX_BATCHES:-8}

SQUASHFS_DIR=${SQUASHFS_DIR:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}
TRAIN_MONTH=${TRAIN_MONTH:-2025-12}
HELDOUT_MONTH=${HELDOUT_MONTH:-2026-01}
TRAIN_RANGE=${TRAIN_RANGE:-2025-12-01,2025-12-31}
HELDOUT_RANGE=${HELDOUT_RANGE:-2026-01-01,2026-01-31}

MROOT=${TMPDIR:-/tmp}/ce_decomp_squashfs_${SLURM_JOB_ID}
cleanup() {
    for d in "$MROOT"/*/; do
        [ -d "$d" ] || continue
        mountpoint -q "$d" 2>/dev/null && fusermount -u "$d" 2>/dev/null || true
    done
}
trap cleanup EXIT

for ym in "$TRAIN_MONTH" "$HELDOUT_MONTH"; do
    SHARD="$SQUASHFS_DIR/shard_${ym}.squashfs"
    [ -f "$SHARD" ] || { echo "FATAL: missing shard $SHARD" >&2; exit 2; }
    mkdir -p "$MROOT/$ym"
    squashfuse "$SHARD" "$MROOT/$ym"
    echo "[squashfs] mounted $ym"
done

OUT_DIR=artifacts/ce_by_position
mkdir -p "$OUT_DIR"
TAG=$(basename "$CKPT_ROOT")

run_slice() {
    local label=$1 month=$2 range=$3
    echo "=== slice $label ($month, $range) ==="
    PYTHONPATH=src python run/base_model/evaluate_ce_by_position.py \
        --checkpoint-root "$CKPT_ROOT" \
        --step "$CKPT_STEP" \
        --data-root "$MROOT/$month" \
        --tickers "$TICKERS" \
        --date-range "$range" \
        --slice-label "$label" \
        --batch-size "$BATCH_SIZE" \
        --max-batches "$MAX_BATCHES" \
        --output "$OUT_DIR/${TAG}_${label}.json"
}

run_slice train_period "$TRAIN_MONTH" "$TRAIN_RANGE"
run_slice heldout "$HELDOUT_MONTH" "$HELDOUT_RANGE"

echo "[done] wrote $OUT_DIR/${TAG}_train_period.json and ${TAG}_heldout.json"
