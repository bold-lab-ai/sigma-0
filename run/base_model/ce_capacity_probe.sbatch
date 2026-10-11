#!/bin/bash
#SBATCH --job-name=ce-capacity-probe
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/ce_capacity_%j.out

# Capacity probe for the profiling PR: the committed per-position decomposition
# (evaluate_ce_by_position.py), run on the 8M and 78M model-zoo lineages at a
# nearest-common checkpoint pair (steps 27360 / 26670) and at each lineage's
# endpoint (29790 / 46050), on the same two GOOG slices as the j5749206
# artifacts (2025-12 training-period, 2026-01 held-out).
#
# Idempotent: a combo whose output JSON already exists and is non-empty is
# skipped, so a re-run (or an attached copy racing a queued copy) cannot
# duplicate work or clobber finished results.

set -euo pipefail

QUANT_ROOT=${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}
export CONDA_PREFIX=$QUANT_ROOT/miniforge3
export PATH=$CONDA_PREFIX/bin:$PATH

REPO=${REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"

ZOO=${ZOO:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/checkpoints}
SQUASHFS_DIR=${SQUASHFS_DIR:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}
TICKERS=${TICKERS:-GOOG}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_BATCHES=${MAX_BATCHES:-8}
TRAIN_MONTH=2025-12
HELDOUT_MONTH=2026-01
TRAIN_RANGE=2025-12-01,2025-12-31
HELDOUT_RANGE=2026-01-01,2026-01-31

MROOT=${TMPDIR:-/tmp}/ce_capacity_squashfs_${SLURM_JOB_ID:-manual}
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

# lineage-dir  step  short-name
COMBOS=(
    "j3443014_zkrtl2ef_3443014 27360 zoo8M"
    "j3443014_zkrtl2ef_3443014 29790 zoo8M"
    "j3417629_pw8u0edj_3417629 26670 zoo78M"
    "j3417629_pw8u0edj_3417629 46050 zoo78M"
)

run_slice() {
    local root=$1 step=$2 short=$3 label=$4 month=$5 range=$6
    local out="$OUT_DIR/${short}_s${step}_${label}.json"
    if [ -s "$out" ]; then
        echo "[skip] $out already exists"
        return 0
    fi
    echo "=== $short step $step slice $label ($month, $range) ==="
    PYTHONPATH=src python run/base_model/evaluate_ce_by_position.py \
        --checkpoint-root "$ZOO/$root" \
        --step "$step" \
        --data-root "$MROOT/$month" \
        --tickers "$TICKERS" \
        --date-range "$range" \
        --slice-label "$label" \
        --batch-size "$BATCH_SIZE" \
        --max-batches "$MAX_BATCHES" \
        --output "$out"
}

for combo in "${COMBOS[@]}"; do
    read -r root step short <<< "$combo"
    run_slice "$root" "$step" "$short" train_period "$TRAIN_MONTH" "$TRAIN_RANGE"
    run_slice "$root" "$step" "$short" heldout "$HELDOUT_MONTH" "$HELDOUT_RANGE"
done

echo "[done] capacity probe JSONs are in $OUT_DIR"
