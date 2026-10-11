#!/bin/bash
#SBATCH --job-name=ce-decomp-period
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/ce_decomp_period_%j.out

# How much does CE move with the period alone, on data the model never trained on?
#
# The first wave measured a memorisation term by differencing GOOG 2025-12 (trained
# on) against a leak-free valset draw over 2025-01..11. That difference is only
# memorisation if CE is stable across periods inside the training window. The wave's
# own precision slice said otherwise: drawn over 2022-2025 it read 0.949 against
# 0.742 for 2025 alone -- a period effect three times the size of the Dec-to-Jan
# shift it was meant to control for.
#
# So this measures the period curve directly. Every slice below is leak-free, inside
# the training window, same ticker and same checkpoint; only the period differs. The
# year slices say how much CE moves between years; the two 2025 halves say how much
# it moves inside the year the memorisation term was differenced across, which is the
# quantity that bounds that term's contamination.
#
# Sample budget per slice is set by what GOOG has in the valset shard: 198 in 2022,
# 101 in 2023, 134 in 2024, and 71 / 84 in the two halves of 2025.

set -euo pipefail

QUANT_ROOT=${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}
export CONDA_PREFIX=$QUANT_ROOT/miniforge3
export PATH=$CONDA_PREFIX/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"

CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the checkpoint run directory}
CKPT_STEP=${CKPT_STEP:-latest}
TICKERS=${TICKERS:-GOOG}
BATCH_SIZE=${BATCH_SIZE:-8}

VALSET_SHARD=${VALSET_SHARD:-/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/valset_ce_eval_20260730/valset/shard_valset_v1_30720.squashfs}
VALSET_SHA256_PREFIX=${VALSET_SHA256_PREFIX:-ffcb71d90d96}
[ -f "$VALSET_SHARD" ] || { echo "FATAL: missing $VALSET_SHARD" >&2; exit 2; }
GOT=$(sha256sum "$VALSET_SHARD" | cut -c1-12)
[ "$GOT" = "$VALSET_SHA256_PREFIX" ] || { echo "FATAL: shard sha256 $GOT != $VALSET_SHA256_PREFIX" >&2; exit 3; }
echo "[valset] sha256 prefix $GOT matches VALSET_V1_REPORT.md"

# TMPDIR unconditionally, not "only if the inherited one looks broken". --export=ALL
# carries a login node's /local/user/<uid> onto the compute node; the first guard tested
# whether that path existed and it did at that moment -- squashfuse mounted under it --
# but torch's tempfile.TemporaryDirectory() still hit ENOENT on the same parent a few
# seconds later (jobs 6249025 and 6267005, both dead inside 40s). A path this session
# creates itself under /tmp has no such window.
TMPDIR=/tmp/u6gb_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
export TMPDIR
MROOT=$TMPDIR/ce_period_${SLURM_JOB_ID:-$$}
cleanup() {
    for d in "$MROOT"/*/; do
        [ -d "$d" ] || continue
        mountpoint -q "$d" 2>/dev/null && fusermount -u "$d" 2>/dev/null || true
    done
}
trap cleanup EXIT
mkdir -p "$MROOT/valset"
squashfuse "$VALSET_SHARD" "$MROOT/valset"
echo "[squashfs] mounted valset_v1"

OUT_DIR=artifacts/ce_by_position
mkdir -p "$OUT_DIR"
TAG=$(basename "$CKPT_ROOT")

run_slice() {
    local label=$1 range=$2 nbatch=$3
    # Resume-safe: an attached step dies with its host allocation, and the first
    # attempt lost four of five slices that way after the parent job's own driver
    # finished eight hours inside its time limit. Skipping finished slices makes a
    # re-run cost only what is missing.
    if [ -s "$OUT_DIR/${TAG}_${label}.json" ]; then
        echo "=== slice $label: already on disk, skipping ==="
        return 0
    fi
    echo "=== slice $label ($range, $((BATCH_SIZE * nbatch)) sequences) ==="
    PYTHONPATH=src python run/base_model/evaluate_ce_by_position.py \
        --checkpoint-root "$CKPT_ROOT" --step "$CKPT_STEP" \
        --data-root "$MROOT/valset" --tickers "$TICKERS" \
        --date-range "$range" --slice-label "$label" \
        --batch-size "$BATCH_SIZE" --max-batches "$nbatch" \
        --output "$OUT_DIR/${TAG}_${label}.json"
}

run_slice valset_2022    2022-01-01,2022-12-31 16
run_slice valset_2023    2023-01-01,2023-12-31 12
run_slice valset_2024    2024-01-01,2024-12-31 16
run_slice valset_2025h1  2025-01-01,2025-05-31 8
run_slice valset_2025h2  2025-06-01,2025-11-30 8

echo "[done] five period slices in $OUT_DIR"
