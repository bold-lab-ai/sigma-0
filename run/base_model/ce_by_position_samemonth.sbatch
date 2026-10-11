#!/bin/bash
#SBATCH --job-name=ce-samemonth
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/ce_decomp_samemonth_%j.out

# The memorisation term without any period correction at all.
#
# The measurement so far differenced GOOG 2025-12 (trained on) against a leak-free
# draw over 2025-01..11, and CE is not flat across that span: it falls monotonically
# year over year and keeps falling inside 2025. Correcting for that took the term from
# +0.058 to +0.024, but the correction is a two-point slope extrapolated 3.5 months, so
# it cannot carry the conclusion by itself.
#
# This removes the need for it. valset_v1 excludes only GOOG's 2025-12 (that month was
# fully consumed by a fine-tune); for every other ticker the frozen set does contain
# 2025-12 windows. So the same tickers can be scored twice in the same month, once on
# data the model trained on and once on data it did not, with the period held exactly
# fixed rather than modelled.
#
# Seven tickers with enough 2025-12 samples in the shard (NVDA 29, GOOGL 21, PLTR 13,
# TSLA 11, AMD 11, AMZN 10, AAPL 9) give 8 sequences each under uniform ticker batching.

set -euo pipefail

QUANT_ROOT=${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}
export CONDA_PREFIX=$QUANT_ROOT/miniforge3
export PATH=$CONDA_PREFIX/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"

CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT}
CKPT_STEP=${CKPT_STEP:-latest}
TICKERS=${TICKERS:-NVDA,GOOGL,PLTR,TSLA,AMD,AMZN,AAPL}
BATCH_SIZE=${BATCH_SIZE:-8}
NBATCH=${NBATCH:-7}
MONTH=${MONTH:-2025-12}
RANGE=${RANGE:-2025-12-01,2025-12-31}

SQUASHFS_DIR=${SQUASHFS_DIR:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}
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
MROOT=$TMPDIR/ce_samemonth_${SLURM_JOB_ID:-$$}
cleanup() {
    for d in "$MROOT"/*/; do
        [ -d "$d" ] || continue
        mountpoint -q "$d" 2>/dev/null && fusermount -u "$d" 2>/dev/null || true
    done
}
trap cleanup EXIT

mkdir -p "$MROOT/valset"; squashfuse "$VALSET_SHARD" "$MROOT/valset"
SHARD="$SQUASHFS_DIR/shard_${MONTH}.squashfs"
[ -f "$SHARD" ] || { echo "FATAL: missing $SHARD" >&2; exit 2; }
mkdir -p "$MROOT/$MONTH"; squashfuse "$SHARD" "$MROOT/$MONTH"
echo "[squashfs] mounted valset_v1 and $MONTH"

OUT_DIR=artifacts/ce_by_position; mkdir -p "$OUT_DIR"
TAG=$(basename "$CKPT_ROOT")

run_slice() {
    local label=$1 root=$2
    if [ -s "$OUT_DIR/${TAG}_${label}.json" ]; then
        echo "=== slice $label: already on disk, skipping ==="; return 0
    fi
    echo "=== slice $label ($root, $RANGE, $((BATCH_SIZE * NBATCH)) sequences) ==="
    PYTHONPATH=src python run/base_model/evaluate_ce_by_position.py \
        --checkpoint-root "$CKPT_ROOT" --step "$CKPT_STEP" \
        --data-root "$root" --tickers "$TICKERS" \
        --date-range "$RANGE" --slice-label "$label" \
        --batch-size "$BATCH_SIZE" --max-batches "$NBATCH" \
        --output "$OUT_DIR/${TAG}_${label}.json"
}

run_slice samemonth_trained "$MROOT/$MONTH"
run_slice samemonth_leakfree "$MROOT/valset"

echo "[done] the pair differs only in leakage; the month is identical"
