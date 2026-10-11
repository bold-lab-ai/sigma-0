#!/bin/bash
#SBATCH --job-name=ce-decomp-valset
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/ce_decomp_valset_%j.out

# Separates memorisation from temporal shift in the per-position CE decomposition.
#
# The first run of this decomposition compared GOOG 2025-12 (inside the training
# window, and trained on) against GOOG 2026-01 (after the window, and never trained
# on). Those two slices differ in BOTH leakage and time period, so the +0.087 nats
# between them is the two effects added together and neither is measured. Whichever
# dominates changes where the entropy floor sits, and in opposite directions.
#
# valset_v1 fills the missing cell: a frozen, leak-free draw over the same 488
# tickers and the same 2022-2025 window (VALSET_V1_REPORT.md section 4 audits every
# training run's consumption depth; the deepest reaches 16.63% of its permutation
# and the set is drawn beyond 20%). Its materialised shard is layout-identical to a
# training month, so nothing here needs a code change -- section 6.2, whose one
# documented caveat is that random start offsets must be off, which
# evaluate_ce_by_position.py already does (rand_offset=False).
#
# Four slices, one ticker, one checkpoint:
#
#   train_period_v2  GOOG 2025-12         trained on        in window
#   valset_2025      GOOG 2025-01..11     never trained on  in window   <- the new cell
#   valset_all       GOOG 2022..2025-11   never trained on  in window   <- precision + a
#                                                                          within-window
#                                                                          period contrast
#   heldout_v2       GOOG 2026-01         never trained on  after it
#
# train_period_v2 -> valset_2025 is memorisation with the year held fixed.
# valset_2025 -> heldout_v2 is temporal shift with leakage held fixed.
# valset_2025 -> valset_all says how much period alone moves CE inside the window,
# which is the free control on that second reading.
#
# Sample count: the first run used 64 sequences per slice, which test_ce_eval_spec.md
# line 42 puts sixty times under the budget for sigma <= 0.002 nats. GOOG has 588
# samples in the valset shard, so 128 sequences is affordable for the three matched
# slices and 512 for the precision one.

set -euo pipefail

QUANT_ROOT=${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}
export CONDA_PREFIX=$QUANT_ROOT/miniforge3
export PATH=$CONDA_PREFIX/bin:$PATH
# Coexist with whatever else holds this node's other GPUs: take what we need, not 90%.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"

CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the checkpoint run directory}
CKPT_STEP=${CKPT_STEP:-latest}
TICKERS=${TICKERS:-GOOG}
BATCH_SIZE=${BATCH_SIZE:-8}
MATCHED_BATCHES=${MATCHED_BATCHES:-16}     # 128 sequences = 64,000 messages
PRECISION_BATCHES=${PRECISION_BATCHES:-64} # 512 sequences = 256,000 messages

SQUASHFS_DIR=${SQUASHFS_DIR:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}
VALSET_SHARD=${VALSET_SHARD:-/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/scaling_law_plots/valset_ce_eval_20260730/valset/shard_valset_v1_30720.squashfs}
VALSET_SHA256_PREFIX=${VALSET_SHA256_PREFIX:-ffcb71d90d96}

[ -f "$VALSET_SHARD" ] || { echo "FATAL: missing $VALSET_SHARD" >&2; exit 2; }
echo "[valset] hashing the frozen shard"
GOT=$(sha256sum "$VALSET_SHARD" | cut -c1-12)
if [ "$GOT" != "$VALSET_SHA256_PREFIX" ]; then
    # The whole value of a frozen set is that it is the same bytes every time; a
    # silent swap would move every number here without moving anything visible.
    echo "FATAL: valset shard sha256 starts $GOT, expected $VALSET_SHA256_PREFIX" >&2
    exit 3
fi
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
MROOT=$TMPDIR/ce_valset_${SLURM_JOB_ID:-$$}
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
for ym in 2025-12 2026-01; do
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
    local label=$1 root=$2 range=$3 nbatch=$4
    echo "=== slice $label ($root, $range, $((BATCH_SIZE * nbatch)) sequences) ==="
    PYTHONPATH=src python run/base_model/evaluate_ce_by_position.py \
        --checkpoint-root "$CKPT_ROOT" \
        --step "$CKPT_STEP" \
        --data-root "$root" \
        --tickers "$TICKERS" \
        --date-range "$range" \
        --slice-label "$label" \
        --batch-size "$BATCH_SIZE" \
        --max-batches "$nbatch" \
        --output "$OUT_DIR/${TAG}_${label}.json"
}

run_slice train_period_v2 "$MROOT/2025-12" 2025-12-01,2025-12-31 "$MATCHED_BATCHES"
run_slice valset_2025     "$MROOT/valset"  2025-01-01,2025-11-30 "$MATCHED_BATCHES"
run_slice heldout_v2      "$MROOT/2026-01" 2026-01-01,2026-01-31 "$MATCHED_BATCHES"
run_slice valset_all      "$MROOT/valset"  2022-01-01,2025-11-30 "$PRECISION_BATCHES"

echo "[done] four slices in $OUT_DIR/${TAG}_{train_period_v2,valset_2025,heldout_v2,valset_all}.json"
