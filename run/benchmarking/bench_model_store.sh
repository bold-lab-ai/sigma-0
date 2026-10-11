#!/bin/bash
# Benchmark one pinned model-store checkpoint on one node.
#
# Four ranks of autoregressive inference over a frozen window set, then
# LOB-Bench scoring with the divergence-over-time curve. The reason this lives
# in the repository rather than in a scratch directory: the environment a
# checkpoint needs is a property of the checkpoint, recorded in
# src/lobpipeline/model_store_pins.json, and an ad-hoc launcher that does not
# read it scores pre-2026-04-22 weights under a normalizer they never saw. That
# happened; see docs/experiments/model_zoo/rmsnorm_epsilon.ipynb.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required; see python -m lobpipeline.model_store list}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
QUANT_ROOT="${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}"
PYTHON="${PYTHON:-$QUANT_ROOT/miniforge3/bin/python}"
BENCHMARK_ROOT="${LOBBENCH_ROOT:-$WORKDIR/src/lobpipeline/scorer}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
N_SEQ="${N_SEQ:-3136}"
BSZ="${BSZ:-8}"
N_COND="${N_COND:-250}"
N_GEN="${N_GEN:-250}"
GEN_SEED="${GEN_SEED:-2026}"
HORIZON="${HORIZON:-50}"
STOCK="${STOCK:-GOOG}"
PERIOD="${PERIOD:-2026-01}"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

# The pin decides the normalizer, not the caller. An unknown name fails here,
# before anything is staged, rather than defaulting to the modern epsilon.
eval "$("$PYTHON" -m lobpipeline.model_store env "$MODEL_NAME")"
export MAMBA3_LEGACY_NORM
[ -n "${MAMBA3_LEGACY_NORM:-}" ] || unset MAMBA3_LEGACY_NORM

read -r CKPT STEP < <("$PYTHON" - "$MODEL_NAME" <<'PY'
import sys
from lobpipeline.model_store import resolve
entry = resolve(sys.argv[1])
print(entry["path"], entry["step"])
PY
)

OUT="$OUT_ROOT/$MODEL_NAME"
INF="$OUT/inference"
mkdir -p "$INF"
echo "[bench] $MODEL_NAME step=$STEP legacy_norm=${MAMBA3_LEGACY_NORM:-<unset>}"
echo "[bench] $N_SEQ sequences, ${N_COND} prompt + ${N_GEN} generated, seed $GEN_SEED"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TMPDIR="${TMPDIR:-/tmp/$USER/bench_$MODEL_NAME}"
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/matplotlib"; mkdir -p "$MPLCONFIGDIR"

source "$QUANT_ROOT/AlphaTrade/lob_pipeline/pipeline/_squashfs_helpers.sh"
infer_squashfs_setup "$STOCK" "${MONTHS:-$PERIOD}" \
    "${SQUASHFS_DIR:-/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs}"
infer_squashfs_setup_wide_book "$STOCK" "${MONTHS:-$PERIOD}" \
    "${WIDE_SQUASHFS_DIR:-/lus/lfs1aip2/projects/public/s5e/quant_team/recon_2026-05/output/squashfs}" \
    "${WIDE_LEVELS:-500}"
trap infer_squashfs_cleanup EXIT

PIDS=()
for RANK in 0 1 2 3; do
    # --architecture is deliberately not passed: the checkpoint declares it, and
    # the registry's ssm_type names the SSM (s5) rather than the architecture
    # (nsa), so asserting it here fails runs that would otherwise work.
    CUDA_VISIBLE_DEVICES=$RANK "$PYTHON" -u -B "$WORKDIR/run/base_model/runtime/inference.py" \
        --token_mode=26tok \
        --ckpt_path="$CKPT" --checkpoint_step="$STEP" \
        --data_dir="$INFER_DATA_DIR_NODE" \
        --wide_book_dir="$INFER_WIDE_BOOK_DIR_NODE" --wide_levels="${WIDE_LEVELS:-500}" \
        --save_dir="$INF" --stock="$STOCK" \
        --n_sequences="$N_SEQ" --batch_size="$BSZ" \
        --n_cond_msgs="$N_COND" --n_gen_msgs="$N_GEN" \
        --seed="$GEN_SEED" --rank="$RANK" --world_size=4 \
        > "$OUT/inference_rank${RANK}.log" 2>&1 &
    PIDS+=("$!")
done
RC=0
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || { echo "[bench] rank $i failed" >&2; RC=1; }; done
[ "$RC" -eq 0 ] || { echo "[bench] $MODEL_NAME inference failed"; exit 1; }

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "$PYTHON" -u -B \
    "$WORKDIR/run/benchmarking/run_lobbench.py" \
    --run-name="${MODEL_NAME//_/-}" --inference-dir="$INF" \
    --output-dir="$OUT/lobbench" --stock="$STOCK" --period="$PERIOD" \
    --benchmark-root="$BENCHMARK_ROOT" --python="$PYTHON" \
    --n-workers="${LOBBENCH_WORKERS:-48}" --run-id=store \
    --divergence-horizon="$HORIZON" \
    --summary="$OUT/lobbench_summary.json" > "$OUT/scoring.log" 2>&1
echo "[bench] $MODEL_NAME complete: $OUT/lobbench_summary.json"
