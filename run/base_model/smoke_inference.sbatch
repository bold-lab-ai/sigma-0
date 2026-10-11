#!/bin/bash
#SBATCH --job-name=sigma0-infer-smoke
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=115000M

set -euo pipefail

export WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export QUANT_ROOT="${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}"
export PATH="$QUANT_ROOT/miniforge3/bin:$PATH"
export PYTHONPATH="$WORKDIR/src:$WORKDIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to a completed model-zoo checkpoint}"
: "${DATA_DIR:?set DATA_DIR to the inference-day preprocessed data directory}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a writable result directory}"

cd "$WORKDIR"
module load cuda/12.6

ARCHITECTURE_ARG=()
if [ -n "${ARCHITECTURE:-}" ]; then
    ARCHITECTURE_ARG=("--architecture=$ARCHITECTURE")
fi

python -u -B run/base_model/runtime/inference.py \
    "${ARCHITECTURE_ARG[@]}" \
    --token_mode=26tok \
    --ckpt_path="$CHECKPOINT_PATH" \
    --data_dir="$DATA_DIR" \
    --save_dir="$OUTPUT_ROOT" \
    --stock="${STOCK:-GOOG}" \
    --n_sequences="${N_SEQUENCES:-8}" \
    --batch_size="${INFERENCE_BATCH_SIZE:-1}" \
    --n_cond_msgs="${N_COND_MSGS:-16}" \
    --n_gen_msgs="${N_GEN_MSGS:-4}" \
    --seed="${SEED:-42}"
