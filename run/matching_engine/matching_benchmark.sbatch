#!/bin/bash
#SBATCH --job-name=alpha-match-bench
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=agent_outputs/%x_%j.out
#SBATCH --error=agent_outputs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/lus/lfs1aip2/projects/public/u6gb/sigma-0}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

OUT_DIR="agent_outputs/alphatrade_matching_bench_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

python run/matching_engine/alphatrade_matching_benchmark.py \
  --orders "${ALPHATRADE_BENCH_ORDERS:-1000000}" \
  --warmup-orders "${ALPHATRADE_BENCH_WARMUP_ORDERS:-100000}" \
  --repeats "${ALPHATRADE_BENCH_REPEATS:-7}" \
  --output "${OUT_DIR}/result.json"
