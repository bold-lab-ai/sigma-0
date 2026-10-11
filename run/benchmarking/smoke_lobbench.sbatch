#!/bin/bash
#SBATCH --job-name=sigma0-lobbench-smoke
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python run/benchmarking/run_lobbench.py --run-name smoke

