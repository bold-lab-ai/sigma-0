#!/usr/bin/env bash
# Build envs/lobbench_venv — the python used to RUN LOB-Bench scoring, i.e. the
# interpreter handed to `run/benchmarking/run_lobbench.py --python`, which
# `lobpipeline.lob_bench.scoring.build_lobbench_plan` then invokes against
# DEFAULT_BENCHMARK_ROOT. NOT the training env: the scoring dependency set
# (pandas 3, plotly, statsmodels, sklearn) conflicts with the training stack,
# and LOB-Bench must never be imported into the training process.
#
# Usage:
#     envs/lobbench_venv_setup.sh [BASE_PY]
#     BASE_PY=/path/to/python envs/lobbench_venv_setup.sh
#
# BASE_PY must be a python >= 3.10 whose site-packages already provide numpy,
# matplotlib and tqdm (the venv is created with --system-site-packages and only
# the packages below are added locally). Reference venv was built from
# miniforge3 python 3.13.13 (see docs/benchmarking.md).
#
# Pinned versions below were read from the working EGGROLL-GAN venv
# (AlphaTrade/experiments/exp_0626_EGGROLL_GAN/third_party/lobbench_venv,
# `pip list --format=freeze`, 2026-07-15).

set -euo pipefail
umask 002   # group (brics.*) inherits rw on files / rwx on dirs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PY="${1:-${BASE_PY:-}}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/lobbench_venv}"

if [[ -z "${BASE_PY}" ]]; then
    echo "usage: $0 <BASE_PY>   (or export BASE_PY=/path/to/python)" >&2
    echo "BASE_PY must provide numpy, matplotlib, tqdm via its site-packages." >&2
    exit 2
fi
if [[ -e "${VENV_DIR}" ]]; then
    echo "[lobbench-venv] ${VENV_DIR} already exists — remove it first to rebuild." >&2
    exit 1
fi

echo "[lobbench-venv] base python: ${BASE_PY}"
"${BASE_PY}" -m venv --system-site-packages "${VENV_DIR}"

"${VENV_DIR}/bin/pip" install --no-cache-dir \
    joblib==1.5.3 \
    narwhals==2.22.1 \
    pandas==3.0.3 \
    patsy==1.0.2 \
    plotly==6.8.0 \
    scikit-learn==1.9.0 \
    scipy==1.17.1 \
    seaborn==0.13.2 \
    statsmodels==0.14.6 \
    threadpoolctl==3.6.0

# Shared-filesystem group access (pip creates files with the caller's umask, but
# be explicit — see CLAUDE.md group-permissions rule).
chmod -R g+rwX "${VENV_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
import matplotlib, numpy, pandas, plotly, scipy, seaborn, sklearn, statsmodels, tqdm
print("[lobbench-venv] import check OK:",
      f"pandas {pandas.__version__}, scipy {scipy.__version__},",
      f"statsmodels {statsmodels.__version__}")
PY
echo "[lobbench-venv] done -> ${VENV_DIR}"
