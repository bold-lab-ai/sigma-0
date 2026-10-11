#!/bin/bash
# Build K3 State Scan CUDA FFI kernel for GH200 (sm_90a, aarch64)
#
# Usage: bash src/base_model/kernels/mamba3_cuda/build_state_scan.sh
# Output: src/base_model/kernels/mamba3_cuda/build/state_scan_ffi.cpython-312-aarch64-linux-gnu.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
FFI_DIR="${SCRIPT_DIR}/ffi"
BUILD_DIR="${SCRIPT_DIR}/build"

mkdir -p "${BUILD_DIR}"

# NVCC (GH200 HPC SDK)
NVCC=${NVCC:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/bin/nvcc}
if [ ! -f "${NVCC}" ]; then
    echo "NVCC not found at ${NVCC}, trying module cuda/12.6..."
    module load cuda/12.6 2>/dev/null || true
    NVCC=$(which nvcc 2>/dev/null || echo "")
    if [ -z "${NVCC}" ]; then
        echo "ERROR: nvcc not found. Load cuda module or set NVCC env var."
        exit 1
    fi
fi

# Python/JAX/nanobind include paths
JAX_INCLUDE=$(python3 -c "import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")
NB_INCLUDE=$(python3 -c "import nanobind; print(nanobind.include_dir())")
NB_SRC=$(python3 -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'src'))")
NB_EXT=$(python3 -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'ext', 'robin_map', 'include'))")
PY_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

OUTPUT="${BUILD_DIR}/state_scan_ffi${EXT_SUFFIX}"

echo "Building K3 State Scan kernel..."
echo "  NVCC: ${NVCC}"
echo "  JAX include: ${JAX_INCLUDE}"
echo "  Output: ${OUTPUT}"
echo ""

"${NVCC}" \
    -arch=sm_90a \
    -std=c++17 \
    --expt-relaxed-constexpr \
    -Xcompiler -fPIC \
    -shared \
    -O3 \
    --use_fast_math \
    -I"${SRC_DIR}" \
    -I"${JAX_INCLUDE}" \
    -I"${NB_INCLUDE}" \
    -I"${NB_EXT}" \
    -I"${PY_INCLUDE}" \
    "${SRC_DIR}/siso_state_scan.cu" \
    "${SRC_DIR}/reduce_y_off.cu" \
    "${FFI_DIR}/state_scan_ffi.cu" \
    "${FFI_DIR}/state_scan_ffi_nb.cpp" \
    "${NB_SRC}/nb_combined.cpp" \
    -o "${OUTPUT}" \
    -lcudart

echo ""
echo "Build successful: ${OUTPUT}"
ls -lh "${OUTPUT}"
