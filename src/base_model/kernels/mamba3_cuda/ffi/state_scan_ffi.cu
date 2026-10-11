/******************************************************************************
 * FFI Handlers for K3 v3 (N-split)
 *
 * Two FFI targets:
 *   1. siso_state_scan: writes partial output (N_SPLITS, B, L, H, P)
 *   2. reduce_y_off:    sums partials -> (B, L, H, P)
 *****************************************************************************/

#include "xla/ffi/api/ffi.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ffi = xla::ffi;
using BF16 = ffi::Buffer<ffi::DataType::BF16>;
using F32  = ffi::Buffer<ffi::DataType::F32>;

// No split count is defined here on purpose. SS_N_SPLITS lives in
// src/siso_state_scan.cu and the Python wrapper mirrors it as _N_SPLITS; a
// third copy on the host side is a constant that can disagree with the kernel
// it describes. The partial-output leading dimension comes from the caller.

// Launchers
extern "C" void siso_state_scan_launch(
    const __nv_bfloat16* K_in,
    const __nv_bfloat16* V_in,
    const __nv_bfloat16* Q_in,
    const float* ADT_in,
    __nv_bfloat16* Y_off_partial,
    int B, int L, int H,
    cudaStream_t stream);

extern "C" void reduce_y_off_launch(
    const __nv_bfloat16* Y_off_partial,
    __nv_bfloat16* Y_off_out,
    int64_t total_elems,
    int n_splits,
    cudaStream_t stream);

// ============================================================================
// Handler 1: State scan (outputs partial)
// ============================================================================
ffi::Error StateScanHost(
    cudaStream_t stream,
    BF16 K_in,      // (B, L, H, N)
    BF16 V_in,      // (B, L, H, P)
    BF16 Q_in,      // (B, L, H, N)
    F32  ADT_in,    // (B, H, L)
    ffi::Result<BF16> Y_partial)  // (N_SPLITS, B, L, H, P)
{
    auto K_dims = K_in.dimensions();
    const int B = static_cast<int>(K_dims[0]);
    const int L = static_cast<int>(K_dims[1]);
    const int H = static_cast<int>(K_dims[2]);

    siso_state_scan_launch(
        reinterpret_cast<const __nv_bfloat16*>(K_in.typed_data()),
        reinterpret_cast<const __nv_bfloat16*>(V_in.typed_data()),
        reinterpret_cast<const __nv_bfloat16*>(Q_in.typed_data()),
        ADT_in.typed_data(),
        reinterpret_cast<__nv_bfloat16*>(Y_partial->typed_data()),
        B, L, H, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StateScan, StateScanHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<BF16>()   // K_in
        .Arg<BF16>()   // V_in
        .Arg<BF16>()   // Q_in
        .Arg<F32>()    // ADT_in
        .Ret<BF16>()   // Y_partial (N_SPLITS, B, L, H, P)
);

// ============================================================================
// Handler 2: Reduction (sum partials along N_SPLITS axis)
// ============================================================================
ffi::Error ReduceYOffHost(
    cudaStream_t stream,
    BF16 Y_partial,              // (N_SPLITS, B, L, H, P)
    ffi::Result<BF16> Y_out)     // (B, L, H, P)
{
    auto dims = Y_partial.dimensions();
    const int n_splits = static_cast<int>(dims[0]);
    const int B = static_cast<int>(dims[1]);
    const int L = static_cast<int>(dims[2]);
    const int H = static_cast<int>(dims[3]);
    const int P = static_cast<int>(dims[4]);
    const int64_t total = (int64_t)B * L * H * P;

    reduce_y_off_launch(
        reinterpret_cast<const __nv_bfloat16*>(Y_partial.typed_data()),
        reinterpret_cast<__nv_bfloat16*>(Y_out->typed_data()),
        total, n_splits, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ReduceYOff, ReduceYOffHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<BF16>()   // Y_partial
        .Ret<BF16>()   // Y_out
);
