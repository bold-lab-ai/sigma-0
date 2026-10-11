/******************************************************************************
 * State Scan Kernel: Common constants and helpers
 *
 * Target: NVIDIA GH200 Hopper (sm_90a)
 *****************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>  // for uint32_t in cp_async_16B

using namespace nvcuda;

// Compile-time constants (must match Python-side)
constexpr int WARP_SIZE = 32;
constexpr int SS_BLK    = 256;  // threads per block = 8 warps (was 128)
constexpr int SS_GEMM_WARPS = 4;  // warps used for GEMMs (m-tile per warp)

#define SS_CS  64    // chunk_size
#define SS_N   128   // d_state (headdim_qk) [exp_R1_Mamba3 baseline]
#define SS_P   64    // headdim_v

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// BF16 conversion
__device__ __forceinline__ float bf16_to_f(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ __nv_bfloat16 f_to_bf16(float v) {
    return __float2bfloat16(v);
}

// ============================================================================
// cp.async helpers (Hopper/Ampere async copy from global to shared memory)
// Bypasses L1, transfers in 16-byte chunks. Allows compute to overlap with
// HBM loads. Critical for hiding latency in low-occupancy kernels.
// ============================================================================

// Issue a 16-byte async copy from global to shared
__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                 :: "r"(s), "l"(gmem), "n"(16));
}

// Mark all preceding cp_async ops as belonging to a "group"
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most N groups are pending
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Wait for ALL pending cp_async ops
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// ============================================================================
// WMMA GEMM helpers (from aramis's V5 kernel, proven on GH200)
// ============================================================================

// C += A @ B  (A: row-major MxK, B: row-major KxN, C: row-major MxN)
__device__ __forceinline__ void wmma_nn(
    const __nv_bfloat16* A, int la,
    const __nv_bfloat16* B, int lb,
    float* C, int lc,
    int M, int K, int N, bool clr, int wid)
{
    int m0 = wid * WMMA_M;
    if (m0 >= M) return;
    for (int tn = 0; tn < N; tn += WMMA_N) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (clr) wmma::fill_fragment(acc, 0.f);
        else wmma::load_matrix_sync(acc, C + m0*lc + tn, lc, wmma::mem_row_major);
        for (int tk = 0; tk < K; tk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bf;
            wmma::load_matrix_sync(af, A + m0*la + tk, la);
            wmma::load_matrix_sync(bf, B + tk*lb + tn, lb);
            wmma::mma_sync(acc, af, bf, acc);
        }
        wmma::store_matrix_sync(C + m0*lc + tn, acc, lc, wmma::mem_row_major);
    }
}

// C += A^T @ B  (A: row-major KxM stored col-major, B: row-major KxN)
__device__ __forceinline__ void wmma_tn(
    const __nv_bfloat16* A, int la,
    const __nv_bfloat16* B, int lb,
    float* C, int lc,
    int M, int K, int N, bool clr, int wid)
{
    int m0 = wid * WMMA_M;
    if (m0 >= M) return;
    for (int tn = 0; tn < N; tn += WMMA_N) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (clr) wmma::fill_fragment(acc, 0.f);
        else wmma::load_matrix_sync(acc, C + m0*lc + tn, lc, wmma::mem_row_major);
        for (int tk = 0; tk < K; tk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> af;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bf;
            wmma::load_matrix_sync(af, A + tk*la + m0, la);
            wmma::load_matrix_sync(bf, B + tk*lb + tn, lb);
            wmma::mma_sync(acc, af, bf, acc);
        }
        wmma::store_matrix_sync(C + m0*lc + tn, acc, lc, wmma::mem_row_major);
    }
}

// C += A @ B^T  (A: row-major MxK, B: col-major NxK)
__device__ __forceinline__ void wmma_nt(
    const __nv_bfloat16* A, int la,
    const __nv_bfloat16* B, int lb,
    float* C, int lc,
    int M, int K, int N, bool clr, int wid)
{
    int m0 = wid * WMMA_M;
    if (m0 >= M) return;
    for (int tn = 0; tn < N; tn += WMMA_N) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (clr) wmma::fill_fragment(acc, 0.f);
        else wmma::load_matrix_sync(acc, C + m0*lc + tn, lc, wmma::mem_row_major);
        for (int tk = 0; tk < K; tk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, A + m0*la + tk, la);
            wmma::load_matrix_sync(bf, B + tn*lb + tk, lb);
            wmma::mma_sync(acc, af, bf, acc);
        }
        wmma::store_matrix_sync(C + m0*lc + tn, acc, lc, wmma::mem_row_major);
    }
}
