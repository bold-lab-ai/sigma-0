/******************************************************************************
 * Reduction kernel: Y_off = sum_split(Y_off_partial[split])
 *
 * Input:  Y_off_partial (N_SPLITS, B, L, H, P) bf16
 * Output: Y_off         (B, L, H, P) bf16
 *
 * Simple elementwise sum. Each thread handles ONE (b, l, h, p) element.
 *****************************************************************************/

#include "state_scan_common.cuh"
#include <cstdint>

__global__ void reduce_y_off_kernel(
    const __nv_bfloat16* __restrict__ Y_off_partial,  // (N_SPLITS, B, L, H, P)
    __nv_bfloat16*       __restrict__ Y_off_out,      // (B, L, H, P)
    int64_t total_elems,   // B * L * H * P
    int n_splits)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    float acc = 0.f;
    for (int s = 0; s < n_splits; s++) {
        int64_t src_idx = (int64_t)s * total_elems + idx;
        acc += bf16_to_f(Y_off_partial[src_idx]);
    }
    Y_off_out[idx] = f_to_bf16(acc);
}

extern "C" void reduce_y_off_launch(
    const __nv_bfloat16* Y_off_partial,
    __nv_bfloat16*       Y_off_out,
    int64_t total_elems,
    int n_splits,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int64_t grid_size = (total_elems + block_size - 1) / block_size;
    reduce_y_off_kernel<<<grid_size, block_size, 0, stream>>>(
        Y_off_partial, Y_off_out, total_elems, n_splits);
}
