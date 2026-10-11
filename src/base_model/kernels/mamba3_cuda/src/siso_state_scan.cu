/******************************************************************************
 * K3 v3: State Scan Kernel with N-SPLIT for B=1 parallelism
 *
 * Key change vs v2: state[P][N] is split along N into N_SPLITS chunks.
 * Each CTA owns state[P][N/N_SPLITS] = 64x64 f32 = 16 KB.
 * Grid: (N_SPLITS, H, B) = (8, 12, 1) = 96 CTAs for production B=1 setting.
 * SS_N_SPLITS below is the authoritative value; this comment follows it.
 *
 * SMEM per CTA: ~100 KB -> fits 2 CTAs/SM (228 KB limit).
 * Warps per scheduler doubled: 2 -> 4.
 *
 * Output: Y_off_partial (N_SPLITS, B, L, H, P) bf16
 *   Requires a reduction kernel (reduce_y_off_kernel) to sum partials.
 *
 * Algorithm per CTA:
 *   n_split = blockIdx.x; i_h = blockIdx.y; i_b = blockIdx.z
 *   n_offset = n_split * SS_N_LOCAL   (0, 64, ... 448 at SS_N=512, splits=8)
 *   state_local = zeros(P, N_LOCAL)   // in SMEM, persistent
 *   for c in 0..nc-1:
 *     Load Q[b,c,h, n_offset:n_offset+N_LOCAL]     (this CTA's portion)
 *     Load K[b,c,h, n_offset:n_offset+N_LOCAL]     (this CTA's portion)
 *     Load V[b,c,h, :]                              (full, shared across splits)
 *     Compute Y_off_partial = Q_local @ state_local^T * decay  (for this N slice)
 *     Write Y_off_partial[n_split, b, c, h, :] to HBM
 *     Update state_local += V_scaled^T @ K_local
 *****************************************************************************/

#include "state_scan_common.cuh"
#include <math_constants.h>
#include <cstdint>

// N-split config
// At SS_N=512: splits=8 gives N_LOCAL=64, ~96 KB/CTA, 2 CTAs/SM (sweet spot)
// N_SPLITS=16 tested negatively: V buffer (not N-split) -> 2x traffic, net -15% perf
// At SS_N=128: splits=8 gives N_LOCAL=16 = WMMA_K minimum (cannot split further)
#define SS_N_SPLITS  8
#define SS_N_LOCAL   (SS_N / SS_N_SPLITS)  // 512 / 8 = 64 at SS_N=512

struct StateScanSmem {
    // Persistent across chunks (this CTA's portion of state)
    float state[SS_P][SS_N_LOCAL];                  // 16 KB (vs 32 KB full)

    // Double-buffered chunk inputs
    __nv_bfloat16 K_buf[2][SS_CS][SS_N_LOCAL];      // 16 KB (vs 32)
    __nv_bfloat16 V_buf[2][SS_CS][SS_P];            // 16 KB (V is full, shared)
    __nv_bfloat16 Q_buf[2][SS_CS][SS_N_LOCAL];      // 16 KB (vs 32)
    float adt_buf[2][SS_CS];                         //  512 B

    // UNION: state_cast and V_scaled have mutually-exclusive lifetimes
    // per chunk. state_cast is written+read BEFORE V_scaled overwrites it.
    // At SS_N=512: both [64][64]=8 KB, saves 8 KB (96->88 KB SMEM/CTA).
    union {
        __nv_bfloat16 state_cast[SS_P][SS_N_LOCAL];  // 8 KB (used first)
        __nv_bfloat16 V_scaled[SS_CS][SS_P];         // 8 KB (used after)
    };
    float Y_off[SS_CS][SS_P];                        // 16 KB
    float a_cumsum[SS_CS];                           //  256 B
};
// Total: 16+8+16+16+16+0.5+8+16+0.25 = ~97 KB -> 2 CTAs/SM (228/2=114 KB)

// ============================================================================
// Async load helpers (N-local portion of K/Q, full V)
// ============================================================================

__device__ __forceinline__ void async_load_chunk(
    StateScanSmem& s, int slot,
    const __nv_bfloat16* __restrict__ K_in,
    const __nv_bfloat16* __restrict__ V_in,
    const __nv_bfloat16* __restrict__ Q_in,
    int i_b, int i_h, int cs0, int n_offset,
    int L, int H, int tid)
{
    const int64_t bL = (int64_t)i_b * (int64_t)L;

    // K (N-local portion): CS * N_LOCAL bf16 = 64*64*2 = 8192 bytes = 512 x 16B
    constexpr int K_LOCAL_SLOTS = (SS_CS * SS_N_LOCAL * 2) / 16;  // 512
    for (int i = tid; i < K_LOCAL_SLOTS; i += SS_BLK) {
        int flat = i * 8;
        int sl = flat / SS_N_LOCAL;
        int n_local = flat % SS_N_LOCAL;
        int n_global = n_offset + n_local;
        int gl = cs0 + sl;
        int64_t gidx = ((bL + gl) * H + i_h) * (int64_t)SS_N + n_global;
        cp_async_16B(&s.K_buf[slot][sl][n_local], &K_in[gidx]);
    }
    // Q (N-local portion): same as K
    for (int i = tid; i < K_LOCAL_SLOTS; i += SS_BLK) {
        int flat = i * 8;
        int sl = flat / SS_N_LOCAL;
        int n_local = flat % SS_N_LOCAL;
        int n_global = n_offset + n_local;
        int gl = cs0 + sl;
        int64_t gidx = ((bL + gl) * H + i_h) * (int64_t)SS_N + n_global;
        cp_async_16B(&s.Q_buf[slot][sl][n_local], &Q_in[gidx]);
    }
    // V (full, same for all splits)
    constexpr int V_SLOTS = (SS_CS * SS_P * 2) / 16;  // 512
    for (int i = tid; i < V_SLOTS; i += SS_BLK) {
        int flat = i * 8;
        int sl = flat / SS_P;
        int p  = flat % SS_P;
        int gl = cs0 + sl;
        int64_t gidx = ((bL + gl) * H + i_h) * (int64_t)SS_P + p;
        cp_async_16B(&s.V_buf[slot][sl][p], &V_in[gidx]);
    }
}

__device__ __forceinline__ void load_adt_sync(
    float* dst, const float* ADT,
    int i_b, int i_h, int cs0, int L, int H, int tid)
{
    const int64_t base = ((int64_t)i_b * H + i_h) * (int64_t)L + cs0;
    for (int i = tid; i < SS_CS; i += SS_BLK) {
        dst[i] = ADT[base + i];
    }
}

__device__ __forceinline__ void prefix_sum_seq(const float* in, float* out, int n) {
    float acc = 0.f;
    for (int i = 0; i < n; i++) {
        acc += in[i];
        out[i] = acc;
    }
}

// ============================================================================
// Main kernel
// ============================================================================

// launch_bounds(maxThreads, minBlocksPerSM): hint compiler to target occupancy.
// At SS_N=512, N_SPLITS=16, N_LOCAL=32 -> SMEM ~68 KB/CTA -> 228/68=3 CTAs/SM.
// At SS_N=128, N_SPLITS=8,  N_LOCAL=16 -> SMEM ~55 KB/CTA -> 4 CTAs/SM.
__global__ void __launch_bounds__(SS_BLK, 3)
siso_state_scan_kernel(
    const __nv_bfloat16* __restrict__ K_in,           // (B, L, H, N)
    const __nv_bfloat16* __restrict__ V_in,           // (B, L, H, P)
    const __nv_bfloat16* __restrict__ Q_in,           // (B, L, H, N)
    const float*         __restrict__ ADT_in,         // (B, H, L)
    __nv_bfloat16*       __restrict__ Y_off_partial,  // (N_SPLITS, B, L, H, P)
    int B, int L, int H, int nc)
{
    extern __shared__ char smem_raw[];
    StateScanSmem& s = *reinterpret_cast<StateScanSmem*>(smem_raw);

    const int n_split_idx = blockIdx.x;   // N-split: 0 or 1
    const int i_h = blockIdx.y;
    const int i_b = blockIdx.z;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int n_offset = n_split_idx * SS_N_LOCAL;

    // Initialize running state to zero (this CTA's N-slice only)
    for (int i = tid; i < SS_P * SS_N_LOCAL; i += SS_BLK) {
        s.state[i / SS_N_LOCAL][i % SS_N_LOCAL] = 0.f;
    }

    // Prefetch first chunk
    if (nc > 0) {
        async_load_chunk(s, 0, K_in, V_in, Q_in, i_b, i_h, 0, n_offset, L, H, tid);
        cp_async_commit();
        load_adt_sync(s.adt_buf[0], ADT_in, i_b, i_h, 0, L, H, tid);
    }
    __syncthreads();

    // Output base: Y_off_partial[n_split_idx, b, :, h, :]
    const int64_t partial_offset = (int64_t)n_split_idx * (int64_t)B * L * H * SS_P;

    for (int ci = 0; ci < nc; ci++) {
        const int slot = ci & 1;
        const int next_slot = (ci + 1) & 1;
        const int cs0 = ci * SS_CS;

        // Prefetch next chunk
        if (ci + 1 < nc) {
            const int cs0_next = (ci + 1) * SS_CS;
            async_load_chunk(s, next_slot, K_in, V_in, Q_in,
                             i_b, i_h, cs0_next, n_offset, L, H, tid);
            cp_async_commit();
            load_adt_sync(s.adt_buf[next_slot], ADT_in,
                          i_b, i_h, cs0_next, L, H, tid);
        }

        // Wait for chunk ci's data
        cp_async_wait<1>();
        __syncthreads();

        // A_cumsum (tid 0) — others idle briefly
        if (tid == 0) {
            prefix_sum_seq(s.adt_buf[slot], s.a_cumsum, SS_CS);
        }
        __syncthreads();

        const float A_end = s.a_cumsum[SS_CS - 1];

        // Cast state to bf16
        for (int i = tid; i < SS_P * SS_N_LOCAL; i += SS_BLK) {
            s.state_cast[i / SS_N_LOCAL][i % SS_N_LOCAL] =
                f_to_bf16(s.state[i / SS_N_LOCAL][i % SS_N_LOCAL]);
        }
        __syncthreads();

        // Y_off_partial = Q_local @ state_local^T (uses state ENTERING chunk)
        // Q_local: [CS][N_LOCAL], state_cast: [P][N_LOCAL], result: [CS][P]
        // NT GEMM: A=Q (row MxK), B=state_cast (col NxK)
        if (wid < SS_GEMM_WARPS) {
            wmma_nt(
                &s.Q_buf[slot][0][0], SS_N_LOCAL,
                &s.state_cast[0][0], SS_N_LOCAL,
                &s.Y_off[0][0], SS_P,
                SS_CS, SS_N_LOCAL, SS_P,
                true, wid);
        }
        __syncthreads();

        // Scale Y_off by exp(A_cumsum) and write partial output to HBM
        const int64_t bL_out = (int64_t)i_b * (int64_t)L;
        for (int i = tid; i < SS_CS * SS_P; i += SS_BLK) {
            int si = i / SS_P;
            int pi = i % SS_P;
            float decay_out = expf(s.a_cumsum[si]);
            float y = s.Y_off[si][pi] * decay_out;
            int64_t gidx = partial_offset + ((bL_out + cs0 + si) * H + i_h) * (int64_t)SS_P + pi;
            Y_off_partial[gidx] = f_to_bf16(y);
        }

        // V_scaled = V * decay_states (parallel with Y_off write)
        for (int i = tid; i < SS_CS * SS_P; i += SS_BLK) {
            int si = i / SS_P;
            int pi = i % SS_P;
            float decay = expf(A_end - s.a_cumsum[si]);
            float v_val = bf16_to_f(s.V_buf[slot][si][pi]);
            s.V_scaled[si][pi] = f_to_bf16(v_val * decay);
        }

        // Scale running state by exp(A_end)
        {
            float chunk_decay = expf(A_end);
            for (int i = tid; i < SS_P * SS_N_LOCAL; i += SS_BLK) {
                s.state[i / SS_N_LOCAL][i % SS_N_LOCAL] *= chunk_decay;
            }
        }
        __syncthreads();

        // state += V_scaled^T @ K_local  (TN GEMM)
        if (wid < SS_GEMM_WARPS) {
            wmma_tn(
                &s.V_scaled[0][0], SS_P,
                &s.K_buf[slot][0][0], SS_N_LOCAL,
                &s.state[0][0], SS_N_LOCAL,
                SS_P, SS_CS, SS_N_LOCAL,
                false, wid);
        }
        __syncthreads();
    }
}

extern "C" void siso_state_scan_launch(
    const __nv_bfloat16* K_in,
    const __nv_bfloat16* V_in,
    const __nv_bfloat16* Q_in,
    const float* ADT_in,
    __nv_bfloat16* Y_off_partial,  // NOTE: shape is (N_SPLITS, B, L, H, P)
    int B, int L, int H,
    cudaStream_t stream)
{
    const int nc = L / SS_CS;
    dim3 grid(SS_N_SPLITS, H, B);  // (N_splits, H, B)
    dim3 block(SS_BLK);
    const size_t smem = sizeof(StateScanSmem);

    // Set SMEM attribute per-call: must be set on EACH CUDA device (context).
    // Previously cached with static bool -> failed under multi-GPU (shard_map)
    // because the attribute was only set on the first device used.
    cudaFuncSetAttribute(siso_state_scan_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    siso_state_scan_kernel<<<grid, block, smem, stream>>>(
        K_in, V_in, Q_in, ADT_in, Y_off_partial,
        B, L, H, nc);
}
