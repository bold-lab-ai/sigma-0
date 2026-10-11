"""Mamba3 FLOP counting for scaling law analysis.

Computes training FLOPs using the formula:

    Total FLOPs = correction(d_model) * 6 * N * D

where:
    N = total trainable parameters
    D = total tokens processed = global_step * gBSZ * seq_len
    correction(d_model) comes from the hardware-calibrated lookup table below.

The correction factor is derived **empirically** via NVIDIA GPM tensor-pipe
counters (`nvidia-smi dmon --gpm-metrics=7,12`) measured at BSZ=1 with queued
launches (no per-step sync) to keep the GPU continuously saturated. See
experiments/scaling_law_plots/agent_outputs/flops_corrections_analysis.md
for the measurement methodology and raw data.

The analytical estimate we previously used (which assumed only a ~1.2x factor
from SSD-specific activation-activation matmuls) under-counted real hardware
FLOPs by 2-3x, because (a) the Mamba3 SSD backward costs ~2.5x forward (not
2x), and (b) XLA-emitted `gemm_fusion_dot_*` kernels do more actual tensor-core
work than the HLO op count suggests. Both effects are captured by the
empirical measurement.
"""


# GH200 BF16 tensor core peak FLOPS (per GPU)
GH200_PEAK_BF16_FLOPS = 989e12  # 989 TFLOPS


# ---------------------------------------------------------------------------
# Empirically measured Mamba3 correction factors.
# Source: GPM profiling runs on a single GH200 GPU at BSZ=1, seq_len=13000,
# n_layers=6, d_state=128, expand=2, headdim=64, rope_fraction=0.5.
# See scaling_law_plots/gpm_profile.batch + gpm_results/ and
# flops_corrections_analysis.md for methodology.
# ---------------------------------------------------------------------------
MAMBA3_MEASURED_CORRECTION = {
    # d_model : measured correction = flops_per_step / (6 * N_params * tokens)
    256:  4.92,
    384:  3.43,
    512:  3.36,
    640:  3.09,
    768:  2.94,
    896:  2.95,
    1024: 2.63,
    1088: 3.11,
    1280: 2.48,
    1664: 2.44,
}

# Linear-in-1/d fit to the measured points: corr(d) ≈ 2.0 + 750/d.
# Asymptote ~2.0 reflects that SSD backward is ~2.5x forward plus non-matmul
# activation-activation ops, giving a persistent ~2x multiplier on 6ND.
_MAMBA3_CORRECTION_ASYMPTOTE = 2.0
_MAMBA3_CORRECTION_INV_D_COEF = 750.0


def mamba3_correction_factor(d_model, seq_len=None, **unused_kwargs):
    """Return the Mamba3 FLOP correction factor for a given d_model.

    Prefers the hardware-measured lookup table; falls back to a 1/d_model fit
    for sizes we have not profiled. `seq_len` and other analytical parameters
    are accepted for backward compatibility but ignored — the measurement
    already integrates over all real compute.

    Args:
        d_model: model width (H). Correction is width-dependent; other hparams
                 (chunk_size, d_state, headdim, expand, rope_fraction) are
                 assumed to match the Mamba3 SISO scaling-law config.
        seq_len: ignored (kept for caller compatibility)
        **unused_kwargs: ignored (kept for caller compatibility)

    Returns:
        float: correction factor (>= 1.0) to multiply against 6*N*D.
    """
    if d_model in MAMBA3_MEASURED_CORRECTION:
        return MAMBA3_MEASURED_CORRECTION[d_model]
    # Fallback: extrapolate via the 1/d fit to the measured points.
    return _MAMBA3_CORRECTION_ASYMPTOTE + _MAMBA3_CORRECTION_INV_D_COEF / d_model


def mamba3_correction_factor_analytical(d_model, seq_len, chunk_size=64, d_state=128,
                              headdim=64, expand=2, rope_fraction=0.5):
    """Compute Mamba3 SISO FLOP correction factor.

    The correction = (matmul_flops + extra_flops) / matmul_flops per token.

    Extra FLOPs per token per layer (not captured by parameter count):
        - Intra-chunk C@B^T attention:  C * nh * N
        - Intra-chunk attn × V:         C * nh * hd
        - State accumulation (B⊗V):     nh * hd * N
        - Cross-chunk propagation:       2 * nh * nc * hd * N / C  (L-dependent)
        - State-to-output (C·state):     nh * hd * N
        - RoPE rotations:               ~4 * nh * rope_dim

    Args:
        d_model: model width (H)
        seq_len: sequence length in tokens (L)
        chunk_size: chunk size for SSD algorithm (C)
        d_state: SSM state dimension (N)
        headdim: head dimension for V (hd)
        expand: expansion factor for d_inner
        rope_fraction: fraction of d_state used for RoPE

    Returns:
        float: correction factor (>= 1.0)
    """
    H = d_model
    d_inner = H * expand
    nh = d_inner // headdim
    hd = headdim
    N = d_state
    C = chunk_size
    nc = (seq_len + C - 1) // C  # number of chunks
    rope_dim = int(N * rope_fraction) // 2

    # Matmul FLOPs per token per layer (from 2 * params_per_layer)
    # in_proj: H -> d_in_proj ≈ 4H + 288
    d_in_proj = 2 * d_inner + 2 * N + 3 * nh + rope_dim
    matmul_per_layer = (
        2 * H * d_in_proj +      # in_proj
        2 * d_inner * H +         # out_proj
        2 * H * H                  # half_glu1 gate (out2)
    )

    # Extra FLOPs per token per layer (activation × activation, not in params)
    extra_per_layer = (
        2 * C * nh * N +           # intra-chunk C@B^T attention
        2 * C * nh * hd +          # intra-chunk attention × V
        2 * nh * hd * N +          # state accumulation (B⊗V → state)
        2 * nh * nc * hd * N // C + # cross-chunk state propagation (L-dependent)
        2 * nh * hd * N +          # state-to-output (C·state → output)
        4 * nh * rope_dim          # RoPE rotations
    )

    correction = (matmul_per_layer + extra_per_layer) / matmul_per_layer
    return correction


def compute_flops_per_step(num_params, tokens_per_step, correction=None,
                            d_model=1024, seq_len=13000, **kwargs):
    """Compute FLOPs for one training step (forward + backward).

    Args:
        num_params: total trainable parameter count
        tokens_per_step: global_batch_size * seq_len
        correction: override correction factor (default: auto-compute for Mamba3)
        d_model: model width (for auto-computing correction)
        seq_len: sequence length (for auto-computing correction)
        **kwargs: passed to mamba3_correction_factor

    Returns:
        dict with flops_per_step, correction, tflops_per_step
    """
    if correction is None:
        correction = mamba3_correction_factor(d_model, seq_len, **kwargs)

    flops_per_step = correction * 6 * num_params * tokens_per_step
    tflops_per_step = flops_per_step / 1e12

    return {
        "flops_per_step": flops_per_step,
        "correction_factor": correction,
        "tflops_per_step": tflops_per_step,
        "num_params": num_params,
        "tokens_per_step": tokens_per_step,
    }


def compute_mfu(flops_per_step, step_time_s, num_gpus,
                peak_flops_per_gpu=GH200_PEAK_BF16_FLOPS):
    """Compute Model FLOP Utilization (MFU).

    MFU = achieved_flops / (num_gpus * peak_flops_per_gpu)

    Args:
        flops_per_step: total FLOPs per training step
        step_time_s: wall clock time for one step (seconds)
        num_gpus: total number of GPUs
        peak_flops_per_gpu: theoretical peak BF16 FLOPS per GPU

    Returns:
        dict with mfu_pct, achieved_tflops, achieved_tflops_per_gpu
    """
    if step_time_s <= 0:
        return {"mfu_pct": 0.0, "achieved_tflops": 0.0, "achieved_tflops_per_gpu": 0.0}

    achieved_flops = flops_per_step / step_time_s
    achieved_tflops = achieved_flops / 1e12
    achieved_per_gpu = achieved_flops / num_gpus
    mfu = achieved_per_gpu / peak_flops_per_gpu * 100

    return {
        "mfu_pct": mfu,
        "achieved_tflops": achieved_tflops,
        "achieved_tflops_per_gpu": achieved_tflops / num_gpus if num_gpus > 0 else 0,
    }


def print_flops_summary(flops_info, num_gpus=None):
    """Print a formatted FLOP summary at training start."""
    print(f"[FLOPs] Params: {flops_info['num_params']:,}")
    print(f"[FLOPs] Tokens/step: {flops_info['tokens_per_step']:,}")
    print(f"[FLOPs] Correction factor: {flops_info['correction_factor']:.3f}")
    print(f"[FLOPs] FLOPs/step: {flops_info['flops_per_step']:.3e}")
    print(f"[FLOPs] TFLOPS/step: {flops_info['tflops_per_step']:.2f}")
    if num_gpus:
        print(f"[FLOPs] Peak BF16 ({num_gpus} GPUs): "
              f"{num_gpus * GH200_PEAK_BF16_FLOPS / 1e12:.0f} TFLOPS")
