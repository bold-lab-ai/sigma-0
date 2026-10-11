"""Sliding Window Attention ops — CUDA C++ + XLA FFI backend.

This module will load the compiled libswa_kernels.so and register
the CUDA kernels with JAX's FFI. Until the .so is built, the model
falls back to the pure JAX implementation in s5/nsa.py.

Build the CUDA kernels:
    cd csrc && mkdir -p build && cd build
    cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
    make -j

Usage (once built):
    from base_model.kernels.nsa.swa_ops import sliding_window_attention_cuda
    o = sliding_window_attention_cuda(q, k, v, window_size=512, scale=0.125)
"""
import os
import jax
import jax.numpy as jnp
from pathlib import Path


# <repo>/src/csrc/build, as when this module lived at src/nsa_kernels/swa_ops.py
_LIB_PATH = Path(__file__).resolve().parents[3] / "csrc" / "build" / "libswa_kernels.so"
_CUDA_AVAILABLE = False

if _LIB_PATH.exists():
    try:
        from jax.extend import ffi as ffi_ext
        import ctypes
        _lib = ctypes.CDLL(str(_LIB_PATH))
        ffi_ext.register_ffi_target(
            "swa_fwd", _lib.SwaFwd, platform="CUDA")
        ffi_ext.register_ffi_target(
            "swa_bwd", _lib.SwaBwd, platform="CUDA")
        _CUDA_AVAILABLE = True
    except Exception as e:
        print(f"[NSA] Failed to load CUDA SWA kernels: {e}")
        print("[NSA] Falling back to pure JAX sliding window attention")


def is_cuda_available():
    """Check if CUDA SWA kernels are loaded."""
    return _CUDA_AVAILABLE


@jax.custom_vjp
def sliding_window_attention_cuda(q, k, v, window_size, scale):
    """O(L·W) sliding window causal attention via CUDA kernel.

    Only available after building csrc/. Otherwise raises RuntimeError.
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError(
            "CUDA SWA kernels not built. Run: cd csrc/build && cmake .. && make -j"
        )
    out = jax.ffi.ffi_call(
        "swa_fwd",
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        q, k, v,
        window_size=window_size,
        scale=scale,
    )
    return out


def _swa_fwd(q, k, v, window_size, scale):
    out = sliding_window_attention_cuda(q, k, v, window_size, scale)
    return out, (q, k, v, out)


def _swa_bwd(res, grad_out):
    q, k, v, out = res
    dq, dk, dv = jax.ffi.ffi_call(
        "swa_bwd",
        (jax.ShapeDtypeStruct(q.shape, q.dtype),) * 3,
        grad_out, q, k, v, out,
    )
    return dq, dk, dv, None, None


sliding_window_attention_cuda.defvjp(_swa_fwd, _swa_bwd)
