"""Accelerator kernels: Mamba3 Triton (``mamba3_triton``) and CUDA (``mamba3_cuda``)
state scans, the GDN chunkwise Triton kernel, NSA sliding window.

Everything here is optional at import time; the models fall back to pure JAX.
"""

__all__ = [
    "gdn_triton_kernels",
    "mamba3_cuda",
    "mamba3_triton",
    "nsa",
]
