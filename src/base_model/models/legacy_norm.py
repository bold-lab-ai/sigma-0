"""Legacy-checkpoint Mamba3 BCNorm/out_norm epsilon compatibility shim.

Mamba3 checkpoints saved before commits ffe45d66 / 13f04e52 (2026-04-22) were
trained with `nn.RMSNorm(N)` for B_norm/C_norm and `nn.RMSNorm(d_inner)` for
out_norm. The first positional arg of `flax.linen.RMSNorm` is `epsilon`, so those
calls set epsilon to the feature size (d_state for B/C, d_inner for out_norm),
silently near-disabling the norm (divisor ~= sqrt(mean(x^2) + d_state) ~= 32). The
weights were learned under that regime. The current s5/mamba3.py uses the corrected
epsilon=1e-6, which makes B, C and the output norm ~32x larger than at train time
for those old weights, scrambling the SSM dynamics and yielding near-random logits
(overall CE ~8 instead of ~0.6).

s5/mamba3.py is read-only on the shared cluster account, so this module reproduces
the training-time normalizers via a scoped, idempotent runtime monkeypatch of
`Mamba3SSM.setup`. It is OPT-IN via the env var MAMBA3_LEGACY_NORM=1 and is inert
otherwise, so it never changes behaviour for checkpoints trained AFTER 2026-04-22.

Usage (call once, before any Mamba3 model is built):

    from base_model.models.legacy_norm import maybe_install_mamba3_legacy_norm
    maybe_install_mamba3_legacy_norm(ssm_type)        # honours MAMBA3_LEGACY_NORM

Set MAMBA3_LEGACY_NORM=1 in the environment for pre-2026-04-22 Mamba3 checkpoints
(e.g. the 2026-03-28 78M baseline j3417629); leave it unset for newer ones.
"""
from __future__ import annotations

import os


def legacy_norm_requested() -> bool:
    """True iff MAMBA3_LEGACY_NORM is set to a truthy value."""
    return os.environ.get("MAMBA3_LEGACY_NORM", "").strip() not in ("", "0", "false", "False")


def maybe_install_mamba3_legacy_norm(ssm_type: str | None = None, verbose: bool = True) -> bool:
    """Install the legacy BCNorm/out_norm epsilon monkeypatch when requested.

    Args:
        ssm_type: if given and not 'mamba3', this is a no-op (so callers can pass
            args.ssm_type unconditionally).
        verbose: print a one-line confirmation when the patch is installed.

    Returns True iff the patch is (now or already) active. Idempotent: safe to call
    from multiple entry points; the patch is installed at most once.
    """
    if ssm_type is not None and ssm_type != "mamba3":
        return False
    if not legacy_norm_requested():
        return False

    import base_model.models.mamba3 as _m3mod

    if getattr(_m3mod.Mamba3SSM, "_legacy_norm_installed", False):
        return True

    _orig_setup = _m3mod.Mamba3SSM.setup

    def _legacy_setup(self):
        # B_norm/C_norm are the ONLY nn.RMSNorm constructions inside Mamba3SSM.setup
        # (out_norm uses _FullScaleParam). Temporarily swap nn.RMSNorm for a version
        # that injects epsilon=d_state, reproducing the legacy B/C normalizer.
        _real_rmsnorm = _m3mod.nn.RMSNorm
        _legacy_eps_bc = float(self.d_state)

        def _rmsnorm_legacy(*a, **kw):
            kw.setdefault("epsilon", _legacy_eps_bc)
            return _real_rmsnorm(*a, **kw)

        _m3mod.nn.RMSNorm = _rmsnorm_legacy
        try:
            _orig_setup(self)
        finally:
            _m3mod.nn.RMSNorm = _real_rmsnorm
        # out_norm epsilon was d_inner under the legacy nn.RMSNorm(d_inner).
        self._out_norm_eps = float(self.d_inner)

    _m3mod.Mamba3SSM.setup = _legacy_setup
    _m3mod.Mamba3SSM._legacy_norm_installed = True
    if verbose:
        print("[LegacyNorm] Mamba3 BCNorm eps=d_state, out_norm eps=d_inner "
              "(pre-2026-04-22 checkpoint compatibility ENABLED)")
    return True
