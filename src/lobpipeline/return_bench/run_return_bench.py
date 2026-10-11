"""Return benchmark summary helpers."""

from __future__ import annotations


def run_return_bench(returns: list[float] | tuple[float, ...]) -> dict[str, float]:
    """Compute mean return from a sequence of realized returns."""
    if not returns:
        raise ValueError("returns must be non-empty")
    return {"mean_return": sum(returns) / len(returns)}
