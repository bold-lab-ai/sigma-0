"""Return benchmark package."""

from .run_return_bench import run_return_bench

__all__ = [
    "run_return_bench",
    # Event-anchored bench: mid return N milliseconds after a trade.
    "price_at_horizon",
    "sample_returns",
    "compute_metrics",
    "shuffle_null",
    "clock_diagnostics",
    "distribution_comparison",
]

# Exported lazily so importing this package keeps costing nothing. event_anchored pulls
# in pandas and scipy, and the existing run_return_bench stub needs neither.
_EVENT_EXPORTS = frozenset(__all__) - {"run_return_bench"}


def __getattr__(name: str):
    if name in _EVENT_EXPORTS:
        from . import event_anchored

        return getattr(event_anchored, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
