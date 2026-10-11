"""Downstream metric boundary."""


def downstream_metric_names() -> tuple[str, ...]:
    """Return metric names carried through score manifests."""
    return ("l1", "wasserstein", "ks")
