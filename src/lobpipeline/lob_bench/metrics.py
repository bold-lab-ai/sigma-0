"""LOB bench metric definitions."""


def metric_names() -> tuple[str, ...]:
    """Return canonical LOB bench distance names."""
    return ("l1", "wasserstein", "ks")

