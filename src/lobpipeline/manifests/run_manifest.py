"""Run manifest model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunManifest:
    """Minimal run manifest."""

    run_name: str
    source_path: str | None = None

