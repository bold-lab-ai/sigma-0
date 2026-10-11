"""Score manifest model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreManifest:
    """Minimal score manifest."""

    run_name: str
    score_path: str | None = None

