"""Pipeline configuration contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Minimal pipeline config."""

    name: str
    output_root: str | None = None

