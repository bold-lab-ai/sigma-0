"""Package manifest model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackageManifest:
    """Minimal manifest describing a self-contained run squashfs."""

    run_name: str
    squashfs_path: str | None = None
