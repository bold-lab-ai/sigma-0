"""Storage back-ends for pipeline artifacts.

``sqfs`` packs a directory of many small files into a single read-only SquashFS
image, so that a Lustre project quota is charged one inode instead of tens of
thousands.
"""

from . import sqfs

__all__ = ["sqfs"]
