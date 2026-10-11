"""Data: LOBSTER datasets and loaders, file resolution, preprocessing.

Submodules import jax and torch, so this package stays lazy. Import what you
need: ``base_model.data.lobster_dataloader``, ``.dataloading``, ``.data_files``,
``.wide_book``, ``.preproc``, ``.sequence_dataset``, ``.permutations``,
``.torch_loaders``.
"""

__all__ = [
    "data_files",
    "dataloading",
    "lobster_dataloader",
    "permutations",
    "preproc",
    "sequence_dataset",
    "torch_loaders",
    "wide_book",
]
