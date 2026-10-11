"""Resolving which data files a dataset will see, and in what order.

Kept apart from inference_no_errcorr so that offline tooling can import it without
pulling in flax, jax and the matching engine. Picking window positions is pure CPU
work over .npy headers and has no reason to need any of that.
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import List, Optional

import numpy as np


def resolve_data_files(
        data_dir: str,
        *,
        day_indeces: Optional[List[int]] = None,
        test_split: float = 0.1,
        wide_book_dir: Optional[str] = None,
        verbose: bool = True,
    ):
    """Resolve ``(msg_files, book_files, wide_book_files)`` for a data directory.

    A file's position in these lists *is* the ``file_idx`` that a dataset index
    resolves to, and that position is not simply ``sorted(glob(...))``:
    ``day_indeces`` subsets the list, ``test_split`` keeps only the trailing
    fraction, and truncated .npy files are dropped. Anything computing window
    positions offline has to go through this same function, because an index built
    against a differently-filtered list points at the wrong trading day and nothing
    raises to say so.
    """
    msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
    book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

    if day_indeces is not None:
        # restricts the data to only include certain days.
        msg_files = [msg_files[i] for i in day_indeces]
        book_files = [book_files[i] for i in day_indeces]
    if test_split > 0:
        n_test_files = max(1, int(len(msg_files) * test_split))
        msg_files = msg_files[-n_test_files:]
        book_files = book_files[-n_test_files:]

    # Filter out truncated .npy files (try mmap load, skip on failure)
    valid_pairs = []
    for mf, bf in zip(msg_files, book_files):
        ok = True
        for f in (mf, bf):
            try:
                a = np.load(f, mmap_mode='r')
                _ = a.shape  # force header parse
                del a
            except Exception as e:
                if verbose:
                    print(f"[get_dataset] Skipping bad file {os.path.basename(f)}: {e}")
                ok = False
                break
        if ok:
            valid_pairs.append((mf, bf))
    if verbose and len(valid_pairs) < len(msg_files):
        print(f"[get_dataset] Kept {len(valid_pairs)}/{len(msg_files)} file pairs after validation")
    msg_files, book_files = zip(*valid_pairs) if valid_pairs else ([], [])

    # Build wide_book_files list by matching dates from book_files
    wide_book_files = None
    if wide_book_dir is not None:
        wide_book_files = []
        for bf in book_files:
            basename = os.path.basename(bf)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
            if date_match is None:
                raise ValueError(f"Cannot extract date from book file: {basename}")
            date_str = date_match.group(1)
            wide_candidates = sorted(glob(
                os.path.join(wide_book_dir, f'*{date_str}*orderbook*proc.npy')))
            if len(wide_candidates) == 0:
                raise FileNotFoundError(
                    f"No wide book file found for date {date_str} in {wide_book_dir}")
            wide_book_files.append(wide_candidates[0])
        if verbose:
            print(f"[get_dataset] Wide book files: {len(wide_book_files)} matched from {wide_book_dir}")

    return list(msg_files), list(book_files), wide_book_files
