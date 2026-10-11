"""Shape contracts for selecting a simulator depth from a wide-book source."""

from __future__ import annotations

import operator
from typing import Any


def select_wide_book_levels(book_l2: Any, levels: int) -> Any:
    """Return exactly ``levels`` LOBSTER levels from a wider L2 snapshot.

    LOBSTER stores four columns per level:
    ask price, ask size, bid price, bid size.  A source may contain more
    levels than the simulator protocol uses, but it must never contain fewer.
    """

    if isinstance(levels, bool):
        raise ValueError("wide-book levels must be a positive integer")
    try:
        levels = operator.index(levels)
    except TypeError as exc:
        raise ValueError("wide-book levels must be a positive integer") from exc
    if levels <= 0:
        raise ValueError("wide-book levels must be a positive integer")

    try:
        source_width = operator.index(book_l2.shape[-1])
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("wide-book input must have a final column dimension") from exc
    if source_width % 4 != 0:
        raise ValueError(
            f"wide-book width must be divisible by 4, found {source_width}"
        )

    required_width = levels * 4
    if source_width < required_width:
        raise ValueError(
            f"wide-book source has {source_width // 4} levels, "
            f"but simulator requires {levels}"
        )
    return book_l2[..., :required_width]
