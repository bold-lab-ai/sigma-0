from __future__ import annotations

import numpy as np
import pytest

from base_model.data.wide_book import select_wide_book_levels


def test_wider_source_is_sliced_to_requested_simulator_depth() -> None:
    source = np.arange(2 * 4 * 500).reshape(2, 4 * 500)

    selected = select_wide_book_levels(source, 100)

    assert selected.shape == (2, 4 * 100)
    np.testing.assert_array_equal(selected, source[:, : 4 * 100])


def test_exact_source_depth_is_preserved() -> None:
    source = np.arange(4 * 100).reshape(1, 4 * 100)

    selected = select_wide_book_levels(source, 100)

    np.testing.assert_array_equal(selected, source)


def test_source_shallower_than_simulator_is_rejected() -> None:
    source = np.zeros((1, 4 * 99), dtype=np.int32)

    with pytest.raises(ValueError, match="source has 99 levels"):
        select_wide_book_levels(source, 100)


@pytest.mark.parametrize("levels", [0, -1, True, 1.5])
def test_invalid_simulator_depth_is_rejected(levels: object) -> None:
    source = np.zeros((1, 4 * 100), dtype=np.int32)

    with pytest.raises(ValueError, match="positive integer"):
        select_wide_book_levels(source, levels)  # type: ignore[arg-type]
