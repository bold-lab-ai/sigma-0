import numpy as np
import pytest

pytest.importorskip("torch")

from lobpipeline.policy_bench.nanosigma.data import NA_VAL  # noqa: E402
from lobpipeline.policy_bench.daydata import build_add_index  # noqa: E402
from lobpipeline.policy_bench.signal import AddRegistry, price_rel_clip, processed_row  # noqa: E402

NA4 = [NA_VAL] * 4
WIDE = 999_999


def _l10(ask, bid):
    r = np.zeros(43, np.int64)
    r[3], r[5] = ask, bid
    return r


def _row(oid, evt, price_rel, size, t_s, t_ns):
    r = np.full(14, NA_VAL, np.int64)
    r[[0, 1, 4, 5, 8, 9]] = oid, evt, price_rel, size, t_s, t_ns
    return r


def test_price_rel_is_relative_to_the_floored_mid_and_truncated():
    reg = AddRegistry()
    # mid (10_100 + 10_000) / 2 = 10_050, floored to 10_000
    r = processed_row(7, 1, 1, 10_000, 50, 0, 1, 34_200, 5, _l10(10_100, 10_000), reg, WIDE)
    assert list(r[:10]) == [7, 1, 1, 10_000, 0, 50, 0, 1, 34_200, 5]
    assert list(r[10:]) == NA4
    assert processed_row(7, 1, 0, 10_300, 50, 0, 0, 0, 0, _l10(10_100, 10_000), reg, WIDE)[4] == 3
    assert processed_row(7, 1, 1, 9_950, 50, 0, 0, 0, 0, _l10(10_100, 10_000), reg, WIDE)[4] == 0    # toward zero
    assert processed_row(7, 1, 0, 10_050, 50, 0, 0, 0, 0, _l10(10_100, 10_000), reg, WIDE)[4] == 0
    # AKAM 2026-01-05 row 1720: (8_905 - 848_900) / 100 = -8399.95 is stored as -8399
    assert processed_row(7, 1, 1, 8_905, 1, 0, 0, 0, 0, _l10(851_000, 846_900), reg, WIDE)[4] == -8399


def test_refs_come_from_the_first_add_of_the_order():
    reg = AddRegistry()
    add = processed_row(7, 1, 1, 10_000, 50, 0, 1, 34_200, 5, _l10(10_100, 10_000), reg, WIDE)
    reg.observe(add)
    reg.observe(_row(7, 1, 99, 99, 99, 99))                      # a later add of the same id is ignored
    cancel = processed_row(7, 3, 1, 10_000, 50, 1, 0, 34_201, 5, _l10(10_300, 10_000), reg, WIDE)
    assert cancel[4] == -1 and list(cancel[10:]) == [0, 50, 34_200, 5]
    unknown = processed_row(8, 4, 1, 10_000, 10, 0, 0, 34_201, 5, _l10(10_100, 10_000), reg, WIDE)
    assert list(unknown[10:]) == NA4


def test_price_rel_clip_depends_on_the_file():
    reg = AddRegistry()
    regular = price_rel_clip("AAPL/AAPL_2026-01-05_34200000_57600000_message_10_proc.npy")
    extended = price_rel_clip("ABT/ABT_2026-01-13_24900000_57900000_message_10_proc.npy")
    assert (regular, extended) == (999, 999_999)
    far = 10_000 - 186_300                                        # 1,863 ticks below the mid
    assert processed_row(7, 1, 1, far, 1, 0, 0, 0, 0, _l10(10_100, 10_000), reg, regular)[4] == -999
    assert processed_row(7, 1, 1, far, 1, 0, 0, 0, 0, _l10(10_100, 10_000), reg, extended)[4] == -1863
    assert processed_row(7, 1, 0, 10_000 + 99_900, 1, 0, 0, 0, 0, _l10(10_100, 10_000), reg, regular)[4] == 999


def test_size_is_clipped_like_the_files():
    r = processed_row(7, 1, 1, 10_000, 25_000, 0, 0, 0, 0, _l10(10_100, 10_000), AddRegistry(), WIDE)
    assert r[5] == 9999


def test_registry_from_rows_keeps_the_first_add():
    msg = np.stack([_row(1, 1, 3, 10, 1, 2), _row(1, 1, 4, 11, 1, 3), _row(2, 3, 0, 5, 1, 4), _row(2, 1, -2, 6, 1, 5)])
    reg = AddRegistry.from_rows(msg)
    assert reg.ref(1) == (3, 10, 1, 2) and reg.ref(2) == (-2, 6, 1, 5) and reg.ref(9) == tuple(NA4)


def test_add_index_answers_for_the_rows_before_the_decision():
    msg = np.stack([_row(5, 1, 3, 10, 1, 2), _row(5, 3, 0, 4, 1, 3), _row(6, 1, -9999, 6, 1, 4),
                    _row(5, 1, 8, 8, 1, 5), _row(7, 1, 2, 9, 1, 6)])
    ix = build_add_index(msg[:, :], chunk=2)                     # chunks split the day
    assert ix.flags.c_contiguous and ix.tolist() == [[5, 6, 7], [0, 2, 4], [3, -9999, 2],
                                                     [10, 6, 9], [1, 1, 1], [2, 4, 6]]
    reg = AddRegistry(ix, before=3)
    assert reg.ref(5) == (3, 10, 1, 2) and reg.ref(6) == (-9999, 6, 1, 4)
    assert reg.ref(7) == tuple(NA4) and reg.ref(4) == tuple(NA4) and reg.ref(9) == tuple(NA4)
    reg.observe(_row(6, 1, 0, 0, 0, 0))                          # a known order whose price_rel is -9999
    reg.observe(_row(7, 1, 1, 1, 1, 1))                          # added after the decision
    assert reg.ref(6) == (-9999, 6, 1, 4) and reg.ref(7) == (1, 1, 1, 1)
    assert AddRegistry(np.zeros((6, 0), np.int64)).ref(5) == tuple(NA4)
