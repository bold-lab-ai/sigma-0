"""Regression pins for the validation class, D-V1 through D-V3, and for D-R6.

The diagnostic is the thing that decides which defect you are looking at, so a
wrong diagnostic is worse than a missing one: it produces a confident wrong
attribution. These tests pin the distinctions the report is supposed to draw.

No engine import, no jax. See docs/fidelity_defect_taxonomy.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from post_training.heuristic_learning.fidelity import (
    INIT_ID,
    LEGACY_LOOKUP_MISS,
    LOOKUP_MISS,
    describe_stream,
    init_id_span,
)


# columns: type, side, qty, price, oid, tid, time_s, time_ns
def _msg(type_, oid, t=0):
    return [type_, 1, 10, 100_00, oid, 0, 34200 + t, 0]


def _stream(rows):
    return np.asarray(rows, dtype=np.int64)


# --------------------------------------------------------------- constants --
def test_mirrored_constants_still_match_the_engine():
    """These are copied so the diagnostic needs no engine. Copies rot."""
    from matching_engine.jaxob import jaxob_constants as cst
    assert INIT_ID == cst.INITID
    assert LOOKUP_MISS == cst.NEGATIVE_RETURN_ID


def test_init_id_span_matches_what_init_msgs_from_l2_writes():
    """`init_id - arange(0, 2*L)`, so the ids run from INIT_ID down 2L-1 steps."""
    assert init_id_span(10) == (-21, -2)
    assert init_id_span(500) == (-1001, -2)
    assert init_id_span(1) == (-3, -2)


# ------------------------------------------------------------------- D-R6 --
def test_the_miss_sentinel_is_disjoint_from_the_init_id_space():
    """D-R6, now inverted: this test used to assert the collision existed.

    LOOKUP_MISS must sit outside [INIT_ID - (2L-1), INIT_ID] for every depth
    anyone will plausibly run. At -99 it did not: from L=49 upwards the
    sentinel named a real init order, and cancel_order matches on id alone, so
    a failed lookup removed quantity from it. At the generator's L=500 that was
    880 messages per run, roughly half of them landing on a bid level.
    """
    assert LOOKUP_MISS < init_id_span(100_000)[0], (
        "the sentinel must stay below any init id a plausible book_depth hands out"
    )
    for depth in (10, 100, 500, 10_000, 100_000):
        lo, hi = init_id_span(depth)
        assert not (lo <= LOOKUP_MISS <= hi), f"collision at book_depth={depth}"


def test_the_config_refuses_a_depth_that_would_recreate_the_collision():
    """Changing the number fixes today. Refusing to construct fixes the class:
    a deep enough book reaches any fixed negative sentinel, and the failure is
    silent because both generator and replay commit it identically, so no
    fidelity assertion can ever see it."""
    from matching_engine.jaxob.jaxob_config import JAXLOB_Configuration

    JAXLOB_Configuration(book_depth=500)          # the generator's depth: fine
    JAXLOB_Configuration(book_depth=100_000)      # far beyond anything used: fine
    with pytest.raises(ValueError, match="miss sentinel"):
        JAXLOB_Configuration(book_depth=500_000_000)


def test_the_legacy_sentinel_is_still_recognised():
    """A constant that reaches persisted records is part of the file format.
    Records written before the change carry -99, and reading them as
    out-of-depth init references instead of misses would misattribute an
    entire class of defect."""
    s = _stream([_msg(3, LEGACY_LOOKUP_MISS, 0), _msg(3, LOOKUP_MISS, 1)])
    r = describe_stream(s, prior_ids=[], n_levels=10)
    assert r.n_lookup_miss_sentinel == 2
    assert r.n_init_ref_out_of_depth == 0, "a miss must not be filed as out-of-depth"


# ------------------------------------------------------------------- D-V3 --
def test_benign_positive_ids_are_not_counted_as_bad():
    """A real stream carries 12 to 38 cancels of pre-window orders and still
    replays exactly. Lumping those with fatal references is what made one
    counter useless."""
    s = _stream([_msg(1, 555, 0), _msg(3, 999_111, 1), _msg(3, 999_222, 2)])
    r = describe_stream(s, prior_ids=[], n_levels=10)
    assert r.n_unresolvable_ids == 2
    assert r.n_unresolved_positive == 2
    assert r.n_init_ref_in_depth == 0
    assert r.n_init_ref_out_of_depth == 0
    assert r.n_lookup_miss_sentinel == 0
    assert r.first_bad_id_step == -1, "benign ids must not set the onset marker"


def test_the_four_categories_are_separated():
    s = _stream([
        _msg(1, 555, 0),          # a new order, becomes known
        _msg(3, 555, 1),          # resolvable, not counted at all
        _msg(3, 999_111, 2),      # positive, unseen  -> benign
        _msg(3, LOOKUP_MISS, 3),  # the generator's lookup failed
        _msg(3, -8, 4),           # init order, inside a 10-level replay
        _msg(3, -86, 5),          # init order, far beyond a 10-level replay
    ])
    r = describe_stream(s, prior_ids=[], n_levels=10)
    assert r.n_cancel_delete == 5
    assert r.n_unresolvable_ids == 4
    assert r.n_unresolved_positive == 1
    assert r.n_lookup_miss_sentinel == 1
    assert r.n_init_ref_in_depth == 1
    assert r.n_init_ref_out_of_depth == 1


def test_depth_reclassifies_in_range_and_out_of_range():
    """The same id is fatal at one width and fine at another. Getting n_levels
    wrong silently turns fatal references into benign ones, which is the exact
    failure mode this split exists to prevent."""
    s = _stream([_msg(3, -86, 0)])
    assert describe_stream(s, n_levels=10).n_init_ref_out_of_depth == 1
    assert describe_stream(s, n_levels=10).n_init_ref_in_depth == 0
    assert describe_stream(s, n_levels=500).n_init_ref_out_of_depth == 0
    assert describe_stream(s, n_levels=500).n_init_ref_in_depth == 1


# ------------------------------------------------------- onset vs magnitude --
def test_first_bad_id_step_records_position_not_just_volume():
    """Causation needs order, not counts. A bad id after divergence has begun
    can only have amplified it; four of eight measured windows diverge with no
    bad id beforehand at all."""
    s = _stream([_msg(1, 1, 0), _msg(1, 2, 1), _msg(3, -8, 2), _msg(3, -9, 3)])
    r = describe_stream(s, n_levels=10)
    assert r.first_bad_id_step == 2
    assert r.n_init_ref_in_depth == 2


def test_a_stream_with_no_cancels_reports_cleanly():
    s = _stream([_msg(1, 1, 0), _msg(1, 2, 1)])
    r = describe_stream(s, n_levels=10)
    assert r.n_cancel_delete == 0
    assert r.n_unresolvable_ids == 0
    assert r.first_bad_id_step == -1


def test_report_dict_exposes_every_new_field():
    """The breakdown has to reach the caller, or splitting it changed nothing."""
    r = describe_stream(_stream([_msg(3, -8, 0)]), n_levels=10).as_dict()
    for k in ("n_unresolved_positive", "n_lookup_miss_sentinel",
              "n_init_ref_in_depth", "n_init_ref_out_of_depth", "first_bad_id_step"):
        assert k in r, f"{k} missing from the report"


# ------------------------------------------------------------------- D-R5 --
def test_provenance_decodes_the_three_facts_it_packs():
    from post_training.heuristic_learning.fidelity import (
        PROV_NAN_FIRED, PROV_PRICE_DIFFERS_FROM_REF, decode_provenance)

    assert decode_provenance(0) == {"resolution": "not-a-cancel",
                                    "nan_branch_fired": False,
                                    "price_differs_from_ref": False}
    assert decode_provenance(1)["resolution"] == "exact-timestamp-hit"
    assert decode_provenance(2)["resolution"] == "price-fallback"
    assert decode_provenance(3)["resolution"] == "total-miss"
    # The bits are independent: a miss that also tripped the NaN branch and
    # whose ref price disagreed is 3 + 4 + 8, and all three must survive.
    d = decode_provenance(3 | PROV_NAN_FIRED | PROV_PRICE_DIFFERS_FROM_REF)
    assert d == {"resolution": "total-miss", "nan_branch_fired": True,
                 "price_differs_from_ref": True}


def test_provenance_summary_counts_what_two_open_defects_needed():
    """D-R1's firing rate and D-R4's disagreement rate were both recorded as
    unmeasured, because the record carried nothing to measure them with."""
    from post_training.heuristic_learning.fidelity import summarise_provenance

    s = summarise_provenance([0, 1, 1, 2, 3, 3 | 4, 1 | 8])
    assert s["resolution_not-a-cancel"] == 1
    assert s["resolution_exact-timestamp-hit"] == 3
    assert s["resolution_price-fallback"] == 1
    assert s["resolution_total-miss"] == 2
    assert s["nan_branch_fired"] == 1
    assert s["price_differs_from_ref"] == 1


def test_the_price_ref_bit_is_not_labelled_as_a_defect():
    """D-R4's control, pinned as a naming rule.

    The bit fires on 55.8% of REAL cancels, which are consistent by
    construction, so its inequality carries no information about the model. It
    was called `ref_price_disagrees` and read as a defect count until the
    control ran. The name is part of the fix: a field whose name asserts a
    refuted conclusion will be re-read as that conclusion by the next person.
    """
    from post_training.heuristic_learning import fidelity

    assert not hasattr(fidelity, "PROV_REF_DISAGREES"), (
        "the old name asserted a conclusion the control refuted"
    )
    assert hasattr(fidelity, "PROV_PRICE_DIFFERS_FROM_REF")
    keys = fidelity.summarise_provenance([0])
    assert "price_differs_from_ref" in keys
    assert "ref_price_disagrees" not in keys
