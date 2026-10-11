"""Pinned world-model checkpoints for the agentic market-making loop.

The point of these tests is to keep two decisions from silently rotting:

1. Both pins are intentionally NOT the latest step in their run directory.
   A future "just take the newest checkpoint" cleanup would look like an
   improvement and would quietly swap the models out from under the loop.
2. Resolution is a dict lookup, never a directory scan. Enumerating step
   directories to find the newest is the metadata-storm pattern this
   filesystem's policy bans.
"""

from __future__ import annotations

import dataclasses

import pytest

from post_training.heuristic_learning.checkpoints import (
    MAMBA3_293M_ARCH,
    PINS,
    CheckpointDrift,
    resolve,
    verify,
)


def test_two_pins_cover_the_two_distinct_roles() -> None:
    roles = {pin.role for pin in PINS.values()}
    assert roles == {"background_order_flow", "price_curve_signal"}


def test_pins_are_deliberately_not_the_latest_step() -> None:
    # Ranking by distributional fit and ranking by direction accuracy disagree,
    # and neither peaks at the last step. If someone "modernises" these to the
    # newest step, this fails loudly instead of degrading the loop silently.
    bg = resolve("bg-flow-293m-s150360")
    assert (bg.step, bg.latest_step_in_dir) == (150360, 168200)
    assert bg.is_latest is False

    sig = resolve("signal-293m-s120000")
    assert (sig.step, sig.latest_step_in_dir) == (120000, 143150)
    assert sig.is_latest is False


def test_both_pins_share_the_293m_architecture() -> None:
    for pin in PINS.values():
        assert pin.arch == MAMBA3_293M_ARCH
    assert MAMBA3_293M_ARCH["d_model"] == 2048
    assert MAMBA3_293M_ARCH["ssm_type"] == "mamba3"
    # book_depth drives the wide-book inference path; a mismatch here is a
    # silent shape error much later.
    assert MAMBA3_293M_ARCH["book_depth"] == 500


def test_every_pin_records_why_that_step_was_chosen() -> None:
    for pin in PINS.values():
        assert pin.rationale.strip(), f"{pin.name} has no rationale"


def test_unknown_pin_names_list_the_known_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        resolve("m3-does-not-exist")
    assert "bg-flow-293m-s150360" in str(excinfo.value)


def test_verify_rejects_metadata_drift() -> None:
    tampered = dataclasses.replace(resolve("bg-flow-293m-s150360"), metadata_sha256="0" * 64)
    if not tampered.step_dir.is_dir():
        pytest.skip("checkpoint root not readable from this host")
    with pytest.raises(CheckpointDrift, match="metadata drift"):
        verify(tampered)


def test_verify_rejects_a_missing_step() -> None:
    absent = dataclasses.replace(resolve("bg-flow-293m-s150360"), step=999999999)
    with pytest.raises(CheckpointDrift, match="step directory absent"):
        verify(absent)


@pytest.mark.parametrize("name", sorted(PINS))
def test_pinned_checkpoints_are_present_and_unchanged_on_disk(name: str) -> None:
    pin = resolve(name)
    if not pin.step_dir.is_dir():
        pytest.skip(f"{pin.step_dir} not readable from this host")
    verify(pin)  # raises CheckpointDrift on incomplete or changed checkpoints
