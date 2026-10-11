"""The model-store pin table, and the traps it exists to encode.

Two halves. The first needs nothing but the repository and runs anywhere: the
table parses, every entry pins an explicit step and a digest, and the pairs
that are easy to confuse are still distinct. The second needs the shared store
behind `models/` and skips without it, because a clone off the cluster has no
payload to check.
"""

from __future__ import annotations

import json

import pytest

from lobpipeline import model_store


PINS = json.loads(model_store.pins_path().read_text())["models"]


def _store_available() -> bool:
    return model_store.store_path().is_dir()


store_only = pytest.mark.skipif(
    not _store_available(), reason="models/ store not linked or not readable"
)


def test_every_entry_pins_a_step_and_a_digest() -> None:
    assert len(PINS) >= 25
    for name, e in PINS.items():
        assert isinstance(e["step"], int) and e["step"] > 0, name
        assert len(e["metadata_sha256"]) == 64, name
        assert e["source"].startswith("/lus/"), name
        # nothing in the table may express "whichever step is newest"
        assert "latest" not in json.dumps(e).lower(), name


def test_the_two_78m_models_at_step_28730_stay_distinct() -> None:
    """Same size, same step, different training corpus — the standing trap."""
    goog, sp500 = PINS["m3-goog-78m-u6gb"], PINS["m3-sp500-78m-s42"]
    assert goog["step"] == sp500["step"] == 28730
    assert goog["d_model"] == sp500["d_model"] == 1024
    assert goog["source"] != sp500["source"]
    assert goog["corpus"] != sp500["corpus"]


def test_message_only_and_book_variants_are_labelled() -> None:
    assert PINS["m3-megacap8-8p2m-msgonly"]["use_book_data"] is False
    assert PINS["m3-sp500-78m-s42"]["use_book_data"] is True


def test_pre_fix_norm_checkpoints_are_flagged() -> None:
    """Weights trained before the 2026-04-22 norm fix must not look ordinary."""
    assert PINS["m3-megacap8-8p2m-msgonly"]["mamba3_legacy_norm"] == "verify"
    assert PINS["m3-sp500-78m-s42"]["mamba3_legacy_norm"] is False


def test_unknown_model_raises_with_a_hint() -> None:
    with pytest.raises(model_store.ModelStoreError):
        model_store.resolve("m3-sp500-9000m")


@store_only
def test_resolve_reaches_the_payload() -> None:
    m = model_store.resolve("m3-sp500-78m-s42")
    assert m["step"] == 28730
    assert m["path"].endswith("m3-sp500-78m-s42/checkpoint")
    assert model_store.metadata_path("m3-sp500-78m-s42").is_file()


@store_only
@pytest.mark.parametrize("name", ["m3-sp500-78m-s42", "m3-megacap8-8p2m-msgonly"])
def test_pinned_metadata_still_hashes_to_the_recorded_digest(name: str) -> None:
    assert model_store.verify(name), f"{name}: metadata changed since it was pinned"


# --- the RMSNorm epsilon regime is a property of the checkpoint -------------


def test_pre_fix_checkpoints_ask_for_the_legacy_epsilon() -> None:
    """`flax.linen.RMSNorm`'s first positional argument is `epsilon`.

    Weights trained before 2026-04-22 learned under epsilon = the feature count,
    so evaluating them at 1e-6 leaves every normalized tensor up to sqrt(d) too
    large. Measured on GOOG 2026-01, three generation seeds: WS-21 0.8379 ->
    0.2643 for m3-goog-6m-zerobook. The pin is what says which regime applies.
    """
    from lobpipeline.model_store import legacy_norm_required

    assert legacy_norm_required("m3-goog-6m-zerobook") is True
    assert legacy_norm_required("m3-megacap8-78m-wce") is True
    assert legacy_norm_required("m3-sp500-78m-s42") is False


def test_the_environment_says_which_epsilon_and_unsets_the_other() -> None:
    """An empty value is the instruction, not a missing one.

    A launcher that loops over models in one shell inherits whatever the
    previous model exported, so the fixed-epsilon case has to say so rather
    than stay silent.
    """
    from lobpipeline.model_store import inference_env

    assert inference_env("m3-goog-6m-zerobook") == {"MAMBA3_LEGACY_NORM": "1"}
    assert inference_env("m3-sp500-78m-s42") == {"MAMBA3_LEGACY_NORM": ""}


def test_an_unknown_model_refuses_rather_than_defaulting() -> None:
    """Defaulting is how two checkpoints came to be scored under the wrong
    normalizer; an unknown name must stop the run."""
    import pytest

    from lobpipeline.model_store import ModelStoreError, legacy_norm_required

    with pytest.raises(ModelStoreError, match="unknown model"):
        legacy_norm_required("m3-not-a-real-model")


def test_the_benchmark_launcher_reads_the_pin() -> None:
    """The launcher must ask the pin, not the caller.

    Checking the source rather than running it: the script needs four GPUs and
    a squashfs mount, neither of which a unit test has.
    """
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2]
              / "run" / "benchmarking" / "bench_model_store.sh").read_text()
    assert "lobpipeline.model_store env" in script, (
        "bench_model_store.sh must derive MAMBA3_LEGACY_NORM from the pin"
    )
    passed = [line for line in script.splitlines()
              if "--architecture" in line and not line.lstrip().startswith("#")]
    assert not passed, (
        "the checkpoint declares its architecture; the registry's ssm_type "
        f"names the SSM and asserting it breaks the NSA checkpoints: {passed}"
    )
