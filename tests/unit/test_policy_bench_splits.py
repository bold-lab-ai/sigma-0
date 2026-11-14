"""The bench's date splits: disjoint, in time order, after the models' data, enforced on episodes."""

import pytest

from lobpipeline.policy_bench.splits import MODEL_DATA_END, SPLITS, check_episodes, in_split, split_of
from lobpipeline.policy_bench.types import Episode


def test_splits_are_in_time_order_after_the_models_data():
    spans = [SPLITS[k] for k in ("train", "dev", "test")]
    assert MODEL_DATA_END < spans[0][0]
    for (lo, hi), (nlo, _) in zip(spans, spans[1:]):
        assert lo <= hi < nlo


def test_each_date_has_one_split():
    assert split_of("2025-12-31") is None
    assert split_of("2026-01-02") == "train"
    assert split_of("2026-01-15") == "train"
    assert split_of("2026-01-16") == "dev"
    assert split_of("2026-01-30") == "dev"
    assert split_of("2026-02-02") == "test"
    assert split_of("2026-02-13") == "test"
    assert split_of("2026-02-17") is None


def test_in_split_keeps_only_its_dates():
    entries = [dict(date=d) for d in ("2025-12-15", "2026-01-05", "2026-01-20", "2026-02-03")]
    assert [e["date"] for e in in_split(entries, "train")] == ["2026-01-05"]
    assert [e["date"] for e in in_split(entries, "dev")] == ["2026-01-20"]
    with pytest.raises(ValueError):
        in_split(entries, "validation")


def test_check_episodes_refuses_a_date_of_another_split():
    dev = [Episode(dict(date="2026-01-20"), 5000), Episode(dict(date="2026-01-28"), 5000)]
    check_episodes(dev, "dev")
    with pytest.raises(ValueError, match="outside the test split"):
        check_episodes(dev, "test")
    with pytest.raises(ValueError, match="outside the dev split"):
        check_episodes(dev + [Episode(dict(date="2026-01-08"), 5000)], "dev")
