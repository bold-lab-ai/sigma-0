"""The scorer in this repository is a recorded merge, not a directory of files.

These tests are cheap and run everywhere: they read the tree, they never import
it. Importing would pull in the scoring dependency set, which by design is not
installed alongside sigma-0.
"""

from __future__ import annotations

import json

import pytest

from lobpipeline.lob_bench.scoring import (
    LOBBENCH_FEATURES,
    benchmark_revision,
    declared_features,
    mode_features,
)
from lobpipeline.scorer import (
    MERGE_RECORD_PATH,
    SCORER_ROOT,
    merged_benchmark_root,
    merged_tree,
    read_merge_record,
    recorded_revision,
    source_files,
    tree_sha256,
)

# Every module the scorer actually runs. The merge must not have moved any of
# them, which is what makes it safe to run against numbers already published.
PRODUCTION_MODULES = (
    "run_bench.py",
    "scoring.py",
    "metrics.py",
    "partitioning.py",
    "eval.py",
    "data_loading.py",
    "plotting.py",
    "impact.py",
    "run_plotting.py",
    "merge_shards.py",
)


def test_the_tree_still_hashes_to_what_the_record_says() -> None:
    """An edited file is a different tree, and must say so."""
    tree = merged_tree("lob_bench")
    record = read_merge_record("lob_bench")
    assert tree_sha256(tree) == record["tree_sha256"]
    assert len(source_files(tree)) == record["file_count"]


def test_importing_the_tree_does_not_change_what_it_hashes_to(tmp_path) -> None:
    """Byte-compiling the tree must not invalidate its record.

    Python writes .pyc files beside any package it imports, and pytest imports
    this one. A hash that counted those would turn "somebody ran the tests" into
    "this is no longer the merge it claims to be".
    """
    (tmp_path / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = tree_sha256(tmp_path)

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-312.pyc").write_bytes(b"\x00compiled")
    (tmp_path / "stray.pyc").write_bytes(b"\x00compiled")

    assert tree_sha256(tmp_path) == before
    assert [p.name for p in source_files(tmp_path)] == ["mod.py"]


def test_the_record_names_a_base_and_both_sides() -> None:
    """A merge has three inputs; recording fewer would hide one of them."""
    record = read_merge_record("lob_bench")
    for side in ("base", "ours", "theirs"):
        commit = record[side]["commit"]
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)
    assert record["base"]["commit"] != record["ours"]["commit"]
    assert record["base"]["commit"] != record["theirs"]["commit"]
    assert record["ours"]["commit"] != record["theirs"]["commit"]


def test_an_unknown_tree_lists_the_recorded_ones() -> None:
    with pytest.raises(KeyError, match="recorded trees"):
        read_merge_record("lob_pipeline")


def test_the_merged_root_is_addressable_as_a_benchmark_root() -> None:
    """The scorer is found at <root>/lob_bench/run_bench.py, as for any root."""
    root = merged_benchmark_root()
    assert (root / "lob_bench" / "run_bench.py").is_file()


def test_a_merged_root_reports_its_tree_hash_not_a_git_head() -> None:
    """A recorded revision is only useful if it resolves off this machine."""
    record = read_merge_record("lob_bench")
    assert recorded_revision(SCORER_ROOT) == record["tree_sha256"]
    assert benchmark_revision(SCORER_ROOT) == record["tree_sha256"]


def test_a_plain_directory_has_no_merge_record() -> None:
    assert recorded_revision(SCORER_ROOT / "lob_bench") is None


def test_the_merge_moved_no_production_module() -> None:
    """The merge took tests, scripts and docs, and left the scorer alone.

    This is the property that lets the merged tree be run against numbers that
    are already published: none of the code that computes them changed.
    """
    tree = merged_tree("lob_bench")
    for module in PRODUCTION_MODULES:
        assert (tree / module).is_file(), module


def test_the_merged_driver_still_declares_the_ws21_scores() -> None:
    """Theirs defines *_levels; ours replaced them. The merge kept ours."""
    declared = declared_features(SCORER_ROOT, "uncond")
    assert declared == LOBBENCH_FEATURES
    assert not [name for name in declared if name.endswith("_levels")]
    assert mode_features(SCORER_ROOT, "uncond") == LOBBENCH_FEATURES


def test_every_mode_is_reachable_on_the_merged_tree() -> None:
    for mode in ("uncond", "cond", "context", "time_lagged"):
        names = mode_features(SCORER_ROOT, mode)
        assert names, f"{mode} declares nothing"
        assert all(isinstance(name, str) for name in names)


def test_the_record_carries_what_the_merge_brought_and_what_it_refused() -> None:
    record = read_merge_record("lob_bench")
    assert record["what_theirs_brought"]
    assert record["conflicts_resolved_to_ours"]
    assert "_levels" in json.dumps(record["not_taken"])


def test_the_record_flags_the_demonstration_script_as_not_a_test() -> None:
    """It has no asserts and describes a return structure the code dropped.

    Carrying it in unlabelled would let a reader count 223 lines of coverage
    that cannot fail, and believe a contract that has not held since contextual
    scoring started weighting regimes and summing.
    """
    record = read_merge_record("lob_bench")
    note = record["known_defects_carried_in"]["tests/test_contextual_scoring.py"]
    assert "0 asserts" in note

    script = merged_tree("lob_bench") / "tests" / "test_contextual_scoring.py"
    body = script.read_text(encoding="utf-8")
    assert "assert " not in body
    assert "raise " not in body


def test_the_real_test_theirs_brought_does_assert() -> None:
    """test_parallel.py is the one that checks something and can fail."""
    script = merged_tree("lob_bench") / "tests" / "test_parallel.py"
    body = script.read_text(encoding="utf-8")
    assert body.count("assert ") >= 3
    assert "allclose" in body
