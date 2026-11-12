from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import pytest

from lobpipeline.lob_bench.scoring import (
    LOBBENCH_FEATURES,
    LobbenchDivergenceSummary,
    LobbenchSummary,
)
from lobpipeline.lob_bench.standard_eval import (
    CANONICAL_INDEX_FILE,
    STANDARD_EVAL_FORMAT,
    build_record,
    directory_digest,
    load_protocol,
    pooled_intervals,
    standard_scorer,
)
from lobpipeline.scorer import read_merge_record
from run.benchmarking.run_standard_eval import main as run_standard_eval

DISTANCES = ("l1", "wasserstein", "ks")


def _index(tmp_path: Path, indices: list[int], *, name: str = "idx.txt") -> Path:
    path = tmp_path / name
    text = "".join(f"{index}\n" for index in indices)
    path.write_text(text, encoding="utf-8")
    metadata = {
        "conditioning_messages": 250,
        "generated_messages": 250,
        "sequence_messages": 500,
        "n_samples": len(indices),
        "dataset_length": 1000,
        "period": "2026-01",
        "ticker": "GOOG",
        "sample_indices_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    path.with_name(path.name + ".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return path


def _pooled(revision: str, *, ws21: float | None = 0.5) -> LobbenchSummary:
    return LobbenchSummary(
        run_name="r",
        stock="GOOG",
        period="2026-01",
        mode="uncond",
        score_path="/x/scores.pkl",
        benchmark_root="/x",
        benchmark_revision=revision,
        feature_count=21,
        expected_feature_count=21,
        ws21=ws21,
        ks21=0.2,
        l1_21=0.3,
        distances={"l1": 0.3, "wasserstein": 0.5, "ks": 0.2},
        feature_scores={f: {d: 0.1 for d in DISTANCES} for f in LOBBENCH_FEATURES},
    )


def _divergence(revision: str, horizon: int = 100) -> LobbenchDivergenceSummary:
    curves = {f: {d: [0.1, None] for d in DISTANCES} for f in LOBBENCH_FEATURES}
    return LobbenchDivergenceSummary(
        run_name="r",
        stock="GOOG",
        period="2026-01",
        mode="divergence",
        score_path="/x/div.pkl",
        benchmark_root="/x",
        benchmark_revision=revision,
        horizon_length=horizon,
        n_horizons=2,
        horizon_axis_messages=(horizon, 2 * horizon),
        feature_count=21,
        expected_feature_count=21,
        is_full_panel=True,
        scores_without_windows=(),
        mean_curves={d: [0.1, None] for d in DISTANCES},
        final_over_first={d: None for d in DISTANCES},
        feature_curves=curves,
        feature_curve_ci={f: {d: [None, None] for d in DISTANCES} for f in LOBBENCH_FEATURES},
    )


def _record(tmp_path: Path, **overrides):
    revision = read_merge_record()["tree_sha256"]
    arguments = dict(
        run_name="r",
        protocol=load_protocol(_index(tmp_path, [3, 7])),
        scorer_tree_sha256=revision,
        pooled=_pooled(revision),
        divergence=_divergence(revision),
        intervals={},
        inputs={"data_gen": {"sha256": "g"}},
        provenance={"n_workers": 2},
    )
    arguments.update(overrides)
    return build_record(**arguments)


def test_canonical_protocol_comes_from_the_frozen_index() -> None:
    protocol = load_protocol()
    assert protocol.canonical
    assert protocol.index_sha256 == (
        "0c41de5111b8996b3ac00ca8f481c98b5453dd4c9562e56f9730d364518faf25"
    )
    assert (protocol.stock, protocol.period, protocol.sequences) == ("GOOG", "2026-01", 3136)
    assert (protocol.conditioning_messages, protocol.generated_messages) == (250, 250)
    assert protocol.mode == "uncond"
    assert protocol.divergence_horizon == 100


def test_a_modified_index_file_is_refused(tmp_path: Path) -> None:
    copy = tmp_path / CANONICAL_INDEX_FILE.name
    copy.write_bytes(CANONICAL_INDEX_FILE.read_bytes().replace(b"116\n", b"117\n", 1))
    metadata = CANONICAL_INDEX_FILE.with_name(CANONICAL_INDEX_FILE.name + ".metadata.json")
    copy.with_name(copy.name + ".metadata.json").write_bytes(metadata.read_bytes())
    with pytest.raises(ValueError, match="differs from the frozen value"):
        load_protocol(copy)


def test_a_small_index_is_allowed_but_never_standard(tmp_path: Path) -> None:
    protocol = load_protocol(_index(tmp_path, [3, 7]))
    assert not protocol.canonical
    assert protocol.sequences == 2


def test_the_merged_scorer_is_the_recorded_tree() -> None:
    root, tree = standard_scorer()
    assert (root / "lob_bench" / "run_bench.py").is_file()
    assert tree == read_merge_record()["tree_sha256"]


def test_pooled_intervals_are_read_from_the_score_pickle(tmp_path: Path) -> None:
    scores = {
        feature: {
            "l1": (0.1, (0.05, 0.15), None),
            "wasserstein": (0.2, (math.nan, 0.3), None),
            "ks": 0.3,
        }
        for feature in LOBBENCH_FEATURES
    }
    path = tmp_path / "scores.pkl"
    with gzip.open(path, "wb") as handle:
        pickle.dump((scores, {}), handle)
    intervals = pooled_intervals(path)
    assert intervals["spread"]["l1"] == [0.05, 0.15]
    # A non-finite bound or a bare point is no interval, not an invented one.
    assert intervals["spread"]["wasserstein"] is None
    assert intervals["spread"]["ks"] is None


def test_directory_digest_sees_bytes_and_names(tmp_path: Path) -> None:
    directory = tmp_path / "data_gen"
    directory.mkdir()
    (directory / "a.csv").write_text("1,2\n", encoding="utf-8")
    first = directory_digest(directory)
    (directory / "a.csv").write_text("1,3\n", encoding="utf-8")
    second = directory_digest(directory)
    (directory / "a.csv").rename(directory / "b.csv")
    third = directory_digest(directory)
    assert len({first["sha256"], second["sha256"], third["sha256"]}) == 3
    assert first["files"] == 1 and first["bytes"] == 4


def test_record_carries_headline_features_divergence_and_inputs(tmp_path: Path) -> None:
    record = _record(
        tmp_path, intervals={"spread": {"l1": [0.0, 0.2]}}
    )
    assert record["format"] == STANDARD_EVAL_FORMAT
    assert record["standard"] is False
    assert record["headline"] == {"ws21": 0.5, "ks21": 0.2, "l1_21": 0.3}
    assert list(record["features"]) == list(LOBBENCH_FEATURES)
    assert record["features"]["spread"]["l1"] == {"point": 0.1, "interval": [0.0, 0.2]}
    assert record["features"]["spread"]["ks"]["interval"] is None
    # The curve's nulls survive: the record does not shrink the panel.
    assert record["divergence"]["mean_curves"]["ks"] == [0.1, None]
    assert record["inputs"]["data_gen"]["sha256"] == "g"
    json.dumps(record)


def test_no_record_without_the_headline(tmp_path: Path) -> None:
    revision = read_merge_record()["tree_sha256"]
    with pytest.raises(ValueError, match="ws21 not produced"):
        _record(tmp_path, pooled=_pooled(revision, ws21=None))


def test_no_record_when_the_summary_names_another_scorer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="summary names scorer"):
        _record(tmp_path, pooled=_pooled("0" * 64))


def test_no_record_at_a_non_standard_horizon(tmp_path: Path) -> None:
    revision = read_merge_record()["tree_sha256"]
    with pytest.raises(ValueError, match="divergence horizon 50"):
        _record(tmp_path, divergence=_divergence(revision, horizon=50))


def _inference_tree(root: Path, indices: list[int], manifest_indices: list[int]) -> Path:
    for name in ("data_gen", "data_real", "data_cond"):
        (root / name).mkdir(parents=True)
    for index in indices:
        rows = "0.0,1,1,1,100,1\n" * 250
        book = "1,1,1,1\n" * 250
        stem = f"GOOG_2026-01-02_{{}}_real_id_{index}"
        (root / "data_gen" / f"{stem.format('message')}_gen_id_0.csv").write_text(rows)
        (root / "data_gen" / f"{stem.format('orderbook')}_gen_id_0.csv").write_text(book)
        (root / "data_real" / f"{stem.format('message')}.csv").write_text(rows)
        (root / "data_real" / f"{stem.format('orderbook')}.csv").write_text(book)
        (root / "data_cond" / f"{stem.format('message')}.csv").write_text(rows)
        (root / "data_cond" / f"{stem.format('orderbook')}.csv").write_text(book + "1,1,1,1\n")
    digest = hashlib.sha256(
        json.dumps(manifest_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (root / "sample_indices_rank0.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "world_size": 1,
                "dataset_length": 1000,
                "all_indices_sha256": digest,
                "all_indices": manifest_indices,
                "rank_indices": manifest_indices,
            }
        )
    )
    return root


def test_windows_that_differ_from_the_index_are_refused_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _index(tmp_path, [3, 7])
    tree = _inference_tree(tmp_path / "inference", [3, 8], manifest_indices=[3, 8])

    def no_scoring(*args, **kwargs):
        raise AssertionError("scoring must not start on a wrong window set")

    monkeypatch.setattr("run.benchmarking.run_standard_eval.build_lobbench_plan", no_scoring)
    with pytest.raises(ValueError, match="does not exactly match the canonical"):
        run_standard_eval(
            [
                "--inference-dir", str(tree),
                "--output-dir", str(tmp_path / "out"),
                "--run-name", "r",
                "--world-size", "1",
                "--index-file", str(index),
            ]
        )
