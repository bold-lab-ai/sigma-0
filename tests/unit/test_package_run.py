import json
from pathlib import Path

import pytest

from lobpipeline.pipeline.package_run import build_package_plan, stage_package


def _make_run(tmp_path: Path) -> tuple[Path, Path]:
    pred = tmp_path / "run" / "predictions"
    pred.mkdir(parents=True)
    (pred / "GOOG_2026-01.npy").write_bytes(b"\x00" * 16)
    scores = tmp_path / "run" / "scores.json"
    scores.write_text(json.dumps({"l1": {"mean": 0.0}}), encoding="utf-8")
    return pred, scores


def test_build_package_plan_paths(tmp_path: Path) -> None:
    pred, scores = _make_run(tmp_path)
    plan = build_package_plan("smoke", pred, scores, tmp_path / "out")
    assert plan.run_name == "smoke"
    assert plan.squashfs_path.endswith("smoke.squashfs")
    assert plan.index_path.endswith("index_smoke.json")


def test_build_package_plan_rejects_empty_run_name(tmp_path: Path) -> None:
    pred, scores = _make_run(tmp_path)
    with pytest.raises(ValueError):
        build_package_plan("", pred, scores, tmp_path / "out")


def test_build_package_plan_rejects_missing_predictions(tmp_path: Path) -> None:
    _, scores = _make_run(tmp_path)
    with pytest.raises(NotADirectoryError):
        build_package_plan("smoke", tmp_path / "nope", scores, tmp_path / "out")


def test_build_package_plan_rejects_missing_scores(tmp_path: Path) -> None:
    pred, _ = _make_run(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_package_plan("smoke", pred, tmp_path / "nope.json", tmp_path / "out")


def test_stage_package_is_self_contained(tmp_path: Path) -> None:
    pred, scores = _make_run(tmp_path)
    plan = build_package_plan("smoke", pred, scores, tmp_path / "out")
    staging = stage_package(plan, tmp_path / "stage")

    assert (staging / "manifest.json").is_file()
    assert (staging / "scores.json").is_file()
    assert (staging / "predictions" / "GOOG_2026-01.npy").is_file()

    manifest = json.loads((staging / "manifest.json").read_text())
    assert manifest["run_name"] == "smoke"
    assert manifest["metrics"] == ["l1", "wasserstein", "ks"]
    assert "scores.json" in manifest["contents"]
    assert "predictions/GOOG_2026-01.npy" in manifest["contents"]
    assert "manifest.json" not in manifest["contents"]
