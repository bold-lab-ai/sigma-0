from __future__ import annotations

import json
from pathlib import Path

import pytest

from run.benchmarking.validate_model_zoo_evaluation import (
    validate_checkpoint,
    validate_inference,
)


def _checkpoint(tmp_path: Path, *, latest: int = 30_000) -> Path:
    root = tmp_path / "run"
    step = root / str(latest)
    (step / "metadata").mkdir(parents=True)
    (step / "state").mkdir()
    (step / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
    (step / "state" / "_METADATA").write_text("{}", encoding="utf-8")
    (step / "state" / "manifest.ocdbt").write_text("x", encoding="utf-8")
    payload = {
        "config": {
            "architecture": "mamba3",
            "ssm_type": "mamba3",
            "jax_seed": 5,
            "token_mode": "26tok",
            "n_layers": 6,
            "d_model": 256,
            "msg_seq_len": 500,
            "book_transform": True,
            "book_depth": 500,
            "max_train_batches": 30_000,
            "curtail_epochs": 29_999,
            "micro_bsz": 8,
            "hierarchical": True,
            "local_steps_k": 10,
            "tickers": ["GOOG"],
            "train_date_range": ["2022-01-01", "2025-11-30"],
        }
    }
    (step / "metadata" / "metadata").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_checkpoint_gate_requires_exact_latest_30k_and_identity(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)
    result = validate_checkpoint(root, "mamba3", 5, 30_000)
    assert result["checkpoint_step"] == 30_000
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_checkpoint(root, "mamba3", 42, 30_000)

    (root / "31000").mkdir()
    with pytest.raises(ValueError, match="latest checkpoint is 31000"):
        validate_checkpoint(root, "mamba3", 5, 30_000)


def _write_csv(path: Path, rows: int) -> None:
    path.write_text("1,2\n" * rows, encoding="utf-8")


def test_inference_gate_proves_unique_common_windows(tmp_path: Path) -> None:
    root = tmp_path / "inference"
    for directory in ("data_gen", "data_real", "data_cond"):
        target = root / directory
        target.mkdir(parents=True)
        for index in range(4):
            _write_csv(target / f"GOOG_message_id_{index}.csv", 3)
            _write_csv(
                target / f"GOOG_orderbook_id_{index}.csv",
                4 if directory == "data_cond" else 3,
            )

    indices = [11, 7, 19, 3]
    digest = __import__("hashlib").sha256(
        json.dumps(indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for rank in range(2):
        payload = {
            "rank": rank,
            "world_size": 2,
            "dataset_length": 100,
            "all_indices_sha256": digest,
            "all_indices": indices,
            "rank_indices": indices[rank::2],
        }
        (root / f"sample_indices_rank{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    canonical = tmp_path / "GOOG_4.txt"
    canonical.write_text("\n".join(map(str, indices)) + "\n", encoding="utf-8")
    result = validate_inference(
        root,
        expected=4,
        rows=3,
        world_size=2,
        expected_indices_file=canonical,
        expected_dataset_length=100,
    )
    assert result["sample_indices_sha256"] == digest
    assert result["sample_indices_file"] == str(canonical.resolve())
    assert result["inventory"]["data_gen"]["messages"] == 4
    assert result["inventory"]["data_cond"]["book_rows_per_file"] == 4

    canonical.write_text("11\n7\n19\n4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_inference(
            root,
            expected=4,
            rows=3,
            world_size=2,
            expected_indices_file=canonical,
            expected_dataset_length=100,
        )
