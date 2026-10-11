#!/usr/bin/env python3
"""Strict gates for fixed-step model-zoo inference and LOBBench artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_WIDTHS = {
    "mamba3": 256,
    "s5": 512,
    "gdn": 256,
    "kda": 256,
    "transformer": 224,
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_checkpoint(
    root: Path,
    architecture: str,
    seed: int,
    step: int,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    _require(root.is_dir(), f"checkpoint manager root is absent: {root}")
    numeric_steps = sorted(
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    _require(numeric_steps, f"no numeric checkpoints under {root}")
    _require(
        numeric_steps[-1] == step,
        f"latest checkpoint is {numeric_steps[-1]}, expected exactly {step}: {root}",
    )

    step_dir = root / str(step)
    required = (
        step_dir / "_CHECKPOINT_METADATA",
        step_dir / "metadata" / "metadata",
        step_dir / "state" / "_METADATA",
        step_dir / "state" / "manifest.ocdbt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    _require(not missing, f"incomplete checkpoint {step_dir}; missing={missing}")

    payload = _load_json(step_dir / "metadata" / "metadata")
    config = payload.get("config", {})
    _require(isinstance(config, dict), f"checkpoint metadata has no config: {step_dir}")
    actual_architecture = config.get("architecture") or config.get("ssm_type")
    expected = {
        "architecture": architecture,
        "jax_seed": seed,
        "token_mode": "26tok",
        "n_layers": 6,
        "d_model": MODEL_WIDTHS[architecture],
        "msg_seq_len": 500,
        "book_transform": True,
        "book_depth": 500,
        "max_train_batches": 30000,
        "curtail_epochs": 29999,
        "micro_bsz": 8,
        "hierarchical": True,
        "local_steps_k": 10,
    }
    actual = {**config, "architecture": actual_architecture}
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    _require(not mismatches, f"checkpoint contract mismatch: {mismatches}")
    _require(config.get("tickers") == ["GOOG"], f"unexpected tickers: {config.get('tickers')}")
    _require(
        config.get("train_date_range") == ["2022-01-01", "2025-11-30"],
        f"unexpected training period: {config.get('train_date_range')}",
    )

    return {
        "architecture": architecture,
        "seed": seed,
        "checkpoint_step": step,
        "checkpoint_path": str(root),
        "metadata_path": str(step_dir / "metadata" / "metadata"),
        "checkpoint_metadata_sha256": hashlib.sha256(
            (step_dir / "metadata" / "metadata").read_bytes()
        ).hexdigest(),
    }


def _csv_inventory(
    directory: Path,
    expected: int,
    message_rows: int,
    book_rows: int | None = None,
) -> dict[str, int]:
    _require(directory.is_dir(), f"missing inference directory: {directory}")
    if book_rows is None:
        book_rows = message_rows
    message_files = sorted(directory.glob("*message*.csv"))
    book_files = sorted(directory.glob("*orderbook*.csv"))
    _require(
        len(message_files) == expected and len(book_files) == expected,
        f"{directory}: expected {expected} message/book files; "
        f"found {len(message_files)}/{len(book_files)}",
    )
    empty = [str(path) for path in (*message_files, *book_files) if path.stat().st_size == 0]
    _require(not empty, f"empty inference files: {empty[:5]}")

    bad_rows: list[tuple[str, int, int]] = []
    for files, expected_rows in (
        (message_files, message_rows),
        (book_files, book_rows),
    ):
        for path in files:
            with path.open("rb") as handle:
                count = sum(1 for _ in handle)
            if count != expected_rows:
                bad_rows.append((str(path), count, expected_rows))
                if len(bad_rows) == 5:
                    break
        if len(bad_rows) == 5:
            break
    _require(
        not bad_rows,
        f"unexpected inference row counts (path, actual, expected): {bad_rows}",
    )
    return {
        "messages": len(message_files),
        "books": len(book_files),
        "message_rows_per_file": message_rows,
        "book_rows_per_file": book_rows,
    }


def validate_inference(
    root: Path,
    expected: int,
    rows: int,
    world_size: int,
    expected_indices_file: Path | None = None,
    expected_dataset_length: int | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    inventories = {
        "data_gen": _csv_inventory(root / "data_gen", expected, rows),
        "data_real": _csv_inventory(root / "data_real", expected, rows),
        # Conditioning books include the state before the first conditioning
        # message, hence N messages correspond to N+1 book states.
        "data_cond": _csv_inventory(
            root / "data_cond", expected, rows, book_rows=rows + 1
        ),
    }

    manifests = [root / f"sample_indices_rank{rank}.json" for rank in range(world_size)]
    missing = [str(path) for path in manifests if not path.is_file()]
    _require(not missing, f"missing sample-index manifests: {missing}")
    payloads = [_load_json(path) for path in manifests]
    hashes = {payload.get("all_indices_sha256") for payload in payloads}
    _require(len(hashes) == 1, f"sample-window drift across ranks: {sorted(hashes)}")
    global_lists = {tuple(payload.get("all_indices", ())) for payload in payloads}
    _require(len(global_lists) == 1, "sample-index lists differ despite their recorded hashes")
    all_indices = next(iter(global_lists))
    _require(len(all_indices) == expected, f"expected {expected} global indices, got {len(all_indices)}")
    _require(len(set(all_indices)) == expected, "evaluation indices contain duplicates")

    expected_indices_file_sha256 = None
    if expected_indices_file is not None:
        expected_indices_file = expected_indices_file.expanduser().resolve()
        _require(
            expected_indices_file.is_file(),
            f"canonical sample-index file is absent: {expected_indices_file}",
        )
        try:
            expected_indices = tuple(
                int(line.strip())
                for line in expected_indices_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot read canonical sample indices from {expected_indices_file}: {exc}"
            ) from exc
        _require(
            all_indices == expected_indices,
            "generated sample-index list does not exactly match the canonical LOBBench file",
        )
        expected_indices_file_sha256 = hashlib.sha256(
            expected_indices_file.read_bytes()
        ).hexdigest()

    assigned: list[int] = []
    for rank, payload in enumerate(payloads):
        _require(payload.get("rank") == rank, f"rank manifest identity mismatch: {manifests[rank]}")
        _require(payload.get("world_size") == world_size, f"world-size mismatch: {manifests[rank]}")
        if expected_dataset_length is not None:
            _require(
                payload.get("dataset_length") == expected_dataset_length,
                f"dataset-length mismatch: {manifests[rank]}",
            )
        rank_indices = payload.get("rank_indices", [])
        _require(
            rank_indices == list(all_indices[rank::world_size]),
            f"rank {rank} is not the expected interleaved partition",
        )
        assigned.extend(rank_indices)
    _require(len(assigned) == expected and set(assigned) == set(all_indices), "rank partition is incomplete")

    return {
        "inference_dir": str(root),
        "sequences": expected,
        "rows_per_sequence": rows,
        "world_size": world_size,
        "dataset_length": payloads[0].get("dataset_length"),
        "sample_indices_sha256": next(iter(hashes)),
        "sample_indices_file": (
            str(expected_indices_file) if expected_indices_file is not None else None
        ),
        "sample_indices_file_sha256": expected_indices_file_sha256,
        "inventory": inventories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("root", type=Path)
    checkpoint.add_argument("--architecture", choices=tuple(MODEL_WIDTHS), required=True)
    checkpoint.add_argument("--seed", type=int, required=True)
    checkpoint.add_argument("--step", type=int, default=30000)

    inference = subparsers.add_parser("inference")
    inference.add_argument("root", type=Path)
    inference.add_argument("--expected-sequences", type=int, required=True)
    inference.add_argument("--rows", type=int, required=True)
    inference.add_argument("--world-size", type=int, required=True)
    inference.add_argument("--expected-indices-file", type=Path)
    inference.add_argument("--expected-dataset-length", type=int)

    args = parser.parse_args()
    if args.command == "checkpoint":
        result = validate_checkpoint(args.root, args.architecture, args.seed, args.step)
    else:
        result = validate_inference(
            args.root,
            args.expected_sequences,
            args.rows,
            args.world_size,
            args.expected_indices_file,
            args.expected_dataset_length,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
