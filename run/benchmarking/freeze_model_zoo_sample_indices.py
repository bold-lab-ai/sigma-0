#!/usr/bin/env python3
"""Freeze uniform LOBBench dataset indices from a SquashFS shape index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def build_payload(
    source: Path,
    ticker: str,
    sequence_messages: int,
    n_samples: int,
    seed: int,
) -> tuple[list[int], dict[str, object]]:
    source_bytes = source.read_bytes()
    shard_index = json.loads(source_bytes)
    shapes = shard_index["shapes"]
    prefix = f"{ticker}/"
    message_keys = sorted(
        key for key in shapes if key.startswith(prefix) and "_message_" in key
    )
    if not message_keys:
        raise ValueError(f"no {ticker} message arrays in {source}")

    file_rows: list[dict[str, object]] = []
    total_sequences = 0
    for message_key in message_keys:
        book_key = message_key.replace("_message_", "_orderbook_", 1)
        if book_key not in shapes:
            raise ValueError(f"missing paired order book for {message_key}")
        message_rows = int(shapes[message_key]["shape"][0])
        book_rows = int(shapes[book_key]["shape"][0])
        # LOBSTER_Dataset(inference=True) needs the post-message book row.
        usable_rows = min(message_rows, max(book_rows - 1, 0))
        sequences = usable_rows // sequence_messages
        total_sequences += sequences
        file_rows.append(
            {
                "message_file": message_key,
                "message_rows": message_rows,
                "book_rows": book_rows,
                "usable_rows": usable_rows,
                "sequences": sequences,
            }
        )

    if n_samples > total_sequences:
        raise ValueError(
            f"requested {n_samples} samples from only {total_sequences} sequences"
        )
    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(total_sequences, size=n_samples, replace=False)
    ).astype(np.int64).tolist()
    payload: dict[str, object] = {
        "version": 1,
        "ticker": ticker,
        "period": shard_index.get("shard"),
        "test_split": 1.0,
        "conditioning_messages": 250,
        "generated_messages": 250,
        "sequence_messages": sequence_messages,
        "n_samples": n_samples,
        "selection_algorithm": "numpy.random.default_rng(seed).choice(replace=False), sorted",
        "selection_seed": seed,
        "dataset_length": total_sequences,
        "source_shape_index": str(source.resolve()),
        "source_shape_index_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "files": file_rows,
    }
    return indices, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticker", default="GOOG")
    parser.add_argument("--sequence-messages", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=3136)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    indices, payload = build_payload(
        args.source,
        args.ticker,
        args.sequence_messages,
        args.n_samples,
        args.seed,
    )
    index_text = "".join(f"{index}\n" for index in indices)
    payload["sample_indices_sha256"] = hashlib.sha256(
        index_text.encode("utf-8")
    ).hexdigest()
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.check:
        if args.output.read_text(encoding="utf-8") != index_text:
            raise SystemExit(f"frozen indices differ: {args.output}")
        if metadata_path.read_text(encoding="utf-8") != metadata_text:
            raise SystemExit(f"frozen metadata differs: {metadata_path}")
        print(f"verified {len(indices)} indices from {payload['dataset_length']} windows")
        return

    args.output.write_text(index_text, encoding="utf-8")
    metadata_path.write_text(metadata_text, encoding="utf-8")
    print(f"wrote {len(indices)} indices from {payload['dataset_length']} windows")


if __name__ == "__main__":
    main()
