#!/usr/bin/env python3
"""Validate inference artifacts with separate conditioning/generated row counts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_runtime_validator(path: Path) -> ModuleType:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"runtime validator is absent: {path}")
    spec = importlib.util.spec_from_file_location(
        "sigma0_runtime_model_zoo_validator",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load runtime validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(args: argparse.Namespace) -> dict[str, Any]:
    runtime = load_runtime_validator(args.runtime_validator)
    original_inventory = runtime._csv_inventory
    conditioning_book_rows = (
        args.conditioning_book_rows
        if args.conditioning_book_rows is not None
        else args.conditioning_rows + 1
    )

    def inventory(directory: Path, expected: int, rows: int) -> dict[str, int]:
        if directory.name != "data_cond":
            return original_inventory(directory, expected, rows)

        message_files = sorted(directory.glob("*message*.csv"))
        book_files = sorted(directory.glob("*orderbook*.csv"))
        if len(message_files) != expected or len(book_files) != expected:
            raise ValueError(
                f"{directory}: expected {expected} message/book files; "
                f"found {len(message_files)}/{len(book_files)}"
            )
        empty = [
            str(path)
            for path in (*message_files, *book_files)
            if path.stat().st_size == 0
        ]
        if empty:
            raise ValueError(f"empty inference files: {empty[:5]}")

        bad_rows: list[tuple[str, int, int]] = []
        for paths, expected_rows in (
            (message_files, args.conditioning_rows),
            (book_files, conditioning_book_rows),
        ):
            for path in paths:
                with path.open("rb") as handle:
                    count = sum(1 for _ in handle)
                if count != expected_rows:
                    bad_rows.append((str(path), count, expected_rows))
                    if len(bad_rows) == 5:
                        break
            if len(bad_rows) == 5:
                break
        if bad_rows:
            raise ValueError(
                "unexpected conditioning row counts "
                f"(path, actual, expected): {bad_rows}"
            )
        return {
            "messages": len(message_files),
            "books": len(book_files),
            "message_rows_per_file": args.conditioning_rows,
            "book_rows_per_file": conditioning_book_rows,
        }

    runtime._csv_inventory = inventory
    result = runtime.validate_inference(
        args.root,
        args.expected_sequences,
        args.generated_rows,
        args.world_size,
        args.expected_indices_file,
        args.expected_dataset_length,
    )
    result["generated_rows_per_sequence"] = args.generated_rows
    result["conditioning_rows_per_sequence"] = args.conditioning_rows
    result["conditioning_book_rows_per_sequence"] = conditioning_book_rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-validator", type=Path, required=True)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-sequences", type=int, required=True)
    parser.add_argument("--generated-rows", type=int, required=True)
    parser.add_argument("--conditioning-rows", type=int, required=True)
    parser.add_argument("--conditioning-book-rows", type=int)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--expected-indices-file", type=Path)
    parser.add_argument("--expected-dataset-length", type=int)
    args = parser.parse_args()
    print(json.dumps(validate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
