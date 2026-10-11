#!/usr/bin/env python3
"""Inspect and lightly validate an Orbax base_model checkpoint."""

from __future__ import annotations

import argparse
import json

from base_model.training import sample_checkpoint_arrays, summarize_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Orbax checkpoint manager root or concrete step directory")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step when passing a manager root")
    parser.add_argument(
        "--sample-arrays",
        type=int,
        default=0,
        help="open and read this many parameter arrays with TensorStore",
    )
    parser.add_argument(
        "--no-verify-shapes",
        dest="verify_shapes",
        action="store_false",
        default=True,
        help="skip TensorStore shape verification; trust the on-disk write_shape "
        "(faster, but under-reports element counts for sharded checkpoints)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text lines")
    args = parser.parse_args()

    summary = summarize_checkpoint(args.checkpoint, step=args.step, verify_shapes=args.verify_shapes)
    samples = sample_checkpoint_arrays(
        args.checkpoint,
        step=args.step,
        limit=args.sample_arrays,
    )

    if args.json:
        payload = {
            **summary,
            "sampled_arrays": [
                {
                    "name": item.name,
                    "metadata_shape": list(item.metadata_shape),
                    "tensor_shape": list(item.tensor_shape),
                    "dtype": item.dtype,
                }
                for item in samples
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"checkpoint={summary['manager_path']} step={summary['step']}")
    print(f"step_path={summary['step_path']}")
    print(
        "config="
        f"ssm_type={summary['model']} "
        f"d_model={summary['d_model']} "
        f"n_layers={summary['n_layers']} "
        f"msg_seq_len={summary['msg_seq_len']} "
        f"use_book_data={summary['use_book_data']} "
        f"opt_config={summary['opt_config']}"
    )
    print(f"leaf_counts={summary['leaf_counts']}")
    print(f"metadata_elements={summary['metadata_elements']}")
    for item in samples:
        print(
            "sample_array="
            f"{item.name} "
            f"metadata_shape={item.metadata_shape} "
            f"tensor_shape={item.tensor_shape} "
            f"dtype={item.dtype}"
        )


if __name__ == "__main__":
    main()
