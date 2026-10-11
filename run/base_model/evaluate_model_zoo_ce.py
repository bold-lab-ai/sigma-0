#!/usr/bin/env python3
"""Evaluate model-zoo checkpoints on a fixed held-out CE sample.

Checkpoint selection is intentionally based only on December 2025 teacher-
forced CE. January 2026 LOBBench data is never opened by this program.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import jax

from base_model.training.checkpoint_selection import convergence_step, select_checkpoints
from base_model.data.dataloading import create_lobster_prediction_dataset
from base_model.training.state import init_train_state, load_checkpoint
from base_model.data.lobster_dataloader import LOBSTER_Dataset
from base_model.sharding.sharding_utils import create_state_shardings, initialize_mesh
from base_model.training.train_helpers import create_jit_eval_step, validate


def _checkpoint_steps(root: Path) -> list[int]:
    steps = sorted(
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
        and (path / "state").is_dir()
        and (path / "metadata" / "metadata").is_file()
    )
    if not steps:
        raise ValueError(f"no complete checkpoint steps found under {root}")
    return steps


def _metadata(root: Path, step: int) -> dict[str, Any]:
    with (root / str(step) / "metadata" / "metadata").open() as handle:
        value = json.load(handle)
    if "config" not in value:
        raise ValueError(f"checkpoint step {step} has no config metadata")
    return value


def _json_scalar_default(value: Any) -> Any:
    """Convert NumPy/JAX scalar values while rejecting non-scalar objects."""
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return scalar
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(
            payload, handle, indent=2, sort_keys=True,
            default=_json_scalar_default,
        )
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-architecture")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tickers", default="GOOG")
    parser.add_argument("--date-range", default="2025-12-01,2025-12-31")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--targets", default="0.70,0.65,0.60")
    parser.add_argument("--target-tolerance", type=float, default=0.01)
    parser.add_argument("--steps", default="all", help="all or comma-separated checkpoint steps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_batches < 1:
        parser.error("batch-size and max-batches must be positive")
    date_parts = tuple(part.strip() for part in args.date_range.split(","))
    if len(date_parts) != 2:
        parser.error("date-range must be START,END")
    tickers = tuple(part.strip() for part in args.tickers.split(",") if part.strip())
    targets = [float(part) for part in args.targets.split(",") if part.strip()]

    available_steps = _checkpoint_steps(args.checkpoint_root)
    if args.steps == "all":
        steps = available_steps
    else:
        requested = [int(part) for part in args.steps.split(",") if part.strip()]
        missing = sorted(set(requested) - set(available_steps))
        if missing:
            raise ValueError(f"requested incomplete/missing checkpoint steps: {missing}")
        steps = sorted(set(requested))

    first_metadata = _metadata(args.checkpoint_root, steps[0])
    config = dict(first_metadata["config"])
    if config.get("token_mode") != "26tok":
        raise ValueError(
            f"checkpoint token_mode must be '26tok', got {config.get('token_mode')!r}"
        )
    if (args.expected_architecture is not None
            and config.get("architecture") != args.expected_architecture):
        raise ValueError(
            "checkpoint architecture mismatch: "
            f"expected {args.expected_architecture!r}, got {config.get('architecture')!r}"
        )
    if (args.expected_seed is not None
            and int(config.get("jax_seed")) != args.expected_seed):
        raise ValueError(
            "checkpoint seed mismatch: "
            f"expected {args.expected_seed}, got {config.get('jax_seed')!r}"
        )
    for step in steps[1:]:
        other = _metadata(args.checkpoint_root, step)["config"]
        for key in ("architecture", "d_model", "n_layers", "token_mode", "jax_seed"):
            if other.get(key) != config.get(key):
                raise ValueError(f"checkpoint config drift at step {step}: {key}")

    model_args = Namespace(**config)
    model_args.USE_WANDB = False
    model_args.micro_bsz = args.batch_size
    model_args.num_devices = 1
    model_args.is_distributed = False
    model_args.process_index = 0
    model_args.process_count = 1
    model_args.local_steps_k = 0
    model_args.hierarchical = False
    model_args.restore = None
    model_args.restore_step = None

    dataset_result = create_lobster_prediction_dataset(
        args.data_root,
        seed=42,
        mask_fn=LOBSTER_Dataset.no_mask,
        msg_seq_len=int(model_args.msg_seq_len),
        micro_bsz=args.batch_size,
        num_devices=1,
        use_book_data=bool(model_args.use_book_data),
        use_simple_book=bool(model_args.use_simple_book),
        book_transform=bool(model_args.book_transform),
        book_ablation="real",
        book_depth=int(model_args.book_depth),
        n_data_workers=0,
        shuffle_train=False,
        rand_offset=False,
        val_split=0.0,
        test_split=0.0,
        tickers=list(tickers),
        data_root=args.data_root,
        train_date_range=date_parts,
        test_date_range=None,
        token_mode="26tok",
        uniform_ticker_batches=True,
        max_train_batches=args.max_batches,
    )
    (_, validation_loader, _, _, _, n_classes, seq_len, _, book_seq_len,
     book_dim, validation_size) = dataset_result
    requested_sequences = args.batch_size * args.max_batches
    if validation_size < requested_sequences:
        raise ValueError(
            f"validation dataset has {validation_size} sequences, fewer than "
            f"the requested fixed sample of {requested_sequences}"
        )

    # Materialise once. Re-iterating a seeded RandomSampler advances its RNG,
    # which would otherwise give every checkpoint a different held-out sample.
    # Keeping these raw batches in host RAM also avoids rereading SquashFS for
    # every checkpoint.
    validation_batches = list(
        itertools.islice(validation_loader, args.max_batches)
    )
    if len(validation_batches) != args.max_batches:
        raise RuntimeError(
            f"validation loader yielded {len(validation_batches)} batches, "
            f"expected {args.max_batches}"
        )

    initial_state, model_cls = init_train_state(
        model_args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        train_size=validation_size,
        print_shapes=True,
    )
    validation_model = model_cls(training=False, step_rescale=1)
    mesh = initialize_mesh(1, hierarchical=False, tp_size=1)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "architecture": config.get("architecture"),
        "seed": config.get("jax_seed"),
        "token_mode": "26tok",
        "validation": {
            "tickers": list(tickers),
            "date_range": list(date_parts),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "sampling_seed": 42,
            "sampling": "uniform_ticker_batches_without_replacement",
            "sequences": requested_sequences,
            "token_values": requested_sequences * seq_len,
            "dataset_size": validation_size,
        },
        "targets": targets,
        "target_tolerance": args.target_tolerance,
        "results": [],
        "selection": {"best_validation": None, "iso_ce": {}},
        "convergence_step": None,
    }

    for step in steps:
        restored = load_checkpoint(
            initial_state,
            str(args.checkpoint_root),
            step=step,
            train=False,
        )
        state = restored["model"]
        state = jax.device_put(state, create_state_shardings(state, mesh))
        jit_eval_step = create_jit_eval_step(
            mesh, state, has_book_data=bool(model_args.use_book_data)
        )
        metrics = validate(
            state,
            validation_model.apply,
            validation_batches,
            seq_len,
            1,
            bool(model_args.batchnorm),
            1,
            0,
            curtail_epoch=args.max_batches - 1,
            ignore_times=bool(model_args.ignore_times),
            mesh=mesh,
            jit_eval_step_fn=jit_eval_step,
            silent=True,
        )
        validation_ce, validation_accuracy = float(metrics[0]), float(metrics[1])
        row = {
            "step": step,
            "validation_ce": validation_ce,
            "validation_accuracy": validation_accuracy,
        }
        payload["results"].append(row)
        payload["selection"] = select_checkpoints(
            payload["results"], targets, args.target_tolerance
        )
        payload["convergence_step"] = convergence_step(payload["results"])
        _write_json(args.output, payload)
        print(
            f"[ce-eval] step={step} validation_ce={validation_ce:.8f} "
            f"accuracy={validation_accuracy:.8f}"
        )

    print(f"[ce-eval] wrote {args.output}")


if __name__ == "__main__":
    main()
