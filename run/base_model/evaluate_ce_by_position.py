#!/usr/bin/env python3
"""Teacher-forced per-position / per-field CE for one checkpoint on one slice.

The 26-position mean hides which positions still carry loss. This program
decomposes it against two free reference lines per position:

  unigram   the corpus's own marginal entropy at that position -- model CE at
            the unigram line means no conditional structure was learned there;
  ceiling   ln(#syntactically valid tokens) from the encoding's validity
            matrix -- the largest CE the position can have under the syntax.

Run it twice per checkpoint (a training-period slice and a held-out slice) and
diff the JSONs: per-position train ~ held-out everywhere reads as "at the
wall", a gap concentrated in a few fields reads as "the aggregate is diluted".

Follows evaluate_model_zoo_ce.py: the slice is built by pointing the training
loader at a date range, deterministically (no shuffle, no random offsets), and
batches are materialised once so the sample is fixed. Never opens the
January-2026 LOBBench period.
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
import numpy as onp

from base_model.data.dataloading import create_lobster_prediction_dataset
from base_model.encoding.encoding import Message_Tokenizer, set_token_mode
from base_model.training.state import init_train_state, load_checkpoint
from base_model.data.lobster_dataloader import LOBSTER_Dataset
from base_model.sharding.sharding_utils import create_state_shardings, initialize_mesh
from base_model.training.train_helpers import (TIME_END_I, TIME_START_I, create_jit_eval_step,
                               validate)
from base_model.inference.validation import syntax_validation_matrix


def _metadata(root: Path, step: int) -> dict[str, Any]:
    with (root / str(step) / "metadata" / "metadata").open() as handle:
        value = json.load(handle)
    if "config" not in value:
        raise ValueError(f"checkpoint step {step} has no config metadata")
    return value


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _active_positions(tpm: int) -> list[int]:
    full = int(Message_Tokenizer.MSG_LEN)
    if tpm == full:
        return list(range(full))
    dropped = TIME_END_I - TIME_START_I + 1
    if tpm != full - dropped:
        raise ValueError(f"cannot map {tpm} positions onto the {full}-token layout")
    return [p for p in range(full) if not (TIME_START_I <= p <= TIME_END_I)]


def _field_of_position() -> list[str]:
    """Position index -> field name, derived from the encoding constants."""
    names: list[str] = []
    for field, width in zip(Message_Tokenizer.FIELDS, Message_Tokenizer.TOK_LENS):
        names.extend([field] * int(width))
    if len(names) != int(Message_Tokenizer.MSG_LEN):
        raise ValueError("TOK_LENS no longer sums to MSG_LEN")
    return names


def _unigram_entropy(labels: onp.ndarray, tpm: int, n_classes: int) -> onp.ndarray:
    """Per-position marginal entropy (nats) of the evaluated sample."""
    per_msg = labels.reshape(-1, tpm)
    out = onp.zeros(tpm, dtype=onp.float64)
    for pos in range(tpm):
        counts = onp.bincount(per_msg[:, pos], minlength=n_classes).astype(onp.float64)
        total = counts.sum()
        probs = counts[counts > 0] / total
        out[pos] = float(-(probs * onp.log(probs)).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", default="latest",
                        help="checkpoint step to score, or 'latest'")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tickers", default="GOOG")
    parser.add_argument("--date-range", required=True, help="START,END")
    parser.add_argument("--slice-label", required=True,
                        help="how this slice should be named in the JSON, "
                             "e.g. train_period or heldout")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    date_parts = tuple(part.strip() for part in args.date_range.split(","))
    if len(date_parts) != 2:
        parser.error("date-range must be START,END")
    tickers = tuple(part.strip() for part in args.tickers.split(",") if part.strip())

    available = _checkpoint_steps(args.checkpoint_root)
    step = available[-1] if args.step == "latest" else int(args.step)
    if step not in available:
        raise ValueError(f"step {step} not in {available}")

    config = dict(_metadata(args.checkpoint_root, step)["config"])
    token_mode = config.get("token_mode")
    if token_mode is None:
        # Pre-registry checkpoints did not record token_mode; the training
        # era only supported 26tok (init_train refuses anything else).
        print("[warn] checkpoint config has no token_mode; assuming 26tok")
        config["token_mode"] = "26tok"
    else:
        # base_model.encoding.encoding is a runtime selector now, so a checkpoint's own token_mode
        # is honoured instead of refused. This is process-wide and must happen
        # before the dataset is built, since the dataloader tokenises at load time.
        set_token_mode(token_mode)
        print(f"[token-mode] {token_mode} selected from the checkpoint config")

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
        token_mode=config["token_mode"],
        uniform_ticker_batches=True,
        max_train_batches=args.max_batches,
    )
    (_, slice_loader, _, _, _, n_classes, seq_len, _, book_seq_len,
     book_dim, slice_size) = dataset_result
    requested = args.batch_size * args.max_batches
    if slice_size < requested:
        raise ValueError(f"slice has {slice_size} sequences, fewer than {requested}")

    # Materialise once: a fixed sample, and no SquashFS rereads.
    batches = list(itertools.islice(slice_loader, args.max_batches))
    if len(batches) != args.max_batches:
        raise RuntimeError(f"loader yielded {len(batches)}, wanted {args.max_batches}")

    initial_state, model_cls = init_train_state(
        model_args, n_classes=n_classes, seq_len=seq_len, book_dim=book_dim,
        book_seq_len=book_seq_len, train_size=slice_size, print_shapes=False,
    )
    eval_model = model_cls(training=False, step_rescale=1)
    mesh = initialize_mesh(1, hierarchical=False, tp_size=1)

    restored = load_checkpoint(initial_state, str(args.checkpoint_root),
                               step=step, train=False)
    state = restored["model"]
    state = jax.device_put(state, create_state_shardings(state, mesh))
    jit_eval_step = create_jit_eval_step(
        mesh, state, has_book_data=bool(model_args.use_book_data))

    ignore_times = bool(model_args.ignore_times)
    metrics = validate(
        state, eval_model.apply, batches, seq_len, 1,
        bool(model_args.batchnorm), 1, 0,
        curtail_epoch=args.max_batches - 1,
        ignore_times=ignore_times,
        log_ce_tables=True,
        mesh=mesh,
        jit_eval_step_fn=jit_eval_step,
        silent=True,
    )
    aggregate_ce, aggregate_acc = float(metrics[0]), float(metrics[1])
    ce_means, acc_means = onp.asarray(metrics[2]), onp.asarray(metrics[3])
    tpm = int(ce_means.shape[0])
    positions = _active_positions(tpm)
    field_names = _field_of_position()

    # References. The syntax matrix covers all 26 positions; the unigram line
    # is computed on the same active positions the CE was computed on.
    valid = onp.asarray(syntax_validation_matrix())
    ceiling_all = onp.log(valid.sum(axis=1).clip(min=1)).astype(onp.float64)
    labels = onp.concatenate([onp.asarray(b[1]) for b in batches], axis=0)
    labels_msgs = labels.reshape(labels.shape[0], -1, int(Message_Tokenizer.MSG_LEN))
    if tpm != int(Message_Tokenizer.MSG_LEN):
        labels_msgs = labels_msgs[:, :, positions]
    unigram = _unigram_entropy(labels_msgs.reshape(-1, tpm), tpm, int(n_classes))

    per_position = []
    for i, pos in enumerate(positions):
        per_position.append({
            "position": pos,
            "field": field_names[pos],
            "ce": float(ce_means[i]),
            "acc": float(acc_means[i]),
            "unigram_nats": float(unigram[i]),
            "ceiling_nats": float(ceiling_all[pos]),
        })

    total_ce = float(ce_means.sum())
    per_field: list[dict[str, Any]] = []
    for field in dict.fromkeys(field_names):  # order-preserving unique
        rows = [r for r in per_position if r["field"] == field]
        if not rows:
            continue  # both time fields drop out under ignore_times
        per_field.append({
            "field": field,
            "width": len(rows),
            "ce": float(onp.mean([r["ce"] for r in rows])),
            "acc": float(onp.mean([r["acc"] for r in rows])),
            "unigram_nats": float(onp.mean([r["unigram_nats"] for r in rows])),
            "ceiling_nats": float(onp.mean([r["ceiling_nats"] for r in rows])),
            "ce_share_pct": float(100.0 * onp.sum([r["ce"] for r in rows]) / total_ce)
            if total_ce > 0 else 0.0,
        })

    payload = {
        "schema_version": 1,
        "checkpoint": {
            "root": str(args.checkpoint_root.resolve()),
            "step": step,
            "architecture": config.get("architecture"),
            "d_model": config.get("d_model"),
            "n_layers": config.get("n_layers"),
            "seed": config.get("jax_seed"),
            "ignore_times": ignore_times,
            # Load-bearing for message-level perplexity: a cross-tokenization
            # table is only checkable if each record says which tokenizer scored
            # it. Without this the widths in per_field are the only clue.
            "token_mode": config["token_mode"],
            "tokens_per_message": int(Message_Tokenizer.MSG_LEN),
        },
        "slice": {
            "label": args.slice_label,
            "tickers": list(tickers),
            "date_range": list(date_parts),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "sequences": requested,
            "messages": requested * int(model_args.msg_seq_len),
        },
        "aggregate": {
            "ce": aggregate_ce,
            "acc": aggregate_acc,
            "ce_per_message_nats": total_ce,
        },
        "per_position": per_position,
        "per_field": per_field,
    }
    _write_json(args.output, payload)
    print(f"[ce-by-position] wrote {args.output}")
    print(f"  aggregate ce={aggregate_ce:.5f} acc={aggregate_acc:.4f} "
          f"({tpm} positions, slice={args.slice_label})")
    top = sorted(per_field, key=lambda r: -r["ce_share_pct"])[:5]
    for row in top:
        print(f"  {row['field']:<12} ce={row['ce']:.4f} "
              f"share={row['ce_share_pct']:.1f}% unigram={row['unigram_nats']:.4f} "
              f"ceiling={row['ceiling_nats']:.4f}")


if __name__ == "__main__":
    main()
