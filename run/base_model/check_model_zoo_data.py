#!/usr/bin/env python3
"""Validate the exact model-zoo training corpus and pull one real batch."""

from __future__ import annotations

import argparse

from base_model.data.dataloading import (
    UniformTickerDistributedSampler,
    create_lobster_prediction_dataset,
)
from base_model.data.lobster_dataloader import LOBSTER_Dataset
from base_model.training.train_helpers import prep_batch


PAPER_TICKERS = ("GOOG",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-11-30")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tickers", default=",".join(PAPER_TICKERS))
    args = parser.parse_args()
    tickers = tuple(part.strip() for part in args.tickers.split(",") if part.strip())

    result = create_lobster_prediction_dataset(
        args.data_root,
        seed=42,
        mask_fn=LOBSTER_Dataset.no_mask,
        msg_seq_len=500,
        micro_bsz=args.batch_size,
        # Reproduce one production rank: 8 sequences/GPU x 4 local GPUs,
        # participating in a four-rank (four-node) global batch of 128.
        num_devices=4,
        use_book_data=True,
        use_simple_book=False,
        book_transform=True,
        n_data_workers=0,
        shuffle_train=False,
        rand_offset=False,
        val_split=0.0,
        test_split=0.0,
        use_distributed_sampler=True,
        process_rank=0,
        process_count=4,
        uniform_ticker_batches=True,
        max_train_batches=30_000,
        tickers=list(tickers),
        data_root=args.data_root,
        train_date_range=(args.start, args.end),
        test_date_range=None,
        token_mode="26tok",
    )
    dataset, trainloader, _, _, _, n_classes, seq_len, _, _, book_dim, train_size = result

    assert n_classes == 2112, n_classes
    assert seq_len == 13_000, seq_len
    assert book_dim == 503, book_dim
    assert train_size > 0, train_size
    assert set(dataset.tickers) == set(tickers), dataset.tickers

    sampler = trainloader.sampler
    assert isinstance(sampler, UniformTickerDistributedSampler), type(sampler)
    assert sampler.per_process_bsz == args.batch_size * 4, sampler.per_process_bsz
    assert sampler.global_bsz == 128, sampler.global_bsz
    ticker_counts = sampler.ticker_sequence_counts
    assert min(ticker_counts.values()) >= 480_000, ticker_counts
    assert len(sampler) == 30_000 * 32, len(sampler)

    batch = next(iter(trainloader))
    (message, book), labels, (message_times, book_times) = prep_batch(
        batch, seq_len, num_devices=1
    )
    per_process_batch = args.batch_size * 4
    assert tuple(message.shape) == (per_process_batch, 13_000), message.shape
    assert tuple(labels.shape) == (per_process_batch, 13_000), labels.shape
    assert tuple(book.shape) == (per_process_batch, 500, 503), book.shape
    assert tuple(message_times.shape) == (per_process_batch, 13_000), message_times.shape
    assert tuple(book_times.shape) == (per_process_batch, 13_000), book_times.shape

    print(
        "MODEL_ZOO_DATA_OK "
        f"tickers={','.join(tickers)} train_size={train_size} "
        f"min_ticker_sequences={min(ticker_counts.values())} "
        f"message={tuple(message.shape)} book={tuple(book.shape)} "
        f"labels={tuple(labels.shape)}"
    )


if __name__ == "__main__":
    main()
