from pathlib import Path
from types import SimpleNamespace

from base_model.data.dataloading import UniformTickerDistributedSampler


def _dataset(tmp_path: Path):
    files = []
    for ticker in ("AAPL", "GOOG"):
        ticker_dir = tmp_path / ticker
        files.extend(
            ticker_dir / f"{ticker}_2025-01-0{day}_message.npy"
            for day in (1, 2)
        )
    # Ten sequences per file, with files contiguous by ticker.
    return SimpleNamespace(
        message_files=[str(path) for path in files],
        _seqs_cumsum=[0, 10, 20, 30, 40],
    )


def test_uniform_sampler_builds_ticker_homogeneous_global_batches(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    rank0 = UniformTickerDistributedSampler(
        dataset, ("AAPL", "GOOG"), per_process_bsz=2,
        num_replicas=2, rank=0, seed=42, num_batches=4,
    )
    rank1 = UniformTickerDistributedSampler(
        dataset, ("AAPL", "GOOG"), per_process_bsz=2,
        num_replicas=2, rank=1, seed=42, num_batches=4,
    )
    samples0, samples1 = list(rank0), list(rank1)

    seen_tickers = []
    all_indices = []
    for batch_idx in range(4):
        global_batch = (
            samples0[batch_idx * 2:(batch_idx + 1) * 2]
            + samples1[batch_idx * 2:(batch_idx + 1) * 2]
        )
        tickers = {"AAPL" if index < 20 else "GOOG" for index in global_batch}
        assert len(tickers) == 1
        seen_tickers.extend(tickers)
        all_indices.extend(global_batch)

    assert seen_tickers.count("AAPL") == 2
    assert seen_tickers.count("GOOG") == 2
    assert len(all_indices) == len(set(all_indices))


def test_uniform_sampler_resume_skips_complete_global_batches(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    full = UniformTickerDistributedSampler(
        dataset, ("AAPL", "GOOG"), per_process_bsz=2,
        num_replicas=2, rank=0, seed=137, num_batches=4,
    )
    resumed = UniformTickerDistributedSampler(
        dataset, ("AAPL", "GOOG"), per_process_bsz=2,
        num_replicas=2, rank=0, seed=137, num_batches=4, start_batch=2,
    )

    assert list(resumed) == list(full)[4:]
    assert len(resumed) == 4
