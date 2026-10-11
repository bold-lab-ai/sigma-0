#!/usr/bin/env python3
"""Verify the three deterministic Mamba3 START-token replay cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from base_model.encoding.encoding import Vocab
from base_model.inference import validation as valh


DT_NS_DECODED_INDEX = 7
TIME_S_DECODED_INDEX = 8
TIME_NS_DECODED_INDEX = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_root = args.task_root.expanduser().resolve()
    original_root = args.original_root.expanduser().resolve()
    manifest_path = require_file(task_root / "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    vocab = Vocab()
    start_token = int(vocab.START_TOK)
    delta_ns_start, delta_ns_end = valh.get_idx_from_field("delta_t_ns")
    current_mask = np.asarray(
        valh.syntax_validation_matrix(block_start_tok=False)
    )
    blocked_mask = np.asarray(
        valh.syntax_validation_matrix(block_start_tok=True)
    )
    canonical_indices = [
        int(line)
        for line in Path(manifest["canonical_indices"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    canonical_index_digest = hashlib.sha256(
        json.dumps(canonical_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sampling_contract = manifest["sampling_contract"]
    base_seed = int(sampling_contract["base_seed"])
    world_size = int(sampling_contract["world_size"])
    batch_size = int(sampling_contract["batch_size"])

    results = []
    for case in manifest["cases"]:
        dataset_index = int(case["dataset_index"])
        date = case["date"]
        row = int(case["generated_message_row_zero_based"])
        rank = int(case["rank"])
        rank_batch = int(case["rank_batch"])
        slot = int(case["slot"])
        output_subdir = Path(case["output_subdir"])
        trace_stem = f"GOOG_{date}_real_id_{dataset_index}"
        csv_name = (
            f"GOOG_{date}_message_real_id_{dataset_index}_gen_id_0.csv"
        )

        trace_npz = require_file(
            task_root / output_subdir / "token_trace" / f"{trace_stem}.npz"
        )
        trace_json = require_file(
            task_root / output_subdir / "token_trace" / f"{trace_stem}.json"
        )
        replay_index_json = require_file(
            task_root
            / output_subdir
            / "inference"
            / f"sample_indices_rank{rank}.json"
        )
        replay_csv = require_file(
            task_root / output_subdir / "inference" / "data_gen" / csv_name
        )
        original_csv = require_file(
            original_root / "evaluation" / "inference" / "data_gen" / csv_name
        )

        with np.load(trace_npz) as trace:
            tokens = np.asarray(trace["generated_tokens"])
            decoded = np.asarray(trace["decoded_messages"])
        trace_sidecar = json.loads(trace_json.read_text(encoding="utf-8"))
        replay_index = json.loads(
            replay_index_json.read_text(encoding="utf-8")
        )

        if row <= 0 or row >= tokens.shape[0]:
            raise ValueError(
                f"invalid target row {row} for dataset index {dataset_index}"
            )
        if tokens.shape[1] <= delta_ns_end:
            raise ValueError(
                f"unexpected token shape for dataset index {dataset_index}: "
                f"{tokens.shape}"
            )

        token_slice = tokens[row, delta_ns_start:delta_ns_end]
        all_start_locations = np.argwhere(tokens == start_token).tolist()
        all_start_locations_in_delta_ns = [
            location
            for location in all_start_locations
            if delta_ns_start <= location[1] < delta_ns_end
        ]
        start_positions = np.flatnonzero(tokens[row] == start_token).tolist()
        start_positions_in_delta_ns = [
            position
            for position in start_positions
            if delta_ns_start <= position < delta_ns_end
        ]
        decoded_delta_ns = int(decoded[row, DT_NS_DECODED_INDEX])
        previous_time = [
            int(decoded[row - 1, TIME_S_DECODED_INDEX]),
            int(decoded[row - 1, TIME_NS_DECODED_INDEX]),
        ]
        current_time = [
            int(decoded[row, TIME_S_DECODED_INDEX]),
            int(decoded[row, TIME_NS_DECODED_INDEX]),
        ]
        time_delta_ns = (
            (current_time[0] - previous_time[0]) * 1_000_000_000
            + current_time[1]
            - previous_time[1]
        )
        rank_indices = canonical_indices[rank::world_size]
        replay_start = rank_batch * batch_size
        expected_batch_indices = rank_indices[
            replay_start : replay_start + batch_size
        ]
        rng = jax.random.key(base_seed + rank)
        rng, _ = jax.random.split(rng)
        for _ in range(rank_batch):
            rng, _ = jax.random.split(rng)
        probe_rng, _ = jax.random.split(rng)
        _, probe_batch_rng = jax.random.split(probe_rng)
        slot_rng_keys = jax.random.split(probe_batch_rng, batch_size)
        observed_slot_rng_key = (
            jax.random.key_data(slot_rng_keys[slot]).tolist()
        )

        original_sha = sha256(original_csv)
        replay_sha = sha256(replay_csv)
        checks = {
            "original_csv_sha_matches_frozen_manifest": (
                original_sha == case["original_csv_sha256"]
            ),
            "replay_csv_is_byte_identical": replay_sha == original_sha,
            "start_is_in_target_delta_t_ns": bool(
                start_positions_in_delta_ns
            ),
            "current_mask_allows_start_at_observed_position": all(
                bool(current_mask[position, start_token])
                for position in start_positions_in_delta_ns
            ),
            "blocked_mask_rejects_start_at_observed_position": all(
                not bool(blocked_mask[position, start_token])
                for position in start_positions_in_delta_ns
            ),
            "decoded_delta_t_ns_is_minus_9999": decoded_delta_ns == -9999,
            "timestamp_moves_back_9999_ns": time_delta_ns == -9999,
            "trace_sidecar_agrees": (
                trace_sidecar["dataset_index"] == dataset_index
                and trace_sidecar["date"] == date
                and trace_sidecar["token_shape"] == list(tokens.shape)
                and trace_sidecar["decoded_message_shape"]
                == list(decoded.shape)
                and trace_sidecar["start_token_id"] == start_token
                and trace_sidecar["start_token_locations"]
                == all_start_locations
                and trace_sidecar["start_token_locations_in_delta_t_ns"]
                == all_start_locations_in_delta_ns
            ),
            "replay_index_manifest_agrees": (
                replay_index["all_indices_sha256"]
                == canonical_index_digest
                and replay_index["world_size"] == world_size
                and replay_index["rank"] == rank
                and replay_index["effective_generation_seed"]
                == base_seed + rank
                and replay_index["replay_rank_batch"] == rank_batch
                and replay_index["rank_indices"] == expected_batch_indices
            ),
            "target_slot_rng_key_matches_manifest": (
                observed_slot_rng_key == case["slot_rng_key_data"]
            ),
        }
        results.append(
            {
                "dataset_index": dataset_index,
                "date": date,
                "generated_message_row_zero_based": row,
                "token_slice_delta_t_ns": token_slice.tolist(),
                "start_token_positions_in_message": start_positions,
                "start_token_positions_in_delta_t_ns": (
                    start_positions_in_delta_ns
                ),
                "mask_allows_start": {
                    str(position): bool(current_mask[position, start_token])
                    for position in range(delta_ns_start, delta_ns_end)
                },
                "decoded_delta_t_ns": decoded_delta_ns,
                "previous_time": previous_time,
                "current_time": current_time,
                "time_delta_ns": int(time_delta_ns),
                "replay_batch_indices": expected_batch_indices,
                "target_slot_rng_key_data": observed_slot_rng_key,
                "original_csv_sha256": original_sha,
                "replay_csv_sha256": replay_sha,
                "checks": checks,
                "causal_chain_closed": all(checks.values()),
            }
        )

    payload = {
        "experiment": manifest["experiment"],
        "diagnostic_runtime_commit": manifest[
            "diagnostic_runtime_commit"
        ],
        "start_token_id": start_token,
        "delta_t_ns_token_slice": [delta_ns_start, delta_ns_end],
        "cases": results,
        "all_cases_closed": all(
            result["causal_chain_closed"] for result in results
        ),
    }

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace trace result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    output_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_tmp.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if not payload["all_cases_closed"]:
        raise SystemExit("one or more replay cases did not close the causal chain")


if __name__ == "__main__":
    main()
