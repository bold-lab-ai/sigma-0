#!/usr/bin/env python3
"""Run the standard LOB-Bench evaluation of one generation run.

The protocol is fixed in ``lobpipeline.lob_bench.standard_eval``: the frozen
sample index, the merged scorer, ``uncond`` with the divergence curve at the
default horizon. This launcher checks the inference tree before scoring, scores
it once, and writes ``standard_eval.json`` beside the driver's own summaries.

    PYTHONPATH=src python run/benchmarking/run_standard_eval.py \\
        --inference-dir <evaluation>/inference.sqfs \\
        --output-dir <outside the repository> \\
        --run-name mamba3_seed42 \\
        --python "$WORKDIR/envs/lobbench_venv/bin/python" \\
        --generation-manifest <evaluation>/generation_complete.json

``--no-divergence`` skips the divergence curve, which is a second scoring
pass. The record then has ``divergence: null``.

``--index-file`` exists for smoke tests on a small window set. A run on any
index other than the canonical one is recorded with ``"standard": false``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from lobpipeline.lob_bench.scoring import (  # noqa: E402
    DEFAULT_DIVERGENCE_HORIZON,
    build_lobbench_plan,
    execute_lobbench_plan,
    summarize_plan_divergence,
)
from lobpipeline.lob_bench.standard_eval import (  # noqa: E402
    CANONICAL_INDEX_FILE,
    build_record,
    directory_digest,
    load_protocol,
    pooled_intervals,
    run_provenance,
    standard_scorer,
    write_record,
)
from lobpipeline.storage import sqfs  # noqa: E402
from run.benchmarking.validate_model_zoo_evaluation import validate_inference  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--inference-dir",
        required=True,
        help="directory with data_gen/, data_real/, data_cond/ and the rank index files, or its .sqfs image",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--python", default=sys.executable, help="scoring interpreter")
    parser.add_argument("--n-workers", type=int, default=48)
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="ranks the generation ran on; one sample_indices_rank<r>.json each",
    )
    parser.add_argument("--index-file", default=str(CANONICAL_INDEX_FILE))
    parser.add_argument("--generation-manifest", default=None)
    parser.add_argument(
        "--no-divergence",
        action="store_true",
        help="skip the divergence curve; the record says so and the headline is unchanged",
    )
    args = parser.parse_args(argv)

    protocol = load_protocol(args.index_file, divergence=not args.no_divergence)
    benchmark_root, scorer_tree = standard_scorer()
    output_dir = Path(args.output_dir).expanduser().resolve()

    with sqfs.resolve(args.inference_dir) as inference_dir:
        inference_dir = Path(inference_dir)
        # Rows, file counts and the exact window list, before any scoring.
        checked = validate_inference(
            inference_dir,
            protocol.sequences,
            protocol.generated_messages,
            args.world_size,
            Path(protocol.index_file),
            protocol.dataset_length,
        )
        inputs = {
            "inference_dir": str(inference_dir),
            "sample_indices_sha256": checked["sample_indices_sha256"],
            "sample_indices_file_sha256": checked["sample_indices_file_sha256"],
            "inventory": checked["inventory"],
            "data_gen": directory_digest(inference_dir / "data_gen"),
            "data_real": directory_digest(inference_dir / "data_real"),
            "data_cond": directory_digest(inference_dir / "data_cond"),
        }
        plan = build_lobbench_plan(
            run_name=args.run_name,
            inference_dir=inference_dir,
            output_dir=output_dir,
            stock=protocol.stock,
            period=protocol.period,
            benchmark_root=benchmark_root,
            python=args.python,
            n_workers=args.n_workers,
            mode=protocol.mode,
            divergence_horizon=protocol.divergence_horizon or DEFAULT_DIVERGENCE_HORIZON,
            with_divergence=protocol.divergence_horizon is not None,
        )
        pooled = execute_lobbench_plan(plan)
        divergence = (
            summarize_plan_divergence(plan) if protocol.divergence_horizon is not None else None
        )

    record = build_record(
        run_name=args.run_name,
        protocol=protocol,
        scorer_tree_sha256=scorer_tree,
        pooled=pooled,
        divergence=divergence,
        intervals=pooled_intervals(plan.score_path),
        inputs=inputs,
        provenance=run_provenance(
            python=args.python,
            n_workers=args.n_workers,
            generation_manifest=args.generation_manifest,
        ),
    )
    destination = write_record(record, Path(plan.results_dir) / "standard_eval.json")
    print(json.dumps({"standard_eval": str(destination), "standard": record["standard"],
                      **record["headline"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
