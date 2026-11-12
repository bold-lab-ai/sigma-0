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

Optional extras, all off by default and none of them changing the headline:
``--kl`` (KL divergence per feature, read from the score pickle, no rescoring),
``--conditional``, ``--context`` and ``--time-lagged`` (one more driver run
each). ``--all-extras`` turns all four on. A part that did not run is ``null``
in the record, and ``provenance.seconds`` says what each part cost.

``--index-file`` exists for smoke tests on a small window set. A run on any
index other than the canonical one is recorded with ``"standard": false``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    OPTIONAL_MODES,
    STANDARD_TIME_LAG,
    build_record,
    directory_digest,
    load_protocol,
    optional_mode_section,
    pooled_intervals,
    pooled_kl,
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
    extras = parser.add_argument_group("optional extras (off by default)")
    extras.add_argument("--kl", action="store_true",
                        help="KL(real || generated) per feature, from the score pickle")
    extras.add_argument("--conditional", action="store_true",
                        help="the driver's cond mode: one more driver run")
    extras.add_argument("--context", action="store_true",
                        help="the driver's context mode (W1 and KS only): one more driver run")
    extras.add_argument("--time-lagged", action="store_true",
                        help=f"the driver's time_lagged mode at lag {STANDARD_TIME_LAG}: "
                             "one more driver run")
    extras.add_argument("--all-extras", action="store_true",
                        help="all four extras above")
    args = parser.parse_args(argv)
    if args.all_extras:
        args.kl = args.conditional = args.context = args.time_lagged = True
    wanted_modes = [
        mode
        for mode, on in (("cond", args.conditional), ("context", args.context),
                         ("time_lagged", args.time_lagged))
        if on
    ]
    assert set(wanted_modes) <= set(OPTIONAL_MODES)
    seconds: dict[str, float] = {}

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
        started = time.monotonic()
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
        seconds["headline" if divergence is None else "headline_and_divergence"] = round(
            time.monotonic() - started, 1
        )
        kl = None
        if args.kl:
            started = time.monotonic()
            kl = pooled_kl(plan.score_path)
            seconds["kl"] = round(time.monotonic() - started, 1)
        optional = {}
        for mode in wanted_modes:
            started = time.monotonic()
            time_lag = STANDARD_TIME_LAG if mode == "time_lagged" else None
            extra_plan = build_lobbench_plan(
                run_name=args.run_name,
                inference_dir=inference_dir,
                output_dir=output_dir,
                stock=protocol.stock,
                period=protocol.period,
                benchmark_root=benchmark_root,
                python=args.python,
                n_workers=args.n_workers,
                mode=mode,
                time_lag=time_lag,
            )
            optional[mode] = optional_mode_section(
                execute_lobbench_plan(extra_plan),
                scorer_tree_sha256=scorer_tree,
                time_lag=time_lag,
            )
            seconds[mode] = round(time.monotonic() - started, 1)

    record = build_record(
        run_name=args.run_name,
        protocol=protocol,
        scorer_tree_sha256=scorer_tree,
        pooled=pooled,
        divergence=divergence,
        kl=kl,
        optional=optional,
        intervals=pooled_intervals(plan.score_path),
        inputs=inputs,
        provenance=run_provenance(
            python=args.python,
            n_workers=args.n_workers,
            generation_manifest=args.generation_manifest,
            seconds=seconds,
        ),
    )
    destination = write_record(record, Path(plan.results_dir) / "standard_eval.json")
    print(json.dumps({"standard_eval": str(destination), "standard": record["standard"],
                      **record["headline"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
