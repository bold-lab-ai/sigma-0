#!/usr/bin/env python3
"""Check that package boundaries import.

The check is tiered because the tiers have different prerequisites, and
collapsing them hides which half is broken:

  core   — imports with the standard library alone: the package boundaries
           themselves plus the stdlib-only glue (lobpipeline, manifests,
           scoring plans, the lazy base_model facade). Runs in any login-node
           shell, no conda environment needed.
  engine — the JAX-LOB matching engine and the RL environments (needs jax,
           and pandas for the LOBSTER loader).
  model  — the base_model stack (needs jax, pandas, torch).

Full check, under the training conda environment (see docs/environments.md):

    PYTHONPATH=src python run/migrate/check_imports.py

Login-node shells without jax can still validate the boundary layer:

    PYTHONPATH=src python3 run/migrate/check_imports.py --tiers core

`core` is exactly the set the previous flat MODULES tuple covered, so the
cheap check is no weaker than it was; `engine` and `model` are the modules
that were never covered because importing them needs the real stack.
"""

from __future__ import annotations

import argparse
import importlib


TIERS: dict[str, tuple[str, ...]] = {
    "core": (
        "environment",
        "environment.jaxen",
        "environment.jaxrl",
        "matching_engine",
        "matching_engine.jaxlobster",
        "matching_engine.jaxob",
        "matching_engine.utils",
        "base_model",
        "base_model.config",
        "base_model.data",
        "base_model.encoding",
        "base_model.models",
        "base_model.training",
        "base_model.inference",
        "base_model.evaluation",
        "base_model.sharding",
        "base_model.kernels",
        "base_model.utils",
        "lobpipeline",
        "lobpipeline.pipeline",
        "lobpipeline.lob_bench",
        "lobpipeline.model_store",
        "lobpipeline.return_bench",
        "lobpipeline.manifests",
    ),
    "engine": (
        "matching_engine.jaxob.jorderbook",
        "matching_engine.jaxob.JaxOrderBookArrays",
        "matching_engine.jaxlobster.data_loading",
    ),
    "model": (
        "base_model.data.preproc",
        "base_model.encoding.encoding",
        "base_model.encoding.encoding_26tok",
        "base_model.data.lobster_dataloader",
        "base_model.models.lob_seq_model",
        "base_model.inference.generate",
        "base_model.models.mamba3_jax",
        "base_model.models.seq_model",
        "base_model.models.es_fold",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiers",
        default="core,engine,model",
        help="comma-separated subset of: core,engine,model",
    )
    args = parser.parse_args()

    requested = [tier.strip() for tier in args.tiers.split(",") if tier.strip()]
    unknown = [tier for tier in requested if tier not in TIERS]
    if unknown:
        raise SystemExit(f"unknown tiers: {unknown}; expected subset of {sorted(TIERS)}")

    total = 0
    for tier in requested:
        for module in TIERS[tier]:
            importlib.import_module(module)
            total += 1
        print(f"import_check tier={tier} ok modules={len(TIERS[tier])}")
    print(f"import_check=ok tiers={','.join(requested)} modules={total}")


if __name__ == "__main__":
    main()
