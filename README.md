# sigma-0

The team's code base for limit-order-book sequence models: the matching engine,
the replay environments, the base model with its encodings, training and
inference, the LOB-Bench and return-bench pipeline, and post-training. One
importable `src/` root, one set of launchers, one set of tests.

Results do not live here. They go to the results database at
`/lus/lfs1aip2/projects/public/u6gb/results`. Read `AGENTS.md` before your
first commit; the rules in it are the ones that have cost the team time.

## Layout

```text
src/
├── base_model/       the LOB sequence-model stack
│   ├── encoding/       runtime selector + 1/22/23/24/26-token tokenizers
│   ├── data/           LOBSTER datasets, loaders, file resolution, preprocessing
│   ├── models/         registry, Mamba3, S5, GDN/KDA, Transformer, NSA, hybrid, LOB model
│   ├── kernels/        Mamba3 Triton and CUDA state scans, GDN and NSA kernels
│   ├── training/       train loop, state init and Orbax restore, launch plans
│   ├── inference/      generator, evaluation, validation, launch plans
│   ├── evaluation/     message perplexity
│   └── sharding/       device mesh and shardings
├── matching_engine/  JAX-LOB engine: jaxob, jaxlobster, utils
├── environment/      replay environments and RL baselines: jaxen, jaxrl
├── lobpipeline/      LOB-Bench adapter + vendored scorer, return bench, manifests, model store
├── post_training/    closed-loop market making (heuristic_learning)
└── state/            state model: the book as a video on a frame grid
run/                  Slurm launchers and runtimes, one directory per subsystem
configs/              train, inference, bench and package configs
tests/                unit, integration, regression, smoke
ci/                   evidence recorder, auditor and the stdlib guards
tools/                data build, diagnostics, dev scripts
docs/                 the wiki (start at docs/README.md)
```

## First run

```bash
bash bootstrap_fork.sh     # data/ symlinks, log dir, core import + launcher syntax gates
source setup_env.sh        # PYTHONPATH=src, umask 002
```

Which interpreter to use is in `docs/environments.md`. Then, on a login node:

```bash
PYTHONPATH=src python -m compileall -q src tests run tools ci
PYTHONPATH=src python run/migrate/check_imports.py --tiers core          # stdlib only
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests -q               # training conda
PYTHONPATH=src python run/base_model/train_base_model.py --config configs/train/mamba3_smoke.yaml
```

The last command prints the `sbatch` command for a one-node training smoke
without submitting it. `docs/training.md` and `docs/inference.md` take it from
there.

## Models

The six backbones share one data path, one outer LOB model, one training loop
and one inference implementation. Pick one with `--architecture` or the
`ARCHITECTURE` environment variable: `mamba3` (default), `s5`, `gdn`, `kda`,
`transformer`, `nsa`, plus `hybrid_mamba3`. Trained checkpoints are resolved by
name through `lobpipeline.model_store` (`docs/checkpoints.md`).

## Where to read next

| you want to | read |
|---|---|
| know the rules | `AGENTS.md`, `docs/conventions.md` |
| understand the packages and where the code came from | `docs/architecture.md` |
| set up an interpreter or a multi-node run | `docs/environments.md` |
| find data, or add a data source | `docs/data.md` |
| train, resume, or smoke-test a model | `docs/training.md` |
| generate order flow, inject an order | `docs/inference.md` |
| score a run | `docs/benchmarking.md` |
| load a blessed checkpoint | `docs/checkpoints.md` |
| understand why CI is red | `docs/ci-and-evidence.md` |
| work in the shared checkout or attach to an allocation | `docs/shared-checkout.md` |
| see what was tried and where the results are | `docs/experiments/README.md` |

## Citation

```bibtex
@misc{an2026sigma0,
      title={{Sigma-0}: Unlock Large Scale Quant Foundation Model},
      author={Junming An and Kang Li and Aramis Marti-Shahandeh and Alexandre Bismuth and Sascha Frey and Yuhe Gao and Valentin Mohl and George Nigmatulin and Aaron Rose and Bidipta Sarkar and Mihai Cucuringu and Stefan Zohren and Leandro S{\'a}nchez-Betancourt and Ani Calinescu and Jakob Nicolaus Foerster},
      year={2026},
      howpublished={\url{https://github.com/bold-lab-ai/sigma-0}},
      url={https://github.com/bold-lab-ai/sigma-0},
}
```
