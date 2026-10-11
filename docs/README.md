# sigma-0 documentation

Reference pages that stay true, plus an index of the experiments this code has
run. Dated status lives in pull requests and in the results database, not here.

## Pages

| page | what it answers |
|---|---|
| [conventions.md](conventions.md) | the platform rules: Lustre safety, checkpoints, group permissions, reproducibility |
| [architecture.md](architecture.md) | what each package does, how data flows through them, where the code came from |
| [environments.md](environments.md) | which interpreter runs what, the multi-node native stack, the scoring venv |
| [data.md](data.md) | data formats, which loader consumes which, the symlink table, the hard data rules |
| [training.md](training.md) | the launcher chain, configs, knobs, what a smoke proves |
| [inference.md](inference.md) | the generator, its inputs, checkpoint pinning, do(Q) injection |
| [benchmarking.md](benchmarking.md) | LOB-Bench scoring modes and sharding, the event-anchored return bench |
| [checkpoints.md](checkpoints.md) | the model store, pinning by step, the legacy-norm switch |
| [ci-and-evidence.md](ci-and-evidence.md) | the cluster measures and CI audits: every record, how to re-record it, every guard |
| [shared-checkout.md](shared-checkout.md) | traps of the shared checkout and of Slurm allocations, how to attach and how to merge |
| [lessons.md](lessons.md) | the lessons that cost time, grouped by theme |
| [fidelity_defect_taxonomy.md](fidelity_defect_taxonomy.md) | the defect classes and the register (`defect_register.json`, rendered by `tools/render_register.py`) |
| [experiments/README.md](experiments/README.md) | every experiment: entry points, configs, evidence, where the numbers are |

## Conventions for these pages

Short sentences. Absolute Lustre paths only where a reader must go there. Every
number carries the job ID or commit it came from. When code changes behaviour,
the page that describes it changes in the same PR. Write-ups of finished
experiments go under `experiments/<slug>/`; figures and result files go to the
results database and are linked, not committed.
