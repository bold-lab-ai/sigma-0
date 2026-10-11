# Environments

The repository says which code runs; this says which interpreter and which
native libraries it runs under. Both halves are load-bearing, a checkpoint
restored under the wrong orbax, or a multi-node run without the NCCL source
build, fails in ways that look like model problems.

Everything here was verified against the live cluster on **2026-07-15**; paths
are Isambard-AI (GH200, aarch64, Slingshot/CXI).

---

## 1. The two conda families

There are exactly two supported Python environment families. Which one you need
is decided by **the orbax version that wrote the checkpoint you want to load**,
not by preference.

### Family 1, `quant` miniforge3 base (training, current checkpoints)

- Prefix: `/projects/public/s5e/quant_team/quant/miniforge3`
  (or `/projects/public/s5e/quant_team/quant/miniforge3/envs/$CONDA_ENV` for a
  named sub-env; the node wrapper convention is `CONDA_ENV=base` → the prefix
  itself).
- Python 3.12. **Live** versions probed 2026-07-15 with
  `miniforge3/bin/python -c "import jax, flax, optax, orbax.checkpoint ..."`:

  | package            | version        |
  |--------------------|----------------|
  | jax                | 0.9.0.1        |
  | flax               | 0.12.2         |
  | optax              | 0.2.6          |
  | orbax-checkpoint   | 0.11.32        |
  | torch              | 2.8.0+cu129    |
  | zstandard          | 0.23.0         |
  | wandb              | 0.21.3         |
  | numpy              | 2.3.3          |
  | pandas             | 2.3.2          |

  This is a **live shared env**, versions drift as the team upgrades it.
  Re-probe before pinning numbers into a paper; `envs/requirements-train.txt`
  records the (unpinned) training dependency list with provenance.

### Family 2, `lobs5` (legacy / old checkpoints)

- Prefix: `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/envs/lobs5`
- Python 3.11, **pinned** (load-bearing pins per the LOB_Mech_Interp
  requirements + README):

  | package            | version   |
  |--------------------|-----------|
  | jax / jaxlib       | 0.7.2     |
  | jax-cuda12-pjrt/plugin | 0.7.2 |
  | flax               | 0.12.0    |
  | optax              | 0.2.2     |
  | orbax-checkpoint   | **0.5.10** |
  | chex               | 0.1.86    |

### Which family when (the rule)

- **A checkpoint written with orbax X needs the family whose orbax matches.**
  Old checkpoints (e.g. the 8.2M Mamba3 mech-interp model, written with
  jax 0.7.2 / orbax 0.5.10) do NOT restore cleanly under orbax 0.11.x, use
  `lobs5`. The 78M s28730 line and everything trained recently was written by
  the base env, use Family 1.
- **New training always uses Family 1** (the base env). Never start a new run
  on `lobs5`.

### Activation pattern (do NOT `conda activate`)

`conda.sh` in the shared miniforge3 has hardcoded broken paths to a personal
home directory. Activate by prefix instead (this is what
`run/base_model/node_wrapper.sh` does):

```bash
CONDA_ENV=${CONDA_ENV:-base}
if [ "$CONDA_ENV" = "base" ]; then
  export CONDA_PREFIX=/projects/public/s5e/quant_team/quant/miniforge3
else
  export CONDA_PREFIX=/projects/public/s5e/quant_team/quant/miniforge3/envs/$CONDA_ENV
fi
export PATH=$CONDA_PREFIX/bin:$PATH

module load cuda/12.6
# Re-prepend conda bin AFTER module load: cuda/12.6 puts the system ptxas
# (12.6, PTX ISA 8.5) ahead of conda's ptxas (12.9, PTX ISA 8.9). Triton 3.4.0
# emits PTX 8.7 → needs ptxas 12.7+, i.e. conda's must win.
export PATH=$CONDA_PREFIX/bin:$PATH
```

---

## 2. Native library stack for multinode (NCCL / OFI / libfabric)

Required for any multi-node JAX run on Slingshot. All three pieces, in order:

1. **NCCL 2.29.3 source build**, `/projects/public/s5e/quant_team/quant/nccl-2.29.3/lib`.
   Built from source (commit 25368a7) with GCC 12.3 + CUDA 12.6, sm_90.
   Fixes the ARM CAS weak-compare-exchange failure in `proxy.cc` on GH200
   (aarch64). History: the `lobmax` conda NCCL 2.29.2 still has the weak-CAS
   codepath under GCC < 10; the old `lob` env's 2.28.9 has the ARM CAS bug
   outright. The source build is the definitive fix.

2. **Why `LD_PRELOAD` and not just `LD_LIBRARY_PATH`**: `xla_cuda_plugin.so`
   carries a `DT_RPATH` (not `RUNPATH`) pointing at conda's
   `nvidia/nccl/lib`. `DT_RPATH` is searched BEFORE `LD_LIBRARY_PATH`, so a
   path override is silently bypassed. Preloading forces our build to load
   first regardless of RPATH resolution:

   ```bash
   NCCL_LIB_OVERRIDE=/projects/public/s5e/quant_team/quant/nccl-2.29.3/lib
   export LD_PRELOAD=$NCCL_LIB_OVERRIDE/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}
   ```

   Sanity check: `strings $NCCL_LIB_OVERRIDE/libnccl.so.2 | grep "NCCL version"`
   must say **cuda12.6** (the source build), not cuda12.9 (conda's copy).

3. **aws-ofi-nccl 1.18.0 + Cray libfabric**, the NCCL↔Slingshot bridge:

   ```bash
   AWS_OFI_NCCL_LIB=${AWS_OFI_NCCL_LIB:-/projects/public/s5e/quant_team/quant/aws-ofi-nccl-1.18.0/lib}
   export LD_LIBRARY_PATH=$AWS_OFI_NCCL_LIB:/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH
   ```

   The system aws-ofi-nccl 1.8.1 was 10 versions behind and caused the
   512-node (2048-GPU) NCCL comm-init hang; 1.18.0 is the fix (upgraded
   2026-02-25). `AWS_OFI_NCCL_LIB` stays overridable as the rollback lever.

The reference implementation of all of this (plus the XLA flags, NCCL_BUFFSIZE,
CXI resilience tiers by node count) is the node wrapper under
`run/base_model/`, treat that as the single source of truth for tuning values.

---

## 3. JAX persistent compilation cache

```bash
export JAX_COMPILATION_CACHE_DIR="/projects/u6gb/quant/jax_cache_lobs5"
```

Cache key = HLO module fingerprint (model config + sharding + device count),
so a resume with the same model and node count skips XLA recompilation
(~30–120 s per resume). The directory must be on the shared filesystem so all
nodes across jobs hit the same cache.

---

## 4. Scoring venv (LOB-Bench)

Scoring runs in a **separate CPU venv**, not the training conda env: the
LOB-Bench dependency set (pandas 3, plotly, statsmodels, sklearn) conflicts
with the training stack, and the benchmark must never be imported into the
training process.

- Training-side python: `$CONDA_PREFIX/bin/python3` (Family 1).
- Scoring-side python: the venv's `bin/python`, handed to
  `run/benchmarking/run_lobbench.py --python`, which
  `lobpipeline.lob_bench.scoring.build_lobbench_plan` then invokes against
  `DEFAULT_BENCHMARK_ROOT`
  (`/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/lob_pipeline`,
  readable from u6gb accounts, verified 2026-08-15).

Setup script: **`envs/lobbench_venv_setup.sh`** → `envs/lobbench_venv`
(gitignored). It pins the scoring deps that the working EGGROLL venv had.

The benchmark implementation itself stays out of this repo on purpose: sigma-0
records the scorer's Git revision in every summary rather than carrying a
second copy that can drift (see `lobpipeline/lob_bench/scoring.py`).

---

## 5. `MAMBA3_LEGACY_NORM` convention

Moved to `docs/checkpoints.md`, next to the checkpoints it applies to. Short version: unset for anything trained after 2026-04-22, `1` only for the pre-fix Mamba3 checkpoints, and the wrong value scrambles the dynamics without crashing.

---

## 6. Compute-node conventions

- **`export TMPDIR=/tmp`**, set BEFORE any Python import (wandb creates its
  tempdir at import time). Also: `/local/user` scratch has been wiped ~30–40 s
  into workq jobs since 2026-06-10, so `/tmp` is the safe node-local target.
- Stage model weights and data from Lustre to `$TMPDIR` before training starts;
  write checkpoints/logs node-local and rsync to Lustre at the end (see
  `docs/data.md` for the full data rules).
- **`umask 002`** everywhere; every file on the shared filesystem must end up
  `group = brics.u6gb` with group rwx on dirs / rw on files. After any
  in-place rewrite (`sed -i`, `tee`, `>`), re-run `chmod g+rw`, temp-file
  renames silently strip group bits.

---

## Files under `envs/`

- `requirements-train.txt`, the training dependency list (Family 1), copied
  with provenance from the Mamba3 GOOG pretraining repo.
- `lobbench_venv_setup.sh`, scoring venv bootstrap (maintained separately).
