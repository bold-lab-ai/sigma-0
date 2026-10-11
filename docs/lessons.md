# Lessons

What the June to September 2026 work taught, kept only where it still applies.
Each line cost at least one job or one afternoon. The dated logs these were
distilled from are in git history at commit `127d76b` (`learnt_lessons.md`,
`findings.md`).

## Launching

- A green `sbatch --test-only` proves scheduling, not the runtime environment
  or the data root. The first Python assertion in `train_full_autoreg.batch`
  runs before `node_wrapper.sh`, so the conda `PATH` must be right in the
  submitting shell.
- Pass comma-bearing variables (`TRAIN_DATE_RANGE`) through the environment,
  not through `--export=ALL,VAR=a,b`.
- A one-task job inherits the full node's default memory request unless
  `--mem` is set; a sub-node job needs `--gres=gpu:N` explicitly, and
  `--gpu-bind=map_gpu:0,1,2,3` on a one-GPU allocation warns or fails.
- `#SBATCH --output` will not create directories. Log to an existing parent
  and create per-job directories inside the script.
- The restored launcher chain is `train_full_autoreg.batch` then
  `node_wrapper.sh` then `run/base_model/runtime/train.py`; each layer can fail
  on its own. Record the exact boundary a smoke reached, not "training ran".
- Reaching `node_wrapper.sh`, initialising four GPUs, and completing one
  optimizer step are three different milestones. Say which one you hit.

## Data

- `lob_preproc_linf` holds inference snapshot `.npz`, not the paired
  message and orderbook training files `discover_ticker_files` wants.
- Public preprocessing directories can still be unusable when their symlinks
  point into a private project path. Check readability by exact path first.
- A `DATA_INDEX_JSON` sidecar in the repository is the right way to point a
  smoke at a shared root without writing into it.
- The inference runtime's wide-book selector requires an L500 source; an L10
  day fails with "wide-book width must be divisible by 4". Mount the L500
  shard (`docs/data.md`, section 1.3).
- Never decompress `.npy.zst` inside the batch loop; pre-decompress once and
  leave a breadcrumb.

## Checkpoints

- Standard Orbax restore fails when the saved optax structure differs from the
  code's; a parameters-only fallback is what makes legacy checkpoints trainable.
- A checkpoint restores by Flax class names and pytree structure. Import paths
  and module names do not matter; class names do.
- Loading a Mamba3 checkpoint with the wrong `MAMBA3_LEGACY_NORM` does not
  crash. It scrambles the dynamics (CE about 8 instead of about 0.6). Record the
  value beside the checkpoint (`docs/checkpoints.md`).
- Resolve a step explicitly. Ranking by KS and by direction accuracy disagree,
  so "latest" is not "best", and listing a checkpoint directory is the metadata
  pattern that got the team suspended.
- A `COMPLETED 0:0` segment can be a time-slice exit, not a finished epoch.
  Read the training log's boundary and the saved step.
- A successful restore plus one curtailed step is not numerical compatibility.
  Compare the immediate loss with the checkpoint-era loss.

## Inference and validity

- A special token that is merely rare is not safe. Every structurally illegal
  token must be masked for every architecture; full-distribution sampling finds
  any nonzero probability.
- Sentinels must never enter arithmetic. `NA_VAL = -9999` becoming a
  nanosecond delta produced three negative inter-arrivals in 780,864 and
  failed a whole LOB-Bench run's strict gate, correctly.
- Artifact counts do not validate generation. Check timestamp monotonicity,
  nanosecond range, finite values and illegal-token counts.
- Deterministic replay needs the whole batch: rank, slot, seed split history.
  Replaying one sample changes its slot key.
- A debugging flag that changes sampling or inputs (`overfit_debug`) is not an
  observation.
- The conditioning CSV has `N` message rows and `N+1` book rows because the
  book includes the initial snapshot. That is the contract, not a bug.

## Evaluation

- A low teacher-forced loss is a signal to evaluate, not a generative result.
  The held-out distribution score decides.
- A one-feature end-to-end LOB-Bench smoke proves connectivity. It is not
  WS-21, and must never be labelled as such.
- Freeze ticker scope, split, seeds, conditioning length, generation length,
  sample count and scorer commit before formal scoring. Protocol documents and
  live manifests drift.
- Score tables from the older canonical pipeline stay useful only while their
  repository, scorer commit and protocol are named beside them.
- The pooled WS-21 hides error compounding; the divergence curve shows it
  (`docs/benchmarking.md`).

## Process

- Validate the merge result, not the branch. Main was red once because a PR
  was validated on a branch cut before the failing test existed.
- Recorded evidence that cannot go stale silently is worth having; evidence
  without source hashes is not.
- `srun --overlap` removes queue latency and reserves nothing. Gate on
  physical GPU occupancy before starting, and know that the gate is not a lease.
- Never edit a shell script while a long-lived bash is executing it. Bash reads
  the modified inode and can parse a half-written file.
- A `state.json` that says `waiting` is not a live process. Check the process
  and its log.
- Commit titles are not commits. Read the diff before describing what a change
  does.
- When a request narrows after an interruption, finish the narrowed target and
  say what was abandoned.
