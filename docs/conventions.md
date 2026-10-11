# Platform conventions

Rules that apply to every run out of this repository, and to every fork of it.
Each one is here because it has cost the team something at least once; the
detail lives in the other pages under `docs/` (`data.md`, `training.md`, `checkpoints.md`, `ci-and-evidence.md`, `shared-checkout.md`).

## Lustre / HPC safety

These follow the 2026-05-08 Isambard incident, in which the team's jobs were
suspended for metadata storms on the Lustre MDTs. The Acceptable Use Policy
attributes every command to the human account, including commands issued by an
agent, so "the tooling did it" is not a defence.

- No `find`, recursive `ls`, `du`, `tree`, `locate`, or large globs on `/lus`
  or `/projects`. Recursive lookups go through `lfs find <narrow_path>`, which
  batches MDT requests.
- Known exact paths only: `cat` / `head` / single-file `grep` are fine;
  `grep -r` only on small trees or with `--include`.
- Discovery ceiling in code: one `os.scandir` per known directory. Never
  enumerate a checkpoint directory to find out which step is newest.
- One output directory per job (`${SLURM_JOB_ID}` in the path); pre-stage model
  weights and hot data to `$TMPDIR` before the loader starts; loggers
  (wandb / tensorboard) write node-local and sync at end of run.
- Stagger submissions (~10 per wave, `sleep 30+` between waves); always pass an
  explicit `--nodes` and verify `%D` on the queued job.
- `/home` is VAST NFS, not Lustre, scoped `find` / `du` are fine *there*, but
  the quota is small: check `quota -s` before big writes. A full quota kills
  jobs at startup with `RaisedSignal:53` and an empty log, which looks like
  anything except a quota problem.
- No persistent agents and no auto-restart wrappers (systemd-user, cron,
  watchdog) on login or compute nodes.

Login nodes: editing, submitting, `squeue`/`sacct`/`sinfo`, light scripting.
Not: compiling, inference, preprocessing, batch grep over large files.

## Checkpoints

- Orbax layout:
  `<CKPT_BASE>/j<JOBID>_<WANDB>_<JOBID>/<step>/{_CHECKPOINT_METADATA, metadata/, state/}`.
  The per-step `metadata/metadata` JSON carries the full training config, and is
  the ground truth for what a checkpoint actually is.
- **Resolve a checkpoint by explicit step, never by enumeration.** Training
  resume takes `RESUME_FROM_STEP`; the pinned world models name their step and
  a SHA-256 over `<step>/metadata/metadata`
  (`src/post_training/heuristic_learning/checkpoints.py`,
  `tests/unit/test_agentic_mm_checkpoints.py`).
  `ls -t`, `ls | sort`, and step-directory globs are the metadata-storm pattern
  that got the team suspended, and "newest" is in any case not the same claim as
  "the one this result was measured on".
- Write checkpoints and training logs node-local first when the cadence is
  high; rsync to Lustre at save points. Checkpoint interval >= 500 steps.

## Group permissions

- Everything under shared paths (`/lus/...`, `/projects/...`) must end up
  group-readable and group-writable: `umask 002` (set by `setup_env.sh`) plus a
  reflexive `chmod g+rw <file>` / `chmod -R g+rwX <dir>` after writing.
- In-place tools (`sed -i`, `tee`, `>` redirects) materialise a temp file under
  a root-side umask and rename it over the target, silently stripping the
  group bits. Re-`chmod` after them; treat it as part of the write.
- Group on this account family is `brics.u6gb`; the parent `public/u6gb`
  directory is setgid, so new files inherit it.

## Reproducibility

- Record provenance where the thing lives: source checkout and date for
  vendored or copied code, job IDs for results, seeds and date ranges for runs.
- Engine semantics are frozen against sealed results. A behavioural change to
  matching or order lookup in `src/matching_engine/` is an explicit, documented
  experiment delta, never a silent platform edit. The defect register
  (`docs/defect_register.json`, rendered by `tools/render_register.py`) is where
  such a change gets stated and evidenced.
- Numbers quoted in docs carry the job ID they came from, and say whether they
  are a single day, a panel, or a full sweep.
