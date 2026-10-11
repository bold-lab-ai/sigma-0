# AGENTS.md: how to work in sigma-0

This file is for any agent (Claude, Codex, or a person) that edits, runs, or
submits work from this repository. Read it fully before the first command.
`CLAUDE.md` imports it. The detailed platform rules are in `docs/conventions.md`;
this file is the short list of what keeps the repository clean and the cluster
safe.

## 1. What this repository is

sigma-0 is the team's single code base for limit-order-book sequence models:
the matching engine (`src/matching_engine`), the RL environments
(`src/environment`), the base model with its encodings, data, training and
inference (`src/base_model`), the LOB-Bench and return-bench pipeline
(`src/lobpipeline`), and post-training (`src/post_training`, `src/state`).
Launchers live in `run/`,
configs in `configs/`, tests in `tests/`, CI checks in `ci/`, documentation in
`docs/` (index: `docs/README.md`).

The repository holds code, configs, launchers, tests and documentation. It does
not hold results. Results go to the results database (section 4).

## 2. Where to work

- Never edit, commit, or run anything inside the shared checkout at
  `/lus/lfs1aip2/projects/public/u6gb/sigma-0`. It belongs to another person's
  agent and its `main` is stale.
- Work in your own clone or worktree. Name a worktree after its branch and
  remove it when the PR merges or is closed. A worktree is not an archive.
- Start every shell with `source setup_env.sh` (sets `PYTHONPATH=src` and
  `umask 002`). Run `bash bootstrap_fork.sh` once per clone.

## 3. What may be committed

Commit source code, tests, launchers (`run/`), configs, CI checks, and
documentation. Nothing else. In particular, never commit:

- checkpoints, logs, Slurm `.out`/`.err` files, wandb directories, `__pycache__`
- result files: JSON metrics, CSV tables, PNG or PDF figures, HTML exports,
  generated notebooks (one exception: the notebook inputs under
  `docs/experiments/trainloss_profiling_20260826/data/`, which two guards hash and read)
- agent state: plans, progress logs, findings, handoffs, `events.jsonl`,
  `state.json`, session summaries. Put durable knowledge in `docs/`; put
  results in the results database; keep the rest out of git
- scratch scripts (`_probe.py`, `tmp_*.sh`, one-off diagnostics), screenshots,
  files whose names contain spaces or shell fragments
- new files at the repository root. The root holds `README.md`, `AGENTS.md`,
  `CLAUDE.md`, `pyproject.toml`, `setup_env.sh`, `bootstrap_fork.sh`,
  `.gitignore`, and the tracked symlinks. Anything else goes in a subdirectory
- personal absolute paths (`/home/<you>/...`, `.../sigma-0-worktrees/<branch>/...`).
  Launchers take `WORKDIR` and data roots from the environment or a config

If a file matches `.gitignore`, do not force-add it. Fix the reason instead.

## 4. Results and outputs

- Job output (checkpoints, logs, generated data) goes to a directory outside the
  repository, keyed by `${SLURM_JOB_ID}` or an experiment tag, node-local first
  and synced to Lustre at the end of the run (`docs/conventions.md`).
- Every finished result, including negative ones, is filed in the results
  database at `/lus/lfs1aip2/projects/public/u6gb/results`. Its `AGENTS.md`
  explains the format. One command:
  ```bash
  /lus/lfs1aip2/projects/public/u6gb/results/_system/add_result.sh <files> --prompt "<what, job id, finding>"
  ```
- A PR is not a results page. A PR body says what changed in the code and how
  it was verified. Numbers that support the change go in the results database
  and are linked from the PR.

## 5. Branches, commits, pull requests

- Branch from `origin/main`. Name it `feat/<topic>-<yyyymmdd>` (or `fix/`,
  `docs/`, `chore/`). One topic per branch.
- The PR base is `main`. Do not open a PR against another open PR.
- Rebase on `main` before asking for review and after `main` moves.
- Commit when a unit of work compiles, its tests pass, and it makes sense on its
  own. Do not commit every edit. Message format: `type(scope): imperative
  summary`, types `feat fix refactor config data exp bench docs chore ci`.
- PR body, in this order: what changed, how it was verified (the commands and
  their outcome), what is deliberately not done. Short sentences. No status
  updates, no progress tables, no colour-coded logs. Do not rewrite the body
  after review has started; add a comment instead.
- Ask for review from a person. Agents do not merge their own PRs and do not
  enable auto-merge.
- Keep PRs small enough to read in one sitting. A code PR that adds more than
  about two thousand lines is a sign that outputs or a vendored tree slipped in.

## 6. Tests and CI

Before pushing, run from the repository root:

```bash
PYTHONPATH=src python -m compileall -q src tests run tools ci
PYTHONPATH=src python run/migrate/check_imports.py --tiers core
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests -q -p no:cacheprovider
python3 ci/check_evidence.py
python3 ci/check_english_only.py --list
python3 ci/check_lobbench_merge.py
python3 ci/check_state_posttraining.py
python3 ci/check_ce_decomposition.py
python3 ci/check_message_perplexity.py
python3 ci/check_legacy_imports.py
bash -n <every .sh, .batch, .sbatch you touched>
```

The evidence rule: `ci/evidence/*.json` record a SHA-256 of every source file a
measured number depends on. If you edit one of those files, the audit fails and
names it. The fix is to re-run `ci/record_evidence.py` on the cluster and commit
the new record, never to edit a hash by hand. Tests are hashed too: after any
change under `tests/`, re-record with `--suite`.

Do not delete or weaken an existing test to make a PR green. Lowering the test
count floor needs a reason in the PR body. All text in the repository is
English (`ci/check_english_only.py`).

## 7. Cluster conduct (the rules that got the team suspended once)

Full list in `docs/conventions.md`. The ones agents break most:

- No `find`, recursive `ls`, `du`, `tree`, `locate`, or large globs on `/lus`
  or `/projects`. Read exact paths. If you must recurse, use
  `lfs find <narrow path> -maxdepth N`. If you do not know what a directory
  holds, ask rather than list it.
- Never locate the newest checkpoint by listing a directory. Read the pinned
  step from the config, the manifest, or `latest_checkpoint.json`.
- Every file you write on a shared path ends up `group=brics.u6gb`, `g+rw`
  (`umask 002`, then `chmod g+rw` after `sed -i`, `tee`, or `>`).
- Submit jobs one at a time or in waves of about ten with `sleep 30` between
  waves. Always pass `--nodes` explicitly. Check `squeue` before and after.
- No persistent agent, watcher, `tmux` loop, `nohup`, cron, or auto-restart
  wrapper on a login or compute node. A script that polls a log every few
  seconds is a watcher. If something was stopped, it was stopped on purpose.
- Login nodes are for editing, submitting, `squeue`, `sacct`, and light
  scripting. Compiling, inference, preprocessing and batch grep belong in a job.
- The Acceptable Use Policy attributes every command an agent runs to the human
  account. There is no "the agent did it".

## 8. Source layout rules

- New code goes into an existing package under `src/`. Do not add new
  top-level modules or packages without a PR that says why.
- New model, encoding, data, training or inference code goes under
  `src/base_model`. The retired `s5` and `lob` packages are gone;
  `ci/check_legacy_imports.py` rejects any import of them.
- The vendored LOB-Bench scorer under `src/lobpipeline/scorer/lob_bench` is
  edited only with a matching update to `src/lobpipeline/scorer/merge.json`.
- Engine semantics in `src/matching_engine` are frozen against sealed results.
  A behavioural change is an explicit, documented experiment, recorded in
  `docs/defect_register.json`.
- Placeholder modules, `.gitkeep` files in non-empty directories, and empty
  packages are not committed. Create the code when it exists.

## 9. Documentation

- `docs/` holds reference material that stays true: `conventions.md`,
  `architecture.md`, `environments.md`, `data.md`, `training.md`,
  `inference.md`, `benchmarking.md`, `checkpoints.md`, `ci-and-evidence.md`,
  `shared-checkout.md`, `lessons.md`. Experiment write-ups live under
  `docs/experiments/<slug>/` and are listed in `docs/experiments/README.md`.
  Nothing in `docs/` is a dated status report or a per-session log.
- When you change behaviour, update the page that describes it in the same PR.
- Write short sentences. No em dashes. Every number carries the job ID or
  commit it came from.
