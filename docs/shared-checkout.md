# Working on the cluster: the shared checkout, allocations, merging

Each item here cost somebody hours. None is discoverable from an error message.

## 1. The shared checkout is shared

`/lus/lfs1aip2/projects/public/u6gb/sigma-0` has `.git` files owned by several
accounts. Do not work in it (`AGENTS.md`). If you must read from it:

| symptom | cause | what to do |
|---|---|---|
| `git fetch` fails with Permission denied on `.git/FETCH_HEAD` | the file belongs to someone else | `git fetch --no-write-fetch-head` |
| `refusing to fetch into branch 'main' checked out at ...` | `main` is checked out there; this is protection, not permissions | cut branches from `origin/main` in your own clone |
| `git checkout -b fix/...` fails with Permission denied | `.git/refs/heads/fix/` belongs to another user and is not group-writable | use a prefix you own, or your own clone |
| `insufficient permission for adding an object to repository database`, intermittently | eleven `.git/objects/xx` directories were created without group write; about 4% of new objects land in one | the August 2026 fix rebuilt them (`mv xx xx.orig; mkdir xx; cp -p`); if it recurs, repeat it and run `git fsck` |
| `git merge-tree` reports a conflict you cannot explain | it could not write its result objects (same cause) | fix the object directories first |

Group-writable is the rule: `umask 002`, `chmod g+rw` after in-place edits
(`sed -i`, `tee`, `>` create a temp file under a private umask and rename it
over the target).

## 2. Slurm traps

| symptom | cause | fix |
|---|---|---|
| `srun: error: Only allocated 1 nodes asked for 4` | the shell inherits `SLURM_*` from its own allocation; `--jobid` does not override it | `for v in $(compgen -v SLURM_); do unset $v; done` before `srun` |
| a job dies about 13 s after launch, log frozen, no error | `tmux new-session -d` started inside an `srun` step is killed with the step | make the script itself the srun step |
| a monitor reports progress for an hour on a dead process | it grepped for `DONE` or an error string; a killed process writes neither | alarm on log mtime staleness and on GPU compute PIDs |
| "these nodes are empty" and they were not | `squeue` reports allocation ownership, not GPU occupancy | attach and count compute PIDs (`nvidia-smi --query-compute-apps=pid --format=csv,noheader`) |
| `--pty` lands on one node | `--pty` requires exactly one task | multi-node must be non-interactive |
| `taskset: failed to set pid affinity: Invalid argument` inside a test | a partial-node allocation's cgroup does not contain CPUs 0-3, which `tests/integration/test_trade_anchored_windows.py` pins | run the suite on `--exclusive` nodes |
| `RESOURCE_EXHAUSTED` at the first train step of `mamba3_smoke.yaml` | 78M parameters, 13,000 tokens, 10 samples per GPU does not fit a GH200 on the pure-JAX path | `PER_GPU_BSZ=1` for a smoke (jobs 6347534 and 6347641, 2026-09-05) |
| `Backend 'cuda' is not in the list of known backends` from a DataLoader worker | a forked worker initialising JAX CUDA | `N_DATA_WORKERS=0` for smokes |
| a first-tier `%j` output directory does not exist at submit time | `#SBATCH --output` cannot create directories | write to an existing log directory; create per-job directories inside the script |
| `/local/user` scratch vanishes 30 to 40 s into a workq job | node-local scratch policy since June 2026 | `TMPDIR=/tmp`, set before any Python import |
| `Error writing persistent compilation cache` warnings | the shared JAX cache directory is over quota | harmless for correctness; slower compiles |

## 3. Attaching to an allocation you already hold

```bash
( for v in $(compgen -v SLURM_); do unset $v; done
  srun --jobid=<ALLOC> --overlap --nodes=1 --ntasks=1 --cpu-bind=none bash -c 'hostname; nproc' )
```

Before starting GPU work on it: `nvidia-smi --query-compute-apps=pid
--format=csv,noheader | wc -l` must print `0` on that node. `--overlap` does
not reserve HBM; a step that starts later can still land on the same GPUs.
`tools/dev/attach_run.sh <jobid> <node> <script> [VAR=VAL ...]` does the
attach correctly: absolute interpreter, variables through `--export`, the
payload script is the step.

## 4. Merging a branch

Validate the merge result, not the branch. A branch green on its own has been
broken by what it merged into.

```bash
git worktree add --detach /path/to/_merge_check origin/main
git -C /path/to/_merge_check merge --no-ff --no-edit <branch>
# then, on a compute node:
cd /path/to/_merge_check && JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests -q
```

`tools/dev/pr_autopilot.sh open|verify|merge|drive` scripts that flow. Merging
stays a human action (`AGENTS.md` section 5).

## 5. Worktrees

One worktree per branch, named after it, outside the repository directory,
removed when the PR closes. A worktree of sigma-0 holds about 450 tracked
files; the u6gb project hit its inode limit on 2026-09-05 with 154 of them
alive.
