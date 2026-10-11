# Agentic Market Making — Handoff

Written 2026-07-31. Everything below is an absolute path you can paste directly.

---

## 1. What this is

We are building a closed loop where a **rule-based market-making policy** quotes at
the touch, a **frozen world model** supplies the background order flow and a
forward mid-price curve, a **deterministic matching engine** settles the fills,
and a **headless LLM call rewrites the policy** from the resulting PnL.

Four parts: (1) the rule-based policy quotes at the touch → (2) the frozen world model
generates background order flow and a forward mid curve → (3) the deterministic matching
engine settles the fills → (4) `claude -p` reads the PnL and rewrites the policy code.

**The one rule you must not break.** The world model learns an *observational*
distribution `p(market)`; market making is an *intervention*,
`p(market | do(my quotes))`. So the generator must **never** produce the fills of
your own orders — the matching engine settles those. Otherwise adverse selection
is under-counted and simulated PnL is systematically optimistic.

---

## 2. Where everything is

| What | Path |
|---|---|
| Repo (GitHub) | `git@github.com:KangOxford/sigma-0.git` |
| Shared checkout | `/lus/lfs1aip2/projects/public/u6gb/sigma-0` |
| **Active worktree** | `/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/agentic-mm-runtime-20260731` |
| Branch | `feat/agentic-mm-stage-a` (base `origin/main`) |
| Python | `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python` |

### Code you will touch

All paths relative to the worktree above.

| Path | Role |
|---|---|
| `src/post_training/heuristic_learning/checkpoints.py` | content-hash pinning and validation for both world-model checkpoints |
| `src/post_training/heuristic_learning/episode_builder.py` | CSV → engine messages → episode, with a clock-monotonicity gate and a replay consistency self-check |
| `run/environment/build_episodes.sbatch` | **Stage A**: generate the frozen market |
| `run/environment/preflight_claude_p.sbatch` | compute-node `claude -p` connectivity preflight |
| `run/environment/pr_autopilot.sh` | full PR flow: open → validate the real merge result → comment → merge only when green |
| `src/lob/inference_no_errcorr.py` | the generation stack, including `replay_real_msgs_single` and `compute_gt_divergence` |
| `run/base_model/runtime/inference.py` | the entry point that actually runs autoregressive generation |
| `tests/unit/test_agentic_mm_checkpoints.py`, `test_gt_divergence.py` | 14 tests |

### Data (read-only, verified readable)

| What | Path |
|---|---|
| L10 message + book (SP500, monthly shards) | `/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/shard_2026-01.squashfs` |
| That shard's index (**readable without mounting**) | `.../lob_preproc_sp500_squashfs/index_2026-01.json` |
| L500 wide book (one shard per ticker) | `/lus/lfs1aip2/projects/public/s5e/quant_team/recon_2026-05/output/squashfs/shard_<TICKER>_L500_npy_2026-01.squashfs` |
| Existing generation output (the P0 fixture, 3136 sequences) | `/lus/lfs1aip2/projects/public/u6gb/s5e_alphatrade/lob_pipeline/inference_results/j4559297_290M_step168200_L500_jan2026_GOOG_4606760/` |

⚠️ The `dir_name=/lus/lfs1aip2/projects/s5e/lob_preproc` recorded in the training
metadata is **Permission denied** for the u6gb account. Do not use it; use the two
squashfs paths above.

### Checkpoints (two of them, deliberately not the latest step)

Root: `/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/checkpoints/`

| Role | Directory | step | Why this step |
|---|---|---|---|
| background order flow | `j4559297_u52a0g05_4559297` | **150360** | best LOBbench fit (mean KS 0.0835), on the train-loss plateau. The newer 168200 is about 3% worse |
| price-curve signal | `j4553948_tz2tmn5z_4553948` | **120000** | highest direction accuracy (0.5523). The newest is 143150 |

**Do not "just update to the latest checkpoint."** Ranking by KS and ranking by
direction accuracy do not agree; these two steps were chosen separately.
`tests/unit/test_agentic_mm_checkpoints.py` will fail to remind you.

### Outputs

`/lus/lfs1aip2/projects/public/u6gb/tasks/agentic_mm_<UTC_TIMESTAMP>/`
(a fresh timestamped directory per run, never overwritten; the current one is
`agentic_mm_20260731T164839Z`)

---

## 3. How to check what is running

### Environment setup (needed in every new shell)

```bash
export WT=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/agentic-mm-runtime-20260731
export PY=/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python
```

### 3.1 The SLURM layer

```bash
squeue -u kangli.u6gb -o "%.10i %.28j %.9T %.12M %.12L %.5D %N"
```

⚠️ **`squeue` tells you who owns an allocation, not whether anything inside it is
burning GPU.** A `RUNNING` `u6gb-*-node-chain` may be entirely idle, or packed by
someone else. Use 3.2 to decide.

### 3.2 Whether the GPUs are actually free (**the only basis for starting work**)

```bash
# Critical: strip the inherited SLURM_* first, or srun uses your current shell's
# allocation and reports "Only allocated 1 nodes asked for 4"
( for v in $(compgen -v SLURM_); do unset "$v"; done
  srun --jobid=<ALLOCATION_ID> --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpu-bind=none \
    bash -c 'echo "$(hostname) | $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr "\n" " ") | PIDs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"' )
```

Reading it: `PIDs=0` with 1–4 MiB of memory means genuinely free. Any PID means someone
is running; **do not touch it and do not squeeze in beside it**.

### 3.3 How far Stage A has got

```bash
LOG=$WT/run/environment/episode_logs/attached2_5836919.out   # the current run's log
grep -c '^\[stageA\] --- ' "$LOG"     # generation phases started, 9 in total
tail -20 "$LOG"
ls -l $WT/run/environment/episode_logs/*rank*.log            # per-rank progress
```

⚠️ **"Still alive" cannot be judged by the absence of errors in the log.** A killed
process writes neither `DONE` nor an error; it simply **stops writing**. The two correct
signals are:

```bash
# (a) are the rank logs' mtimes still advancing? Stalled for over 15 minutes means dead
stat -c '%n %s %y' $WT/run/environment/episode_logs/*rank*.log

# (b) is there a compute PID of ours on the GPU (see 3.2)
```

### 3.4 Always check the exit code after a job ends

```bash
sacct -j <JOBID> --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList -P
```

Only `0:0` is success. **"Not in squeue" does not mean "succeeded."** When it is not
`0:0`, scan every rank log for the first error.

---

## 4. How to run it

### Option A: ordinary queueing (recommended, least trouble)

```bash
cd $WT
squeue -u kangli.u6gb -o "%.10i %.28j %.9T"      # pre-submit dedup check; this is a hard rule
sbatch run/environment/build_episodes.sbatch
```

### Option B: attach to an existing allocation (saves queue time, with conditions)

Only do this when 3.2 confirms `PIDs=0` on the target node.

```bash
( for v in $(compgen -v SLURM_); do unset "$v"; done
  srun --jobid=<ALLOCATION_ID> --overlap --nodes=1 --ntasks=1 --nodelist=<NODE> \
       --gpus-per-node=4 --cpu-bind=none \
       bash $WT/run/environment/build_episodes.sbatch > $WT/run/environment/episode_logs/attached.out 2>&1 ) &
```

⚠️ **Do not start a background process with `tmux new-session -d` from inside an srun
step** — the tmux server is killed along with that step and the job dies silently within
seconds. **Let the script itself be the srun step** (as above); the job then lives
exactly as long as the step does.

⚠️ Two copies must not run together: once the attached one is confirmed alive (both
signals in 3.3 pass), cancel the queued one and verify with both `squeue` and `sacct`.

### Tunable parameters (all environment variables, all with defaults)

```bash
STOCK=GS MONTHS=2026-01 N_SEQUENCES=8 BATCH_SIZE=2 \
N_COND_MSGS=500 N_GEN_MSGS=900 BG_SEED=2026 \
SIGNAL_SEEDS="3001 3002 3003 3004 3005 3006 3007 3008" \
sbatch run/environment/build_episodes.sbatch
```

**Cost out a ticker before switching to it.** A one-minute window means a minute of
message flow, and that is a per-ticker constant:

| ticker | msg/s | messages per minute |
|---|---|---|
| GOOGL | 540.6 | 32,436 |
| GOOG | 241.5 | 14,488 |
| **GS (currently used)** | ~13 | **~800** |
| median (e.g. KEY) | 7.7 | 463 |
| quietest (NVR) | 1.3 | 77 |

How to compute it (no squashfs mount needed):

```bash
$PY -c "
import json, collections
sh=json.load(open('/lus/lfs1aip2/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/index_2026-01.json'))['shapes']
r=collections.defaultdict(list)
for k,v in sh.items():
    if '_message_' in k: r[k.split('/')[0]].append(v['shape'][0]/23400.0)
for t in ['GS','GOOG','KEY']:
    m=sum(r[t])/len(r[t]); print(f'{t}: {m:.1f} msg/s -> {m*60:.0f} msgs per minute')
"
```

A ticker also has to satisfy two conditions: (a) it is inside the training universe —
both checkpoints trained on the full SP500 (~490 tickers, per the metadata's `tickers`
field), so essentially all are; and (b) it has a corresponding L500 wide book shard
(§2). Prefer high-priced names, because **their spread is wider in ticks**, which is
what gives at-touch quoting something to capture.

### Running the tests

```bash
cd $WT && JAX_PLATFORMS=cpu PYTHONPATH=src $PY -m pytest tests -q
# expected: 102 passed, 7 skipped
```

---

## 5. What is done versus what is next

### Completed and merged into `main`

| PR | Contents | State |
|---|---|---|
| #1 | lands the real Mamba3 generation stack (`src/lob`, `src/s5`) plus two inference fixes | MERGED |
| #2 | closed-loop market-making foundations: checkpoint pinning, sim-vs-real replay comparison, `claude -p` preflight, PR autopilot | MERGED |
| #3 | seven training configs switched to W&B online, entity `oxford-lob` | MERGED |

`main = 74728b0`. Full `pytest`: 102 passed, 7 skipped. Headless `claude -p` is verified
to work from a compute node (job 5849614, `COMPLETED 0:0`, 10 seconds).

### In progress

**Stage A** — generating the frozen market. Attached to allocation 5836919 on nid010138,
writing to `/lus/lfs1aip2/projects/public/u6gb/tasks/agentic_mm_20260731T164839Z/`.
Nine generation phases (one background flow plus eight signal seeds).

### The gate that has not passed (**whoever picks this up needs to know**)

P0 replay consistency: only 1 of 3 sequences reproduces the generator's own mid path
exactly (`exact_mid` = 0.832 / 0.992 / 1.000).

- The initial state is **verified correct**: `cond_book[-1] == gen_book[0] ==` the
  post-reset state, value for value
- The first divergence is at step 97, on `DELETE oid=814600586 qty=10`, where 14 shares
  land one price level away
- Two hypotheses were **refuted**, not worked around: replaying the conditioning messages
  made things worse (0.664 vs 0.832), and engine capacity `nOrders` 100 versus 1050 is
  bit-identical
- Root cause: `init_msgs_from_l2` collapses each level of an L2 snapshot into one
  synthetic order (`INITID=-2`), so a delete carrying a real order id from the
  conditioning window finds no target and the engine removes quantity from elsewhere
- **This is a limitation of the archived fixture** (it stored L10 books only), not a
  defect in the new code. The generator itself runs on L500 plus a conditioning replay.
  **Re-judge this gate once Stage A's own L500 conditioning book exists.**

### Next steps, in order

1. Wait for Stage A's nine phases → re-judge the P0 gate with the L500 conditioning book
2. `mm_sim.py` / `mm_accounting.py`: closed-loop replay, step-wise matching, end-of-window
   inventory marked to market
3. `policy_api.py`: the policy contract, an AST allowlist, and a timed subprocess
   (**this auto-executes LLM output, so the gate is not optional**)
4. `opro_driver.py`: meta-prompt → `claude -p` → extract code → evaluate → append to history
5. A 5-round OPRO smoke; replay the same policies against the real order flow to produce
   a `Δ_sim` vs `Δ_real` comparison

⚠️ **Stated in advance**: eight windows **will not pass a 2σ significance threshold**,
and that is expected. What the smoke has to demonstrate is that the loop turns and that
the score genuinely feeds back into the code — **not that the policy works**. Any
significance claim waits for the full multi-init × multi-seed × per-regime evaluation.

---

## 6. Traps that will cost you hours

| # | Trap | How it shows up | What to do |
|---|---|---|---|
| 1 | attaching again from an already-attached shell | `srun: error: Only allocated 1 nodes asked for 4` | run `for v in $(compgen -v SLURM_); do unset $v; done` first. `--jobid` does not override an inherited `SLURM_NNODES` |
| 2 | `tmux new-session -d` from inside an srun step | the job dies silently within seconds and the log freezes | let the script itself be the srun step |
| 3 | judging liveness by the absence of errors | a monitor runs for an hour without alerting on a process that died long ago | check the rank logs' **mtime** plus the GPU **compute PID** |
| 4 | `--pty` with multiple nodes | it only lands on the first node | `--pty` supports a single task; use non-interactive for multi-node, or one terminal per `--nodelist` |
| 5 | `--allowed-tools ""` to disable tools | **no effect**; the tool list comes back full | use `--disallowed-tools "Bash" "Write" "Edit" ...` |
| 6 | re-applying `(d*2)-1` when reading messages back from CSV | direction becomes {-3, 1}; the engine **does not complain** and matches the wrong side silently | direction in the CSV is already {-1,1}; use it as is |
| 7 | inferring the training data from `dir_name` | you conclude only GOOG was trained on and think switching tickers leaves the distribution | read the metadata's `tickers` field (it is the full SP500) |
| 8 | changing `nOrders` | every LOBbench `_levels` metric shifts silently | it is pinned for reproducibility; leave it alone |
| 9 | `ls` on a large Lustre directory, `find`, `du -sh` | a metadata storm; the whole group's jobs can be suspended by the administrators | use exact paths, `lfs find`, or `os.scandir` with an early break |
| 10 | branch prefixes in the shared repo | `refs/heads/fix/` belongs to someone else and is not group-writable, so `fix/xxx` cannot be created | use `feat/xxx`, or confirm directory ownership first |
| 11 | local `main` will not update | `refusing to fetch into branch 'main' checked out at ...` | the main checkout holds it; cut branches from `origin/main` instead. Fetch with `git fetch --no-write-fetch-head` (`.git/FETCH_HEAD` belongs to someone else) |

---

## 7. Workflow conventions (team rules, not suggestions)

- Work on a **separate branch** in a worktree → open a **PR whose body states everything**,
  including the gates that did not pass → update locally once the remote `main` merges →
  finally merge into `main`
- Run the `squeue` dedup check **before** every `sbatch` and paste the result
- Monitor **after** every `sbatch` at the 1/5/15/30-minute checkpoints; when a job
  disappears, check the `sacct` exit code; when it is not `0:0`, scan **every** rank log
  for the first error
- Every experiment uses `USE_WANDB=True` with `WANDB_MODE=online` and entity
  `oxford-lob`, smoke tests included
- Data output always goes to a **new timestamped directory**, never overwriting; write
  atomically (`.tmp` → rename); resume skips what already exists
- Checkpoints and training logs go to `$TMPDIR` (node-local) and are `rsync`'d back to
  Lustre once at the end
- **Delete nothing** unless the user names the path and confirms

## 8. Reference documents

| Document | Contents |
|---|---|
| `docs/agentic_mm_progress_20260731.md` | the detailed progress report (9 sections, with the full reasoning and a risk register) |
| Notion "agentic trading" | problem formulation and per-round progress callouts |
| `/lus/lfs1aip2/projects/public/u6gb/{plans,findings,progress,learnt_lessons}.md` | per-round records (timestamped ids) |
