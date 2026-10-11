# Benchmarking

## LOB-Bench

Login-node smoke:

```bash
PYTHONPATH=src python run/benchmarking/run_lobbench.py --run-name smoke
```

Compute-node smoke:

```bash
sbatch run/benchmarking/smoke_lobbench.sbatch
```

### Where the scorer lives

`src/lobpipeline/lob_bench/scoring.py` (the adapter this repository wrote) builds a command for a driver at
`<benchmark root>/lob_bench/run_bench.py`, runs it, and records that scorer's
revision in every summary. Scoring runs under its own interpreter, built by
`envs/lobbench_venv_setup.sh`, because the scoring dependency set conflicts with
the training stack.

There are two roots to point it at.

| Root | What it is | Revision recorded |
|---|---|---|
| `DEFAULT_BENCHMARK_ROOT` | the FLAIROx `lob_pipeline` checkout on Lustre | that checkout's `HEAD` |
| `src/lobpipeline/scorer` | the same scorer, with Sat-A/lob_bench merged in | the merged tree's SHA-256 |

### Two directories are called `lob_bench`, and they are not the same thing

| Path | What it is | How it is used |
|---|---|---|
| `src/lobpipeline/lob_bench/` | the adapter this repository wrote | imported, as `lobpipeline.lob_bench.scoring` |
| `src/lobpipeline/scorer/lob_bench/` | the merged scorer | never imported; run as a program under its own interpreter |

The inner name is load-bearing rather than careless. The adapter locates the
driver as `<benchmark root>/lob_bench/run_bench.py`, and the external checkout has
that layout, so naming the in-repo tree the same way lets one code path serve both
roots. Renaming it would buy a clearer directory listing at the cost of a second
branch in the adapter for the in-repo case.

The default is unchanged. The in-repo copy exists because the default's recorded
revision does not resolve anywhere else: that checkout sits on `try-merge-daily`,
a branch with no upstream, so the hash in every summary written so far names a
scorer only to the machine that ran it. A tree hash can be recomputed from the
files it names.

### The merged scorer

`merge.json` records a three-way merge, not a copy: the common ancestor on
`peernagy/lob_bench`, both sides, and the SHA-256 of the result.

```bash
PYTHONPATH=src python -c "
from lobpipeline.scorer import merged_benchmark_root, read_merge_record
print(merged_benchmark_root())
r = read_merge_record('lob_bench')
print(r['ours']['commit'][:8], '+', r['theirs']['commit'][:8],
      'over', r['base']['commit'][:8])"
```

`merged_tree()` re-hashes on every call and refuses a tree that no longer matches
the record, so an edited file is a different tree rather than a silent local
patch.

**No production module moved.** Every `.py` the scorer runs is byte-identical to
the Lustre checkout's, so the merged tree can be run against numbers that are
already published. What the merge took from Sat-A is tests, the metrics-shard
SLURM array, two guides and a README. The four `*_levels` scores theirs still
defines were not taken: ours replaced them with BBO-relative `*_ticks` after the
2026-03-19 node restarts, when those four drifted about sixteen times as much as
the other seventeen.

### Two things in the merged tests

`tests/test_parallel.py` is worth running. It checks that the vectorised
bootstrap fast paths return the same losses and the same intervals as the
reference implementation, which nothing in this repository checked before:

```bash
cd src/lobpipeline/scorer/lob_bench
PYTHONPATH=. JAX_PLATFORMS=cpu python tests/test_parallel.py
```

`tests/test_contextual_scoring.py` is not a test. It has no asserts and no
failure path, and it prints a return structure of
`dict[metric: dict[regime_id: ...]]` that `compute_metrics_context` stopped
producing when it started weighting regimes by observation count and summing.
Read it as documentation of an earlier design.

### Scoring modes

The driver has five modes and sigma-0 drives all five.

| `--mode` | Driver flag | What it compares | Carries WS-21 |
|---|---|---|---|
| `uncond` (default) | `--unconditional` | 21 marginal distributions, plus the divergence curve below by default | yes |
| `cond` | `--conditional` | one score given another, averaged over the conditioning bins | no |
| `context` | `--context` | the same, but the bins come from the conditioning data | no |
| `time_lagged` | `--time_lagged` | a score at *t* against a score at *t - lag* | no |
| `divergence` | `--divergence` | each window of `--divergence-horizon` generated messages against the full real distribution | no (same 21 names, per window) |

```bash
PYTHONPATH=src python run/benchmarking/run_lobbench.py \
    --run-name my-run --inference-dir <dir with data_gen/ data_real/ data_cond/> \
    --output-dir artifacts/lobbench --stock GOOG --period 2026-01 \
    --mode context --run-id myrun
```

`ws21`, `ks21` and `l1_21` are filled in only by a complete `uncond` run. Every
other mode leaves them null and reports its own means under `distances`, because
WS-21 names a mean over 21 specific scores and nothing else may be published
under that name.

### The divergence curve is the headline number

A pooled score cannot show error compounding: it averages the first generated
window with the last one. The divergence curve keeps the time axis -- the
generated stream is cut into windows of `--divergence-horizon` messages
(default 100) and each window is scored against the full real distribution, so
index *k* of a curve is the distance for generated messages `(k*h, (k+1)*h]`.
A curve that rises with *k* is a model drifting away from the market as its
own output feeds back into its state, which is exactly what the pooled WS-21
hides. Measured here (mamba3 seed_42, GOOG, 3136 sequences of 250 messages,
horizon 50): spread KS rises 0.128 -> 0.494 across the five windows while the
pooled number reads 0.36.

Because that is the first thing to look at, a default `uncond` run now emits
the curve alongside the pooled scores -- same driver invocation, same loaded
data -- and `run_lobbench.py` prints the curve summary first. The curve lands
in `scores_div_<stock>_<run>_<horizon>_<run_id>.pkl` and is summarized to
`lobbench_divergence_summary.json`:

| Field | Meaning |
|---|---|
| `mean_curves` | per distance, the across-feature mean at each horizon window |
| `feature_curves` | per feature and distance, the raw per-window points |
| `final_over_first` | last window over first window of each mean curve; > 1 is compounding |
| `horizon_axis_messages` | right edge of each window, in generated messages |

A window too thin to score becomes `null` in the curve (and poisons the panel
mean at that index rather than shrinking the feature set quietly).
`--no-divergence` opts a run out; `--mode divergence` runs the curve alone.
The driver's `--div_error_bounds` companion file (`scores_div_..._REAL_...`)
is not wired: its point estimates are a structural zero (the "real vs real"
baseline compares one sample against a copy of itself, so only its bootstrap
spread carries information) and it covers l1 only.

For sequences of 250 generated messages, horizon 50 gives five points and
horizon 100 gives two; pick the horizon so the curve has enough windows to
show a trend and each window has enough messages to score.

### Do not read the contextual L1

In `--context` the driver overwrites the `group` column with the regime id and
then splits the frame by it, so each frame `l1_by_group` sees has a single group
and its histogram has one bin. The result is a structural zero: measured on
GOOG 2023 it reported 3.25e-05 while an L1 over the same frame's regime shares
was 0.1325.

A contextual L1 of exactly 1.0 for some regime is also not a measurement. It is
the guard that fires when one side of the comparison has no observations in that
regime.

Wasserstein and KS are unaffected; they read the `score` column. The fix belongs
in the scorer, not here.

### Which scores a mode produces

`uncond` and `divergence` share the pinned list `LOBBENCH_FEATURES` -- the
driver scores the same unconditional config for both, pooled or per window --
because WS-21 is defined over exactly those 21 names. The conditioned modes'
names are read out of the driver's source at plan time, so enabling another
entry upstream needs no change here:

```bash
PYTHONPATH=src python -c "
from lobpipeline.lob_bench.scoring import mode_features
print(mode_features('<benchmark root>', 'context'))"
```

A driver that stops declaring the pinned 21 fails the plan rather than quietly
publishing a WS-21 that averages a different set.

### Scoring one run as several shards

Shards split the score names, not the data, so a shard is a subset of the
columns of the same run. Score them, then merge:

```bash
# three shards of the 21 unconditional scores
sbatch --array=0-2 \
    --export=ALL,RUN_NAME=my-run,INFERENCE_DIR=<dir>,LOBBENCH_OUTPUT=<dir>,SHARD_COUNT=3 \
    run/benchmarking/lobbench_shard_array.batch

# merge once every shard is on disk
sbatch --dependency=afterok:<array job id> \
    --export=ALL,LOBBENCH_MERGE=1,RUN_NAME=my-run,INFERENCE_DIR=<dir>,LOBBENCH_OUTPUT=<dir>,SHARD_COUNT=3 \
    run/benchmarking/lobbench_shard_array.batch
```

Directly, without SLURM:

```bash
PYTHONPATH=src python run/benchmarking/run_lobbench.py ... --shard-index 0 --shard-count 3
PYTHONPATH=src python run/benchmarking/run_lobbench.py ... --merge --shard-count 3
```

The merge derives each shard's path from the same naming function that wrote it,
so a shard that never ran is a missing file. It also refuses two shards that
scored the same name, and refuses a merged set that is not the set that was
asked for. The driver itself only warns when asked for a score it does not have
and then writes the rest, so without these checks a typo would cost a short
pickle and a zero exit code.

A sharded run and an unsharded one produce the same point estimates: those are
computed on the full data before any resampling. The bootstrap intervals differ,
because the driver's random generator is a mutable default argument shared
across a process, so its state depends on how many scores that process computed
first. Compare `feature_scores`, not `ci`, when checking one against the other.

`model_zoo_lobbench_score_array.batch` is unchanged and still unsharded: its
array index is already architecture times seed, and it guards a published WS-21.

## The event-anchored return bench

What the mid-price does N milliseconds after a real trade executes, and whether
a model's generated order flow predicts it. Two departures from LOB-Bench: the
horizon is wall-clock time, not a message count, and the anchor is the instant
of execution, not a window boundary. Code: `src/lobpipeline/return_bench/`
(`event_anchored.py` is the metric, `run_event_bench.py` the CLI),
`run/base_model/eval/pick_trade_anchors.py` picks the anchors. The metric is
hashed into `ci/evidence/return_bench.json`; `ci/mutation_check.py` plants five
defects and the tests must catch all of them.

```bash
# 1. anchors (CPU, login node is fine)
python3 run/base_model/eval/pick_trade_anchors.py --data_dir <GOOG day directory> \
    --out <task dir>/anchors/goog_anchors.json \
    --n_cond_msgs 250 --n_gen_msgs 250 --n_samples 1024 --test_split 0.02 --seed 42
# 2. generate with the anchors (docs/inference.md), then
# 3. score
PYTHONPATH=src python3 -m lobpipeline.return_bench.run_event_bench \
    --infer_dir <inference dir> --horizons 1,5,10,25,50 --shuffle-null
# 4. self-check: real as generated must score exactly 1.0
PYTHONPATH=src python3 -m lobpipeline.return_bench.run_event_bench --infer_dir <any result dir> --identity-check
```

The finding that matters for reading its numbers: a rule that predicts the sign
from the side the anchoring trade hit beats every model on both direction
metrics, so direction there is bookkeeping, not prediction. The write-up is
`docs/experiments/return_bench_event_anchored/README.md`.

## The model-zoo protocol

The frozen comparison of the six backbones at about 8.1M parameters (GOOG
January 2026, corrected WS-21, checkpoints selected on December 2025 CE) is
`docs/experiments/model_zoo/protocol.md`. Its launchers are
`run/base_model/model_zoo_*.batch` and `run/benchmarking/model_zoo_*`.

## Inference trees are stored as one SquashFS image

One generation writes 25,092 files into `<evaluation>/inference/`. The project
quota on Lustre is a fixed number of inodes, so file count, not bytes, is what
runs out, and scoring then walks those files again with about 12,545 `glob`
calls against the metadata server. Both costs go away if the tree is stored as a
single image and read back through a read-only mount.

### Writing

`run/benchmarking/selftrain_checkpoint_generation.batch` defaults to
`LOBBENCH_PACK=1`. The four ranks write to node-local `/tmp`, the tree is packed
and verified there, and only two files reach Lustre:

```
<evaluation>/inference.sqfs          the image
<evaluation>/index_inference.json    per-file sha256, counts, families
```

`LOBBENCH_PACK=0` restores the previous behaviour byte for byte. `mksquashfs` is
not installed system-wide and there is no module for it, so it has to come from a
conda prefix. The launcher resolves it before generating, in this order, and
fails in seconds rather than after eleven hours if it cannot:

```
$SIGMA0_SQUASHFS_BIN  ->  PATH  ->  the prefix holding $PYTHON
```

For a launcher that must not be edited, such as a frozen runtime worktree,
`run/benchmarking/pack_gen_wrapper.sh` gets the same result without touching it,
by pointing `TASK_ROOT` at node-local storage and packing afterwards.

### Reading

Nothing on the read side changed. `run/benchmarking/run_lobbench.py` passes
`--inference-dir` through `lobpipeline.storage.sqfs.resolve`, which returns the
path unchanged when it is a directory and mounts `<path>.sqfs` when it is not.
Everything downstream, including the vendored scorer, still receives a plain
directory path.

Measured on one tree (Slurm job 6423589): mounting costs 4.7 ms, and
`inventory_inference` takes 0.818 s from the image against 5.014 s from the
Lustre directory, with identical counts.

Two consequences worth knowing:

- The mount lives inside the `with` block, at a fresh path each run, so the
  three `bench_data/<run>/<stock>/<period>/data_*` symlinks that
  `stage_inference` creates dangle afterwards. The next run replaces them;
  `_replace_directory_link` redirects a dangling or a differently-pointed
  symlink and only refuses a real directory. `tests/unit/test_sqfs.py` pins this.
- `/etc/fuse.conf` has no `user_allow_other`, so a mount is visible only to the
  user who made it. The scorer runs as a child of the same user, which is
  enough. Scoring across accounts would not work.

### Verification is not optional

`mksquashfs` has silently truncated files in this project before
(`src/base_model/data/lobster_dataloader.py`), and LOB-Bench parses with
`pandas.read_csv(on_bad_lines="skip")` and
`polars.read_csv(ignore_errors=True, truncate_ragged_lines=True)`. A truncated
tail is absorbed into a shorter DataFrame with nothing raised, which would move
a score silently. Equal file counts do not detect it. `sqfs.verify` therefore
mounts the image and compares every file by sha256, and there is no flag to skip
it.

The sidecar also records the sha256 of the whole image, which is a different
check rather than a duplicate one: images are padded to a 4 KiB boundary, so a
tail truncation can remove only padding and every file still reads correctly.
`sqfs.verify_sidecar` checks both and needs no source tree, so an image can be
re-verified long after the tree it came from is gone.

### Changing a few records without rebuilding

An image may have a sibling `<name>.overlay/` directory. `resolve` layers it over
the image with `fuse-overlayfs`, so a file placed there replaces the one in the
image and a whiteout hides it:

```python
from lobpipeline.storage import sqfs
sqfs.overlay_override(inference_dir, "data_gen/GOOG_..._real_id_7_gen_id_0.csv", data)
sqfs.overlay_exclude(inference_dir, "data_gen/GOOG_..._real_id_9_gen_id_0.csv")
```

An excluded record disappears from `glob`, which is how `Simple_Loader` finds
files, so it stops existing for scoring. A symlink farm would also work and is
deliberately not used: it has previously exhausted node-local inodes, and a farm
built inside an `if` condition can come out silently incomplete because bash
suppresses `errexit` there.

### Packing trees that already exist

```bash
python run/benchmarking/pack_inference.py --dry-run <evaluation-dir>
python run/benchmarking/pack_inference.py <evaluation-dir> [...]
```

It packs, verifies every file, then renames the source aside as
`inference_deprecated_<UTC>`. It does not delete, and renaming does not release
an inode, so this command does not lower the quota reading by itself. What it
produces is an image that can be scored and restored, so that a later deletion,
which is a human decision, does not lose anything.
