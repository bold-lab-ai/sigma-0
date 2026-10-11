# Data

Formats, which loader consumes which, and the hard rules a launcher must
satisfy before `sbatch`. Verified against the live filesystem 2026-07-15
(symlink targets re-checked 2026-08-15). Companion files:
`tools/data_symlinks.tsv` (canonical symlink table),
`$BENCHMARK_ROOT/pipeline/_squashfs_helpers.sh` in the FLAIROx `lob_pipeline` checkout (mount helpers; `run/benchmarking/squashfs_helpers_no_delete.sh` wraps them with an unmount-only cleanup),
`tools/data_build/make_snapshots.py` (wide-book snapshot builder).

---

## 1. Data formats

### 1.1 Paired canonical `.npy` (message + orderbook)

The unit of training/eval data is one trading day of one ticker, as a PAIR:

```
<TICKER>_<YYYY-MM-DD>_<startns>_<endns>_message_<L>_proc.npy
<TICKER>_<YYYY-MM-DD>_<startns>_<endns>_orderbook_<L>_proc.npy
```

- Output of the canonical preprocessing chain (DataCleaner + tokenizer
  preproc). `L` is the book depth (10 for the canonical corpus).
- Message npy column order (`base_model/encoding/encoding_26tok.py`, `Message_Tokenizer.preproc`):
  `0 order_id, 1 event_type, 2 direction, 3 price_abs, 4 price(rel), 5 size,
  6 delta_t_s, 7 delta_t_ns, 8 time_s, 9 time_ns`, then `*_ref` features.
- Orderbook npy row `R` = book state BEFORE message `R` (row-aligned).
- Consumed by `LOBSTER_Dataset` (`base_model/data/lobster_dataloader.py`) for training and
  by `get_dataset()` in `base_model/inference/generate.py` for rollout/eval, whose
  `*message*.npy` / `*book*.npy` globs rely on the filename embedding the date,
  so a lexical sort is chronological.

### 1.2 `.npy.zst` (zstd-compressed npy)

Same content, compressed. Two consumption paths:

- **zstandard fastpath**: pre-decompress the whole day set ONCE to a local
  `.npy` mirror before the data loader starts, and leave a
  `decompress_done.json` breadcrumb so restarts skip the work. This is the fix
  that took GOOG pretraining from 4.16 to 0.65 s/step, never decompress inside
  the batch loop.
- CLI tools (`tools/data_build/make_snapshots.py`) shell out to
  `/usr/bin/zstd -d` into `--scratch` for one-shot reads.

`FORBID_RAW_NPYZST=1` (see §4) bans directory-scanning raw `.npy.zst` trees for
SP500 sweeps, the SquashFS shards below are the sanctioned path.

### 1.3 SquashFS monthly shards (SP500 corpus)

Canonical store: `/projects/public/s5e/quant_team/lob_preproc_sp500_squashfs`
(= `data/SP500-2022-2025` in this repo):

```
shard_YYYY-MM.squashfs       # one month, EVERY ticker as a subdir inside
index_YYYY-MM.json           # shard-local index (per-month manifest)
```

Layout inside a shard (verified 2026-05-14, `shard_2023-01.squashfs`):

```
<mount>/<TICKER>/<TICKER>_<date>_<startns>_<endns>_message_<L>_proc.npy
<mount>/<TICKER>/<TICKER>_<date>_<startns>_<endns>_orderbook_<L>_proc.npy
<mount>/index.json
```

**Mount pattern**: source the helpers inside the srun-launched (per-node) bash block:

```bash
BENCHMARK_ROOT=<lob_pipeline checkout> source "$REPO/run/benchmarking/squashfs_helpers_no_delete.sh"
trap 'infer_squashfs_cleanup || true' EXIT
infer_squashfs_setup "$TICKER" "2023-01,2023-02" "$SHARD_DIR"
# -> $INFER_DATA_DIR_NODE : per-node symlink farm flattening all requested
#    months for the ticker; matches the *message*/*book* loader globs.
```

Mounts and farms live under `${TMPDIR:-/tmp}`, scoped by
`SLURM_JOB_ID` / `SLURM_PROCID` / pid, so ranks and jobs never collide. The
pseudo-ticker `SP500` pools ALL ticker subdirs into one combined farm.

Wide-book variant (`infer_squashfs_setup_wide_book`) mounts L500 shards named
`shard_<STOCK>_L<LEVEL>_npy_<YYYY-MM>.squashfs` and exports
`INFER_WIDE_BOOK_DIR_NODE` for `--wide_book_dir`.

Scale sanity: a single 8-node job over 48 months = `8 × 48 = 384` squashfuse
mounts and 48 shard-local `index.json` reads per node, far cheaper on the
Lustre MDT than globbing hundreds of thousands of raw `.npy.zst` files.

### 1.4 Wide-L500 snapshot npz (simulator init)

Built by `tools/data_build/make_snapshots.py`; full schema, alignment rules and
build gates are in that module's docstring. Summary:

```
<TICKER>_<date>_L500_linf_snapshots_<n_win>.npz
  snapshots           int32 (n_win, 4*levels)   book BEFORE window-start msg
  local_seq_indices   int64                     window index idx
  target_market_rows  int64                     idx * n_messages
  n_levels, date, ticker
```

Consumed by `LOBSTER_Dataset` via `wide_book_files` to initialise the JaxLOB
simulator. Deep L500 init is required for engine-truthful rollouts: an
L10-only init produces phantom-liquidity artefacts, because the levels the
engine never saw behave as if empty. Build gates: coverage >= 0.90,
crossed <= 0.02, era-aware price-align floor (see the builder).

---

## 2. Which loader consumes which

| Format | Consumer | Use |
|---|---|---|
| paired `.npy` (L10) | `LOBSTER_Dataset` (`base_model/data/lobster_dataloader.py`) | training batches |
| paired `.npy` (L10) | `get_dataset()` globs (`base_model/inference/generate.py`) | rollout / eval context |
| `.npy.zst` | pre-decompress step (fastpath) → then as `.npy` | training on compressed stores |
| SquashFS shard | squashfuse farm → then the `.npy` globs above | SP500 multi-ticker sweeps |
| wide-L500 npz | `LOBSTER_Dataset` `wide_book_files` | JaxLOB simulator init |

---

## 3. Symlink table (`tools/data_symlinks.tsv`)

Nothing hardcodes a data path. Canonical stores are linked under `data/`, and
`bootstrap_fork.sh` creates the links from the table:

1. Read `tools/data_symlinks.tsv` (3 columns: `link_name`, `target`, `note`).
2. For each row, verify the target is readable from YOUR account with a cheap
   exact-path test (`stat <target>`, or `ls <target> | head -3` only if the
   directory is known-small). Private-project paths (not under
   `projects/public/`) are often not traversable across accounts.
3. `ln -sfn <target> data/<link_name>` for each verified row.

`bootstrap_fork.sh` does exactly that, skips unreadable targets with a warning
rather than guessing, and leaves existing links alone. Excluded or
unverifiable targets are recorded as comment lines in the tsv, keep that
convention when adding rows.

---

## 4. HARD rules (Lustre safety, non-negotiable)

These follow the 2026-05-08 Isambard incident (team jobs suspended for MDT
metadata storms). Every launcher must comply BEFORE sbatch:

1. **`SQUASHFS_MULTI_MODE=1` with an explicit month list.**
   `SQUASHFS_MONTHS=2022-01,...,2025-12`, an explicit CSV, ALWAYS.
   `SQUASHFS_MONTHS=all` is forbidden.
2. **`FORBID_RAW_NPYZST=1` for SP500 sweeps.** Falling back to raw
   `lob_preproc_sp500/<TICKER>/*.npy.zst` directory scanning is not allowed;
   the compressed monthly shards are the only sanctioned SP500 path.
3. **Stage weights AND data to `$TMPDIR` before training starts.** The data
   loader never opens cold Lustre paths mid-run. Use `TMPDIR=/tmp`: the
   `/local/user` scratch has been wiped ~30–40 s into workq jobs since
   2026-06-10, which presents as a mid-run file-not-found, not as a wipe.
4. **Checkpoints and logs go node-local first, rsync to Lustre at the end** (or
   at coarse intervals, rank 0 only). Checkpoint resolution is by explicit step
   (`docs/conventions.md`), never by listing the directory. wandb / tensorboard
   write locally and sync at end of run.
5. General: no `find` / recursive `ls` / `du` / `tree` / large globs on `/lus`;
   one `os.scandir` per known directory is the discovery ceiling; per-job
   output directories keyed by `${SLURM_JOB_ID}`; staggered submission
   (~10 jobs per wave, `sleep 30+` between waves); checkpoint interval >= 500
   steps.
6. Permissions: `umask 002`; everything on shared paths ends up
   `group = brics.u6gb`, `g+rwX`; re-`chmod g+rw` after any in-place rewrite
   (`sed -i` / `tee` / `>` silently strip group bits via temp-file rename).
