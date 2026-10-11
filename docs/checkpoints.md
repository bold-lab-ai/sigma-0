# Checkpoints

## The model store: blessed checkpoints, by name

Not to be confused with the *model zoo*, which in this repo means the
architecture registry, mamba3 / s5 / gdn / kda / transformer / nsa, selected
with `--architecture` (`docs/architecture.md`). This is about
*checkpoints*: which trained weights exist, where they are, and which step of
each one a result was measured on.

### The shape

Models are treated the way data already is:

| | payload | table in the repo |
|---|---|---|
| data | `data/SP500-2022-2025` → shared store | `tools/data_symlinks.tsv` |
| models | `models/` → `/lus/lfs1aip2/projects/public/u6gb/model_zoo` | `src/lobpipeline/model_store_pins.json` |

`bootstrap_fork.sh` does not create `models/`; it is tracked in git as a
symlink, like `data/SP500-2022-2025`. Off the cluster it dangles, and the
store-dependent half of `tests/unit/test_model_store_pins.py` skips.

```python
from lobpipeline.model_store import list_models, resolve, verify

resolve("m3-megacap8-8p2m-msgonly")["path"]   # models/<name>/checkpoint
resolve("m3-megacap8-8p2m-msgonly")["step"]   # 40760, pinned, not discovered
verify("m3-megacap8-8p2m-msgonly")            # metadata still hashes as recorded
```

### The contract

Same as `src/post_training/heuristic_learning/checkpoints.py`, applied to more
checkpoints:

- **One explicit step per entry, pinned.** Nothing reads a checkpoint
  directory, nothing resolves "newest", and no entry can express it, a test
  asserts the word does not appear in the table. For several of these runs the
  blessed step is *not* the last one on disk; later is not better, it is a
  different model, and a number measured on one does not transfer to the other.
- **A SHA-256 of `<step>/metadata/metadata`.** `verify()` reads that one file
  at a known path and compares. It never enumerates.

Both properties are also the Lustre rule: enumerating step directories to find
the newest is the metadata-storm pattern (`docs/conventions.md`).

### What an entry buys over a path

The paths themselves are unusable as institutional memory: they live in a
rescue mirror under directory names like
`s5e_archive/projects_public_s5e_quant_team_quant/AlphaTrade/experiments/exp_R11_book-ablation/checkpoints/j3751501_3y3i51wn_3751501`,
and that prefix moved once already (the 2026-08-15 workspace reorg, which is
why the table was re-pinned that day). More to the point, the fields record
the distinctions that have actually caused mistakes:

- `use_book_data`, message-only vs book+messages. `m3-megacap8-8p2m-msgonly`
  is the message-only 8.2M; most of the rest are dual-stream.
- `mamba3_legacy_norm`, `"verify"` means the weights predate the 2026-04-22
  norm fix, so restore parity has to be checked before generation is trusted
  (see `src/mamba3_legacy_norm.py`). Loading these wrong does not crash.
- the **two distinct 78M models at step 28730**: `m3-goog-78m-u6gb` is GOOG-only,
  `m3-sp500-78m-s42` is the SP500 ladder point. Same size, same step, different
  training corpus. A test pins them apart.

### What is in it

25 entries as of 2026-08-15, all verified to resolve:

- the **SP500 Mamba3 scaling ladder**, seed 42, book+messages: `m3-sp500-{0p2m,
  1m, 4m, 6m, 10m, 14m, 23m, 50m, 78m, 120m, 200m, 350m}-s42`, plus
  `m3-sp500-290m-s42`.
- longer-trained variants: `m3-sp500-78m-s5-ext`, `m3-sp500-6m-s5-longdata`.
- `m3-goog-78m-u6gb` (GOOG-only 78M), `m3-78m-ftgoog-s42` (GOOG fine-tune).
- 8-megacap-ticker models: `m3-megacap8-8p2m-msgonly` (message-only),
  `m3-megacap8-78m-wce`, `m3-megacap8-79m-mimo`, `m3-goog-6m-zerobook`.
- non-Mamba3 comparisons: `o8-transformer-125m`, `o2-nsa-200m`,
  `o2b-rope-ctx1000`, `o2b-nsa-rope`.

`MODEL_CARD.md` beside each entry in the store has the longer provenance.

### Adding one

Bless a step, then record it: the entry needs the source run directory, the
step, and the SHA-256 of that step's `metadata/metadata` (which is also where
every descriptive field comes from, it carries the full training config).
Never point an entry at a step you have not measured something on.

## `MAMBA3_LEGACY_NORM`

RMSNorm epsilon compatibility switch (code default vs weight-era). The
mechanism is `src/base_model/models/legacy_norm.py`, an opt-in runtime monkeypatch of
`Mamba3SSM.setup`:

- **unset (or `0`)**, **fixed-norm checkpoints**: the s28730 anchor family and
  anything trained after the 2026-04-22 fix (commits ffe45d66 / 13f04e52).
  This is the default and what `setup_env.sh` leaves in place.
- `MAMBA3_LEGACY_NORM=1`, **legacy-norm checkpoints only** (pre-2026-04-22,
  e.g. the 2026-03-28 78M baseline j3417629). Never use for new training.

Loading a checkpoint with the wrong setting does not crash, it silently
scrambles the SSM dynamics (overall CE ~8 instead of ~0.6 in the case that
motivated the shim). Always record the value next to the checkpoint.

---
