# Training

## The launcher chain

```text
run/base_model/train_base_model.py --config <yaml>      reads the flat key: value config, prints the sbatch command
  -> env <config env_*> sbatch --nodes=N run/base_model/train_full_autoreg.batch
      -> srun, one task per node -> run/base_model/node_wrapper.sh          conda by prefix, CUDA module, NCCL/OFI stack,
                                                                             XLA flags, node-local TMPDIR, data staging
          -> python run/base_model/runtime/train.py                          argparse -> base_model.training.train.train
```

`train_base_model.py` never submits; it prints the command, with `--test-only`
unless `--execute` is given. Run what it prints. `WORKDIR` is the checkout the
job runs from (`legacy_workdir: .` resolves to the current directory);
`node_wrapper.sh` puts `$WORKDIR/src` on `PYTHONPATH`.

## Smokes

Login node, no submission:

```bash
PYTHONPATH=src python run/base_model/train_base_model.py --config configs/train/mamba3_smoke.yaml
bash -n run/base_model/train_full_autoreg.batch run/base_model/node_wrapper.sh
```

Compute node, one node for ten minutes, W&B offline:

```bash
CMD=$(PYTHONPATH=src python run/base_model/train_base_model.py --config configs/train/mamba3_smoke.yaml --execute | sed -n 's/^command=//p')
eval "${CMD/ sbatch / WANDB_MODE=offline USE_WANDB=False PER_GPU_BSZ=1 N_DATA_WORKERS=0 sbatch }"
```

What the smoke config proves, measured 2026-09-05 on GH200 nodes:

| setting | outcome | job |
|---|---|---|
| `mamba3_smoke.yaml` as committed (78M, 500 messages, 10 samples per GPU) | reaches the first train step, then `RESOURCE_EXHAUSTED` asking for 269 GiB per GPU on the pure-JAX path | 6347534 |
| the same with a DataLoader worker | the worker died initialising CUDA before the first step | 6347597 (on `main` at 127d76b) |
| `PER_GPU_BSZ=1 N_DATA_WORKERS=0` | two optimizer steps, epoch-end checkpoint, exit 0 | 6347641 |

A smoke reaching `node_wrapper.sh`, initialising four GPUs, and completing an
optimizer step are three different milestones. Say which one you hit.

## Configs

Flat `key: value` files under `configs/train/`, read by
`base_model.config.read_key_value_config`; `include:` names a parent. Keys
prefixed `env_` become environment variables for the launcher.

| config | purpose | note |
|---|---|---|
| `mamba3_smoke.yaml` | one-node training smoke on GOOG 2022, three train days | too large at 10 per GPU; see above |
| `base_model_default.yaml` | eight nodes, forty epochs, the 75m preset | the production shape |
| `model_zoo_smoke_{mamba3,s5,gdn,kda,transformer,nsa}.yaml` | tiny one-GPU configuration per architecture | share `model_zoo_smoke_base.yaml` |
| `mamba3_restore_{single_gpu,single_node,multi_node}.yaml` | resume from a legacy R1 checkpoint | `RESTORE_PATH` is the manager root |
| `mamba3_sp500.yaml` | SP500 SquashFS corpus | requires `SQUASHFS_MULTI_MODE=1` and an explicit month list |
| `selftrain12h_*.yaml`, `selftrain_*.yaml`, `dfm_smoke_1gpu.yaml` | the July 2026 self-training and DFM runs | historical: `legacy_workdir` points at the `openreview-v2` checkout, so they do not run from this repository |

`configs/train/goog_2022_smoke_index.json` is the index sidecar the smoke uses
so the shared data root is never written to.

## Environment knobs the launcher reads

| knob | default | effect |
|---|---|---|
| `ARCHITECTURE` | `mamba3` | registry name; `SSM_TYPE` and `MODEL_TYPE` from old configs are interpreted at load and must not conflict |
| `TOKEN_MODE` | `26tok` | the encoding; the launcher asserts it against the batch script's token count |
| `MODEL_PRESET` | `75m` | width and depth preset |
| `PER_GPU_BSZ`, `GRAD_ACCUM_STEPS`, `MSG_SEQ_LEN` | 10, 1, 500 | batch shape |
| `CURTAIL_EPOCHS` | unset | stop after this many epochs (smokes use 1) |
| `CHECKPOINT_EVERY` | see script | steps between mid-epoch checkpoints; never below 500 on Lustre |
| `NO_AUTO_RESUME` | unset | disables the resubmit-on-exit chain |
| `N_DATA_WORKERS` | 1 | DataLoader workers; 0 for smokes |
| `LOG_EVERY` | 250 | steps between `step_loss` and `lr` points |
| `LOG_GRAD_NORMS` | off | per-group gradient norms and `clip_ratio` at the loss cadence |
| `LOG_CE_BY_POSITION` | off | per-position CE tables to W&B (the profiling experiment) |
| `WANDB_MODE`, `USE_WANDB`, `WANDB_ENTITY`, `WANDB_PROJECT` | online, True, oxford-lob, per config | every experiment tracks online; smokes may go offline |
| `DATA_ROOT`, `DATA_INDEX_JSON`, `TICKERS`, `TRAIN_DATE_RANGE`, `TEST_DATE_RANGE` | per config | data selection; ranges contain commas, pass them through the environment |

Mamba3 numerics and kernels, read by `src/base_model/models/mamba3.py` at
trace time and forwarded across auto-resume by the batch script:

| knob | default | effect |
|---|---|---|
| `MAMBA3_SSD_IMPL` | `xla` | the chunked SSD: `xla` is the einsum reference in `mamba3_jax.py`; `pallas` is the fused kernel pair in `mamba3_pallas.py` (one program per chunk and head; B and C read once for all heads and the rotated query and scaled key built in registers; nothing of size chunk x chunk or per-head (L, heads, d_state) written to HBM; recompute in the backward). Left in JAX: the angle cumsum, the decay cumsum and the cross-chunk state passing. Same numbers to rounding; layers whose tiles are not powers of two of at least 16 (chunk, d_state, headdim, and the rope ranges R, R, d_state - 2R) fall back to `xla` with a warning |
| `MAMBA3_PALLAS_DTYPE` | `float32` | dtype the tensor cores see inside the Pallas kernels (`bfloat16` halves the operand bytes); accumulation is f32 either way |
| `DTYPE` (`--dtype`) | `float32` | `bfloat16` runs the Mamba3 in_proj, out_proj and GLU projections, the transformer and NSA blocks, and the Pallas SSD kernel operands in bf16. Parameters, optimizer state, norms, the decay math (softplus, exp, cumsum), the QK-dot skip, the cross-chunk recurrence, the residual stream and the log-softmax stay float32. The XLA SSD reference always receives float32 inputs. s5, gdn and kda refuse the flag because nothing in them reads it |
| `SINGLE_NODE_SHARD_MAP` | unset | `1` keeps `HIERARCHICAL=True` on one node: a (1, N) `nodes x gpus` mesh and the same shard_map training step multi-node jobs run. Unset, a single node uses the historical 1-D `data` mesh under jit. Needed for `MAMBA3_SSD_IMPL=pallas` on one node: a custom kernel under plain jit has no sharding rule and is replicated on every GPU. The trainer refuses `MAMBA3_SSD_IMPL=pallas` on one node with several GPUs unless this is set |
| `MAMBA3_STATE_PASSING` | `blocked` | Pallas path only: how the cross-chunk state is carried. `blocked` does 16-chunk blocks plus a block-level carry (0.68 ms per layer forward, job 6394522); `dense` is the reference's (nc+1) x (nc+1) contraction (1.08 ms). Same numbers to rounding |
| `GRAD_ALLREDUCE_DTYPE` | `float32` | dtype the cross-node gradient mean travels in on the shard_map path. `bfloat16` halves the inter-node bytes; the intra-node NVLink reduction, parameters and optimizer stay float32. Opt-in; the loss gate for it is still owed |
| `MAMBA3_CONTRACTION_PRECISION` | `highest` | `highest` keeps the SSD contractions in full f32; `default` allows TF32 (both paths). Production has always run `highest`; every 2026-09 speed benchmark ran `default` |

Hybrid attention (`ARCHITECTURE=hybrid_mamba3`), settable like any model arg;
defaults in `src/base_model/models/registry.py`:

| arg | default | effect |
|---|---|---|
| `hybrid_attn_layers` | derived | attention positions; unset follows the Nemotron-H rule in the 0.26 to 0.71 depth band |
| `hybrid_attn_heads` | 10 | head count, reduced until it divides `d_model` |
| `hybrid_attn_d_ff` | `4*d_model` | FFN width of the attention blocks |
| `hybrid_attn_flash` | true | Pallas causal kernel; refuses head dims it cannot take |
| `hybrid_attn_positional_encoding` | false | Nemotron-H runs attention without positional encoding |
| `attention_max_cache_len` | 26000 | KV cache length for generation; must cover conditioning plus generation (defect D-T5) |

## Throughput

Measured September 2026 on GH200 nodes for the 75m preset (Mamba3, d_model
1024, six fused layers plus two message layers and one book post-layer at
full length, 13,000 tokens per sequence, one sequence per GPU). Every row is
a same-node, interleaved arm of 400 optimizer steps at seed 42; the step
time is the mean of the timing windows at steps 200 to 400. Tokens per
second are per node.

### The recipe

```
SINGLE_NODE_SHARD_MAP=1          # one node only; multi-node jobs already take this path
DTYPE=bfloat16                    # bf16 projections, f32 parameters and numerics elsewhere
MAMBA3_SSD_IMPL=pallas            # fused SSD kernels (bf16 tiles when the layer is bf16)
MAMBA3_CONTRACTION_PRECISION=default   # TF32 for the remaining f32 contractions
```

| configuration | s/step | tok/s | MFU (bf16 peak) | job |
|---|---:|---:|---:|---|
| production numerics (f32, HIGHEST, XLA SSD) | 0.158 | 328k | 10.3% | 6393377 |
| TF32 contractions | 0.128 | 405k | 12.7% | 6393377 |
| TF32 + `DTYPE=bfloat16` | 0.107 | 484k | 15.2% | 6393377 |
| TF32 + bf16 + Pallas SSD, shard_map path | 0.092 | 565k | 17.7% | 6393966, 6394522 |
| the same at `MAMBA3_CONTRACTION_PRECISION=highest` | 0.099 | 522k | 16.4% | 6394522 |
| the same with phase 1 in Pallas as well | 0.082 | 634k | 19.9% | 6422282 |
| the same with the chunk_scan backward as three kernels (current kernels) | 0.072 to 0.075 | 714k | 22.4% | 6425384 |

The last row is the current `MAMBA3_SSD_IMPL=pallas`: phase 1 (biases, RoPE,
trapezoidal scale, QK-dot) runs in its own Pallas kernel pair on the shared B
and C, so no per-head (L, heads, d_state) float32 tensor is built by XLA. On
one node, same node, interleaved, two repeats each (job 6422282): production
numerics 0.155 s/step, TF32 0.129, the recipe with the previous kernels 0.092
and 0.091, with the current kernels 0.082 and 0.082; two sequences per GPU
0.164 (0.082 per sequence). Per layer the JAX side of the SSD fell from 1.9 to
0.5 ms forward+backward and the two phase-1 kernels cost 0.35 / 0.84 ms; the
GPU time of a layer-step went from 8.9 to 7.9 ms, all of it out of XLA's
elementwise, reduce and relayout fusions (4.4 to 3.0 ms), the SSD kernels
unchanged. Folding phase 1 into the SSD kernels themselves was tried first
and regressed to 0.158 s/step (job 6416484): the compiler capped those
kernels at 32 registers per thread and spilled to local memory (job 6421949).

The chunk_scan backward as three kernels (gate and state, value, query and
key) is the current form: on one node, same node, two repeats each (job
6425384), 0.083 and 0.083 s/step with the single backward kernel against
0.072 and 0.075 with the three; production numerics 0.156 on that node. Per
layer-step the backward kernel went from 2.16 ms to 0.97 (0.44 + 0.25 +
0.28), everything else unchanged, GPU time 7.9 to 7.0 ms. Per layer on one
GPU, bf16 tiles, forward+backward: 7.76 to 6.56 ms; eight warps are slower
(7.00), so four stay the default.

Same-configuration repeats on one node agree to about 2 percent in step
time. The Pallas path under the single-node jit (no `SINGLE_NODE_SHARD_MAP`)
is 2.8 times slower than XLA: a custom call has no sharding rule and the
partitioner runs the whole batch on every GPU (job 6393829, one sequence per
device on four devices: 11.4 ms forward against 2.35 on one device).

### Four nodes

Same arms on four nodes (16 GPUs, the hierarchical shard_map path every
multi-node job runs), job 6395905. The step is the single-node step plus a
fixed cost: fitting t = c x batch + k to the batch 1 and 2 arms gives
c = 85 ms per sequence and k = 58 ms per step for the recipe, c = 149 and
k = 61 for production numerics. k is the inter-node gradient reduction (314 MB
of float32 over Slingshot) plus per-step synchronisation, so the per-GPU batch
is the lever at scale: it amortises k.

| configuration | s/step | tok/s (16 GPUs) | MFU |
|---|---:|---:|---:|
| production numerics, batch 1 | 0.210 | 0.99M | 7.7% |
| TF32 contractions, batch 1 | 0.180 | 1.15M | 9.0% |
| recipe, batch 1 | 0.143 | 1.45M | 11.4% |
| production numerics, batch 2 | 0.359 | 1.16M | 9.1% |
| recipe, batch 2 | 0.228 | 1.82M | 14.3% |
| recipe + `GRAD_ALLREDUCE_DTYPE=bfloat16`, batch 1 | 0.125 | 1.65M | 13.0% |
| recipe + `GRAD_ALLREDUCE_DTYPE=bfloat16`, batch 2 | 0.215 | 1.93M | 15.1% |

The bf16 wire rows are job 6396072 (two repeats each, 0.124 to 0.127 s/step
against 0.146 for the float32 wire). At 400 steps their loss is not worse than
the float32-wire arms at the same seed; the longer multi-seed comparison is
part of the loss gate still owed.

Multi-node arms need a checkpoint directory every node can see: with a
node-local `CHECKPOINT_BASE_DIR` the epoch-end save waits for a directory only
rank 0 has, the other ranks reach the shutdown barrier alone, and the job
hangs about fifteen minutes before the coordination service gives up (job
6395171).

### Eight nodes

Same arms on eight nodes (32 GPUs, the hierarchical shard_map path), the
75m preset, seed 42, jobs 6425735 and 6425791 (two node sets, agreeing to
1 percent). Batch-2 arms used a 14-day GOOG 2022 index so the epoch reaches
the steady timing windows. The one-node reference for the same code is
0.156 to 0.160 s/step for production numerics and 0.072 to 0.075 for the
recipe (jobs 6425384, 6425753).

| configuration | s/step | tok/s (32 GPUs) | MFU |
|---|---:|---:|---:|
| production numerics, batch 1 | 0.266 to 0.270 | 1.55M | 6.1% |
| recipe, float32 gradient wire, batch 1 | 0.185 | 2.25M | 8.8% |
| recipe with `GRAD_ALLREDUCE_DTYPE=bfloat16`, batch 1 | 0.147 to 0.149 | 2.84M | 11.1% |
| production numerics, batch 2 | 0.420 | 1.98M | 7.7% |
| recipe with the bf16 wire, batch 2 | 0.216 to 0.223 | 3.72M to 3.84M | 15.0% |

Fitting t = c x batch + k on one node set: the recipe costs 71 ms per
sequence and 78 ms of fixed per-step cost at eight nodes (75 and 40 at four
nodes, bf16 wire); production numerics 150 ms per sequence and 120 ms fixed
(149 and 61 at four nodes). The per-sequence cost is the single-node number;
the fixed cost doubles from four to eight nodes and the bf16 wire is worth 20
percent of the step at eight. Speedup against production numerics: 2.1x on
one node, 1.83x at eight nodes and batch 1, 1.91x at batch 2.

### Other architectures and sizes

One node, same node, interleaved production numerics against the recipe,
400 steps, seed 42, per-GPU batch 1, the same data pipeline for every row
(jobs 6425753, 6425796, 6425851). The recipe is `SINGLE_NODE_SHARD_MAP=1
DTYPE=bfloat16` plus `MAMBA3_SSD_IMPL=pallas MAMBA3_CONTRACTION_PRECISION=default`
where Mamba3 layers exist. Parameter counts are the trainer's own. The loss
column is the paired difference, recipe minus production, at step 400 of the
same seed and data order; same-configuration repeats at 400 steps differ by up
to 0.06 nats in this campaign, so single-seed differences below that are
noise. The last column carries that difference forward to the length the
model zoo trained each model for. It is an approximation and assumes the
difference does not drift with steps, which the two longer measurements
support (78M: three seeds, no trend between steps 400 and 2,300; 293M: the
early gap closes by step 500 at both seeds) but which no full-length run has
verified.

| architecture | params | production s/step | recipe s/step | speedup | recipe tok/s | recipe MFU | loss diff at 400 steps | extrapolated over a longer period |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Mamba3 SISO, d_model 256, 11 layers | 10.8M | 0.073 | 0.038 | 1.93x | 1.39M | 11.2% | +0.01 | +0.01 over 40.8k steps |
| Mamba3 SISO, d_model 768, 6 layers | 46.4M | 0.121 | 0.055 | 2.21x | 948k | 19.6% | +0.05 | +0.05 over 63.4k |
| Mamba3 SISO, d_model 1024, 6 layers (75m preset) | 78.5M | 0.160 | 0.074 | 2.16x | 701k | 22.0% | -0.06 | -0.001 over 28.7k (three seeds x 2,300 steps, job 6395871) |
| Mamba3 SISO, d_model 1280, 6 layers | 119.3M | 0.207 | 0.093 | 2.22x | 557k | 25.0% | -0.02 | -0.02 over 30.0k |
| Mamba3 SISO, d_model 2048, 6 layers | 293.3M | 0.349 | 0.157 | 2.23x | 332k | 34.9% | +0.16 | -0.03 (seed 42) and +0.14 (seed 43) over steps 500 to 900, job 6425925; not resolved at two seeds; 168k steps in the zoo |
| hybrid_mamba3, 75m preset, XLA attention | 83.4M | 0.171 | 0.082 | 2.08x | 624k | 7.9% | +0.03 | +0.03 (no zoo run) |
| hybrid_mamba3, 75m preset, 16 heads, Pallas attention | 83.4M | 0.170 | 0.078 | 2.19x | 665k | 8.4% | -0.01 | -0.01 (no zoo run) |
| transformer, d_model 1024, 6 layers, Pallas attention | 125.4M | 0.307 | 0.134 | 2.29x | 388k | 7.4% | 0.00 | 0.00 over 70.5k |

Any loss difference here is the recipe's numerics, bf16 projections and tiles
and TF32 contractions, not the kernels: the compiled kernels reproduce the
reference to 1e-5 forward and 8e-6 on gradients at equal precision
(`tests/integration/test_mamba3_pallas_gpu.py`, jobs 6422282 and 6425384). At
293M the recipe trails production by 0.3 nats at step 300 and 0.13 at 400 and
is within a few hundredths from step 500 on at seed 42; at seed 43 it stays
0.09 to 0.21 above from step 500 to 900. TF32 alone and bf16 without the
kernels land within 0.07 nats of production at step 900 (job 6425925). A
multi-seed gate at that size is owed before the recipe is used at d_model 2048.
The 78M row is the canonical model; the three-seed gate is its long-horizon
number.

Three findings about `main` from these runs, none caused by this branch:

- `hybrid_mamba3` at the 75m preset with its defaults (10 heads reduced to 8,
  head_dim 128, Pallas attention) does not compile in float32 on a GH200: the
  attention kernel asks for 256 KB of shared memory against 227 KB available
  (job 6425755). It compiles in bf16 (0.078 s/step), with 16 heads, or with
  `HYBRID_ATTN_FLASH=False`.
- The transformer at d_model 1024 with dense attention over 13,000 tokens needs
  117 GB in float32 and does not fit; `USE_FLASH=True` is required at this
  length (job 6425755).
- `nsa` at d_model 1024 with 12 layers does not train at 13,000 tokens: one
  step at most, then a hang and the 600-second NCCL termination, in float32 and
  bf16, on the shard_map and the plain jit paths (jobs 6425755, 6425913,
  6425972). It is not in the table.

`s5`, `gdn` and `kda` are unchanged by construction: `--dtype` is refused for
them, the kernels are Mamba3-only, and shard_map costs nothing on one node.
Mamba3 MIMO is not implemented in this repository (the SSD is the SISO
variant), so the zoo's MIMO checkpoint has no row.

### Where a step goes

| measurement | value | job |
|---|---:|---|
| one full-length Mamba3 layer, forward + backward + its optimizer share | 13.0 ms | 6393377 (N_LAYERS 6 vs 2) |
| nine full-length layers | 117 of 128 ms | same |
| Muon Newton-Schulz (vs AdamW) | ~1 ms | 6393377 |
| SSD block of one layer, XLA reference at TF32, forward / forward+backward | 2.6 / 7.5 ms | 6393829 |
| of which phases 1 to 2 (RoPE, biases, scaling, chunking) | 1.0 / 1.9 ms | 6394522 |
| SSD kernels + state passing, bf16 tiles, forward / forward+backward | 1.3 / 5.1 ms | 6394522 |
| state passing alone, dense / blocked, forward | 1.08 / 0.68 ms | 6394522 |
| SSD block of one layer, phase-1 kernels and the single backward kernel, bf16 tiles, forward / forward+backward | 1.7 / 7.6 ms | 6422282 |
| SSD block of one layer, current Pallas path (three backward kernels), bf16 tiles, forward / forward+backward | 1.7 / 6.6 ms | 6425384 |
| of which the JAX side (angle cumsum, weights, padding) | 0.4 / 0.5 ms | 6422282 |
| of which the phase-1 kernel pair | 0.35 / 0.84 ms | 6422282 |
| the previous kernels on the same node (phase 1 in XLA) | 2.1 / 8.8 ms | 6422282 |
| batch 2 or 4 per GPU on the recipe, per sequence | 0.094 s (flat) | 6394522 |

GPM counters during the reference step: SM active 95%, memory controller
busy 64 to 68%, tensor pipe 14 to 15%, FP32 pipe 3% (jobs 6393341, 6393377).
Nsight Compute on the SSD kernels: every kernel that contains a matmul
compiles to 32 registers per thread with heavy local-memory traffic (3.5e8
local loads per twelve launches of the scan forward, 8.7e8 for the single
backward kernel, job 6422282), while the elementwise phase-1 kernels get 228
and 255 registers and no spills. The compiled PTX carries no register
directive (job 6425111: only `.reqntid 128`), the Pallas API exposes only
warps and stages, and eight warps (128 registers) are slower, so the cap is
ptxas's own choice and not a knob. Splitting the backward into three kernels
did not change the register count or the local traffic; it halved the time
by shortening each kernel's dependency chain and tripling the programs in
flight (job 6425384). The forward scan kernel has the same pattern at a
smaller scale (0.2 ms per layer-step); why matmul kernels are compiled this
way is the open question for the next step.

### Loss under the recipe

bf16 projections alone: loss within 0.02 nats of the f32 run from step 1000
to 1700 at the same seed and data order (job 6393377, 2000 steps). The full
recipe: 0.035 nats above the f32 reference on average over steps 1000 to
1700, one seed (job 6394522); repeats of one configuration differ by 0.03 to
0.05 nats at these steps, so the difference is not resolved. A multi-seed
gate at a longer horizon is still owed before the recipe becomes a default.

## Checkpoints

Orbax layout `<CKPT_BASE>/j<JOBID>_<WANDB>_<JOBID>/<step>/{_CHECKPOINT_METADATA, metadata/, state/}`.
`metadata/metadata` carries the full training config and is the ground truth
for what a checkpoint is. Resume with `RESTORE_PATH` and `RESUME_FROM_STEP`;
the auto-resume in the batch script finds its own job's directory with
`lfs find --maxdepth 1`, the one sanctioned recursive lookup. Write checkpoints
node-local first when the cadence is high. Loading and pinning are in
`docs/checkpoints.md`.

## Where the tables came from

`docs/training-launcher-appendix.md` holds the historical parameter tables and
baseline provenance that used to sit at the bottom of the batch script. Dated,
unverified, kept for the record.
