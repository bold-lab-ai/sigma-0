# Architecture

## The packages

```text
src/
  base_model/          the LOB sequence-model stack (details below)
  matching_engine/     JAX-LOB engine: price-time priority book, LOBSTER loaders, utils
  environment/         replay environments (jaxen: MARLEnv, MarketMakingAgent) and RL baselines (jaxrl)
  lobpipeline/         LOB-Bench adapter + vendored scorer, return bench, run/score/package manifests, model store
  post_training/       heuristic_learning: the closed-loop market-making simulator and its fidelity checks
  state/               the state model: the book resampled as frames on a wall-clock grid, with four post-training methods
```

`base_model` is the part that trains and generates:

| subpackage | holds | entry points |
|---|---|---|
| `encoding/` | `encoding.py`, the runtime selector with the `Message_Tokenizer` and `Vocab` proxies; `encoding_{1,22,23,24,26}tok.py`, one tokenizer per width | `set_token_mode`, `Message_Tokenizer.MSG_LEN` |
| `data/` | `lobster_dataloader.py` (LOBSTER datasets, windows, wide-book init), `dataloading.py` (loaders, ticker-balanced sampler), `data_files.py` (which files a run sees, in what order), `wide_book.py`, `preproc.py`, `sequence_dataset.py`, `permutations.py`, `torch_loaders.py` | `create_lobster_prediction_dataset`, `resolve_data_files` |
| `models/` | `registry.py` (the only place an architecture name becomes a layer factory), `seq_model.py` and `layers.py` (the stacked encoder and the sequence layer), one file per backbone (`mamba3.py`, `ssm.py` for S5, `gdn.py` for GDN and KDA, `transformer.py`, `nsa.py`), `mamba3_jax.py` (pure-JAX chunked SSD reference), `mamba3_pallas.py` (fused Pallas kernels for the SSD, phase 1 included, `MAMBA3_SSD_IMPL=pallas`), `mamba3_ops.py` (Triton bridge), `state_scan_ops.py` (CUDA bridge), `lob_seq_model.py` (the outer LOB model with the book encoder), `legacy_norm.py` | `build_backbone`, `resolve_architecture` |
| `kernels/` | `mamba3_triton/` (SISO forward and backward Triton kernels), `mamba3_cuda/` (state scan `.cu` + build script), `gdn_triton_kernels.py`, `nsa/` | optional; the models fall back to pure JAX when a kernel is absent |
| `training/` | `train.py` (the loop), `train_helpers.py` (train step, validation, CE tables), `state.py` (state init, Orbax save and restore, partial restore), `checkpoint_inspect.py`, `checkpoint_selection.py`, `launch_plan.py` (config to `sbatch` command) | `train`, `init_train_state`, `load_checkpoint`, `build_train_plan` |
| `inference/` | `generate.py` (autoregressive generation with the engine in the loop, replay, do(Q) injection), `evaluation.py`, `validation.py` (syntax masks), `launch_plan.py` | `sample_new`, `generate_batched` |
| `evaluation/` | `message_perplexity.py`, the loss number that survives a change of tokenizer | |
| `sharding/` | `sharding_utils.py`: mesh, state and data shardings | |
| `utils/` | `log.py`, `cli.py` | |

Checkpoints restore by Flax class name and pytree structure, not by import
path. The September 2026 move of this code out of `src/s5` and `src/lob` did
not touch any saved checkpoint (verified by summarising the pinned 78M model
under both trees, job 6347532).

## How data flows

```text
LOBSTER day files (message + orderbook .npy, L10)
  -> base_model.data.lobster_dataloader.LOBSTER_Dataset        windows of N messages, book state before each
  -> base_model.encoding                                       26 tokens per message (the default), or 22/23/24
  -> base_model.models.lob_seq_model + a registry backbone     next-token prediction, teacher forced
  -> base_model.training.train                                 Orbax checkpoints: <root>/j<JOB>_<WANDB>_<JOB>/<step>/
  -> base_model.inference.generate.sample_new                  conditioning + generation, every message through
                                                               matching_engine.jaxob so the book stays consistent
  -> data_cond/ data_gen/ data_real/ CSVs                      one run directory per job
  -> lobpipeline (LOB-Bench adapter, return bench)             scores, divergence curves, evidence records
  -> post_training.heuristic_learning                          frozen episodes replayed with a policy in the loop
```

The engine (`matching_engine.jaxob`) is used twice: inside generation, so that
generated orders act on a real book, and inside the post-training simulator,
which settles the policy's own fills. The generator never produces the fills of
our own orders; that is the one rule the closed loop must not break.

## Where the code came from

sigma-0 was assembled in June 2026 from four trees. Provenance matters because
sealed numbers were measured on the originals.

| package | source | how it was taken |
|---|---|---|
| `matching_engine` | AlphaTrade `gymnax_exchange` (`.../s5e/quant_team/quant/AlphaTrade/LOBS5/Alphatrade/gymnax_exchange`) | copied; `jaxob_constants.py` byte-identical, `JaxOrderBookArrays.py` differs by 8 import lines (`gymnax_exchange.` to `matching_engine.`) |
| `environment` | the same AlphaTrade tree, `jaxen` and `jaxrl` | copied, imports rewritten |
| `base_model` (as `lob/` and `s5/`) | `AlphaTrade/experiments/exp_R1_Mamba3` for the LOB stack and Mamba3; `exp_O8_self_attention` for S5; `exp_O2b_rope` for Transformer and NSA; `exp_O2b_nsa_infer_hardened` for the hybrid adapter, the attention-capable LOB model and the prefill inference path | copied, then adapted: the strict registry, a common `__call_rnn__` boundary, in-repo engine imports, NSA blocks derived from the 26-token encoding |
| `lobpipeline.scorer/lob_bench` | FLAIROx `lob_pipeline` merged three-way with Sat-A `lob_bench` | `src/lobpipeline/scorer/merge.json` records base, both sides and the tree hash; `ci/check_lobbench_merge.py` refuses a drifted tree |
| `run/base_model/train_full_autoreg.batch`, `node_wrapper.sh` | `/lus/lfs1aip2/projects/public/u6gb/openreview-v2` | restored byte for byte in June 2026, then edited here |
| `post_training.heuristic_learning` | written in this repository (PR #4 and its follow-ups) | |
| `state` | written in this repository (PR #38) | |
| do(Q) injection, `replay_real_msgs_single`, `compute_gt_divergence` | Jess Carr's evaluation work, via LOBelia `380b35f` | archive at `/lus/lfs1aip2/projects/public/u6gb/lob_evals_jcarr/` |

Source hashes of the initially ported files (26-token encoder, Mamba3, GDN,
S5, Transformer, NSA, layers, LOB model, prefill inference) are in the git
history of `docs/model_zoo/provenance.md` at commit `127d76b`.

## The September 2026 reorganisation

Until PR #79 the real code lived in `src/s5` (backbones, kernels) and `src/lob`
(data, encodings, training, inference) and `src/base_model` was a facade of
re-exports. PR #79 moved the modules into `base_model` byte-identically, then
rewrote the imports. For one cycle `src/lob` and `src/s5` stayed as alias
packages (a `sys.meta_path` finder returning the real module objects) so that
open branches and the four files hashed by `ci/evidence` kept importing. The
follow-up PR re-measured that evidence and deleted the aliases, the root
shims and the `logs_lobs5/` log directory name. `ci/check_legacy_imports.py`
now refuses any `s5.*`, `lob.*` or root-module import: one means a stale
rebase.

The `s5` that survives is the architecture name: S5 is one of the six
registered backbones (`configs/train/model_zoo_smoke_s5.yaml`).
