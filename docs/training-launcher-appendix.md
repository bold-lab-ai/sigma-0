# Launcher appendix: train_full_autoreg.batch

Historical reference that lived at the bottom of `run/base_model/train_full_autoreg.batch` as an inert here-document until September 2026. It is the record of where the launcher's defaults came from. It has not been re-verified; treat every number as dated.

```text

CAVEAT NOT UPDATED YET MAY BE WRONG !!!


================================================================================
ignore-times-shard-map Branch: Training Data / Test Data / Default Metrics Overview
================================================================================

Based on HEAD: fe24db5, default 32 nodes (128 GPUs) configuration

--------------------------------------------------------------------------------

1. DATA OVERVIEW

1.1 Dataset Partitioning

Partition     Data Path                                                           Days      Source              Method
---------     ---------                                                           ----      ------              ------
Train Set     /path/to/home/user/.../GOOG/2022                     249 d     GOOG 2022 Full      Main directory files
Val Set       Randomly sampled from training set                                  ~25 d     GOOG 2022 Subset    Date-level random sampling
Test Set      /path/to/data/JAN2023/GOOG_24tok_preproc                      9 d       GOOG Jan 2023       Temporal extrapolation

Key Design: The validation set is sampled at the "date" level from the training set (not random message sampling) to ensure time-series independence. The test set uses independent data from Jan 2023 to verify temporal extrapolation.

1.2 Data Format: 26 Token Message Encoding (Active Encoding)

Each LOB message is 26 tokens, divided into 8 new event fields (16 tok) + 4 reference fields (10 tok).

Message Structure (26 tokens):
┌────────────────── New Event Fields (16 tok) ──────────────────┐┌───── Reference Fields (10 tok) ──────┐
│ idx:  0    1    2-4     5-6   7    8-10   11-12  13-15        ││ 16-18    19-20  21-22  23-25         │
│       │    │    │       │     │    │      │      │            ││ │        │      │      │             │
│      evt  dir  price   size dt_s  dt_ns  time_s time_ns      ││ p_ref   sz_ref ts_ref tns_ref       │
│           (sign+hi+lo) (h+l)                                 ││(sign+h+l)(h+l)                      │
└───────────────────────────────────────────────────────────────┘└──────────────────────────────────────┘
                                           ^---- ignore_times removes these 5 tokens (idx 11-15)

Token Index   Field Name      Count   Type         Range                  Description
-----------   ----------      -----   ----         -----                  -----------
0             event_type      1       event_type   {1,2,3,4}              1=New, 2=Cancel, 3=Delete, 4=Trade
1             direction       1       direction    {0,1}                  0=Ask, 1=Bid
2-4           price           3       sign+price   sign{-1,+1}, 2x[0,999] Base-1000 relative price (ticks)
5-6           size            2       size_digit   2x[0,99]              Base-100 order quantity (0-9999)
7             delta_t_s       1       time         [0, 999]               Interval (seconds)
8-10          delta_t_ns      3       time         3x[0,999]              Interval (nanoseconds)
11-12         time_s          2       time         2x[0,999]              Exchange time sec (removed by ignore_times)
13-15         time_ns         3       time         3x[0,999]              Exchange time ns (removed by ignore_times)
16-18         price_ref       3       sign+price   Same as price          Reference price (sign + 2 base-1000 digits)
19-20         size_ref        2       size_digit   2x[0,99]              Reference quantity (base-100)
21-22         time_s_ref      2       time         2x[0,999]              Reference time sec
23-25         time_ns_ref     3       time         3x[0,999]              Reference time ns

TOK_LENS array: (1, 1, 3, 2, 1, 3, 2, 3, 3, 2, 2, 3)  -> sum = 26
NEW_MSG_LEN  = 16   (new event fields only, excluding _ref fields)

1.3 Vocabulary (26tok)

Encoding Domain       Size    Token Range    Description
---------------       ----    -----------    -----------
Special tokens        4       0-3            MASK=0, HIDDEN=1, NA=2, START=3
time                  1000    4-1003         Digits 0-999, delimiter at [3,6,9,12]
event_type            4       1004-1007      Types 1-4
size_digit            100     1008-1107      Base-100 digits 0-99
price                 1000    1108-2107      Price digits 0-999
sign                  2       2108-2109      Sign {-1, +1}
direction             2       2110-2111      Direction {0, 1}
Total n_classes       2112                   Returned by Vocab.__len__()

1.4 Sequence Dimensions (26tok)

Parameter             Value      Formula        Description
---------             -----      -------        -----------
msg_seq_len           500        (CLI arg)      History messages per sample
MSG_LEN               26         (constant)     Tokens per message
seq_len (total)       13000      500 x 26       Total tokens per sample
Effective loss tok    10500      500 x 21       After removing time_s/ns (5 tok/msg)
n_classes             2112       Vocab size     Output softmax classes

--------------------------------------------------------------------------------

2. IGNORE_TIMES MECHANISM DETAILS

2.1 Core Implementation

ignore_times=True does NOT change model input (still 26 tok); it only affects the loss and accuracy calculation.

Implementation location: lob/train/train_helpers.py

Context                     Function                     Lines        Purpose
-------                     --------                     -----        -------
Training loss               train_step() -> loss_fn()    L728-733     Remove idx 11-15 from CE loss
Hierarchical train loss     sharded_step() version       L1171-1176   Same, shard_map version
Validation/test loss        eval_step()                  L1008-1015   Remove idx 11-15 from CE loss
Validation/test accuracy    eval_step()                  L1017-1024   Remove idx 11-15 from accuracy

2.2 Mathematical Effect

Original CE loss:     L = mean(CE[0:26])             - average over all 26 positions
ignore_times CE:      L = mean(CE[0:11] + CE[16:26]) - average over 21 positions only

(+) denotes concatenation. Removed tokens: idx 11-12 (time_s), 13-15 (time_ns).

2.3 Design Motivation

Reason                      Explanation
------                      -----------
Time unpredictability       Absolute timestamps (HH:MM:SS) are meaningless for prediction
Loss perturbation           5/26 (19.2%) of loss comes from unpredictable time tokens
Preserves delta_t           delta_t_s/ns are NOT removed - intervals are informative

--------------------------------------------------------------------------------

3. 32-NODE TRAINING CONFIGURATION

3.1 SLURM / Hardware Configuration

Parameter               Value                     Description
---------               -----                     -----------
--nodes                 32                        SLURM node count
--gres=gpu              4                         GPUs per node
Total GPUs              128                       32 x 4
GPU model               NVIDIA GH200 (sm_90a)     96 GB HBM3, Grace CPU
Available memory        ~85.5 GB / GPU            96 GB SKU
Intra-node interconnect NV6                       ~159 GB/s per GPU pair
Per-GPU bandwidth       ~478 GB/s                 18 NVLinks x 26.56 GB/s
Inter-node interconnect Slingshot                 HPE Cray
--time                  00:30:00 / 24:00:00       Test / Production limits
--ntasks-per-node       1                         1 process per node (4 GPUs)
Account                 <your_account>                 an HPC cluster
Partition               workq                     an HPC cluster

3.2 Batch Size Configuration

Parameter               Value      Formula
---------               -----      -------
PER_GPU_BSZ             8          Env var, default 8
GPUS_PER_NODE           4          Fixed
NNODES                  32         SLURM
TOTAL_GPUS              128        32 x 4
GLOBAL_BSZ              1024       8 x 128
PER_PROCESS_BSZ         32         8 x 4 (DataLoader batch size)

3.3 Model Architecture (75M)

Parameter               Value                     Description
---------               -----                     -----------
d_model                 1024                      Hidden dimension H
n_layers                12                        Number of S5 blocks
n_message_layers        2                         Message encoder layers
n_book_pre_layers       1                         Order book pre-processing
n_book_post_layers      1                         Order book post-processing
blocks                  16                        SSM discretization blocks J
ssm_size_base           1024                      SSM latent state size P
block_size              64                        P / J = 1024 / 16
conj_sym                True                      Conjugate symmetry constraint
clip_eigs               True                      Force stability
bidirectional           False                     Unidirectional S5
activation_fn           half_glu1                 GLU variant
C_init                  trunc_standard_normal     Output initialization
discretization          zoh                       Zero-order hold
mode                    pool / none               Aggregation method
prenorm                 True                      Pre-LayerNorm
batchnorm               False                     Uses LayerNorm
Total params            ~75M
                                        |

3.4 Optimizer Configuration

Parameter       Value      Description
---------       ----       -----------
opt_config      standard   SSM params use Adam, others use AdamW
ssm_lr_base     5e-4       SSM param LR (B, Lambda, log_step, etc.)
lr_factor       1          Global LR = lr_factor x ssm_lr_base
Global LR       5e-4       For C, dense, encoder, etc.
weight_decay    0.05       Applies only to regular param group
p_dropout       0.0        No dropout
jax_seed        42         Random seed
LR schedule     optax      Computed inside JIT

3.5 Learning Rate Schedule

Phase                Formula                                       Steps                   Description
-----                -------                                       -----                   -----------
Warmup (linear)      lr = base x (step+1) / warmup                 1 epoch                 Linear ramp from 0 to 5e-4
Cosine Decay         lr = (base-min) x 0.5 x (1+cos(...)) + min    39 epochs               Decay from 5e-4 to 0
lr_min               0.0                                           -                       Lower bound

32-Node Specific Values:
Parameter            Value      Calculation
---------            -----      -----------
steps_per_epoch      ~549       train_size / (micro_bsz x num_devices x proc_count)
warmup_steps         ~549       1 epoch
cosine_steps         ~21,411    39 x 549
total_steps          ~21,960    40 x 549

3.6 Distributed Training / Sharding

Parameter              32N Value               Description
---------              ---------               -----------
Mesh type              2D (nodes, gpus)        Auto-enabled with hierarchical=True
Mesh shape             (32, 4)                 32 nodes x 4 GPUs
Data sharding          P((nodes,gpus), None)   Batch dim split across 128 GPUs
Param sharding         P() (replicated)        Each GPU holds full copy
Gradient sync          Hierarchical AllReduce  pmean(gpus) -> pmean(nodes)
Level 1 (NVLink)       jax.lax.pmean           Intra-node (4 GPUs), ~478 GB/s
Level 2 (Slingshot)    jax.lax.pmean           Across 32 nodes
shard_autotuning       False                   Avoids 16N scaling regression

3.7 NCCL Communication Configuration

Env Variable           Value              Description
------------           -----              -----------
NCCL_BUFFSIZE          2097152 (2MB)      Critical: 16N speedup from 34.8% to 83.3%
NCCL_TIMEOUT           3600 (1h)          Prevents false timeouts
NCCL_MIN_NCHANNELS     4                  Slingshot best practice
NCCL_NCHANNELS_PER_NET 4                  GH200 cluster recommended
NCCL_DEBUG             INFO               Set to WARN after debugging
NCCL_P2P_DISABLE       0 (default)        Setting to 1 causes 25x slowdown

3.8 XLA / JAX Environment

Env Variable / Flag                         32N Value      Description
-------------------                         ---------      -----------
XLA_PYTHON_CLIENT_PREALLOCATE               true           GPU memory pooling
XLA_PYTHON_CLIENT_MEM_FRACTION              0.80           Dual NCCL communicator limit
JAX_PLATFORMS                               cuda           Force CUDA backend
TF_GPU_ALLOCATOR                            cuda_malloc    Async GPU memory
CUDA_MODULE_LOADING                         EAGER          Load modules at startup
--xla_gpu_all_reduce_combine_threshold      128MB          Combine AllReduce ops
--xla_gpu_enable_latency_hiding_scheduler   true           Overlap compute/comm
--xla_gpu_enable_highest_priority_async     true           High-priority streams
--xla_gpu_shard_autotuning                  false          Avoid autotuner overhead

3.9 Data Loading Configuration

Parameter               Value                 Description
---------               -----                 -----------
dataset                 lobster-prediction    Dataset type
msg_seq_len             500                   History length per sample
use_book_data           True                  Include order book
use_simple_book         False                 Not volume image representation
book_transform          True                  On-the-fly transformation
book_depth              500                   Tick levels
merging                 padded                Message + book merge method
masking                 none                  Full autoregressive
n_data_workers          12                    CPU loading processes
random_offsets_train    True                  Random offset start per epoch
shuffle_train           True                  Shuffle training batches
ignore_times            True                  Branch core: remove absolute time
Distributed sampler     DistributedSampler    Rank-based auto-sharding

3.10 Training Loop Configuration

Parameter               Value                 Description
---------               -----                 -----------
epochs                  40                    Production epochs
early_stop_patience     1000                  Effectively disabled
lr_patience             4                     Informational (managed by optax)
reduce_factor           1.0                   No actual decay (cosine instead)
cosine_anneal           True                  Use cosine schedule
warmup_end              1 (epoch)             1 epoch linear warmup
curtail_epochs          None / 10             Full / truncated testing

3.11 Checkpoint Configuration

Parameter               Value                              Description
---------               -----                              -----------
Save interval           Every epoch                        save_interval_steps=1
Max kept                10                                 max_to_keep=10
Permanent keeps         Every 5th                          Epoch 0, 5, 10...
Async write             False                              fork() breaks NCCL
Save directory          checkpoints/...                    Rank 0 writes
Format                  Orbax PyTree                       CheckpointManager
Re-sharding             jax.jit(identity...)               Consistent NamedSharding

3.12 W&B Configuration

Parameter               Value              Description
---------               -----              -----------
USE_WANDB               True
wandb_project           lobs5-75M-B1
wandb_entity            anonymous
Rank 0                  online             Syncs to cloud
Other ranks             offline            Prevents duplicate cloud runs

--------------------------------------------------------------------------------

4. TRAINING METRICS

4.1 Per-Epoch W&B Metrics

W&B Metric Name         Type        Computation              Source Code       Description
---------------         ----        -----------              -----------       -----------
Training Loss           Train       mean(batch_losses)       lob/train/train.py:452  Mean CE over epoch
Val loss                Val         mean(concat_loss)        lob/train/train.py:453  Validation CE loss
Val Accuracy            Val         mean(accuracies)         lob/train/train.py:454  Token-level accuracy
Test Loss               Test        mean(concat_loss)        lob/train/train.py:455  Jan 2023 CE loss
Test Accuracy           Test        mean(accuracies)         lob/train/train.py:456  Jan 2023 accuracy
count                   Internal    Early stop counter       lob/train/train.py:457  val_loss patience
Learning rate count     Internal    LR decay counter         lob/train/train.py:458  patience counter
Opt acc                 Internal    Best val accuracy        lob/train/train.py:459  Tracked best acc
lr                      Hyperparam  lr_schedule(step)        lob/train/train.py:460  Current global LR
ssm_lr                  Hyperparam  ssm_lr_schedule(step)    lob/train/train.py:461  Current SSM LR
block_size              Config      ssm_size / blocks        lob/train/train.py:48   Logged at epoch 0

4.2 W&B Run Summary (Final)

Summary Metric          Description                                  Source
--------------          -----------                                  ------
Best Val Loss           Best validation loss across all epochs       lob/train/train.py:480
Best Val Accuracy       Best validation accuracy across all epochs   lob/train/train.py:481
Best Epoch              Epoch that achieved best Val Accuracy        lob/train/train.py:482
Best Test Loss          Test loss at the best Val Acc epoch          lob/train/train.py:483
Best Test Accuracy      Test accuracy at the best Val Acc epoch      lob/train/train.py:484

4.3 Checkpoint-Stored Metrics

Key                     Type        Description
---                     ----        -----------
metrics.loss_train      float       Last epoch training loss
metrics.loss_val_ar     float       Validation loss (AR mode)
metrics.loss_test_rnn   float       Test loss
metrics.acc_val_ar      float       Validation accuracy (AR mode)
metrics.acc_test_rnn    float       Test accuracy
config                  dict        Full args dictionary
model                   PyTree      Full TrainState (params + opt)

4.4 Low-Level Loss / Accuracy Functions

Function                Location                    Formula
--------                --------                    -------
cross_entropy_loss      lob/train/train_helpers.py:443    -logits[label] (log-softmax)
compute_accuracy        lob/train/train_helpers.py:452    argmax == label

Note: Vectorized over batch and sequence dimensions.

4.5 Impact of ignore_times on Metrics

Dimension                     No ignore_times   With ignore_times   Difference
---------                     ---------------   -----------------   ----------
Loss tokens / message         26                21                  -19.2%
Loss tokens / sample          13000             10500               500x26 -> 500x21
Model input tokens            13000             13000               Unchanged
Gradient dimensions           Full              Full                Time grads = 0

--------------------------------------------------------------------------------

5. OPTIONAL / ADVANCED METRICS

5.1 Per-Token CE Table (log_ce_tables=True)

Column Format           Description                          Epoch Dimension
-------------           -----------                          ---------------
tok                     Token index [0, seq_len)             Fixed
val_ce_{epoch}          Validation per-token CE              New/epoch
test_ce_{epoch}         Test per-token CE                    New/epoch
val_acc_{epoch}         Validation per-token accuracy        New/epoch
test_acc_{epoch}        Test per-token accuracy              New/epoch
train_ce_{epoch}        Training per-token CE                New/epoch

5.2 Console Output Metrics

Printed each epoch (examples):
- Train Loss: {:.5f}, Val Loss (AR): {:.5f}, Test Loss (RNN): {:.5f}
- Val Accuracy: {:.4f}, Test Accuracy: {:.4f}
- Best Val Loss: {:.5f} at Epoch {N}
- Epoch {N}, Batch {M} + Memory usage

5.3 Hierarchical AllReduce Metrics (hierarchical=True)

- pmean(loss, 'gpus') : Average loss across 4 local GPUs
- pmean(loss, 'nodes'): Average loss across 32 nodes
- pmean(grads, 'gpus') -> pmean(grads, 'nodes'): Gradient sync (core)

---

## 6. Performance Benchmarks (Verified Data Points)

### 6.1 Known Scaling Efficiency

| Config        | Nodes | GPUs | Step Time (s/step) | Efficiency | Key Config | Job ID  |
|:--------------|:------|:-----|:-------------------|:-----------|:-----------|:--------|
| 1N baseline   | 1     | 4    | ~0.29              | 100%       | 1D mesh    | -       |
| 4N hier.      | 4     | 16   | -                  | 114.6%     | 2D mesh    | -       |
| 16N hier.     | 16    | 64   | 0.66               | 83.3%      | BUFF=2MB   | 2426303 |
| **32N hier.** | **32**| **128** | **0.94**        | **58.5%**  | **BUFF=2MB** | **2424846** |

### 6.2 32N Throughput Estimates

| Metric                | Value              | Formula                    |
|:----------------------|:-------------------|:---------------------------|
| Step time             | 0.94 s/step        |                            |
| Tokens per step       | 13,312,000         | 1024 (BSZ) x 13000 (seq_len) |
| **Token throughput**  | **~14.2M tok/s**   | 13,312,000 / 0.94          |
| Samples per step      | 1024               | GLOBAL_BSZ                 |
| **Sample throughput** | **~1089 samp/s**   | 1024 / 0.94                |
| Steps per epoch       | ~549               |                            |
| **Time per epoch**    | **~8.6 min**       | 549 x 0.94 / 60            |
| **40 epoch total**    | **~5.7 h**         | 8.6 x 40 / 60              |
| **128 GPU-hours**     | **~731 GPU-h**     | 5.7 x 128                  |

---

## 7. Complete Parameter Reference: argparse vs baseline_full_autoreg vs ignore-times-shard-map

### 7.1 Three-Way Comparison

Source: argparse defaults from run_train.py, baseline from baseline_full_autoreg:bin/run_experiments/run_lobster_padded_large.sh,
this branch from train_full_autoreg.batch (32N).  **Bold** = differs from argparse default.

| #  | Parameter               | argparse Default           | baseline_full_autoreg (7 GPU, 1N)  | ignore-times-shard-map (32N)  | Category       |
|:---|:------------------------|:---------------------------|:-----------------------------------|:------------------------------|:---------------|
| 1  | --USE_WANDB             | True                       | True                               | True                          | W&B            |
| 2  | --wandb_project         | "LOBS5v2"                  | "LOBS5v2" (default)                | **"lobs5-75M-B1"**           | W&B            |
| 3  | --wandb_entity          | "anonymous"                   | "anonymous" (default)                 | **"anonymous"**            | W&B            |
| 4  | --dir_name              | "./data/LOBS5v2Cached/"    | **"/home/.../GOOG/2022"**          | **GOOG 2022 (this cluster)**     | Data           |
| 5  | --test_dir_name         | None                       | N/A (not in baseline)              | **JAN2023 path**             | Data           |
| 6  | --dataset               | "lobster-prediction"       | "lobster-prediction"               | "lobster-prediction"          | Data           |
| 7  | --masking               | "causal"                   | **"none"**                         | **"none"**                   | Data           |
| 8  | --use_book_data         | False                      | **True**                           | **True**                     | Data           |
| 9  | --merging               | "projected"                | **"padded"**                       | **"padded"**                 | Data           |
| 10 | --use_simple_book       | False                      | False                              | False                         | Data           |
| 11 | --book_transform        | False                      | **True**                           | **True**                     | Data           |
| 12 | --book_depth            | 500                        | 500 (default)                      | 500 (default)                 | Data           |
| 13 | --msg_seq_len           | 500                        | 500                                | 500                           | Data           |
| 14 | --micro_bsz             | 16                         | **4** (28/7 GPUs)                  | **8** (PER_GPU_BSZ)          | Data           |
| 15 | --num_devices           | 1                          | **7**                              | **4** (GPUS_PER_NODE)        | Hardware       |
| 16 | --n_data_workers        | 0                          | **7**                              | **12**                       | Data           |
| 17 | --random_offsets_train  | True                       | True                               | True                          | Data           |
| 18 | --shuffle_train         | True                       | True                               | True                          | Data           |
| 19 | --debug_overfit         | False                      | False                              | False                         | Debug          |
| 20 | --debug_loading         | False                      | False                              | False                         | Debug          |
| 21 | --enable_profiler       | False                      | False                              | False                         | Debug          |
| 22 | --ignore_times          | False                      | False (default)                    | **True**                     | **Core**       |
| 23 | --log_ce_tables         | False                      | N/A (not in baseline)              | False (default)               | Logging        |
| 24 | --hierarchical          | False                      | N/A (not in baseline)              | **True** (env var)           | **Distributed**|
| 25 | --d_model               | 32                         | **1024**                           | **1024**                     | Model          |
| 26 | --n_layers              | 6                          | **12**                             | **12**                       | Model          |
| 27 | --n_message_layers      | 2                          | 2 (default)                        | 2 (default)                   | Model          |
| 28 | --n_book_pre_layers     | 1                          | 1 (default)                        | 1 (default)                   | Model          |
| 29 | --n_book_post_layers    | 1                          | 1 (default)                        | 1 (default)                   | Model          |
| 30 | --ssm_size_base         | 32                         | **1024**                           | **1024**                     | Model          |
| 31 | --blocks                | 8                          | **16**                             | **16**                       | Model          |
| 32 | --C_init                | "trunc_standard_normal"    | "trunc_standard_normal"            | "trunc_standard_normal"       | Model          |
| 33 | --discretization        | "zoh"                      | "zoh" (default)                    | "zoh" (default)               | Model          |
| 34 | --mode                  | "none"                     | "none" (default)                   | "none" (default)              | Model          |
| 35 | --activation_fn         | "half_glu1"                | "half_glu1"                        | "half_glu1"                   | Model          |
| 36 | --conj_sym              | True                       | True (default)                     | True (default)                | Model          |
| 37 | --clip_eigs             | False                      | **True**                           | **True**                     | Model          |
| 38 | --bidirectional         | False                      | False                              | False                         | Model          |
| 39 | --dt_min                | 0.001                      | 0.001 (default)                    | 0.001 (default)               | Model          |
| 40 | --dt_max                | 0.1                        | 0.1 (default)                      | 0.1 (default)                 | Model          |
| 41 | --prenorm               | True                       | True                               | True                          | Optimizer      |
| 42 | --batchnorm             | True                       | **False**                          | **False**                    | Optimizer      |
| 43 | --bn_momentum           | 0.95                       | 0.95 (default)                     | 0.95 (default)                | Optimizer      |
| 44 | --epochs                | 100                        | **400**                            | **40**                       | Optimizer      |
| 45 | --early_stop_patience   | 1000                       | 1000 (default)                     | 1000 (default)                | Optimizer      |
| 46 | --ssm_lr_base           | 1e-3                       | **5e-4**                           | **5e-4**                     | Optimizer      |
| 47 | --lr_factor             | 1                          | 1                                  | 1                             | Optimizer      |
| 48 | --dt_global             | False                      | False                              | False                         | Optimizer      |
| 49 | --lr_min                | 0                          | 0 (default)                        | 0 (default)                   | Optimizer      |
| 50 | --cosine_anneal         | True                       | True (default)                     | True (default)                | Optimizer      |
| 51 | --warmup_end            | 1                          | 1                                  | 1                             | Optimizer      |
| 52 | --lr_patience           | 1000000                    | **4**                              | **4**                        | Optimizer      |
| 53 | --reduce_factor         | 0.9                        | 0.9 (default, ACTIVE)              | **1.0** (DEPRECATED, no-op)  | Optimizer      |
| 54 | --p_dropout             | 0.0                        | 0.0                                | 0.0                           | Optimizer      |
| 55 | --weight_decay          | 0.05                       | 0.05                               | 0.05                          | Optimizer      |
| 56 | --opt_config            | "standard"                 | "standard"                         | "standard"                    | Optimizer      |
| 57 | --jax_seed              | 1919                       | **42**                             | **42**                       | Optimizer      |
| 58 | --curtail_epochs        | None                       | None (default)                     | None (production)             | Optimizer      |
| 59 | --restore               | None                       | **checkpoint path**                | None (or env var)             | Checkpoint     |
| 60 | --restore_step          | None                       | **23**                             | None (or env var)             | Checkpoint     |
|    | (LR management)         | -                          | inject_hyperparams (manual)        | **optax schedule** (in JIT)  | Architecture   |
|    | (Distributed)           | -                          | pmap (single-node only)            | **jit + shard_map** (multi-N)| Architecture   |
|    | (Gradient sync)         | -                          | pmap pmean (local devices)         | **hierarchical AllReduce**   | Architecture   |

### 7.2 Key Differences: baseline vs this branch

| Aspect                   | baseline_full_autoreg                | ignore-times-shard-map                | Why changed                                           |
|:-------------------------|:-------------------------------------|:--------------------------------------|:------------------------------------------------------|
| **ignore_times**         | False (predict all 26 tokens)        | **True** (skip 5 time tokens)         | Time tokens unpredictable, pollute gradients           |
| **test_dir_name**        | N/A (val=test from same dir)         | **JAN2023** (separate dir)            | Temporal extrapolation evaluation                      |
| **hierarchical**         | N/A (single-node pmap)               | **True** (2D mesh shard_map)          | Multi-node DDP with NVLink+Slingshot hierarchy         |
| **epochs**               | 400                                  | **40**                                | 10x fewer epochs (128 GPU compensates)                 |
| **micro_bsz**            | 4 (28/7 GPUs)                       | **8** (PER_GPU_BSZ)                   | Hardware: 7 GPU -> 4 GPU/node                          |
| **num_devices**           | 7                                   | **4** (per node)                      | GH200 = 4 GPU/node (vs original 7-GPU server)         |
| **n_data_workers**        | 7                                   | **12**                                | More CPU cores on GH200 (72 per NUMA)                  |
| **LR management**        | inject_hyperparams (Python scalars)  | **optax schedule** (inside JIT)       | inject_hyperparams caused NCCL deadlock on multi-node  |
| **reduce_factor**         | 0.9 (actively used)                 | **1.0** (DEPRECATED, no-op)           | Replaced by optax cosine schedule                      |
| **wandb_project**         | "LOBS5v2"                           | **"lobs5-75M-B1"**                   | Experiment tracking separation                         |
| **wandb_entity**          | "anonymous"                            | **"anonymous"**                     | Different W&B account                                  |

---

## 8. Data Flow Overview (Pipeline ASCII Diagram)

```
+----------------------------------------------------------------------+
|                        DATA PIPELINE (32 Nodes)                       |
+----------------------------------------------------------------------+

     GOOG/2022/*.npy (249 days)          JAN2023/*.npy (9 days)
             |                                   |
             v                                   v
     +------------------------------------------------------+
     |              LOBSTER.setup()                          |
     |  train_files = 249 days (all)                         |
     |  val_files   = ~25 days (random 10% from train)      |
     |  test_files  = 9 days (separate directory)            |
     +------------------------------------------------------+
             |                    |                   |
             v                    v                   v
     +-------------+    +-------------+    +-------------+
     | LOBSTER_     |    | LOBSTER_     |    | LOBSTER_     |
     | Dataset      |    | Dataset      |    | Dataset      |
     | (train)      |    | (val)        |    | (test)       |
     | mask=none    |    | mask=none    |    | mask=none    |
     | rand_offset  |    |              |    |              |
     +------+------+    +------+------+    +------+------+
            |                  |                   |
            v                  v                   v
     +-----------------------------------------------------+
     |        torch DataLoader (per rank)                   |
     |   batch_size=micro_bsz*num_devices (per-process), workers=12 |
     |   DistributedSampler (epoch-aware shuffle)            |
     |   Output: (messages[per_process_bsz,500,26], book[...]) |
     +-----------------------+-----------------------------+
                             |
                             v
     +-----------------------------------------------------+
     |  prep_batch() + make_array_from_process_local_data   |
     |  -> inputs:  (msg[per_process_bsz,13000], book[...])  |
     |  -> labels:  [per_process_bsz, 13000]                |
     |  -> times:   [per_process_bsz, 13000]                |
     |  Sharding: P(('nodes','gpus'), None)                  |
     +-----------------------+-----------------------------+
                             |
                             v
     +-----------------------------------------------------+
     |     shard_map(sharded_step, mesh=2D(32,4))           |
     |                                                      |
     |  Per-shard (1 GPU, micro_bsz=8):                      |
     |    logits = model(msg[8,13000], book[8,...])          |
     |    ce = cross_entropy_loss(logits, labels)            |
     |    if ignore_times:                                   |
     |       ce = ce[:, 0:11] + ce[:, 16:26]  (per msg)     |
     |    loss = mean(ce)                                   |
     |    grads = grad(loss)                                |
     |                                                      |
     |    # Hierarchical AllReduce:                         |
     |    grads = pmean(grads, 'gpus')   <- NVLink 478GB/s  |
     |    grads = pmean(grads, 'nodes')  <- Slingshot        |
     |    state = apply_gradients(grads)                    |
     +-----------------------+-----------------------------+
                             |
                             v
     +-----------------------------------------------------+
     |  Per Epoch:                                          |
     |    W&B log: {Training Loss, Val loss, Val Accuracy,  |
     |              Test Loss, Test Accuracy, lr, ssm_lr,   |
     |              count, lr_count, opt_acc}                |
     |    Checkpoint: Orbax save (rank 0 writes)            |
     |    Early stop check (patience=1000, disabled)        |
     +-----------------------------------------------------+
```

---

## 9. Key Code Location Index

| Feature                        | File                          | Lines     | Description                          |
|:-------------------------------|:------------------------------|:----------|:-------------------------------------|
| **Batch script**               | `train_full_autoreg.batch`     | Full file | SLURM + environment + parameters     |
| Data path hardcoded            | `train_full_autoreg.batch`     | 94-96     | DATA_DIR, TEST_DIR                   |
| `--ignore_times=True`          | `train_full_autoreg.batch`     | 281       | Branch core configuration            |
| Parameter definitions          | `run_train.py`                 | 44-188    | All argparse parameters              |
| Multi-node initialization      | `run_train.py`                 | 191-234   | JAX distributed + SLURM              |
| Training main loop             | `lob/train/train.py`                 | 21-498    | Complete training logic              |
| W&B logging                    | `lob/train/train.py`                 | 449-476   | Per-epoch metrics                    |
| W&B summary                    | `lob/train/train.py`                 | 480-484   | Best metrics                         |
| Checkpoint saving              | `lob/train/train.py`                 | 374-404   | Orbax + multi-node sync              |
| LR schedule creation           | `lob/train/train_helpers.py`         | 68-111    | Warmup + cosine                      |
| cross_entropy_loss             | `lob/train/train_helpers.py`         | 443       | `-logits[label]`                     |
| compute_accuracy               | `lob/train/train_helpers.py`         | 452       | `argmax == label`                    |
| train_step                     | `lob/train/train_helpers.py`         | 685-752   | JIT training step                    |
| ignore_times (train)           | `lob/train/train_helpers.py`         | 728-733   | Remove idx 9-13 from loss            |
| hierarchical train_step        | `lob/train/train_helpers.py`         | 1115-1216 | shard_map version                    |
| ignore_times (hierarchical)    | `lob/train/train_helpers.py`         | 1171-1176 | Hierarchical version                 |
| eval_step                      | `lob/train/train_helpers.py`         | 946-1026  | Validation/test step                 |
| ignore_times (eval loss)       | `lob/train/train_helpers.py`         | 1008-1015 | Eval CE                              |
| ignore_times (eval accuracy)   | `lob/train/train_helpers.py`         | 1017-1024 | Eval accuracy                        |
| Message_Tokenizer              | `lob/encode/encoding.py`              | 279-358   | Token encoding definition            |
| Vocab construction             | `lob/encode/encoding.py`              | 220-278   | Vocabulary construction              |
| encode_msg                     | `lob/encode/encoding.py`              | 94-132    | Message encoding function            |
| decode_msg                     | `lob/encode/encoding.py`              | 158-186   | Message decoding function            |
| TIME_START_I / TIME_END_I      | `lob/train_helpers.py`               | 100-101   | Canonical encoder constants: 11, 15  |



================================================================================
10. LEGACY ENCODING ARCHIVE (22-token and 24-token Modes)
================================================================================

This section preserves the layouts, vocabularies, and performance characteristics
of the legacy 22-token and 24-token modes. The active training pipeline ONLY supports
26-token mode.

--------------------------------------------------------------------------------

10.1 Legacy 22-Token Mode Data Format

Each LOB message is 22 tokens, divided into 8 new event fields (18 tok) + 4 reference fields (4 tok).

Message Structure (22 tokens):
┌──────────────────── New Event Fields (18 tok) ────────────────────┐┌── Reference Fields (4 tok) ──┐
│ idx:  0    1    2-3   4    5    6-8    9-10   11-13              │ 14-15  16   17-18  19-21     │
│       │    │    │     │    │    │      │      │                  │  │      │    │      │        │
│      evt  dir  price size dt_s dt_ns  time_s time_ns            │ p_ref sz_r ts_ref tns_ref   │
└───────────────────────────────────────────────────────────────────┘└─────────────────────────────┘
                                        ^---- ignore_times removes these 5 tokens (idx 9-13)

Token Index   Field Name      Count   Type         Range                Description
-----------   ----------      -----   ----         -----                -----------
0             event_type      1       event_type   {1,2,3,4}            1=New, 2=Cancel, 3=Delete, 4=Trade
1             direction       1       direction    {0,1}                0=Ask, 1=Bid
2-3           price           2       sign+price   sign+/-1, val 999    Relative price (ticks)
4             size            1       size         [0, 9999]            Order quantity (shares)
5             delta_t_s       1       time         [0, 999]             Interval (seconds)
6-8           delta_t_ns      3       time         3x[0,999]            Interval (nanoseconds)
9-10          time_s          2       time         2x[0,999]            Exchange time sec (removed)
11-13         time_ns         3       time         3x[0,999]            Exchange time ns (removed)
14-15         price_ref       2       sign+price   Same as price        Reference price
16            size_ref        1       size         [0, 9999]            Reference quantity
17-18         time_s_ref      2       time         2x[0,999]            Reference time sec
19-21         time_ns_ref     3       time         3x[0,999]            Reference time ns

10.2 Legacy 22-Token Mode Vocabulary

Encoding Domain       Size    Token Range    Description
---------------       ----    -----------    -----------
Special tokens        4       0-3            MASK=0, HIDDEN=1, NA=2, START=3
time                  1000    4-1003         Digits 0-999, delimiter at [3,6,9,12]
event_type            4       1004-1007      Types 1-4
size                  10000   1008-11007     Quantity 0-9999
price                 1000    11008-12007    Price digits 0-999
sign                  2       12008-12009    Sign {-1, +1}
direction             2       12010-12011    Direction {0, 1}
Total n_classes       ~12012                 Returned by Vocab.__len__()

10.3 Legacy 22-Token Mode Sequence Dimensions

Parameter             Value      Formula      Description
---------             -----      -------      -----------
msg_seq_len           500        (CLI arg)    History messages per sample
MSG_LEN               22         (constant)   Tokens per message
seq_len (total)       11000      500 x 22     Total tokens per sample
Effective loss tok    8500       500 x 17     After removing time_s/ns
n_classes             ~12012     Vocab size   Output softmax classes

--------------------------------------------------------------------------------

10.4 Legacy 24-Token Mode Layout

24tok uses a base-1000 price and base-100 size but has only
1 token for price and price_ref, restricting the range. Vocab size is 2112.

Token layout:
  24tok: [evt:0, dir:1, price:2-3, size:4-5, dt_s:6, dt_ns:7-9, time_s:10-11, time_ns:12-14,
          price_ref_sign:15, price_ref:16, size_ref:17-18, time_ref:19-23]
  (22tok was: size=4, dt_s=5, time_s=9-10, time_ns=11-13 - indices shifted +1 by size 1→2 tokens)

--------------------------------------------------------------------------------

10.5 G2 Experiment Evidence (32N, 40ep comparison)

Comparing legacy performance across tokenizations:
  24tok: Val Acc 78.87%, Test Acc 75.89% (Job 2440076, W&B pmk5zbad)
  22tok: Val Acc 76.61%, Test Acc 73.45% (Job 2458353, W&B cgdexweb)

--------------------------------------------------------------------------------

10.6 Integrated 26-Token Inference

The in-repository runtime (`run/base_model/runtime/inference.py`) and generator
(`lob/inference_no_errcorr.py`) use the canonical 26-token encoder. Recurrent
models share the common carry adapter; Transformer and NSA use parallel prefix
prefill followed by cached autoregressive decoding. Checkpoint architecture
metadata is resolved through the same strict registry as training.


#SBATCH --contiguous
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid010655,nid010052,nid010442,nid010093,nid010990,nid010499,nid011290,nid011263,nid010993,nid[010226-010255],nid[010340-010347],nid011108

# TRAIN_FULL_AUTOREG_APPENDIX
```
