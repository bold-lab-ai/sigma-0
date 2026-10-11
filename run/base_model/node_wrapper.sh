#!/bin/bash
# ========== Node Wrapper ==========
# Called by srun with --export=ALL: all env vars from batch script are available.
# 1 process per node, all GPUs visible.

# Group-writable by default: new files/dirs (checkpoints, wandb, logs) are g+rw
umask 002

# TMPDIR/WANDB_DIR must be set BEFORE any Python import. Do not trust Slurm's
# shared /local/user/<uid> TMPDIR: a sibling-job epilog can remove that path
# while this process is importing (observed in matched KDA smoke 5788482_3).
# A rank-local directory below node-local /tmp is isolated from that cleanup.
SIGMA0_JOB_TMPDIR="${SIGMA0_JOB_TMPDIR:-/tmp/${USER:-${LOGNAME:-unknown}}/sigma0/${SLURM_JOB_ID:-nojid}_${SLURM_PROCID:-0}}"
mkdir -p "$SIGMA0_JOB_TMPDIR"
[ -d "$SIGMA0_JOB_TMPDIR" ] && [ -w "$SIGMA0_JOB_TMPDIR" ] || {
    echo "[Wrapper] FATAL: job-local temporary directory is unavailable: $SIGMA0_JOB_TMPDIR" >&2
    exit 1
}
export TMPDIR="$SIGMA0_JOB_TMPDIR"
export WANDB_DIR="${WANDB_DIR:-$TMPDIR/wandb}"
mkdir -p "$WANDB_DIR"

# Default production runs to W&B online mode, but allow smoke jobs to run
# offline when the configured entity/project is not writable from this account.
export WANDB_MODE="${WANDB_MODE:-online}"

# Per-node logging: all nodes write to individual log files for debugging
# Uses exec (process-local redirect), NOT srun --output (which overflows at 32N+)
LOG_DIR="${NODE_LOG_DIR:-${WORKDIR:-${SLURM_SUBMIT_DIR:-.}}/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || true
exec > "$LOG_DIR/training_${SLURM_JOB_ID}_node${SLURM_PROCID:-0}.log" 2>&1

echo "========================================"
echo "[Wrapper] Running on node: $(hostname)"
echo "[Wrapper] SLURM_NODEID: ${SLURM_NODEID:-N/A}"
echo "[Wrapper] SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "[Wrapper] SLURM_LOCALID: ${SLURM_LOCALID:-N/A}"
echo "[Wrapper] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# De-anonymization hook (double-blind release): real Isambard-AI paths live in a
# gitignored credentials/ dir, never committed. If present, source it so QUANT_ROOT
# (and SQUASHFS_DIR etc.) resolve to real paths; otherwise the /path/to/... default
# below remains and a fresh clone must set QUANT_ROOT itself. See credentials/real_env.sh.
_CRED_ENV="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}/credentials/real_env.sh"
[ -f "$_CRED_ENV" ] && { echo "[Wrapper] sourcing de-anonymized paths: $_CRED_ENV"; . "$_CRED_ENV"; }

# Activate conda env directly (conda.sh has hardcoded broken paths to user's home)
# Post-migration: /path/to/miniforge3 is gone; canonical path is now
# /path/to/quant/miniforge3.
CONDA_ENV=${CONDA_ENV:-base}
QUANT_ROOT=${QUANT_ROOT:-/path/to/quant}
if [ "$CONDA_ENV" = "base" ]; then
  export CONDA_PREFIX=$QUANT_ROOT/miniforge3
else
  export CONDA_PREFIX=$QUANT_ROOT/miniforge3/envs/$CONDA_ENV
fi
export PATH=$CONDA_PREFIX/bin:$PATH
echo "[Wrapper] Conda env: $CONDA_ENV ($CONDA_PREFIX)"
echo "[Wrapper] Python: $(which python) ($(python --version 2>&1))"

# Load CUDA module (for cuDNN/cuBLAS shared libs)
module load cuda/12.6
# Re-prepend conda bin: module load cuda/12.6 puts system ptxas (12.6, PTX ISA 8.5)
# before conda's ptxas (12.9, PTX ISA 8.9). Triton 3.4.0 emits PTX 8.7 → needs 12.7+.
export PATH=$CONDA_PREFIX/bin:$PATH

# Set LD_LIBRARY_PATH
TORCH_LIB=$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib
# NCCL override: use custom-built NCCL 2.29.3 (fixes ARM CAS weak failure in proxy.cc)
# Built from source (commit 25368a7) with GCC 12.3 (strong CAS on aarch64) + CUDA 12.6, sm_90.
# History: lobmax conda had 2.29.2 (still has weak CAS under GCC < 10 codepath),
#          lob env has 2.28.9 (ARM CAS bug). Source build is the definitive fix.
NCCL_LIB_OVERRIDE=$QUANT_ROOT/nccl-2.29.3/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$NCCL_LIB_OVERRIDE:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# CRITICAL: xla_cuda_plugin.so has DT_RPATH (not RUNPATH) pointing to conda's nvidia/nccl/lib.
# DT_RPATH is searched BEFORE LD_LIBRARY_PATH, so our override above is silently bypassed.
# LD_PRELOAD forces our NCCL to load first, overriding any RPATH resolution.
# The base conda env ships NCCL 2.29.3+cuda12.9 (also fixed), but LD_PRELOAD guarantees
# our GCC 12.3 source build loads regardless of which conda env is active.
export LD_PRELOAD=$NCCL_LIB_OVERRIDE/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}

# NCCL OFI plugin for cross-node communication via Slingshot/libfabric
# Upgraded from system aws-ofi-nccl 1.8.1 → 1.18.0 (2026-02-25)
# 1.8.1 was 10 versions behind, caused 512N (2048 GPU) NCCL comm init hang.
# Fallback: AWS_OFI_NCCL_LIB=/path/to/nccl-plugins/.../aws-ofi-nccl-1.8.1-.../lib
AWS_OFI_NCCL_LIB=${AWS_OFI_NCCL_LIB:-$QUANT_ROOT/aws-ofi-nccl-1.18.0/lib}
export LD_LIBRARY_PATH=$AWS_OFI_NCCL_LIB:/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH
echo "[OFI] aws-ofi-nccl: $AWS_OFI_NCCL_LIB"

# Verify NCCL version (must be 2.29.x, NOT 2.28.x)
echo "[NCCL] Override lib path: $NCCL_LIB_OVERRIDE"
echo "[NCCL] Library: $(ls -la $NCCL_LIB_OVERRIDE/libnccl.so.2 2>/dev/null || echo NOT_FOUND)"
NCCL_VER=$(strings $NCCL_LIB_OVERRIDE/libnccl.so.2 2>/dev/null | grep "^NCCL version.*compiled" | head -1)
echo "[NCCL] Version: ${NCCL_VER:-UNKNOWN}"
echo "[NCCL] LD_PRELOAD: $LD_PRELOAD"
# Sanity: our build says "cuda12.6", conda's says "cuda12.9"
if echo "$NCCL_VER" | grep -q "cuda12.6"; then
    echo "[NCCL] OK: source-built NCCL confirmed (cuda12.6)"
else
    echo "[NCCL] WARNING: Expected cuda12.6 (source build), got: $NCCL_VER"
fi

# JAX environment
export XLA_PYTHON_CLIENT_PREALLOCATE=true
# 2D mesh (hierarchical) creates two NCCL communicators (gpus+nodes), more buffer needed
# 32+ nodes: 0.80 (XLA HLO planner is greedy — 0.85 just makes it allocate MORE, not leave headroom)
#   Job 2439132: MEM=0.80 → XLA requested 73.5 GiB (96.7% of 76 GB) → OOM epoch 11
#   Job 2439639: MEM=0.85 → XLA requested 78.7 GiB (96.4% of 81.6 GB) → OOM step 0
#   Fix: reduce PER_GPU_BSZ (12→10), not increase MEM_FRACTION
# 8-16N: 0.85 (moderate overhead)
# Override with MEM_FRACTION env var for BSZ tuning (e.g. MEM_FRACTION=0.85)
if [ -z "${MEM_FRACTION}" ]; then
  if [ "${NNODES}" -ge 32 ] && [ "${HIERARCHICAL}" = "True" ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
  elif [ "${NNODES}" -ge 8 ] && [ "${HIERARCHICAL}" = "True" ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80  # was 0.85, OOM on NCCL buffer alloc (job 2898567)
  else
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
  fi
else
  export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}
fi
export JAX_PLATFORMS="cuda"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Persistent compilation cache: skip XLA recompilation on resume (ref: MaxText pyconfig.py:310)
# Cache key = HLO module fingerprint (includes model config + sharding + device info)
# Same model + same node count → cache hit → skip compile (~30-120s saved per resume)
# Also auto-enables xla_gpu_per_fusion_autotune_cache (JAX 0.9.0.1 default behavior)
# Cache dir must be on shared Lustre filesystem (accessible by all nodes across jobs)
export JAX_COMPILATION_CACHE_DIR="$QUANT_ROOT/jax_cache_lobs5"

# CAVEAT — XLA FLAGS FOR MULTI-HOST AUTOTUNER
# autotune_level=0 is FORBIDDEN — XLA AutoTune (kernel fusion) is why we use JAX
#
# JAX 0.9.0 had multi-host autotuner crash: autotuner.cc:260 DEVICE_TYPE_INVALID
# Root cause: non-deterministic iteration + unsorted sharding across hosts
# Fixed in JAX 0.9.0.1 via XLA#36579 + XLA#36755
# CAVEAT: DO NOT re-enable these flags unless downgrading below JAX 0.9.0.1
# CAVEAT: ssm-stable, HyperscaleES, MaxText all use XLA defaults (no flags)
# Rollback to JAX 0.9.0: pip install jax==0.9.0 jaxlib==0.9.0 jax-cuda12-pjrt==0.9.0 jax-cuda12-plugin==0.9.0
# export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false --xla_gpu_shard_autotuning=false}"
# Force all CUDA modules to load at startup (vs LAZY default which loads on first use).
# Why EAGER: in multi-node training, lazy loading causes non-deterministic load timing
# across nodes → NCCL collective timeouts and XLA autotuner device-binding races.
# Trade-off: slightly slower startup, but eliminates mid-training CUDA load stalls.
# 128N+: NCCL comm init + first collective can take 10-30 min at 2048 GPU scale
if [ "${NNODES}" -ge 128 ]; then
  FIRST_COLLECTIVE_TIMEOUT=1800  # 30 min (default 600s = 10 min)
fi

# XLA AllReduce fusion for multi-node scaling (ref: MaxText GPU config)
# Without this, XLA creates 101 independent AllReduce ops (one per param tensor).
# 16-rank ring on 4N has 91ms/op launch latency → 101*91ms = 9.2s per step.
# With 128MB threshold, XLA combines into 2-3 large AllReduces → ~3*91ms = 0.27s.
# Profiler evidence: jobs 2381959(2N) vs 2381960(4N), 101 ops/step both configs.
# Latency hiding scheduler overlaps remaining AllReduce with compute.
export XLA_FLAGS="${XLA_FLAGS} \
  --xla_gpu_all_reduce_combine_threshold_bytes=134217728 \
  --xla_gpu_enable_latency_hiding_scheduler=true \
  --xla_gpu_enable_highest_priority_async_stream=true \
  --xla_gpu_nccl_terminate_on_error=true \
  --xla_gpu_nccl_termination_timeout_seconds=600 \
  --xla_gpu_first_collective_call_terminate_timeout_seconds=${FIRST_COLLECTIVE_TIMEOUT:-600}"
# NOTE (G11): xla_gpu_first_collective_call_terminate_timeout_seconds is the correct flag
# for thunk init rendezvous timeout. xla_gpu_executable_terminate_timeout_seconds does NOT
# exist in JAX 0.9.0.1 (FATAL: Unknown flag, Job 2476156).
# CAVEAT: do NOT use --xla_gpu_all_reduce_blueconnect_num_devices_per_host=4 with shard_map
# BlueConnect decomposes AllReduce into RS+AR+AG, but shard_map already does 2-level decomposition.
# Result: 3x slowdown (3.55 s/step vs baseline 1.18 s/step). Verified job 2440967.

# Disable shard_autotuning for mini-epoch training:
# XLA re-autotunes train_step after every eval, causing ~45s + 30-step ramp per mini-epoch boundary.
# Evidence: Job 2476205 (2N, MINI_EPOCHS=3): step 99→100 drops from 6 it/s to 116s/it.
# Also beneficial at 8+ nodes: 32N off=0.94 s/step (58.4% eff) vs 16N on=1.58 s/step (34.8% eff).
# The old CAVEAT (JIT >30min at 16N) was for 1D DDP without shard_map; not applicable to 2D mesh.
if [ "${HIERARCHICAL}" = "True" ]; then
  export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_shard_autotuning=false"
  echo "[XLA] hierarchical + mini-epoch: shard_autotuning disabled"
fi

export CUDA_MODULE_LOADING=EAGER
echo "[XLA] XLA_FLAGS=${XLA_FLAGS}"

# NCCL config
# NCCL_TIMEOUT=3600 did NOT prevent a 6h hang at 32N (job 2426449, epoch 14 step 356).
# Multiple timeout mechanisms for broader NCCL version compatibility:
export NCCL_TIMEOUT=600                     # 10 min (NCCL 2.19+, seconds)
export NCCL_BLOCKING_WAIT=0                 # Non-blocking wait mode (NCCL 2.x)
export NCCL_ASYNC_ERROR_HANDLING=1           # Async error handling (NCCL 2.13+)
export NCCL_LAUNCH_ORDER_IMPLICIT=1         # Implicit ordering for multi-communicator ops (NCCL 2.26+)

# NCCL_BUFFSIZE=2MB — CRITICAL for multi-node shard_map performance
# Experimentally verified (C5a, 2026-02-21):
#   16N + BUFF=default(4MB): 1.58 s/step (34.8% eff) — Jobs 2424845, 2425406, 2426301
#   16N + BUFF=2MB:          0.66 s/step (83.3% eff) — Job 2426303  → 2.4x faster!
#   32N + BUFF=2MB:          0.94 s/step (58.5% eff) — Job 2424846
# With 40 NCCL channels (default on GH200), 4MB per channel saturates Slingshot
# injection bandwidth. 2MB reduces per-chunk size, improving pipeline efficiency.
# This is the sole cause of the previous 16N<32N efficiency anomaly.
if [ "${NNODES}" -ge 2 ]; then
  export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-2097152}
  echo "[NCCL] Multi-node: NCCL_BUFFSIZE=${NCCL_BUFFSIZE} ($((NCCL_BUFFSIZE / 1048576))MB)"
fi

# GH200 cluster Slingshot best practices (ref: docs.example.com, NCCL Issue #1272)
# Experimentally: no measurable impact alone (1.60 vs 1.58 s/step), but recommended.
export NCCL_MIN_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=4
# CAVEAT — DO NOT re-enable NCCL_P2P_DISABLE=1
# GH200 nodes HAVE NV6 (6x NVLink bonded, 478 GB/s) between all 4 GPUs.
# P2P_DISABLE=1 forces intra-node comm to SHM (CPU memcpy, ~50 GB/s),
# causing 25x slowdown on 4N+ (11.2 s/step vs 0.46 s/step).
# 2N is unaffected because each NCCL comm has localRanks=1 (no intra-node comm).
# Verified: nvidia-smi topo -m shows NV6 between all GPU pairs.
#export NCCL_P2P_DISABLE=1

# NCCL collective algorithm — both TREE and RING are slow on 4N Slingshot
# NCCL_ALGO=TREE: crash, AllGather does not support TREE — job 2382073
# NCCL_ALGO=allreduce:tree: 21.83 s/step, TREE 2.2x SLOWER than RING — job 2382095
# NCCL_ALGO default RING: 9.78 s/step — job 2381259
# Root cause is XLA AllReduce combine threshold (see XLA_FLAGS above), not NCCL algo.
#export NCCL_ALGO="allreduce:tree"

# Slingshot/CXI optimization — PERFORMANCE tuning
# TESTED job 2382150: 13.36 s/step — 37% WORSE than baseline (9.78).
# Root cause of regression: NCCL_CROSS_NIC=1 + NCCL_NET_GDR_LEVEL=PHB, NOT the CXI hang-prevention vars.
# These two are harmful on our Slingshot topology. Do not re-enable without single-variable benchmarking.
#export NCCL_CROSS_NIC=1                   # CAVEAT: tested harmful (job 2382150)
#export NCCL_NET_GDR_LEVEL=PHB             # CAVEAT: tested harmful (job 2382150)
#export NCCL_PROTO=^LL128                  # CAVEAT: tested harmful — 23% regression (1.12 vs 0.91 s/step, job 2447647 vs 2447130)
# FI_CXI_DEFAULT_CQ_SIZE, TX_SIZE, RX_MATCH_MODE: moved to ≥128N block below (was TODO)

# Slingshot CXI resilience — prevent NCCL deadlocks at 8+ nodes
# These are RESILIENCE tuning — no normal-path performance impact, only fault recovery.
# Root cause analysis (2026-02-23, HLO profiling + deep research):
#   - eval_step has ZERO NCCL collectives (HLO verified, Job 2440089)
#   - train_step has 336 AllReduce/step (169 intra-node + 167 inter-node)
#   - 32N × 336 × 549 steps/epoch × 6 epochs = ~1.1M collectives before hang
#   - NCCL 2.29.3 fixes ARM CAS weak failure (exactly our GH200 ARM platform)
#   - CXI eager message race condition is CSCS+this cluster+ALCF consensus workaround
if [ "${SLURM_NNODES:-1}" -ge 8 ]; then
  # --- Existing resilience ---
  export FI_CXI_RDZV_RETRIES=100           # default=5, survive transient Slingshot fabric errors
  export FI_CXI_OFLOW_BUF_SIZE=8388608     # 8MB overflow buffer (prevent CXI ENOMEM under bursty traffic)
  export FI_CXI_OFLOW_BUF_COUNT=6          # default=1, more overflow buffers for burst absorption
  export FI_CXI_REQ_BUF_SIZE=8388608       # 8MB request buffer (reduce flow control stalls)
  export FI_CXI_REQ_BUF_COUNT=6            # default=1, more request buffers for 128+ GPU bursts

  # --- NEW: CXI hang prevention (CSCS + this cluster + ALCF consensus) ---
  # Disable eager messages to prevent CXI race condition under high concurrency.
  # CXI eager path buffer management has race condition at 128 GPU bursty traffic.
  # Setting all three to 0 forces rendezvous-only path for all message sizes.
  export FI_CXI_RDZV_GET_MIN=0             # Disable eager GET minimum size
  export FI_CXI_RDZV_THRESHOLD=0           # Force all messages through rendezvous
  export FI_CXI_RDZV_EAGER_SIZE=0          # No eager data in rendezvous
  export FI_CXI_RDZV_PROTO=alt_read        # Alternate read protocol (ALCF verified up to 540 nodes)

  # --- NEW: Host register deadlock prevention (this cluster docs) ---
  # Multi-process per GPU → page locking competition → deadlock
  export FI_CXI_DISABLE_HOST_REGISTER=1    # Prevent host buffer GPU registration deadlock
  export FI_MR_CACHE_MONITOR=userfaultfd   # Memory registration cache monitor

  # --- NEW: Prevent GPU-aware MPI + NCCL collision (CSCS docs) ---
  export MPICH_GPU_SUPPORT_ENABLED=0       # "easily leads to deadlocks" - CSCS

  echo "[CXI] ${SLURM_NNODES}N: Full CXI resilience (RDZV_RETRIES=100, eager=off, alt_read, host_reg=off)"
fi

# === 512-NODE (2048 GPU) SCALING: CXI resource limits ===
# Job 2476315: 512N hangs after "Connected all rings" — never reaches first collective.
# CSCS documented: "NCCL alltoall benchmarks stop at 256 GPUs...gets stuck on 512+"
# Root cause: CXI completion queue (default=512 entries) overflows at ~256 peers.
# Each rank needs ~N_peers entries in CQ; 2048 ranks → 400% overflow → silent hang.
# 4N/64N work because CQ utilization stays under 100%.
if [ "${SLURM_NNODES:-1}" -ge 128 ]; then
  export FI_CXI_DEFAULT_CQ_SIZE=262144       # default=512, overflow at ~256 peers (CSCS uses 131072; 2x for dual-comm 2D mesh)
  export FI_CXI_DEFAULT_TX_SIZE=32768         # default=256, insufficient for 512+ peer connections
  export FI_CXI_RX_MATCH_MODE=software        # hardware match table exhausts at 500+ endpoints

  # NCCL bootstrap: 2048 ranks → TCP all-to-one bottleneck on rank 0
  export NCCL_SOCKET_RETRY_CNT=100            # default=34, more retries for congested accept()
  export NCCL_SOCKET_RETRY_SLEEP_MSEC=200     # default=100ms, backoff to reduce stampede

  # Limit NCCL channels to reduce CXI connection count (2 comms × N_channels × N_peers)
  export NCCL_MAX_NCHANNELS=16                # default auto (up to 32), cap at 16 for 512N

  ulimit -s 16384                              # 16MB stack (NCCL graph search at 2048 ranks)

  echo "[CXI] ${SLURM_NNODES}N: 2048-GPU scaling (CQ=262144, TX=32768, RX=software, channels≤16)"
fi

# Multi-node JAX distributed info
echo "[Wrapper] SLURM_PROCID=${SLURM_PROCID:-0} (process rank)"
echo "[Wrapper] SLURM_NNODES=${SLURM_NNODES:-1} (total nodes)"
echo "[Wrapper] JAX_COORDINATOR_ADDRESS=${JAX_COORDINATOR_ADDRESS:-none}"
# NCCL debug for cross-node verification (set to WARN after verified)
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# CUDA config
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$HOME/.nv/ComputeCache" || true

echo "[Wrapper] Available GPUs:"
nvidia-smi --list-gpus | head -4

# === FLOPs profiling (set ENABLE_DMON=1) ===
# Background nvidia-smi dmon captures HMMA/FP32/FP16 activity per GPU per second
# for the lifetime of the python training process. CSV is per-node, named by
# SLURM_JOB_ID + SLURM_PROCID, written under DMON_OUTPUT_DIR (or $WORKDIR/dmon_logs).
# Default ENABLE_DMON=0 → no behavior change for production runs.
DMON_PID=""
DMON_LOG=""
cleanup_dmon() {
    if [ -n "$DMON_PID" ]; then
        kill "$DMON_PID" 2>/dev/null || true
        wait "$DMON_PID" 2>/dev/null || true
        echo "[dmon] stopped, log: $DMON_LOG ($(wc -l < "$DMON_LOG" 2>/dev/null || echo 0) lines)"
    fi
}
trap cleanup_dmon EXIT
if [ "${ENABLE_DMON:-0}" = "1" ]; then
    DMON_DIR="${DMON_OUTPUT_DIR:-${WORKDIR:-.}/dmon_logs}"
    mkdir -p "$DMON_DIR" 2>/dev/null || true
    DMON_LOG="$DMON_DIR/dmon_${SLURM_JOB_ID}_node${SLURM_PROCID:-0}.csv"
    nvidia-smi dmon --gpm-metrics=2,3,7,12,13 -d 1 -c 3600 -o T -f "$DMON_LOG" &
    DMON_PID=$!
    echo "[dmon] launched PID=$DMON_PID -> $DMON_LOG"
fi

# === Squashfs data mount (optional, gated by SQUASHFS_MODE) ===
# When SQUASHFS_MODE=1, the per-node squashfuse mount of the YYYY-MM shard
# replaces the on-Lustre DATA_ROOT for the python training process. Closes
# the metadata-storm anti-pattern (488 ticker-dirs × N workers stat'ing
# 1M+ inodes at startup) and turns dataloader IO into block reads of one
# pre-striped Lustre file per node. Pilot single-month: SQUASHFS_MONTH=YYYY-MM
# selects the shard. Multi-month support will derive the shard list from
# TRAIN_DATE_RANGE+TEST_DATE_RANGE and mount each under a YYYY-MM subdir.
SQUASHFS_MOUNT=""
SQUASHFS_MULTI_MOUNT_ROOT=""
cleanup_squashfs() {
    # Single-mount cleanup
    if [ -n "$SQUASHFS_MOUNT" ] && mountpoint -q "$SQUASHFS_MOUNT" 2>/dev/null; then
        fusermount -u "$SQUASHFS_MOUNT" 2>/dev/null || true
        echo "[squashfs] unmounted $SQUASHFS_MOUNT"
    fi
    # Multi-mount cleanup: unmount each subdir under the root
    if [ -n "$SQUASHFS_MULTI_MOUNT_ROOT" ] && [ -d "$SQUASHFS_MULTI_MOUNT_ROOT" ]; then
        for d in "$SQUASHFS_MULTI_MOUNT_ROOT"/*/; do
            [ -d "$d" ] || continue
            if mountpoint -q "$d" 2>/dev/null; then
                fusermount -u "$d" 2>/dev/null || true
                echo "[squashfs] unmounted $d"
            fi
        done
    fi
}
# === Multi-shard mount (SQUASHFS_MULTI_MODE=1) ===
# When SQUASHFS_MULTI_MODE=1, mount N shards (one per YYYY-MM month) under
# $TMPDIR/sp500_squashfs/<YYYY-MM>/, then export DATA_ROOT as the comma-joined
# list of mount points. The dataloader's discover_ticker_files now accepts
# comma-separated multi-root and combines per-ticker file lists across roots.
#
# SQUASHFS_MONTHS env: "all" (mount every shard_*.squashfs found) OR a
# comma-separated list of YYYY-MM (e.g. "2023-09,2023-04,2023-08").
if [ "${SQUASHFS_MULTI_MODE:-0}" = "1" ]; then
    SQUASHFS_DIR="${SQUASHFS_DIR:-/path/to/project/lob_preproc_sp500_squashfs}"
    SQUASHFS_MULTI_MOUNT_ROOT="${SQUASHFS_MULTI_MOUNT_ROOT:-$TMPDIR/sp500_squashfs_${SLURM_JOB_ID:-manual}_${SLURM_PROCID:-0}}"
    SQUASHFS_MONTHS="${SQUASHFS_MONTHS:-all}"

    if [ "$SQUASHFS_MONTHS" = "all" ]; then
        echo "[squashfs] FATAL: SQUASHFS_MONTHS=all is disabled. Pass an explicit comma-separated month list." >&2
        exit 1
    fi
    MONTHS=$(echo "$SQUASHFS_MONTHS" | tr ',' ' ')

    if [ -z "$MONTHS" ]; then
        echo "[squashfs] FATAL: no months resolved (SQUASHFS_MONTHS=$SQUASHFS_MONTHS)" >&2
        exit 1
    fi

    n_total=$(echo "$MONTHS" | wc -w)
    echo "[squashfs] multi-mode: mounting $n_total shards from $SQUASHFS_DIR"
    mkdir -p "$SQUASHFS_MULTI_MOUNT_ROOT"
    DATA_ROOT_LIST=""
    n_mounted=0
    for ym in $MONTHS; do
        SHARD="$SQUASHFS_DIR/shard_${ym}.squashfs"
        MOUNT="$SQUASHFS_MULTI_MOUNT_ROOT/${ym}"
        if [ ! -f "$SHARD" ]; then
            echo "[squashfs] WARN: shard missing for $ym ($SHARD), skipping"
            continue
        fi
        mkdir -p "$MOUNT"
        if mountpoint -q "$MOUNT" 2>/dev/null; then
            echo "[squashfs]   $ym already mounted (reusing)"
        else
            if ! squashfuse "$SHARD" "$MOUNT" 2>/dev/null; then
                echo "[squashfs] FATAL: squashfuse failed for $ym" >&2
                exit 1
            fi
        fi
        DATA_ROOT_LIST="${DATA_ROOT_LIST:+$DATA_ROOT_LIST,}$MOUNT"
        n_mounted=$((n_mounted + 1))
    done
    echo "[squashfs] mounted $n_mounted/$n_total shards"

    trap 'cleanup_dmon; cleanup_squashfs' EXIT

    export DATA_ROOT_ORIG="${DATA_ROOT:-}"
    export DATA_ROOT="$DATA_ROOT_LIST"
    echo "[squashfs] DATA_ROOT (multi) = $n_mounted comma-joined paths"
    # Show first 2 + count
    echo "[squashfs]   first: $(echo "$DATA_ROOT" | cut -d, -f1)"
    [ "$n_mounted" -gt 1 ] && echo "[squashfs]   last:  $(echo "$DATA_ROOT" | rev | cut -d, -f1 | rev)"

elif [ "${SQUASHFS_MODE:-0}" = "1" ]; then
    SQUASHFS_DIR="${SQUASHFS_DIR:-/path/to/project/lob_preproc_sp500_squashfs}"
    SQUASHFS_MOUNT="${SQUASHFS_MOUNT_OVERRIDE:-$TMPDIR/sp500_squashfs_${SLURM_JOB_ID:-manual}_${SLURM_PROCID:-0}}"
    SQUASHFS_MONTH="${SQUASHFS_MONTH:-2026-01}"

    SHARD="$SQUASHFS_DIR/shard_${SQUASHFS_MONTH}.squashfs"
    if [ ! -f "$SHARD" ]; then
        echo "[squashfs] FATAL: shard not found: $SHARD" >&2
        exit 1
    fi

    mkdir -p "$SQUASHFS_MOUNT"
    if mountpoint -q "$SQUASHFS_MOUNT" 2>/dev/null; then
        echo "[squashfs] already mounted at $SQUASHFS_MOUNT (reusing)"
    else
        if ! squashfuse "$SHARD" "$SQUASHFS_MOUNT"; then
            echo "[squashfs] FATAL: squashfuse failed: $SHARD -> $SQUASHFS_MOUNT" >&2
            exit 1
        fi
        n_visible=$(ls "$SQUASHFS_MOUNT" 2>/dev/null | wc -l)
        echo "[squashfs] mounted $SHARD at $SQUASHFS_MOUNT ($n_visible tickers visible)"
    fi

    # Chain cleanup: dmon + squashfs unmount on EXIT
    trap 'cleanup_dmon; cleanup_squashfs' EXIT

    # Override DATA_ROOT for the python launch
    export DATA_ROOT_ORIG="${DATA_ROOT:-}"
    export DATA_ROOT="$SQUASHFS_MOUNT"
    echo "[squashfs] DATA_ROOT overridden: $DATA_ROOT (was: $DATA_ROOT_ORIG)"

    # Expose data index. Prefer in-shard $DATA_ROOT/index.json (future builds).
    # If absent (jan 2026 was built before in-shard index emit), fall back to
    # sidecar $SQUASHFS_DIR/index_<MONTH>.json. Either path makes the dataloader
    # skip ~9740 stat+header reads at startup.
    if [ -f "$SQUASHFS_MOUNT/index.json" ]; then
        echo "[squashfs] in-shard index found: $SQUASHFS_MOUNT/index.json"
    else
        SIDECAR="$SQUASHFS_DIR/index_${SQUASHFS_MONTH}.json"
        if [ -f "$SIDECAR" ]; then
            export DATA_INDEX_JSON="$SIDECAR"
            echo "[squashfs] sidecar index exposed: DATA_INDEX_JSON=$SIDECAR"
        else
            echo "[squashfs] WARN: no index found (in-shard or sidecar); dataloader will fall back to walk"
        fi
    fi
fi

if [ "${FORBID_RAW_NPYZST:-1}" = "1" ] \
   && [ "${SQUASHFS_MULTI_MODE:-0}" != "1" ] \
   && [ "${SQUASHFS_MODE:-0}" != "1" ]; then
    echo "[squashfs] FATAL: raw lob_preproc_sp500/*.npy.zst training is disabled." >&2
    echo "[squashfs] Set SQUASHFS_MULTI_MODE=1 with an explicit SQUASHFS_MONTHS list." >&2
    exit 1
fi

# Run training
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src:$WORKDIR:${PYTHONPATH:-}"
# The encoder module sits at a different path depending on which tree WORKDIR
# points at: sigma-0's own src/ exposes base_model.encoding.encoding, while
# the openreview-v2 tree (the one legacy_workdir selects) has
# lob.encode.encoding. The guard's purpose is to confirm the
# 26-token encoder is in use, which every layout can answer, so try them in
# order rather than hard-failing on the layout. Job 5856867 died here with
# ModuleNotFoundError: No module named 'lob.encoding'.
python -c "
try:
    from base_model.encoding.encoding import Message_Tokenizer
except ModuleNotFoundError:
    from lob.encode.encoding import Message_Tokenizer
assert Message_Tokenizer.MSG_LEN == 26, Message_Tokenizer.MSG_LEN
print('[guard] 26-token encoder confirmed via', Message_Tokenizer.__module__)
" || {
    echo "FATAL: sigma-0 runtime is not using the canonical 26-token encoder"
    exit 1
}

# Fixed held-out CE evaluation reuses this wrapper so CUDA/JAX, SquashFS, and
# library setup are byte-for-byte identical to training. The validation batch
# sets these explicit paths and the wrapper exits before entering train.py.
if [ "${MODEL_ZOO_CE_EVAL:-0}" = "1" ]; then
    : "${CE_CHECKPOINT_ROOT:?CE_CHECKPOINT_ROOT is required}"
    : "${CE_OUTPUT_JSON:?CE_OUTPUT_JSON is required}"
    exec python -u -B "$WORKDIR/run/base_model/evaluate_model_zoo_ce.py" \
        --checkpoint-root "$CE_CHECKPOINT_ROOT" \
        --expected-architecture "${ARCHITECTURE:?ARCHITECTURE is required}" \
        --expected-seed "${JAX_SEED:?JAX_SEED is required}" \
        --data-root "$DATA_ROOT" \
        --tickers "${CE_TICKERS:-GOOG}" \
        --date-range "${CE_DATE_RANGE:-2025-12-01,2025-12-31}" \
        --batch-size "${CE_BATCH_SIZE:-8}" \
        --max-batches "${CE_MAX_BATCHES:-8}" \
        --targets "${CE_TARGETS:-0.70,0.65,0.60}" \
        --target-tolerance "${CE_TARGET_TOLERANCE:-0.01}" \
        --output "$CE_OUTPUT_JSON"
fi

# ============================================
# IGNORE_TIMES default: False (all tokens in loss)
# ============================================
# IGNORE_TIMES=False means ALL tokens (event + time) enter loss computation.
# IGNORE_TIMES=True skips time tokens in loss (only event tokens contribute).
#
# Changed to False (2026-02-24) based on KTL (Keep-Time-Large) experiment evidence:
#
#   | Config                    | Val Acc  | Test Acc | Val Loss | Job ID  | W&B       |
#   |---------------------------|----------|----------|----------|---------|-----------|
#   | IGNORE_TIMES=False (KTL)  | 80.28%   | 77.82%  | 1.017    | 2458440 | ew3af26l  |
#   | IGNORE_TIMES=True  (G0)   | 76.61%   | 73.45%  | 1.177    | 2458353 | cgdexweb  |
#   | Delta                     | +3.67pp  | +4.37pp | -0.160   |         |           |
#
# Both runs: 75M model, 32N (128 GPU), BSZ=10, lr=1e-3, 40 epochs.
# KTL was still improving when cancelled at E37 — ceiling likely higher.
# Conclusion: time prediction provides causal signal that improves event prediction.
# Override: IGNORE_TIMES=True sbatch ... (to revert to old behavior)
# ============================================

# PROFILER_PREFIX: an optional command placed in front of the interpreter, for
# example `nsys profile -o ... -y 150 -d 40`. Empty by default, so production
# launches are unchanged; word-splitting of the value is intended.
${PROFILER_PREFIX:-} python -u -B "$WORKDIR/run/base_model/runtime/train.py" \
    --USE_WANDB="${USE_WANDB:-True}" \
    --wandb_project="${WANDB_PROJECT:-lobs5-360M-G30}" \
    --wandb_entity="${WANDB_ENTITY:-anonymous}" \
    --C_init=trunc_standard_normal \
    --prenorm=True \
    --batchnorm=False \
    --bidirectional=False \
    --dataset=lobster-prediction \
    --merging=padded \
    --dir_name="$DATA_DIR" \
    ${TEST_DIR:+--test_dir_name="$TEST_DIR"} \
    --clip_eigs=True \
    --activation_fn=half_glu1 \
    --dt_global=False \
    --epochs="${EPOCHS:-1}" \
    --jax_seed=${JAX_SEED:-42} \
    --opt_config="${OPT_CONFIG:-standard}" \
    --p_dropout=0.0 \
    --warmup_end="$WARMUP_END" \
    --weight_decay="${WEIGHT_DECAY:-0.05}" \
    --msg_seq_len="${MSG_SEQ_LEN:-500}" \
    --use_book_data=${USE_BOOK_DATA:-True} \
    --use_simple_book=False \
    --book_transform=True \
    --masking="${MASKING:-none}" \
    ${DFM:+--dfm=$DFM} \
    ${DFM_RESIDUAL:+--dfm_residual=$DFM_RESIDUAL} \
    ${DFM_STAGE:+--dfm_stage=$DFM_STAGE} \
    ${DFM_LR:+--dfm_lr=$DFM_LR} \
    ${DFM_BETA_MAX:+--dfm_beta_max=$DFM_BETA_MAX} \
    ${DFM_SEQ_CHUNK:+--dfm_seq_chunk=$DFM_SEQ_CHUNK} \
    ${DFM_DIST_CKPT:+--dfm_dist_ckpt=$DFM_DIST_CKPT} \
    ${DFM_DIST_STEP:+--dfm_dist_step=$DFM_DIST_STEP} \
    --num_devices="$GPUS_PER_NODE" \
    --n_data_workers="${N_DATA_WORKERS:-12}" \
    --prefetch_factor="${PREFETCH_FACTOR:-2}" \
    --debug_loading=False \
    --enable_profiler=False \
    --random_offsets_train=True \
    --shuffle_train=True \
    --uniform_ticker_batches="${UNIFORM_TICKER_BATCHES:-False}" \
    --max_train_batches="${MAX_TRAIN_BATCHES:-0}" \
    --debug_overfit=False \
    --ignore_times="${IGNORE_TIMES:-False}" \
    --lr_patience=4 \
    --d_model="$D_MODEL" \
    --n_layers="$N_LAYERS" \
    --blocks="$BLOCKS" \
    --ssm_size_base="$SSM_SIZE_BASE" \
    --ssm_lr_base="$SSM_LR_BASE" \
    --lr_factor="$LR_FACTOR" \
    --micro_bsz="$PER_GPU_BSZ" \
    ${CURTAIL_EPOCHS:+--curtail_epochs=$CURTAIL_EPOCHS} \
    --mini_epochs=$MINI_EPOCHS \
    --val_split="$VAL_SPLIT" \
    ${RESTORE_PATH:+--restore=$RESTORE_PATH} \
    ${RESTORE_STEP:+--restore_step=$RESTORE_STEP} \
    ${RESTORE_RESET_SCHEDULE:+--restore_reset_schedule=$RESTORE_RESET_SCHEDULE} \
    ${PARTIAL_RESTORE:+--partial_restore} \
    ${RESUME_FROM_STEP:+--resume_from_step=$RESUME_FROM_STEP} \
    --architecture="${ARCHITECTURE:-mamba3}" \
    --token_mode="${TOKEN_MODE:-26tok}" \
    ${GDN_NUM_HEADS:+--gdn_num_heads=$GDN_NUM_HEADS} \
    ${GDN_HEAD_DIM:+--gdn_head_dim=$GDN_HEAD_DIM} \
    ${GDN_EXPAND_V:+--gdn_expand_v=$GDN_EXPAND_V} \
    ${GDN_USE_CONV:+--gdn_use_conv=$GDN_USE_CONV} \
    ${GDN_CHUNK_SIZE:+--gdn_chunk_size=$GDN_CHUNK_SIZE} \
    ${MAMBA3_D_STATE:+--mamba3_d_state=$MAMBA3_D_STATE} \
    ${MAMBA3_EXPAND:+--mamba3_expand=$MAMBA3_EXPAND} \
    ${MAMBA3_HEADDIM:+--mamba3_headdim=$MAMBA3_HEADDIM} \
    ${MAMBA3_CHUNK_SIZE:+--mamba3_chunk_size=$MAMBA3_CHUNK_SIZE} \
    ${MAMBA3_ROPE_FRACTION:+--mamba3_rope_fraction=$MAMBA3_ROPE_FRACTION} \
    ${MAMBA3_USE_TRITON:+--mamba3_use_triton=$MAMBA3_USE_TRITON} \
    ${MAMBA3_USE_CUDA:+--mamba3_use_cuda=$MAMBA3_USE_CUDA} \
    ${HYBRID_ATTN_LAYERS:+--hybrid_attn_layers=$HYBRID_ATTN_LAYERS} \
    ${HYBRID_ATTN_HEADS:+--hybrid_attn_heads=$HYBRID_ATTN_HEADS} \
    ${HYBRID_ATTN_D_FF:+--hybrid_attn_d_ff=$HYBRID_ATTN_D_FF} \
    ${HYBRID_ATTN_FLASH:+--hybrid_attn_flash=$HYBRID_ATTN_FLASH} \
    ${HYBRID_ATTN_PE:+--hybrid_attn_positional_encoding=$HYBRID_ATTN_PE} \
    ${N_HEADS:+--n_heads=$N_HEADS} \
    ${D_FF:+--d_ff=$D_FF} \
    ${DTYPE:+--dtype=$DTYPE} \
    ${USE_FLASH:+--use_flash=$USE_FLASH} \
    --remat="${REMAT:-0}" \
    ${ATTENTION_MAX_CACHE_LEN:+--attention_max_cache_len=$ATTENTION_MAX_CACHE_LEN} \
    ${BLOCK_SIZE:+--block_size=$BLOCK_SIZE} \
    ${BLOCK_COUNTS:+--block_counts=$BLOCK_COUNTS} \
    ${WINDOW_SIZE:+--window_size=$WINDOW_SIZE} \
    ${PE_TYPE:+--pe_type=$PE_TYPE} \
    ${ROPE_BASE:+--rope_base=$ROPE_BASE} \
    ${HIERARCHICAL:+--hierarchical=$HIERARCHICAL} \
    ${LOCAL_STEPS_K:+--local_steps_k=$LOCAL_STEPS_K} \
    ${DILOCO_OUTER:+--diloco_outer=$DILOCO_OUTER} \
    ${DILOCO_OUTER_LR:+--diloco_outer_lr=$DILOCO_OUTER_LR} \
    ${DILOCO_OUTER_MOMENTUM:+--diloco_outer_momentum=$DILOCO_OUTER_MOMENTUM} \
    ${TP_SIZE:+--tp_size=$TP_SIZE} \
    ${GRAD_ACCUM_STEPS:+--grad_accum_steps=$GRAD_ACCUM_STEPS} \
    ${TICKERS:+--tickers=$TICKERS} \
    ${DATA_ROOT:+--data_root="$DATA_ROOT"} \
    ${TRAIN_DATE_RANGE:+--train_date_range=$TRAIN_DATE_RANGE} \
    ${TEST_DATE_RANGE:+--test_date_range=$TEST_DATE_RANGE} \
    ${MUON_LR:+--muon_lr=$MUON_LR} \
    ${MUON_WD:+--muon_wd=$MUON_WD} \
    ${LOG_CE_TABLES:+--log_ce_tables=$LOG_CE_TABLES} \
    ${HIERARCHICAL_NOBOOK:+--hierarchical_nobook=$HIERARCHICAL_NOBOOK} \
    ${BOOK_ABLATION:+--book_ablation=$BOOK_ABLATION} \
    ${SKIP_TEST_EVAL:+--skip_test_eval} \
    --checkpoint_every_n_steps="$CHECKPOINT_EVERY" \
    --max_job_hours="$MAX_JOB_HOURS"
