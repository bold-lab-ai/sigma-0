#!/bin/bash
# Stage A: generate the frozen market that every candidate market-making policy
# will be scored against.
#
# Two roles, two checkpoints, one shared set of windows:
#   background order flow  <- bg checkpoint,     one seed
#   price-curve signal     <- signal checkpoint, J seeds on the SAME windows
#
# The shared windows are not something this script has to arrange. Index
# selection inside runtime/inference.py is pinned to seed 42 and is independent
# of --seed (see its "Index selection stays fixed at 42" note), so varying
# --seed re-rolls the generation randomness while leaving the conditioning
# windows identical. That is exactly the J-rollouts-from-one-history sampler the
# signal needs, with no new generation code.
#
# Ticker choice is load-bearing. A one-minute window costs one minute of message
# flow, and that is a per-ticker constant: 32,436 messages on GOOGL, 14,488 on
# GOOG, but 463 at the SP500 median (measured from index_2026-01.json over 487
# tickers). GS sits at 787/minute, which is affordable, and its high share price
# means the spread is usually more than one tick, so quoting at the touch has
# something to capture. Both checkpoints were trained on the full SP500
# universe (their metadata lists ~490 tickers), so GS is in distribution, and
# 2026-01 is after the 2022-2025 training range, so the window is held out.

#SBATCH --job-name=sigma0-agentic-mm-episodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --time=03:00:00
# Logs go to a neutral job-id-keyed location, NOT into a worktree. A #SBATCH
# directive cannot reference $WORKDIR -- Slurm reads these lines before any
# shell expansion happens -- so a worktree-relative output path can only ever
# be a hardcoded one, and it then points at the wrong tree the moment WORKDIR
# is overridden (D-X3). Keying everything on the job id instead means the log,
# the per-rank logs and the manifest all join on one value, and the manifest
# records which worktree and which commit produced the data.
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/slurm_logs/%j.out

set -euo pipefail

# No default, on purpose (D-X2). A stale default silently produces new data
# from old code, which is exactly what happened on 2026-07-31: episodes
# generated at 16:48 from a worktree cut before the 09:39-11:00 fixes, with
# nothing in the pipeline able to say so.
WORKDIR="${WORKDIR:?WORKDIR must be set explicitly, e.g. WORKDIR=/lus/.../sigma-0-worktrees/<name> sbatch ...}"
QUANT_ROOT="${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/lus/lfs1aip2/projects/public/u6gb/s5e_alphatrade/lob_pipeline}"
PYTHON="${PYTHON:-$QUANT_ROOT/miniforge3/bin/python}"

STOCK="${STOCK:-GS}"
MONTHS="${MONTHS:-2026-01}"
N_SEQUENCES="${N_SEQUENCES:-8}"        # windows; must divide by world_size * batch_size
BATCH_SIZE="${BATCH_SIZE:-2}"
N_COND_MSGS="${N_COND_MSGS:-500}"      # the model's training context length
N_GEN_MSGS="${N_GEN_MSGS:-900}"        # ~68s at GS's 787 msg/min, i.e. one minute plus margin
BG_SEED="${BG_SEED:-2026}"
# The L2 width the simulator initialises from. Load-bearing rather than
# cosmetic: it sets book_depth, nOrders and the range of synthetic init ids,
# and a replay that does not match it reconstructs a different book (D-I3).
WIDE_LEVELS="${WIDE_LEVELS:-500}"
SIGNAL_SEEDS="${SIGNAL_SEEDS:-3001 3002 3003 3004 3005 3006 3007 3008}"
WORLD_SIZE=4

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
TASK_ROOT="${TASK_ROOT:-/lus/lfs1aip2/projects/public/u6gb/tasks/agentic_mm_${STAMP}}"
LOG_DIR="${SIGMA_LOG_ROOT:-/lus/lfs1aip2/projects/public/u6gb/slurm_logs}"
mkdir -p "$LOG_DIR"

# --- gate 0: cheap checks, before the allocation is spent -------------------
# Two seconds here would have saved the run that died two seconds in.
source "$WORKDIR/run/environment/preflight.sh"
preflight_or_die "$WORKDIR" python3 rsync

# SHARD is this invocation's identity when several nodes split the work. The
# stages are independent -- one background flow plus J signal seeds, all over
# the same conditioning windows -- so they parallelise across nodes with no
# coordination beyond not writing to each other's directories.
#
# RUN_BACKGROUND / SIGNAL_SEEDS select which stages *this* invocation runs.
# A single-node run leaves both at their defaults and does everything.
SHARD="${SHARD:-solo}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
WRITE_MANIFEST="${WRITE_MANIFEST:-1}"

# Everything heavy stays on node-local scratch; only the finished artifacts are
# copied back at the end.
JOB_TMP="${TMPDIR:-/tmp}/agentic_mm_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
mkdir -p "$MPLCONFIGDIR"
GEN_ROOT="$JOB_TMP/generated"
mkdir -p "$GEN_ROOT"

echo "[stageA] node=$(hostname) job=${SLURM_JOB_ID:-manual} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[stageA] stock=$STOCK months=$MONTHS windows=$N_SEQUENCES cond=$N_COND_MSGS gen=$N_GEN_MSGS"
echo "[stageA] task_root=$TASK_ROOT"

# --- gate 1: the pinned checkpoints are present and unchanged ---------------
# Content hash, not recency. Both pins are deliberately NOT the newest step in
# their directory, so verify() is called with require_latest left off.
export PYTHONPATH="$WORKDIR/src:$WORKDIR:${PYTHONPATH:-}"
"$PYTHON" - <<'PY'
from post_training.heuristic_learning.checkpoints import PINS, verify
for name, pin in sorted(PINS.items()):
    verify(pin)
    print(f"[stageA] checkpoint ok: {name} step={pin.step} role={pin.role} "
          f"(latest in dir is {pin.latest_step_in_dir}; pinned on purpose)")
PY

BG_CKPT="/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/checkpoints/j4559297_u52a0g05_4559297"
BG_STEP=150360
SIG_CKPT="/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3/checkpoints/j4553948_tz2tmn5z_4553948"
SIG_STEP=120000

# --- environment ------------------------------------------------------------
module load cuda/12.6
CONDA_PREFIX="$QUANT_ROOT/miniforge3"
NVIDIA_SITE="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$TORCH_LIB:$NVIDIA_SITE/cuda_nvrtc/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusparse/lib:$NVIDIA_SITE/cuda_cupti/lib:$NVIDIA_SITE/cufft/lib:$NVIDIA_SITE/nvjitlink/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/nccl/lib:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cudnn/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
# These checkpoints were trained with the current norm implementation.
unset MAMBA3_LEGACY_NORM || true
"$PYTHON" -c 'import jax; b=jax.default_backend(); assert b=="gpu", b; print(f"[stageA] jax backend={b} devices={jax.device_count()}")'

# --- data: mount the L10 stream and the L500 wide book ----------------------
source "$WORKDIR/run/benchmarking/squashfs_helpers_no_delete.sh"
infer_squashfs_setup "$STOCK" "$MONTHS"
infer_squashfs_setup_wide_book "$STOCK" "$MONTHS" \
    "/lus/lfs1aip2/projects/public/s5e/quant_team/recon_2026-05/output/squashfs" "$WIDE_LEVELS"
trap infer_squashfs_cleanup EXIT
echo "[stageA] data_dir=$INFER_DATA_DIR_NODE"
echo "[stageA] wide_book_dir=$INFER_WIDE_BOOK_DIR_NODE"

# --- generation -------------------------------------------------------------
run_generation() {
    local tag="$1" ckpt="$2" step="$3" seed="$4"
    local out="$GEN_ROOT/$tag"
    if [ -f "$out/.complete" ]; then
        echo "[stageA] $tag already complete, skipping"   # resume: never redo finished work
        return 0
    fi
    mkdir -p "$out"
    echo "[stageA] --- $tag  ckpt=$(basename "$ckpt")@$step seed=$seed"
    local pids=() rc=0
    for r in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$r TMPDIR="$JOB_TMP/rank_$r" \
        "$PYTHON" -u -B "$WORKDIR/run/base_model/runtime/inference.py" \
            --architecture=mamba3 --token_mode=26tok \
            --ckpt_path="$ckpt" --checkpoint_step="$step" \
            --data_dir="$INFER_DATA_DIR_NODE" \
            --wide_book_dir="$INFER_WIDE_BOOK_DIR_NODE" --wide_levels="$WIDE_LEVELS" \
            --save_dir="$out" --stock="$STOCK" \
            --n_sequences="$N_SEQUENCES" --batch_size="$BATCH_SIZE" \
            --n_cond_msgs="$N_COND_MSGS" --n_gen_msgs="$N_GEN_MSGS" \
            --test_split=1.0 --seed="$seed" \
            --rank="$r" --world_size="$WORLD_SIZE" \
            > "$LOG_DIR/${SLURM_JOB_ID:-manual}_${tag}_rank${r}.log" 2>&1 &
        pids+=("$!")
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "[stageA] $tag rank $i FAILED (see ${SLURM_JOB_ID:-manual}_${tag}_rank${i}.log)" >&2; rc=1; }
    done
    [ "$rc" -eq 0 ] || return 1
    touch "$out/.complete"
}

if [ "$RUN_BACKGROUND" = "1" ]; then
    run_generation "background_seed${BG_SEED}" "$BG_CKPT" "$BG_STEP" "$BG_SEED"
fi
for s in $SIGNAL_SEEDS; do
    run_generation "signal_seed${s}" "$SIG_CKPT" "$SIG_STEP" "$s"
done

# --- provenance + copy back -------------------------------------------------
mkdir -p "$TASK_ROOT"
if [ "$WRITE_MANIFEST" = "1" ]; then
"$PYTHON" - "$TASK_ROOT/manifest.json" <<PY
import json, sys
from post_training.heuristic_learning.checkpoints import PINS, metadata_sha256
manifest = {
    "experiment": "agentic_mm_stage_a_episodes",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "stamp": "${STAMP}",
    # Code provenance (D-X2). Without these the only thing an artifact says
    # about its own code is its mtime, and mtime is not freshness: the
    # 2026-07-31 episodes are newer than the fixes they do not contain.
    "workdir": "${WORKDIR}",
    # The node is part of the configuration, not incidental metadata (D-X7).
    # Two runs with identical checkpoint, seed, window and code produce
    # byte-identical output on the same node and completely different output on
    # different nodes: measured 8/8 identical on nid010148, 8/8 different
    # between nid010147 and nid010148. Cross-device floating point flips a
    # sampling decision and autoregression amplifies it from there.
    #
    # This does not break the frozen-market design, since a market is frozen by
    # being written to disk and every policy replays that same record. It does
    # mean a record that does not name its node cannot be regenerated, so the
    # record names it.
    "node": "$(hostname)",
    "git_sha": "${PREFLIGHT_GIT_SHA}",
    "git_dirty_src_run": ${PREFLIGHT_GIT_DIRTY},
    "stock": "${STOCK}", "months": "${MONTHS}",
    "n_windows": ${N_SEQUENCES}, "n_cond_msgs": ${N_COND_MSGS}, "n_gen_msgs": ${N_GEN_MSGS},
    "background_seed": ${BG_SEED},
    "signal_seeds": [int(s) for s in "${SIGNAL_SEEDS}".split()],
    "index_selection_seed": 42,
    # The engine configuration this run actually used (D-T3). Without it a
    # replay must guess, and JAXLOB_Configuration's defaults are not what the
    # generator runs: book_depth 10 against 500 here. Guessing wrong produces a
    # different book from the same messages, which is class T in one line.
    #
    # book_depth and nOrders are derived from wide_levels exactly as
    # sample_new derives them, so the record cannot disagree with itself.
    "engine": {
        "wide_levels": ${WIDE_LEVELS},
        "book_depth": ${WIDE_LEVELS},
        "nOrders": max(100, ${WIDE_LEVELS} * 2 + 50),
        "tick_size": 100,
        "init_id": -2,
    },
    "token_mode": "26tok",
    "checkpoints": {
        name: {"run_dir": p.run_dir, "step": p.step, "role": p.role,
               "metadata_sha256_expected": p.metadata_sha256,
               "metadata_sha256_actual": metadata_sha256(p),
               "is_latest_in_dir": p.is_latest, "rationale": p.rationale}
        for name, p in sorted(PINS.items())
    },
}
json.dump(manifest, open(sys.argv[1], "w"), indent=2)
print("[stageA] wrote", sys.argv[1])
PY
fi

# Each shard rsyncs only the tags it produced. Node-local scratch during the
# run, one copy back at the end, and distinct destination subtrees so parallel
# shards never write the same file.
echo "[stageA] shard=$SHARD copying generated data to $TASK_ROOT/generated"
rsync -a "$GEN_ROOT/" "$TASK_ROOT/generated/"
echo "[stageA] shard=$SHARD DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
