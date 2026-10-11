#!/bin/bash
# Run an unmodified generation launcher against node-local storage, then copy
# back one SquashFS image instead of 25,092 files.
#
# Why a wrapper rather than an edit. The evaluations in flight use a frozen
# runtime worktree pinned to an exact commit, and the whole point of that pin is
# that nobody edits it. But the launcher derives its output paths from one
# argument:
#
#     TASK_ROOT  ->  OUTPUT_ROOT="$TASK_ROOT/evaluation"
#                ->  INFERENCE_DIR="$OUTPUT_ROOT/inference"
#
# so pointing TASK_ROOT at node-local /tmp moves every one of those files off
# Lustre without changing a byte of the frozen code. Afterwards this script
# packs, verifies, and copies back only the image, the sidecar, and the small
# files that cannot be recovered from it.
#
# Usage:
#   run/benchmarking/pack_gen_wrapper.sh <real-task-root> <launcher> [launcher args...]
#
# Everything the launcher needs in the environment is passed through unchanged
# except TASK_ROOT, which this script overrides.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <real-task-root> <launcher> [args...]" >&2
    exit 64
fi

REAL_TASK_ROOT="$1"; shift
LAUNCHER="$1"; shift

: "${LAUNCHER_WORKDIR:?LAUNCHER_WORKDIR is required (it holds pack_inference.py)}"
PYTHON="${PYTHON:-python3}"

# Unique per invocation, not per allocation. Under `srun --overlap` several
# steps share one SLURM_JOB_ID, so a job-derived name lets two concurrent
# wrappers write into each other's tree.
mkdir -p "/tmp/${USER:-unknown}"
LOCAL_ROOT="$(mktemp -d "/tmp/${USER:-unknown}/sigma0_gen_XXXXXXXX")"
LOCAL_EVAL="$LOCAL_ROOT/evaluation"
REAL_EVAL="$REAL_TASK_ROOT/evaluation"
mkdir -p "$LOCAL_EVAL" "$REAL_EVAL"

echo "[wrapper] real task root : $REAL_TASK_ROOT"
echo "[wrapper] local task root: $LOCAL_ROOT"

# 3,136 sequences produced 699 MB on the measured trees, and /tmp is tmpfs, so
# it comes out of node memory. Check before generating, not after.
TMP_FREE_MB="$(df -Pm /tmp | awk 'NR==2 {print $4}')"
if [ "${TMP_FREE_MB:-0}" -lt 4096 ]; then
    echo "[wrapper] only ${TMP_FREE_MB} MB free on /tmp; refusing to start" >&2
    exit 2
fi
echo "[wrapper] /tmp free=${TMP_FREE_MB} MB"

# Resolve mksquashfs now rather than after eleven hours of generation.
if [ -n "${SIGMA0_SQUASHFS_BIN:-}" ] && [ -x "$SIGMA0_SQUASHFS_BIN/mksquashfs" ]; then
    :
elif command -v mksquashfs >/dev/null 2>&1; then
    SIGMA0_SQUASHFS_BIN="$(dirname "$(command -v mksquashfs)")"
else
    echo "[wrapper] mksquashfs not found; set SIGMA0_SQUASHFS_BIN" >&2
    exit 2
fi
export SIGMA0_SQUASHFS_BIN
echo "[wrapper] mksquashfs=$SIGMA0_SQUASHFS_BIN/mksquashfs"

TASK_ROOT="$LOCAL_ROOT" "$LAUNCHER" "$@"

LOCAL_INFERENCE="$LOCAL_EVAL/inference"
if [ ! -d "$LOCAL_INFERENCE" ]; then
    echo "[wrapper] launcher produced no $LOCAL_INFERENCE; nothing to pack" >&2
    exit 3
fi

# Pack and verify every file by sha256 against the tree that produced it. The
# verify is not optional: mksquashfs has silently truncated files in this
# project before, and LOB-Bench reads with on_bad_lines="skip", so a truncated
# tail would become a shorter DataFrame and move the score with nothing failing.
PYTHONPATH="$LAUNCHER_WORKDIR/src:${PYTHONPATH:-}" \
    "$PYTHON" -u "$LAUNCHER_WORKDIR/run/benchmarking/pack_inference.py" \
    --keep-source \
    --report "$LOCAL_EVAL/pack_report.json" \
    "$LOCAL_INFERENCE"

# Copy back the image, its sidecar, and the small files that the image does not
# make redundant: the scores, the manifests, the rank logs, and the sample
# indices that say which contexts were drawn.
mkdir -p "$REAL_EVAL/inference_index"
rsync -a "$LOCAL_EVAL/inference.sqfs" "$REAL_EVAL/inference.sqfs"
rsync -a "$LOCAL_EVAL/index_inference.json" "$REAL_EVAL/index_inference.json"
cp -p "$LOCAL_INFERENCE"/sample_indices_rank*.json "$REAL_EVAL/inference_index/" 2>/dev/null || true
# An `if` rather than `[ -f x ] && rsync`: under `set -e` the && form makes the
# loop exit non-zero when the last file happens to be absent, which would abort
# the script after the image had already been copied back.
for name in generation_complete.json inference_inventory.json pack_report.json \
            inference_rank0.log inference_rank1.log inference_rank2.log inference_rank3.log; do
    if [ -f "$LOCAL_EVAL/$name" ]; then
        rsync -a "$LOCAL_EVAL/$name" "$REAL_EVAL/$name"
    fi
done

IMAGE_BYTES="$(stat -c%s "$REAL_EVAL/inference.sqfs")"
echo "[wrapper] copied back inference.sqfs ($IMAGE_BYTES bytes) to $REAL_EVAL"
echo "[wrapper] node-local tree left at $LOCAL_INFERENCE; it goes with the node"
