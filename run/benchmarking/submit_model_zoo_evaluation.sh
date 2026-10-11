#!/bin/bash
# Submit correlated full-checkpoint smoke -> generation -> CPU WS-21 scoring.
# TASK_IDS is an explicit comma-separated list so unfinished architectures
# cannot accidentally enter the evaluation matrix.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR="${WORKDIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
QUANT_ROOT="${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}"
PYTHON="${PYTHON:-$QUANT_ROOT/miniforge3/bin/python}"
SIGMA0_SHARED_ROOT="${SIGMA0_SHARED_ROOT:-/projects/u6gb/public/sigma-0}"
MODEL_ZOO_ARTIFACT_ROOT="${MODEL_ZOO_ARTIFACT_ROOT:-$SIGMA0_SHARED_ROOT/artifacts/model_zoo_26tok/paper_runs_goog_20260727}"
EVALUATION_OUTPUT_DIRNAME="${EVALUATION_OUTPUT_DIRNAME:-evaluation_30k}"
LOBBENCH_ROOT="${LOBBENCH_ROOT:-$QUANT_ROOT/AlphaTrade/lob_pipeline}"
TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
GENERATION_TIME="${GENERATION_TIME:-04:00:00}"
LOBBENCH_N_SEQUENCES="${LOBBENCH_N_SEQUENCES:-3136}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
N_COND_MSGS="${N_COND_MSGS:-250}"
N_GEN_MSGS="${N_GEN_MSGS:-250}"
SOURCE_SMOKE_JOB_IDS="${SOURCE_SMOKE_JOB_IDS:-}"
CHECKPOINT_STEP=30000
ARCHITECTURES=(mamba3 s5 gdn kda transformer)
TRAIN_SEEDS=(5 42 137)

SMOKE_BATCH="$WORKDIR/run/benchmarking/model_zoo_checkpoint_inference_smoke_array.batch"
GENERATION_BATCH="$WORKDIR/run/benchmarking/model_zoo_generation_array.batch"
SCORE_BATCH="$WORKDIR/run/benchmarking/model_zoo_lobbench_score_array.batch"
EXPECTED_COMMIT=$(git -C "$WORKDIR" rev-parse HEAD)
LOBBENCH_REVISION=$(git -C "$LOBBENCH_ROOT" rev-parse HEAD)

if [ -n "$(git -C "$WORKDIR" status --porcelain --untracked-files=no)" ]; then
    echo "Refusing evaluation submission from a dirty tracked worktree" >&2
    exit 2
fi
if ! git -C "$LOBBENCH_ROOT" diff --quiet HEAD -- \
        lob_bench ':(exclude)lob_bench/hf_data_git' pipeline/_squashfs_helpers.sh; then
    echo "Refusing production evaluation with modified tracked LOBBench code" >&2
    exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENT must be positive, got $MAX_CONCURRENT" >&2
    exit 2
fi
if ! [[ "$EVALUATION_OUTPUT_DIRNAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EVALUATION_OUTPUT_DIRNAME must be one safe path component" >&2
    exit 2
fi
if [ $((LOBBENCH_N_SEQUENCES % (4 * INFERENCE_BATCH_SIZE))) -ne 0 ]; then
    echo "Evaluation pool would require padded duplicate windows" >&2
    exit 2
fi
if [ "$N_COND_MSGS" -ne 250 ] || [ "$N_GEN_MSGS" -ne 250 ]; then
    echo "Paper evaluation requires exactly 250 conditioning + 250 generated messages" >&2
    exit 2
fi
SAMPLE_INDICES_FILE="${SAMPLE_INDICES_FILE:-$WORKDIR/run/benchmarking/model_zoo_GOOG_jan2026_3136_seq500_seed42.txt}"
SAMPLE_INDICES_METADATA="${SAMPLE_INDICES_METADATA:-$SAMPLE_INDICES_FILE.metadata.json}"
if [ ! -f "$SAMPLE_INDICES_FILE" ]; then
    echo "Missing canonical LOBBench sample-index file: $SAMPLE_INDICES_FILE" >&2
    exit 2
fi
if [ ! -f "$SAMPLE_INDICES_METADATA" ]; then
    echo "Missing sample-index provenance: $SAMPLE_INDICES_METADATA" >&2
    exit 2
fi
SAMPLE_INDEX_COUNT=$(awk 'NF {count++} END {print count+0}' "$SAMPLE_INDICES_FILE")
SAMPLE_UNIQUE_COUNT=$(awk 'NF {print $1}' "$SAMPLE_INDICES_FILE" | sort -n -u | wc -l)
if [ "$SAMPLE_INDEX_COUNT" -ne "$LOBBENCH_N_SEQUENCES" ] || \
        [ "$SAMPLE_UNIQUE_COUNT" -ne "$LOBBENCH_N_SEQUENCES" ]; then
    echo "Canonical index file must contain exactly $LOBBENCH_N_SEQUENCES unique indices" >&2
    exit 2
fi
SAMPLE_INDICES_SHA256=$(sha256sum "$SAMPLE_INDICES_FILE" | awk '{print $1}')
read -r EXPECTED_DATASET_LENGTH METADATA_SAMPLE_SHA256 < <(
    "$PYTHON" -c 'import json, sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p["dataset_length"], p["sample_indices_sha256"])' \
        "$SAMPLE_INDICES_METADATA"
)
if [ "$METADATA_SAMPLE_SHA256" != "$SAMPLE_INDICES_SHA256" ]; then
    echo "Sample-index file does not match its provenance metadata" >&2
    exit 2
fi

IFS=',' read -r -a REQUESTED_TASKS <<< "$TASK_IDS"
[ "${#REQUESTED_TASKS[@]}" -gt 0 ] || { echo "TASK_IDS is empty" >&2; exit 2; }
declare -A SEEN=()
for TASK_ID in "${REQUESTED_TASKS[@]}"; do
    if ! [[ "$TASK_ID" =~ ^([0-9]|1[0-4])$ ]]; then
        echo "TASK_IDS must contain only explicit IDs 0..14, got: $TASK_ID" >&2
        exit 2
    fi
    if [ -n "${SEEN[$TASK_ID]:-}" ]; then
        echo "Duplicate evaluation task: $TASK_ID" >&2
        exit 2
    fi
    SEEN[$TASK_ID]=1
    ARCH_INDEX=$((TASK_ID / ${#TRAIN_SEEDS[@]}))
    SEED_INDEX=$((TASK_ID % ${#TRAIN_SEEDS[@]}))
    ARCHITECTURE=${ARCHITECTURES[$ARCH_INDEX]}
    TRAIN_SEED=${TRAIN_SEEDS[$SEED_INDEX]}
    CHECKPOINT_PATH="$MODEL_ZOO_ARTIFACT_ROOT/checkpoints/$ARCHITECTURE/seed_$TRAIN_SEED/run"
    "$PYTHON" "$WORKDIR/run/benchmarking/validate_model_zoo_evaluation.py" checkpoint \
        "$CHECKPOINT_PATH" --architecture "$ARCHITECTURE" --seed "$TRAIN_SEED" \
        --step "$CHECKPOINT_STEP" >/dev/null
    echo "[preflight] task=$TASK_ID architecture=$ARCHITECTURE seed=$TRAIN_SEED checkpoint=$CHECKPOINT_STEP"
done

SMOKE_MODE="fresh-checkpoint-inference"
SKIP_SLURM_SMOKE=0
if [ -n "${SOURCE_SMOKE_JOB_IDS:-}" ]; then
    IFS=':' read -r -a SOURCE_IDS <<< "$SOURCE_SMOKE_JOB_IDS"
    for TASK_ID in "${REQUESTED_TASKS[@]}"; do
        if [ "$TASK_ID" -ge "${#SOURCE_IDS[@]}" ] || ! [[ "${SOURCE_IDS[$TASK_ID]}" =~ ^[0-9]+$ ]]; then
            echo "SOURCE_SMOKE_JOB_IDS has no numeric entry for task $TASK_ID" >&2
            exit 2
        fi
        ARCH_INDEX=$((TASK_ID / ${#TRAIN_SEEDS[@]}))
        SEED_INDEX=$((TASK_ID % ${#TRAIN_SEEDS[@]}))
        ARCHITECTURE=${ARCHITECTURES[$ARCH_INDEX]}
        TRAIN_SEED=${TRAIN_SEEDS[$SEED_INDEX]}
        SOURCE_INFERENCE="$MODEL_ZOO_ARTIFACT_ROOT/inference_smoke_30k/$ARCHITECTURE/seed_$TRAIN_SEED/job_${SOURCE_IDS[$TASK_ID]}/inference"
        "$PYTHON" "$WORKDIR/run/benchmarking/validate_model_zoo_evaluation.py" inference \
            "$SOURCE_INFERENCE" --expected-sequences 8 --rows 32 --world-size 1 >/dev/null
        echo "[preflight] validated existing inference task=$TASK_ID source_job=${SOURCE_IDS[$TASK_ID]}"
    done
    # The source artifacts have just passed the same strict validator locally.
    # Do not submit a redundant CPU-only gate to the GPU-only workq partition:
    # such jobs can remain at priority 1 indefinitely and strand aftercorr.
    SMOKE_MODE="existing-artifact-prevalidated"
    SKIP_SLURM_SMOKE=1
fi

ARRAY_SPEC="$TASK_IDS%$MAX_CONCURRENT"
EXPORTS="ALL,WORKDIR=$WORKDIR,QUANT_ROOT=$QUANT_ROOT,MODEL_ZOO_ARTIFACT_ROOT=$MODEL_ZOO_ARTIFACT_ROOT,EVALUATION_OUTPUT_DIRNAME=$EVALUATION_OUTPUT_DIRNAME,LOBBENCH_ROOT=$LOBBENCH_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,EXPECTED_LOBBENCH_REVISION=$LOBBENCH_REVISION,LOBBENCH_N_SEQUENCES=$LOBBENCH_N_SEQUENCES,INFERENCE_BATCH_SIZE=$INFERENCE_BATCH_SIZE,GENERATION_SEED=${GENERATION_SEED:-2026},N_COND_MSGS=$N_COND_MSGS,N_GEN_MSGS=$N_GEN_MSGS,SAMPLE_INDICES_FILE=$SAMPLE_INDICES_FILE,EXPECTED_SAMPLE_INDICES_SHA256=$SAMPLE_INDICES_SHA256,EXPECTED_DATASET_LENGTH=$EXPECTED_DATASET_LENGTH,SOURCE_SMOKE_JOB_IDS=${SOURCE_SMOKE_JOB_IDS:-}"
SUBMISSION_ROOT="$MODEL_ZOO_ARTIFACT_ROOT/evaluation_submissions"
mkdir -p "$SUBMISSION_ROOT/slurm"

if [ "${DRY_RUN:-0}" = "1" ]; then
    if [ "$SKIP_SLURM_SMOKE" -eq 1 ]; then
        echo "existing inference artifacts validated locally; no redundant Slurm smoke"
        echo "sbatch --array=$ARRAY_SPEC --time=$GENERATION_TIME --export=$EXPORTS $GENERATION_BATCH"
    else
        echo "sbatch --array=$ARRAY_SPEC --export=$EXPORTS $SMOKE_BATCH"
        echo "sbatch --array=$ARRAY_SPEC --dependency=aftercorr:<smoke_job> --time=$GENERATION_TIME --export=$EXPORTS $GENERATION_BATCH"
    fi
    echo "sbatch --array=$ARRAY_SPEC --dependency=aftercorr:<generation_job> --export=$EXPORTS $SCORE_BATCH"
    exit 0
fi

GENERATION_DEPENDENCY=()
if [ "$SKIP_SLURM_SMOKE" -eq 1 ]; then
    SMOKE_JOB="prevalidated"
    DEPENDENCY_CHAIN="prevalidated inference artifacts -> generation --aftercorr--> lobbench"
else
    SMOKE_JOB=$(sbatch --parsable \
        --array="$ARRAY_SPEC" \
        --output="$SUBMISSION_ROOT/slurm/inference_smoke_%A_%a.out" \
        --error="$SUBMISSION_ROOT/slurm/inference_smoke_%A_%a.err" \
        --export="$EXPORTS" \
        "$SMOKE_BATCH")
    SMOKE_JOB=${SMOKE_JOB%%;*}
    GENERATION_DEPENDENCY=(--dependency="aftercorr:$SMOKE_JOB")
    DEPENDENCY_CHAIN="smoke --aftercorr--> generation --aftercorr--> lobbench"
fi

GENERATION_JOB=$(sbatch --parsable \
    --array="$ARRAY_SPEC" \
    "${GENERATION_DEPENDENCY[@]}" \
    --time="$GENERATION_TIME" \
    --output="$SUBMISSION_ROOT/slurm/generation_%A_%a.out" \
    --error="$SUBMISSION_ROOT/slurm/generation_%A_%a.err" \
    --export="$EXPORTS" \
    "$GENERATION_BATCH")
GENERATION_JOB=${GENERATION_JOB%%;*}

SCORE_JOB=$(sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --dependency="aftercorr:$GENERATION_JOB" \
    --output="$SUBMISSION_ROOT/slurm/lobbench_%A_%a.out" \
    --error="$SUBMISSION_ROOT/slurm/lobbench_%A_%a.err" \
    --export="$EXPORTS" \
    "$SCORE_BATCH")
SCORE_JOB=${SCORE_JOB%%;*}

MANIFEST="$SUBMISSION_ROOT/submission_${SMOKE_JOB}_${GENERATION_JOB}_${SCORE_JOB}.json"
"$PYTHON" - "$MANIFEST" <<PY
import json
import os
import sys
from pathlib import Path

payload = {
    "sigma0_commit": "$EXPECTED_COMMIT",
    "lobbench_revision": "$LOBBENCH_REVISION",
    "task_ids": [int(value) for value in "$TASK_IDS".split(",")],
    "architectures": ["mamba3", "s5", "gdn", "kda", "transformer"],
    "training_seeds": [5, 42, 137],
    "checkpoint_step": $CHECKPOINT_STEP,
    "evaluation_output_dirname": "$EVALUATION_OUTPUT_DIRNAME",
    "smoke_job": "$SMOKE_JOB",
    "generation_job": "$GENERATION_JOB",
    "lobbench_job": "$SCORE_JOB",
    "dependency_chain": "$DEPENDENCY_CHAIN",
    "smoke_mode": "$SMOKE_MODE",
    "source_smoke_job_ids": "$SOURCE_SMOKE_JOB_IDS".split(":") if "$SOURCE_SMOKE_JOB_IDS" else [],
    "generation": {
        "stock": "GOOG",
        "period": "2026-01",
        "sequences": $LOBBENCH_N_SEQUENCES,
        "conditioning_messages": $N_COND_MSGS,
        "generated_messages": $N_GEN_MSGS,
        "generation_seed": ${GENERATION_SEED:-2026},
        "world_size": 4,
        "batch_size_per_gpu": $INFERENCE_BATCH_SIZE,
        "padding_policy": "forbidden",
        "sample_indices_file": "$SAMPLE_INDICES_FILE",
        "sample_indices_file_sha256": "$SAMPLE_INDICES_SHA256",
        "sample_indices_metadata": "$SAMPLE_INDICES_METADATA",
        "dataset_length": $EXPECTED_DATASET_LENGTH,
    },
    "scoring": {
        "primary": "LOBBench unconditional WS-21",
        "distances": ["wasserstein", "ks", "l1"],
        "cpu_only": True,
    },
}
destination = Path(sys.argv[1])
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "smoke_job=$SMOKE_JOB"
if [ "$SKIP_SLURM_SMOKE" -eq 1 ]; then
    echo "generation_job=$GENERATION_JOB (released after local artifact validation)"
else
    echo "generation_job=$GENERATION_JOB (aftercorr:$SMOKE_JOB)"
fi
echo "lobbench_job=$SCORE_JOB (aftercorr:$GENERATION_JOB)"
echo "manifest=$MANIFEST"
