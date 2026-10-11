#!/bin/bash
# Submit training and its correlated checkpoint -> inference -> WS-21 array.
# This command is intentionally separate from the smoke jobs: run it only after
# all architecture training and tiny inference/LOBBench gates are green.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR="${WORKDIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
QUANT_ROOT="${QUANT_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant}"
TRAIN_BATCH="$WORKDIR/run/base_model/model_zoo_production_array.batch"
VALIDATION_BATCH="$WORKDIR/run/base_model/model_zoo_validation_array.batch"
EVAL_BATCH="$WORKDIR/run/benchmarking/model_zoo_lobbench_array.batch"
EXPECTED_COMMIT=$(git -C "$WORKDIR" rev-parse HEAD)
SIGMA0_SHARED_ROOT="${SIGMA0_SHARED_ROOT:-/projects/u6gb/public/sigma-0}"
MODEL_ZOO_ARTIFACT_ROOT="${MODEL_ZOO_ARTIFACT_ROOT:-$SIGMA0_SHARED_ROOT/artifacts/model_zoo_26tok}"
LOBBENCH_CE_TARGET="${LOBBENCH_CE_TARGET:-0.65}"
CE_TARGETS="${CE_TARGETS:-0.70,0.65,0.60}"
OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-30000}"
LOBBENCH_N_SEQUENCES="${LOBBENCH_N_SEQUENCES:-3136}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INFERENCE_WORLD_SIZE=4
if [ $((LOBBENCH_N_SEQUENCES % (INFERENCE_WORLD_SIZE * INFERENCE_BATCH_SIZE))) -ne 0 ]; then
    echo "LOBBench pool $LOBBENCH_N_SEQUENCES is not divisible by world_size=$INFERENCE_WORLD_SIZE x batch_size=$INFERENCE_BATCH_SIZE" >&2
    exit 2
fi
WINDOWS_PER_RANK=$((LOBBENCH_N_SEQUENCES / INFERENCE_WORLD_SIZE))
LOBBENCH_ROOT="${LOBBENCH_ROOT:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/lob_pipeline}"
LOBBENCH_REVISION=$(git -C "$LOBBENCH_ROOT" rev-parse HEAD)
export CE_TARGETS OPTIMIZER_STEPS LOBBENCH_CE_TARGET LOBBENCH_ROOT
export LOBBENCH_N_SEQUENCES INFERENCE_BATCH_SIZE
export QUANT_ROOT

# The large hf_data_git checkout is a mutable data cache and is not consumed by
# sigma-0 (inference files are staged explicitly). The scorer and mount helper,
# however, must exactly match LOBBENCH_REVISION.
if ! git -C "$LOBBENCH_ROOT" diff --quiet HEAD -- \
    lob_bench ':(exclude)lob_bench/hf_data_git' pipeline/_squashfs_helpers.sh; then
    echo "Refusing production submission with modified LOBBench scorer code" >&2
    exit 2
fi

case ",$CE_TARGETS," in
    *",$LOBBENCH_CE_TARGET,"*) ;;
    *)
        echo "LOBBENCH_CE_TARGET=$LOBBENCH_CE_TARGET is absent from CE_TARGETS=$CE_TARGETS" >&2
        exit 2
        ;;
esac

if [ -n "$(git -C "$WORKDIR" status --porcelain --untracked-files=no)" ]; then
    echo "Refusing production submission from a dirty tracked worktree" >&2
    exit 2
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "sbatch --parsable --export=ALL,WORKDIR=$WORKDIR,QUANT_ROOT=$QUANT_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,MODEL_ZOO_ARTIFACT_ROOT=$MODEL_ZOO_ARTIFACT_ROOT $TRAIN_BATCH"
    echo "sbatch --parsable --dependency=aftercorr:<train_job_id> --export=ALL,WORKDIR=$WORKDIR,QUANT_ROOT=$QUANT_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,MODEL_ZOO_ARTIFACT_ROOT=$MODEL_ZOO_ARTIFACT_ROOT $VALIDATION_BATCH"
    echo "sbatch --parsable --dependency=aftercorr:<validation_job_id> --export=ALL,WORKDIR=$WORKDIR,EXPECTED_COMMIT=$EXPECTED_COMMIT,MODEL_ZOO_ARTIFACT_ROOT=$MODEL_ZOO_ARTIFACT_ROOT,LOBBENCH_CE_TARGET=$LOBBENCH_CE_TARGET $EVAL_BATCH"
    exit 0
fi

mkdir -p "$MODEL_ZOO_ARTIFACT_ROOT"
if [ -e "$MODEL_ZOO_ARTIFACT_ROOT/submission.json" ]; then
    echo "Refusing to replace existing production manifest: $MODEL_ZOO_ARTIFACT_ROOT/submission.json" >&2
    exit 2
fi

TRAIN_JOB=$(sbatch --parsable \
    --export="ALL,WORKDIR=$WORKDIR,QUANT_ROOT=$QUANT_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,MODEL_ZOO_ARTIFACT_ROOT=$MODEL_ZOO_ARTIFACT_ROOT" \
    "$TRAIN_BATCH")
TRAIN_JOB=${TRAIN_JOB%%;*}
VALIDATION_JOB=$(sbatch --parsable \
    --dependency="aftercorr:$TRAIN_JOB" \
    --export="ALL,WORKDIR=$WORKDIR,QUANT_ROOT=$QUANT_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,MODEL_ZOO_ARTIFACT_ROOT=$MODEL_ZOO_ARTIFACT_ROOT" \
    "$VALIDATION_BATCH")
VALIDATION_JOB=${VALIDATION_JOB%%;*}
EVAL_JOB=$(sbatch --parsable \
    --dependency="aftercorr:$VALIDATION_JOB" \
    --export="ALL,WORKDIR=$WORKDIR,EXPECTED_COMMIT=$EXPECTED_COMMIT,MODEL_ZOO_ARTIFACT_ROOT=$MODEL_ZOO_ARTIFACT_ROOT,LOBBENCH_CE_TARGET=$LOBBENCH_CE_TARGET" \
    "$EVAL_BATCH")
EVAL_JOB=${EVAL_JOB%%;*}

cat > "$MODEL_ZOO_ARTIFACT_ROOT/submission.json" <<EOF
{
  "commit": "$EXPECTED_COMMIT",
  "quant_root": "$QUANT_ROOT",
  "lobbench_revision": "$LOBBENCH_REVISION",
  "lobbench_scorer_tracked_code_clean": true,
  "training_job": "$TRAIN_JOB",
  "validation_job": "$VALIDATION_JOB",
  "lobbench_job": "$EVAL_JOB",
  "architectures": ["mamba3", "s5", "gdn", "kda", "transformer", "nsa"],
  "training_seeds": [5, 42, 137],
  "generation_seed": ${GENERATION_SEED:-2026},
  "token_mode": "26tok",
  "wandb": {
    "mode": "online",
    "entity": "oxford-lob",
    "project": "icaif_model_paper",
    "cloud_runs": "one per architecture/seed from global rank 0"
  },
  "parameter_matching": {
    "reference": "mamba3_d256_l6",
    "trainable_parameters": {
      "mamba3": 8099567,
      "s5": 8201069,
      "gdn": 8021621,
      "kda": 7916209,
      "transformer": 8041203,
      "nsa": 8056828
    }
  },
  "training_protocol": {
    "nodes": 4,
    "gpus_per_node": 4,
    "micro_batch_per_gpu": 8,
    "gradient_accumulation": 1,
    "global_batch": 128,
    "distributed_optimizer": "K=10 local Adam/AdamW steps; pmean over 4 GPUs/node per step; plain parameter pmean over 4 nodes per group",
    "optimizer": "AdamW/Adam parameter groups",
    "peak_learning_rate": 0.0005,
    "warmup_fraction": 0.01,
    "weight_decay": 0.05,
    "maximum_gradient_norm": 1.0,
    "messages_per_sequence": 500,
    "tokens_per_message": 26,
    "token_values_per_update": 1664000,
    "maximum_optimizer_updates": $OPTIMIZER_STEPS,
    "ticker_sampling": "GOOG-only global batches",
    "training_period": ["2022-01-01", "2025-11-30"],
    "tickers": ["GOOG"]
  },
  "checkpoint_selection": {
    "split": "2025-12",
    "validation_ce_target": $LOBBENCH_CE_TARGET,
    "available_targets": "$CE_TARGETS"
  },
  "test_protocol": {
    "period": "2026-01",
    "metric_family": "LOBBench WS-21",
    "sequences": $LOBBENCH_N_SEQUENCES,
    "inference_world_size": $INFERENCE_WORLD_SIZE,
    "inference_batch_size": $INFERENCE_BATCH_SIZE,
    "windows_per_rank": $WINDOWS_PER_RANK,
    "padding_policy": "forbidden",
    "conditioning_messages": ${N_COND_MSGS:-500},
    "generated_messages": ${N_GEN_MSGS:-500}
  }
}
EOF

echo "training_job=$TRAIN_JOB"
echo "validation_job=$VALIDATION_JOB (aftercorr:$TRAIN_JOB)"
echo "lobbench_job=$EVAL_JOB (aftercorr:$VALIDATION_JOB, validation_ce_target=$LOBBENCH_CE_TARGET)"
echo "manifest=$MODEL_ZOO_ARTIFACT_ROOT/submission.json"
