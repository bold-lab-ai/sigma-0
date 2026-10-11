from pathlib import Path

import pytest

from base_model.inference import build_inference_plan
from base_model.config import read_key_value_config
from base_model.training import build_train_plan


def test_training_architecture_override_is_canonical(tmp_path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        "\n".join(
            [
                "name: archived-smoke",
                "legacy_batch: run/base_model/train_full_autoreg.batch",
                "legacy_workdir: .",
                "env_SSM_TYPE: gdn",
                "env_TOKEN_MODE: 24tok",
            ]
        ),
        encoding="utf-8",
    )
    plan = build_train_plan(config, architecture="transformer")
    assert plan.environment["ARCHITECTURE"] == "transformer"
    assert plan.environment["TOKEN_MODE"] == "26tok"
    assert "SSM_TYPE" not in plan.environment


def test_config_include_keeps_architecture_override(tmp_path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text("architecture: mamba3\nenv_D_MODEL: 64\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(
        "include: parent.yaml\narchitecture: nsa\nenv_BLOCK_SIZE: 26\n",
        encoding="utf-8",
    )
    values = read_key_value_config(child)
    assert values["architecture"] == "nsa"
    assert values["env_D_MODEL"] == "64"
    assert values["env_BLOCK_SIZE"] == "26"


def test_inference_command_carries_architecture_and_26tok(tmp_path) -> None:
    config = tmp_path / "inference.yaml"
    config.write_text(
        "\n".join(
            [
                "name: nsa-smoke",
                "stock: GOOG",
                "architecture: nsa",
                "checkpoint_path: /checkpoints/nsa",
                "data_dir: /data/goog",
                "output_root: /outputs/nsa",
                "n_sequences: 8",
                "n_cond_msgs: 16",
                "n_gen_msgs: 4",
                "seed: 137",
            ]
        ),
        encoding="utf-8",
    )
    plan = build_inference_plan(config)
    command = plan.command(python="python3")
    assert "--architecture=nsa" in command
    assert "--token_mode=26tok" in command
    assert "--n_cond_msgs=16" in command
    assert "--n_gen_msgs=4" in command


@pytest.mark.parametrize(
    "architecture",
    ("mamba3", "s5", "gdn", "kda", "transformer", "nsa"),
)
def test_model_zoo_smoke_configs_are_selectable(architecture: str) -> None:
    plan = build_train_plan(
        f"configs/train/model_zoo_smoke_{architecture}.yaml"
    )
    assert plan.environment["ARCHITECTURE"] == architecture
    assert plan.environment["TOKEN_MODE"] == "26tok"
    assert plan.extra_args == ("--gres=gpu:1", "--ntasks-per-node=1")


def test_gpu_smoke_array_covers_registry_order() -> None:
    script = Path("run/base_model/model_zoo_smoke_array.batch").read_text(
        encoding="utf-8"
    )
    assert "ARCHITECTURES=(mamba3 s5 gdn kda transformer nsa)" in script
    assert "#SBATCH --array=0-5" in script
    assert "export TOKEN_MODE=26tok" in script
    assert "export GPUS_PER_NODE=1" in script


def test_attention_inference_expands_short_smoke_cache() -> None:
    script = Path("run/base_model/runtime/inference.py").read_text(encoding="utf-8")
    assert "n_messages_conditional + n_gen_msgs" in script
    assert "args.attention_max_cache_len = required_cache_len" in script


def test_production_batch_has_one_global_batch_invariant() -> None:
    script = Path("run/base_model/model_zoo_production_array.batch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --nodes=4" in script
    assert "export GPUS_PER_NODE=4" in script
    assert "export PER_GPU_BSZ=8" in script
    assert "export TARGET_GLOBAL_BSZ=128" in script
    assert "ACTUAL_GLOBAL_BSZ=$((PER_GPU_BSZ * GPUS_PER_NODE * SLURM_NNODES * GRAD_ACCUM_STEPS))" in script
    assert "export SSM_LR_BASE=0.0005" in script
    assert "export TICKERS=GOOG" in script
    assert "export UNIFORM_TICKER_BATCHES=True" in script
    assert "export MAX_TRAIN_BATCHES=$((OPTIMIZER_STEPS * GRAD_ACCUM_STEPS))" in script

    trainer = Path("src/base_model/training/train.py").read_text(encoding="utf-8")
    assert 'os.environ.get("EXPECTED_PARAMS")' in trainer
    assert "parameter-count mismatch" in trainer
    assert "[Protocol] Parameter-count match" in trainer
    assert 'os.environ.get("WANDB_RUN_ID")' in trainer
    assert "resume=wandb_resume" in trainer

    data_smoke = Path("run/base_model/check_model_zoo_data.py").read_text(
        encoding="utf-8"
    )
    assert "uniform_ticker_batches=True" in data_smoke
    assert "process_count=4" in data_smoke
    assert "sampler.global_bsz == 128" in data_smoke

    data_batch = Path("run/base_model/model_zoo_data_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    assert "SIGMA0_JOB_TMPDIR=" in data_batch
    assert 'export TMPDIR="$SIGMA0_JOB_TMPDIR"' in data_batch


def test_kda_resume_is_exact_and_preserves_tracking_identity() -> None:
    production = Path("run/base_model/model_zoo_production_array.batch").read_text(
        encoding="utf-8"
    )
    resume = Path("run/base_model/model_zoo_kda_resume_array.batch").read_text(
        encoding="utf-8"
    )
    assert 'RESUME_STEP="${MODEL_ZOO_RESUME_STEP:-}"' in production
    assert 'RESUME_FROM_STEP="$RESUME_STEP"' in production
    assert "Refusing non-latest resume" in production
    assert "RESTORE_RESET_SCHEDULE=False" in production
    assert "WANDB_RUN_ID is required for a production continuation" in production
    assert 'MODEL_ZOO_CONFIG_ONLY:-0' in production
    assert "#SBATCH --array=9-11%3" in resume
    assert "MODEL_ZOO_RESUME_STEP=28000" in resume
    assert "OPTIMIZER_STEPS=30000" in resume
    assert "WANDB_RUN_ID=e2fdixnm" in resume
    assert "WANDB_RUN_ID=kahh38wg" in resume
    assert "WANDB_RUN_ID=w8wsnxa9" in resume


def test_validation_reuses_fixed_sample_and_checks_checkpoint_identity() -> None:
    evaluator = Path("run/base_model/evaluate_model_zoo_ce.py").read_text(
        encoding="utf-8"
    )
    wrapper = Path("run/base_model/node_wrapper.sh").read_text(encoding="utf-8")
    assert "validation_batches = list(" in evaluator
    assert "uniform_ticker_batches=True" in evaluator
    assert "max_train_batches=args.max_batches" in evaluator
    assert '"sampling": "uniform_ticker_batches_without_replacement"' in evaluator
    assert "default=_json_scalar_default" in evaluator
    assert "--expected-architecture" in evaluator
    assert "--expected-architecture" in wrapper
    assert "--expected-seed" in wrapper
    assert 'parser.add_argument("--tickers", default="GOOG")' in evaluator
    assert '--tickers "${CE_TICKERS:-GOOG}"' in wrapper


def test_lobbench_uses_exact_final_checkpoints_and_cpu_scoring() -> None:
    smoke = Path(
        "run/benchmarking/model_zoo_checkpoint_inference_smoke_array.batch"
    ).read_text(encoding="utf-8")
    generation = Path("run/benchmarking/model_zoo_generation_array.batch").read_text(
        encoding="utf-8"
    )
    scoring = Path(
        "run/benchmarking/model_zoo_lobbench_score_array.batch"
    ).read_text(encoding="utf-8")
    submitter = Path(
        "run/benchmarking/submit_model_zoo_evaluation.sh"
    ).read_text(encoding="utf-8")

    for script in (smoke, generation, scoring):
        assert "ARCHITECTURES=(mamba3 s5 gdn kda transformer)" in script
        assert "nsa" not in script.lower()
    assert "EVALUATION_CHECKPOINT_STEP:-30000" in smoke
    assert "EVALUATION_CHECKPOINT_STEP:-30000" in generation
    assert "validate_model_zoo_evaluation.py\" checkpoint" in smoke
    assert "validate_model_zoo_evaluation.py\" checkpoint" in generation
    assert "LOBBENCH_CE_TARGET" not in generation
    assert "VALIDATION_JSON" not in generation
    assert 'LOBBENCH_N_SEQUENCES:-3136' in generation
    assert 'INFERENCE_BATCH_SIZE:-8' in generation
    assert "N_SEQUENCES % (WORLD_SIZE * BATCH_SIZE)" in generation
    assert "Refusing padded evaluation pool" in generation
    assert '--n_sequences="$N_SEQUENCES"' in generation
    assert 'N_COND_MSGS="${N_COND_MSGS:-250}"' in generation
    assert 'N_GEN_MSGS="${N_GEN_MSGS:-250}"' in generation
    assert '--test_split=1.0' in generation
    assert '--sample_indices_file="$SAMPLE_INDICES_FILE"' in generation
    assert '--expected-indices-file "$SAMPLE_INDICES_FILE"' in generation
    assert '--expected-dataset-length "${EXPECTED_DATASET_LENGTH:' in generation
    assert "sample_indices_rank" in Path(
        "run/base_model/runtime/inference.py"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:4" in generation
    assert "#SBATCH --gres" not in scoring
    assert "feature_count\") != 21" in scoring
    assert "--run-id=production30k" in scoring
    assert 'TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8}"' in submitter
    assert 'N_COND_MSGS="${N_COND_MSGS:-250}"' in submitter
    assert 'N_GEN_MSGS="${N_GEN_MSGS:-250}"' in submitter
    assert "Paper evaluation requires exactly 250 conditioning + 250 generated messages" in submitter
    assert "sample_indices_file_sha256" in submitter
    assert "model_zoo_GOOG_jan2026_3136_seq500_seed42.txt" in submitter
    assert '"dataset_length": $EXPECTED_DATASET_LENGTH' in submitter
    assert 'dependency="aftercorr:$SMOKE_JOB"' in submitter
    assert 'dependency="aftercorr:$GENERATION_JOB"' in submitter
    assert "UPSTREAM_SMOKE_JOB" not in submitter
    assert "existing-artifact-prevalidated" in submitter
    assert "no redundant Slurm smoke" in submitter
    assert 'SMOKE_JOB="prevalidated"' in submitter
    assert '"checkpoint_step": $CHECKPOINT_STEP' in submitter
    assert '"padding_policy": "forbidden"' in submitter
    assert '"primary": "LOBBench unconditional WS-21"' in submitter
    assert "modified tracked LOBBench code" in submitter


def test_node_wrapper_uses_isolated_job_local_tmpdir() -> None:
    wrapper = Path("run/base_model/node_wrapper.sh").read_text(encoding="utf-8")
    assert "SIGMA0_JOB_TMPDIR=" in wrapper
    assert 'export TMPDIR="$SIGMA0_JOB_TMPDIR"' in wrapper
    assert "sibling-job epilog" in wrapper


def test_e2e_smoke_covers_all_architectures_and_real_lobbench() -> None:
    script = Path("run/benchmarking/model_zoo_e2e_smoke_array.batch").read_text(
        encoding="utf-8"
    )
    assert "ARCHITECTURES=(mamba3 s5 gdn kda transformer nsa)" in script
    assert "#SBATCH --array=0-5%3" in script
    assert "--checkpoint_step=\"$CHECKPOINT_STEP\"" in script
    assert "infer_squashfs_setup_wide_book" in script
    assert "--features=spread" in script
    assert "lobbench_summary.json" in script
    assert 'NVIDIA_SITE="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"' in script
    assert 'TORCH_LIB="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"' in script
    assert 'NVIDIA_SITE/cusparse/lib' in script
    assert "[cuda-preflight]" in script
