from pathlib import Path

from base_model.inference import run_inference
from base_model.training import run_train


def test_training_plan_targets_self_contained_runtime() -> None:
    plan = run_train()
    assert plan.batch_script == "run/base_model/train_full_autoreg.batch"
    assert plan.nodes == 1
    assert (Path(plan.environment["WORKDIR"]) / "src/base_model/models/registry.py").is_file()
    assert "PYTHONPATH" not in plan.environment
    assert plan.environment["ARCHITECTURE"] == "mamba3"
    assert plan.environment["TOKEN_MODE"] == "26tok"
    assert "miniforge3/bin" in plan.environment["PATH"]
    # Every experiment tracks to W&B online under the oxford-lob entity, with no
    # exception for smoke runs. An untracked smoke is precisely the run whose
    # configuration nobody can reconstruct afterwards, which is why the rule has
    # no exception. This assertion previously required offline, encoding an
    # older convention that the configs themselves no longer follow.
    assert plan.environment["USE_WANDB"] == "True"
    assert plan.environment["WANDB_MODE"] == "online"
    assert plan.environment["WANDB_ENTITY"] == "oxford-lob"
    assert plan.environment["TICKERS"] == "2022"
    assert plan.environment["DATA_ROOT"].endswith("GOOG2016TO2021")
    assert plan.environment["DATA_INDEX_JSON"].endswith("configs/train/goog_2022_smoke_index.json")
    assert plan.environment["TRAIN_DATE_RANGE"] == "2022-01-03,2022-01-05"
    assert plan.environment["TEST_DATE_RANGE"] == "2022-01-06,2022-01-07"
    assert "env " in plan.shell_command(test_only=True)
    assert "--export=ALL" in plan.shell_command(test_only=True)
    assert plan.shell_command(test_only=True).endswith("run/base_model/train_full_autoreg.batch")


def test_inference_plan_has_output_root() -> None:
    plan = run_inference()
    assert plan.stock == "GOOG"
    assert plan.architecture == "mamba3"
    assert plan.output_root.endswith("goog_2026")
