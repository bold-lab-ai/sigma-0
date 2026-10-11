"""Training launch planning for the restored base_model autoregressive job."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from base_model.config import prefixed_values, read_key_value_config


@dataclass(frozen=True)
class TrainingLaunchPlan:
    """A concrete SLURM launch plan derived from a train config."""

    name: str
    config_path: str
    batch_script: str
    workdir: str
    nodes: int
    time_limit: str
    job_name: str
    environment: dict[str, str]
    extra_args: tuple[str, ...] = ()

    def sbatch_command(self, test_only: bool = False) -> list[str]:
        """Build the shell command without executing it."""
        command = ["env"]
        command.extend(f"{key}={value}" for key, value in sorted(self.environment.items()))
        command.append("sbatch")
        command.append("--export=ALL")
        if test_only:
            command.append("--test-only")
        command.extend(
            [
                f"--nodes={self.nodes}",
                f"--time={self.time_limit}",
                f"--job-name={self.job_name}",
            ]
        )
        command.extend(self.extra_args)
        command.append(self.batch_script)
        return command

    def shell_command(self, test_only: bool = False) -> str:
        """Return a copy-pasteable shell command."""
        return " ".join(shlex.quote(part) for part in self.sbatch_command(test_only=test_only))


def build_train_plan(
    config_path: str | Path,
    *,
    architecture: str | None = None,
) -> TrainingLaunchPlan:
    """Create a launch plan for the restored `train_full_autoreg.batch` entry."""
    config = read_key_value_config(config_path)
    batch_script = config.get("legacy_batch", "train_full_autoreg.batch")
    workdir = config.get("legacy_workdir", "")
    if not workdir:
        raise ValueError(f"{config_path}: legacy_workdir is required for full_autoreg launches")
    if workdir == ".":
        workdir = str(Path.cwd().resolve())

    env = {"WORKDIR": workdir, **prefixed_values(config, "env_")}
    from base_model.models import resolve_architecture

    if architecture is not None:
        selected_architecture = resolve_architecture(architecture)
    else:
        selected_architecture = resolve_architecture(
            config.get("architecture") or env.get("ARCHITECTURE"),
            ssm_type=env.get("SSM_TYPE"),
            model_type=env.get("MODEL_TYPE"),
        )
    env["ARCHITECTURE"] = selected_architecture
    env.pop("SSM_TYPE", None)
    env.pop("MODEL_TYPE", None)
    env["TOKEN_MODE"] = "26tok"
    quant_root = env.get("QUANT_ROOT")
    if quant_root and "PATH" not in env:
        env["PATH"] = f"{quant_root}/miniforge3/bin:/usr/bin:/bin:/usr/sbin:/sbin"

    return TrainingLaunchPlan(
        name=config.get("name", Path(config_path).stem),
        config_path=str(config_path),
        batch_script=batch_script,
        workdir=workdir,
        nodes=int(config.get("sbatch_nodes", "1")),
        time_limit=config.get("sbatch_time", "00:10:00"),
        job_name=config.get("sbatch_job_name", config.get("name", "base_model-train")),
        environment=env,
        # e.g. "--gres=gpu:1 --mem=115000M" for sub-node (single-GPU) launches
        extra_args=tuple(shlex.split(config.get("sbatch_extra_args", ""))),
    )


def run_train(
    config_path: str | Path | None = None,
    dry_run: bool = True,
    *,
    architecture: str | None = None,
) -> TrainingLaunchPlan:
    """Return the train launch plan for `config_path`.

    `dry_run` is retained for compatibility with earlier CLI calls; this function
    always plans a launch and never starts a job by itself.
    """
    selected = config_path or "configs/train/mamba3_smoke.yaml"
    return build_train_plan(selected, architecture=architecture)
