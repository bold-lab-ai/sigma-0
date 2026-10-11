"""Inference launch planning for base_model generated order streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from base_model.config import read_key_value_config


@dataclass(frozen=True)
class InferencePlan:
    """A validated inference request."""

    name: str
    config_path: str
    stock: str | None
    universe: str | None
    period: str | None
    output_root: str
    architecture: str
    checkpoint_path: str | None
    data_dir: str | None
    n_sequences: int
    n_cond_msgs: int
    n_gen_msgs: int
    seed: int

    def command(self, python: str = "python") -> list[str]:
        if not self.checkpoint_path or not self.data_dir:
            raise ValueError(
                f"{self.config_path}: checkpoint_path and data_dir are required "
                "to execute inference"
            )
        command = [
            python,
            "run/base_model/runtime/inference.py",
            f"--architecture={self.architecture}",
            "--token_mode=26tok",
            f"--ckpt_path={self.checkpoint_path}",
            f"--data_dir={self.data_dir}",
            f"--save_dir={self.output_root}",
            f"--n_sequences={self.n_sequences}",
            f"--n_cond_msgs={self.n_cond_msgs}",
            f"--n_gen_msgs={self.n_gen_msgs}",
            f"--seed={self.seed}",
        ]
        if self.stock:
            command.append(f"--stock={self.stock}")
        return command

    def shell_command(self, python: str = "python") -> str:
        return " ".join(shlex.quote(part) for part in self.command(python=python))


def build_inference_plan(
    config_path: str | Path,
    *,
    architecture: str | None = None,
) -> InferencePlan:
    """Create a validated inference plan from a flat config file."""
    config = read_key_value_config(config_path)
    stock = config.get("stock")
    universe = config.get("universe")
    if not stock and not universe:
        raise ValueError(f"{config_path}: provide either stock or universe")
    from base_model.models import resolve_architecture

    selected_architecture = resolve_architecture(
        architecture or config.get("architecture") or config.get("model")
    )
    return InferencePlan(
        name=config.get("name", Path(config_path).stem),
        config_path=str(config_path),
        stock=stock,
        universe=universe,
        period=config.get("period"),
        output_root=config.get("output_root", "artifacts/inference"),
        architecture=selected_architecture,
        checkpoint_path=config.get("checkpoint_path"),
        data_dir=config.get("data_dir"),
        n_sequences=int(config.get("n_sequences", "1024")),
        n_cond_msgs=int(config.get("n_cond_msgs", "500")),
        n_gen_msgs=int(config.get("n_gen_msgs", "500")),
        seed=int(config.get("seed", "42")),
    )


def run_inference(
    config_path: str | Path | None = None,
    dry_run: bool = True,
    *,
    architecture: str | None = None,
) -> InferencePlan:
    """Return a validated inference plan.

    `dry_run` is retained for command compatibility; this function does not
    launch GPU inference.
    """
    selected = config_path or "configs/inference/goog_2026.yaml"
    return build_inference_plan(selected, architecture=architecture)
