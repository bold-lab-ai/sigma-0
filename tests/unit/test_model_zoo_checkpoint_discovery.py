import json
from pathlib import Path

import pytest

from run.base_model.find_model_zoo_checkpoint import find_checkpoint


def _checkpoint(root: Path, name: str, step: int, **config: object) -> Path:
    manager = root / name
    step_dir = manager / str(step)
    (step_dir / "state").mkdir(parents=True)
    (step_dir / "metadata").mkdir()
    (step_dir / "metadata" / "metadata").write_text(
        json.dumps({"config": config}), encoding="utf-8"
    )
    return manager


def test_find_checkpoint_filters_architecture_shape_and_token_mode(tmp_path) -> None:
    expected = _checkpoint(
        tmp_path,
        "mamba-smoke",
        21,
        architecture="mamba3",
        token_mode="26tok",
        d_model=64,
        n_layers=2,
    )
    _checkpoint(
        tmp_path,
        "mamba-wrong-width",
        30_000,
        architecture="mamba3",
        token_mode="26tok",
        d_model=256,
        n_layers=6,
    )
    _checkpoint(
        tmp_path,
        "transformer-smoke",
        21,
        architecture="transformer",
        token_mode="26tok",
        d_model=64,
        n_layers=2,
    )

    match = find_checkpoint(tmp_path, "mamba3", d_model=64, n_layers=2)

    assert match.manager == expected.resolve()
    assert match.step == 21


def test_find_checkpoint_rejects_missing_match(tmp_path) -> None:
    with pytest.raises(ValueError, match="no complete 26-token kda checkpoint"):
        find_checkpoint(tmp_path, "kda", d_model=64, n_layers=2)
