from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_training_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "run/base_model/train_base_model.py"), "--help"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout


def test_restored_batch_files_parse() -> None:
    for script in ("run/base_model/train_full_autoreg.batch", "run/base_model/node_wrapper.sh"):
        result = subprocess.run(
            ["bash", "-n", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.returncode == 0
