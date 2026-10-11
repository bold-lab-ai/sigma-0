from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_package_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "run/benchmarking/run_package.py"), "--help"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout
    assert "--execute" in result.stdout


def test_package_sbatch_parses() -> None:
    result = subprocess.run(
        ["bash", "-n", "run/benchmarking/smoke_package.sbatch"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.returncode == 0
