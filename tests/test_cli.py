from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

from equity_alpha.cli import app


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_module_execution_prints_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "equity_alpha.cli", "version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "0.1.0" in result.stdout


def test_cli_sample_data_writes_file(tmp_path) -> None:
    output = tmp_path / "sample.parquet"

    result = CliRunner().invoke(
        app,
        ["sample-data", "--output", str(output), "--assets", "12", "--days", "180"],
    )

    assert result.exit_code == 0
    assert output.exists()


def test_cli_run_accepts_research_controls(tmp_path) -> None:
    output_dir = tmp_path / "reports"

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--output-dir",
            str(output_dir),
            "--assets",
            "24",
            "--days",
            "260",
            "--horizon",
            "10",
            "--cost-bps",
            "2.5",
            "--train-days",
            "90",
            "--test-days",
            "30",
            "--purge-days",
            "10",
            "--embargo-days",
            "3",
            "--neural-epochs",
            "3",
            "--no-include-neural",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "research_report.md").exists()
    assert "cost_adjusted_sharpe" in result.stdout


def test_cli_run_rejects_invalid_neural_epochs(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--output-dir",
            str(tmp_path / "reports"),
            "--neural-epochs",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "neural-epochs" in result.stdout + result.stderr
