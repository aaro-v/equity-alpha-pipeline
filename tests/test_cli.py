from __future__ import annotations

from typer.testing import CliRunner

from equity_alpha.cli import app


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_sample_data_writes_file(tmp_path) -> None:
    output = tmp_path / "sample.parquet"

    result = CliRunner().invoke(
        app,
        ["sample-data", "--output", str(output), "--assets", "12", "--days", "180"],
    )

    assert result.exit_code == 0
    assert output.exists()
