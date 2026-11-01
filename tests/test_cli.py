from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from equity_alpha.cli import app


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_sample_data_writes_four_files(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["sample-data", "--out", str(tmp_path), "--assets", "18"])
    assert result.exit_code == 0, result.exception
    assert {p.name for p in tmp_path.iterdir()} == {
        "panel.parquet",
        "prices_daily.parquet",
        "fundamentals_quarterly.parquet",
        "universe.csv",
    }


def test_cli_loads_parquet_and_records_source(tmp_path: Path) -> None:
    data, reports = tmp_path / "data", tmp_path / "reports"
    runner = CliRunner()
    generated = runner.invoke(
        app, ["sample-data", "--out", str(data), "--assets", "24", "--days", "400"]
    )
    assert generated.exit_code == 0, generated.exception
    result = runner.invoke(
        app,
        [
            "run",
            "--panel",
            str(data / "panel.parquet"),
            "--output-dir",
            str(reports),
            "--train-days",
            "90",
            "--test-days",
            "30",
            "--horizon",
            "10",
            "--purge-days",
            "10",
            "--no-include-neural",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert (reports / "research_report.md").exists()
    metadata = json.loads((reports / "run.json").read_text())
    assert metadata["data_source"] == (data / "panel.parquet").as_posix()
    assert "sample_data" not in metadata["parameters"]
    assert metadata["panel_trading_days"] == 400
    assert metadata["parameters"]["horizon"] == 10
    assert set(pd.read_csv(reports / "metrics.csv")["model"]) == {"composite", "ridge", "tree"}
    assert not list(reports.glob("*.parquet"))


def test_cli_rejects_insufficient_purge() -> None:
    result = CliRunner().invoke(app, ["run", "--horizon", "22"])
    assert result.exit_code != 0
    assert "purge_days must cover horizon" in result.stdout + result.stderr


def test_build_data_calls_builder_without_network(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        "equity_alpha.data.market_data.build.build", lambda out, limit: calls.append((out, limit))
    )
    result = CliRunner().invoke(app, ["build-data", "--out", str(tmp_path), "--limit", "2"])
    assert result.exit_code == 0, result.exception
    assert calls == [(tmp_path, 2)]
