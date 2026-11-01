from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from equity_alpha.data import SampleDataConfig
from equity_alpha.research.models import ModelConfig
from equity_alpha.validation import WalkForwardConfig
from equity_alpha.workflow.pipeline import PipelineConfig, run_research_pipeline


def test_artifacts_reconcile_to_portfolio_and_save_no_asset_rows(tmp_path: Path) -> None:
    config = PipelineConfig(
        sample_data=SampleDataConfig(n_assets=24, n_days=400),
        walk_forward=WalkForwardConfig(train_days=90, test_days=30, purge_days=21),
        models=ModelConfig(include_neural=False),
    )
    result = run_research_pipeline(config, tmp_path)
    metrics = pd.read_csv(tmp_path / "metrics.csv").set_index(["variant", "model"])
    daily = pd.read_csv(tmp_path / "portfolio_returns.csv")
    for key, frame in daily.groupby(["variant", "model"]):
        assert metrics.loc[key, "portfolio_return"] == pytest.approx(
            (1 + frame["net_return"]).prod() - 1
        )
        assert frame["wealth"].iloc[-1] - 1 == pytest.approx(metrics.loc[key, "portfolio_return"])
    for path in tmp_path.glob("*.csv"):
        assert "asset" not in pd.read_csv(path).columns
    assert result.report_path.exists()
    assert json.loads((tmp_path / "run.json").read_text())["parameters"]["horizon"] == 21
    assert (tmp_path / "equity_curve.png").stat().st_size > 1000
    assert (tmp_path / "ic_by_year.png").stat().st_size > 1000
