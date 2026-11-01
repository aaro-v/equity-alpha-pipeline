from __future__ import annotations

import pandas as pd
import pytest

from equity_alpha.data import SampleDataConfig
from equity_alpha.research.models import ModelConfig
from equity_alpha.validation import WalkForwardConfig, make_walk_forward_splits
from equity_alpha.workflow.pipeline import PipelineConfig, run_research_pipeline


def test_purge_separates_training_label_end_from_test() -> None:
    dates = pd.bdate_range("2020-01-01", periods=320)
    horizon = 21
    config = WalkForwardConfig(train_days=120, test_days=30, purge_days=horizon)
    splits = make_walk_forward_splits(dates, config)
    for split in splits:
        label_end = dates.get_loc(split.train_dates[-1]) + horizon
        assert label_end < dates.get_loc(split.test_dates[0])
    assert splits[0].test_dates[-1] + pd.offsets.BDay(1) == splits[1].test_dates[0]


def test_config_rejects_insufficient_purge() -> None:
    with pytest.raises(ValueError, match="purge_days must be at least horizon"):
        PipelineConfig(horizon=22)


def test_pipeline_compares_models_with_composite() -> None:
    config = PipelineConfig(
        sample_data=SampleDataConfig(n_assets=24, n_days=330, seed=23),
        walk_forward=WalkForwardConfig(train_days=90, test_days=30, purge_days=15),
        models=ModelConfig(include_neural=True, neural_epochs=2),
        horizon=15,
    )
    result = run_research_pipeline(config)
    assert {"composite", "ridge", "tree", "neural"} == set(result.metrics["model"])
    assert result.predictions["fold"].nunique() >= 2
    assert result.portfolio_returns["net_return"].notna().all()
    assert result.metrics["ic_observations"].max() < result.predictions["date"].nunique()
