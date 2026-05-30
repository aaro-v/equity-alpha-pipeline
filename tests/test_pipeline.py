from __future__ import annotations

import pandas as pd
import pytest

from equity_alpha.data import SampleDataConfig, generate_sample_panel
from equity_alpha.features import build_research_frame
from equity_alpha.models import ModelConfig
from equity_alpha.pipeline import PipelineConfig, run_research_pipeline
from equity_alpha.validation import WalkForwardConfig, make_walk_forward_splits


def test_walk_forward_splits_are_purged() -> None:
    panel = generate_sample_panel(SampleDataConfig(n_assets=16, n_days=260))
    research = build_research_frame(panel, horizon=10)
    config = WalkForwardConfig(train_days=90, test_days=30, purge_days=10, embargo_days=3)

    splits = make_walk_forward_splits(research["date"], config)

    assert splits
    for split in splits:
        assert split.train_dates.max() < split.test_dates.min()
        assert len(split.train_dates) <= config.train_days
        assert len(split.test_dates) == config.test_days


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (WalkForwardConfig(train_days=0), "train_days must be positive"),
        (WalkForwardConfig(test_days=0), "test_days must be positive"),
        (WalkForwardConfig(purge_days=-1), "purge_days must be non-negative"),
        (WalkForwardConfig(embargo_days=-1), "embargo_days must be non-negative"),
    ],
)
def test_walk_forward_splits_reject_invalid_config(
    config: WalkForwardConfig, message: str
) -> None:
    dates = pd.bdate_range("2021-01-01", periods=300)

    with pytest.raises(ValueError, match=message):
        make_walk_forward_splits(dates, config)


def test_pipeline_runs_and_returns_metrics() -> None:
    config = PipelineConfig(
        sample_data=SampleDataConfig(n_assets=28, n_days=320, seed=23),
        walk_forward=WalkForwardConfig(train_days=120, test_days=30, purge_days=15, embargo_days=5),
        models=ModelConfig(include_neural=False),
        horizon=15,
    )

    result = run_research_pipeline(config)

    assert {"ridge", "tree"} <= set(result.metrics["model"])
    assert result.predictions["fold"].nunique() >= 2
    assert result.portfolio_returns["net_return"].notna().all()
    assert result.sector_attribution["rank_ic"].notna().any()
