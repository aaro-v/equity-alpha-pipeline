from __future__ import annotations

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
