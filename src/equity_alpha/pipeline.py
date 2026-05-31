from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from equity_alpha.data import SampleDataConfig, generate_sample_panel
from equity_alpha.features import build_research_frame
from equity_alpha.models import ModelConfig, predict_baselines
from equity_alpha.portfolio import evaluate_predictions
from equity_alpha.reports import write_artifacts
from equity_alpha.validation import WalkForwardConfig, make_walk_forward_splits


@dataclass(frozen=True)
class PipelineConfig:
    sample_data: SampleDataConfig = SampleDataConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    models: ModelConfig = ModelConfig()
    horizon: int = 21
    cost_bps: float = 5.0


@dataclass(frozen=True)
class PipelineResult:
    metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    sector_attribution: pd.DataFrame
    predictions: pd.DataFrame
    report_path: Path | None = None


def run_research_pipeline(
    config: PipelineConfig | None = None, output_dir: Path | None = None
) -> PipelineResult:
    """Run the deterministic sample cross-sectional alpha pipeline."""

    config = config or PipelineConfig()
    _validate_pipeline_config(config)
    panel = generate_sample_panel(config.sample_data)
    research = build_research_frame(panel, horizon=config.horizon)
    splits = make_walk_forward_splits(research["date"], config.walk_forward)
    if not splits:
        msg = "not enough dates for the requested walk-forward configuration"
        raise ValueError(msg)

    fold_predictions: list[pd.DataFrame] = []
    for split in splits:
        train = research.loc[research["date"].isin(split.train_dates)]
        test = research.loc[research["date"].isin(split.test_dates)]
        predictions = predict_baselines(train, test, config=config.models)
        fold_predictions.append(predictions.assign(fold=split.fold))

    all_predictions = pd.concat(fold_predictions, ignore_index=True)
    metrics, portfolio_returns, sector_attribution = evaluate_predictions(
        all_predictions, cost_bps=config.cost_bps, holding_period_days=config.horizon
    )
    report_path = None
    if output_dir is not None:
        report_path = write_artifacts(
            output_dir=output_dir,
            metrics=metrics,
            portfolio_returns=portfolio_returns,
            sector_attribution=sector_attribution,
            predictions=all_predictions,
        )
    return PipelineResult(
        metrics=metrics,
        portfolio_returns=portfolio_returns,
        sector_attribution=sector_attribution,
        predictions=all_predictions,
        report_path=report_path,
    )


def _validate_pipeline_config(config: PipelineConfig) -> None:
    if config.horizon <= 0:
        msg = "horizon must be positive"
        raise ValueError(msg)
    if config.cost_bps < 0:
        msg = "cost_bps must be non-negative"
        raise ValueError(msg)
