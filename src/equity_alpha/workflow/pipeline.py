from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from equity_alpha.data import SampleDataConfig, generate_sample_panel
from equity_alpha.portfolio import evaluate_predictions
from equity_alpha.portfolio.construction import ic_series
from equity_alpha.reporting.artifacts import write_artifacts
from equity_alpha.research.diagnostics import factor_diagnostics, feature_coverage
from equity_alpha.research.features import build_research_frame
from equity_alpha.research.models import ModelConfig, predict_baselines, validate_model_config
from equity_alpha.validation import WalkForwardConfig, make_walk_forward_splits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    sample_data: SampleDataConfig = SampleDataConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    models: ModelConfig = ModelConfig()
    horizon: int = 21
    cost_bps: float = 5.0
    panel_path: Path | None = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.walk_forward.purge_days < self.horizon:
            raise ValueError("purge_days must be at least horizon")
        validate_model_config(self.models)


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
    """Run one frozen configuration on a parquet panel or a generated sample."""
    config = config or PipelineConfig()
    panel = (
        pd.read_parquet(config.panel_path)
        if config.panel_path
        else generate_sample_panel(config.sample_data)
    )
    panel["asset"] = panel["asset"].astype(str)
    panel["sector"] = panel["sector"].astype(str)
    logger.info("Building features for %s rows", f"{len(panel):,}")
    research = build_research_frame(panel, horizon=config.horizon)
    splits = make_walk_forward_splits(research["date"], config.walk_forward)
    if not splits:
        raise ValueError("not enough dates for the requested walk-forward configuration")
    fold_predictions, folds = [], []
    for split in splits:
        train = research.loc[research["date"].isin(split.train_dates)]
        test = research.loc[research["date"].isin(split.test_dates)]
        logger.info(
            "Fold %s: %s to %s",
            split.fold,
            split.test_dates.min().date(),
            split.test_dates.max().date(),
        )
        with threadpool_limits(limits=4):
            predictions = predict_baselines(train, test, config=config.models)
        fold_predictions.append(predictions.assign(fold=split.fold))
        folds.append(
            {
                "fold": split.fold,
                "train_start": split.train_dates.min(),
                "train_end": split.train_dates.max(),
                "test_start": split.test_dates.min(),
                "test_end": split.test_dates.max(),
                "train_rows": len(train),
                "test_rows": len(test),
                "test_assets": test["asset"].nunique(),
            }
        )
    all_predictions = pd.concat(fold_predictions, ignore_index=True)
    logger.info("Evaluating daily portfolios")
    metrics, portfolios, sectors = evaluate_predictions(
        all_predictions, panel, cost_bps=config.cost_bps, holding_period_days=config.horizon
    )
    report_path = None
    if output_dir is not None:
        logger.info("Computing factor diagnostics")
        decay, spreads = factor_diagnostics(
            research, panel, all_predictions["date"], config.horizon, config.cost_bps
        )
        coverage, coverage_by_date = feature_coverage(
            research.loc[research["date"].isin(all_predictions["date"])]
        )
        metadata = _metadata(config, panel, all_predictions)
        report_path = write_artifacts(
            output_dir,
            metrics,
            portfolios,
            sectors,
            ic_series(all_predictions),
            decay,
            spreads,
            coverage,
            coverage_by_date,
            pd.DataFrame(folds),
            metadata,
        )
    return PipelineResult(metrics, portfolios, sectors, all_predictions, report_path)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _metadata(config: PipelineConfig, panel: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    parameters = asdict(config)
    if config.panel_path is not None:
        parameters.pop("sample_data")
        parameters["panel_path"] = config.panel_path.as_posix()
    return {
        "parameters": parameters,
        "data_source": config.panel_path.as_posix()
        if config.panel_path
        else "synthetic (generated)",
        "panel_start": str(panel["date"].min().date()),
        "panel_end": str(panel["date"].max().date()),
        "panel_rows": len(panel),
        "panel_assets": panel["asset"].nunique(),
        "panel_trading_days": panel["date"].nunique(),
        "test_start": str(predictions["date"].min().date()),
        "test_end": str(predictions["date"].max().date()),
        "test_assets": predictions["asset"].nunique(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "pyproject.toml")),
        "python": platform.python_version(),
        "versions": {
            p: version(p)
            for p in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "matplotlib", "pyarrow"]
        },
        "portfolio_convention": (
            "long 100%, short 100%; shares held; close execution; final liquidation"
        ),
        "ic_convention": "non-overlapping target-horizon signal dates; daily series descriptive",
    }
