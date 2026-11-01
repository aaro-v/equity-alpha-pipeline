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
from equity_alpha.research.attribution import factor_attribution
from equity_alpha.research.diagnostics import factor_diagnostics, feature_coverage
from equity_alpha.research.features import build_research_frame, forward_returns
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
    execution_lag: int = 1
    robustness: bool = True
    bootstrap_samples: int = 2000
    bootstrap_block_days: int = 63

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.execution_lag < 0:
            raise ValueError("horizon must be positive and execution_lag non-negative")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.walk_forward.purge_days < max(self.horizon, self.horizon + self.execution_lag - 1):
            raise ValueError("purge_days must be at least horizon + execution_lag - 1 and horizon")
        if self.bootstrap_samples < 1 or self.bootstrap_block_days < 1:
            raise ValueError("bootstrap samples and block days must be positive")
        validate_model_config(self.models)


@dataclass(frozen=True)
class PipelineResult:
    metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    sector_attribution: pd.DataFrame
    predictions: pd.DataFrame
    report_path: Path | None = None


def _fit_folds(research, panel, config, variant, training_log):
    splits = make_walk_forward_splits(research["date"], config.walk_forward)
    if not splits:
        raise ValueError("not enough dates for the requested walk-forward configuration")
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    predictions, folds = [], []
    for split in splits:
        if calendar.get_loc(
            split.train_dates[-1]
        ) + config.horizon + config.execution_lag > calendar.get_loc(split.test_dates[0]):
            raise ValueError("a training label ends after the test signal close")
        train = research.loc[research["date"].isin(split.train_dates)]
        test = research.loc[research["date"].isin(split.test_dates)]
        logger.info(
            "%s fold %s: %s to %s",
            variant,
            split.fold,
            split.test_dates.min().date(),
            split.test_dates.max().date(),
        )
        logs = []
        with threadpool_limits(limits=4):
            prediction = predict_baselines(
                train,
                test,
                config=config.models,
                horizon=config.horizon,
                execution_lag=config.execution_lag,
                training_log=logs,
            )
        predictions.append(prediction.assign(fold=split.fold))
        training_log.extend(dict(row, variant=variant, fold=split.fold) for row in logs)
        folds.append(
            {
                "variant": variant,
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
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def run_research_pipeline(
    config: PipelineConfig | None = None, output_dir: Path | None = None
) -> PipelineResult:
    """Run purged forecasts, execution comparisons and size-controlled portfolios."""
    config = config or PipelineConfig()
    panel = (
        pd.read_parquet(config.panel_path)
        if config.panel_path
        else generate_sample_panel(config.sample_data)
    )
    panel["asset"] = panel["asset"].astype(str)
    panel["sector"] = panel["sector"].astype(str)
    logger.info("Building features for %s rows", f"{len(panel):,}")
    research = build_research_frame(panel, config.horizon, config.execution_lag)
    training_log = []
    forecasts, folds = _fit_folds(research, panel, config, "main", training_log)
    variants = [("main", forecasts, config.execution_lag, False)]
    if config.robustness:
        if config.execution_lag:
            same = forecasts.drop(columns=["label", "future_return"]).merge(
                forward_returns(panel, config.horizon, 0)[
                    ["date", "asset", "label", "future_return"]
                ],
                on=["date", "asset"],
                how="inner",
            )
            variants.append(("same_close", same, 0, False))
        variants.append(("cap_neutral_book", forecasts, config.execution_lag, True))
        logger.info("Building size-neutralized features")
        neutral = build_research_frame(panel, config.horizon, config.execution_lag, True)
        neutral_forecasts, neutral_folds = _fit_folds(
            neutral, panel, config, "size_neutral_features", training_log
        )
        folds = pd.concat([folds, neutral_folds], ignore_index=True)
        variants.extend(
            [
                ("size_neutral_features", neutral_forecasts, config.execution_lag, False),
                ("size_neutral", neutral_forecasts, config.execution_lag, True),
            ]
        )
    metric_rows, portfolios, sectors = [], [], []
    for name, prediction, lag, neutral_book in variants:
        logger.info("Evaluating %s", name)
        metrics, daily, sector = evaluate_predictions(
            prediction,
            panel,
            cost_bps=config.cost_bps,
            holding_period_days=config.horizon,
            execution_lag=lag,
            cap_neutral=neutral_book,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_block_days=config.bootstrap_block_days,
            seed=config.models.seed,
        )
        metric_rows.append(metrics.assign(variant=name, execution_lag=lag))
        portfolios.append(daily.assign(variant=name, execution_lag=lag))
        sectors.append(sector.assign(variant=name))
    metrics = pd.concat(metric_rows, ignore_index=True)
    portfolios = pd.concat(portfolios, ignore_index=True)
    sectors = pd.concat(sectors, ignore_index=True)
    report_path = None
    if output_dir is not None:
        logger.info("Computing factor diagnostics and attribution")
        decay, spreads = factor_diagnostics(
            research,
            panel,
            forecasts["date"],
            config.horizon,
            config.cost_bps,
            config.execution_lag,
        )
        coverage, coverage_by_date = feature_coverage(
            research.loc[research["date"].isin(forecasts["date"])]
        )
        attribution, loadings = factor_attribution(portfolios, spreads)
        report_path = write_artifacts(
            output_dir,
            metrics,
            portfolios,
            sectors,
            ic_series(forecasts),
            decay,
            spreads,
            coverage,
            coverage_by_date,
            folds,
            _metadata(config, panel, forecasts),
            attribution,
            loadings,
            pd.DataFrame(training_log),
        )
    return PipelineResult(metrics, portfolios, sectors, forecasts, report_path)


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
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pyarrow"]
    if config.models.include_neural:
        packages.append("torch")
    else:
        parameters["models"] = {"include_neural": False, "seed": config.models.seed}
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
        "versions": {p: version(p) for p in packages},
        "portfolio_convention": (
            f"gross exposure two; shares held; execution lag {config.execution_lag}; "
            "final liquidation"
        ),
        "ic_convention": "non-overlapping target-horizon signal dates; daily series descriptive",
        "attribution_convention": (
            "OLS with intercept; adjusted returns retain alpha; "
            "net factors matched by execution lag"
        ),
        "multiple_comparisons": {
            "forecasts": 4 if config.models.include_neural else 3,
            "factors": 7,
            "interval_adjustment": "none (pointwise)",
        },
    }
