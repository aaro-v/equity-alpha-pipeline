from __future__ import annotations

import json

import pandas as pd
import pytest

from equity_alpha.reporting.artifacts import _ic_stability_summary, _signal_spread_summary
from equity_alpha.reports import write_artifacts


def test_signal_spread_summary_compares_top_and_bottom_scores() -> None:
    predictions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset": "A",
                "sector": "Energy",
                "label": 0.04,
                "future_return": 0.04,
                "ridge": 0.9,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset": "B",
                "sector": "Technology",
                "label": 0.01,
                "future_return": 0.01,
                "ridge": 0.1,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset": "C",
                "sector": "Financials",
                "label": -0.01,
                "future_return": -0.01,
                "ridge": -0.8,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "asset": "A",
                "sector": "Energy",
                "label": 0.03,
                "future_return": 0.03,
                "ridge": 0.7,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "asset": "B",
                "sector": "Technology",
                "label": 0.00,
                "future_return": 0.00,
                "ridge": 0.2,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "asset": "C",
                "sector": "Financials",
                "label": -0.02,
                "future_return": -0.02,
                "ridge": -0.6,
            },
        ]
    )

    summary = _signal_spread_summary(predictions)

    row = summary.iloc[0]
    assert row["model"] == "ridge"
    assert row["dates"] == 2
    assert row["long_avg_label"] == pytest.approx(0.035)
    assert row["short_avg_label"] == pytest.approx(-0.015)
    assert row["label_spread"] == pytest.approx(0.05)
    assert row["future_return_spread"] == pytest.approx(0.05)


def test_ic_stability_summary_reports_daily_cross_sectional_persistence() -> None:
    predictions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset": "A",
                "sector": "Energy",
                "label": 0.04,
                "future_return": 0.04,
                "ridge": 0.9,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset": "B",
                "sector": "Technology",
                "label": 0.01,
                "future_return": 0.01,
                "ridge": 0.1,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset": "C",
                "sector": "Financials",
                "label": -0.01,
                "future_return": -0.01,
                "ridge": -0.8,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "asset": "A",
                "sector": "Energy",
                "label": -0.02,
                "future_return": -0.02,
                "ridge": 0.8,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "asset": "B",
                "sector": "Technology",
                "label": 0.00,
                "future_return": 0.00,
                "ridge": 0.1,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "asset": "C",
                "sector": "Financials",
                "label": 0.03,
                "future_return": 0.03,
                "ridge": -0.7,
            },
        ]
    )

    summary = _ic_stability_summary(predictions)

    row = summary.iloc[0]
    assert row["model"] == "ridge"
    assert row["dates"] == 2
    assert row["valid_ic_dates"] == 2
    assert row["positive_ic_rate"] == pytest.approx(0.5)
    assert row["avg_rank_ic"] == pytest.approx(0.0)
    assert row["rank_ic_volatility"] == pytest.approx(1.0)
    assert row["positive_rank_ic_rate"] == pytest.approx(0.5)
    assert row["worst_rank_ic"] == pytest.approx(-1.0)
    assert row["best_rank_ic"] == pytest.approx(1.0)


def test_write_artifacts_escapes_markdown_table_values(tmp_path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": "tree",
                "mean_ic": 0.223456,
                "mean_rank_ic": 0.334567,
                "portfolio_return": 0.445678,
                "positive_return_rate": 1.0,
                "cost_adjusted_sharpe": 2.734567,
                "max_drawdown": -0.025678,
                "mean_turnover": 0.66789,
            },
            {
                "model": "ridge|linear",
                "mean_ic": 0.123456,
                "mean_rank_ic": 0.234567,
                "portfolio_return": 0.345678,
                "positive_return_rate": 1.0,
                "cost_adjusted_sharpe": 1.234567,
                "max_drawdown": -0.045678,
                "mean_turnover": 0.56789,
            }
        ]
    )
    portfolio_returns = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "gross_return": 0.01,
                "gross_exposure": 2.0,
                "net_exposure": 0.0,
                "long_names": 1,
                "short_names": 1,
                "turnover": 1.0,
                "net_return": 0.009,
                "model": "ridge|linear",
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "gross_return": 0.02,
                "gross_exposure": 2.0,
                "net_exposure": 0.0,
                "long_names": 1,
                "short_names": 1,
                "turnover": 0.5,
                "net_return": 0.0195,
                "model": "ridge|linear",
            },
            {
                "date": pd.Timestamp("2024-01-04"),
                "gross_return": -0.03,
                "gross_exposure": 1.8,
                "net_exposure": 0.2,
                "long_names": 1,
                "short_names": 1,
                "turnover": 0.5,
                "net_return": -0.0305,
                "model": "ridge|linear",
            },
        ]
    )
    sector_attribution = pd.DataFrame(
        [{"model": "ridge|linear", "sector": "Energy|Utilities", "rank_ic": 0.345678}]
    )
    predictions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset": "A",
                "sector": "Energy|Utilities",
                "label": 0.01,
                "future_return": 0.01,
                "ridge|linear": 0.2,
                "fold": 1,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "asset": "B",
                "sector": "Financials",
                "label": -0.01,
                "future_return": -0.01,
                "ridge|linear": None,
                "fold": 2,
            }
        ]
    )

    report_path = write_artifacts(
        tmp_path,
        metrics=metrics,
        portfolio_returns=portfolio_returns,
        sector_attribution=sector_attribution,
        predictions=predictions,
    )

    report = report_path.read_text(encoding="utf-8")
    assert r"ridge\|linear" in report
    assert r"Energy\|Utilities" in report
    assert "0.1235" in report
    assert "## Benchmark Comparison" in report
    assert (
        "| model | benchmark | rank_ic_delta | sharpe_delta | "
        "portfolio_return_delta |"
    ) in report
    assert "| tree | tree | 0.0000 | 0.0000 | 0.0000 |" in report
    assert "| ridge\\|linear | tree | -0.1000 | -1.5000 | -0.1000 |" in report
    assert "## IC Stability" in report
    assert (
        "| model | dates | valid_ic_dates | avg_ic | ic_volatility | positive_ic_rate | "
        "avg_rank_ic | rank_ic_volatility | positive_rank_ic_rate | worst_rank_ic | "
        "best_rank_ic |"
    ) in report
    assert (
        "| ridge\\|linear | 2 | 0 | nan | nan | nan | nan | nan | nan | nan | nan |"
    ) in report

    assert "## Validation Summary" in report
    assert (
        "| folds | start_date | end_date | prediction_rows | unique_assets | "
        "avg_assets_per_date | models |"
    ) in report
    assert "| 2 | 2024-01-02 | 2024-01-03 | 2 | 2 | 1.0000 | ridge\\|linear |" in report
    assert "## Prediction Coverage" in report
    assert (
        "| model | prediction_rows | dates | min_assets_per_date | avg_assets_per_date | "
        "missing_score_rate | unique_score_rate |"
    ) in report
    assert "| ridge\\|linear | 2 | 2 | 1 | 1.0000 | 0.5000 | 0.5000 |" in report
    assert "## Score Dispersion Diagnostics" in report
    assert (
        "| model | dates | score_mean | score_std | avg_daily_score_std | "
        "min_daily_unique_scores | avg_daily_unique_score_rate |"
    ) in report
    assert "| ridge\\|linear | 1 | 0.2000 | 0.0000 | 0.0000 | 1 | 1.0000 |" in report
    assert "## Signal Spread Diagnostics" in report
    assert (
        "| model | dates | long_avg_label | short_avg_label | label_spread | "
        "long_avg_future_return | short_avg_future_return | future_return_spread |"
    ) in report
    assert (
        "| ridge\\|linear | 0 | nan | nan | nan | nan | nan | nan |"
    ) in report
    assert "## Fold Breadth" in report
    assert (
        "| fold | start_date | end_date | dates | prediction_rows | unique_assets | "
        "min_assets_per_date | avg_assets_per_date | model_count |"
    ) in report
    assert (
        "| 1 | 2024-01-02 | 2024-01-02 | 1 | 1 | 1 | 1 | 1.0000 | 1 |"
    ) in report
    assert (
        "| 2 | 2024-01-03 | 2024-01-03 | 1 | 1 | 1 | 1 | 1.0000 | 1 |"
    ) in report
    assert "## Fold Stability" in report
    assert "| model | fold | start_date | end_date | rank_ic | net_return |" in report
    assert "| ridge\\|linear | 1 | 2024-01-02 | 2024-01-02 | nan | 0.0090 |" in report
    assert "| ridge\\|linear | 2 | 2024-01-03 | 2024-01-03 | nan | 0.0195 |" in report
    assert "## Fold Robustness" in report
    assert (
        "| model | folds | avg_fold_rank_ic | rank_ic_std | positive_fold_rate | "
        "worst_fold_net_return | best_fold_net_return |"
    ) in report
    assert "| ridge\\|linear | 2 | nan | nan | 1.0000 | 0.0090 | 0.0195 |" in report
    assert "## Portfolio Diagnostics" in report
    assert (
        "| model | periods | avg_gross_return | avg_net_return | avg_gross_exposure | "
        "avg_abs_net_exposure | avg_turnover | avg_cost_drag |"
    ) in report
    assert (
        "| ridge\\|linear | 3 | 0.0000 | -0.0007 | 1.9333 | 0.0667 | "
        "0.6667 | 0.0007 |"
    ) in report
    assert "## Turnover Stability" in report
    assert (
        "| model | periods | avg_turnover | turnover_volatility | p95_turnover | "
        "max_turnover | high_turnover_rate |"
    ) in report
    assert (
        "| ridge\\|linear | 3 | 0.6667 | 0.2357 | 0.9500 | 1.0000 | 0.3333 |"
    ) in report
    assert "## Return Consistency" in report
    assert (
        "| model | periods | mean_net_return | median_net_return | cumulative_return | "
        "return_skew | positive_period_rate |"
    ) in report
    assert (
        "| ridge\\|linear | 3 | -0.0007 | 0.0090 | -0.0027 | "
        "-1.4281 | 0.6667 |"
    ) in report
    assert "## Tail Risk Diagnostics" in report
    assert (
        "| model | worst_period_net_return | net_return_var_5 | "
        "net_return_expected_shortfall_5 | net_return_volatility | positive_period_rate | "
        "max_drawdown | max_drawdown_periods |"
    ) in report
    assert (
        "| ridge\\|linear | -0.0305 | -0.0265 | -0.0305 | 0.0215 | 0.6667 | "
        "-0.0305 | 1 |"
    ) in report
    assert "## Underwater Diagnostics" in report
    assert (
        "| model | periods | ending_growth | peak_growth | underwater_period_rate | "
        "max_drawdown | max_drawdown_start | max_drawdown_trough | "
        "recovered_after_trough |"
    ) in report
    assert (
        "| ridge\\|linear | 3 | 0.9973 | 1.0287 | 0.3333 | -0.0305 | "
        "2024-01-03 | 2024-01-04 | False |"
    ) in report
    assert "## Book Exposure Diagnostics" in report
    assert (
        "| model | side | avg_names_per_date | top_sector | top_sector_weight | "
        "avg_label | avg_future_return |"
    ) in report
    assert (
        "| ridge\\|linear | short | 1.0000 | Energy\\|Utilities | "
        "0.5000 | 0.0000 | 0.0000 |"
    ) in report
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "equity_curve.png").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_format"] == 1
    artifacts = {artifact["path"]: artifact for artifact in manifest["artifacts"]}
    assert set(artifacts) == {
        "metrics.csv",
        "portfolio_returns.csv",
        "sector_attribution.csv",
        "predictions.parquet",
        "equity_curve.png",
        "research_report.md",
    }
    assert artifacts["metrics.csv"]["rows"] == 2
    assert artifacts["metrics.csv"]["columns"] == list(metrics.columns)
    assert artifacts["metrics.csv"]["bytes"] == (tmp_path / "metrics.csv").stat().st_size
    assert len(artifacts["metrics.csv"]["sha256"]) == 64


def test_write_artifacts_rejects_missing_required_columns(tmp_path) -> None:
    metrics, portfolio_returns, sector_attribution, predictions = _minimal_artifact_frames()
    bad_predictions = predictions.drop(columns=["future_return"])

    with pytest.raises(
        ValueError, match="predictions missing required columns: future_return"
    ):
        write_artifacts(
            tmp_path,
            metrics=metrics,
            portfolio_returns=portfolio_returns,
            sector_attribution=sector_attribution,
            predictions=bad_predictions,
        )


def test_write_artifacts_requires_model_score_columns(tmp_path) -> None:
    metrics, portfolio_returns, sector_attribution, predictions = _minimal_artifact_frames()
    metadata_only = predictions.drop(columns=["ridge"])

    with pytest.raises(
        ValueError, match="predictions must include at least one model score column"
    ):
        write_artifacts(
            tmp_path,
            metrics=metrics,
            portfolio_returns=portfolio_returns,
            sector_attribution=sector_attribution,
            predictions=metadata_only,
        )


def _minimal_artifact_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame(
        [
            {
                "model": "ridge",
                "mean_rank_ic": 0.1,
                "portfolio_return": 0.02,
                "cost_adjusted_sharpe": 0.8,
            }
        ]
    )
    portfolio_returns = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "model": "ridge",
                "gross_return": 0.01,
                "net_return": 0.009,
                "turnover": 0.5,
            }
        ]
    )
    sector_attribution = pd.DataFrame(
        [{"model": "ridge", "sector": "Energy", "rank_ic": 0.1}]
    )
    predictions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "asset": "A",
                "sector": "Energy",
                "label": 0.01,
                "future_return": 0.01,
                "fold": 1,
                "ridge": 0.2,
            }
        ]
    )
    return metrics, portfolio_returns, sector_attribution, predictions
