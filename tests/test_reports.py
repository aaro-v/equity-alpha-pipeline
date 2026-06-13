from __future__ import annotations

import pandas as pd

from equity_alpha.reports import write_artifacts


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
                "turnover": 1.0,
                "net_return": 0.009,
                "model": "ridge|linear",
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "gross_return": 0.02,
                "turnover": 0.5,
                "net_return": 0.0195,
                "model": "ridge|linear",
            },
            {
                "date": pd.Timestamp("2024-01-04"),
                "gross_return": -0.03,
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
                "ridge|linear": -0.2,
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
    assert "## Validation Summary" in report
    assert (
        "| folds | start_date | end_date | prediction_rows | unique_assets | "
        "avg_assets_per_date | models |"
    ) in report
    assert "| 2 | 2024-01-02 | 2024-01-03 | 2 | 2 | 1.0000 | ridge\\|linear |" in report
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
        "| model | periods | avg_gross_return | avg_net_return | avg_turnover | "
        "avg_cost_drag |"
    ) in report
    assert "| ridge\\|linear | 3 | 0.0000 | -0.0007 | 0.6667 | 0.0007 |" in report
    assert "## Tail Risk Diagnostics" in report
    assert (
        "| model | worst_period_net_return | net_return_volatility | "
        "positive_period_rate | max_drawdown | max_drawdown_periods |"
    ) in report
    assert "| ridge\\|linear | -0.0305 | 0.0215 | 0.6667 | -0.0305 | 1 |" in report
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "equity_curve.png").exists()
