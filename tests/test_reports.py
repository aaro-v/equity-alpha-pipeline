from __future__ import annotations

import pandas as pd

from equity_alpha.reports import write_artifacts


def test_write_artifacts_escapes_markdown_table_values(tmp_path) -> None:
    metrics = pd.DataFrame(
        [
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
    assert "## Validation Summary" in report
    assert (
        "| folds | start_date | end_date | prediction_rows | unique_assets | "
        "avg_assets_per_date | models |"
    ) in report
    assert "| 2 | 2024-01-02 | 2024-01-03 | 2 | 2 | 1.0000 | ridge\\|linear |" in report
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "equity_curve.png").exists()
