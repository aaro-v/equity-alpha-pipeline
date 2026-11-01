from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_alpha.portfolio import drawdown_series, evaluate_predictions
from equity_alpha.portfolio.construction import ic_series


def _example() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2022-01-03", periods=5)
    panel = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "asset": ["A", "B"] * 5,
            "sector": ["Tech", "Energy"] * 5,
            "daily_return": [0, 0, 0.1, 0, 0.1, 0, 0, 0, 0, 0],
        }
    )
    # Daily horizon labels are deliberately huge: portfolio must never book them.
    predictions = panel.iloc[:6][["date", "asset", "sector"]].copy()
    predictions["future_return"] = 99.0
    predictions["label"] = [1.0, -1.0] * 3
    predictions["ridge"] = [1.0, 0.0] * 3
    return predictions, panel


def test_daily_pnl_does_not_book_overlapping_forward_labels() -> None:
    predictions, panel = _example()
    metrics, daily, _ = evaluate_predictions(
        predictions, panel, cost_bps=0, holding_period_days=2, execution_lag=0
    )
    assert daily["gross_return"].tolist() == pytest.approx([0, 0.1, 0.1, 0, 0])
    assert metrics.iloc[0]["portfolio_return"] == pytest.approx(0.21)
    expected = np.sqrt(252) * daily["net_return"].mean() / daily["net_return"].std(ddof=1)
    assert metrics.iloc[0]["cost_adjusted_sharpe"] == pytest.approx(expected)
    assert daily["rebalance"].tolist() == [True, False, True, False, False]


def test_costs_use_drifted_holdings_and_include_exit() -> None:
    predictions, panel = _example()
    metrics, daily, sectors = evaluate_predictions(
        predictions, panel, cost_bps=10, holding_period_days=2, execution_lag=0
    )
    # Entry 0.002; first two days gain 0.21; rebalance trades 0.210 dollars;
    # liquidation closes 1.208 dollars on each side and costs 0.002416.
    assert metrics.iloc[0]["portfolio_return"] == pytest.approx(1.208 - 0.000210 - 0.002416 - 1)
    assert daily.iloc[2]["turnover"] == pytest.approx(0.210 / 1.098)
    assert daily.iloc[-1]["turnover"] > 0
    assert sectors["net_return_contribution"].sum() == pytest.approx(daily["net_return"].sum())
    assert daily["cost_return"].to_numpy() == pytest.approx(daily["turnover"].to_numpy() * 0.001)


def test_missing_held_return_is_an_error() -> None:
    predictions, panel = _example()
    panel.loc[2, "daily_return"] = np.nan
    with pytest.raises(ValueError, match="missing daily return"):
        evaluate_predictions(predictions, panel, holding_period_days=2, execution_lag=0)


def test_next_close_execution_does_not_receive_the_execution_days_return() -> None:
    predictions, panel = _example()
    metrics, daily, _ = evaluate_predictions(
        predictions, panel, cost_bps=0, holding_period_days=2, execution_lag=1
    )
    assert metrics.iloc[0]["portfolio_return"] == pytest.approx(0.1)
    assert daily.iloc[0]["date"] == panel["date"].sort_values().unique()[1]
    assert daily.iloc[0]["gross_return"] == 0
    assert daily.iloc[0]["signal_date"] == predictions["date"].min()


def test_shuffled_labels_destroy_cross_sectional_ic() -> None:
    rng = np.random.default_rng(42)
    dates, assets = 80, 150
    score = rng.normal(size=(dates, assets))
    labels = score + rng.normal(scale=0.2, size=score.shape)
    base = pd.DataFrame(
        {
            "date": np.repeat(pd.bdate_range("2020-01-01", periods=dates), assets),
            "ridge": score.ravel(),
            "label": labels.ravel(),
        }
    )
    assert ic_series(base, ["ridge"])["rank_ic"].mean() > 0.9
    base["label"] = np.array([rng.permutation(row) for row in labels]).ravel()
    assert abs(ic_series(base, ["ridge"])["rank_ic"].mean()) < 0.03


def test_drawdown_includes_initial_loss() -> None:
    assert drawdown_series(pd.Series([-0.1, 0.05, 0.1])).tolist() == pytest.approx(
        [-0.1, -0.055, 0]
    )
