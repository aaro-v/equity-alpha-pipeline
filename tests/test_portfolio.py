from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_alpha.portfolio import drawdown_series, evaluate_predictions, max_drawdown


def test_sharpe_annualization_uses_holding_period() -> None:
    predictions = _simple_predictions()

    default_metrics, _, _ = evaluate_predictions(
        predictions, cost_bps=0.0, quantile=0.25, holding_period_days=21
    )
    faster_metrics, _, _ = evaluate_predictions(
        predictions, cost_bps=0.0, quantile=0.25, holding_period_days=5
    )

    default_sharpe = default_metrics.loc[default_metrics["model"] == "ridge"].iloc[0][
        "cost_adjusted_sharpe"
    ]
    faster_sharpe = faster_metrics.loc[faster_metrics["model"] == "ridge"].iloc[0][
        "cost_adjusted_sharpe"
    ]

    assert faster_sharpe == pytest.approx(default_sharpe * np.sqrt(21 / 5))


def test_positive_return_rate_reports_net_return_hit_rate() -> None:
    metrics, _, _ = evaluate_predictions(_simple_predictions(), cost_bps=0.0, quantile=0.25)

    hit_rate = metrics.loc[metrics["model"] == "ridge"].iloc[0]["positive_return_rate"]

    assert hit_rate == pytest.approx(0.75)


def test_portfolio_returns_include_book_exposure_diagnostics() -> None:
    _, portfolio_returns, _ = evaluate_predictions(
        _simple_predictions(), cost_bps=0.0, quantile=0.25
    )

    ridge_returns = portfolio_returns.loc[portfolio_returns["model"] == "ridge"]

    assert ridge_returns["long_names"].tolist() == [1, 1, 1, 1]
    assert ridge_returns["short_names"].tolist() == [1, 1, 1, 1]
    assert ridge_returns["gross_exposure"].tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert ridge_returns["net_exposure"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_drawdown_includes_loss_from_initial_capital() -> None:
    returns = pd.Series([-0.10, 0.05, 0.10])

    drawdowns = drawdown_series(returns)

    assert drawdowns.tolist() == pytest.approx([-0.10, -0.055, 0.0])
    assert max_drawdown(returns) == pytest.approx(-0.10)


def test_max_drawdown_handles_empty_returns() -> None:
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


def test_sharpe_annualization_rejects_non_positive_holding_period() -> None:
    with pytest.raises(ValueError, match="holding_period_days must be positive"):
        evaluate_predictions(_simple_predictions(), holding_period_days=0)


def test_evaluation_rejects_negative_costs() -> None:
    with pytest.raises(ValueError, match="cost_bps must be non-negative"):
        evaluate_predictions(_simple_predictions(), cost_bps=-0.1)


def test_evaluation_rejects_missing_prediction_columns() -> None:
    predictions = _simple_predictions().drop(columns=["future_return", "sector"])

    with pytest.raises(
        ValueError, match="predictions missing required columns: future_return, sector"
    ):
        evaluate_predictions(predictions)


def test_evaluation_rejects_frames_without_model_forecasts() -> None:
    protected_columns = ["date", "asset", "sector", "label", "future_return"]
    predictions = _simple_predictions().loc[:, protected_columns]

    with pytest.raises(ValueError, match="at least one model forecast column"):
        evaluate_predictions(predictions)


def test_evaluation_rejects_duplicate_asset_dates() -> None:
    predictions = pd.concat(
        [_simple_predictions(), _simple_predictions().iloc[[0]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="at most one row per date and asset"):
        evaluate_predictions(predictions)


def test_evaluation_rejects_missing_required_values() -> None:
    predictions = _simple_predictions()
    predictions.loc[0, "label"] = np.nan

    with pytest.raises(ValueError, match="missing required values: label"):
        evaluate_predictions(predictions)


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, "not-a-number"])
def test_evaluation_rejects_non_finite_model_scores(bad_score: object) -> None:
    predictions = _simple_predictions()
    if isinstance(bad_score, str):
        predictions["ridge"] = predictions["ridge"].astype(object)
    predictions.loc[0, "ridge"] = bad_score

    with pytest.raises(
        ValueError, match="model forecast columns must be numeric and finite: ridge"
    ):
        evaluate_predictions(predictions)


@pytest.mark.parametrize("quantile", [0.0, -0.1, 0.75])
def test_evaluation_rejects_invalid_quantile(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile must be greater than 0 and at most 0.5"):
        evaluate_predictions(_simple_predictions(), quantile=quantile)


def _simple_predictions() -> pd.DataFrame:
    rows = []
    date_returns = [
        (0.010, -0.004),
        (0.018, -0.002),
        (-0.006, 0.003),
        (0.014, -0.005),
    ]
    for offset, (long_return, short_return) in enumerate(date_returns):
        date = pd.Timestamp("2022-01-03") + pd.offsets.BDay(offset)
        rows.extend(
            [
                {
                    "date": date,
                    "asset": "A",
                    "sector": "Technology",
                    "label": long_return,
                    "future_return": long_return,
                    "ridge": 0.9,
                },
                {
                    "date": date,
                    "asset": "B",
                    "sector": "Health Care",
                    "label": 0.001,
                    "future_return": 0.001,
                    "ridge": 0.2,
                },
                {
                    "date": date,
                    "asset": "C",
                    "sector": "Financials",
                    "label": -0.001,
                    "future_return": -0.001,
                    "ridge": -0.2,
                },
                {
                    "date": date,
                    "asset": "D",
                    "sector": "Energy",
                    "label": short_return,
                    "future_return": short_return,
                    "ridge": -0.8,
                },
            ]
        )
    return pd.DataFrame(rows)
