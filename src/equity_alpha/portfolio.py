from __future__ import annotations

import numpy as np
import pandas as pd

from equity_alpha.models import model_columns


def evaluate_predictions(
    predictions: pd.DataFrame,
    cost_bps: float = 5.0,
    quantile: float = 0.2,
    holding_period_days: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate model forecasts with IC and long-short portfolio metrics."""

    _validate_inputs(
        predictions=predictions,
        cost_bps=cost_bps,
        quantile=quantile,
        holding_period_days=holding_period_days,
    )

    metric_rows: list[dict[str, float | str]] = []
    return_rows: list[pd.DataFrame] = []
    sector_rows: list[dict[str, float | str]] = []

    models = model_columns(predictions)
    if not models:
        msg = "predictions must include at least one model forecast column"
        raise ValueError(msg)

    for model in models:
        ic_by_date = predictions.groupby("date").apply(
            lambda frame, col=model: _safe_corr(frame[col], frame["label"])
        )
        rank_ic_by_date = predictions.groupby("date").apply(
            lambda frame, col=model: _safe_corr(frame[col].rank(), frame["label"].rank())
        )
        portfolio = _portfolio_returns(predictions, model, cost_bps=cost_bps, quantile=quantile)
        returns = portfolio["net_return"]
        annualizer = np.sqrt(252 / holding_period_days)
        return_std = returns.std(ddof=0)
        sharpe = 0.0 if return_std == 0 else annualizer * returns.mean() / return_std
        drawdown = _max_drawdown(returns)

        metric_rows.append(
            {
                "model": model,
                "mean_ic": float(ic_by_date.mean()),
                "mean_rank_ic": float(rank_ic_by_date.mean()),
                "portfolio_return": float(returns.sum()),
                "positive_return_rate": float((returns > 0).mean()),
                "cost_adjusted_sharpe": float(sharpe),
                "max_drawdown": float(drawdown),
                "mean_turnover": float(portfolio["turnover"].mean()),
            }
        )
        return_rows.append(portfolio.assign(model=model))
        sector_rows.extend(_sector_attribution(predictions, model))

    metrics = pd.DataFrame(metric_rows).sort_values("cost_adjusted_sharpe", ascending=False)
    portfolio_returns = pd.concat(return_rows, ignore_index=True)
    sector_attribution = pd.DataFrame(sector_rows)
    return metrics, portfolio_returns, sector_attribution


def _validate_inputs(
    predictions: pd.DataFrame, cost_bps: float, quantile: float, holding_period_days: int
) -> None:
    required_columns = {"date", "asset", "sector", "label", "future_return"}
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        msg = f"predictions missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)
    if holding_period_days <= 0:
        msg = "holding_period_days must be positive"
        raise ValueError(msg)
    if cost_bps < 0:
        msg = "cost_bps must be non-negative"
        raise ValueError(msg)
    if not 0 < quantile <= 0.5:
        msg = "quantile must be greater than 0 and at most 0.5"
        raise ValueError(msg)


def _portfolio_returns(
    predictions: pd.DataFrame, model: str, cost_bps: float, quantile: float
) -> pd.DataFrame:
    previous_weights: pd.Series | None = None
    rows: list[dict[str, float | pd.Timestamp]] = []

    for date, frame in predictions.sort_values(["date", model]).groupby("date"):
        n_names = max(1, int(len(frame) * quantile))
        longs = frame.nlargest(n_names, model)
        shorts = frame.nsmallest(n_names, model)
        weights = pd.Series(0.0, index=frame["asset"])
        weights.loc[longs["asset"]] = 1.0 / n_names
        weights.loc[shorts["asset"]] = -1.0 / n_names

        gross_return = float(
            (weights.reindex(frame["asset"]).to_numpy() * frame["future_return"]).sum()
        )
        if previous_weights is None:
            turnover = float(weights.abs().sum())
        else:
            aligned = pd.concat([previous_weights, weights], axis=1).fillna(0.0)
            turnover = float((aligned.iloc[:, 1] - aligned.iloc[:, 0]).abs().sum())
        net_return = gross_return - turnover * cost_bps / 10_000
        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "turnover": turnover,
                "net_return": net_return,
            }
        )
        previous_weights = weights

    return pd.DataFrame(rows)


def _max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min())


def _sector_attribution(predictions: pd.DataFrame, model: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for sector, frame in predictions.groupby("sector"):
        rank_ic = frame.groupby("date").apply(
            lambda date_frame, col=model: _safe_corr(
                date_frame[col].rank(), date_frame["label"].rank()
            )
        )
        rows.append({"model": model, "sector": sector, "rank_ic": float(rank_ic.mean())})
    return rows


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right))
