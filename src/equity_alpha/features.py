from __future__ import annotations

import numpy as np
import pandas as pd


RAW_FEATURES: tuple[str, ...] = (
    "value",
    "momentum",
    "quality",
    "low_volatility",
    "size",
    "beta",
    "liquidity",
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(f"{name}_neutral_rank" for name in RAW_FEATURES)


def build_research_frame(panel: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Build point-in-time features and forward excess-return labels."""

    required = {
        "date",
        "asset",
        "sector",
        "price",
        "daily_return",
        "benchmark_return",
        "market_cap",
        "dollar_volume",
        "book_to_price",
        "gross_margin",
        "accruals",
        "debt_to_equity",
        "fundamentals_available_date",
    }
    missing = required.difference(panel.columns)
    if missing:
        msg = f"panel missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["fundamentals_available_date"] = pd.to_datetime(data["fundamentals_available_date"])
    data = data.loc[data["fundamentals_available_date"] <= data["date"]]
    data = data.sort_values(["asset", "date"]).reset_index(drop=True)

    by_asset = data.groupby("asset", group_keys=False)
    data["future_return"] = by_asset["price"].shift(-horizon) / data["price"] - 1.0
    data["future_benchmark_return"] = by_asset["benchmark_return"].transform(
        lambda returns: returns.rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    )
    data["label"] = data["future_return"] - data["future_benchmark_return"]

    data["value"] = np.log(data["book_to_price"].clip(lower=1e-6))
    data["momentum"] = by_asset["price"].pct_change(126).shift(5)
    data["quality"] = data["gross_margin"] - data["accruals"] - 0.15 * np.log1p(data["debt_to_equity"])
    data["low_volatility"] = -by_asset["daily_return"].rolling(63, min_periods=21).std().reset_index(
        level=0, drop=True
    )
    data["size"] = -np.log(data["market_cap"].clip(lower=1.0))
    data["liquidity"] = np.log(data["dollar_volume"].clip(lower=1.0))

    benchmark = data["benchmark_return"]
    rolling_cov = by_asset.apply(
        lambda asset_frame: asset_frame["daily_return"]
        .rolling(63, min_periods=21)
        .cov(benchmark.loc[asset_frame.index])
    )
    if isinstance(rolling_cov.index, pd.MultiIndex):
        rolling_cov.index = rolling_cov.index.droplevel(0)
    market_var = by_asset["benchmark_return"].rolling(63, min_periods=21).var().reset_index(
        level=0, drop=True
    )
    data["beta"] = rolling_cov / market_var.reindex(data.index)

    for feature in RAW_FEATURES:
        data[f"{feature}_z"] = data.groupby("date")[feature].transform(_winsorized_zscore)

    for feature in RAW_FEATURES:
        data[f"{feature}_neutral"] = data.groupby("date", group_keys=False).apply(
            lambda frame, col=f"{feature}_z": _neutralize(frame, col)
        )
        data[f"{feature}_neutral_rank"] = data.groupby("date")[f"{feature}_neutral"].rank(pct=True) - 0.5

    useful_columns = [
        "date",
        "asset",
        "sector",
        "label",
        "future_return",
        "future_benchmark_return",
        *FEATURE_COLUMNS,
    ]
    research = data[useful_columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return research.sort_values(["date", "asset"], ignore_index=True)


def _winsorized_zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    lower = values.quantile(0.01)
    upper = values.quantile(0.99)
    clipped = values.clip(lower, upper)
    std = clipped.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (clipped - clipped.mean()) / std


def _neutralize(frame: pd.DataFrame, column: str) -> pd.Series:
    y = frame[column].astype(float)
    if y.notna().sum() < 3:
        return pd.Series(np.nan, index=frame.index)

    sectors = pd.get_dummies(frame["sector"], dtype=float, drop_first=False)
    beta = frame["beta_z"].astype(float).fillna(0.0).to_numpy().reshape(-1, 1)
    intercept = np.ones((len(frame), 1))
    x = np.column_stack([intercept, sectors.to_numpy(), beta])
    mask = y.notna().to_numpy()
    coef, *_ = np.linalg.lstsq(x[mask], y.to_numpy()[mask], rcond=None)
    fitted = x @ coef
    return pd.Series(y.to_numpy() - fitted, index=frame.index)
