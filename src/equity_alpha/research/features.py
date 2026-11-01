from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURES = ("value", "momentum", "quality", "low_volatility", "size", "beta", "liquidity")
FEATURE_COLUMNS = tuple(f"{name}_neutral_rank" for name in RAW_FEATURES)
WARMUP_DAYS = 131


def forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Close t to close t+h returns on the common trading calendar, excluding day t."""
    prices = panel.pivot(index="date", columns="asset", values="price").sort_index()
    future = prices.shift(-horizon) / prices - 1.0
    benchmark = panel.groupby("date")["benchmark_return"].first().reindex(prices.index)
    wealth = (1.0 + benchmark.astype(float)).cumprod()
    benchmark_forward = wealth.shift(-horizon) / wealth - 1.0
    result = future.rename_axis(columns="asset").stack().rename("future_return").reset_index()
    result["future_benchmark_return"] = result["date"].map(benchmark_forward)
    result["label"] = result["future_return"] - result["future_benchmark_return"]
    return result


def build_research_frame(panel: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Construct trailing factors and calendar-aligned labels without dropping missing ranks."""
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
        raise ValueError(f"panel missing required columns: {sorted(missing)}")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("panel has duplicate date/asset rows")
    data = panel.copy().sort_values(["date", "asset"], ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data["asset"] = data["asset"].astype(str)
    data["sector"] = data["sector"].astype(str)
    available = pd.to_datetime(data["fundamentals_available_date"])
    usable = available.notna() & (available <= data["date"])
    fundamental_cols = ["book_to_price", "gross_margin", "accruals", "debt_to_equity"]
    data.loc[~usable, fundamental_cols] = np.nan

    prices = data.pivot(index="date", columns="asset", values="price").sort_index()
    returns = data.pivot(index="date", columns="asset", values="daily_return").reindex_like(prices)
    benchmark = data.groupby("date")["benchmark_return"].first().reindex(prices.index)
    momentum = prices.shift(5) / prices.shift(131) - 1.0
    vol = returns.rolling(63, min_periods=21).std()
    market = pd.DataFrame(
        np.broadcast_to(benchmark.to_numpy()[:, None], returns.shape),
        index=returns.index,
        columns=returns.columns,
    ).where(returns.notna())
    cov = (returns * market).rolling(63, min_periods=21).mean()
    cov -= returns.rolling(63, min_periods=21).mean() * market.rolling(63, min_periods=21).mean()
    variance = (market**2).rolling(63, min_periods=21).mean()
    variance -= market.rolling(63, min_periods=21).mean() ** 2
    keys = pd.MultiIndex.from_frame(data[["date", "asset"]])
    for name, wide in {
        "momentum": momentum,
        "low_volatility": -vol,
        "beta": cov / variance,
    }.items():
        data[name] = wide.stack().reindex(keys).to_numpy()
    data["value"] = np.log(data["book_to_price"].where(data["book_to_price"] > 0))
    data["quality"] = (
        data["gross_margin"] - data["accruals"] - 0.15 * np.log1p(data["debt_to_equity"])
    )
    data["size"] = -np.log(data["market_cap"].where(data["market_cap"] > 0))
    data["liquidity"] = np.log(data["dollar_volume"].where(data["dollar_volume"] > 0))
    data[list(RAW_FEATURES)] = data[list(RAW_FEATURES)].replace([np.inf, -np.inf], np.nan)
    groups = data.groupby("date")[list(RAW_FEATURES)]
    lower, upper = groups.quantile(0.01), groups.quantile(0.99)
    clipped = data[list(RAW_FEATURES)].clip(
        lower=lower.reindex(data["date"]).set_axis(data.index),
        upper=upper.reindex(data["date"]).set_axis(data.index),
    )
    mean = clipped.groupby(data["date"]).transform("mean")
    std = clipped.groupby(data["date"]).transform("std").replace(0, np.nan)
    z = (clipped - mean) / std
    for factor in RAW_FEATURES:
        data[f"{factor}_observed"] = data[factor].notna()
        neutral = _neutralize(z[factor], z["beta"], data, factor != "beta")
        ranks = neutral.groupby(data["date"]).rank(pct=True)
        ranks -= ranks.groupby(data["date"]).transform("mean")
        data[f"{factor}_neutral_rank"] = ranks.fillna(0.0)
    count = prices.notna().cumsum().stack().reindex(keys).to_numpy()
    data = data.loc[count > WARMUP_DAYS]
    labels = forward_returns(panel.assign(asset=panel["asset"].astype(str)), horizon)
    data = data.merge(labels, on=["date", "asset"], how="inner").dropna(subset=["label"])
    columns = [
        "date",
        "asset",
        "sector",
        "label",
        "future_return",
        "future_benchmark_return",
        *RAW_FEATURES,
        *FEATURE_COLUMNS,
        *(f"{f}_observed" for f in RAW_FEATURES),
    ]
    return data[columns].sort_values(["date", "asset"], ignore_index=True)


def _neutralize(y: pd.Series, beta: pd.Series, data: pd.DataFrame, use_beta: bool) -> pd.Series:
    """Frisch-Waugh residuals from sector intercepts and a common date-specific beta slope."""
    sector_keys = [data["date"], data["sector"]]
    y_res = y - y.groupby(sector_keys).transform("mean")
    if not use_beta:
        return y_res
    b = beta.fillna(0.0).where(y.notna())
    b_res = b - b.groupby(sector_keys).transform("mean")
    numerator = (b_res * y_res).groupby(data["date"]).transform("sum")
    denominator = b_res.pow(2).groupby(data["date"]).transform("sum")
    slope = (numerator / denominator.replace(0, np.nan)).fillna(0.0)
    return y_res - slope * b_res
