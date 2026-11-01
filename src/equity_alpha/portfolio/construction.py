from __future__ import annotations

import numpy as np
import pandas as pd

from equity_alpha.portfolio.metrics import max_drawdown
from equity_alpha.portfolio.uncertainty import moving_block_sharpe_interval
from equity_alpha.research.models import model_columns


def ic_series(predictions: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Daily cross-sectional Pearson and Spearman correlations, with ties preserved."""
    columns = columns or model_columns(predictions)
    rows = []
    grouped = predictions.groupby("date")
    for model in columns:
        for date, frame in grouped:
            sample = frame[[model, "label"]].dropna()
            valid = len(sample) >= 3 and sample[model].nunique() > 1
            ic = sample[model].corr(sample["label"]) if valid else np.nan
            rank_ic = sample[model].rank().corr(sample["label"].rank()) if valid else np.nan
            rows.append({"date": date, "model": model, "ic": ic, "rank_ic": rank_ic})
    return pd.DataFrame(rows, columns=["date", "model", "ic", "rank_ic"])


def nonoverlapping_dates(dates: pd.Series, calendar: pd.DatetimeIndex, horizon: int) -> list:
    selected, next_position = [], -1
    for date in sorted(pd.unique(dates)):
        position = calendar.get_loc(date)
        if position >= next_position:
            selected.append(date)
            next_position = position + horizon
    return selected


def evaluate_predictions(
    predictions: pd.DataFrame,
    daily_data: pd.DataFrame,
    cost_bps: float = 5.0,
    quantile: float = 0.2,
    holding_period_days: int = 21,
    execution_lag: int = 1,
    cap_neutral: bool = False,
    bootstrap_samples: int = 2000,
    bootstrap_block_days: int = 63,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold shares for h days; account for daily P&L and trades against drifted holdings."""
    if holding_period_days < 1:
        raise ValueError("holding_period_days must be positive")
    if cost_bps < 0 or not 0 < quantile <= 0.5:
        raise ValueError("costs must be non-negative and quantile in (0, 0.5]")
    if predictions.duplicated(["date", "asset"]).any():
        raise ValueError("predictions must contain at most one row per date and asset")
    models = model_columns(predictions)
    if not models:
        raise ValueError("predictions must include at least one model forecast column")
    if not np.isfinite(predictions[models].to_numpy(dtype=float)).all():
        raise ValueError("model forecast columns must be numeric and finite")
    calendar = pd.DatetimeIndex(sorted(daily_data["date"].unique()))
    sampled = nonoverlapping_dates(predictions["date"], calendar, holding_period_days)
    correlations = ic_series(predictions.loc[predictions["date"].isin(sampled)], models)
    metrics, portfolios, sectors = [], [], []
    for model in models:
        portfolio, attribution = portfolio_returns(
            predictions,
            daily_data,
            model,
            cost_bps,
            quantile,
            holding_period_days,
            execution_lag,
            cap_neutral,
        )
        returns = portfolio["net_return"]
        std = returns.std(ddof=1)
        corr = correlations.loc[correlations["model"].eq(model)]
        rank = corr["rank_ic"].dropna()
        rank_std = rank.std(ddof=1)
        annual_traded = portfolio["turnover"].sum() * 252 / len(portfolio)
        lower, upper = moving_block_sharpe_interval(
            returns, bootstrap_block_days, bootstrap_samples, seed
        )
        metrics.append(
            {
                "model": model,
                "mean_ic": corr["ic"].mean(),
                "mean_rank_ic": rank.mean(),
                "rank_ic_t_stat": rank.mean() / rank_std * np.sqrt(len(rank))
                if rank_std > 0
                else 0,
                "ic_observations": len(rank),
                "portfolio_return": (1 + returns).prod() - 1,
                "annualized_return": (1 + returns).prod() ** (252 / len(returns)) - 1,
                "cost_adjusted_sharpe": np.sqrt(252) * returns.mean() / std if std > 0 else 0,
                "max_drawdown": max_drawdown(returns),
                "sharpe_ci_lower": lower,
                "sharpe_ci_upper": upper,
                "mean_abs_size_exposure": portfolio.loc[
                    portfolio["rebalance"], "rebalance_size_exposure"
                ]
                .abs()
                .mean(),
                # Gross traded dollars / four: two sides, each with purchases and sales.
                "annual_one_way_turnover_per_side": annual_traded / 4,
            }
        )
        portfolios.append(portfolio.assign(model=model))
        sectors.append(attribution.assign(model=model))
    return pd.DataFrame(metrics), pd.concat(portfolios), pd.concat(sectors)


def portfolio_returns(
    predictions: pd.DataFrame,
    daily_data: pd.DataFrame,
    model: str,
    cost_bps: float,
    quantile: float,
    horizon: int,
    execution_lag: int = 1,
    cap_neutral: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = daily_data.pivot(index="date", columns="asset", values="daily_return").sort_index()
    returns.columns = returns.columns.astype(str)
    calendar = returns.index
    if execution_lag < 0:
        raise ValueError("execution_lag must be non-negative")
    signal_frames = dict(iter(predictions.groupby("date")))
    schedule = nonoverlapping_dates(predictions["date"], calendar, horizon)
    schedule = [
        d for d in schedule if calendar.get_loc(d) + execution_lag + horizon < len(calendar)
    ]
    if not schedule:
        raise ValueError("no complete holding interval available")
    signals = {calendar[calendar.get_loc(d) + execution_lag]: signal_frames[d] for d in schedule}
    signal_set = set(signals)
    first = calendar.get_loc(schedule[0]) + execution_lag
    last = calendar.get_loc(schedule[-1]) + execution_lag + horizon
    assets = returns.columns
    sector_by_asset = daily_data.drop_duplicates("asset").set_index("asset")["sector"]
    sector_names = sector_by_asset.reindex(assets).astype(str).to_numpy()
    sector_totals = dict.fromkeys(sorted(set(sector_names)), 0.0)
    positions = np.zeros(len(assets))
    nav, expiry = 1.0, -1
    rows = []
    for position in range(first, last + 1):
        date, old_nav = calendar[position], nav
        changes = returns.iloc[position].to_numpy(dtype=float)
        if np.any((positions != 0) & ~np.isfinite(changes)):
            raise ValueError(f"missing daily return for a held asset on {date}")
        pnl_by_asset = positions * np.nan_to_num(changes)
        gross_pnl = pnl_by_asset.sum()
        positions += pnl_by_asset
        nav += gross_pnl
        if nav <= 0:
            raise ValueError(f"portfolio NAV exhausted on {date}")
        traded = np.zeros(len(assets))
        rebalance = date in signal_set
        size_exposure, signal_date = np.nan, pd.NaT
        if rebalance:
            frame = signals[date].sort_values([model, "asset"], kind="stable")
            if cap_neutral:
                frame = frame.loc[frame["log_market_cap"].notna()]
            n = min(max(1, int(len(frame) * quantile)), len(frame) // 2)
            if n == 0:
                raise ValueError("portfolio needs at least two assets")
            target = pd.Series(0.0, index=assets)
            weights = pd.Series(0.0, index=frame["asset"])
            weights.loc[frame.tail(n)["asset"]] = 1 / n
            weights.loc[frame.head(n)["asset"]] = -1 / n
            if cap_neutral:
                weights = market_cap_neutral_weights(
                    weights, frame.set_index("asset")["log_market_cap"]
                )
            target.loc[weights.index] = weights * nav
            signal_date = frame["date"].iloc[0]
            if "log_market_cap" in frame:
                size_exposure = float(
                    weights @ frame.set_index("asset")["log_market_cap"].fillna(0)
                )
            traded = np.abs(target.to_numpy() - positions)
            positions = target.to_numpy(copy=True)
            expiry = position + horizon
        elif position == expiry:
            traded = np.abs(positions)
            positions = np.zeros(len(assets))
        costs = traded * cost_bps / 10_000
        nav -= costs.sum()
        if nav <= 0:
            raise ValueError(f"portfolio NAV exhausted after costs on {date}")
        for sector in sector_totals:
            mask = sector_names == sector
            sector_totals[sector] += float((pnl_by_asset[mask] - costs[mask]).sum() / old_nav)
        rows.append(
            {
                "date": date,
                "gross_return": gross_pnl / old_nav,
                "cost_return": costs.sum() / old_nav,
                "net_return": nav / old_nav - 1,
                "turnover": traded.sum() / old_nav,
                "rebalance": rebalance,
                "signal_date": signal_date,
                "rebalance_size_exposure": size_exposure,
                "gross_exposure": np.abs(positions).sum() / nav,
                "net_exposure": positions.sum() / nav,
            }
        )
    attribution = pd.DataFrame(
        [
            {"sector": sector, "net_return_contribution": value}
            for sector, value in sector_totals.items()
        ]
    )
    return pd.DataFrame(rows), attribution


def market_cap_neutral_weights(weights: pd.Series, log_cap: pd.Series) -> pd.Series:
    """Project dollar weights off intercept/log cap, preserving gross exposure two."""
    cap = log_cap.reindex(weights.index).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(cap)), cap - cap.mean()])
    original = weights.to_numpy(dtype=float)
    neutral = original - design @ np.linalg.lstsq(design, original, rcond=None)[0]
    gross = np.abs(neutral).sum()
    neutral = neutral * (2 / gross) if gross > 1e-10 else np.zeros(len(cap))
    return pd.Series(neutral, index=weights.index)
