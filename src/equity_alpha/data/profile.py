from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_profile(panel_path: Path, out: Path) -> None:
    """Save aggregate distribution checks without exporting company observations."""
    panel = pd.read_parquet(panel_path)
    panel["asset"] = panel["asset"].astype(str)
    panel["sector"] = panel["sector"].astype(str)
    returns = panel.pivot(index="date", columns="asset", values="daily_return").astype(float)
    market = panel.groupby("date")["benchmark_return"].first().reindex(returns.index).astype(float)
    vol = returns.std() * np.sqrt(252)
    beta = returns.apply(lambda r: r.cov(market) / market.loc[r.notna()].var())
    corr = returns.corr().to_numpy()
    sectors = panel.drop_duplicates("asset").set_index("asset")["sector"].reindex(returns.columns)
    within = sectors.to_numpy()[:, None] == sectors.to_numpy()[None, :]
    upper = np.triu(np.ones(corr.shape, dtype=bool), 1)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    fields = [
        "book_to_price",
        "gross_margin",
        "accruals",
        "debt_to_equity",
        "market_cap",
        "dollar_volume",
    ]
    months = panel.groupby(panel["date"].dt.to_period("M"))["date"].max()
    monthly = panel.loc[panel["date"].isin(months)]
    profile = {
        "rows": len(panel),
        "assets": panel["asset"].nunique(),
        "trading_days": len(returns),
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "quantile_levels": quantiles,
        "annual_volatility": vol.quantile(quantiles).to_list(),
        "beta": beta.quantile(quantiles).to_list(),
        "pairwise_correlation": float(np.nanmean(corr[upper])),
        "within_sector_correlation": float(np.nanmean(corr[upper & within])),
        "between_sector_correlation": float(np.nanmean(corr[upper & ~within])),
        "pooled_excess_kurtosis": float(panel["daily_return"].kurt()),
        "abs_daily_return_above_5pct": float(panel["daily_return"].abs().gt(0.05).mean()),
        "fundamentals": {
            field: {
                "missing_share": float(monthly[field].isna().mean()),
                "quantiles": monthly[field].quantile(quantiles).to_list(),
            }
            for field in fields
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
