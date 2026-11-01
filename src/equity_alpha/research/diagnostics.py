from __future__ import annotations

import numpy as np
import pandas as pd

from equity_alpha.portfolio.construction import (
    ic_series,
    nonoverlapping_dates,
    portfolio_returns,
)
from equity_alpha.research.features import FEATURE_COLUMNS, RAW_FEATURES, forward_returns


def factor_diagnostics(
    research: pd.DataFrame,
    panel: pd.DataFrame,
    test_dates: pd.Series,
    horizon: int,
    cost_bps: float,
    execution_lag: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor IC decay and investable quintile spreads restricted to test signal dates."""
    test = research.loc[research["date"].isin(test_dates)]
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rows, spreads = [], []
    for h in sorted({1, 5, 10, 21, 63, horizon}):
        labels = forward_returns(panel, h, execution_lag)[
            ["date", "asset", "label", "future_return"]
        ]
        selected = nonoverlapping_dates(test["date"], calendar, h)
        sample = test.loc[test["date"].isin(selected)].drop(columns=["label", "future_return"])
        sample = sample.merge(labels, on=["date", "asset"]).dropna(subset=["label"])
        for variant, columns in [
            ("raw", list(RAW_FEATURES)),
            ("neutral_rank", list(FEATURE_COLUMNS)),
        ]:
            correlations = ic_series(sample, columns)
            for factor, column in zip(RAW_FEATURES, columns, strict=True):
                values = correlations.loc[correlations["model"].eq(column)]
                rank = values["rank_ic"].dropna()
                std = rank.std(ddof=1)
                rows.append(
                    {
                        "factor": factor,
                        "variant": variant,
                        "horizon": h,
                        "mean_ic": values["ic"].mean(),
                        "mean_rank_ic": rank.mean(),
                        "rank_ic_std": std,
                        "rank_ic_t_stat": (
                            rank.mean() / std * np.sqrt(len(rank)) if std > 0 else 0
                        ),
                        "observations": len(rank),
                        "share_positive": (rank > 0).mean(),
                    }
                )
    for lag in sorted({0, execution_lag}):
        for factor, column in zip(RAW_FEATURES, FEATURE_COLUMNS, strict=True):
            predictions = test[["date", "asset", "sector", "label", "future_return", column]]
            portfolio, _ = portfolio_returns(
                predictions, panel, column, cost_bps, 0.2, horizon, lag
            )
            spreads.append(portfolio.assign(factor=factor, execution_lag=lag))
    return pd.DataFrame(rows), pd.concat(spreads, ignore_index=True)


def feature_coverage(research: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [f"{factor}_observed" for factor in RAW_FEATURES]
    by_date = (
        research.groupby("date")[columns]
        .mean()
        .rename(columns=dict(zip(columns, RAW_FEATURES, strict=True)))
    )
    summary = pd.DataFrame(
        {
            "factor": RAW_FEATURES,
            "observed_share": [research[c].mean() for c in columns],
            "imputed_share": [1 - research[c].mean() for c in columns],
        }
    )
    return summary, by_date.reset_index()
