from __future__ import annotations

import numpy as np
import pandas as pd


def factor_attribution(
    portfolios: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive OLS on matched daily net returns, retaining alpha in adjusted returns."""
    summaries, loadings = [], []
    for (variant, model), group in portfolios.groupby(["variant", "model"]):
        lag = int(group["execution_lag"].iloc[0])
        factors = factor_returns.loc[factor_returns["execution_lag"].eq(lag)].pivot(
            index="date", columns="factor", values="net_return"
        )
        sample = group.set_index("date")[["net_return"]].join(factors, how="inner").dropna()
        names = list(factors.columns)
        x, y = sample[names].to_numpy(), sample["net_return"].to_numpy()
        design = np.column_stack([np.ones(len(x)), x])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        error = y - design @ coefficients
        sse = np.sum(error**2)
        total = np.sum((y - y.mean()) ** 2)
        adjusted = y - x @ coefficients[1:]
        size = names.index("size")
        size_design = np.column_stack([np.ones(len(x)), x[:, size]])
        size_coef = np.linalg.lstsq(size_design, y, rcond=None)[0]
        size_error = y - size_design @ size_coef
        size_adjusted = y - x[:, size] * size_coef[1]
        summaries.append(
            {
                "variant": variant,
                "model": model,
                "observations": len(y),
                "r_squared": 1 - sse / total if total > 0 else np.nan,
                "alpha_daily": coefficients[0],
                "alpha_annual": coefficients[0] * 252,
                "residual_sharpe": _sharpe(adjusted),
                "size_only_r_squared": 1 - np.sum(size_error**2) / total if total > 0 else np.nan,
                "size_only_residual_sharpe": _sharpe(size_adjusted),
            }
        )
        for index, factor in enumerate(names):
            reduced = np.delete(design, index + 1, axis=1)
            reduced_error = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
            reduced_sse = np.sum(reduced_error**2)
            loadings.append(
                {
                    "variant": variant,
                    "model": model,
                    "factor": factor,
                    "loading": coefficients[index + 1],
                    "annual_mean_contribution": coefficients[index + 1] * x[:, index].mean() * 252,
                    "partial_r_squared": (reduced_sse - sse) / reduced_sse
                    if reduced_sse > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(loadings)


def _sharpe(values: np.ndarray) -> float:
    std = values.std(ddof=1)
    return float(np.sqrt(252) * values.mean() / std) if std > 1e-12 else float("nan")
