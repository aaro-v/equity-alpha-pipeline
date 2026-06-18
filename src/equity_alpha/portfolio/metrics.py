from __future__ import annotations

import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return drawdowns relative to the initial capital and subsequent peaks."""

    wealth = (1.0 + returns.astype(float)).cumprod()
    running_peak = wealth.cummax().clip(lower=1.0)
    return wealth / running_peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Return the worst peak-to-trough loss, including losses from initial capital."""

    drawdowns = drawdown_series(returns)
    return 0.0 if drawdowns.empty else float(drawdowns.min())
