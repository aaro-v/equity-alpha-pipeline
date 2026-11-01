from __future__ import annotations

import numpy as np
import pandas as pd


def moving_block_sharpe_interval(
    returns: pd.Series,
    block_days: int = 63,
    samples: int = 2000,
    seed: int = 17,
) -> tuple[float, float]:
    """Pointwise percentile interval from overlapping, non-circular moving blocks."""
    values = returns.to_numpy(dtype=float)
    n = len(values)
    if n < 2 * block_days or not np.isfinite(values).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sharpe = []
    blocks = int(np.ceil(n / block_days))
    for begin in range(0, samples, 100):
        starts = rng.integers(0, n - block_days + 1, (min(100, samples - begin), blocks))
        indices = (starts[:, :, None] + np.arange(block_days)).reshape(len(starts), -1)[:, :n]
        draws = values[indices]
        standard_deviation = draws.std(axis=1, ddof=1)
        ratios = np.divide(
            np.sqrt(252) * draws.mean(axis=1),
            standard_deviation,
            out=np.full(len(draws), np.nan),
            where=standard_deviation > 1e-12,
        )
        sharpe.extend(ratios)
    finite = np.asarray(sharpe)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper)
