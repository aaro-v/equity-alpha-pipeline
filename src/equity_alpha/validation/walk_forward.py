from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 504
    test_days: int = 63
    purge_days: int = 21
    embargo_days: int = 0


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def make_walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex, config: WalkForwardConfig | None = None
) -> list[WalkForwardSplit]:
    """Create rolling walk-forward splits with a purge gap before each test window."""

    config = config or WalkForwardConfig()
    _validate_walk_forward_config(config)
    unique_dates = pd.DatetimeIndex(
        pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values()
    )
    splits: list[WalkForwardSplit] = []

    test_start = config.train_days + config.purge_days
    fold = 0
    while test_start + config.test_days <= len(unique_dates):
        train_end = test_start - config.purge_days
        train_start = max(0, train_end - config.train_days)
        test_end = test_start + config.test_days

        train_dates = unique_dates[train_start:train_end]
        test_dates = unique_dates[test_start:test_end]
        if len(train_dates) and len(test_dates):
            splits.append(
                WalkForwardSplit(fold=fold, train_dates=train_dates, test_dates=test_dates)
            )
            fold += 1

        test_start += config.test_days + config.embargo_days

    return splits


def _validate_walk_forward_config(config: WalkForwardConfig) -> None:
    if config.train_days <= 0:
        msg = "train_days must be positive"
        raise ValueError(msg)
    if config.test_days <= 0:
        msg = "test_days must be positive"
        raise ValueError(msg)
    if config.purge_days < 0:
        msg = "purge_days must be non-negative"
        raise ValueError(msg)
    if config.embargo_days < 0:
        msg = "embargo_days must be non-negative"
        raise ValueError(msg)
