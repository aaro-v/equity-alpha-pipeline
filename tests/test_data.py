from __future__ import annotations

import pandas as pd

from equity_alpha.data import SampleDataConfig, generate_sample_panel


def test_sample_panel_is_deterministic() -> None:
    config = SampleDataConfig(n_assets=12, n_days=40, seed=11)

    first = generate_sample_panel(config)
    second = generate_sample_panel(config)

    pd.testing.assert_frame_equal(first, second)


def test_sample_panel_has_point_in_time_columns() -> None:
    panel = generate_sample_panel(SampleDataConfig(n_assets=8, n_days=30))

    assert {
        "date",
        "asset",
        "sector",
        "price",
        "daily_return",
        "benchmark_return",
        "market_cap",
        "dollar_volume",
        "fundamentals_available_date",
    }.issubset(panel.columns)
    assert (panel["fundamentals_available_date"] <= panel["date"]).all()
    assert panel.groupby("asset")["price"].nunique().min() > 1
