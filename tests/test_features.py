from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_alpha.data import SampleDataConfig, generate_sample_panel
from equity_alpha.research.features import FEATURE_COLUMNS, build_research_frame, forward_returns


def test_missing_quality_does_not_remove_companies() -> None:
    panel = generate_sample_panel(SampleDataConfig(n_assets=30, n_days=220))
    panel.loc[panel["asset"].eq("EQ000"), "gross_margin"] = np.nan
    research = build_research_frame(panel)
    assert research.groupby("date")["asset"].nunique().min() == 30
    missing = research.loc[research["asset"].eq("EQ000")]
    assert missing["quality_neutral_rank"].eq(0).all()
    assert not missing["quality_observed"].any()
    assert research[list(FEATURE_COLUMNS)].abs().max().max() <= 0.5
    assert research[list(FEATURE_COLUMNS)].notna().all().all()


@pytest.mark.parametrize("neutralize_size", [False, True])
def test_future_perturbations_do_not_change_past_features(neutralize_size: bool) -> None:
    panel = generate_sample_panel(SampleDataConfig(n_assets=24, n_days=240))
    cutoff = sorted(panel["date"].unique())[190]
    original = build_research_frame(panel, neutralize_size=neutralize_size)
    modified = panel.copy()
    future = modified["date"] > cutoff
    modified.loc[future, "price"] *= 1.5
    modified.loc[future, "daily_return"] += 0.2
    modified.loc[future, "gross_margin"] = -0.5
    modified.loc[future, "market_cap"] *= 1.5
    changed = build_research_frame(modified, neutralize_size=neutralize_size)
    columns = ["date", "asset", *FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(
        original.loc[original["date"] <= cutoff, columns].reset_index(drop=True),
        changed.loc[changed["date"] <= cutoff, columns].reset_index(drop=True),
    )


def test_forward_label_compounds_next_days_and_uses_calendar_gaps() -> None:
    dates = pd.bdate_range("2022-01-03", periods=4)
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "asset": ["A"] * 4 + ["B"] * 4,
            "price": [100, 110, 121, 150] * 2,
            "benchmark_return": [0.5, 0.01, 0.02, 0.03] * 2,
        }
    )
    panel = panel.drop(index=5)  # B is absent on day 1: never shift by observed rows.
    labels = forward_returns(panel, 2).set_index(["date", "asset"])
    assert labels.loc[(dates[0], "A"), "label"] == pytest.approx(0.21 - 0.0302)
    assert labels.loc[(dates[0], "B"), "future_return"] == pytest.approx(0.21)
    delayed = forward_returns(panel, 1, execution_lag=1).set_index(["date", "asset"])
    assert delayed.loc[(dates[0], "A"), "label"] == pytest.approx(0.1 - 0.02)


def test_unavailable_fundamentals_are_masked_without_changing_trailing_prices() -> None:
    panel = generate_sample_panel(SampleDataConfig(n_assets=24, n_days=200))
    panel["fundamentals_available_date"] = pd.Timestamp("2030-01-01")
    result = build_research_frame(panel)
    assert not result["quality_observed"].any()
    assert result["quality_neutral_rank"].eq(0).all()
    assert result["momentum_observed"].all()
