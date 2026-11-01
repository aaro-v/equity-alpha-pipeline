from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_alpha.data.market_data.build import _asof
from equity_alpha.data.market_data.sec import instant, trailing_twelve_months
from equity_alpha.research.features import _neutralize


def test_asof_waits_until_after_filing_and_expires_stale_data() -> None:
    dates = pd.Series(pd.to_datetime(["2021-02-05", "2021-02-08", "2022-06-01"]))
    filing = pd.DataFrame(
        {
            "end": [pd.Timestamp("2020-12-31")],
            "filed": [pd.Timestamp("2021-02-05")],
            "val": [42.0],
        }
    )
    values, available = _asof(dates, filing)
    assert np.isnan(values.iloc[0])
    assert values.iloc[1] == 42
    assert available.iloc[1] == dates.iloc[1]
    assert np.isnan(values.iloc[2])


def test_sec_keeps_first_filed_balance_sheet_value() -> None:
    facts = [
        {"end": "2020-12-31", "val": 100, "filed": "2021-02-05", "form": "10-K"},
        {"end": "2020-12-31", "val": 200, "filed": "2022-02-05", "form": "10-K"},
    ]
    payload = {"facts": {"us-gaap": {"Assets": {"units": {"USD": facts}}}}}
    result = instant(payload, ["Assets"])
    assert result.iloc[0]["val"] == 100
    assert result.iloc[0]["filed"] == pd.Timestamp("2021-02-05")


def test_ttm_uses_latest_filing_among_quarterly_components() -> None:
    dates = [
        ("2020-01-01", "2020-03-31", "2020-05-01"),
        ("2020-04-01", "2020-06-30", "2020-08-01"),
        ("2020-07-01", "2020-09-30", "2021-03-01"),
        ("2020-10-01", "2020-12-31", "2021-02-01"),
    ]
    facts = [
        {"start": start, "end": end, "val": 10, "filed": filed, "form": "10-Q"}
        for start, end, filed in dates
    ]
    payload = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": facts}}}}}
    result = trailing_twelve_months(payload, ["Revenues"])
    assert result.iloc[0]["val"] == 40
    assert result.iloc[0]["filed"] == pd.Timestamp("2021-03-01")


def test_vectorized_neutralization_matches_least_squares_with_missing_values() -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "date": np.repeat(pd.bdate_range("2020-01-01", periods=2), 20),
            "sector": ["A", "B"] * 20,
        }
    )
    beta = pd.Series(rng.normal(size=40))
    y = pd.Series(rng.normal(size=40))
    y.iloc[[2, 6, 31]] = np.nan
    result = _neutralize(y, beta, frame, True)
    for _, block in frame.groupby("date"):
        indices = block.index
        x = np.column_stack([pd.get_dummies(block["sector"], dtype=float), beta.loc[indices]])
        mask = y.loc[indices].notna()
        coefficients = np.linalg.lstsq(x[mask], y.loc[indices][mask], rcond=None)[0]
        expected = y.loc[indices] - x @ coefficients
        assert result.loc[indices][mask].to_numpy() == pytest.approx(expected[mask].to_numpy())
