"""Point-in-time company fundamentals from SEC EDGAR XBRL ``companyfacts``.

Every value is tagged with the date its filing was accepted by EDGAR
(``filed``). When a figure is re-reported later (comparative columns,
amendments), only the FIRST filing is kept, so the series reflects what an
investor could actually have known at the time, not restated history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .http import get_json

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT"}

# Concept fallbacks, in priority order (taxonomy names changed over time, e.g. ASC 606).
REVENUE = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
]
COST_OF_REVENUE = [
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
    "CostOfServices",
    "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
]
GROSS_PROFIT = ["GrossProfit"]
NET_INCOME = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
OPERATING_CASH_FLOW = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
EQUITY = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
ASSETS = ["Assets"]
LIABILITIES = ["Liabilities"]
SHARE_SOURCES = [
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("dei", "EntityCommonStockSharesOutstanding"),
]
QUARTER_DAYS = 91.31


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "end": pd.Series(dtype="datetime64[ns]"),
            "val": pd.Series(dtype=float),
            "filed": pd.Series(dtype="datetime64[ns]"),
        }
    )


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["val"] = frame["val"].astype(float)
    return frame


def company_facts(cik: int) -> dict:
    # Raw payloads are 1-10 MB each; callers cache the extracted series instead.
    return get_json(FACTS_URL.format(cik=int(cik)), cache=False)


def _facts(payload: dict, concepts: list[str], taxonomy: str = "us-gaap") -> pd.DataFrame:
    """Stack facts for the given concepts; earlier concepts win when periods overlap."""

    frames = []
    pool = payload.get("facts", {}).get(taxonomy, {})
    for rank, concept in enumerate(concepts):
        units = pool.get(concept, {}).get("units", {})
        values = units.get("USD") or units.get("shares") or []
        if not values:
            continue
        frame = pd.DataFrame(values)
        frame = frame[frame["form"].isin(FORMS)]
        if frame.empty:
            continue
        frame["rank"] = rank
        frames.append(frame)
    columns = ["start", "end", "val", "filed", "rank"]
    if not frames:
        return pd.DataFrame(columns=columns)
    data = pd.concat(frames, ignore_index=True)
    if "start" not in data:
        data["start"] = pd.NaT
    data["start"] = pd.to_datetime(data["start"])
    data["end"] = pd.to_datetime(data["end"])
    data["filed"] = pd.to_datetime(data["filed"])
    data = data.dropna(subset=["end", "val", "filed"])
    # First availability per period; concept priority breaks ties on the same filing date.
    data = data.sort_values(["filed", "rank"])
    keys = ["start", "end"] if data["start"].notna().any() else ["end"]
    data = data.drop_duplicates(subset=keys, keep="first")
    return data[columns].sort_values("end").reset_index(drop=True)


def instant(payload: dict, concepts: list[str]) -> pd.DataFrame:
    """Balance-sheet item: one row per period end -> (end, val, filed)."""

    data = _facts(payload, concepts)
    if data.empty:
        return _empty()
    data = data.sort_values("filed").drop_duplicates("end", keep="first")
    return data[["end", "val", "filed"]].sort_values("end").reset_index(drop=True)


def trailing_twelve_months(payload: dict, concepts: list[str]) -> pd.DataFrame:
    """Flow item summed over the last four fiscal quarters -> (end, val, filed).

    Discrete quarters come from 3-month facts when reported, otherwise from
    differences of year-to-date facts sharing the same fiscal-year start
    (e.g. Q4 = FY - 9M, cash-flow Q2 = 6M - 3M). The ``filed`` date of a TTM
    value is the latest filing among its components.
    """

    data = _facts(payload, concepts)
    data = data.dropna(subset=["start"])
    if data.empty:
        return _empty()
    # Count fiscal quarters per fact. Tolerant enough for 52/53-week retail calendars
    # (12- and 16-week quarters, 24/28/36/40-week year-to-date periods).
    data["days"] = (data["end"] - data["start"]).dt.days
    data["nq"] = np.rint(data["days"] / QUARTER_DAYS).astype(int)
    fits = (data["days"] - data["nq"] * QUARTER_DAYS).abs() <= 25
    data = data[data["nq"].between(1, 4) & fits]

    quarters: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    for row in data[data["nq"] == 1].itertuples():
        quarters[row.end] = (float(row.val), row.filed)
    ytd = {(row.start, row.nq): row for row in data[data["nq"] > 1].itertuples()}
    for (start, nq), row in ytd.items():
        if row.end in quarters:
            continue
        previous = ytd.get((start, nq - 1))
        if previous is None and nq == 2:
            q1 = data[(data["start"] == start) & (data["nq"] == 1)]
            previous = next(q1.itertuples(), None)
        if previous is None:
            continue
        quarters[row.end] = (float(row.val) - float(previous.val), max(row.filed, previous.filed))

    if len(quarters) < 4:
        return _empty()
    q = pd.DataFrame(
        [(end, val, filed) for end, (val, filed) in quarters.items()],
        columns=["end", "val", "filed"],
    ).sort_values("end", ignore_index=True)

    annual = {row.end: row for row in data[data["nq"] == 4].itertuples()}
    rows = []
    for i in range(3, len(q)):
        window = q.iloc[i - 3 : i + 1]
        gaps = window["end"].diff().dt.days.dropna()
        span = (window["end"].iloc[-1] - window["end"].iloc[0]).days
        if not (gaps.between(56, 126).all() and 230 <= span <= 320):
            continue
        end = window["end"].iloc[-1]
        fy = annual.get(end)
        if fy is not None:  # an audited annual figure is more reliable than a sum
            rows.append((end, float(fy.val), max(fy.filed, window["filed"].iloc[-1])))
        else:
            rows.append((end, float(window["val"].sum()), window["filed"].max()))
    return pd.DataFrame(rows, columns=["end", "val", "filed"])


def shares(payload: dict) -> pd.DataFrame:
    """Share count per filing -> (end, val, filed).

    Priority: diluted weighted-average shares (covers every share class), basic
    weighted-average, balance-sheet shares outstanding, then cover-page shares
    (dei). One value per filing date, from the best concept that filing reports.
    """

    frames = []
    for rank, (taxonomy, concept) in enumerate(SHARE_SOURCES):
        data = _facts(payload, [concept], taxonomy=taxonomy)
        if data.empty:
            continue
        data["days"] = (data["end"] - data["start"]).dt.days.fillna(0)
        data["rank"] = rank
        frames.append(data)
    if not frames:
        return _empty()
    data = pd.concat(frames, ignore_index=True)
    data = data[data["val"] >= 1e6]  # drop placeholder values (e.g. a cover page reporting 1)
    data = data.sort_values(["filed", "rank", "end", "days"], ascending=[True, True, False, True])
    data = data.drop_duplicates("filed", keep="first")
    return data[["end", "val", "filed"]].sort_values("filed").reset_index(drop=True)


def fundamentals(cik: int) -> dict[str, pd.DataFrame]:
    """All series needed by the equity factor pipeline, each as (end, val, filed)."""

    payload = company_facts(cik)
    revenue = trailing_twelve_months(payload, REVENUE)
    gross = trailing_twelve_months(payload, GROSS_PROFIT)
    # Many filers tag GrossProfit only in some years; fill the other periods with
    # revenue - cost of revenue (reported GrossProfit wins where both exist).
    cost = trailing_twelve_months(payload, COST_OF_REVENUE)
    if not revenue.empty and not cost.empty:
        merged = revenue.merge(cost, on="end", suffixes=("_rev", "_cost"))
        derived = pd.DataFrame(
            {
                "end": merged["end"],
                "val": merged["val_rev"] - merged["val_cost"],
                "filed": merged[["filed_rev", "filed_cost"]].max(axis=1),
            }
        )
        derived = derived[~derived["end"].isin(gross["end"])]
        gross = pd.concat([gross, derived], ignore_index=True).sort_values("end", ignore_index=True)
    series = {
        "revenue_ttm": revenue,
        "gross_profit_ttm": gross,
        "net_income_ttm": trailing_twelve_months(payload, NET_INCOME),
        "operating_cash_flow_ttm": trailing_twelve_months(payload, OPERATING_CASH_FLOW),
        "total_equity": instant(payload, EQUITY),
        "total_assets": instant(payload, ASSETS),
        "total_liabilities": instant(payload, LIABILITIES),
        "diluted_shares": shares(payload),
    }
    return {name: _typed(frame) for name, frame in series.items()}
