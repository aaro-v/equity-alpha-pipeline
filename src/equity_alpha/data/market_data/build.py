"""Real-data panel for the cross-sectional equity pipeline.

Universe      current S&P 500 constituents excluding Financials and Real Estate
              (gross margin and accruals are not meaningful for banks and REITs)
Prices        Yahoo Finance daily bars; ``price`` is the split- and
              dividend-adjusted close
Sectors       GICS sector from the Wikipedia S&P 500 constituents table
Fundamentals  SEC EDGAR XBRL filings; first-filed values only, usable from the
              business day after the filing date

Outputs, written to ``--out`` (default ``data/real``):

    panel.parquet                  columns expected by build_research_frame
    prices_daily.parquet           date, asset, close, adj_close, volume
    fundamentals_quarterly.parquet asset, field, period_end, value, filed
    universe.csv                   identifiers, sector and coverage per company
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.tseries.offsets import BDay

from . import sec, yahoo
from .http import CACHE_DIR, get

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
EXCLUDED_SECTORS = {"Financials", "Real Estate"}
BENCHMARK = "SPY"
PRICE_START = "2011-01-01"
PANEL_START = "2012-01-01"
MAX_FUNDAMENTAL_AGE_DAYS = 450  # stop using a filing if no newer one arrives
DEBT_TO_EQUITY_CAP = 50.0

PANEL_COLUMNS = [
    "date",
    "asset",
    "sector",
    "price",
    "daily_return",
    "benchmark_return",
    "market_cap",
    "dollar_volume",
    "book_to_price",
    "gross_margin",
    "accruals",
    "debt_to_equity",
    "fundamentals_available_date",
]


def load_universe() -> pd.DataFrame:
    html = get(WIKI_URL, headers={"User-Agent": "real-datasets-builder/1.0"}).decode()
    table = pd.read_html(io.StringIO(html))[0]
    table = table.rename(
        columns={
            "Symbol": "asset",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "CIK": "cik",
            "Date added": "date_added",
        }
    )
    table = table[~table["sector"].isin(EXCLUDED_SECTORS)].copy()
    table["yahoo_symbol"] = table["asset"].str.replace(".", "-", regex=False)
    table["cik"] = table["cik"].astype(int)
    return table[["asset", "yahoo_symbol", "name", "sector", "sub_industry", "cik", "date_added"]]


def cached_fundamentals(cik: int) -> pd.DataFrame:
    path = CACHE_DIR / "sec" / f"{cik:010d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for field, frame in sec.fundamentals(cik).items():
        if not frame.empty:
            frames.append(frame.assign(field=field))
    data = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["end", "val", "filed", "field"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)
    return data


def _asof(daily_dates: pd.Series, series: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Latest value available on each date, with availability = next business day after filing."""

    if series.empty:
        nan = pd.Series(np.nan, index=daily_dates.index)
        return nan, pd.Series(pd.NaT, index=daily_dates.index)
    s = series.copy()
    s["filed"] = pd.to_datetime(s["filed"])
    s["end"] = pd.to_datetime(s["end"])
    s["available"] = s["filed"] + BDay(1)
    # If several periods become available on the same day, keep the most recent period end.
    s = s.sort_values(["available", "end"]).drop_duplicates("available", keep="last")
    s = s.sort_values("available")
    s["available"] = s["available"].astype("datetime64[ns]")
    left = pd.DataFrame(
        {"date": pd.to_datetime(daily_dates).astype("datetime64[ns]").to_numpy()},
        index=daily_dates.index,
    )
    merged = pd.merge_asof(
        left.sort_values("date"),
        s[["available", "val"]],
        left_on="date",
        right_on="available",
        direction="backward",
    )
    merged.index = left.sort_values("date").index
    merged = merged.reindex(left.index)
    stale = (merged["date"] - merged["available"]).dt.days > MAX_FUNDAMENTAL_AGE_DAYS
    merged.loc[stale, ["val", "available"]] = [np.nan, pd.NaT]
    return merged["val"], merged["available"]


def company_panel(
    meta: pd.Series,
    history: yahoo.YahooHistory,
    fundamentals: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    bars = history.bars.loc[PRICE_START:]
    frame = pd.DataFrame(index=bars.index)
    frame["price"] = bars["adj_close"]
    frame["daily_return"] = bars["adj_close"].pct_change()
    frame["benchmark_return"] = benchmark_returns.reindex(frame.index)
    frame["dollar_volume"] = (bars["close"] * bars["volume"]).rolling(21, min_periods=10).mean()
    frame = frame.loc[PANEL_START:].reset_index()
    dates = frame["date"]

    field = dict(iter(fundamentals.groupby("field")))
    get_field = lambda name: field.get(name, pd.DataFrame(columns=["end", "val", "filed"]))  # noqa: E731

    shares = get_field("diluted_shares").copy()
    if not shares.empty:  # express share counts on today's split basis (prices are split-adjusted)
        shares["val"] = shares["val"] * yahoo.split_factor_after(
            history.splits, pd.DatetimeIndex(shares["filed"])
        )
    values, available = {}, {}
    for name in [
        "revenue_ttm",
        "gross_profit_ttm",
        "net_income_ttm",
        "operating_cash_flow_ttm",
        "total_equity",
        "total_assets",
        "total_liabilities",
    ]:
        values[name], available[name] = _asof(dates, get_field(name))
    values["diluted_shares"], available["diluted_shares"] = _asof(dates, shares)

    close = bars["close"].reindex(dates).to_numpy()
    market_cap = close * values["diluted_shares"].to_numpy()
    equity = values["total_equity"]
    liabilities = values["total_liabilities"].fillna(values["total_assets"] - equity)
    debt_to_equity = (liabilities / equity).where(equity > 0, DEBT_TO_EQUITY_CAP)
    frame["market_cap"] = market_cap
    frame["book_to_price"] = equity / market_cap
    gross_margin = values["gross_profit_ttm"] / values["revenue_ttm"]
    # Gross profit cannot exceed revenue; values outside [-1, 1] come from mismatched tags.
    frame["gross_margin"] = gross_margin.where(gross_margin.between(-1.0, 1.0))
    frame["accruals"] = (values["net_income_ttm"] - values["operating_cash_flow_ttm"]) / values[
        "total_assets"
    ]
    frame["debt_to_equity"] = debt_to_equity.clip(0.0, DEBT_TO_EQUITY_CAP)
    frame["fundamentals_available_date"] = pd.concat(available, axis=1).max(axis=1)
    frame["asset"] = meta["asset"]
    frame["sector"] = meta["sector"]
    return frame[PANEL_COLUMNS]


def build(out: Path, limit: int | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    if limit:
        universe = universe.head(limit)
    benchmark = yahoo.history(BENCHMARK, start=PRICE_START)
    benchmark_returns = benchmark.bars["adj_close"].pct_change()

    panels, prices, fundamentals_rows, coverage = [], [], [], []
    for i, meta in enumerate(universe.itertuples(index=False), start=1):
        meta = pd.Series(meta._asdict())
        try:
            history = yahoo.history(meta["yahoo_symbol"], start=PRICE_START)
            fundamentals = cached_fundamentals(int(meta["cik"]))
        except Exception as exc:  # keep going; report the gap in universe.csv
            print(f"[{i}/{len(universe)}] {meta['asset']}: skipped ({exc})", file=sys.stderr)
            coverage.append({"asset": meta["asset"], "status": f"skipped: {exc}"})
            continue
        panel = company_panel(meta, history, fundamentals, benchmark_returns)
        panels.append(panel)
        bars = history.bars.loc[PRICE_START:, ["close", "adj_close", "volume"]].reset_index()
        prices.append(bars.assign(asset=meta["asset"]))
        fundamentals_rows.append(fundamentals.assign(asset=meta["asset"]))
        usable = panel.dropna(subset=["book_to_price", "gross_margin", "accruals", "market_cap"])
        coverage.append(
            {
                "asset": meta["asset"],
                "status": "ok",
                "first_price_date": bars["date"].min().date(),
                "panel_rows": len(panel),
                "rows_with_all_fundamentals": len(usable),
                "first_full_fundamentals_date": usable["date"].min().date()
                if len(usable)
                else None,
            }
        )
        if i % 25 == 0:
            print(f"[{i}/{len(universe)}] processed", file=sys.stderr)

    universe = universe.merge(pd.DataFrame(coverage), on="asset", how="left")
    universe.to_csv(out / "universe.csv", index=False)

    price_table = pd.concat(prices, ignore_index=True)[
        ["date", "asset", "close", "adj_close", "volume"]
    ]
    price_table["asset"] = price_table["asset"].astype("category")
    price_table["date"] = price_table["date"].astype("datetime64[us]")
    price_table = price_table.astype(
        {"close": "float32", "adj_close": "float32", "volume": "float64"}
    )
    price_table.to_parquet(out / "prices_daily.parquet", index=False, compression="zstd")

    fundamentals_table = pd.concat(fundamentals_rows, ignore_index=True).rename(
        columns={"end": "period_end", "val": "value"}
    )[["asset", "field", "period_end", "value", "filed"]]
    fundamentals_table["asset"] = fundamentals_table["asset"].astype(
        pd.ArrowDtype(pa.large_string())
    )
    fundamentals_table["field"] = fundamentals_table["field"].astype(object)
    fundamentals_table[["period_end", "filed"]] = fundamentals_table[
        ["period_end", "filed"]
    ].astype("datetime64[us]")
    fundamentals_table.to_parquet(
        out / "fundamentals_quarterly.parquet", index=False, compression="zstd"
    )

    panel = pd.concat(panels, ignore_index=True).sort_values(["date", "asset"], ignore_index=True)
    float_cols = [
        c
        for c in PANEL_COLUMNS
        if c not in {"date", "asset", "sector", "fundamentals_available_date"}
    ]
    panel[float_cols] = panel[float_cols].astype("float32")
    panel["asset"] = panel["asset"].astype("category")
    panel["sector"] = panel["sector"].astype("category")
    panel["date"] = panel["date"].astype("datetime64[us]")
    panel["fundamentals_available_date"] = panel["fundamentals_available_date"].astype(
        "datetime64[ns]"
    )
    panel.to_parquet(out / "panel.parquet", index=False, compression="zstd")
    ok = universe["status"].eq("ok").sum()
    print(f"equity panel: {ok}/{len(universe)} companies, {len(panel):,} rows -> {out}")
