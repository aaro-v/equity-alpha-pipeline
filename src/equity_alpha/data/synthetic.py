from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

SECTORS = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Utilities",
)
FIELDS = (
    "revenue_ttm",
    "gross_profit_ttm",
    "net_income_ttm",
    "operating_cash_flow_ttm",
    "total_equity",
    "total_assets",
    "total_liabilities",
    "diluted_shares",
)


@dataclass(frozen=True)
class SampleDataConfig:
    n_assets: int = 100
    n_days: int = 756
    start: str = "2018-01-02"
    seed: int = 7


def generate_sample_tables(config: SampleDataConfig | None = None) -> dict[str, pd.DataFrame]:
    """Market/sector Student-t process, persistent quarterly filings and weak latent alpha."""
    config = config or SampleDataConfig()
    if config.n_assets <= 0 or config.n_days <= 0:
        raise ValueError("n_assets and n_days must be positive")
    rng = np.random.default_rng(config.seed)
    n, t = config.n_assets, config.n_days
    dates = pd.bdate_range(config.start, periods=t).astype("datetime64[us]")
    assets = np.array([f"EQ{i:03d}" for i in range(n)])
    sector_idx = np.arange(n) % len(SECTORS)
    sectors = np.array(SECTORS)[sector_idx]
    caps = 10 ** rng.normal(10.4, 0.5, n)
    liquidity = 10 ** rng.normal(8.29, 0.47, n)
    beta = np.clip(rng.normal(1.0, 0.36, n), 0.15, 2.2)
    idio_vol = np.exp(rng.normal(np.log(0.0145), 0.32, n))
    value, quality = rng.normal(size=(2, n))
    market = 0.00045 + rng.standard_t(5, t) * np.sqrt(3 / 5) * 0.0104
    sector = rng.standard_t(5, (t, len(SECTORS))) * np.sqrt(3 / 5) * 0.006
    noise = rng.standard_t(5, (t, n)) * np.sqrt(3 / 5) * idio_vol
    alpha = 0.00006 * quality + 0.000025 * value
    returns = np.clip(market[:, None] * beta + sector[:, sector_idx] + noise + alpha, -0.85, 1.5)
    initial = 50 * np.exp(rng.normal(0, 0.35, n))
    prices = initial * np.cumprod(1 + returns, axis=0)
    shares = caps / initial
    distressed = rng.random(n) < 0.12
    dollar_volume = liquidity * np.exp(rng.normal(0, 0.15, (t, n)))
    frames, filings, universe = [], [], []
    for i, asset in enumerate(assets):
        # Include a prior quarter so short test samples also start with available filings.
        periods = pd.date_range(dates[0] - pd.Timedelta(days=180), dates[-1], freq="QE")
        q = len(periods)
        lag = np.clip(rng.normal(35, 15, q).astype(int), 20, 75)
        filed = periods + pd.to_timedelta(lag, unit="D")
        available = (filed + pd.offsets.BDay(1)).astype("datetime64[ns]")
        persistent_value = value[i] + np.cumsum(rng.normal(0, 0.06, q))
        persistent_quality = quality[i] + np.cumsum(rng.normal(0, 0.04, q))
        equity = caps[i] * np.exp(-1.47 + 0.85 * persistent_value)
        margin = np.clip(0.438 + 0.22 * persistent_quality, 0.03, 0.95)
        transport = sectors[i] == "Industrials" and (i // len(SECTORS)) % 3 != 2
        if sectors[i] in {"Utilities", "Energy"} or transport:
            margin[:] = np.nan
        debt = np.minimum(50, np.exp(0.57 - 0.9 * quality[i] + rng.normal(0, 0.08, q)))
        if distressed[i]:
            debt[:] = 50.0
        total_assets = equity * (1 + debt)
        revenue = caps[i] * 0.65 * np.exp(rng.normal(0, 0.03, q))
        accrual = -0.041 + 0.045 * rng.normal() + rng.normal(0, 0.018, q)
        accrual[rng.random(q) < 0.057] = np.nan
        reported_shares = np.full(q, shares[i])
        reported_shares[rng.random(q) < 0.046] = np.nan
        income = revenue * 0.12
        fields = {
            "revenue_ttm": revenue,
            "gross_profit_ttm": revenue * margin,
            "net_income_ttm": income,
            "operating_cash_flow_ttm": income - accrual * total_assets,
            "total_equity": equity,
            "total_assets": total_assets,
            "total_liabilities": total_assets - equity,
            "diluted_shares": reported_shares,
        }
        for field, values in fields.items():
            filings.append(
                pd.DataFrame(
                    {
                        "asset": asset,
                        "field": field,
                        "period_end": periods,
                        "value": values,
                        "filed": filed,
                    }
                )
            )
        f = pd.DataFrame(
            {
                "available": available,
                "equity": equity,
                "gross_margin": margin,
                "accruals": accrual,
                "debt_to_equity": debt,
                "shares": reported_shares,
            }
        ).sort_values("available")
        daily = pd.merge_asof(
            pd.DataFrame({"date": dates.astype("datetime64[ns]")}),
            f,
            left_on="date",
            right_on="available",
        )
        frame = pd.DataFrame(
            {
                "date": dates,
                "asset": asset,
                "sector": sectors[i],
                "price": prices[:, i],
                "daily_return": returns[:, i],
                "benchmark_return": market,
                "market_cap": daily["shares"].to_numpy() * prices[:, i],
                "dollar_volume": dollar_volume[:, i],
                "book_to_price": daily["equity"].to_numpy()
                / (daily["shares"].to_numpy() * prices[:, i]),
                "gross_margin": daily["gross_margin"],
                "accruals": daily["accruals"],
                "debt_to_equity": daily["debt_to_equity"],
                "fundamentals_available_date": daily["available"].astype("datetime64[ns]"),
            }
        )
        frames.append(frame)
        full = frame.dropna(subset=["book_to_price", "gross_margin", "accruals", "market_cap"])
        universe.append(
            {
                "asset": asset,
                "yahoo_symbol": asset,
                "name": f"Synthetic company {i}",
                "sector": sectors[i],
                "sub_industry": "Transportation" if transport else "Synthetic",
                "cik": i + 1,
                "date_added": str(dates[0].date()),
                "status": "ok",
                "first_price_date": str(dates[0].date()),
                "panel_rows": t,
                "rows_with_all_fundamentals": len(full),
                "first_full_fundamentals_date": str(full["date"].min().date())
                if len(full)
                else None,
            }
        )
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "asset"], ignore_index=True)
    floats = panel.select_dtypes(include="number").columns
    panel[floats] = panel[floats].astype("float32")
    panel["asset"] = panel["asset"].astype("category")
    panel["sector"] = panel["sector"].astype("category")
    panel["date"] = panel["date"].astype("datetime64[us]")
    price_table = panel[["date", "asset", "price"]].rename(columns={"price": "adj_close"}).copy()
    price_table["close"] = price_table["adj_close"]
    price_table["volume"] = (panel["dollar_volume"] / price_table["close"]).astype("float64")
    fundamentals = pd.concat(filings, ignore_index=True)
    fundamentals["asset"] = fundamentals["asset"].astype(pd.ArrowDtype(pa.large_string()))
    fundamentals["field"] = fundamentals["field"].astype(object)
    fundamentals[["period_end", "filed"]] = fundamentals[["period_end", "filed"]].astype(
        "datetime64[us]"
    )
    return {
        "panel.parquet": panel,
        "prices_daily.parquet": price_table[["date", "asset", "close", "adj_close", "volume"]],
        "fundamentals_quarterly.parquet": fundamentals,
        "universe.csv": pd.DataFrame(universe),
    }


def generate_sample_panel(config: SampleDataConfig | None = None) -> pd.DataFrame:
    return generate_sample_tables(config)["panel.parquet"]


def write_sample_data(out: Path, config: SampleDataConfig | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for filename, frame in generate_sample_tables(config).items():
        if filename.endswith("csv"):
            frame.to_csv(out / filename, index=False)
        else:
            frame.to_parquet(out / filename, index=False, compression="zstd")
