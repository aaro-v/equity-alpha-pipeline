from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SECTORS: tuple[str, ...] = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Utilities",
)


@dataclass(frozen=True)
class SampleDataConfig:
    """Configuration for the deterministic synthetic equity panel."""

    n_assets: int = 90
    n_days: int = 640
    start: str = "2018-01-02"
    seed: int = 7


def generate_sample_panel(config: SampleDataConfig | None = None) -> pd.DataFrame:
    """Generate a deterministic point-in-time cross-sectional equity sample.

    The data is synthetic by design. It has enough structure to exercise a
    research pipeline without implying live-market profitability.
    """

    config = config or SampleDataConfig()
    rng = np.random.default_rng(config.seed)

    dates = pd.bdate_range(config.start, periods=config.n_days, name="date")
    assets = np.array([f"EQ{i:03d}" for i in range(config.n_assets)])
    sectors = np.array([SECTORS[i % len(SECTORS)] for i in range(config.n_assets)])

    market_caps = np.exp(rng.normal(9.8, 0.9, config.n_assets))
    base_liquidity = np.exp(rng.normal(15.2, 0.7, config.n_assets))
    beta = rng.normal(1.0, 0.25, config.n_assets)
    quality = rng.normal(0.0, 1.0, config.n_assets)
    value = rng.normal(0.0, 1.0, config.n_assets)
    idio_vol = rng.uniform(0.009, 0.025, config.n_assets)

    market_return = rng.normal(0.00025, 0.009, len(dates))
    sector_shocks = rng.normal(0.0, 0.004, (len(dates), len(SECTORS)))
    latent_alpha = 0.00035 * quality + 0.00025 * value - 0.00012 * np.log(market_caps)

    rows: list[pd.DataFrame] = []
    prices = np.full(config.n_assets, 50.0) * np.exp(rng.normal(0.0, 0.25, config.n_assets))
    shares = market_caps / prices

    for t, date in enumerate(dates):
        sector_idx = np.array([SECTORS.index(sector) for sector in sectors])
        shock = sector_shocks[t, sector_idx]
        noise = rng.normal(0.0, idio_vol)
        daily_return = beta * market_return[t] + shock + latent_alpha + noise
        prices = np.maximum(1.0, prices * (1.0 + daily_return))
        market_cap = shares * prices

        drift = np.sin(t / 45.0) * 0.05
        book_to_price = np.exp(-0.2 + 0.45 * value + drift + rng.normal(0.0, 0.03, config.n_assets))
        gross_margin = np.clip(0.35 + 0.08 * quality + rng.normal(0.0, 0.02, config.n_assets), 0.05, 0.9)
        accruals = np.clip(0.08 - 0.04 * quality + rng.normal(0.0, 0.015, config.n_assets), -0.2, 0.4)
        debt_to_equity = np.exp(0.1 - 0.2 * quality + rng.normal(0.0, 0.08, config.n_assets))
        dollar_volume = base_liquidity * np.exp(rng.normal(0.0, 0.25, config.n_assets))

        # Synthetic fundamentals are assumed to have become available five
        # business days earlier, making the point-in-time filter explicit.
        available_idx = max(0, t - 5)
        frame = pd.DataFrame(
            {
                "date": date,
                "asset": assets,
                "sector": sectors,
                "price": prices,
                "daily_return": daily_return,
                "benchmark_return": market_return[t],
                "market_cap": market_cap,
                "dollar_volume": dollar_volume,
                "book_to_price": book_to_price,
                "gross_margin": gross_margin,
                "accruals": accruals,
                "debt_to_equity": debt_to_equity,
                "fundamentals_available_date": dates[available_idx],
            }
        )
        rows.append(frame)

    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["date", "asset"], ignore_index=True)
