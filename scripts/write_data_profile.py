"""Generate the data profile from current panel summaries and out-of-sample factor results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
reports = root / "reports" / "real"
profile = json.loads((reports / "data_profile.json").read_text())
run = json.loads((reports / "run.json").read_text())
factors = pd.read_csv(reports / "factor_ics.csv")
factors = factors.loc[factors["variant"].eq("neutral_rank")]

return_lines = [
    "| Statistic | p10 | p25 | p50 | p75 | p90 |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for label, key in [("Annualized stock volatility", "annual_volatility"), ("Beta to SPY", "beta")]:
    return_lines.append("| " + label + " | " + " | ".join(f"{v:.3f}" for v in profile[key]) + " |")
returns_table = "\n".join(return_lines)
fundamental_lines = [
    "| Field | Missing share | p10 | p25 | p50 | p75 | p90 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for field, values in profile["fundamentals"].items():
    scale = 1e9 if field == "market_cap" else 1e6 if field == "dollar_volume" else 1
    label = field + (
        " (USD billions)" if scale == 1e9 else " (USD millions)" if scale == 1e6 else ""
    )
    quantiles = " | ".join(f"{v / scale:.3f}" for v in values["quantiles"])
    fundamental_lines.append(f"| {label} | {values['missing_share']:.2%} | {quantiles} |")
fundamentals_table = "\n".join(fundamental_lines)
factor_lines = [
    "| Factor | Mean rank IC | IC standard deviation | IC t-statistic | Observations |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for row in factors.itertuples():
    factor_lines.append(
        f"| {row.factor} | {row.mean_rank_ic:.4f} | {row.rank_ic_std:.4f} "
        f"| {row.rank_ic_t_stat:.2f} | {row.observations} |"
    )
factor_table = "\n".join(factor_lines)
text = f"""# Equity panel data profile

The real panel contains {profile["assets"]} companies, {profile["trading_days"]:,}
trading dates and {profile["rows"]:,} company-date observations, from
{profile["start"]} through {profile["end"]}. Full-panel distributions below use that
entire period. Their source is `reports/real/data_profile.json`.

## Returns

Volatility is each company's sample standard deviation of daily total returns,
annualized by sqrt(252). Beta is estimated against SPY over each company's
available observations. The table reports cross-sectional quantiles of those
company-level statistics.

{returns_table}

Mean pairwise correlation is {profile["pairwise_correlation"]:.3f}, with
{profile["within_sector_correlation"]:.3f} within sectors and
{profile["between_sector_correlation"]:.3f} between sectors. Pooled daily-return
excess kurtosis is {profile["pooled_excess_kurtosis"]:.2f};
{profile["abs_daily_return_above_5pct"]:.2%} of company-date returns exceed 5% in
absolute value. Pairwise correlations exclude self-correlations.

## Fundamentals

These distributions pool observations at the last trading date of each calendar
month. Ratios are reported in their original units; market capitalization and
dollar volume are scaled as indicated.

{fundamentals_table}

## Factor results

Factor statistics use out-of-sample signals from {run["test_start"]} through
{run["test_end"]}, a subset of the panel period. They match
`reports/real/factor_ics.csv` and the results report. Rank IC uses the centered
neutralized factor ranks against the next {run["parameters"]["horizon"]} trading
days' excess return, sampled on non-overlapping signal dates.

{factor_table}

The universe is selected from current constituents and therefore contains
survivorship bias. Beta is sector-neutralized; the other factors are neutralized
against both sector and beta.

Regenerate the panel summary and this document with:

```bash
equity-alpha profile-data --panel data/real/panel.parquet --out reports/real/data_profile.json
python scripts/write_data_profile.py
```

Factor statistics are produced by the real-panel research run described in the README.
"""
(root / "docs" / "DATA_PROFILE.md").write_text(text, encoding="utf-8")
