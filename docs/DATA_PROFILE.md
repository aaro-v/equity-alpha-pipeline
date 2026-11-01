# Equity panel data profile

The real panel contains 397 companies, 3,729
trading dates and 1,391,906 company-date observations, from
2012-01-03 through 2026-10-30. Full-panel distributions below use that
entire period. Their source is `reports/real/data_profile.json`.

## Returns

Volatility is each company's sample standard deviation of daily total returns,
annualized by sqrt(252). Beta is estimated against SPY over each company's
available observations. The table reports cross-sectional quantiles of those
company-level statistics.

| Statistic | p10 | p25 | p50 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Annualized stock volatility | 0.220 | 0.254 | 0.298 | 0.382 | 0.466 |
| Beta to SPY | 0.554 | 0.770 | 0.985 | 1.184 | 1.449 |

Mean pairwise correlation is 0.281, with
0.379 within sectors and
0.266 between sectors. Pooled daily-return
excess kurtosis is 62.41;
2.90% of company-date returns exceed 5% in
absolute value. Pairwise correlations exclude self-correlations.

## Fundamentals

These distributions pool observations at the last trading date of each calendar
month. Ratios are reported in their original units; market capitalization and
dollar volume are scaled as indicated.

| Field | Missing share | p10 | p25 | p50 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| book_to_price | 4.62% | 0.037 | 0.108 | 0.231 | 0.412 | 0.621 |
| gross_margin | 29.36% | 0.175 | 0.300 | 0.438 | 0.622 | 0.759 |
| accruals | 5.65% | -0.103 | -0.068 | -0.041 | -0.020 | 0.003 |
| debt_to_equity | 0.00% | 0.569 | 0.981 | 1.762 | 3.483 | 50.000 |
| market_cap (USD billions) | 4.61% | 6.860 | 12.570 | 24.934 | 56.755 | 139.302 |
| dollar_volume (USD millions) | 0.04% | 49.371 | 101.816 | 196.179 | 399.296 | 809.149 |

## Factor results

Factor statistics use out-of-sample signals from 2014-08-13 through
2026-08-21, a subset of the panel period. They match
`reports/real/factor_ics.csv` and the results report. Rank IC uses the centered
neutralized factor ranks against the next 21 trading
days' excess return after the execution close, sampled on non-overlapping signal
dates. Execution follows the signal by 1 trading day(s).

| Factor | Mean rank IC | IC standard deviation | IC t-statistic | Observations |
| --- | ---: | ---: | ---: | ---: |
| value | -0.0108 | 0.1129 | -1.15 | 144 |
| momentum | 0.0050 | 0.1332 | 0.45 | 144 |
| quality | -0.0079 | 0.0756 | -1.26 | 144 |
| low_volatility | -0.0065 | 0.0907 | -0.86 | 144 |
| size | 0.0241 | 0.0922 | 3.13 | 144 |
| beta | 0.0184 | 0.1686 | 1.31 | 144 |
| liquidity | -0.0190 | 0.0868 | -2.63 | 144 |

The universe is selected from current constituents and therefore contains
survivorship bias. Beta is sector-neutralized; the other factors are neutralized
against both sector and beta.

Regenerate the panel summary and this document with:

```bash
equity-alpha profile-data --panel data/real/panel.parquet --out reports/real/data_profile.json
python scripts/write_data_profile.py
```

Factor statistics are produced by the real-panel research run described in the README.
