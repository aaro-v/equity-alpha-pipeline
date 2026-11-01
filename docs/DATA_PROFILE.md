# Real data profile: equity panel

Summary statistics of `data/panel.parquet` and its companion files, computed
from the real data. Use them to calibrate the synthetic generator so that the
synthetic panel behaves like the real one, and as a sanity reference when
reading backtest output. Nothing here is raw vendor data.

The synthetic generator should reproduce, roughly: the volatility and beta
distributions; a one-factor-plus-sector correlation structure (about 0.30
overall, 0.40 within sector); fat-tailed daily returns; fundamentals that move
in steps on quarterly filing dates about 35 days after period end, with very
persistent cross-sectional ranks; missing gross margin for whole sectors
(utilities, transports, integrated oil); and weak signals. No feature has a
monthly rank IC above about 0.025 in absolute value on the real data, so a
synthetic signal much stronger than that will make every model look better
than it can be on real data.

Universe: 397 companies, 3729 trading days (2012-01-03 to 2026-10-30), 1,391,906 rows.

| GICS sector | companies |
|---|---|
| Industrials | 83 |
| Information Technology | 75 |
| Health Care | 60 |
| Consumer Discretionary | 47 |
| Consumer Staples | 34 |
| Utilities | 31 |
| Materials | 24 |
| Communication Services | 22 |
| Energy | 21 |

Daily returns (total return, adjusted close)

| statistic | value |
|---|---|
| benchmark (SPY) annualised return / vol | 0.152 / 0.165 |
| stock annualised vol, p10/p25/p50/p75/p90 | 0.22 / 0.254 / 0.298 / 0.382 / 0.466 |
| beta to SPY, p10/p25/p50/p75/p90 | 0.544 / 0.77 / 0.994 / 1.19 / 1.52 |
| mean pairwise correlation (2015 onward) | 0.301 |
| mean correlation, same sector / different sector | 0.406 / 0.285 |
| daily return kurtosis (pooled) / share of days with abs return > 5% | 62.4 / 0.0272 |

Cross-sectional fundamentals (month-end samples pooled over time)

| field | missing share | p10 / p25 / p50 / p75 / p90 |
|---|---|---|
| book_to_price | 0.046 | 0.037 / 0.109 / 0.231 / 0.413 / 0.622 |
| gross_margin | 0.294 | 0.175 / 0.3 / 0.438 / 0.622 / 0.759 |
| accruals | 0.057 | -0.103 / -0.0679 / -0.0411 / -0.0203 / 0.00269 |
| debt_to_equity | 0.000 | 0.569 / 0.98 / 1.76 / 3.48 / 50 |
| log10 market_cap (USD) | 0.046 | 9.83 / 10.1 / 10.4 / 10.8 / 11.1 |
| log10 dollar_volume (21d avg, USD) | 0.000 | 7.69 / 8 / 8.29 / 8.6 / 8.9 |

Filings: median lag from period end to filing 35 days (p90 59); median spacing between balance-sheet filings 91 days. Fundamentals change in steps on filing dates, not daily.
Rank persistence quarter to quarter (cross-sectional rank correlation, 63 trading days apart): book_to_price 0.98, gross_margin 0.99, accruals 0.85, debt_to_equity 0.98.

Monthly rank IC of each neutralised feature against the 21-day forward excess return (non-overlapping months, 2013 onward)

| feature | mean rank IC | IC std | t-stat | share positive |
|---|---|---|---|---|
| value | -0.0188 | 0.101 | -2.39 | 0.40 |
| momentum | +0.0105 | 0.138 | +0.98 | 0.50 |
| quality | -0.0063 | 0.091 | -0.88 | 0.49 |
| low_volatility | -0.0017 | 0.094 | -0.24 | 0.48 |
| size | +0.0231 | 0.100 | +2.95 | 0.60 |
| beta | +0.0000 | 0.223 | +0.00 | 0.51 |
| liquidity | -0.0172 | 0.093 | -2.37 | 0.44 |

The positive size IC is partly survivorship bias: the universe is today's
S&P 500, so small companies in the sample are the ones that went on to grow
into the index. Value had a weak decade in US large caps, so its negative IC
over 2013-2026 is expected, not a bug.
