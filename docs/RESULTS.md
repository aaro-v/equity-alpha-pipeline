# Results and validation

The real experiment uses 48 rolling folds, with signal dates from
2014-08-13 through 2026-08-21. It covers 394 distinct
eligible companies after the trailing-history requirement, from a panel of
397 current constituents. Incomplete final test windows are omitted.
Each forecast has 144 non-overlapping monthly IC observations.
The source commit and exact dependency versions are recorded in each run's `run.json`.

Ridge's realized daily net Sharpe is 0.71, versus
0.03 for the equal-weight rank composite. Its annualized
net return is 9.1% and maximum drawdown is
-22.2%. Its monthly rank IC is 0.0156, with a
non-overlapping IC t-statistic of 1.45; this is limited evidence
of persistent forecasting ability. Boosting has a positive realized result and the
neural baseline has very little rank IC. Favorable realized portfolio statistics
alone do not resolve survivorship or execution assumptions.

## Factor evidence

| Factor | Monthly rank IC | Non-overlapping IC t-statistic |
| --- | ---: | ---: |
| value | -0.0118 | -1.24 |
| momentum | 0.0047 | 0.41 |
| quality | -0.0075 | -1.18 |
| low_volatility | -0.0072 | -0.92 |
| size | 0.0247 | 3.14 |
| beta | 0.0156 | 1.09 |
| liquidity | -0.0209 | -2.83 |

![Factor IC decay](../reports/real/factor_decay.png)

The positive size result is consistent with the known current-constituent
survivorship problem: historically smaller firms in this sample are firms that
survived to become constituents. Value, quality and liquidity have negative average
neutralized IC; combining all predefined factor directions equally can cancel
weak effects. Learning their signs within historical training windows can explain
some improvement over that benchmark without requiring a strong nonlinear signal.

The expected small-factor range is appropriate for the neutralized rows above.
Raw low-volatility and size rank IC in `factor_ics.csv` fall outside that range;
those raw factors retain sector and market exposures. Beta is deliberately only
sector-neutralized and is not comparable with the numerically degenerate result
from regressing beta on itself. No forecast mean rank IC exceeds the suspicious
large-signal sanity threshold, and model Sharpe, drawdown and annual one-way
turnover all fall within its typical ranges. Daily overlapping IC is used only for
the descriptive annual plot. Non-overlapping samples remove mechanical target
intersection; their simple t-statistics do not account for every source of serial
dependence or multiple testing.

## Accounting and availability

The former evaluator booked a new forward holding-period return on every date.
The current evaluator books actual daily total returns of the held shares, charges
costs against drifted holdings, and liquidates at the end. The hand-calculated test
covers compounding, rebalancing drift, entry costs and exit costs. Sharpe is
annualized from daily observations.

Benchmark labels compound the next trading days and exclude the signal day.
Momentum lags remain within the asset and common trading calendar. Future-data
perturbation, date-wise shuffled labels, purge boundaries, first-filed SEC values,
filing availability and vectorized neutralization have behavioral tests. Missing
quality is imputed after neutralization rather than eliminating whole sectors.
Selection from current constituents and missing future endpoints still limit the
historical universe; these checks do not make the source data survivorship-free.

## Data rebuild

The live builder successfully downloaded 397 companies
with 0 skipped downloads. It reproduced all
1,391,906 supplied panel rows. Market cap, dollar volume and all
fundamental ratios match exactly on common observations. The largest adjusted-price
difference is 0.00048828 dollars,
and the largest daily-return difference is
0.00000185; these reflect
small changes in adjusted Yahoo history. Missingness matches throughout.
The canonical three-company rebuild also matches the committed file schemas.

Reproduce the aggregate comparison, keeping both directories private:

```bash
python scripts/check_rebuild.py --reference data/real --rebuilt data/real/rebuilt \
  --out reports/real/rebuild_check.json
```

## Synthetic check

The saved synthetic panel contains 100 assets over
2,016 business days, with the same model hyperparameters and
walk-forward layout as the real run. Pairwise correlation is
0.26 and within-sector correlation is
0.36; median annual volatility is
29.6%. Gross margin is absent for
30% of month-end rows.
Student-t tails are heavier than Gaussian tails but much less extreme than the
real panel's crisis and event tails. The industrial missing-margin rule was corrected
when distribution checks exposed an unreachable condition; model settings,
returns, random seeds and transaction costs were unchanged.

Synthetic forecast rank IC spans -0.0230 to
0.0171, and daily net Sharpe spans
-1.15 to 0.42.
Values outside the real-data typical ranges are reported as generated, without
changing parameters to make the synthetic model results look favorable.

Regenerate this document with `python scripts/write_result_notes.py` after the
report commands in the README. All values above come from saved aggregate outputs.
