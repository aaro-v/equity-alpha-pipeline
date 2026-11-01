"""Write interpretation and validation notes from saved aggregate outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
real = root / "reports" / "real"
synthetic = root / "reports" / "synthetic"
metrics = pd.read_csv(real / "metrics.csv").set_index("model")
factors = pd.read_csv(real / "factor_ics.csv")
factors = factors.loc[factors["variant"].eq("neutral_rank")].set_index("factor")
folds = pd.read_csv(real / "fold_summary.csv")
run = json.loads((real / "run.json").read_text())
rebuild = json.loads((real / "rebuild_check.json").read_text())
synth_metrics = pd.read_csv(synthetic / "metrics.csv")
profile = json.loads((synthetic / "data_profile.json").read_text())
ridge, composite = metrics.loc["ridge"], metrics.loc["composite"]
lines = ["| Factor | Monthly rank IC | Non-overlapping IC t-statistic |", "| --- | ---: | ---: |"]
for factor, row in factors.iterrows():
    lines.append(f"| {factor} | {row.mean_rank_ic:.4f} | {row.rank_ic_t_stat:.2f} |")
factor_table = "\n".join(lines)
text = f"""# Results and validation

The real experiment uses {len(folds)} rolling folds, with signal dates from
{run["test_start"]} through {run["test_end"]}. It covers {run["test_assets"]} distinct
eligible companies after the trailing-history requirement, from a panel of
{run["panel_assets"]} current constituents. Incomplete final test windows are omitted.
Each forecast has {int(ridge.ic_observations)} non-overlapping monthly IC observations.
The source commit and exact dependency versions are recorded in each run's `run.json`.

Ridge's realized daily net Sharpe is {ridge.cost_adjusted_sharpe:.2f}, versus
{composite.cost_adjusted_sharpe:.2f} for the equal-weight rank composite. Its annualized
net return is {ridge.annualized_return:.1%} and maximum drawdown is
{ridge.max_drawdown:.1%}. Its monthly rank IC is {ridge.mean_rank_ic:.4f}, with a
non-overlapping IC t-statistic of {ridge.rank_ic_t_stat:.2f}; this is limited evidence
of persistent forecasting ability. Boosting has a positive realized result and the
neural baseline has very little rank IC. Favorable realized portfolio statistics
alone do not resolve survivorship or execution assumptions.

## Factor evidence

{factor_table}

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

The live builder successfully downloaded {rebuild["successful_downloads"]} companies
with {rebuild["skipped_downloads"]} skipped downloads. It reproduced all
{rebuild["common_rows"]:,} supplied panel rows. Market cap, dollar volume and all
fundamental ratios match exactly on common observations. The largest adjusted-price
difference is {rebuild["columns"]["price"]["max_absolute_difference"]:.8f} dollars,
and the largest daily-return difference is
{rebuild["columns"]["daily_return"]["max_absolute_difference"]:.8f}; these reflect
small changes in adjusted Yahoo history. Missingness matches throughout.
The canonical three-company rebuild also matches the committed file schemas.

Reproduce the aggregate comparison, keeping both directories private:

```bash
python scripts/check_rebuild.py --reference data/real --rebuilt data/real/rebuilt \\
  --out reports/real/rebuild_check.json
```

## Synthetic check

The saved synthetic panel contains {profile["assets"]} assets over
{profile["trading_days"]:,} business days, with the same model hyperparameters and
walk-forward layout as the real run. Pairwise correlation is
{profile["pairwise_correlation"]:.2f} and within-sector correlation is
{profile["within_sector_correlation"]:.2f}; median annual volatility is
{profile["annual_volatility"][2]:.1%}. Gross margin is absent for
{profile["fundamentals"]["gross_margin"]["missing_share"]:.0%} of month-end rows.
Student-t tails are heavier than Gaussian tails but much less extreme than the
real panel's crisis and event tails. The industrial missing-margin rule was corrected
when distribution checks exposed an unreachable condition; model settings,
returns, random seeds and transaction costs were unchanged.

Synthetic forecast rank IC spans {synth_metrics.mean_rank_ic.min():.4f} to
{synth_metrics.mean_rank_ic.max():.4f}, and daily net Sharpe spans
{synth_metrics.cost_adjusted_sharpe.min():.2f} to {synth_metrics.cost_adjusted_sharpe.max():.2f}.
Values outside the real-data typical ranges are reported as generated, without
changing parameters to make the synthetic model results look favorable.

Regenerate this document with `python scripts/write_result_notes.py` after the
report commands in the README. All values above come from saved aggregate outputs.
"""
(root / "docs" / "RESULTS.md").write_text(text, encoding="utf-8")
