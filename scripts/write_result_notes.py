"""Write result notes from the saved experiment tables."""

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
real_profile = json.loads((real / "data_profile.json").read_text())
names = {
    "composite": "Equal-weight composite",
    "ridge": "Ridge",
    "tree": "Gradient boosting",
    "neural": "Neural network",
}
best_model = metrics["cost_adjusted_sharpe"].idxmax()
best = metrics.loc[best_model]
evidence = (
    "The monthly IC t-statistics provide limited evidence of a stable forecasting effect."
    if metrics["rank_ic_t_stat"].abs().max() < 2
    else "The IC estimates remain subject to serial dependence and multiple comparisons."
)
ic_lines = [
    "| Forecast | Monthly rank IC | IC t-statistic | Observations |",
    "| --- | ---: | ---: | ---: |",
]
for model, row in metrics.iterrows():
    ic_lines.append(
        f"| {names[model]} | {row.mean_rank_ic:.4f} | {row.rank_ic_t_stat:.2f} "
        f"| {int(row.ic_observations)} |"
    )
ic_table = "\n".join(ic_lines)
lines = ["| Factor | Monthly rank IC | IC t-statistic |", "| --- | ---: | ---: |"]
for factor, row in factors.iterrows():
    lines.append(f"| {factor} | {row.mean_rank_ic:.4f} | {row.rank_ic_t_stat:.2f} |")
factor_table = "\n".join(lines)
rebuild_lines = [
    "| Field | Maximum absolute difference | Missingness disagreements |",
    "| --- | ---: | ---: |",
]
for field, comparison in rebuild["columns"].items():
    difference = comparison["max_absolute_difference"]
    difference_text = f"{difference:.8g}" if difference is not None else "n/a"
    rebuild_lines.append(
        f"| {field} | {difference_text} | {comparison['missingness_disagreements']} |"
    )
rebuild_table = "\n".join(rebuild_lines)
text = f"""# Results and validation

The real experiment uses {len(folds)} rolling folds, with signal dates from
{run["test_start"]} through {run["test_end"]}. It covers {run["test_assets"]} distinct
eligible companies after the trailing-history requirement, from a panel of
{run["panel_assets"]} current constituents. Incomplete final test windows are omitted.
Each run's `run.json` records the modeling source commit, dependency versions,
active parameters and panel dimensions. Generator settings are recorded when the
pipeline generates a panel in memory; file-based runs describe the loaded panel.

{names[best_model]} has the highest realized daily net Sharpe,
{best.cost_adjusted_sharpe:.2f}, with an annualized net return of
{best.annualized_return:.1%} and maximum drawdown of {best.max_drawdown:.1%}.

{ic_table}

{evidence}
Current-constituent survivorship and idealized close execution limit the economic
interpretation of the portfolio results. Borrow and financing costs are excluded.

## Factor evidence

The factor statistics use the same out-of-sample signal window as the model
forecasts. Raw and neutralized statistics are available in
`reports/real/factor_ics.csv`; the table below shows centered neutralized ranks.

{factor_table}

![Factor IC decay](../reports/real/factor_decay.png)

Raw factors retain sector and market exposures. Beta is neutralized against sector
only. The universe consists of current constituents, so historical size
associations reflect that selection.

Horizon-spaced IC samples avoid mechanical target overlap. They can still be
serially dependent, and the reported t-statistics use the sample-mean formula
without corrections for multiple comparisons. The annual plot averages daily IC
with overlapping targets and is descriptive.

## Accounting and availability

Portfolio returns compound the daily P&L of held shares. Costs apply to traded
dollars against drifted holdings, including entry and final liquidation. Sector
contributions sum arithmetic daily net returns and therefore do not add to the
compounded portfolio return.

Benchmark labels compound the next trading days and exclude the signal day.
Momentum lags use the common trading calendar. First-filed fundamentals become
available on the business day after filing. Missing factor ranks are imputed
after neutralization. Current-constituent selection and missing future endpoints
limit the historical universe.

## Data rebuild

The rebuild recorded {rebuild["successful_downloads"]} successful company downloads
and {rebuild["skipped_downloads"]} skipped downloads. The snapshot comparison covers
{rebuild["common_rows"]:,} common panel rows across {rebuild["common_assets"]} companies.
Adjusted prices can differ across downloads; the saved comparison records the
differences in each numeric field.

{rebuild_table}

The aggregate comparison can be regenerated from two local datasets:

```bash
python scripts/check_rebuild.py --reference data/real --rebuilt data/real/rebuilt \\
  --out reports/real/rebuild_check.json
```

## Synthetic results

The synthetic panel contains {profile["assets"]} assets over
{profile["trading_days"]:,} business days, with the same model hyperparameters and
walk-forward layout as the real run. Pairwise correlation is
{profile["pairwise_correlation"]:.2f} and within-sector correlation is
{profile["within_sector_correlation"]:.2f}; median annual volatility is
{profile["annual_volatility"][2]:.1%}. Gross margin is absent for
{profile["fundamentals"]["gross_margin"]["missing_share"]:.0%} of month-end rows.

Pooled excess return kurtosis is {profile["pooled_excess_kurtosis"]:.2f}, compared with
{real_profile["pooled_excess_kurtosis"]:.2f} in the real panel. The Student-t process
has less extreme tails than the observed crisis and event returns.

Synthetic forecast rank IC spans {synth_metrics.mean_rank_ic.min():.4f} to
{synth_metrics.mean_rank_ic.max():.4f}, and daily net Sharpe spans
{synth_metrics.cost_adjusted_sharpe.min():.2f} to {synth_metrics.cost_adjusted_sharpe.max():.2f}.
The individual forecasts appear in `reports/synthetic/metrics.csv`.

This document is generated by `python scripts/write_result_notes.py` from the
saved aggregate tables.
"""
(root / "docs" / "RESULTS.md").write_text(text, encoding="utf-8")
