"""Regenerate README results directly from the saved real-data experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
reports = root / "reports" / "real"
run = json.loads((reports / "run.json").read_text())
profile = json.loads((reports / "data_profile.json").read_text())
metrics = pd.read_csv(reports / "metrics.csv").set_index("model")
coverage = pd.read_csv(reports / "feature_coverage.csv").set_index("factor")
names = {
    "composite": "Equal-weight composite",
    "ridge": "Ridge",
    "tree": "Gradient boosting",
    "neural": "Neural network",
}
rows = [
    "| Forecast | Monthly rank IC | Net annual return | Daily net Sharpe | Max drawdown "
    "| Annual one-way turnover per side |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for model, name in names.items():
    m = metrics.loc[model]
    rows.append(
        f"| {name} | {m.mean_rank_ic:.4f} | {m.annualized_return:.1%} | "
        f"{m.cost_adjusted_sharpe:.2f} | {m.max_drawdown:.1%} | "
        f"{m.annual_one_way_turnover_per_side:.2f}x |"
    )
table = "\n".join(rows)
text = f"""# Cross-sectional equity alpha

## Research question

Can linear, tree and small neural models turn familiar equity factors into better
out-of-sample forecasts than an equal-weight composite? This project tests that
question with rolling historical training windows and a long-short quintile book.

## Data

The saved experiment uses {profile["assets"]:,} current S&P 500 constituents outside
Financials and Real Estate, with {profile["rows"]:,} observations from
{profile["start"]} through {profile["end"]}. Yahoo Finance supplies adjusted daily
prices, SEC EDGAR supplies first-filed financial statements, and Wikipedia supplies
the current constituent list and sectors. Vendor data is kept locally in ignored
`data/real/` and `.cache/`; it is excluded from the repository because it cannot be
redistributed under the data source's terms.

Fundamentals enter the panel on the business day after filing and expire when
older than the builder's staleness limit. Price features use trailing observations.
Missing fundamentals remain in the universe: missing factor ranks are imputed to
zero after centering. Quality is observed for {coverage.loc["quality", "observed_share"]:.1%}
of out-of-sample company dates. Recently listed names need sufficient price history
before becoming eligible. The universe is selected from current constituents, so
historical results contain survivorship bias.

The data builder writes daily prices, quarterly fundamentals, a universe table and
a modeling panel. The deterministic synthetic generator writes the same schemas,
using market and sector factors, Student-t innovations, quarterly filing dates,
persistent fundamentals and weak latent alpha. The saved distribution checks in
[the real profile](reports/real/data_profile.json) and
[the synthetic profile](reports/synthetic/data_profile.json) show the approximation;
synthetic data does not reproduce historical crisis tails.

## Method

Value, momentum, quality, low volatility, size, beta and liquidity are winsorized,
standardized, neutralized and ranked within each date. Positive book-to-price is
required for value. Sector and beta exposures are removed before ranking; beta
itself is only sector-neutralized, since regressing it on itself would leave
numerical noise. The benchmark is the mean of the seven centered ranks.

The target is stock total return minus compounded SPY return over the next 21
trading days, excluding the signal day. Rolling windows contain 504 training days,
a 21-day purge and 63 test days; test windows are contiguous. Ridge, histogram
gradient boosting and a small ReLU network use the settings fixed in
[the experiment protocol](docs/EXPERIMENT.md), without tuning on test results.

At each rebalance the top and bottom quintiles receive long and short exposure
equal to the pre-trade NAV on each side. Shares are held for 21 trading days and
marked daily. Costs are 5 bp per dollar traded against drifted holdings, including
initial entry and final liquidation. Returns compound daily and Sharpe uses daily
net P&L. The signal-close execution assumption is idealized. Linear factor
neutralization does not guarantee beta or sector neutrality of the quantile book.

IC summaries and their t-statistics use non-overlapping target-horizon signal dates.
Factor diagnostics include raw and neutralized IC, decay at 1, 5, 10, 21 and 63 days,
and daily returns of each factor's quintile spread. The annual IC chart averages
daily IC with overlapping targets and is descriptive, without daily-sample t-tests.

## Real-data results

Out-of-sample signals run from {run["test_start"]} through {run["test_end"]}.
The table is generated from [saved metrics](reports/real/metrics.csv); run parameters,
dependency versions and the source commit are in [run.json](reports/real/run.json).

{table}

Ridge and gradient boosting have higher realized Sharpe than the composite in this
sample. Their mean rank IC is small, and the non-overlapping IC t-statistics do not
provide strong evidence of a stable forecasting effect. The neural model adds
little rank information. Signs, costs and drawdowns are reported without selecting
a more favorable configuration. See [the result notes](docs/RESULTS.md) for factor
findings, comparisons with the expected ranges and rebuild verification.

![Cumulative out-of-sample net long-short return](reports/real/equity_curve.png)

![Rank IC by year](reports/real/ic_by_year.png)

[The synthetic run](reports/synthetic/research_report.md) uses the same model and
walk-forward settings and serves as a reproducible distribution and accounting
check; its results are not evidence about market performance.

## Limitations

Current-constituent survivorship, imperfect XBRL concept matching, revised vendor
prices, idealized close execution, unmodeled borrow and financing costs, and a
single fixed universe limit the economic interpretation of the backtest.

## Reproduce

Create and activate a Python virtual environment, then install and check the package:

```bash
pip install -e ".[dev,data]"
ruff check .
pytest -q
```

Rebuild private data locally (set `SEC_USER_AGENT` to a suitable name and contact
before downloading), then run the real experiment:

```bash
equity-alpha build-data --out data/real
equity-alpha run --panel data/real/panel.parquet --output-dir reports/real
equity-alpha profile-data --panel data/real/panel.parquet --out reports/real/data_profile.json
```

The universe and provider responses can change. The committed report describes the
supplied snapshot, and a fresh download need not reproduce it bit for bit. To compare
against a local snapshot, rebuild into a separate directory and run
`scripts/check_rebuild.py` with `--reference`, `--rebuilt` and `--out` paths.

No vendor access is needed for a synthetic run. The default generator is a small
sample; the saved synthetic report uses a longer history:

```bash
equity-alpha sample-data --out data/synthetic --days 2016
equity-alpha run --panel data/synthetic/panel.parquet --output-dir reports/synthetic
equity-alpha profile-data --panel data/synthetic/panel.parquet \\
  --out reports/synthetic/data_profile.json
python scripts/write_readme.py
python scripts/write_result_notes.py
python scripts/render_figures.py reports/real reports/synthetic
```

Without `--panel`, `equity-alpha run` generates the default synthetic panel in memory.
Use `--no-include-neural` for a faster run. CI generates synthetic data and runs the
same parquet-loading path; the real-schema check is skipped when private data is absent.
Reports contain aggregate tables and figures, with no saved company-level predictions.
"""
(root / "README.md").write_text(text, encoding="utf-8")
