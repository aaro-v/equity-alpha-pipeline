# Cross-Sectional Equity Alpha Research Pipeline

End-to-end Python research pipeline for cross-sectional equity alpha research:
factor construction, point-in-time hygiene, sector and beta neutralization,
ranking, portfolio formation, purged walk-forward validation, and model
comparison across linear, tree-based, and small neural-network forecasters.

This repository is being built as a resume-ready research engineering project.
Generated sample results are reproducible and use deterministic synthetic data
unless explicitly documented otherwise.

## Target Resume Bullets

- End-to-end pipeline for cross-sectional equity alpha: factor construction,
  sector/beta neutralization, ranking, and portfolio formation on a US equity
  universe with point-in-time data hygiene.
- Implemented purged, embargoed walk-forward validation; evaluated factors with
  IC, rank IC, decay curves, turnover, and cost-adjusted Sharpe across regimes
  and sectors.
- Compared linear, gradient-boosted, and small neural-net return forecasters;
  analyzed factor crowding and decay via rolling cross-sectional regressions
  and bootstrap stability tests.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Run The Sample Pipeline

```bash
equity-alpha run --output-dir reports/generated
```

The command generates deterministic synthetic data in memory, constructs
point-in-time factors, creates forward excess-return labels, runs purged
walk-forward validation, fits ridge/tree/neural baselines, forms long-short
portfolios, and writes artifacts:

- `reports/generated/metrics.csv`
- `reports/generated/portfolio_returns.csv`
- `reports/generated/sector_attribution.csv`
- `reports/generated/predictions.parquet`
- `reports/generated/equity_curve.png`
- `reports/generated/research_report.md`

For a faster smoke run without PyTorch training:

```bash
equity-alpha run --no-include-neural --assets 50 --days 500
```

To write only the deterministic sample panel:

```bash
equity-alpha sample-data --output data/cache/sample_panel.parquet
```

## Implemented Scope

- Deterministic synthetic US-equity-like panel with prices, sectors, market
  caps, fundamentals, liquidity, benchmark returns, and explicit availability
  dates.
- Point-in-time factor construction for value, momentum, quality, volatility,
  size, beta, and liquidity signals.
- Sector and beta neutralization, cross-sectional ranking, and future labels
  that are excluded from features.
- Purged and embargo-stepped walk-forward validation.
- Ridge, histogram gradient boosting, and small PyTorch neural-network
  forecasters.
- IC, rank IC, turnover, cost-adjusted Sharpe, drawdown, sector attribution,
  long-short portfolio returns, and generated Markdown/CSV/PNG artifacts.

The generated outputs are for engineering validation only. They use synthetic
data and should not be read as evidence of live-market profitability.
