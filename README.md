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
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
