# Project Brief

Build a resume-ready cross-sectional equity alpha research pipeline using
Python, pandas, NumPy, scikit-learn, and PyTorch.

The project should work offline with deterministic synthetic point-in-time
sample data while clearly documenting what would change for live institutional
data. It should avoid unsupported profitability claims.

Core deliverables:

- deterministic US-equity-like panel data with prices, sectors, market caps,
  fundamentals, benchmark returns, and point-in-time availability dates
- factor construction for value, momentum, quality, volatility, size, beta, and
  liquidity-style signals
- sector and beta neutralization, cross-sectional ranking, portfolio formation
- purged and embargoed walk-forward validation
- IC, rank IC, decay curves, turnover, cost-adjusted Sharpe, drawdown, sector
  attribution, regime attribution
- linear, gradient-boosted, and small PyTorch neural-net forecasters
- factor crowding and decay analysis via rolling cross-sectional regressions
  and bootstrap stability tests
- CLI commands, tests, generated report, and clean README
