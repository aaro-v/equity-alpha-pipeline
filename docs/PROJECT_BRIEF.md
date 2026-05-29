# Project Scope

This repository implements an offline cross-sectional equity alpha research
pipeline in Python, pandas, NumPy, scikit-learn, and PyTorch. The pipeline uses
deterministic synthetic data to demonstrate research engineering practices
without making claims about live-market profitability.

The sample panel is designed to resemble a US equity universe with prices,
sectors, market caps, fundamentals, liquidity, benchmark returns, and explicit
point-in-time availability dates. Factors cover value, momentum, quality,
volatility, size, beta, and liquidity-style signals. Feature processing includes
sector and beta neutralization, cross-sectional ranking, and forward-return
labels that are kept out of the feature set.

The modeling workflow runs purged walk-forward validation across ridge,
tree-based, and small neural-network forecasters. Evaluation artifacts include
IC, rank IC, turnover, transaction costs, drawdown, sector attribution,
long-short portfolio returns, generated reports, and reproducible CLI outputs.

The implementation is intended as a compact, inspectable example of the
research workflow that would be adapted for institutional point-in-time market
and fundamentals data in production.
