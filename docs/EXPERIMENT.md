# Experiment protocol

Fixed before examining model test results on 2026-11-01.

Use the supplied panel through 2026-10-30, the current constituent universe,
and the same feature definitions for both real and synthetic data. The target
is the next 21 trading days' compounded stock return minus compounded SPY
return. Features use the signal close; execution at that close is an idealized
assumption. Fundamentals become usable the business day after filing.

Use rolling windows of 504 training dates, a 21-date purge, and 63 test dates,
with no gap between successive test windows. Require 131 price observations
before an asset enters the research frame. Do not adjust settings to improve
test statistics.

Retain ridge alpha 5; histogram boosting with 80 iterations, learning rate
0.05, 15 leaves and L2 0.05; and a 7-input, 12-hidden-unit ReLU network trained
for 25 full-batch Adam epochs with learning rate 0.01 and weight decay 0.001.
Use model seed 17 and synthetic seed 7. Compare all three with an equal-weight
mean of the seven centered factor ranks.

Winsorize factors at each date's 1st and 99th percentiles, then remove sector
and beta exposures using the available observations. For the beta factor itself,
remove sector only: regressing beta on itself would produce numerical noise.
Center cross-sectional percentile ranks and impute missing ranks to zero.

Rebalance every 21 trading dates into the top and bottom quintiles, long 100%
and short 100% of NAV. Hold shares between rebalances, carry cash, and mark the
book using actual daily total returns. Charge 5 bp per dollar traded against
drifted holdings, including entry and final liquidation. Stop trading if NAV
is exhausted. No interest, borrowing fees, or intraday execution are modeled.

Compute Sharpe with sqrt(252) from daily net P&L. IC summaries and t-statistics
use dates spaced by the target horizon. For decay, use horizons 1, 5, 10, 21,
and 63, each with its own non-overlapping sample, on out-of-sample signal dates.
Record daily IC separately for the annual chart. Report all results, including
negative values, and investigate suspicious statistics without parameter tuning.
