# Experiment configuration

The revised specification was recorded on 2026-11-01 before evaluation. The real
panel ends on 2026-10-30 and uses current constituents. All models use the same
504-date rolling training window, 63-date test window and 21-date outer purge.

## Signals and execution

Signals use information available at close t. The main book trades at close t+1,
and its target is stock return minus compounded SPY return from t+1 to t+22.
The last training target must be available by the first test signal close; equality
at that close is allowed because execution occurs the next day. Fundamentals
enter on the business day after filing. Eligibility begins after 131 observed
price dates. Transaction costs are 5 bp per dollar traded, including entry and
final liquidation. Shares are held for 21 trading dates between rebalances.
Borrow, financing and intraday execution costs are excluded.

## Features and models

Factors are winsorized at the 1st and 99th percentiles, standardized and neutralized
against sector and beta before ranking. Beta itself is only sector-neutralized.
Centered percentile ranks are imputed to zero when a factor is missing. The
composite is the equal-weight mean of the seven ranks.

Ridge alpha is 5. Histogram boosting uses 80 iterations, learning rate 0.05,
15 leaves and L2 regularization 0.05. Model seed is 17.

The neural architecture is seven inputs, 12 ReLU hidden units and one output.
Adam uses learning rate 0.001, weight decay 0.001, batch size 4,096 and at most 100
epochs. Features and targets are standardized using the fitting slice only.
The last 20% of training dates form validation, separated from the fitting slice
by 22 trading dates. Early stopping uses patience 10 and minimum improvement
0.0001 in standardized validation MSE, restoring the best weights. Seed 17 fixes
initialization and mini-batch order. Validation history and optimizer updates
are recorded for each fold. The validation slice is never used to fit scalers.

The synthetic report uses 100 assets, 2,016 business days and generator seed 7.

## Robustness variants

Five portfolios are reported: next-close execution; same-close execution using
the same forecasts; size-neutralized features; a log-market-cap-neutral book
using the original features; and both size controls together. The execution
comparison changes the holding interval without retraining the forecasts.

Size-neutral features remove joint sector, beta and log-market-cap exposures
before ranking; the size feature itself is zero. The hedged book projects initial
quintile weights onto the space orthogonal to the intercept and signal-date log
market cap, then scales gross exposure to two. It can include names outside the
quintiles. Names with missing market cap are excluded from this book. The
constraint uses information available at the signal close; exposure can drift
between rebalances. Each variant includes the turnover costs of its own trades.

## Attribution and uncertainty

Attribution regresses daily net model returns on the seven net factor
quintile-spread returns with the same execution convention, including an
intercept. Factor-adjusted Sharpe subtracts the fitted factor component and retains
the intercept. OLS residuals with the intercept removed would have mean zero.
These regressions describe the evaluation sample and are not trading rules.
R-squared measures explained return variance, not a proportion of Sharpe.

Sharpe intervals use 2,000 non-circular moving-block bootstrap draws, 63-date
blocks, seed 17 and percentile 95% intervals. At least two blocks of observations
are required. Intervals are pointwise and are not adjusted for the four forecasts,
seven factors or five robustness variants. IC summaries use non-overlapping
signal dates; their simple t-statistics do not correct for all serial dependence.
The annual chart averages daily IC with overlapping targets and is descriptive.
