# Experiment protocol

The model configuration was recorded on 2026-11-01 before evaluation.

The real-data experiment uses the panel through 2026-10-30 and the current
constituent universe. The target is the next 21 trading days' compounded stock
return minus compounded SPY return. Features use the signal close, with execution
at that close as an idealized assumption. Fundamentals become available on the
business day after filing.

Rolling windows contain 504 training dates, a 21-date purge and 63 test dates.
Successive test windows are contiguous. Eligibility begins after a 131-observation
price warm-up.

The models are ridge regression with alpha 5; histogram boosting with 80 iterations,
learning rate 0.05, 15 leaves and L2 regularization 0.05; and a 7-input,
12-hidden-unit ReLU network. The network trains for 25 full-batch Adam epochs with
learning rate 0.01 and weight decay 0.001. The model seed is 17. The reference
forecast is the equal-weight mean of the seven centered factor ranks.

The synthetic reporting panel uses seed 7, 100 assets and 2,016 business days.
The default synthetic sample contains 756 business days.

Factors are winsorized at each date's 1st and 99th percentiles, standardized and
neutralized against sector and beta exposures. Beta itself is neutralized against
sector only, since regressing beta on itself would leave numerical noise.
Cross-sectional percentile ranks are centered, with missing ranks imputed to zero.

The portfolio rebalances every 21 trading dates into the top and bottom quintiles.
Each side receives exposure equal to the pre-trade NAV. Shares are held between
rebalances and marked using daily total returns. Transaction costs are 5 bp per
dollar traded against drifted holdings, including entry and final liquidation.
A run terminates if NAV is exhausted. Interest, borrowing fees and intraday
execution are excluded.

Sharpe is calculated from daily net returns with an annualization factor of
sqrt(252). IC summaries and t-statistics use dates spaced by the target horizon.
Factor decay is measured at horizons 1, 5, 10, 21 and 63 trading days, each with
its own non-overlapping signal sample. The annual IC chart averages daily IC and
is descriptive because its targets overlap.

## Execution and robustness specification

The follow-up configuration was recorded on 2026-11-01 before evaluating the
revised experiments. Training windows remain 504 dates, test windows 63 dates,
and the outer purge 21 dates. The main signal is observed at close t, traded at
close t+1, and labeled over the following 21 trading days, from t+1 to t+22.
Training labels must be fully available by the first test signal close. Equality
at that close is allowed because orders execute the following day.

The neural architecture remains seven inputs, 12 ReLU hidden units and one output.
Adam uses learning rate 0.001, weight decay 0.001, batch size 4,096 and at most 100
epochs. Features and targets are standardized using the fitting slice only.
The last 20% of training dates form validation, separated from the fitting slice
by 22 trading dates. Early stopping uses patience 10 and minimum improvement
0.0001 in standardized validation MSE, restoring the best weights. Seed 17 fixes
initialization and mini-batch order. Validation history and optimizer updates
are recorded for each fold.

Five portfolio variants are reported: next-close execution; same-close execution
using the same forecasts; size-neutralized features; a log-market-cap-neutral
book using the original features; and both size controls together. Feature
neutralization removes sector, beta and log-market-cap exposures before ranking;
the size factor itself is zero in this variant. The book projects initial
quintile weights onto the space orthogonal to the intercept and signal-date log
market cap, then scales gross exposure to two. This hedge can include stocks
outside the quintiles. Names without observed market cap are excluded from the
hedged book. Neutrality is enforced against information available at the signal
close and can drift between rebalances.

Attribution regresses each model's daily net returns on the seven net factor
quintile-spread returns with the same execution convention. Regressions include
an intercept. Factor-adjusted Sharpe is calculated after subtracting the fitted
factor component while retaining the intercept. Attribution is descriptive and
estimated over the evaluation sample, rather than an executable strategy.

Sharpe uncertainty uses 2,000 moving-block bootstrap draws, 63-date blocks, seed
17 and percentile 95% intervals. At least two blocks of observations are required.
Intervals are pointwise, without adjustment for the four forecasts, seven factors
or the robustness variants.
