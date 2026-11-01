# Results

Signals run from 2014-08-13 through 2026-08-21 across 48 rolling folds and 394
eligible companies. The main setting trades one close after the signal and holds
for 21 trading dates. Training windows contain 504 dates with a 21-date outer
purge; all training targets are available by the test signal close. Each forecast
has 144 non-overlapping monthly IC observations.

## Main experiment

| Forecast | Rank IC | IC t-statistic | Net annual return | Net Sharpe | Sharpe 95% interval | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Equal-weight composite | 0.0038 | 0.47 | -0.3% | 0.02 | [-0.44, 0.57] | -31.5% |
| Ridge | 0.0161 | 1.51 | 9.6% | 0.74 | [0.31, 1.41] | -22.0% |
| Gradient boosting | 0.0118 | 1.19 | 7.1% | 0.59 | [0.18, 1.27] | -24.1% |
| Neural network | 0.0076 | 0.83 | 6.8% | 0.59 | [0.15, 1.16] | -17.4% |

The raw portfolio Sharpe intervals exclude zero for three forecasts, while the
monthly IC estimates remain imprecise. Portfolio returns include 5 bp costs on
actual trades, including entry and final liquidation. Annual returns compound the
daily net P&L; sector contributions sum daily returns arithmetically.

## Factor attribution

Daily net model returns are regressed on the seven daily net factor
quintile-spread returns, with an intercept and matching execution lag. The
factor-adjusted return subtracts fitted factor exposure while retaining the
intercept. Its Sharpe is recorded as `residual_sharpe`; demeaned OLS residuals
would have zero mean. The regressions use the full evaluation sample and are
descriptive rather than trading rules.

| Forecast | Joint R-squared | Annualized intercept | Factor-adjusted Sharpe |
| --- | ---: | ---: | ---: |
| Equal-weight composite | 87.1% | -0.18% | -0.05 |
| Ridge | 58.7% | -1.87% | -0.21 |
| Gradient boosting | 48.5% | -1.29% | -0.14 |
| Neural network | 64.1% | -4.14% | -0.56 |

Ridge's factor loadings and contributions are:

| Factor | Loading | Annual mean-return contribution |
| --- | ---: | ---: |
| Beta | 0.393 | 3.96 percentage points |
| Liquidity | 0.098 | -0.66 percentage points |
| Low volatility | -0.210 | 2.38 percentage points |
| Momentum | 0.255 | 0.29 percentage points |
| Quality | -0.304 | 1.56 percentage points |
| Size | 0.352 | 3.94 percentage points |
| Value | -0.361 | 0.49 percentage points |

Contributions are loading times the factor's arithmetic mean daily return times
252; they do not add to CAGR. Joint R-squared is explained variance across all
seven factors, not the fraction of Sharpe explained by size. Size alone explains
5.8% of ridge's daily variance. Beta has the largest conditional variance
contribution; size and beta contribute similar amounts to its mean return.
Historical size effects are particularly exposed to the selection of firms that
survived into today's index. This sample cannot identify how much of that effect
would persist in a historical constituent universe.

[Attribution](../reports/real/factor_attribution.csv) contains the joint and
size-only regressions for every variant. [Loadings](../reports/real/factor_loadings.csv)
include each factor's mean-return contribution and partial R-squared.

## Execution and size controls

The same-close row uses the same next-close-trained forecasts and changes only
the execution and holding interval. It isolates timing effects rather than
retraining on an earlier target. Size-neutral features remove sector, beta and
log-cap exposures before ranking, with the size feature set to zero. Ranking can
reintroduce exposure, so the book hedge is measured separately.

| Variant | Composite Sharpe | Ridge Sharpe | Boosting Sharpe | Neural Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Next-close | 0.02 | 0.74 | 0.59 | 0.59 |
| Same-close | 0.03 | 0.69 | 0.51 | 0.57 |
| Size-neutral features | 0.26 | 0.30 | 0.36 | 0.29 |
| Cap-neutral book | 0.09 | 0.29 | 0.34 | 0.21 |
| Both size controls | 0.30 | 0.28 | 0.34 | 0.31 |

The cap-neutral book projects quintile weights off the intercept and signal-date
log market cap, then normalizes gross exposure to two. It can trade stocks outside
the quintiles and excludes names without observed market cap. Dollar and log-cap
exposure are zero against signal-date inputs at rebalance; exposure can drift as
positions and market caps change. All resulting trades incur costs. The hedge
changes other correlated exposures and turnover, so its Sharpe difference is not
a causal estimate of the size premium.

Ridge's Sharpe interval is [-0.17, 0.91] for the cap-neutral book and
[-0.20, 0.87] with both size controls. The corresponding neural intervals are
[-0.28, 0.77] and [-0.23, 0.88]. No consistent neural advantage remains after
controls. Each variant's complete metrics, intervals and turnover appear in
[metrics.csv](../reports/real/metrics.csv).

## Uncertainty and comparisons

Intervals are 95% percentile moving-block bootstrap intervals: 2,000 draws,
63-date overlapping blocks, no circular wraparound and seed 17. Fewer than two
blocks of observations yield an unavailable interval. These are pointwise
intervals, unadjusted for four forecasts, seven factors and five robustness
variants. They are not tests of pairwise differences between model Sharpes.
Non-overlapping monthly IC avoids mechanical target overlap, but its simple
t-statistics do not correct for every form of serial dependence or multiple testing.

## Neural training

The fixed network has seven inputs, 12 ReLU hidden units and one output. Features
and target are standardized on the fitting slice. Validation uses the final 20%
of each training window, separated by 22 trading dates. Mini-batch Adam uses
batch size 4,096, learning rate 0.001 and weight decay 0.001, with a 100-epoch cap,
patience 10 and a minimum validation-MSE improvement of 0.0001. The best checkpoint
is restored.

The 48 primary real-data fits use 374 to 2,590 optimizer updates each, with a
median of 526.5. The median selected checkpoint is epoch 5.5; runs last 11 to 70
epochs including the stopping patience. The size-neutral-feature fits have a
median of 534 optimizer updates. [Training diagnostics](../reports/real/neural_training.csv)
record the fitting and validation dates, target scale, best loss, selected epoch
and total updates for every fit.

## Synthetic experiment

The saved synthetic panel has 100 assets and 2,016 business dates. Its primary
net Sharpes are 0.51 for the composite, -0.06 for ridge, -0.19 for boosting and
-0.37 for the neural network. All four primary synthetic Sharpe intervals include
zero. The synthetic neural fits use a median of 190 optimizer updates and select
a median checkpoint at epoch 9. The weak synthetic signal does not establish a
reliable advantage for the learned forecasts.

## Interpretation and limitations

The evidence is consistent with models taking familiar factor exposures,
including a material size tilt, rather than delivering additional factor-adjusted
alpha. Next-close execution preserves positive raw model performance here, but
size controls reduce it and the factor-adjusted Sharpes are negative. The data
use today's constituents and sectors, with imperfect XBRL concept matching and
revised vendor prices. Execution assumes closing-price fills and excludes borrow
and financing costs. Robustness checks reuse the evaluated period, so a separate
historical-constituent sample would be needed for confirmation.

The [data profile](DATA_PROFILE.md) describes the panel and factor distributions.
Each report's `run.json` records its modeling source commit, active settings,
versions and data dimensions. The [experiment configuration](EXPERIMENT.md)
specifies the procedures.
