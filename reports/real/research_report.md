# Walk-forward research results

Source: `data/real/panel.parquet`. Test signals: 2014-08-13 through 2026-08-21.

The main portfolio executes at the next close. IC summaries use non-overlapping signal dates; Sharpe uses daily net returns. Its 95% percentile intervals use moving blocks and are pointwise. 4 forecasts and seven factors are tested, plus the execution and size-control variants; intervals are not adjusted for multiple comparisons. Intervals require at least two blocks of daily observations. The book holds shares between rebalances and pays costs on actual traded dollars, including entry and final liquidation. Sector contributions sum arithmetic daily net returns and do not add to compounded total return.

| model | variant | mean_rank_ic | cost_adjusted_sharpe | sharpe_ci_lower | sharpe_ci_upper | portfolio_return | max_drawdown | annual_one_way_turnover_per_side |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| composite | main | 0.0038 | 0.0168 | -0.4363 | 0.5686 | -0.0342 | -0.3152 | 3.7656 |
| ridge | main | 0.0161 | 0.7407 | 0.3089 | 1.4133 | 2.0064 | -0.2202 | 2.9764 |
| tree | main | 0.0118 | 0.5897 | 0.1826 | 1.2673 | 1.2736 | -0.2411 | 4.4381 |
| neural | main | 0.0076 | 0.5914 | 0.1488 | 1.1579 | 1.1908 | -0.1738 | 4.1947 |
| composite | same_close | 0.0012 | 0.0279 | -0.4200 | 0.5927 | -0.0220 | -0.3372 | 3.7662 |
| ridge | same_close | 0.0157 | 0.6934 | 0.2703 | 1.3655 | 1.7779 | -0.2225 | 2.9803 |
| tree | same_close | 0.0094 | 0.5086 | 0.1101 | 1.1985 | 1.0046 | -0.2368 | 4.4406 |
| neural | same_close | 0.0071 | 0.5657 | 0.1551 | 1.1292 | 1.1021 | -0.1795 | 4.1941 |
| composite | cap_neutral_book | 0.0038 | 0.0925 | -0.3477 | 0.6518 | 0.0526 | -0.2620 | 3.8435 |
| ridge | cap_neutral_book | 0.0161 | 0.2926 | -0.1748 | 0.9067 | 0.4123 | -0.2705 | 2.9975 |
| tree | cap_neutral_book | 0.0118 | 0.3444 | -0.0888 | 0.9632 | 0.5065 | -0.2278 | 4.2768 |
| neural | cap_neutral_book | 0.0076 | 0.2074 | -0.2756 | 0.7745 | 0.2184 | -0.3228 | 4.0050 |
| composite | size_neutral_features | 0.0070 | 0.2643 | -0.1937 | 0.8165 | 0.2730 | -0.2038 | 3.9322 |
| ridge | size_neutral_features | 0.0014 | 0.2967 | -0.1849 | 0.8859 | 0.4616 | -0.2759 | 3.6876 |
| tree | size_neutral_features | -0.0011 | 0.3565 | -0.0854 | 0.9750 | 0.5300 | -0.2397 | 5.3511 |
| neural | size_neutral_features | -0.0016 | 0.2942 | -0.2382 | 0.8565 | 0.3805 | -0.3662 | 5.2489 |
| composite | size_neutral | 0.0070 | 0.3018 | -0.1418 | 0.8458 | 0.3261 | -0.1970 | 4.0331 |
| ridge | size_neutral | 0.0014 | 0.2767 | -0.2007 | 0.8703 | 0.4120 | -0.3044 | 3.7549 |
| tree | size_neutral | -0.0011 | 0.3375 | -0.1061 | 0.9406 | 0.4740 | -0.2525 | 5.3635 |
| neural | size_neutral | -0.0016 | 0.3054 | -0.2327 | 0.8803 | 0.4012 | -0.3648 | 5.3042 |

![Cumulative net return](equity_curve.png)

![Annual rank IC](ic_by_year.png)

![Factor IC decay](factor_decay.png)

![Size controls](size_controls.png)
