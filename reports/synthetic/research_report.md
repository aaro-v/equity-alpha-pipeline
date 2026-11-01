# Walk-forward research results

Source: `data/synthetic/panel.parquet`. Test signals: 2020-07-08 through 2025-08-01.

IC summaries use non-overlapping signal dates; Sharpe uses daily net returns. The book holds shares between rebalances and pays costs on actual traded dollars, including entry and final liquidation. Sector contributions sum arithmetic daily net returns and do not add to compounded total return.

| model | mean_rank_ic | cost_adjusted_sharpe | portfolio_return | max_drawdown | annual_one_way_turnover_per_side |
| --- | --- | --- | --- | --- | --- |
| composite | 0.0171 | 0.4188 | 0.2012 | -0.2619 | 3.0133 |
| ridge | -0.0026 | -0.0873 | -0.1027 | -0.4342 | 3.3795 |
| tree | 0.0104 | 0.1247 | 0.0406 | -0.2318 | 5.7588 |
| neural | -0.0230 | -1.1524 | -0.4042 | -0.4042 | 3.9650 |

![Cumulative net return](equity_curve.png)

![Annual rank IC](ic_by_year.png)

![Factor IC decay](factor_decay.png)
