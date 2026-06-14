from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from equity_alpha.research.models import model_columns


def write_artifacts(
    output_dir: Path,
    metrics: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    sector_attribution: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Path:
    """Write CSV artifacts, an equity-curve chart, and a Markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    portfolio_returns.to_csv(output_dir / "portfolio_returns.csv", index=False)
    sector_attribution.to_csv(output_dir / "sector_attribution.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)

    chart_path = output_dir / "equity_curve.png"
    _plot_equity_curve(portfolio_returns, chart_path)

    report_path = output_dir / "research_report.md"
    report = _render_report(
        metrics, portfolio_returns, sector_attribution, predictions, chart_path.name
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _plot_equity_curve(portfolio_returns: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, frame in portfolio_returns.groupby("model"):
        curve = (1.0 + frame.sort_values("date")["net_return"]).cumprod()
        ax.plot(pd.to_datetime(frame.sort_values("date")["date"]), curve, label=model)
    ax.set_title("Walk-forward long-short equity curve")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _render_report(
    metrics: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    sector_attribution: pd.DataFrame,
    predictions: pd.DataFrame,
    chart_name: str,
) -> str:
    best = metrics.iloc[0]
    fold_summary = _fold_summary(predictions, portfolio_returns)
    sector_summary = (
        sector_attribution.sort_values(["model", "rank_ic"], ascending=[True, False])
        .groupby("model")
        .head(3)
    )
    return "\n".join(
        [
            "# Cross-Sectional Equity Alpha Research Report",
            "",
            "Synthetic, deterministic sample results. These figures validate pipeline mechanics; "
            "they are not live trading claims.",
            "",
            "## Model Summary",
            "",
            _markdown_table(metrics),
            "",
            f"Best sample model by cost-adjusted Sharpe: **{best['model']}**.",
            "",
            "## Benchmark Comparison",
            "",
            _markdown_table(_benchmark_comparison(metrics)),
            "",
            "## Validation Summary",
            "",
            _markdown_table(_validation_summary(predictions)),
            "",
            "## Prediction Coverage",
            "",
            _markdown_table(_prediction_coverage_summary(predictions)),
            "",
            "## Fold Stability",
            "",
            _markdown_table(fold_summary),
            "",
            "## Fold Robustness",
            "",
            _markdown_table(_fold_robustness_summary(fold_summary)),
            "",
            "## Portfolio Diagnostics",
            "",
            _markdown_table(_portfolio_summary(portfolio_returns)),
            "",
            "## Tail Risk Diagnostics",
            "",
            _markdown_table(_tail_risk_summary(portfolio_returns)),
            "",
            "## Top Sector Rank ICs",
            "",
            _markdown_table(sector_summary),
            "",
            "## Equity Curve",
            "",
            f"![Equity curve]({chart_name})",
            "",
            "## Method Notes",
            "",
            "- Features are built only after synthetic availability dates are reached.",
            "- Forward labels use future excess returns and stay out of feature construction.",
            "- Walk-forward splits use a purge gap before each test window and an embargo step.",
            "- Long-short returns include turnover-based transaction costs.",
            "",
        ]
    )


def _benchmark_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    benchmark_model = "ridge" if "ridge" in set(metrics["model"]) else metrics.iloc[0]["model"]
    benchmark = metrics.loc[metrics["model"] == benchmark_model].iloc[0]
    return (
        metrics.assign(
            benchmark=benchmark_model,
            rank_ic_delta=metrics["mean_rank_ic"] - benchmark["mean_rank_ic"],
            sharpe_delta=metrics["cost_adjusted_sharpe"] - benchmark["cost_adjusted_sharpe"],
            portfolio_return_delta=metrics["portfolio_return"] - benchmark["portfolio_return"],
        )[
            [
                "model",
                "benchmark",
                "rank_ic_delta",
                "sharpe_delta",
                "portfolio_return_delta",
            ]
        ]
        .sort_values("sharpe_delta", ascending=False)
        .reset_index(drop=True)
    )


def _portfolio_summary(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    return (
        portfolio_returns.assign(
            cost_drag=portfolio_returns["gross_return"] - portfolio_returns["net_return"]
        )
        .groupby("model", as_index=False)
        .agg(
            periods=("date", "nunique"),
            avg_gross_return=("gross_return", "mean"),
            avg_net_return=("net_return", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_cost_drag=("cost_drag", "mean"),
        )
    )


def _tail_risk_summary(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model, frame in portfolio_returns.sort_values("date").groupby("model"):
        returns = frame["net_return"]
        curve = (1.0 + returns).cumprod()
        drawdown = curve / curve.cummax() - 1.0
        in_drawdown = drawdown < 0
        rows.append(
            {
                "model": model,
                "worst_period_net_return": float(returns.min()),
                "net_return_volatility": float(returns.std(ddof=0)),
                "positive_period_rate": float((returns > 0).mean()),
                "max_drawdown": float(drawdown.min()),
                "max_drawdown_periods": _longest_true_run(in_drawdown),
            }
        )
    return pd.DataFrame(rows).sort_values("max_drawdown").reset_index(drop=True)


def _validation_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_dates = pd.to_datetime(predictions["date"])
    assets_per_date = predictions.groupby("date")["asset"].nunique()
    return pd.DataFrame(
        [
            {
                "folds": int(predictions["fold"].nunique()),
                "start_date": prediction_dates.min().date().isoformat(),
                "end_date": prediction_dates.max().date().isoformat(),
                "prediction_rows": len(predictions),
                "unique_assets": int(predictions["asset"].nunique()),
                "avg_assets_per_date": float(assets_per_date.mean()),
                "models": ", ".join(model_columns(predictions)),
            }
        ]
    )


def _prediction_coverage_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    assets_per_date = predictions.groupby("date")["asset"].nunique()
    total_rows = len(predictions)

    for model in model_columns(predictions):
        scores = predictions[model]
        rows.append(
            {
                "model": model,
                "prediction_rows": total_rows,
                "dates": int(predictions["date"].nunique()),
                "min_assets_per_date": int(assets_per_date.min()),
                "avg_assets_per_date": float(assets_per_date.mean()),
                "missing_score_rate": float(scores.isna().mean()),
                "unique_score_rate": float(scores.nunique(dropna=True) / total_rows),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["missing_score_rate", "unique_score_rate"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _fold_summary(predictions: pd.DataFrame, portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    fold_dates = predictions.loc[:, ["fold", "date"]].drop_duplicates()
    returns_by_fold = portfolio_returns.merge(
        fold_dates, on="date", how="left", validate="many_to_one"
    )
    rows: list[dict[str, float | int | str]] = []
    for model in model_columns(predictions):
        model_returns = returns_by_fold.loc[returns_by_fold["model"] == model]
        for fold, frame in predictions.groupby("fold"):
            fold_returns = model_returns.loc[model_returns["fold"] == fold, "net_return"]
            rows.append(
                {
                    "model": model,
                    "fold": int(fold),
                    "start_date": pd.to_datetime(frame["date"]).min().date().isoformat(),
                    "end_date": pd.to_datetime(frame["date"]).max().date().isoformat(),
                    "rank_ic": _safe_rank_ic(frame, model),
                    "net_return": float(fold_returns.sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(drop=True)


def _fold_robustness_summary(fold_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_summary.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            avg_fold_rank_ic=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
            positive_fold_rate=("net_return", lambda values: float((values > 0).mean())),
            worst_fold_net_return=("net_return", "min"),
            best_fold_net_return=("net_return", "max"),
        )
        .sort_values(["positive_fold_rate", "avg_fold_rank_ic"], ascending=False)
        .reset_index(drop=True)
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_value(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", r"\|")


def _safe_rank_ic(frame: pd.DataFrame, model: str) -> float:
    rank_ic = frame.groupby("date").apply(
        lambda date_frame: _safe_corr(date_frame[model].rank(), date_frame["label"].rank())
    )
    return float(rank_ic.mean())


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right))


def _longest_true_run(values: pd.Series) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
