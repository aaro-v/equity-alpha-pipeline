from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from equity_alpha.portfolio.metrics import drawdown_series
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
    _write_manifest(
        output_dir / "manifest.json",
        artifacts=[
            ("metrics.csv", metrics),
            ("portfolio_returns.csv", portfolio_returns),
            ("sector_attribution.csv", sector_attribution),
            ("predictions.parquet", predictions),
            ("equity_curve.png", None),
            ("research_report.md", None),
        ],
    )
    return report_path


def _write_manifest(
    path: Path, artifacts: list[tuple[str, pd.DataFrame | None]]
) -> None:
    manifest = {
        "artifact_format": 1,
        "artifacts": [
            {
                "path": artifact_path.name,
                "bytes": artifact_path.stat().st_size,
                "sha256": _sha256_file(artifact_path),
                **(
                    {
                        "rows": len(frame),
                        "columns": list(frame.columns),
                    }
                    if frame is not None
                    else {}
                ),
            }
            for name, frame in artifacts
            if (artifact_path := path.parent / name).exists()
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "## IC Stability",
            "",
            _markdown_table(_ic_stability_summary(predictions)),
            "",
            "## Validation Summary",
            "",
            _markdown_table(_validation_summary(predictions)),
            "",
            "## Prediction Coverage",
            "",
            _markdown_table(_prediction_coverage_summary(predictions)),
            "",
            "## Score Dispersion Diagnostics",
            "",
            _markdown_table(_score_dispersion_summary(predictions)),
            "",
            "## Signal Spread Diagnostics",
            "",
            _markdown_table(_signal_spread_summary(predictions)),
            "",
            "## Fold Breadth",
            "",
            _markdown_table(_fold_breadth_summary(predictions)),
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
            "## Turnover Stability",
            "",
            _markdown_table(_turnover_stability_summary(portfolio_returns)),
            "",
            "## Return Consistency",
            "",
            _markdown_table(_return_consistency_summary(portfolio_returns)),
            "",
            "## Tail Risk Diagnostics",
            "",
            _markdown_table(_tail_risk_summary(portfolio_returns)),
            "",
            "## Underwater Diagnostics",
            "",
            _markdown_table(_underwater_summary(portfolio_returns)),
            "",
            "## Book Exposure Diagnostics",
            "",
            _markdown_table(_book_exposure_summary(predictions)),
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
    gross_exposure = portfolio_returns.get(
        "gross_exposure", pd.Series(float("nan"), index=portfolio_returns.index)
    )
    net_exposure = portfolio_returns.get(
        "net_exposure", pd.Series(float("nan"), index=portfolio_returns.index)
    )
    frame = portfolio_returns.assign(
        cost_drag=portfolio_returns["gross_return"] - portfolio_returns["net_return"],
        gross_exposure=gross_exposure,
        abs_net_exposure=net_exposure.abs(),
    )
    return (
        frame
        .groupby("model", as_index=False)
        .agg(
            periods=("date", "nunique"),
            avg_gross_return=("gross_return", "mean"),
            avg_net_return=("net_return", "mean"),
            avg_gross_exposure=("gross_exposure", "mean"),
            avg_abs_net_exposure=("abs_net_exposure", "mean"),
            avg_turnover=("turnover", "mean"),
            avg_cost_drag=("cost_drag", "mean"),
        )
    )


def _ic_stability_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model in model_columns(predictions):
        daily = predictions.groupby("date").apply(
            lambda frame, col=model: pd.Series(
                {
                    "ic": _safe_corr(frame[col], frame["label"]),
                    "rank_ic": _safe_corr(frame[col].rank(), frame["label"].rank()),
                }
            )
        )
        ic = daily["ic"].dropna()
        rank_ic = daily["rank_ic"].dropna()
        rows.append(
            {
                "model": model,
                "dates": int(daily.index.nunique()),
                "valid_ic_dates": int(ic.count()),
                "avg_ic": float(ic.mean()),
                "ic_volatility": float(ic.std(ddof=0)),
                "positive_ic_rate": float((ic > 0).mean()),
                "avg_rank_ic": float(rank_ic.mean()),
                "rank_ic_volatility": float(rank_ic.std(ddof=0)),
                "positive_rank_ic_rate": float((rank_ic > 0).mean()),
                "worst_rank_ic": float(rank_ic.min()),
                "best_rank_ic": float(rank_ic.max()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["avg_rank_ic", "positive_rank_ic_rate"], ascending=False)
        .reset_index(drop=True)
    )


def _turnover_stability_summary(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model, frame in portfolio_returns.sort_values("date").groupby("model"):
        turnover = frame["turnover"].astype(float)
        rows.append(
            {
                "model": model,
                "periods": int(frame["date"].nunique()),
                "avg_turnover": float(turnover.mean()),
                "turnover_volatility": float(turnover.std(ddof=0)),
                "p95_turnover": float(turnover.quantile(0.95)),
                "max_turnover": float(turnover.max()),
                "high_turnover_rate": float((turnover > turnover.mean()).mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["avg_turnover", "turnover_volatility"], ascending=False)
        .reset_index(drop=True)
    )


def _return_consistency_summary(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model, frame in portfolio_returns.sort_values("date").groupby("model"):
        returns = frame["net_return"]
        curve = (1.0 + returns).cumprod()
        rows.append(
            {
                "model": model,
                "periods": int(frame["date"].nunique()),
                "mean_net_return": float(returns.mean()),
                "median_net_return": float(returns.median()),
                "cumulative_return": float(curve.iloc[-1] - 1.0),
                "return_skew": float(returns.skew()),
                "positive_period_rate": float((returns > 0).mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["cumulative_return", "mean_net_return"], ascending=False)
        .reset_index(drop=True)
    )


def _tail_risk_summary(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model, frame in portfolio_returns.sort_values("date").groupby("model"):
        returns = frame["net_return"]
        drawdown = drawdown_series(returns)
        in_drawdown = drawdown < 0
        net_return_var_5 = float(returns.quantile(0.05))
        rows.append(
            {
                "model": model,
                "worst_period_net_return": float(returns.min()),
                "net_return_var_5": net_return_var_5,
                "net_return_expected_shortfall_5": float(
                    returns.loc[returns <= net_return_var_5].mean()
                ),
                "net_return_volatility": float(returns.std(ddof=0)),
                "positive_period_rate": float((returns > 0).mean()),
                "max_drawdown": float(drawdown.min()),
                "max_drawdown_periods": _longest_true_run(in_drawdown),
            }
        )
    return pd.DataFrame(rows).sort_values("max_drawdown").reset_index(drop=True)


def _underwater_summary(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, bool | float | int | str]] = []
    for model, frame in portfolio_returns.sort_values("date").groupby("model"):
        returns = frame["net_return"].astype(float)
        dates = pd.to_datetime(frame["date"]).reset_index(drop=True)
        wealth = (1.0 + returns.reset_index(drop=True)).cumprod()
        running_peak = wealth.cummax().clip(lower=1.0)
        drawdown = wealth / running_peak - 1.0

        trough_position = int(drawdown.to_numpy().argmin())
        max_drawdown = float(drawdown.iloc[trough_position])
        peak_value = float(running_peak.iloc[trough_position])
        peak_candidates = wealth.iloc[: trough_position + 1][
            wealth.iloc[: trough_position + 1] >= peak_value
        ]
        peak_position = (
            int(peak_candidates.index[-1]) if not peak_candidates.empty else trough_position
        )
        recovered_after_trough = bool(
            (wealth.iloc[trough_position + 1 :] >= peak_value).any()
        )

        rows.append(
            {
                "model": model,
                "periods": int(frame["date"].nunique()),
                "ending_growth": float(wealth.iloc[-1]),
                "peak_growth": float(running_peak.max()),
                "underwater_period_rate": float((drawdown < 0).mean()),
                "max_drawdown": max_drawdown,
                "max_drawdown_start": dates.iloc[peak_position].date().isoformat(),
                "max_drawdown_trough": dates.iloc[trough_position].date().isoformat(),
                "recovered_after_trough": recovered_after_trough,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["underwater_period_rate", "max_drawdown"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _book_exposure_summary(predictions: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model in model_columns(predictions):
        for side, selector in [("long", "nlargest"), ("short", "nsmallest")]:
            selected = []
            for _, frame in predictions.groupby("date"):
                n_names = max(1, int(len(frame) * quantile))
                selected.append(getattr(frame, selector)(n_names, model))
            book = pd.concat(selected, ignore_index=True)
            sector_weights = book["sector"].value_counts(normalize=True)
            rows.append(
                {
                    "model": model,
                    "side": side,
                    "avg_names_per_date": float(len(book) / predictions["date"].nunique()),
                    "top_sector": str(sector_weights.index[0]),
                    "top_sector_weight": float(sector_weights.iloc[0]),
                    "avg_label": float(book["label"].mean()),
                    "avg_future_return": float(book["future_return"].mean()),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["model", "side"], ascending=[True, False])
        .reset_index(drop=True)
    )


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


def _score_dispersion_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for model in model_columns(predictions):
        scores = predictions[model].dropna().astype(float)
        daily = (
            predictions.dropna(subset=[model])
            .groupby("date")[model]
            .agg(
                daily_score_std=lambda values: values.astype(float).std(ddof=0),
                daily_unique_scores="nunique",
                daily_rows="count",
            )
        )
        if daily.empty:
            rows.append(
                {
                    "model": model,
                    "dates": 0,
                    "score_mean": float("nan"),
                    "score_std": float("nan"),
                    "avg_daily_score_std": float("nan"),
                    "min_daily_unique_scores": 0,
                    "avg_daily_unique_score_rate": float("nan"),
                }
            )
            continue

        daily_unique_rate = daily["daily_unique_scores"] / daily["daily_rows"]
        rows.append(
            {
                "model": model,
                "dates": int(daily.index.nunique()),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std(ddof=0)),
                "avg_daily_score_std": float(daily["daily_score_std"].mean()),
                "min_daily_unique_scores": int(daily["daily_unique_scores"].min()),
                "avg_daily_unique_score_rate": float(daily_unique_rate.mean()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["avg_daily_unique_score_rate", "avg_daily_score_std"], ascending=False)
        .reset_index(drop=True)
    )


def _signal_spread_summary(predictions: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model in model_columns(predictions):
        spreads = []
        for date, frame in predictions.groupby("date"):
            valid = frame.dropna(subset=[model])
            if len(valid) < 2:
                continue
            n_names = max(1, int(len(valid) * quantile))
            longs = valid.nlargest(n_names, model)
            shorts = valid.nsmallest(n_names, model)
            spreads.append(
                {
                    "date": date,
                    "long_label": float(longs["label"].mean()),
                    "short_label": float(shorts["label"].mean()),
                    "long_future_return": float(longs["future_return"].mean()),
                    "short_future_return": float(shorts["future_return"].mean()),
                }
            )

        spread_frame = pd.DataFrame(spreads)
        if spread_frame.empty:
            rows.append(
                {
                    "model": model,
                    "dates": 0,
                    "long_avg_label": float("nan"),
                    "short_avg_label": float("nan"),
                    "label_spread": float("nan"),
                    "long_avg_future_return": float("nan"),
                    "short_avg_future_return": float("nan"),
                    "future_return_spread": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "model": model,
                "dates": int(spread_frame["date"].nunique()),
                "long_avg_label": float(spread_frame["long_label"].mean()),
                "short_avg_label": float(spread_frame["short_label"].mean()),
                "label_spread": float(
                    (spread_frame["long_label"] - spread_frame["short_label"]).mean()
                ),
                "long_avg_future_return": float(spread_frame["long_future_return"].mean()),
                "short_avg_future_return": float(spread_frame["short_future_return"].mean()),
                "future_return_spread": float(
                    (
                        spread_frame["long_future_return"]
                        - spread_frame["short_future_return"]
                    ).mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["future_return_spread", "label_spread"], ascending=False)
        .reset_index(drop=True)
    )


def _fold_breadth_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    model_count = len(model_columns(predictions))
    for fold, frame in predictions.groupby("fold"):
        dates = pd.to_datetime(frame["date"])
        assets_per_date = frame.groupby("date")["asset"].nunique()
        rows.append(
            {
                "fold": int(fold),
                "start_date": dates.min().date().isoformat(),
                "end_date": dates.max().date().isoformat(),
                "dates": int(frame["date"].nunique()),
                "prediction_rows": len(frame),
                "unique_assets": int(frame["asset"].nunique()),
                "min_assets_per_date": int(assets_per_date.min()),
                "avg_assets_per_date": float(assets_per_date.mean()),
                "model_count": model_count,
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


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
