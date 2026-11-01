from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

from equity_alpha.portfolio.metrics import drawdown_series

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_artifacts(
    output_dir: Path,
    metrics: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    sector_attribution: pd.DataFrame,
    correlations: pd.DataFrame,
    factor_decay: pd.DataFrame,
    factor_spreads: pd.DataFrame,
    coverage: pd.DataFrame,
    coverage_by_date: pd.DataFrame,
    folds: pd.DataFrame,
    metadata: dict,
    attribution: pd.DataFrame,
    loadings: pd.DataFrame,
    neural_training: pd.DataFrame,
) -> Path:
    """Write aggregate research results; never persist asset-level vendor observations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon = metadata["parameters"]["horizon"]
    curves = portfolio_returns.copy().sort_values(["variant", "model", "date"])
    groups = curves.groupby(["variant", "model"])["net_return"]
    curves["wealth"] = groups.transform(lambda r: (1 + r).cumprod())
    curves["drawdown"] = groups.transform(drawdown_series)
    if neural_training.empty:
        neural_training = pd.DataFrame(
            columns=["variant", "fold", "epochs_run", "best_epoch", "optimizer_updates"]
        )
    tables = {
        "metrics": metrics,
        "portfolio_returns": curves,
        "ic_series": correlations,
        "factor_ics": factor_decay.loc[factor_decay["horizon"].eq(horizon)],
        "factor_decay": factor_decay,
        "quintile_spreads": factor_spreads,
        "turnover": curves[
            ["date", "variant", "model", "turnover", "rebalance", "rebalance_size_exposure"]
        ],
        "drawdown": curves[["date", "variant", "model", "drawdown"]],
        "sector_attribution": sector_attribution,
        "feature_coverage": coverage,
        "feature_coverage_by_date": coverage_by_date,
        "fold_summary": folds,
        "factor_attribution": attribution,
        "factor_loadings": loadings,
        "neural_training": neural_training,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.10g")
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    _plot_curves(curves, output_dir)
    _plot_ic(correlations, output_dir)
    _plot_decay(factor_decay, output_dir)
    _plot_size_controls(curves, output_dir)
    report = output_dir / "research_report.md"
    report.write_text(
        "# Walk-forward research results\n\n"
        f"Source: `{metadata['data_source']}`. "
        f"Test signals: {metadata['test_start']} through {metadata['test_end']}.\n\n"
        "The main portfolio executes at the next close. IC summaries use non-overlapping "
        "signal dates; Sharpe uses daily net returns. Its 95% percentile intervals use "
        "moving blocks and are pointwise. "
        f"{metadata['multiple_comparisons']['forecasts']} forecasts and seven factors are tested, "
        "plus the execution and size-control variants; intervals are not adjusted for "
        "multiple comparisons. Intervals require at least two blocks of daily observations. "
        "The book holds shares between rebalances and pays costs on actual traded dollars, "
        "including entry and final liquidation. Sector contributions sum arithmetic daily "
        "net returns and do not add to compounded total return.\n\n"
        + markdown_table(
            metrics,
            [
                "model",
                "variant",
                "mean_rank_ic",
                "cost_adjusted_sharpe",
                "sharpe_ci_lower",
                "sharpe_ci_upper",
                "portfolio_return",
                "max_drawdown",
                "annual_one_way_turnover_per_side",
            ],
        )
        + "\n\n![Cumulative net return](equity_curve.png)\n\n"
        "![Annual rank IC](ic_by_year.png)\n\n"
        "![Factor IC decay](factor_decay.png)\n\n"
        "![Size controls](size_controls.png)\n",
        encoding="utf-8",
    )
    return report


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
        )
    return "\n".join(lines)


def _plot_curves(curves: pd.DataFrame, out: Path) -> None:
    curves = curves.loc[curves["variant"].eq("main")]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for model, frame in curves.groupby("model", sort=False):
        ax.plot(frame["date"], frame["wealth"] - 1, label=model, linewidth=1.4)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(
        xlabel="Date", ylabel="Cumulative net return", title="Out-of-sample long-short quintiles"
    )
    ax.legend(frameon=False, ncol=4)
    ax.grid(alpha=0.15)
    fig.savefig(out / "equity_curve.png", dpi=150)
    plt.close(fig)


def _plot_ic(correlations: pd.DataFrame, out: Path) -> None:
    annual = correlations.assign(year=pd.to_datetime(correlations["date"]).dt.year)
    annual = annual.groupby(["year", "model"])["rank_ic"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    annual.plot.bar(ax=ax)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(
        xlabel="Signal year",
        ylabel="Mean daily cross-sectional rank IC",
        title="Rank IC by year (overlapping targets; descriptive only)",
    )
    ax.legend(frameon=False, ncol=4)
    fig.savefig(out / "ic_by_year.png", dpi=150)
    plt.close(fig)


def _plot_decay(decay: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for factor, frame in decay.loc[decay["variant"].eq("neutral_rank")].groupby("factor"):
        frame = frame.sort_values("horizon")
        ax.plot(frame["horizon"], frame["mean_rank_ic"], marker="o", label=factor)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(
        xlabel="Forward horizon (trading days)",
        ylabel="Mean rank IC",
        title="Factor IC decay (non-overlapping signal samples)",
    )
    ax.set_xticks(sorted(decay["horizon"].unique()))
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.15)
    fig.savefig(out / "factor_decay.png", dpi=150)
    plt.close(fig)


def render_figures(output_dir: Path) -> None:
    """Regenerate report figures from aggregate tables without vendor access."""
    curves = pd.read_csv(output_dir / "portfolio_returns.csv", parse_dates=["date"])
    correlations = pd.read_csv(output_dir / "ic_series.csv", parse_dates=["date"])
    _plot_curves(curves, output_dir)
    _plot_ic(correlations, output_dir)
    _plot_decay(pd.read_csv(output_dir / "factor_decay.csv"), output_dir)
    _plot_size_controls(curves, output_dir)


def _plot_size_controls(curves: pd.DataFrame, out: Path) -> None:
    labels = {
        "main": "Next close",
        "size_neutral_features": "Size-neutral features",
        "cap_neutral_book": "Cap-neutral book",
        "size_neutral": "Both size controls",
    }
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ridge = curves.loc[curves["model"].eq("ridge")]
    for variant, label in labels.items():
        frame = ridge.loc[ridge["variant"].eq(variant)].sort_values("date")
        if not frame.empty:
            ax.plot(frame["date"], frame["wealth"] - 1, label=label)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set(xlabel="Date", ylabel="Cumulative net return", title="Ridge: size controls")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.15)
    fig.savefig(out / "size_controls.png", dpi=150)
    plt.close(fig)
