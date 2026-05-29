from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    report_path.write_text(_render_report(metrics, sector_attribution, chart_path.name), encoding="utf-8")
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


def _render_report(metrics: pd.DataFrame, sector_attribution: pd.DataFrame, chart_name: str) -> str:
    best = metrics.iloc[0]
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
            metrics.to_markdown(index=False, floatfmt=".4f"),
            "",
            f"Best sample model by cost-adjusted Sharpe: **{best['model']}**.",
            "",
            "## Top Sector Rank ICs",
            "",
            sector_summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Equity Curve",
            "",
            f"![Equity curve]({chart_name})",
            "",
            "## Method Notes",
            "",
            "- Features are built only after synthetic availability dates are reached.",
            "- Forward labels use future excess returns and are excluded from feature construction.",
            "- Walk-forward splits use a purge gap before each test window and an embargo step.",
            "- Long-short returns include turnover-based transaction costs.",
            "",
        ]
    )
