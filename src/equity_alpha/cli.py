from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from equity_alpha.data import SampleDataConfig
from equity_alpha.data.synthetic import write_sample_data
from equity_alpha.research.models import ModelConfig
from equity_alpha.validation import WalkForwardConfig
from equity_alpha.workflow.pipeline import PipelineConfig, run_research_pipeline

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the installed package version."""
    from equity_alpha import __version__

    typer.echo(__version__)


@app.command()
def build_data(
    out: Annotated[Path, typer.Option(help="Private data output directory.")] = Path("data/real"),
    limit: Annotated[
        int | None, typer.Option(min=1, help="Limit universe for a spot check.")
    ] = None,
) -> None:
    """Rebuild real market data from Yahoo, SEC EDGAR and Wikipedia (requires [data])."""
    from equity_alpha.data.market_data.build import build

    build(out, limit=limit)


@app.command()
def sample_data(
    out: Annotated[Path, typer.Option(help="Output directory for the four sample files.")] = Path(
        "data/synthetic"
    ),
    assets: Annotated[int, typer.Option(min=10)] = 100,
    days: Annotated[int, typer.Option(min=180)] = 756,
    seed: Annotated[int, typer.Option()] = 7,
) -> None:
    """Generate a deterministic synthetic dataset with the real builder's schema."""
    write_sample_data(out, SampleDataConfig(n_assets=assets, n_days=days, seed=seed))
    typer.echo(f"Wrote synthetic dataset to {out}")


@app.command()
def profile_data(
    panel: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option()] = Path("reports/synthetic/data_profile.json"),
) -> None:
    """Save aggregate market and fundamentals distributions."""
    from equity_alpha.data.profile import write_profile

    write_profile(panel, out)
    typer.echo(f"Profile: {out}")


@app.command()
def run(
    output_dir: Annotated[Path, typer.Option(help="Aggregate report output directory.")] = Path(
        "reports/synthetic"
    ),
    panel: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    assets: Annotated[int, typer.Option(min=20)] = 100,
    days: Annotated[int, typer.Option(min=180)] = 756,
    seed: Annotated[int, typer.Option()] = 7,
    horizon: Annotated[int, typer.Option(min=1)] = 21,
    cost_bps: Annotated[float, typer.Option(min=0.0)] = 5.0,
    train_days: Annotated[int, typer.Option(min=20)] = 504,
    test_days: Annotated[int, typer.Option(min=5)] = 63,
    purge_days: Annotated[int, typer.Option(min=0)] = 21,
    embargo_days: Annotated[int, typer.Option(min=0)] = 0,
    include_neural: Annotated[bool, typer.Option()] = True,
    neural_epochs: Annotated[int, typer.Option(min=1)] = 25,
) -> None:
    """Run purged walk-forward forecasts and daily marked-to-market portfolios."""
    if purge_days < horizon:
        raise typer.BadParameter("purge_days must be at least horizon", param_hint="--purge-days")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = PipelineConfig(
        sample_data=SampleDataConfig(n_assets=assets, n_days=days, seed=seed),
        walk_forward=WalkForwardConfig(train_days, test_days, purge_days, embargo_days),
        models=ModelConfig(include_neural=include_neural, neural_epochs=neural_epochs),
        horizon=horizon,
        cost_bps=cost_bps,
        panel_path=panel,
    )
    result = run_research_pipeline(config, output_dir=output_dir)
    typer.echo(result.metrics.to_string(index=False))
    typer.echo(f"Report: {result.report_path}")


if __name__ == "__main__":
    app()
