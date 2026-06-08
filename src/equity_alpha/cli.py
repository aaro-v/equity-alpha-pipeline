from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from equity_alpha.data import SampleDataConfig, generate_sample_panel
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
def sample_data(
    output: Annotated[Path, typer.Option(help="Output parquet path.")] = Path(
        "data/cache/sample_panel.parquet"
    ),
    assets: Annotated[int, typer.Option(min=10, help="Number of synthetic assets.")] = 90,
    days: Annotated[int, typer.Option(min=180, help="Number of business days.")] = 640,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = 7,
) -> None:
    """Generate deterministic synthetic point-in-time sample data."""

    panel = generate_sample_panel(SampleDataConfig(n_assets=assets, n_days=days, seed=seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    typer.echo(f"Wrote {len(panel):,} rows to {output}")


@app.command()
def run(
    output_dir: Annotated[Path, typer.Option(help="Artifact output directory.")] = Path(
        "reports/generated"
    ),
    assets: Annotated[int, typer.Option(min=20, help="Number of synthetic assets.")] = 90,
    days: Annotated[int, typer.Option(min=180, help="Number of business days.")] = 640,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = 7,
    horizon: Annotated[
        int, typer.Option(min=1, help="Forward-return horizon in business days.")
    ] = 21,
    cost_bps: Annotated[
        float, typer.Option(min=0.0, help="One-way transaction cost in basis points.")
    ] = 5.0,
    train_days: Annotated[
        int, typer.Option(min=20, help="Training window length in business days.")
    ] = 252,
    test_days: Annotated[
        int, typer.Option(min=5, help="Test window length in business days.")
    ] = 63,
    purge_days: Annotated[
        int, typer.Option(min=0, help="Gap between each training and test window.")
    ] = 21,
    embargo_days: Annotated[
        int, typer.Option(min=0, help="Gap after each test window before the next fold.")
    ] = 5,
    include_neural: Annotated[
        bool, typer.Option(help="Fit the small PyTorch neural baseline.")
    ] = True,
    neural_epochs: Annotated[
        int, typer.Option(min=1, help="Training epochs for the neural baseline.")
    ] = 25,
) -> None:
    """Run the full deterministic cross-sectional alpha research pipeline."""

    config = PipelineConfig(
        sample_data=SampleDataConfig(n_assets=assets, n_days=days, seed=seed),
        walk_forward=WalkForwardConfig(
            train_days=train_days,
            test_days=test_days,
            purge_days=purge_days,
            embargo_days=embargo_days,
        ),
        models=ModelConfig(include_neural=include_neural, neural_epochs=neural_epochs),
        horizon=horizon,
        cost_bps=cost_bps,
    )
    result = run_research_pipeline(config, output_dir=output_dir)
    typer.echo(result.metrics.to_string(index=False))
    if result.report_path is not None:
        typer.echo(f"\nReport: {result.report_path}")


if __name__ == "__main__":
    app()
