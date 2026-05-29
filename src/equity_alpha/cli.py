from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from equity_alpha.data import SampleDataConfig, generate_sample_panel
from equity_alpha.models import ModelConfig
from equity_alpha.pipeline import PipelineConfig, run_research_pipeline
from equity_alpha.validation import WalkForwardConfig

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
    days: Annotated[int, typer.Option(min=500, help="Number of business days.")] = 640,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = 7,
    include_neural: Annotated[
        bool, typer.Option(help="Fit the small PyTorch neural baseline.")
    ] = True,
) -> None:
    """Run the full deterministic cross-sectional alpha research pipeline."""

    config = PipelineConfig(
        sample_data=SampleDataConfig(n_assets=assets, n_days=days, seed=seed),
        walk_forward=WalkForwardConfig(),
        models=ModelConfig(include_neural=include_neural),
    )
    result = run_research_pipeline(config, output_dir=output_dir)
    typer.echo(result.metrics.to_string(index=False))
    if result.report_path is not None:
        typer.echo(f"\nReport: {result.report_path}")
