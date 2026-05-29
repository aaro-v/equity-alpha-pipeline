from __future__ import annotations

from pathlib import Path

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
    output: Path = typer.Option(Path("data/cache/sample_panel.parquet"), help="Output parquet path."),
    assets: int = typer.Option(90, min=10, help="Number of synthetic assets."),
    days: int = typer.Option(640, min=180, help="Number of business days."),
    seed: int = typer.Option(7, help="Deterministic random seed."),
) -> None:
    """Generate deterministic synthetic point-in-time sample data."""

    panel = generate_sample_panel(SampleDataConfig(n_assets=assets, n_days=days, seed=seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    typer.echo(f"Wrote {len(panel):,} rows to {output}")


@app.command()
def run(
    output_dir: Path = typer.Option(Path("reports/generated"), help="Artifact output directory."),
    assets: int = typer.Option(90, min=20, help="Number of synthetic assets."),
    days: int = typer.Option(640, min=500, help="Number of business days."),
    seed: int = typer.Option(7, help="Deterministic random seed."),
    include_neural: bool = typer.Option(True, help="Fit the small PyTorch neural baseline."),
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
