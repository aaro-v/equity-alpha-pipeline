from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the installed package version."""
    from equity_alpha import __version__

    typer.echo(__version__)
