"""Regenerate report figures from committed aggregate tables."""

import argparse
from pathlib import Path

from equity_alpha.reporting.artifacts import render_figures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    for directory in parser.parse_args().directories:
        render_figures(directory)
