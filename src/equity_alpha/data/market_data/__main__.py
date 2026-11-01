"""Command-line entry point: ``python -m market_data --out data/real``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real-data equity panel.")
    parser.add_argument("--out", type=Path, default=Path("data/real"), help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Only build the first N companies.")
    args = parser.parse_args()
    build(args.out, limit=args.limit)


if __name__ == "__main__":
    main()
