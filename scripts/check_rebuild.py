"""Compare a fresh private rebuild with a private snapshot; write aggregate differences only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(reference: Path, rebuilt: Path, out: Path) -> None:
    original = pd.read_parquet(reference / "panel.parquet")
    candidate = pd.read_parquet(rebuilt / "panel.parquet")
    for frame in [original, candidate]:
        frame["asset"] = frame["asset"].astype(str)
    common = original.merge(candidate, on=["date", "asset"], suffixes=("_reference", "_rebuilt"))
    fields = original.select_dtypes(include="number").columns
    comparisons = {}
    for field in fields:
        left, right = common[f"{field}_reference"], common[f"{field}_rebuilt"]
        difference = (left - right).abs().dropna()
        comparisons[field] = {
            "common_finite_rows": len(difference),
            "max_absolute_difference": float(difference.max()) if len(difference) else None,
            "mean_absolute_difference": float(difference.mean()) if len(difference) else None,
            "missingness_disagreements": int((left.isna() != right.isna()).sum()),
            "share_close_rtol_1e_4": float(
                np.isclose(left, right, rtol=1e-4, atol=1e-7, equal_nan=True).mean()
            ),
        }
    universe = pd.read_csv(rebuilt / "universe.csv")
    result = {
        "reference": str(reference),
        "rebuilt": str(rebuilt),
        "reference_rows": len(original),
        "rebuilt_rows": len(candidate),
        "common_rows": len(common),
        "common_assets": common["asset"].nunique(),
        "reference_assets": original["asset"].nunique(),
        "rebuilt_assets": candidate["asset"].nunique(),
        "successful_downloads": int(universe["status"].eq("ok").sum()),
        "skipped_downloads": int(universe["status"].ne("ok").sum()),
        "columns": comparisons,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.reference, args.rebuilt, args.out)
