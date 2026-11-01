from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from equity_alpha.data.synthetic import SampleDataConfig, generate_sample_tables, write_sample_data


def test_sample_is_deterministic_and_filings_are_available() -> None:
    config = SampleDataConfig(n_assets=18, n_days=220, seed=11)
    tables = generate_sample_tables(config)
    for name, frame in generate_sample_tables(config).items():
        pd.testing.assert_frame_equal(frame, tables[name])
    panel = tables["panel.parquet"]
    assert (panel["fundamentals_available_date"] <= panel["date"]).all()
    assert panel.loc[panel["sector"].eq("Utilities"), "gross_margin"].isna().all()
    industrials = panel.loc[panel["sector"].eq("Industrials")]
    assert industrials.groupby("asset", observed=True)["gross_margin"].count().eq(0).any()
    assert panel["market_cap"].median() > 1e9
    assert panel["dollar_volume"].median() > 1e7
    filings = tables["fundamentals_quarterly.parquet"]
    assert (filings["filed"] > filings["period_end"]).all()
    assert (filings["filed"] - filings["period_end"]).dt.days.median() >= 25
    changes = panel.groupby("asset", observed=True)["fundamentals_available_date"].nunique()
    assert changes.max() <= 5


def _schema(directory: Path) -> dict:
    result = {}
    for path in directory.glob("*.parquet"):
        result[path.name] = {
            field.name: "category" if pa.types.is_dictionary(field.type) else str(field.type)
            for field in pq.read_schema(path)
        }
    result["universe.csv"] = list(pd.read_csv(directory / "universe.csv").columns)
    return result


def test_sample_matches_committed_schema(tmp_path: Path) -> None:
    write_sample_data(tmp_path, SampleDataConfig(n_assets=18, n_days=180))
    expected = json.loads(Path("src/equity_alpha/data/schema.json").read_text())
    assert _schema(tmp_path) == expected


@pytest.mark.skipif(not Path("data/real/panel.parquet").exists(), reason="private data absent")
def test_real_matches_committed_schema() -> None:
    expected = json.loads(Path("src/equity_alpha/data/schema.json").read_text())
    assert _schema(Path("data/real")) == expected
