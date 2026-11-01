from pathlib import Path

import pandas as pd

from equity_alpha.data import SampleDataConfig
from equity_alpha.workflow.pipeline import PipelineConfig, _metadata


def test_loaded_panel_metadata_omits_unused_generator_settings() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-01", periods=6).repeat(2),
            "asset": ["A", "B"] * 6,
        }
    )
    config = PipelineConfig(
        panel_path=Path("data/example/panel.parquet"),
        sample_data=SampleDataConfig(n_assets=999, n_days=999),
    )
    metadata = _metadata(config, panel, panel.tail(4))

    assert "sample_data" not in metadata["parameters"]
    assert metadata["panel_trading_days"] == 6
    assert metadata["panel_assets"] == 2
    assert metadata["data_source"] == "data/example/panel.parquet"
    assert metadata["parameters"]["panel_path"] == metadata["data_source"]


def test_generated_panel_metadata_keeps_active_generator_settings() -> None:
    sample = SampleDataConfig(n_assets=20, n_days=300, seed=42)
    panel = pd.DataFrame({"date": pd.bdate_range(sample.start, periods=300), "asset": "A"})
    metadata = _metadata(PipelineConfig(sample_data=sample), panel, panel.tail(10))

    assert metadata["parameters"]["sample_data"] == {
        "n_assets": 20,
        "n_days": 300,
        "start": sample.start,
        "seed": 42,
    }
    assert metadata["panel_trading_days"] == 300
    assert metadata["data_source"] == "synthetic (generated)"
