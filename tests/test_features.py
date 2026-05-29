from __future__ import annotations

from equity_alpha.data import SampleDataConfig, generate_sample_panel
from equity_alpha.features import FEATURE_COLUMNS, build_research_frame


def test_research_frame_has_neutral_rank_features_and_labels() -> None:
    panel = generate_sample_panel(SampleDataConfig(n_assets=30, n_days=220, seed=5))

    research = build_research_frame(panel, horizon=21)

    assert set(FEATURE_COLUMNS).issubset(research.columns)
    assert research["label"].notna().all()
    assert research.groupby("date")["asset"].nunique().min() >= 20
    assert research[list(FEATURE_COLUMNS)].max().max() <= 0.5
    assert research[list(FEATURE_COLUMNS)].min().min() >= -0.5


def test_labels_use_future_returns_without_overlapping_last_horizon() -> None:
    panel = generate_sample_panel(SampleDataConfig(n_assets=20, n_days=180, seed=13))

    research = build_research_frame(panel, horizon=15)

    assert research["date"].max() < panel["date"].max()
    assert research.groupby("asset")["date"].max().nunique() == 1
