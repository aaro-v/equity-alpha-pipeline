from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_alpha.portfolio.construction import market_cap_neutral_weights
from equity_alpha.portfolio.uncertainty import moving_block_sharpe_interval
from equity_alpha.research.attribution import factor_attribution
from equity_alpha.research.features import FEATURE_COLUMNS, RAW_FEATURES, _size_neutralize
from equity_alpha.research.models import ModelConfig, _neural_predictions, neural_split


def test_cap_hedge_is_dollar_neutral_size_neutral_and_preserves_gross() -> None:
    weights = pd.Series([0.5, 0.5, 0, 0, 0, 0, -0.5, -0.5])
    cap = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
    result = market_cap_neutral_weights(weights, cap)
    assert result.sum() == pytest.approx(0, abs=1e-12)
    assert result @ cap == pytest.approx(0, abs=1e-12)
    assert result.abs().sum() == pytest.approx(2)
    assert result.iloc[2:6].abs().sum() > 0


def test_joint_neutralization_removes_size_and_beta() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-01"),
            "sector": ["A", "B"] * 30,
            "log_market_cap": rng.normal(size=60),
        }
    )
    beta = pd.Series(rng.normal(size=60))
    y = 3 * data.log_market_cap + 2 * beta + rng.normal(size=60)
    residual = _size_neutralize(y, beta, data, True)
    assert residual.corr(data.log_market_cap) == pytest.approx(0, abs=1e-12)
    assert residual.corr(beta) == pytest.approx(0, abs=1e-12)


def test_bootstrap_is_deterministic_and_marks_short_samples_unavailable() -> None:
    returns = pd.Series(np.random.default_rng(42).normal(0.0003, 0.01, 400))
    first = moving_block_sharpe_interval(returns, block_days=20, samples=200)
    assert first == moving_block_sharpe_interval(returns, block_days=20, samples=200)
    assert np.isfinite(first).all() and first[0] < first[1]
    assert np.isnan(moving_block_sharpe_interval(returns.head(100))).all()
    assert np.isnan(moving_block_sharpe_interval(pd.Series(np.ones(200)))).all()


def test_factor_attribution_recovers_loadings_and_retains_intercept() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    x = rng.normal(0, 0.01, (500, 7))
    design = np.column_stack([np.ones(500), x])
    noise = rng.normal(0, 0.003, 500)
    noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    coefficients = np.array([0.4, -0.2, 0.3, 0.1, -0.1, 0.5, 0.2])
    alpha = 0.0002
    y = alpha + x @ coefficients + noise
    factors = pd.DataFrame(x, columns=RAW_FEATURES, index=dates).rename_axis("date").reset_index()
    factors = factors.melt(id_vars="date", var_name="factor", value_name="net_return")
    factors["execution_lag"] = 1
    portfolios = pd.DataFrame(
        {"date": dates, "net_return": y, "model": "ridge", "variant": "main", "execution_lag": 1}
    )
    summary, loadings = factor_attribution(portfolios, factors)
    got = loadings.set_index("factor").loc[list(RAW_FEATURES), "loading"]
    assert got.to_numpy() == pytest.approx(coefficients)
    assert summary.iloc[0].alpha_daily == pytest.approx(alpha)
    assert summary.iloc[0].residual_sharpe == pytest.approx(
        np.sqrt(252) * alpha / noise.std(ddof=1)
    )


def _training_data() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(1000, 7)), columns=FEATURE_COLUMNS)
    frame["date"] = np.repeat(pd.bdate_range("2020-01-01", periods=100), 10)
    frame["label"] = 0.02 * frame[FEATURE_COLUMNS[0]] - 0.01 * frame[FEATURE_COLUMNS[1]]
    return frame


def test_neural_validation_has_a_label_purge() -> None:
    frame = _training_data()
    fit, validation = neural_split(frame, 6)
    calendar = pd.DatetimeIndex(sorted(frame.date.unique()))
    last_fit = calendar.get_loc(frame.loc[fit, "date"].max())
    first_validation = calendar.get_loc(frame.loc[validation, "date"].min())
    assert last_fit + 6 < first_validation
    assert frame.loc[validation, "date"].nunique() == 20


def test_neural_batches_learn_and_target_scaling_is_affine_invariant() -> None:
    pytest.importorskip("torch")
    train = _training_data()
    config = ModelConfig(include_neural=True, neural_epochs=40, neural_batch_size=32)
    prediction, log = _neural_predictions(train, train.tail(100), FEATURE_COLUMNS, config, 6)
    assert log["optimizer_updates"] > log["epochs_run"]
    assert 1 <= log["best_epoch"] <= log["epochs_run"] <= 40
    assert np.corrcoef(prediction, train.tail(100).label)[0, 1] > 0.9
    transformed = train.assign(label=train.label * 100 + 10)
    transformed_prediction, _ = _neural_predictions(
        transformed, transformed.tail(100), FEATURE_COLUMNS, config, 6
    )
    assert transformed_prediction == pytest.approx(prediction * 100 + 10, rel=1e-4, abs=1e-4)
