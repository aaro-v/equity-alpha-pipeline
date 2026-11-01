from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from equity_alpha.research.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ModelConfig:
    include_neural: bool = False
    neural_epochs: int = 100
    neural_batch_size: int = 4096
    neural_patience: int = 10
    neural_learning_rate: float = 0.001
    seed: int = 17


def predict_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    config: ModelConfig | None = None,
    horizon: int = 21,
    execution_lag: int = 1,
    training_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Fit forecasters using only observations available by the test signal close."""
    config = config or ModelConfig()
    validate_model_config(config)
    x_train = train.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=float)
    x_test = test.loc[:, feature_columns].to_numpy(dtype=float)
    columns = ["date", "asset", "sector", "label", "future_return", "log_market_cap"]
    predictions = test[columns].copy()
    predictions["composite"] = test.loc[:, feature_columns].mean(axis=1)
    ridge = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
    ridge.fit(x_train, y_train)
    predictions["ridge"] = ridge.predict(x_test)
    tree = HistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=config.seed,
    )
    tree.fit(x_train, y_train)
    predictions["tree"] = tree.predict(x_test)
    if config.include_neural:
        forecast, summary = _neural_predictions(
            train, test, feature_columns, config, horizon + execution_lag
        )
        predictions["neural"] = forecast
        if training_log is not None:
            training_log.append(summary)
    return predictions


def model_columns(predictions: pd.DataFrame) -> list[str]:
    protected = {"date", "asset", "sector", "label", "future_return", "fold", "log_market_cap"}
    return [column for column in predictions.columns if column not in protected]


def validate_model_config(config: ModelConfig) -> None:
    if config.neural_epochs <= 0:
        raise ValueError("neural_epochs must be positive")
    if min(config.neural_batch_size, config.neural_patience) <= 0:
        raise ValueError("neural batch size and patience must be positive")
    if config.neural_learning_rate <= 0:
        raise ValueError("neural learning rate must be positive")


def neural_split(train: pd.DataFrame, purge_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the final 20% of dates; fitting labels end before validation signals."""
    dates = pd.DatetimeIndex(sorted(train["date"].unique()))
    start = len(dates) - max(1, int(np.ceil(0.2 * len(dates))))
    fit_end = start - purge_days
    if fit_end < 1:
        raise ValueError("training window too short for purged neural validation")
    return train["date"].isin(dates[:fit_end]).to_numpy(), train["date"].isin(
        dates[start:]
    ).to_numpy()


def _neural_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
    config: ModelConfig,
    purge_days: int,
) -> tuple[np.ndarray, dict]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ValueError('Neural models require pip install -e ".[neural]"') from exc
    torch.manual_seed(config.seed)
    torch.set_num_threads(4)
    fit, validation = neural_split(train, purge_days)
    scaler = StandardScaler().fit(train.loc[fit, columns])
    target = train.loc[fit, "label"].to_numpy(dtype=float)
    target_mean, target_scale = target.mean(), target.std(ddof=0)
    target_scale = max(target_scale, 1e-8)
    x = torch.tensor(scaler.transform(train.loc[fit, columns]), dtype=torch.float32)
    y = torch.tensor(((target - target_mean) / target_scale)[:, None], dtype=torch.float32)
    xv = torch.tensor(scaler.transform(train.loc[validation, columns]), dtype=torch.float32)
    yv = torch.tensor(
        ((train.loc[validation, "label"].to_numpy() - target_mean) / target_scale)[:, None],
        dtype=torch.float32,
    )
    model = nn.Sequential(nn.Linear(len(columns), 12), nn.ReLU(), nn.Linear(12, 1))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.neural_learning_rate, weight_decay=1e-3
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(config.seed)
    best_loss, best_epoch, stale, updates = float("inf"), 0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.neural_epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for indices in order.split(config.neural_batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            updates += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(xv), yv))
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.neural_patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        values = model(torch.tensor(scaler.transform(test.loc[:, columns]), dtype=torch.float32))
    summary = {
        "fit_end": train.loc[fit, "date"].max(),
        "validation_start": train.loc[validation, "date"].min(),
        "fit_rows": int(fit.sum()),
        "validation_rows": int(validation.sum()),
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "optimizer_updates": updates,
        "validation_mse": best_loss,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }
    return values.numpy().ravel() * target_scale + target_mean, summary
