from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from equity_alpha.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ModelConfig:
    include_neural: bool = True
    neural_epochs: int = 25
    seed: int = 17


def predict_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Fit deterministic baseline forecasters and return test predictions."""

    config = config or ModelConfig()
    validate_model_config(config)
    x_train = train.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=float)
    x_test = test.loc[:, feature_columns].to_numpy(dtype=float)

    predictions = test[["date", "asset", "sector", "label", "future_return"]].copy()
    predictions["ridge"] = _ridge_predictions(x_train, y_train, x_test)
    predictions["tree"] = _tree_predictions(x_train, y_train, x_test, config.seed)
    if config.include_neural:
        predictions["neural"] = _neural_predictions(
            x_train, y_train, x_test, epochs=config.neural_epochs, seed=config.seed
        )
    return predictions


def model_columns(predictions: pd.DataFrame) -> list[str]:
    protected = {"date", "asset", "sector", "label", "future_return", "fold"}
    return [column for column in predictions.columns if column not in protected]


def validate_model_config(config: ModelConfig) -> None:
    if config.neural_epochs <= 0:
        msg = "neural_epochs must be positive"
        raise ValueError(msg)


def _ridge_predictions(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _tree_predictions(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int
) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _neural_predictions(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, epochs: int, seed: int
) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 12),
        nn.ReLU(),
        nn.Linear(12, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x_test_scaled, dtype=torch.float32)).numpy().ravel()
