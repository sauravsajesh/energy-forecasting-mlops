from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROCESSED_PATH = Path("data/processed/germany_energy_processed.csv")


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}")
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df = df.sort_index()
    return df


def create_supervised_sequences(
    df: pd.DataFrame,
    target_col: str = "load_MW",
    lookback: int = 24,
    horizon: int = 24,
    feature_cols: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series into supervised X, y.

    X shape: (n_samples, lookback, n_features)
    y shape: (n_samples, horizon)
    """
    if feature_cols is None:
        feature_cols = df.columns.tolist()

    assert target_col in df.columns, f"{target_col} not in dataframe"

    data = df[feature_cols].values.astype("float32")
    target = df[target_col].values.astype("float32")

    X, y = [], []
    total = len(df)

    for i in range(total - lookback - horizon):
        X.append(data[i : i + lookback, :])
        y.append(target[i + lookback : i + lookback + horizon])

    X = np.stack(X)
    y = np.stack(y)

    return X, y


def time_series_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]

    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_default_sequences(
    lookback: int = 24,
    horizon: int = 24,
    target_col: str = "load_MW",
):
    df = load_processed()
    X, y = create_supervised_sequences(
        df,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_series_train_val_test_split(X, y)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_default_sequences()
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape, "y_test: ", y_test.shape)
