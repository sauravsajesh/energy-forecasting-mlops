from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_scalers(df: pd.DataFrame, target_col: str = "load_MW") -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Fit separate scalers for features and target."""
    feature_cols = [c for c in df.columns if c != target_col]
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    feature_scaler.fit(df[feature_cols])
    target_scaler.fit(df[[target_col]])
    
    return feature_scaler, target_scaler


def scale_data(df: pd.DataFrame, feature_scaler, target_scaler, target_col: str = "load_MW") -> Tuple[np.ndarray, np.ndarray]:
    """Scale features and target."""
    feature_cols = [c for c in df.columns if c != target_col]
    X = feature_scaler.transform(df[feature_cols])
    y = target_scaler.transform(df[[target_col]])
    return X, y


def inverse_scale_target(y_scaled: np.ndarray, target_scaler) -> np.ndarray:
    """
    Inverse transform scaled predictions back to original MW units.
    Handles any shape by flattening to (n, 1) then back.
    """
    original_shape = y_scaled.shape
    y_inv = target_scaler.inverse_transform(
        y_scaled.reshape(-1, 1)
    ).reshape(original_shape)
    return y_inv



