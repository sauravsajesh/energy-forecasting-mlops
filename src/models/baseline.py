import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def persistence_forecast(last_window: np.ndarray, horizon: int) -> np.ndarray:
    """
    last_window: shape (lookback,) or (lookback, n_features) but we expect 1D target.
    Returns forecast of shape (horizon,)
    """
    # Simple: repeat last value horizon times
    last_value = last_window[-1]
    return np.full(shape=(horizon,), fill_value=last_value, dtype="float32")


def evaluate_persistence(
    y_true: np.ndarray,
    y_input_windows: np.ndarray,
) -> dict:
    """
    y_true: shape (n_samples, horizon)
    y_input_windows: shape (n_samples, lookback) of target values

    Returns MAE, RMSE, MAPE on horizon-aggregated forecast
    (here we evaluate on first step or mean over horizon).
    """
    n_samples, horizon = y_true.shape
    preds = np.zeros_like(y_true, dtype="float32")

    for i in range(n_samples):
        preds[i] = persistence_forecast(y_input_windows[i], horizon=horizon)

    # Evaluate on all horizon steps (you can later also compare only first hour)
    mae = mean_absolute_error(y_true.flatten(), preds.flatten())
    mse = mean_squared_error(y_true.flatten(), preds.flatten())
    rmse = float(mse ** 0.5)
    mape = np.mean(np.abs((y_true.flatten() - preds.flatten()) / (y_true.flatten() + 1e-6))) * 100

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
