import numpy as np

from src.data.sequence_builder import build_default_sequences
from src.models.baseline import evaluate_persistence


def test_sequence_shapes():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_default_sequences(
        lookback=24,
        horizon=24,
        target_col="load_MW",
    )

    # Basic sanity checks
    assert X_train.ndim == 3
    assert y_train.ndim == 2
    assert X_train.shape[1] == 24
    assert y_train.shape[1] == 24
    assert X_train.shape[0] > 100  # enough samples
    assert X_test.shape[0] > 50

def test_persistence_runs():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_default_sequences(
        lookback=24,
        horizon=24,
        target_col="load_MW",
    )

    # Use only target windows from y as a quick proxy for now
    # (more precise approach is in train_baseline.py)
    y_input = y_test[:, :24]  # fake window, correct shape

    metrics = evaluate_persistence(y_true=y_test, y_input_windows=y_input)
    assert "mae" in metrics and "rmse" in metrics and "mape" in metrics
    assert metrics["mae"] > 0
