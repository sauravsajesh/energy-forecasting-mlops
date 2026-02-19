from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sequence_builder import build_default_sequences, load_processed
from src.models.baseline import evaluate_persistence


def main():
    # Load processed df to get target series
    df = load_processed()
    target = df["load_MW"].values.astype("float32")

    # Build supervised sequences
    lookback = 24
    horizon = 24
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_default_sequences(
        lookback=lookback,
        horizon=horizon,
        target_col="load_MW",
    )

    # For persistence we only need the last window of target values
    # Extract from target series: align with X/y creation
    total = len(df)
    n_samples = total - lookback - horizon

    target_windows = []
    for i in range(n_samples):
        window = target[i : i + lookback]
        target_windows.append(window)
    target_windows = np.stack(target_windows)

    # Split same way as X/y
    n_train = len(X_train)
    n_val = len(X_val)

    target_train = target_windows[:n_train]
    target_val = target_windows[n_train : n_train + n_val]
    target_test = target_windows[n_train + n_val :]

    # Evaluate on test set
    metrics_test = evaluate_persistence(
        y_true=y_test,
        y_input_windows=target_test,
    )

    print("=== Persistence baseline (24h ahead) ===")
    print(f"Test MAE : {metrics_test['mae']:.3f}")
    print(f"Test RMSE: {metrics_test['rmse']:.3f}")
    print(f"Test MAPE: {metrics_test['mape']:.2f}%")

    # Optionally save metrics
    out_dir = Path("artifacts/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics_test).to_json(out_dir / "persistence_test_metrics.json", indent=2)


if __name__ == "__main__":
    main()
