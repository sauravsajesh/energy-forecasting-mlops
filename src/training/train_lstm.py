import os
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.sequence_builder import build_default_sequences, load_processed, build_default_sequences_scaled
from src.models.lstm_model import build_lstm_model
from src.data.scaling import inverse_scale_target

def evaluate_multi_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true, y_pred: shape (n_samples, horizon)
    """
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = float(mse**0.5)
    mape = (
        np.mean(
            np.abs((y_true.flatten() - y_pred.flatten()) / (y_true.flatten() + 1e-6))
        )
        * 100.0
    )

    return {"mae": float(mae), "rmse": rmse, "mape": float(mape)}


def main():
    # Hyperparameters
    lookback = 24
    horizon = 24
    batch_size = 64
    epochs = 50
    units = 64
    dropout = 0.2

    # Load data
    (X_train, y_train, feature_scaler, target_scaler), (X_val, y_val), (X_test, y_test) = build_default_sequences_scaled()
    n_features = X_train.shape[2]

    # Configure MLflow (local tracking for now; Azure ML integration on a later day)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("energy_lstm")

    with mlflow.start_run(run_name="lstm_24x24_v1"):
        mlflow.log_params(
            {
                "lookback": lookback,
                "horizon": horizon,
                "batch_size": batch_size,
                "epochs": epochs,
                "units": units,
                "dropout": dropout,
                "n_features": n_features,
            }
        )

        model = build_lstm_model(
            lookback=lookback,
            n_features=n_features,
            horizon=horizon,
            units=units,
            dropout=dropout,
        )

        callbacks = [
            # Early stopping to avoid overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Log training curves
        for k, v in history.history.items():
            mlflow.log_metric(f"train_{k}", v[-1])

        # Evaluate on test set
        y_pred = model.predict(X_test, batch_size=batch_size)

# Inverse transform predictions and true values
        y_test_inv = inverse_scale_target(y_test, target_scaler)
        y_pred_inv = inverse_scale_target(y_pred, target_scaler)

        metrics = evaluate_multi_horizon(y_true=y_test_inv, y_pred=y_pred_inv)

        print("=== LSTM (24h ahead) Test Metrics ===")
        print(f"MAE : {metrics['mae']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"MAPE: {metrics['mape']:.2f}%")

        mlflow.log_metrics(
            {
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
                "test_mape": metrics["mape"],
            }
        )

        # Log model
        mlflow.keras.log_model(model, artifact_path="model")

        # Save predictions for later analysis
        out_dir = Path("artifacts/lstm")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "y_test.npy", y_test)
        np.save(out_dir / "y_pred.npy", y_pred)


if __name__ == "__main__":
    import tensorflow as tf  # keep import here to avoid circular when tooling
    main()
