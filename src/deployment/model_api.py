import os
from pathlib import Path

import mlflow
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL = None
MODEL_VERSION = "unknown"


def load_model_from_mlruns():
    """Load latest MLflow model from local mlruns."""
    global MODEL, MODEL_VERSION

    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("energy_lstm")

    if experiment is None:
        print("No MLflow experiment found. Model not loaded.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_mape ASC"],
        max_results=1,
    )

    if not runs:
        print("No runs found.")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Loading model from run: {run_id}")
    MODEL = mlflow.keras.load_model(model_uri)
    MODEL_VERSION = run_id[:8]
    print(f"Model loaded. Version: {MODEL_VERSION}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "model-serving",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION
    })


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input' in request body"}), 400

    try:
        input_seq = np.array(data["input"], dtype="float32")

        # Expect shape (lookback, n_features) -> add batch dim
        if input_seq.ndim == 2:
            input_seq = input_seq[np.newaxis, ...]  # (1, lookback, n_features)

        predictions = MODEL.predict(input_seq, verbose=0)

        return jsonify({
            "forecast_24h": predictions[0].tolist(),
            "model_version": MODEL_VERSION,
            "horizon": len(predictions[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_version": MODEL_VERSION,
        "forecast_horizon": 24,
        "input_shape": "(1, 24, n_features)",
        "output_shape": "(1, 24)",
        "description": "LSTM model for 24h German energy consumption forecasting"
    })


if __name__ == "__main__":
    load_model_from_mlruns()
    app.run(host="0.0.0.0", port=5001)
