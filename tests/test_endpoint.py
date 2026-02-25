import requests
import numpy as np

BASE_URL = "http://energy-model-saurav.westeurope.azurecontainer.io:5001"

SAMPLE_INPUT = {
    "features": [[
        [0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8],
        [0.4, 0.3, 0.6, 0.5, 0.7, 0.3, 0.7],
        [0.6, 0.4, 0.5, 0.3, 0.8, 0.4, 0.6],
        [0.5, 0.2, 0.7, 0.4, 0.6, 0.3, 0.9],
        [0.3, 0.5, 0.8, 0.6, 0.5, 0.2, 0.7],
        [0.7, 0.4, 0.6, 0.3, 0.7, 0.5, 0.8],
        [0.4, 0.3, 0.5, 0.7, 0.6, 0.4, 0.6],
        [0.6, 0.5, 0.4, 0.8, 0.5, 0.3, 0.7],
        [0.5, 0.6, 0.7, 0.4, 0.6, 0.5, 0.8],
        [0.3, 0.4, 0.6, 0.5, 0.7, 0.6, 0.5],
        [0.7, 0.5, 0.8, 0.3, 0.6, 0.4, 0.7],
        [0.4, 0.6, 0.5, 0.7, 0.5, 0.3, 0.8],
        [0.6, 0.3, 0.7, 0.4, 0.8, 0.5, 0.6],
        [0.5, 0.4, 0.6, 0.8, 0.5, 0.6, 0.7],
        [0.3, 0.7, 0.5, 0.6, 0.4, 0.5, 0.8],
        [0.8, 0.4, 0.6, 0.5, 0.7, 0.3, 0.6],
        [0.5, 0.3, 0.8, 0.4, 0.6, 0.7, 0.5],
        [0.6, 0.5, 0.4, 0.7, 0.5, 0.4, 0.8],
        [0.4, 0.6, 0.7, 0.5, 0.8, 0.3, 0.6],
        [0.7, 0.4, 0.5, 0.6, 0.4, 0.7, 0.5],
        [0.5, 0.8, 0.6, 0.4, 0.7, 0.5, 0.6],
        [0.3, 0.5, 0.7, 0.8, 0.6, 0.4, 0.7],
        [0.6, 0.4, 0.8, 0.5, 0.5, 0.6, 0.4],
        [0.7, 0.6, 0.5, 0.4, 0.8, 0.5, 0.7]
    ]]
}

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] == True
    print("✅ Health check passed:", r.json())

def test_prediction():
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_INPUT)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    preds = np.array(data["prediction"][0])
    assert preds.shape == (24,), f"Expected 24 outputs, got {preds.shape}"
    assert all(0.0 <= v <= 1.0 for v in preds), "Predictions should be normalised 0–1"
    print("✅ Prediction passed:", np.round(preds, 4))

def test_invalid_input():
    r = requests.post(f"{BASE_URL}/predict", json={"wrong_key": []})
    assert r.status_code in [400, 422, 500]
    print("✅ Invalid input handled:", r.status_code)

if __name__ == "__main__":
    test_health()
    test_prediction()
    test_invalid_input()