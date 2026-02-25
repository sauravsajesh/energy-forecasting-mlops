# Industrial Energy Forecasting - Edge-Cloud MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange)](https://www.tensorflow.org/)
[![Azure](https://img.shields.io/badge/Azure-ACI-0078D4)](https://azure.microsoft.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-blue)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![Live](https://img.shields.io/badge/Endpoint-Live-brightgreen)](http://energy-model-saurav.westeurope.azurecontainer.io:5001/health)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Production-ready ML system for 24-hour ahead energy consumption forecasting
> using real German grid data with end-to-end MLOps and hybrid edge-cloud architecture.
> **Live and serving predictions on Azure Container Instances.**

---

## 🟢 Live Endpoint

| Endpoint | URL |
|----------|-----|
| Health | `GET http://energy-model-saurav.westeurope.azurecontainer.io:5001/health` |
| Predict | `POST http://energy-model-saurav.westeurope.azurecontainer.io:5001/predict` |

**Quick test:**
\`\`\`bash
curl http://energy-model-saurav.westeurope.azurecontainer.io:5001/health
\`\`\`

Expected response:
\`\`\`json
{
  "model_loaded": true,
  "model_version": "blob-v1",
  "service": "model-serving",
  "status": "healthy"
}
\`\`\`

**Prediction request:**
\`\`\`bash
curl -X POST http://energy-model-saurav.westeurope.azurecontainer.io:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[[0.5,0.3,0.7,0.4,0.6,0.2,0.8], ...]]}'
\`\`\`

---

## Key Results

| Metric | LSTM Model | Persistence Baseline | Improvement |
|--------|-----------|---------------------|-------------|
| MAE    | 1,722 MW  | 8,499 MW            | 80% decrease |
| RMSE   | 2,419 MW  | 10,060 MW           | 76% decrease |
| MAPE   | 3.35%     | 17.22%              | 80% decrease |

Dataset: Real German energy grid — Open Power System Data (OPSD) 2015–2023
Forecast horizon: 24 hours ahead | Split: 70/15/15 (no data leakage)

---

## Architecture

\`\`\`
OPSD Data (2015–2023)
        │
        ▼
Edge Preprocessing Service (Docker / ACI)
  - Normalisation, sequence building (24h windows)
  - Exposes REST API at :5002
        │
        ▼
Model Serving Service (Docker / ACI)
  - TensorFlow 2.16.2 LSTM
  - Loads model from Azure Blob Storage (blob-v1)
  - Exposes /health and /predict at :5001
        │
        ├── Azure Container Registry (energyforecastacr)
        │     └── energy-model-serving:v12
        │     └── energy-edge-service:v12
        │
        ├── Azure Blob Storage
        │     └── Model weights (blob-v1)
        │
        └── MLflow Experiment Tracking
              └── Metrics, params, artifacts
\`\`\`

---

## Project Structure

\`\`\`
energy-forecasting-mlops/
├── src/
│   ├── data/
│   │   ├── download_opsd.py
│   │   ├── preprocess_opsd.py
│   │   ├── sequence_builder.py
│   │   └── scaling.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── lstm_model.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   └── train_lstm.py
│   ├── deployment/
│   │   ├── simple_inference.py
│   │   └── deploy_aci.py
│   └── monitoring/
│       └── drift_detector.py
├── edge_service/
│   ├── preprocess_api.py
│   └── Dockerfile
├── model_serving/
│   ├── app.py
│   ├── requirements-deployment.txt
│   └── Dockerfile
├── scripts/
│   ├── build-push.sh
│   └── deploy-aci.sh
├── tests/
│   ├── test_preprocessing.py
│   ├── test_sequences_and_baseline.py
│   ├── test_lstm_pipeline.py
│   └── test_endpoint.py
├── kubernetes/
│   ├── edge-deployment.yaml
│   └── model-deployment.yaml
├── mlruns/
├── artifacts/
├── requirements.txt
└── README.md
\`\`\`

---

## Quick Start

### Prerequisites
- Python 3.10+, Docker Desktop, Azure CLI, Git

### Setup

\`\`\`bash
git clone https://github.com/sauravsajesh/energy-forecasting-mlops.git
cd energy-forecasting-mlops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

### Run Training Pipeline

\`\`\`bash
export PYTHONPATH=src
python src/data/download_opsd.py
python src/data/preprocess_opsd.py
python src/training/train_baseline.py
python src/training/train_lstm.py
pytest tests/ -v
\`\`\`

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Lookback window | 24 hours |
| Forecast horizon | 24 hours |
| LSTM units | 64 |
| Dropout | 0.2 |
| Optimizer | Adam lr=0.001 |
| Batch size | 64 |
| Early stopping | patience=8 |
| Framework | TensorFlow 2.16.2 / Keras 3 |

### Input Features

| Feature | Description |
|---------|-------------|
| \`load_MW\` | German grid consumption (target) |
| \`hour\` | Hour of day 0–23 |
| \`day_of_week\` | 0=Monday, 6=Sunday |
| \`month\` | Month 1–12 |
| \`is_weekend\` | Binary flag |
| \`load_rolling_24h_mean\` | 24h rolling average |
| \`load_rolling_168h_mean\` | 7-day rolling average |

Input tensor shape: \`(batch, 24, 7)\`

---

## Testing

\`\`\`bash
# Unit tests
PYTHONPATH=src pytest tests/ -v

# With coverage
PYTHONPATH=src pytest tests/ --cov=src --cov-report=html

# Live endpoint validation
python tests/test_endpoint.py
\`\`\`

\`tests/test_endpoint.py\` validates:
- \`/health\` returns \`model_loaded: true\`
- \`/predict\` returns valid predictions for shape \`(1, 24, 7)\` input
- Invalid input is gracefully handled

---

## Docker

\`\`\`bash
# Edge service
docker build -t energy-edge-service:v12 edge_service/
docker run -d -p 5002:5002 --name edge-sim energy-edge-service:v12
curl http://localhost:5002/health

# Model serving
docker build -t energy-model-serving:v12 model_serving/
docker run -d -p 5001:5001 --name model-server energy-model-serving:v12
curl http://localhost:5001/health
\`\`\`

---

## Azure Deployment

### Prerequisites
\`\`\`bash
az login
az acr login --name energyforecastacr
\`\`\`

### Build and Push to ACR
\`\`\`bash
bash scripts/build-push.sh
\`\`\`

### Deploy to ACI
\`\`\`bash
bash scripts/deploy-aci.sh
\`\`\`

### Manual deployment
\`\`\`bash
az group create --name rg-energy-mlops --location westeurope
az container create \
  --resource-group rg-energy-mlops \
  --name energy-model-serving \
  --image energyforecastacr.azurecr.io/energy-model-serving:v12 \
  --ports 5001 \
  --dns-name-label energy-model-saurav \
  --location westeurope
\`\`\`

---

## MLflow Tracking

\`\`\`bash
mlflow ui
# Open: http://localhost:5000
\`\`\`

Tracks per run: lookback, horizon, LSTM units, dropout, MAE, RMSE, MAPE

---

## Cost Analysis

| Azure Service | Usage | Cost |
|--------------|-------|------|
| Azure ML Workspace | Free tier | \$0 |
| Container Registry (Basic) | ~3 weeks | ~\$5 |
| Container Instances | ~3 days testing | ~\$13 |
| Blob Storage | <5 GB | ~\$0.50 |
| Networking | Data transfer | ~\$2 |
| **Total** | | **~\$20–28 / \$100 budget** |

---

## Roadmap

- [x] Data acquisition (OPSD German grid data)
- [x] Data preprocessing and feature engineering
- [x] Supervised sequence builder
- [x] Persistence baseline model
- [x] LSTM model (3.35% MAPE)
- [x] MLflow experiment tracking
- [x] Edge preprocessing Docker service
- [x] Azure Container Registry (energyforecastacr) — v12
- [x] Azure Container Instances deployment — live at :5001
- [x] Model serving via Azure Blob Storage (blob-v1)
- [x] Live /health and /predict endpoints validated
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Data drift detection and monitoring
- [ ] Edge service ACI deployment (energy-edge-service)
- [ ] Kubernetes deployment (Minikube)
- [ ] Full documentation and thesis writeup

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| ML Framework | TensorFlow 2.16.2 / Keras 3 |
| Experiment Tracking | MLflow 2.10 |
| Cloud Platform | Microsoft Azure (ACI, ACR, Blob) |
| Containerization | Docker |
| Orchestration | Kubernetes (AKS / Minikube) |
| CI/CD | GitHub Actions (planned) |
| Testing | pytest |
| Code Quality | black, flake8, isort |
| Data Source | Open Power System Data (OPSD) |

---

## Dataset

Open Power System Data — Time Series
- Source: https://data.open-power-system-data.org/time_series/
- Coverage: Germany 2015–2023 hourly
- License: Open Data CC BY 4.0
- Size: ~70,000 hourly samples

---

## Author

**Saurav Sajesh**
Master's Student — AI for Smart Sensors and Actuators
Deggendorf Institute of Technology, Germany

- LinkedIn: https://linkedin.com/in/sauravsajesh
- GitHub: https://github.com/sauravsajesh
- Email: saurav.sajesh2001@gmail.com

---

## License

MIT License — see LICENSE file for details.

---

*Portfolio project demonstrating production MLOps best practices for industrial AI in the German energy sector.*
