# Industrial Energy Forecasting - Edge-Cloud MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[![Azure](https://img.shields.io/badge/Azure-ML-0078D4)](https://azure.microsoft.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-blue)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Production-ready ML system for 24-hour ahead energy consumption forecasting
> using real German grid data with end-to-end MLOps and hybrid edge-cloud architecture.

---

## Key Results

| Metric | LSTM Model | Persistence Baseline | Improvement |
|--------|-----------|---------------------|-------------|
| MAE | 1,722 MW | 8,499 MW | 80% decrease |
| RMSE | 2,419 MW | 10,060 MW | 76% decrease |
| MAPE | 3.35% | 17.22% | 80% decrease |

Dataset: Real German energy grid - Open Power System Data (OPSD) 2015-2023
Forecast horizon: 24 hours ahead | Split: 70/15/15 (no data leakage)

---

## Architecture

Edge Preprocessing (Docker) --> Azure ML Workspace --> MLflow Registry
--> Azure Container Instances --> GitHub Actions CI/CD --> Monitoring + Drift Detection

---

## Project Structure

src/data/          - preprocess_opsd.py, sequence_builder.py, scaling.py
src/models/        - baseline.py, lstm_model.py
src/training/      - train_baseline.py, train_lstm.py
src/deployment/    - simple_inference.py
src/monitoring/    - drift_detector.py
edge_service/      - preprocess_api.py, Dockerfile
kubernetes/        - edge-deployment.yaml, model-deployment.yaml
tests/             - test_preprocessing.py, test_sequences_and_baseline.py, test_lstm_pipeline.py
mlruns/            - MLflow experiment tracking
artifacts/         - baseline and lstm metrics

---

## Quick Start

### Prerequisites
- Python 3.10+, Docker Desktop, Azure CLI, Git

### Setup

    git clone https://github.com/sauravsajesh/energy-forecasting-mlops.git
    cd energy-forecasting-mlops
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### Run pipeline

    export PYTHONPATH=src
    python src/data/download_opsd.py
    python src/data/preprocess_opsd.py
    python src/training/train_baseline.py
    python src/training/train_lstm.py
    pytest tests/ -v

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

### Features

| Feature | Description |
|---------|-------------|
| load_MW | German grid consumption (target) |
| hour | Hour of day 0-23 |
| day_of_week | 0=Monday 6=Sunday |
| month | Month 1-12 |
| is_weekend | Binary flag |
| load_rolling_24h_mean | 24h rolling average |
| load_rolling_168h_mean | 7-day rolling average |

---

## Testing

    PYTHONPATH=src pytest tests/ -v
    PYTHONPATH=src pytest tests/ --cov=src --cov-report=html

---

## Docker

    docker build -t energy-edge-service:v1 edge_service/
    docker run -d -p 5000:5000 --name edge-sim energy-edge-service:v1
    curl -X GET http://localhost:5000/health

---

## Azure Deployment

    az login
    az group create --name rg-energy-mlops --location westeurope
    az ml workspace create --name ml-energy-forecast --resource-group rg-energy-mlops
    python src/deployment/deploy_aci.py

---

## MLflow Tracking

    mlflow ui
    # Open: http://localhost:5000

Tracks: lookback, horizon, units, dropout, MAE, RMSE, MAPE per run

---

## Cost Analysis

| Azure Service | Usage | Cost |
|--------------|-------|------|
| Azure ML Workspace | Free tier | $0 |
| Container Registry | Basic 3 weeks | ~$5 |
| Container Instances | 3 days testing | ~$13 |
| Storage | less than 5GB | ~$0.50 |
| Networking | data transfer | ~$2 |
| Total | | ~$20-28 of $100 budget |

---

## Roadmap

- [x] Data acquisition (OPSD German grid data)
- [x] Data preprocessing and feature engineering
- [x] Supervised sequence builder
- [x] Persistence baseline model
- [x] LSTM model (3.35% MAPE)
- [x] MLflow experiment tracking
- [x] Edge preprocessing Docker service
- [ ] Azure Container Registry integration
- [ ] Azure Container Instances deployment
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Data drift detection and monitoring
- [ ] Kubernetes deployment (Minikube)
- [ ] Full documentation

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| ML Framework | TensorFlow 2.15 / Keras |
| Experiment Tracking | MLflow 2.10 |
| Cloud Platform | Microsoft Azure |
| Containerization | Docker |
| Orchestration | Kubernetes (AKS / Minikube) |
| CI/CD | GitHub Actions |
| Testing | pytest |
| Code Quality | black, flake8, isort |
| Data Source | Open Power System Data (OPSD) |

---

## Dataset

Open Power System Data - Time Series
- Source: https://data.open-power-system-data.org/time_series/
- Coverage: Germany 2015-2023 hourly
- License: Open Data CC BY 4.0
- Size: ~70,000 hourly samples

---

## Author

Saurav Sajesh
Master's Student - AI for Smart Sensors and Actuators
Deggendorf Institute of Technology, Germany

- LinkedIn: https://linkedin.com/in/sauravsajesh
- GitHub: https://github.com/sauravsajesh
- Email: saurav.sajesh2001@gmail.com

---

## License

MIT License - see LICENSE file for details.

---

*Portfolio project demonstrating MLOps best practices for industrial AI in the German market.*
