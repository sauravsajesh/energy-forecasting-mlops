# Industrial Energy Forecasting — Edge-Cloud MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange)](https://www.tensorflow.org/)
[![Azure](https://img.shields.io/badge/Azure-ACI-0078D4)](https://azure.microsoft.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-blue)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> End-to-end MLOps pipeline for 24-hour ahead industrial energy forecasting
> using real German grid data. Fully containerised, cloud-deployed, and automated.

---

## Live Endpoint

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Service and model status |
| `/predict` | POST | 24-hour energy forecast |

```bash
# Health check
curl http://energy-model-saurav.westeurope.azurecontainer.io:5001/health

# Prediction (input shape: 1 sample × 24 hours × 7 features)
curl -X POST http://energy-model-saurav.westeurope.azurecontainer.io:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[[0.5,0.3,0.7,0.4,0.6,0.2,0.8], ...]]}'
Expected health response:

json
{
  "model_loaded": true,
  "model_version": "blob-v1",
  "service": "model-serving",
  "status": "healthy"
}
Results
Metric	LSTM	Persistence Baseline	Improvement
MAE	1,722 MW	8,499 MW	80%
RMSE	2,419 MW	10,060 MW	76%
MAPE	3.35%	17.22%	80%
Dataset: OPSD German grid data 2015–2023 (~70,000 hourly samples)
Split: 70% train / 15% validation / 15% test — no data leakage

Architecture
text
OPSD Data (2015–2023)
        │
        ▼
Edge Preprocessing Service
  └── Docker container — normalisation + 24h sequence building
        │
        ▼
Model Serving Service (Azure ACI)
  └── TensorFlow 2.16.2 LSTM
  └── Model loaded from Azure Blob Storage (blob-v1)
  └── REST API: /health + /predict
        │
        ├── Azure Container Registry (energyforecastacr)
        │     ├── energy-model-serving:v12
        │     └── energy-edge-service:v12
        │
        ├── Azure Blob Storage — model weights
        │
        └── GitHub Actions CI/CD
              └── auto build + push on every commit to main
Project Structure
text
energy-forecasting-mlops/
├── src/
│   ├── data/               # download_opsd.py, preprocess_opsd.py,
│   │                       # sequence_builder.py, scaling.py
│   ├── models/             # baseline.py, lstm_model.py
│   ├── training/           # train_baseline.py, train_lstm.py
│   ├── deployment/         # model_api.py, Dockerfile, deploy_aci.py
│   └── monitoring/         # drift_detector.py
├── edge_service/           # preprocess_api.py, Dockerfile
├── scripts/                # build-push.sh, deploy-aci.sh, benchmark.py
├── tests/                  # test_preprocessing.py, test_lstm_pipeline.py,
│                           # test_endpoint.py
├── .github/workflows/      # deploy.yml — CI/CD
├── artifacts/              # saved metrics and model outputs
├── requirements.txt
├── requirements-deployment.txt
└── README.md
Quick Start
bash
git clone https://github.com/sauravsajesh/energy-forecasting-mlops.git
cd energy-forecasting-mlops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
Run Training Pipeline

bash
export PYTHONPATH=src
python src/data/download_opsd.py
python src/data/preprocess_opsd.py
python src/training/train_baseline.py
python src/training/train_lstm.py
pytest tests/ -v
Validate Live Endpoint

bash
python tests/test_endpoint.py
Model
Parameter	Value
Architecture	LSTM
Lookback window	24 hours
Forecast horizon	24 hours
LSTM units	64
Dropout	0.2
Optimizer	Adam (lr=0.001)
Batch size	64
Early stopping	patience=8
Input shape	(batch, 24, 7)
Features

Feature	Description
load_MW	Grid consumption in MW (target)
hour	Hour of day 0–23
day_of_week	0=Monday, 6=Sunday
month	Month 1–12
is_weekend	Binary flag
load_rolling_24h_mean	24h rolling average
load_rolling_168h_mean	7-day rolling average
CI/CD
Every push to main automatically:

Builds the Docker image from src/deployment/Dockerfile

Pushes to ACR with :latest and :<commit-sha> tags

Runs tests/test_endpoint.py against the live endpoint

Requires GitHub secrets: ACR_USERNAME, ACR_PASSWORD

Azure Infrastructure
Service	Usage	Cost
Container Registry (Basic)	Image storage	~$5
Container Instances	Model serving	~$13
Blob Storage	Model weights	~$0.50
Networking	Data transfer	~$2
Total		~$20 / $200 credit
Tech Stack
Category	Technology
Language	Python 3.12
ML Framework	TensorFlow 2.16.2 / Keras 3
Experiment Tracking	MLflow 2.10
Cloud	Azure ACI, ACR, Blob Storage
Containerisation	Docker
CI/CD	GitHub Actions
Testing	pytest
Data	Open Power System Data (OPSD) CC BY 4.0
Author
Saurav Sajesh
Master's Student — AI for Smart Sensors and Actuators
Deggendorf Institute of Technology, Germany

GitHub: https://github.com/sauravsajesh

LinkedIn: https://linkedin.com/in/sauravsajesh

Email: saurav.sajesh2001@gmail.com

License
MIT License — see LICENSE for details.

Portfolio project demonstrating production MLOps practices for industrial energy forecasting.
