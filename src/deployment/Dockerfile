FROM python:3.11-slim

WORKDIR /app

ENV PIP_TRUSTED_HOST="pypi.org pypi.python.org files.pythonhosted.org"
ENV PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip

COPY requirements-deployment.txt .
RUN pip install -r requirements-deployment.txt

COPY src/deployment/model_api.py .

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "300", "model_api:app"]
