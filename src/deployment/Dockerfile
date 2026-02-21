FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements-deployment.txt .
RUN pip install --no-cache-dir -r requirements-deployment.txt

# Copy source code
COPY src/ src/
COPY mlruns/ mlruns/

ENV PYTHONPATH=src
ENV PYTHONUNBUFFERED=1

EXPOSE 5001

CMD ["python", "src/deployment/model_api.py"]
