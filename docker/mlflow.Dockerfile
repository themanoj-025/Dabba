# ─── Dabba MLflow tracking server ────────────────────────────────────
# Lightweight container running MLflow server on port 5000.
# Uses a persistent volume for the backend store (/mlflow/mlruns).

FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow (only dependency needed)
RUN pip install --no-cache-dir "mlflow>=2.0,<4.0"

# Persistent storage for MLflow runs
VOLUME /mlflow

# Port for MLflow server
EXPOSE 5000

# Default command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/mlflow/mlruns"]
