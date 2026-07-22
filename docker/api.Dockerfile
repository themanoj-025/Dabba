# ─── Dabba API service ──────────────────────────────────────────────
# Serves the FastAPI restaurant intelligence API on port 8000.
# Built with /health endpoint for load-balancer and Docker health checks.
# Uses docker/entrypoint.sh to run Alembic migrations before startup.

FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code — only what the API needs
COPY api/ api/
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini .
COPY docker/entrypoint.sh docker/entrypoint.sh
COPY models/ models/

# Make entrypoint executable
RUN chmod +x docker/entrypoint.sh

# Port for uvicorn
EXPOSE 8000

# Default command — run migrations then start uvicorn
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD []
