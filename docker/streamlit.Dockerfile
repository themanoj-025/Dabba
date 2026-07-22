# ─── Dabba Streamlit dashboard service ─────────────────────────────
# Serves the user-facing dashboard on port 8501.
# Requires the dabba package (src/) and models/ artifacts.

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

# Copy application code
COPY app/ app/
COPY src/ src/
COPY models/ models/

# Port for Streamlit
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501"]
