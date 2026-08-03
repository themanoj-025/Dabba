# Dabba — Docker Guide

## Quick start

```bash
docker compose up -d
```

Starts API (`:8000`), Streamlit dashboard (`:8501`), MLflow (`:5000`),
PostgreSQL (`:5432`), and Redis (`:6379`). The compose file uses
per-service Dockerfiles in `docker/` (`api.Dockerfile`,
`streamlit.Dockerfile`, `mlflow.Dockerfile`).

> The root `Dockerfile` is a legacy combined image kept for backward
> compatibility only — prefer `docker compose`.

## Environment

See `.env.example`. Key vars: `DABBA_LOG_LEVEL`, `DABBA_MLFLOW_TRACKING_URI`,
`DABBA_DATABASE_URL`, `DABBA_REDIS_URL`, optional `DABBA_API_KEY`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| API depends on mlflow/postgres/redis | All three must be healthy first (`depends_on: condition: service_healthy`) |
| Port conflicts | Adjust `ports` per service |
