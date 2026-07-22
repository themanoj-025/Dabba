#!/bin/bash
# ─── Dabba Docker Entrypoint ──────────────────────────────────────────
# Runs Alembic migrations on startup, then launches the app server.
# This ensures the database schema is always up-to-date when the
# container starts, whether using SQLite (dev) or Postgres (prod).
#
# Usage:
#   docker/entrypoint.sh                          # runs migrations + uvicorn
#   docker/entrypoint.sh --skip-migrations        # skip alembic, start directly
#   docker/entrypoint.sh --seed                   # run migrations + seed + start
# ======================================================================

set -euo pipefail

MIGRATE=true
SEED=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --skip-migrations) MIGRATE=false ;;
        --seed) SEED=true ;;
    esac
done

if [ "$MIGRATE" = true ]; then
    echo ">>> Running Alembic migrations (upgrade head)..."
    alembic upgrade head
    echo ">>> Migrations complete."
fi

if [ "$SEED" = true ]; then
    echo ">>> Seeding database from processed CSVs..."
    python -m dabba.database.seed || echo ">>> Seed skipped (no CSVs found)"
    echo ">>> Seeding complete."
fi

echo ">>> Starting uvicorn: api.main:app on port 8000"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
