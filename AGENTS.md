# AGENTS.md — Dabba

> Canonical project instructions. Pointers like `CLAUDE.md` or
> `.github/copilot-instructions.md` should say "See AGENTS.md".

---

## Project overview

**Dabba** — a restaurant-goods and dark-kitchen delivery operations
platform. Core components:

- **Delivery** — order routing, driver dispatch, logistics.
- **Catalog** — restaurant catalogue + dish management.
- **Pricing** — dynamic pricing and margin calculations.
- **Inventory** — stock handling and batch management.

Stack: Python 3.11+ · FastAPI · PostgreSQL · Redis · Celery.

---

## Exact commands

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Lint / typecheck / test
make lint
pre-commit run --all-files
python -m mypy . --ignore-missing-imports
python -m pytest tests/ -v --cov=. --cov-fail-under=70

# Run
uvicorn api.main:app --reload
```

---

## Folder map

| Path | Purpose |
|------|---------|
| `api/` | FastAPI application (routes, services) |
| `delivery/` | Routing + dispatch engine |
| `catalog/` | Restaurant / dish catalogue |
| `pricing/` | Pricing engine |
| `inventory/` | Stock + batch handling |
| `tests/` | pytest suite |
| `.github/workflows/` | CI (ruff, mypy, pytest, gitleaks, trivy) |

## Do / don't

- **Do** treat driver capacity as a hard constraint in routing.
- **Do not** hardcode restaurant or menu data in the API.
- **Do not** commit `.env` files.

## Security rules

- No secrets in the repository; `gitleaks` CI gate gates on hits.
- Rate-limit driver-facing endpoints to avoid abuse.

## AI-assistance convention

Commits authored by AI must carry the trailer:

```text
AI-Assisted: yes | no | partial
```

See `.gitmessage` for the template. Do not rewrite historic commits
retroactively.
