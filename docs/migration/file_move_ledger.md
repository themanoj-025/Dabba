# Dabba — File Move Ledger

## This pass (2026-08-11)

| Old path | New path | Category | Reason | Risk | Verified |
|---|---|---|---|---|---|
| `docs/migration_summary.md` | `docs/migration/migration_summary.md` | Meta/docs | Consolidate migration records under `docs/migration/` per enterprise standard | Low (docs only) | ✅ `git mv` preserved history; no inbound refs found |

## Prior pass (v5.0 modernization, commit `eeaa846`)

The v5.0 pass moved application code into the current layout. Its complete
file-move log is preserved at `docs/migration/migration_summary.md` (§ File
move log) with import/reference updates (§ Import/reference update summary)
and verification (§ Verification report).

## Non-moves (documented decisions)

| Path | Decision | Reason |
|---|---|---|
| `src/dabba/**` | keep | src-layout core package (installed via pyproject) |
| `api/**`, `app/**` | keep | Framework interface layers (FastAPI include_router, Streamlit multipage); Docker/Compose/Makefile entry contract |
| `notebooks/**`, `models/**`, `reports/**`, `data/**`, `docker/**`, `alembic/**` | keep | Canonical artifact locations |
| `setup_kaggle.py`, `mlflow.db` | keep | Kaggle bootstrap + MLflow store (mlflow.db untracked) |
| `.env`, `data/dabba.db*`, `src/dabba.egg-info/`, `.hypothesis/` | leave (untracked) | Secrets/runtime/build artifacts, correctly gitignored |
