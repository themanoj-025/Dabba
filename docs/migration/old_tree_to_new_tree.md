# Dabba — Old Tree → New Tree

## This pass (2026-08-11)

```
Before                                After
──────                                ─────
docs/migration_summary.md      →      docs/migration/migration_summary.md
—                                     docs/module_dependency.md        (new)
—                                     docs/startup_flow.md             (new)
—                                     docs/package_overview.md         (new)
—                                     docs/migration/old_tree_to_new_tree.md (new)
—                                     docs/migration/file_move_ledger.md     (new)
```

## Prior pass (v5.0 modernization, commit `eeaa846`)

Dabba was restructured by the v5.0 pass into the current layout; its record
(scope, changes, file-move log, import updates, verification, risk analysis,
needs-human-review) lives at `docs/migration/migration_summary.md`.
Tree-level view:

```
Before (flat)                         After (canonical)
──────                                ─────
*.py flat modules            →        src/dabba/ package
                                       ├── data/ · database/ · features/ · models/
                                       ├── llm/ · nlp/ · evaluation/ · monitoring/
                                       ├── cache/ · observability/ · config.py
                                       └── pipeline.py
*.py API modules             →        api/ (FastAPI: main, schemas, auth, limiter, routers/)
*.py Streamlit modules       →        app/ (streamlit_app, pages/, components/, utils/)
*.py tests                   →        tests/ (unit + integration + e2e)
*.ipynb                      →        notebooks/
*.pkl/.pt/.npy/.index        →        models/
CSVs                         →        data/raw + data/processed
Dockerfiles                  →        docker/
```

## No-code-move rationale (this pass)

The layout already conforms (src-layout core package, interface packages,
canonical artifact dirs, root metadata only). This pass only consolidates the
migration record under `docs/migration/` and completes the Phase-6 doc suite —
zero code changed.
