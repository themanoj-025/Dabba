# Tracker — Dabba: Living Status Tracker

| Field | Value |
| --- | --- |
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Engineering Lead |
| Status | In Review |

---

## 1. Snapshot Dashboard

| Metric | Value |
| --- | --- |
| Overall % Complete | 55% |
| Current Phase | Phase 3 |
| Tasks Done / Total | 8 / 14 |
| Blockers (open) | 1 |
| Days to Target Launch | 30 |

## 2. Status Legend

🟢 Done | 🟡 In Progress | 🔴 Blocked | ⚪ Not Started | 🔵 In Review

## 3. Phase Progress Bars

| Phase | Progress |
| --- | --- |
| Phase 0: Data | `[████████░░] 100%` |
| Phase 1: Models | `[████████░░] 100%` |
| Phase 2: Reliability | `[████████░░] 100%` |
| Phase 3: LLM Layer | `[███░░░░░░░] 33%` |
| Phase 4: Serving | `[░░░░░░░░░░] 0%` |

## 4. Full Task Table

| TASK | Description | Status | Assignee | Start | Target | Actual | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TASK-0.1 | Kaggle ingestion + cache | 🟢 | Data | 2026-06-15 | 2026-06-17 | — |  |
| TASK-0.2 | Preprocessing + features | 🟢 | Data | 2026-06-18 | 2026-06-21 | — |  |
| TASK-1.1 | Rating model + Optuna | 🟢 | ML | 2026-06-22 | 2026-06-28 | — | MAE 0.0596 |
| TASK-1.2 | ETA model + Optuna | 🟢 | ML | 2026-06-28 | 2026-07-03 | — | MAE 5.789 |
| TASK-1.3 | Benchmark harness | 🟢 | ML | 2026-07-04 | 2026-07-06 | — |  |
| TASK-2.1 | CF (PyTorch) | 🟢 | ML | 2026-07-07 | 2026-07-12 | — |  |
| TASK-2.2 | Reliability score | 🟢 | ML | 2026-07-12 | 2026-07-14 | — |  |
| TASK-2.3 | Drift + Slack alert | 🟢 | ML | 2026-07-15 | 2026-07-17 | — |  |
| TASK-3.1 | Narrator + fallback | 🟡 | Eng | 2026-07-18 | — | — | in progress |
| TASK-3.2 | FAISS RAG | ⚪ | ML | — | — | — |  |
| TASK-3.3 | Concierge chat | ⚪ | Eng | — | — | — |  |
| TASK-4.1 | FastAPI endpoints | ⚪ | Eng | — | — | — |  |
| TASK-4.2 | Streamlit dashboard | ⚪ | FE | — | — | — |  |
| TASK-4.3 | MLflow wiring | ⚪ | ML | — | — | — |  |

## 5. Blockers Log

| ID | Description | Raised | Owner | Impact | Status |
| --- | --- | --- | --- | --- | --- |
| BLK-001 | .env file has non-UTF8 box-drawing chars → Starlette Config UnicodeDecodeError on Windows | 2026-08-01 | Eng | Local test collection errors | 🔴 Open — save .env as UTF-8 or drop decorative chars |

## 6. Changelog

- 2026-08-06: **Documentation suite complete** — 14-file suite consolidated into `docs/`, categorized structure, cross-linked navigation, deployment/git/auth diagrams, quality gate passed (238/238), merged to `main`.
| Date | What shipped |
| --- | --- |
| 2026-08-06 | Docs suite v0.1 |
| 2026-07-17 | Phase 2 complete (reliability + drift) |

## 7. Burndown Summary

```mermaid
pie
    title Tasks by Status
    "Done" : 8
    "In Progress" : 1
    "Not Started" : 5
```

## 8. Next 3 Priorities

1. Finish TASK-3.1 — LLM narrator + fallback.
2. TASK-3.2 — FAISS RAG + fallback.
3. TASK-3.3 — Concierge chat.

## 9. Related Documents

| Document | Relationship |
| --- | --- |
| [ImplementationPlan.md](ImplementationPlan.md) | Tasks |
| [PRD.md](../product/PRD.md) | Features |
| [TechSpec.md](../technical/TechSpec.md) | Components |
| [AppFlow.md](../design/AppFlow.md) | Flows |
| [Design.md](../design/Design.md) | Design |
| [Schema.md](../technical/Schema.md) | Data |
| [Rules.md](Rules.md) | Standards |
| [API.md](../technical/API.md) | Contract |
| [SecurityAndCompliance.md](../technical/SecurityAndCompliance.md) | Security |
| [Testing.md](../technical/Testing.md) | Tests |
| [Deployment.md](../technical/Deployment.md) | Deploy |
| [Glossary.md](../reference/Glossary.md) | Vocabulary |
| [RiskRegister.md](RiskRegister.md) | Risks |
