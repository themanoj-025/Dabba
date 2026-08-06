# Rules — Dabba: Coding Standards & AI-Agent Operating Rules

| Field | Value |
|---|---|
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Engineering Lead |
| Status | In Review |

---

## 1. Guiding Principles

1. Deterministic before probabilistic — ranking works without LLM.
2. Reproducibility — every experiment traceable in MLflow.
3. No silent failures — fallbacks are explicit and logged.
4. Small PRs only.
5. Tests accompany behavior changes.
6. Honest labeling — LLM output marked vs template.

## 2. Code Style

- Python 3.11, type hints required.
- Formatter: black; linter: ruff; isort.
- Structure:

```
dabba/
  data/          # ingestion, preprocessing
  models/        # rating, eta, cf, reliability
  explain/       # narrator
  rag/           # faiss similarity
  chat/          # concierge
  monitor/       # drift detection
  api/           # FastAPI
  dashboard/     # Streamlit
tests/
```

## 3. Git Workflow

- Branches: `feat/<slug>`, `fix/<slug>`, `ml/<experiment>`.
- Commits: Conventional Commits.
- PRs: ≤ 400 lines, CI green.
- Merge: squash to main.

## 4. Testing Requirements

- Coverage ≥ 60%.
- MUST have tests: preprocessing, model interfaces, fallback paths, API endpoints, reliability formula.
- See [Testing.md](Testing.md).

## 5. AI Agent Operating Rules

- Always read Tracker.md and ImplementationPlan.md before starting.
- Never mark a task 🟢 Done without tests passing.
- Never invent requirements not in PRD.md/TechSpec.md — flag ambiguity.
- Always update Schema.md when data model changes.
- Never commit secrets; env vars per SecurityAndCompliance.md.
- Never let LLM features block core ranking (fallbacks mandatory).
- State conflicts rather than silently picking one.

## 6. Security Baseline Rules

- API key auth on API; rate limiting.
- No PII collected beyond restaurant data.
- LLM inputs minimized.
- Dependencies scanned weekly.

## 7. Documentation Rules

- New endpoints → API.md same PR.
- Schema changes → Schema.md same PR.
- New env vars → Deployment.md.

## 8. Prohibited Patterns

| Anti-pattern | Why |
|---|---|
| LLM call blocking ranking path | Availability |
| Logging API keys | Leak |
| Untracked model training | Reproducibility |
| Hardcoded dataset paths | Portability |
| Blanket except | Hides failures |

## 9. Escalation Rules

**Ask a human when:** new data sources, LLM provider changes, model promotion to prod, PII handling.
**Decide autonomously:** refactors, tests, config tuning, fallback improvements.

## 10. Related Documents

| Document | Relationship |
|---|---|
| [Testing.md](Testing.md) | Test requirements |
| [SecurityAndCompliance.md](SecurityAndCompliance.md) | Security |
| [PRD.md](PRD.md) | Requirements |
| [TechSpec.md](TechSpec.md) | Architecture |
| [AppFlow.md](AppFlow.md) | Flows |
| [Design.md](Design.md) | Design |
| [Schema.md](Schema.md) | Data |
| [ImplementationPlan.md](ImplementationPlan.md) | Tasks |
| [Tracker.md](Tracker.md) | Status |
| [API.md](API.md) | Contract |
| [Deployment.md](Deployment.md) | Env vars |
| [Glossary.md](Glossary.md) | Vocabulary |
| [RiskRegister.md](RiskRegister.md) | Risks |
