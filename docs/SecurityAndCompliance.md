# SecurityAndCompliance — Dabba: Security

| Field | Value |
|---|---|
| Version | v0.1 |
| Last Updated | 2026-08-06 |
| Owner | Security Engineer |
| Status | In Review |

---

## 1. Threat Model (STRIDE)

| Threat | Surface | Impact | Mitigation |
|---|---|---|---|
| Spoofing | API key forgery | Data abuse | Key auth + rate limits |
| Tampering | Query params | Wrong rankings | Pydantic validation |
| Repudiation | Model changes | Untracked | MLflow + governance |
| Info disclosure | PII in LLM input | Leak | Minimized inputs, redaction |
| DoS | API flood | Outage | Rate limiting |
| Elevation | Admin actions | Config tamper | Role separation |

## 2. Auth / Authorization

- API: static key + dev mode.
- Dashboard: benchmarks/drift pages admin-only.
- No user accounts in v1 (consumer-facing via dashboard).

## 3. Data Classification

| Data | Class | Handling |
|---|---|---|
| Restaurant data | Public | no encryption needed |
| Review text | Public | VADER local |
| LLM chat inputs | Internal | minimize, redact |
| API keys | Credential | env only |

## 4. Encryption

- In transit: TLS.
- At rest: none sensitive; keys in env/secret manager.

## 5. Compliance Checklist

- [ ] No PII beyond restaurant data
- [ ] API keys env-only
- [ ] Rate limits on API
- [ ] Dependency scans
- [ ] GDPR: minimal collection

## 6. Incident Response Plan (outline)

1. Detect: API error spike.
2. Triage: abuse vs outage.
3. Contain: revoke key / throttle.
4. Remediate + tests.
5. Recover + rotate.
6. Postmortem.

## 7. Related Documents

| Document | Relationship |
|---|---|
| [Rules.md](Rules.md) | Security baseline |
| [API.md](API.md) | Auth + limits |
| [Schema.md](Schema.md) | Sensitive map |
| [TechSpec.md](TechSpec.md) | NFRs |
| [PRD.md](PRD.md) | Goals |
| [AppFlow.md](AppFlow.md) | Flows |
| [Design.md](Design.md) | Design |
| [ImplementationPlan.md](ImplementationPlan.md) | Tasks |
| [Tracker.md](Tracker.md) | Status |
| [Testing.md](Testing.md) | Security tests |
| [Deployment.md](Deployment.md) | Secrets |
| [Glossary.md](Glossary.md) | Vocabulary |
| [RiskRegister.md](RiskRegister.md) | Risks |
