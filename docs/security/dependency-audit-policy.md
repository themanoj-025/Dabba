# Dependency Audit Policy

The `security-audit` CI job runs `pip-audit -r requirements.txt` on every push.

## Rules

1. **Fix by bumping.** If an advisory lists a fixed release, raise the pin to that
   release (or higher) in `requirements.txt`. Advisories are never ignored while a
   fix exists.
2. **Ignore only with no upstream fix.** A vulnerable dependency may be ignored
   *only* when no fixed release has been published. Each ignore must appear in the
   table below with a justification and a review date, and must be passed to the job
   as an explicit `--ignore-vuln <ID>` flag so it is visible in the workflow.
3. **Review on every dependency change** and at least quarterly. When a fix ships,
   remove the ignore and bump the pin (rule 1).

## Current ignores

| Advisory | Package | Why ignored | Review by |
|---|---|---|---|
| `PYSEC-2026-3740` | `nltk` | No fixed release published upstream at time of writing (`nltk>=3.10.3,<4.0`). Re-check on each nltk release. | 2026-12-19 |

## Recently resolved (kept for context)

| Advisory | Package | Resolution |
|---|---|---|
| `PYSEC-2026-1845` | `pytest` | Pin raised from `<9.0` to `>=9.0.3,<10.0` (fixed release 9.0.3). |

## Running locally

```bash
pip-audit -r requirements.txt --ignore-vuln PYSEC-2026-3740
```
