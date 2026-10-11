# DEAD-WEIGHT PURGE v2.0 AUDIT REPORT

**Repository**: `Dabba`  
**Mode**: `AUDIT_ONLY` (Zero repository modifications performed)  
**Baseline Commit SHA**: `5bb30a84222c5cf1c2e7721558ea475c32ac054c`  
**Audit Date**: 2026-10-11  
**Verdict**: **YELLOW** (Dead weight detected; review recommended)

---

## 1. Executive Summary

- **Tracked Files in Git**: 211
- **Untracked Local Files**: 0
- **Total LOC**: 27,560
- **Analyzed Size**: 0.93 MB
- **Total Dead-Weight Candidates Identified**: 101
  - **Tier 0 (Provable Junk/Empty)**: 70
  - **Tier 1 (High Confidence)**: 0
  - **Tier 2 (Medium / Needs Verification)**: 30
  - **Tier 3 (Protected / Hard Vetoed)**: 1

## 2. Findings by Category

| Category | Description | Count |
|---|---|---|
| **A** | Empty & Near-Empty Files | 4 |
| **C** | Junk / Caches / Build Artifacts | 67 |
| **D** | Potential Orphan Modules | 30 |
| **H** | Exact Hash Duplicates | 0 |

### Top Tier-0 Items (Provable Deletion Candidates if EXECUTE_SAFE were used):
- [`A`] `app/py.typed`: 0-byte file with no content
- [`A`] `py.typed`: 0-byte file with no content
- [`A`] `src/py.typed`: 0-byte file with no content
- [`C`] `.cleanup-audit/`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `.cleanup-audit/.claude/`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `.env`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `.hypothesis/`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `.hypothesis/.gitignore`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `.hypothesis/unicode_data/`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `.mypy_cache/`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `.pytest_cache/`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `.ruff_cache/`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `AGENTS.md`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `AI_DISCLOSURE.md`: Untracked build/dependency/cache artifact directory in workspace
- [`C`] `__pycache__/`: Untracked build/dependency/cache artifact directory in workspace
_...and 55 more listed in `manifest.csv`._

## 3. Next Steps & Recommended Actions

1. Review `.purge-reports/manifest.csv` and `.purge-reports/FLAGGED.md`.
2. If safe cleanup is desired, configure `MODE: EXECUTE_SAFE` or generate an approval `PLAN.yaml`.
3. Verify that all untracked artifacts (`__pycache__`, `.venv`, `.pytest_cache`) are properly covered in `.gitignore`.
