---
name: fastapi-verification
description: "Verification loop for FastAPI projects: Alembic migration checks, linting, async tests with coverage, security scans, and deployment readiness before release or PR."
origin: local
---

# FastAPI Verification Loop

Run before PRs, after major changes, and pre-deploy to ensure FastAPI application quality and security.

## When to Activate

- Before opening a pull request for a FastAPI project
- After model changes, Alembic migration updates, or dependency upgrades
- Pre-deployment verification for staging or production
- Running full environment → lint → migrations → tests → security → config pipeline

## Phase 1: Environment Check

```bash
# Python version (match pyproject.toml / .python-version)
python --version

# Verify virtual environment is active
which python  # Should be inside venv or .venv

# Check for outdated dependencies
uv pip list --outdated
# or: pip list --outdated

# Verify critical env vars are set
python -c "
import os
required = ['DATABASE_URL', 'SECRET_KEY']
missing = [v for v in required if not os.environ.get(v)]
if missing:
    print('MISSING:', missing)
else:
    print('All required env vars set')
"
```

If environment is misconfigured, stop and fix before proceeding.

## Phase 2: Code Quality & Formatting

```bash
# Type checking
mypy app/ --ignore-missing-imports

# Linting with ruff (replaces flake8 + isort + pyupgrade)
ruff check . --fix

# Formatting
ruff format . --check
ruff format .  # auto-fix

# Or if using black + isort separately:
black . --check && black .
isort . --check-only && isort .
```

Common issues:
- Missing type hints on public functions (mypy)
- Unused imports, undefined names (ruff)
- Async functions not awaited
- Pydantic v1 patterns used instead of v2

## Phase 3: Alembic Migrations

```bash
# Check current migration state
alembic current

# Verify all migrations are applied
alembic history --verbose

# Check for model changes that need a new migration
# (compare ORM models against the DB)
alembic check
# Returns non-zero if autogenerate would produce changes

# Dry-run next migration (review before applying)
alembic upgrade head --sql | head -50

# Apply migrations (test environment)
alembic upgrade head

# Verify downgrade works (important for rollback safety)
alembic downgrade -1
alembic upgrade head
```

Report:
- Current migration revision
- Any unapplied migrations
- Any ORM changes without a corresponding migration

**Migration safety checklist:**
- [ ] New columns have server defaults or are nullable
- [ ] No destructive changes (DROP COLUMN, DROP TABLE) without data backup
- [ ] Large table migrations tested for lock impact
- [ ] Downgrade script is correct and tested

## Phase 4: Tests + Coverage

```bash
# Run full test suite with coverage
pytest --cov=app --cov-report=term-missing --cov-report=html

# Run fast tests only (exclude slow/integration)
pytest -m "not slow and not integration"

# Run with verbose output to identify failing tests
pytest -v

# Open HTML coverage report
open htmlcov/index.html  # macOS
```

Report:
- Total: X passed, Y failed, Z skipped
- Overall coverage: XX%
- Per-module coverage breakdown

Coverage targets:

| Component | Target |
|-----------|--------|
| Routers | 85%+ |
| Services | 90%+ |
| Auth / security | 90%+ |
| Schemas / validators | 80%+ |
| Overall | 80%+ |

Fail verification if overall coverage drops below 80%.

## Phase 5: Security Scan

```bash
# Dependency vulnerability scan
pip-audit

# Static security analysis
bandit -r app/ -f json -o bandit-report.json
bandit -r app/  # human-readable summary

# Secret detection (gitleaks or trufflehog)
gitleaks detect --source . --verbose
# or: trufflehog filesystem .

# Check for known CVEs in transitive deps
uv pip check
```

Report:
- Vulnerable packages + CVE IDs
- Bandit: HIGH/MEDIUM issues to fix, LOW to review
- Exposed secrets (should be zero; stop deployment if any found)

**Auto-fail conditions:**
- Any HIGH severity bandit finding
- Any known CVE with a fix available (pip-audit)
- Any hardcoded secret detected

## Phase 6: Configuration Review

```python
# run: python -c "$(cat << 'EOF'
from app.config import get_settings

s = get_settings()
checks = {
    "debug is False": not s.debug,
    "secret_key is set": bool(s.secret_key),
    "database_url is set": bool(s.database_url),
    "cors_origins not wildcard": s.cors_origins != ["*"],
    "access_token_expire set": s.access_token_expire_minutes > 0,
}
for check, result in checks.items():
    icon = "✓" if result else "✗"
    print(f"{icon} {check}")
# EOF
# )"
```

## Phase 7: Performance Checks

```bash
# Check for sync blocking calls in async context
# (grep for common offenders: requests, time.sleep, open())
grep -rn "requests\." app/ --include="*.py" | grep -v test
grep -rn "time\.sleep" app/ --include="*.py" | grep -v test

# Check SQLAlchemy for missing eager loading on relationships
# (look for lazy loaded relationships accessed in async context)
grep -rn "lazy=" app/models/ --include="*.py"

# Profile a specific endpoint (optional, dev only)
# pip install pyinstrument
# uvicorn app.main:app & pyinstrument -m httpx GET http://localhost:8000/api/v1/products/
```

Report:
- Sync calls in async routes (blocks the event loop)
- Relationships with unsafe lazy loading for async

## Phase 8: API Schema Validation

```bash
# Generate OpenAPI schema
python -c "
import json
from app.main import app
schema = app.openapi()
with open('openapi.json', 'w') as f:
    json.dump(schema, f, indent=2)
print(f'Schema generated: {len(schema[\"paths\"])} paths')
"

# Validate the schema file is valid JSON
python -c "import json; json.load(open('openapi.json')); print('Valid JSON')"

# Optional: lint with spectral (if installed)
spectral lint openapi.json
```

## Phase 9: Diff Review

```bash
# Show diff statistics
git diff --stat

# Check for common issues in the diff
git diff | grep -n "print("           # Debug statements
git diff | grep -n "TODO\|FIXME\|HACK" # Open issues
git diff | grep -n "debug=True"        # Debug mode
git diff | grep -n "password\s*=\s*['\"]" # Hardcoded passwords
git diff | grep -n "secret\s*=\s*['\"]"   # Hardcoded secrets

# Show all changed files
git diff --name-only
```

Checklist:
- [ ] No `print()` debug statements
- [ ] No hardcoded secrets or credentials
- [ ] No `debug=True` in production config
- [ ] No `TODO`/`FIXME` in critical paths
- [ ] Alembic migration included for model changes
- [ ] Error handling present for all external calls
- [ ] New dependencies added to `pyproject.toml`

## Output Template

```
FASTAPI VERIFICATION REPORT
============================

Phase 1: Environment
  ✓ Python 3.12.2
  ✓ Virtual environment active
  ✓ DATABASE_URL, SECRET_KEY set

Phase 2: Code Quality
  ✓ mypy: no errors
  ✗ ruff: 2 issues (auto-fixed)
  ✓ ruff format: OK

Phase 3: Alembic Migrations
  ✓ Current: abc123 (head)
  ✓ No unapplied migrations
  ✓ alembic check: no new changes detected
  ✓ Downgrade/upgrade roundtrip OK

Phase 4: Tests + Coverage
  Tests: 134 passed, 0 failed, 3 skipped
  Coverage:
    Overall: 86%
    app/routers:  88%
    app/services: 91%
    app/auth:     94%

Phase 5: Security Scan
  ✓ pip-audit: no vulnerabilities
  ✗ bandit: 1 MEDIUM issue (B105 hardcoded_password_string — review)
  ✓ No secrets detected

Phase 6: Configuration
  ✓ debug = False
  ✓ secret_key set
  ✓ database_url set
  ✓ cors_origins not wildcard
  ✓ token expiry configured

Phase 7: Performance
  ✓ No sync blocking calls found
  ✓ Relationships using selectin loading

Phase 8: API Schema
  ✓ OpenAPI schema valid
  ✓ 18 paths exported

Phase 9: Diff Review
  Files changed: 7
  +210, -45 lines
  ✓ No debug statements
  ✓ No hardcoded secrets
  ✓ Migration included

RECOMMENDATION: ⚠️ Review bandit B105 finding before merging

NEXT STEPS:
1. Fix bandit MEDIUM finding
2. Re-run security scan
3. Merge PR
```

## Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Coverage ≥ 80%
- [ ] No security vulnerabilities (pip-audit)
- [ ] No HIGH bandit findings
- [ ] Alembic migrations up to date
- [ ] `debug = False`
- [ ] `secret_key` via env var (not hardcoded)
- [ ] CORS origins locked down (not `["*"]`)
- [ ] HTTPS enforced at reverse proxy
- [ ] Rate limiting on auth endpoints
- [ ] Database connection pooling configured
- [ ] Structured logging configured
- [ ] Error monitoring (Sentry etc.) configured
- [ ] Health check endpoint available (`/healthz`)
- [ ] OpenAPI schema reviewed (no sensitive data exposed)

## Continuous Integration (GitHub Actions)

```yaml
# .github/workflows/fastapi-verification.yml
name: FastAPI Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv sync --frozen

      - name: Code quality
        run: |
          uv run mypy app/ --ignore-missing-imports
          uv run ruff check .
          uv run ruff format . --check

      - name: Check migrations
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/test_db
        run: |
          uv run alembic upgrade head
          uv run alembic check

      - name: Run tests
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:postgres@localhost:5432/test_db
          SECRET_KEY: ci-test-secret-key-not-for-production
        run: uv run pytest --cov=app --cov-report=xml --cov-fail-under=80

      - name: Security scan
        run: |
          uv run pip-audit
          uv run bandit -r app/ -ll  # only MEDIUM+ findings

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
```

## Quick Reference

| Check | Command |
|-------|---------|
| Type check | `mypy app/` |
| Lint + format | `ruff check . && ruff format . --check` |
| Migration state | `alembic current && alembic check` |
| Apply migrations | `alembic upgrade head` |
| Tests + coverage | `pytest --cov=app` |
| Dependency CVEs | `pip-audit` |
| Security lint | `bandit -r app/` |
| Secret scan | `gitleaks detect --source .` |
| Diff stats | `git diff --stat` |

Remember: Automated verification catches common issues but does not replace manual code review and staging environment testing.
