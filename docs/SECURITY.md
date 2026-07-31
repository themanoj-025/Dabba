# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.x     | ✅ Active development |

## Reporting a Vulnerability

Dabba is a portfolio/educational project, but we take security seriously.

If you discover a security vulnerability, please **do not** open a public issue.
Instead, report it privately via:

1. **Email**: [themanoj-025](https://github.com/themanoj-025) (GitHub contact)
2. **GitHub Security Advisory**: Use the "Report a vulnerability" link under the
   repository's "Security" tab.

### What to include
- A clear description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential mitigations you've identified

### Response timeline
- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 5 business days
- **Fix timeline**: Depends on severity (typically 7-30 days for critical issues)

## Security Practices

### API Authentication
- All `/v1/*` endpoints require an `X-API-Key` header when `DABBA_API_KEY` is configured
- The `/health` endpoint is intentionally unauthenticated for health-check access
- Rate limiting via `slowapi` prevents brute-force and DoS attacks

### Input Validation
- All API inputs are validated via Pydantic schemas with strict type/bounds checks
- User-echoed text is HTML-escaped to prevent XSS
- File upload/download paths are restricted to configured directories

### Dependencies
- All Python dependencies are version-pinned in `pyproject.toml`
- Secrets (`kaggle.json`, `.env`) are gitignored
- Docker images are pinned to specific base image tags

### Infrastructure
- Docker containers run with least-privilege user where possible
- Security headers set on all API responses (CSP, X-Frame-Options, etc.)
- CORS restricted to known origins

## Known Security Considerations

1. **LLM API keys** stored in environment variables — use secret management in production
2. **SQLite** is used for development; migrate to Postgres with proper auth for production
3. **Rate limiting** is basic — production deployments should add a WAF/API gateway
