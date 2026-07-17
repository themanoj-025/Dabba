# Contributing to Dabba

Thank you for your interest in contributing! This document provides guidelines for contributing to the Dabba project.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `make setup`
4. Install pre-commit hooks: `pre-commit install`

## Code Style

- **Formatter:** Black (line length 88)
- **Import sorting:** isort (profile = black)
- **Linter:** ruff
- **Type hints:** Required on all function signatures
- **Docstrings:** Google style on all public functions/classes

Run `make format` before committing to auto-format your code.

## Testing

- Write tests for any new functionality
- Run the full test suite before submitting: `make test`
- Aim for meaningful coverage, not just high percentages

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commits
3. Update documentation if needed
4. Ensure all tests pass and linters are clean
5. Submit a pull request with a clear description of changes

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bug reports
- Tag issues appropriately (bug, enhancement, question)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
