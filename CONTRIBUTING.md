# Contributing

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Quality Checks

Run these before opening a change:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov
```

## Guidelines

- Keep changes small and focused.
- Preserve the existing module scripts unless intentionally migrating users to the package CLI.
- Mock Ollama, Wikimedia, QLever, and Java/Triplify calls in tests.
- Do not commit local `.env`, generated outputs, virtual environments, archives, or logs.
