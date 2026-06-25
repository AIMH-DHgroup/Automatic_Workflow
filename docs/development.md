# Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run quality checks:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov
```

Use `.env.example` as the template for local environment variables.
