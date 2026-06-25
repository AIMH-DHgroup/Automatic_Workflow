ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ruff format --check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
mypy src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
pytest --cov
