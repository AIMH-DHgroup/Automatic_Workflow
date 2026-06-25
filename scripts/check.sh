#!/usr/bin/env sh
set -eu

ruff check src tests
ruff format --check src tests
mypy src
pytest --cov
