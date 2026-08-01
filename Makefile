.PHONY: prepare test lint lint-py lint-js lint-validation build clean

# Default directories
PYTHON_SRC := mountaineer_exceptions
JS_SRC := $(PYTHON_SRC)/views
MOUNTAINEER_VERSION := 0.20.0.dev1

prepare:
	uv sync
	uv pip install "mountaineer==$(MOUNTAINEER_VERSION)"
	uv pip install --no-deps --editable .

# Testing
test: prepare
	uv run --no-sync pytest

# Linting
lint: lint-py lint-js

lint-py: prepare
	uv run --no-sync ruff format $(PYTHON_SRC)
	uv run --no-sync ruff check --fix $(PYTHON_SRC)

lint-js:
	cd $(JS_SRC) && npm run lint

# Lint validation
lint-validation: prepare
	echo "Running lint validation for $(PYTHON_SRC)..."
	@(cd . && uv run --no-sync ruff format --check $(PYTHON_SRC))
	@(cd . && uv run --no-sync ruff check $(PYTHON_SRC))
	echo "Running pyright for $(PYTHON_SRC)..."
	@(cd . && uv run --no-sync pyright $(PYTHON_SRC))

# Building
build: prepare
	uv run --no-sync build-exceptions
	uv build

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Help command
help:
	@echo "Available commands:"
	@echo "  make test          - Run pytest"
	@echo "  make lint          - Run all linters"
	@echo "  make lint-py       - Run Python linter (ruff)"
	@echo "  make lint-js       - Run JavaScript linter"
	@echo "  make lint-validation - Run lint validation with ruff format check and pyright"
	@echo "  make build         - Build the package"
	@echo "  make clean         - Remove build artifacts"
