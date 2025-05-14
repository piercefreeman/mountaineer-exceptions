.PHONY: test lint lint-py lint-js lint-validation build clean

# Default directories
PYTHON_SRC := mountaineer_exceptions
JS_SRC := $(PYTHON_SRC)/views

# Testing
test:
	uv run pytest

# Linting
lint: lint-py lint-js

lint-py:
	uv run ruff format $(PYTHON_SRC)
	uv run ruff check --fix $(PYTHON_SRC)

lint-js:
	cd $(JS_SRC) && npm run lint

# Lint validation
lint-validation:
	echo "Running lint validation for $(PYTHON_SRC)..."
	@(cd . && uv run ruff format --check $(PYTHON_SRC))
	@(cd . && uv run ruff check $(PYTHON_SRC))
	echo "Running pyright for $(PYTHON_SRC)..."
	@(cd . && uv run pyright $(PYTHON_SRC))

# Building
build:
	uv run build
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
