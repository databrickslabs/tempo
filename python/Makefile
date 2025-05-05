# Map the desired dbr version to that LTS version of python
PYTHON_DBR154=python3.11
PYTHON_DBR143=python3.10
PYTHON_DBR133=python3.10
PYTHON_DBR122=python3.9
PYTHON_DBR113=python3.9

# Default to 154 for now
DBR ?= 154
TEST_ENVS = dbr154,dbr143,dbr133,dbr122,dbr113
PYTHON_VERSION=$(PYTHON_DBR$(DBR))

# Virtual environment path. Hopefully can use this to manage venvs a bit better once we remove tox
VENV_DIR=.venv
VENV=$(VENV_DIR)/dbr$(DBR)
COVERAGE_FILE=.coverage.dbr$(DBR)

# Default target
.PHONY: help
help:
	@echo "Makefile for managing test environments"
	@echo ""
	@echo "Usage:"
	@echo "  make venv                  Create virtual environment for newest, supported DBR version"
	@echo "  make venv DBR=<version>    Create virtual environment for a specific DBR version"
	@echo "  make test                  Run tests for the newest, supported DBR version"
	@echo "  make test DBR=<version>    Run tests for a specific DBR version"
	@echo "  make test-all              Run tests for all supported DBR versions"
	@echo "  make coverage-report       Generate test coverage report"
	@echo "  make lint                  Run linters"
	@echo "  make type-check            Run type checks"
	@echo "  make build-dist            Build distribution"
	@echo "  make build-docs            Build documentation"
	@echo "  make all-local             Run all local tests and checks"
	@echo "  make clean                 Remove all generated files"
	@echo ""
	@echo "Available Environment Versions:"
	@echo "  154 - Python 3.11 (default)"
	@echo "  143 - Python 3.10"
	@echo "  133 - Python 3.10"
	@echo "  122 - Python 3.9"
	@echo "  113 - Python 3.9"
	@echo ""

# Create virtual environment if it doesn't exist
# Ensure the virtual environment is created
.PHONY: create-env
venv:
	@echo "Creating virtual environment for DBR $(DBR)..."
	@(pyenv versions | grep -q " 3.9" || (echo "Installing Python 3.9..." && pyenv install 3.9))
	@(pyenv versions | grep -q " 3.10" || (echo "Installing Python 3.10..." && pyenv install 3.10))
	@(pyenv versions | grep -q " 3.11" || (echo "Installing Python 3.11..." && pyenv install 3.11))
	pyenv local 3.9 3.10 3.11
	tox --devenv .venv -e dbr${DBR}

# Run specific environment testtests
.PHONY: test
test:
	@echo "Running tests in DBR $(DBR)..."
	tox -e dbr${DBR}

.PHONY: test-all
test-all:
	@echo "Running tests for all environments"
	tox -e ${TEST_ENVS}

# Run linters
.PHONY: lint
lint:
	@echo "Running linters..."
	tox -e lint

# Run type checks
.PHONY: type-check
type-check: 
	@echo "Running type checks..."
	tox -e type-check	

# Build distribution
.PHONY: build-dist
build-dist: 
	@echo "Building distribution..."
	tox -e build-dist

# Build documentation
.PHONY: build-docs
build-docs:
	@echo "Building documentation..."
	tox -e build-docs

# Combine coverage data and generate report
.PHONY: coverage-report
coverage-report:
	@echo "Generating coverage report..."
	tox -e coverage-report


.PHONY: all-local
all-local: lint type-check build-dist build-docs coverage-report

# Clean virtual environments and build artifacts
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf .coverage .coverage.* build dist
