# Check if pyenv is available and add shims to PATH if needed
SHELL := /bin/bash
PYENV_ROOT ?= $(HOME)/.pyenv
export PATH := $(PYENV_ROOT)/shims:$(PYENV_ROOT)/bin:$(PATH)

# Map the desired dbr version to that LTS version of python
PYTHON_DBR154=python3.11
PYTHON_DBR143=python3.10
PYTHON_DBR133=python3.10
PYTHON_DBR122=python3.9
PYTHON_DBR113=python3.9

# All DBR versions use Java 8 to match Databricks Runtime environments
JAVA_VERSION=8.0.422-zulu
# SDKMAN Java path
SDKMAN_DIR ?= $(HOME)/.sdkman
JAVA_HOME_TARGET=$(SDKMAN_DIR)/candidates/java/$(JAVA_VERSION)

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
	@echo "  make check-env             Check Python and Java environment setup"
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
	@echo "  154 - Python 3.11, Java 8 (default)"
	@echo "  143 - Python 3.10, Java 8"
	@echo "  133 - Python 3.10, Java 8"
	@echo "  122 - Python 3.9, Java 8"
	@echo "  113 - Python 3.9, Java 8"
	@echo ""
	@echo "Environment Management:"
	@echo "  - Python versions managed via pyenv"
	@echo "  - Java 8 (Zulu) managed via SDKMAN"
	@echo "  - All DBR versions use Java 8 to match Databricks Runtime"
	@echo ""

# Check if Java 8 is available and install if needed
.PHONY: check-java
check-java:
	@# Check if SDKMAN is installed
	@if [ ! -f "$(SDKMAN_DIR)/bin/sdkman-init.sh" ]; then \
		echo "SDKMAN not found. Installing SDKMAN..."; \
		curl -s "https://get.sdkman.io" | bash; \
	fi
	@# Source SDKMAN and check for Java 8
	@if [ ! -d "$(JAVA_HOME_TARGET)" ]; then \
		echo "Installing Zulu OpenJDK 8 via SDKMAN..."; \
		bash -c "source $(SDKMAN_DIR)/bin/sdkman-init.sh && sdk install java $(JAVA_VERSION)"; \
	fi
	@# Verify Java installation
	@if [ -d "$(JAVA_HOME_TARGET)" ]; then \
		export JAVA_HOME=$(JAVA_HOME_TARGET) && \
		export PATH=$(JAVA_HOME_TARGET)/bin:$$PATH && \
		echo "Java version check:" && \
		$(JAVA_HOME_TARGET)/bin/java -version && \
		echo "JAVA_HOME set to: $(JAVA_HOME_TARGET)"; \
	else \
		echo "Warning: Java 8 installation may have failed. Using system Java."; \
		java -version 2>/dev/null || echo "No Java found!"; \
	fi

# Check if pyenv is available
.PHONY: check-pyenv
check-pyenv:
	@command -v pyenv >/dev/null 2>&1 || { echo "Error: pyenv is not installed. Please install pyenv first."; exit 1; }
	@echo "pyenv is available at: $$(which pyenv)"
	@echo "PATH is: $$PATH"

# Check both Python and Java environments
.PHONY: check-env
check-env: check-pyenv check-java

# Create virtual environment if it doesn't exist
# Ensure the virtual environment is created
.PHONY: create-env
venv: check-env
	@echo "Creating virtual environment for DBR $(DBR)..."
	@(pyenv versions | grep -q " 3.9" || (echo "Installing Python 3.9..." && pyenv install 3.9))
	@(pyenv versions | grep -q " 3.10" || (echo "Installing Python 3.10..." && pyenv install 3.10))
	@(pyenv versions | grep -q " 3.11" || (echo "Installing Python 3.11..." && pyenv install 3.11))
	pyenv local 3.9 3.10 3.11
	export JAVA_HOME=$(JAVA_HOME_TARGET) && \
		export PATH=$(JAVA_HOME_TARGET)/bin:$$PATH && \
		tox --devenv $(VENV) -e dbr${DBR}

# Run specific environment tests
.PHONY: test
test: check-env
	@echo "Running tests in DBR $(DBR)..."
	@# Ensure pyenv Python versions are available
	@(pyenv versions | grep -q " 3.9" || (echo "Installing Python 3.9..." && pyenv install 3.9))
	@(pyenv versions | grep -q " 3.10" || (echo "Installing Python 3.10..." && pyenv install 3.10))
	@(pyenv versions | grep -q " 3.11" || (echo "Installing Python 3.11..." && pyenv install 3.11))
	@pyenv local 3.9 3.10 3.11
	export JAVA_HOME=$(JAVA_HOME_TARGET) && \
		export PATH=$(JAVA_HOME_TARGET)/bin:$$PATH && \
		tox -e dbr${DBR}

.PHONY: test-all
test-all: check-env
	@echo "Running tests for all environments"
	export JAVA_HOME=$(JAVA_HOME_TARGET) && \
		export PATH=$(JAVA_HOME_TARGET)/bin:$$PATH && \
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
