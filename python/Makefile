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
	@echo "  - Python versions: uses system Python if available, otherwise installs pyenv"
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

# Check if required Python version for current DBR is available, install pyenv if needed
.PHONY: check-python
check-python:
	@echo "Checking Python environment for DBR $(DBR)..."
	@REQUIRED_PYTHON=$(PYTHON_DBR$(DBR)); \
	if command -v $$REQUIRED_PYTHON >/dev/null 2>&1; then \
		echo "Required Python version ($$REQUIRED_PYTHON) found in system"; \
		echo "Using system Python: $$(which $$REQUIRED_PYTHON)"; \
	elif command -v pyenv >/dev/null 2>&1; then \
		echo "pyenv is available at: $$(which pyenv)"; \
		echo "Using pyenv for Python version management"; \
	else \
		echo "Required Python version ($$REQUIRED_PYTHON) not found in system and pyenv not installed"; \
		echo "Installing pyenv..."; \
		curl https://pyenv.run | bash; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		echo "Configuring shell for pyenv..."; \
		if [ "$$SHELL" = "/bin/zsh" ] || [ "$$SHELL" = "/usr/bin/zsh" ]; then \
			echo 'export PATH="$$HOME/.pyenv/bin:$$PATH"' >> ~/.zshrc; \
			echo 'eval "$$(pyenv init -)"' >> ~/.zshrc; \
			echo "Added pyenv configuration to ~/.zshrc"; \
		elif [ "$$SHELL" = "/bin/bash" ] || [ "$$SHELL" = "/usr/bin/bash" ]; then \
			echo 'export PATH="$$HOME/.pyenv/bin:$$PATH"' >> ~/.bashrc; \
			echo 'eval "$$(pyenv init -)"' >> ~/.bashrc; \
			echo "Added pyenv configuration to ~/.bashrc"; \
		else \
			echo "Unknown shell ($$SHELL). Please manually add the following to your shell config:"; \
			echo '  export PATH="$$HOME/.pyenv/bin:$$PATH"'; \
			echo '  eval "$$(pyenv init -)"'; \
		fi; \
		echo "pyenv installed successfully"; \
	fi

# Check both Python and Java environments
.PHONY: check-env
check-env: check-python check-java

# Setup Python version for current DBR using either system Python or pyenv
.PHONY: setup-python-versions
setup-python-versions:
	@echo "Setting up Python version for DBR $(DBR)..."
	@REQUIRED_PYTHON=$(PYTHON_DBR$(DBR)); \
	PYTHON_VERSION=$${REQUIRED_PYTHON#python}; \
	if command -v $$REQUIRED_PYTHON >/dev/null 2>&1; then \
		echo "Using system Python: $$(which $$REQUIRED_PYTHON)"; \
	elif command -v pyenv >/dev/null 2>&1; then \
		echo "Using pyenv to install and configure Python $$PYTHON_VERSION"; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		(pyenv versions | grep -q " $$PYTHON_VERSION" || (echo "Installing Python $$PYTHON_VERSION..." && pyenv install $$PYTHON_VERSION)); \
		pyenv local $$PYTHON_VERSION; \
	else \
		echo "Error: Required Python version ($$REQUIRED_PYTHON) not available and pyenv not found"; \
		exit 1; \
	fi

# Setup all Python versions for test-all target
.PHONY: setup-all-python-versions
setup-all-python-versions:
	@echo "Setting up all Python versions for test-all..."
	@if command -v python3.9 >/dev/null 2>&1 && command -v python3.10 >/dev/null 2>&1 && command -v python3.11 >/dev/null 2>&1; then \
		echo "All required Python versions (3.9, 3.10, 3.11) found in system"; \
		echo "Using system Python versions"; \
	elif command -v pyenv >/dev/null 2>&1; then \
		echo "Using pyenv to install and configure all Python versions"; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		(pyenv versions | grep -q " 3.9" || (echo "Installing Python 3.9..." && pyenv install 3.9)); \
		(pyenv versions | grep -q " 3.10" || (echo "Installing Python 3.10..." && pyenv install 3.10)); \
		(pyenv versions | grep -q " 3.11" || (echo "Installing Python 3.11..." && pyenv install 3.11)); \
		pyenv local 3.9 3.10 3.11; \
	else \
		echo "Error: Python versions not available and pyenv not found"; \
		echo "Installing pyenv..."; \
		curl https://pyenv.run | bash; \
		export PATH="$$HOME/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
		echo "Installing required Python versions..."; \
		pyenv install 3.9; \
		pyenv install 3.10; \
		pyenv install 3.11; \
		pyenv local 3.9 3.10 3.11; \
	fi

# Create virtual environment if it doesn't exist
# Ensure the virtual environment is created
.PHONY: create-env
venv: check-env
	@echo "Creating virtual environment for DBR $(DBR)..."
	@$(MAKE) setup-python-versions
	export JAVA_HOME=$(JAVA_HOME_TARGET) && \
		export PATH=$(JAVA_HOME_TARGET)/bin:$$PATH && \
		hatch shell dbr${DBR}

# Run specific environment tests
.PHONY: test
test: check-env
	@echo "Running tests in DBR $(DBR)..."
	@$(MAKE) setup-python-versions
	export JAVA_HOME=$(JAVA_HOME_TARGET) && \
		export PATH=$(JAVA_HOME_TARGET)/bin:$$PATH && \
		hatch run dbr${DBR}:test

.PHONY: test-all
test-all: check-java
	@$(MAKE) setup-all-python-versions
	export JAVA_HOME=$(JAVA_HOME_TARGET) && \
		export PATH=$(JAVA_HOME_TARGET)/bin:$$PATH && \
		for env in dbr154 dbr143 dbr133 dbr122 dbr113; do hatch run $$env:test; done

# Run linters
.PHONY: lint
lint:
	@echo "Running linters..."
	hatch run lint:runLint $()

# Run type checks
.PHONY: type-check
type-check: 
	@echo "Running type checks..."
	hatch run type-check:check

# Build distribution
.PHONY: build-dist
build-dist: 
	@echo "Building distribution..."
	hatch run build-dist:build

# Build documentation
.PHONY: build-docs
build-docs:
	@echo "Building documentation..."
	hatch run build-docs:build

# Combine coverage data and generate report
.PHONY: coverage-report
coverage-report:
	@if [ ! -f .coverage ] && ! ls .coverage.* 1> /dev/null 2>&1; then \
		echo "Error: No coverage data found (.coverage file missing)"; \
		echo "Run tests first with: make test or make test-all"; \
		exit 1; \
	fi
	@if ls .coverage.* 1> /dev/null 2>&1 && [ ! -f .coverage ]; then \
		echo "Combining parallel coverage files..."; \
		hatch run coverage-report:coverage combine; \
	fi
	@echo "Generating coverage report..."
	hatch run coverage-report:run-coverage


.PHONY: all-local
all-local: lint type-check build-dist build-docs coverage-report

# Clean virtual environments and build artifacts
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf .coverage .coverage.* build dist
