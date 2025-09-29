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

# Load version constraints
-include .env.versions

# Default target
.PHONY: help
help:
	@echo "Makefile for managing test environments"
	@echo ""
	@echo "Usage:"
	@echo "  make init                  Complete setup for new developers (recommended)"
	@echo "  make status                Quick status check"
	@echo "  make doctor                Run comprehensive environment health check"
	@echo "  make check-env             Check Python and Java environment setup"
	@echo "  make venv                  Create virtual environment for newest, supported DBR version"
	@echo "  make venv DBR=<version>    Create virtual environment for a specific DBR version"
	@echo ""
	@echo "Testing:"
	@echo "  make test                  Run tests for the newest, supported DBR version"
	@echo "  make test DBR=<version>    Run tests for a specific DBR version"
	@echo "  make test-all              Run tests for all supported DBR versions"
	@echo "  make coverage-report       Generate test coverage report (requires test run first)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make quality               Run all code quality checks (format + lint + type-check)"
	@echo "  make lint                  Run linters"
	@echo "  make format                Format code with Black"
	@echo "  make type-check            Run type checks"
	@echo ""
	@echo "Build:"
	@echo "  make build                 Build distribution and documentation"
	@echo "  make build-dist            Build distribution only"
	@echo "  make build-docs            Build documentation only"
	@echo ""
	@echo "Complete CI:"
	@echo "  make ci                    Run complete CI pipeline (quality + test-all + build + coverage)"
	@echo "  make all                   Alias for 'make ci'"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                 Remove all generated files"
	@echo "  make deep-clean            Remove everything (venvs, caches, artifacts)"
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

# Setup development environment
.PHONY: setup
setup:
	@echo "Setting up development environment..."
	@if [ -f .env.versions ]; then \
		. .env.versions && \
		python -m pip install --upgrade pip --quiet && \
		python -m pip install hatch==$${HATCH_VERSION} "$${CLICK_CONSTRAINT}" --quiet && \
		echo "Development tools installed successfully"; \
	else \
		echo "ERROR: .env.versions file not found"; \
		exit 1; \
	fi

# Complete initialization for new developers - one command to set up everything
.PHONY: init
init:
	@echo " Initializing complete development environment..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@$(MAKE) setup
	@echo ""
	@$(MAKE) check-env
	@echo ""
	@echo " Creating default virtual environment..."
	@$(MAKE) venv
	@echo ""
	@echo " Running smoke test..."
	@$(MAKE) lint 2>&1 | head -5
	@echo ""
	@echo " Development environment initialization complete!"
	@echo ""
	@echo " Next steps:"
	@echo "  1. Review CONTRIBUTING.md for detailed documentation"
	@echo "  2. Run 'make test' to run tests"
	@echo "  3. Run 'make help' to see all available commands"

# Quick status check
.PHONY: status
status:
	@# Check development tools
	@if [ -f .env.versions ]; then \
		. .env.versions && \
		if command -v hatch >/dev/null 2>&1; then \
			HATCH_VER=$$(hatch --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'); \
			if [ "$${HATCH_VER}" = "$${HATCH_VERSION}" ] && python -c "import click; exit(0 if tuple(map(int, click.__version__.split('.'))) < (8, 3, 0) else 1)" 2>/dev/null; then \
				echo "Development tools: OK"; \
			else \
				echo "Development tools: needs update (run 'make setup')"; \
			fi; \
		else \
			echo "Development tools: not installed (run 'make setup')"; \
		fi; \
	else \
		echo "Development tools: config missing"; \
	fi
	@# Check Java
	@if [ -d "$(JAVA_HOME_TARGET)" ]; then \
		echo "Java 8: OK"; \
	else \
		echo "Java 8: not installed"; \
	fi
	@# Check virtual environments
	@if [ -d "${VENV_DIR}" ]; then \
		COUNT=$$(ls -1 ${VENV_DIR} | wc -l | tr -d ' '); \
		echo "Virtual environments: $${COUNT}"; \
	else \
		echo "Virtual environments: none (run 'make venv')"; \
	fi

# Comprehensive environment health check
.PHONY: doctor
doctor:
	@echo " Running comprehensive environment health check..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo " System Information:"
	@echo "  OS: $$(uname -s) $$(uname -m)"
	@echo "  Shell: $$SHELL"
	@echo "  User: $$USER"
	@echo ""
	@echo " Python Status:"
	@which python >/dev/null 2>&1 && echo "   python: $$(which python) ($$(python --version 2>&1))" || echo "   python: not found"
	@which python3 >/dev/null 2>&1 && echo "   python3: $$(which python3) ($$(python3 --version 2>&1))" || echo "   python3: not found"
	@echo ""
	@$(MAKE) check-deps 2>&1 | sed 's/^/  /'
	@echo ""
	@echo " Java Status:"
	@if [ -d "$(JAVA_HOME_TARGET)" ]; then \
		echo "   Java 8 installed at $(JAVA_HOME_TARGET)"; \
	else \
		echo "    Java 8 not installed (will be installed when needed)"; \
	fi
	@echo ""
	@echo " Virtual Environments:"
	@if [ -d "${VENV_DIR}" ]; then \
		for env in ${VENV_DIR}/*; do \
			if [ -d "$$env" ]; then \
				echo "   $$(basename $$env)"; \
			fi; \
		done; \
	else \
		echo "    No virtual environments found"; \
	fi
	@echo ""
	@echo " Diagnosis Complete"
	@if command -v hatch >/dev/null 2>&1 && python -c "import click; exit(0 if tuple(map(int, click.__version__.split('.'))) < (8, 3, 0) else 1)" 2>/dev/null; then \
		echo "   Environment is healthy"; \
	else \
		echo "    Issues detected - run 'make init' to fix"; \
	fi

# Run all code quality checks (format + lint + type-check)
.PHONY: quality
quality: format lint type-check
	@echo "✅ All quality checks passed!"


# Deep clean - remove everything including venvs
.PHONY: deep-clean
deep-clean: clean
	@echo " Deep cleaning - removing all artifacts..."
	@rm -rf ${VENV_DIR}/
	@rm -rf .tox/
	@rm -rf __pycache__/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .coverage*
	@rm -rf htmlcov/
	@rm -rf dist/
	@rm -rf build/
	@rm -rf *.egg-info/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo " Deep clean complete!"

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

# Check dependencies (hatch and click versions)
.PHONY: check-deps
check-deps:
	@echo "Checking development dependencies..."
	@if [ -f .env.versions ]; then \
		. .env.versions && \
		echo "Expected versions from .env.versions:" && \
		echo "  - hatch: $${HATCH_VERSION}" && \
		echo "  - click: < 8.3.0" && \
		echo "" && \
		if command -v hatch >/dev/null 2>&1; then \
			INSTALLED_HATCH=$$(hatch --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+') && \
			echo "Installed hatch: $${INSTALLED_HATCH}" && \
			if [ "$${INSTALLED_HATCH}" != "$${HATCH_VERSION}" ]; then \
				echo "  Warning: hatch version mismatch! Expected $${HATCH_VERSION}, got $${INSTALLED_HATCH}"; \
				echo "  Run 'make setup' to install the correct version"; \
			else \
				echo " Hatch version is correct"; \
			fi; \
		else \
			echo " hatch not found! Run 'make setup' to install"; \
		fi && \
		if python -c "import click" 2>/dev/null; then \
			INSTALLED_CLICK=$$(python -c "import click; print(click.__version__)") && \
			echo "Installed click: $${INSTALLED_CLICK}" && \
			if python -c "import click; exit(0 if tuple(map(int, click.__version__.split('.'))) < (8, 3, 0) else 1)"; then \
				echo "Click version is safe (< 8.3.0)"; \
			else \
				echo "  Warning: Click version $${INSTALLED_CLICK} may cause issues!"; \
				echo "  See: https://github.com/pypa/hatch/issues/2050"; \
				echo "  Run 'make setup' to install the correct version"; \
			fi; \
		else \
			echo " click not found! Run 'make setup' to install"; \
		fi; \
	else \
		echo " Error: .env.versions file not found"; \
		echo "  Cannot verify dependency versions"; \
	fi

# Check both Python, Java environments, and dependencies
.PHONY: check-env
check-env: check-python check-java check-deps

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
	@echo "Running tests for all DBR versions..."
	@$(MAKE) setup-all-python-versions
	export JAVA_HOME=$(JAVA_HOME_TARGET) && \
		export PATH=$(JAVA_HOME_TARGET)/bin:$$PATH && \
		for env in dbr154 dbr143 dbr133 dbr122 dbr113; do \
			echo "Testing $$env..." && \
			hatch run $$env:test || exit 1; \
		done
	@echo "✅ All tests passed!"

# Run linters
.PHONY: lint
lint:
	@echo "Running linters..."
	hatch run lint:runLint

# Format code with Black
.PHONY: format
format:
	@echo "Formatting code with Black..."
	hatch run lint:formatBlack

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
	@echo "✅ Coverage report generated!"


# Build both distribution and documentation
.PHONY: build
build: build-dist build-docs
	@echo "✅ Build complete!"

# Run complete CI pipeline
.PHONY: ci
ci: quality test-all build coverage-report
	@echo "✅ Complete CI pipeline passed!"

# Alias for ci
.PHONY: all
all: ci

# Clean virtual environments and build artifacts
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf .coverage .coverage.* build dist
