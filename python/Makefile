# Tempo developer & CI toolchain, driven by uv.
#
# uv manages the Python interpreter, the locked virtual environment, and the
# build backend. See CONTRIBUTING.md for developer setup (`brew install uv`).
SHELL := /bin/bash

# Ensure that all uv commands don't automatically update the lock file: instead they use the locked dependencies.
export UV_FROZEN := 1

# Pin the build backend (hatchling) with hashes when building distributions.
export UV_BUILD_CONSTRAINT := .build-constraints.txt

# Optional developer-local overrides (git-ignored), e.g. an internal package
# index for environments that cannot reach the public PyPI. See CONTRIBUTING.md.
-include local.mk

# When a network target runs and public PyPI (pypi.org) is unreachable, fall
# back to a developer-configured index (DATABRICKS_PYPI_PROXY, set in local.mk
# or the environment) so uv can still resolve locally. This does nothing unless
# that index is configured, so CI runners and external contributors reach the
# public index and are unaffected. `lock-dependencies` scrubs the committed
# uv.lock back to pypi.org, so a private index never leaks into the lockfile.
# Export UV_DEFAULT_INDEX yourself to override the probe.
NETWORK_GOALS := dev setup lock-dependencies build build-dist build-docs test coverage-report
ifneq ($(filter $(NETWORK_GOALS),$(MAKECMDGOALS)),)
ifndef UV_DEFAULT_INDEX
ifneq ($(strip $(DATABRICKS_PYPI_PROXY)),)
ifneq ($(shell curl -fsS -m 3 -o /dev/null https://pypi.org/simple/ 2>/dev/null && echo up),up)
export UV_DEFAULT_INDEX := $(DATABRICKS_PYPI_PROXY)
$(info make: public PyPI unreachable -- falling back to the developer-configured index.)
endif
endif
endif
endif

# Run a tool inside an isolated dependency group (no project install needed for
# pure source checks). Test/doc targets install the project explicitly.
UV_LINT := uv run --no-project --no-default-groups
# Run a tool with the project installed and the `test` group (Spark stack).
UV_TEST := uv run --no-default-groups --group test

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Tempo Makefile (uv-based)"
	@echo ""
	@echo "Setup:"
	@echo "  make dev               Sync the locked development environment"
	@echo "  make setup             Alias for 'make dev' (used by CI)"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint              Check formatting (black) and lint (flake8)"
	@echo "  make format            Auto-format the code with black"
	@echo "  make type-check        Run mypy"
	@echo "  make quality           format + lint + type-check"
	@echo ""
	@echo "Testing:"
	@echo "  make test              Run the unit test suite under coverage"
	@echo "  make coverage-report   Emit the coverage report and coverage.xml"
	@echo ""
	@echo "Build:"
	@echo "  make build-dist        Build sdist + wheel (hash-pinned backend)"
	@echo "  make build-docs        Build the Sphinx documentation"
	@echo "  make build             build-dist + build-docs"
	@echo ""
	@echo "Maintenance:"
	@echo "  make lock-dependencies Regenerate uv.lock and .build-constraints.txt"
	@echo "  make clean             Remove caches and build artifacts"

# Sync the locked development environment (all dev dependency groups).
.PHONY: dev
dev:
	uv sync

# Alias used by the CI workflows.
.PHONY: setup
setup: dev

.PHONY: lint
lint:
	$(UV_LINT) --group lint black --check --diff ./tempo
	$(UV_LINT) --group lint flake8 --config ./.flake8 ./tempo

.PHONY: format
format:
	$(UV_LINT) --group lint black ./tempo ./tests

.PHONY: type-check
type-check:
	$(UV_LINT) --group type-check mypy --install-types --non-interactive ./tempo

.PHONY: quality
quality: format lint type-check
	@echo "All quality checks passed!"

# Run the unit tests under coverage. The project must be installed, so this uses
# the `test` group (which includes the Spark stack) with the project.
.PHONY: test
test:
	$(UV_TEST) coverage erase
	$(UV_TEST) coverage run -m unittest discover -s tests -p '*_tests.py'

.PHONY: coverage-report
coverage-report:
	$(UV_TEST) coverage report -m
	$(UV_TEST) coverage xml

# Build the distribution with the build backend pinned by hashes.
.PHONY: build-dist
build-dist:
	uv build --require-hashes --build-constraints=.build-constraints.txt

# Sphinx autodoc imports tempo, so the docs build needs the Spark stack too.
.PHONY: build-docs
build-docs:
	uv run --no-default-groups --group docs --group test $(MAKE) --directory ../docs html

.PHONY: build
build: build-dist build-docs
	@echo "Build complete!"

# Regenerate the lock file and the pinned build-backend constraints. This is the
# only target permitted to update uv.lock, so UV_FROZEN is cleared here.
.PHONY: lock-dependencies
lock-dependencies: UV_FROZEN := 0
lock-dependencies:
	@touch .build-constraints.txt
	uv lock
	uv run --no-default-groups --group yq tomlq -r '.["build-system"].requires[]' pyproject.toml | \
		uv pip compile --generate-hashes --universal --no-header - > build-constraints-new.txt
	mv build-constraints-new.txt .build-constraints.txt
	@perl -pi \
		-e 's|registry = "https://[^"]*"|registry = "https://pypi.org/simple"|g;' \
		-e 's|url = "https://[^"]*/packages/([^"]*)"|url = "https://files.pythonhosted.org/packages/$$1"|g;' \
		uv.lock
	@printf 'Stripped registry references from uv.lock.\n'

.PHONY: clean
clean:
	rm -rf .venv htmlcov .mypy_cache .pytest_cache .ruff_cache .coverage .coverage.* coverage.xml build dist
	find . -name '__pycache__' -type d -prune -print0 | xargs -0 rm -rf
