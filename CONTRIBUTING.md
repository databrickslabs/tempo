# Contributing to Tempo

## Quick Start

```bash
cd tempo/python
make dev      # Sync the locked development environment
make quality  # Format, lint, and type-check
make test     # Run the unit test suite
```

## Development Setup

Tempo uses [uv](https://docs.astral.sh/uv/) to manage the Python interpreter,
the locked virtual environment, and the build backend. uv also enforces the
dependency lockdown for the project (every dependency is pinned with a
cryptographic hash in `uv.lock`).

### Prerequisites
- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- `make`
- A JDK (Java 8 recommended, to match the Databricks Runtime)

You do **not** need to install Python yourself: uv downloads and manages the
interpreter declared in `python/.python-version`.

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/databrickslabs/tempo.git
   cd tempo/python
   ```

2. **Sync the development environment**
   ```bash
   make dev
   ```
   This creates `.venv` from `uv.lock` with every development dependency group
   (test, lint, type-check, docs).

## Running Tests

```bash
make test             # Run the unit + pytest suites under coverage
make coverage-report  # Emit the coverage report and coverage.xml
```

Select the Python interpreter for a run with `UV_PYTHON` (CI tests 3.9–3.11):

```bash
UV_PYTHON=3.10 make test
```

### A note on the DBR matrix

Tempo previously ran a test matrix that pinned a **different** Spark / numpy /
pandas / pyarrow set for each Databricks Runtime (DBR 11.3 → 15.4), each on its
own Python version, via Hatch environments.

That model does not translate cleanly to the supply-chain lockdown, which
requires a single `uv.lock` where every dependency is pinned with a hash:

- A single universal lock resolves each dependency group across the **whole**
  supported Python range. The older DBRs pin old numpy/pandas/pyarrow whose
  wheels do not exist on the newer Pythons in the matrix, which would force
  per-dependency `python_version` markers scattered across five conflicting
  groups — fragile and hard to review.
- It is fundamentally at odds with the lockdown's "one cryptographically
  verified dependency set per repository" intent.

We therefore **lock the test stack to the latest supported LTS runtime
(DBR 15.4: PySpark 3.5 / Python 3.11)** and preserve breadth through the
**Python-version matrix (3.9, 3.10, 3.11)** in CI — matching the reference
implementations (`blueprint`, `lsql`, `pytester`). If uv's support for
conflicting dependency groups matures enough to express the full DBR matrix
cleanly, this decision can be revisited.

## Code Quality

```bash
make format      # Auto-format with black
make lint        # Check formatting (black) and lint (flake8)
make type-check  # Run mypy
make quality     # format + lint + type-check
```

The tempo project abides by [`black`](https://black.readthedocs.io/en/stable/index.html)
formatting standards, and uses [`flake8`](https://flake8.pycqa.org/en/latest/)
and [`mypy`](https://mypy.readthedocs.io/en/stable/) for style, type-checking,
and common bad practices.

### Module imports

We organize import statements at the top of each module in the following order,
each section separated by a blank line:
1. Standard Python library imports
2. Third-party library imports
3. PySpark library imports
4. Tempo library imports

Within each section, imports are sorted alphabetically. While it is acceptable
to directly import classes and some commonly used functions, for readability it
is generally preferred to import a package with an alias. When importing
`pyspark.sql.functions`, we use the convention to alias it as `sfn`.

## Building

```bash
make build-dist  # Build sdist + wheel (build backend pinned by hashes)
make build-docs  # Build the Sphinx documentation
make build       # build-dist + build-docs
```

## Managing dependencies

Dependencies are locked in `python/uv.lock`, and the build backend (hatchling)
is pinned with hashes in `python/.build-constraints.txt`. Whenever you change
project dependencies in `pyproject.toml`, regenerate both:

```bash
make lock-dependencies
```

Commit the `pyproject.toml`, `uv.lock`, and `.build-constraints.txt` changes
together. A PR that modifies `uv.lock` without a corresponding dependency change
should be either an intentional refresh (state so in the description) or
rejected. All newly locked third-party dependencies must be at least 7 days old.

### Resolving behind a PyPI block

Some environments cannot reach the public PyPI (for example a machine behind a
supply-chain hosts blocklist). When `pypi.org` is unreachable, the `make`
targets that need an index fall back to an internal index — but only if you
configure one in a git-ignored `python/local.mk`:

```make
# python/local.mk (not committed)
DATABRICKS_PYPI_PROXY := https://<your-internal-pypi-index>/simple/
```

Databricks developers can find the internal index URL in the internal package
guidance. The fallback activates only while `pypi.org` is unreachable, and
`make lock-dependencies` scrubs the committed `uv.lock` back to `pypi.org`, so
the internal index never lands in the lockfile.

## Cleanup

```bash
make clean  # Remove the venv, caches, and build artifacts
```
