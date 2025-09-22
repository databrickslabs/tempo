# Contributing to Tempo

## Quick Start

```bash
cd tempo/python
make init  # Complete setup for new developers
```

This will set up your complete development environment.

## Development Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11
- pip
- make

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/databrickslabs/tempo.git
   cd tempo/python
   ```

2. **Initialize development environment**
   ```bash
   make init  # Complete setup (recommended)
   ```

   Or if you prefer manual steps:
   ```bash
   make setup         # Install development tools
   make check-env     # Verify environment
   make venv          # Create virtual environment for default DBR
   ```

3. **Check status**
   ```bash
   make status        # Quick status check
   make doctor        # Comprehensive health check
   ```

## Python Version Management

Since we test across multiple Python versions, it is recommended to use `pyenv` to manage Python versions on your computer.

### Pyenv Setup

`pyenv` does not install via `pip` on all platforms, so see the [pyenv documentation](https://github.com/pyenv/pyenv#installation) for installation instructions.
Be sure to carefully follow the instructions to configure your shell environment to use `pyenv`, otherwise subsequent commands will not work as intended.

Use `pyenv` to install the following Python versions for testing:
```bash
pyenv install 3.9 3.10 3.11
```

You will probably want to set one of these versions as your global Python version. This will be the version of Python that is used when you run `python` commands in your terminal.
For example, to set Python 3.9 as your global Python version, run the following command:
```bash
pyenv global 3.9
```

Within the `tempo/python` folder, run the below command to create a `.python-version` file that will tell `pyenv` which Python version to use when running commands in this directory:
```bash
pyenv local 3.9 3.10 3.11
```

## Running Tests

### Test a specific DBR version
```bash
make test           # Test with default DBR version (154)
make test DBR=143   # Test with specific DBR version
```

### Test all DBR versions
```bash
make test-all       # Test all DBR versions
```

### DBR Versions Supported
- **DBR 154** - Python 3.11 (default)
- **DBR 143** - Python 3.10
- **DBR 133** - Python 3.10
- **DBR 122** - Python 3.9
- **DBR 113** - Python 3.9

## Code Quality

### Run linters and formatters
```bash
make lint           # Run linters
make format         # Format code with Black
make type-check     # Run type checking
```

### Run all checks at once
```bash
make all-local      # Run all local tests and checks
```

## Building

```bash
make build-dist     # Build distribution packages
make build-docs     # Build documentation
make coverage-report # Generate test coverage report
```

## Code Style & Standards

The tempo project abides by [`black`](https://black.readthedocs.io/en/stable/index.html) formatting standards,
as well as using [`flake8`](https://flake8.pycqa.org/en/latest/) and [`mypy`](https://mypy.readthedocs.io/en/stable/)
to check for effective code style, type-checking and common bad practices.

To test your code against these standards, run:
```bash
make lint
make type-check
```

To have `black` automatically format your code, run:
```bash
make format
```

In addition, we apply some project-specific standards:

### Module imports

We organize import statements at the top of each module in the following order, each section being separated by a blank line:
1. Standard Python library imports
2. Third-party library imports
3. PySpark library imports
4. Tempo library imports

Within each section, imports are sorted alphabetically. While it is acceptable to directly import classes and some functions that are
going to be commonly used, for the sake of readability, it is generally preferred to import a package with an alias and then use the alias
to reference the package's classes and functions.

When importing `pyspark.sql.functions`, we use the convention to alias this package as `sfn`, which is both distinctive and short.

## Troubleshooting

### Dependency Issues
If you see errors during setup or when running commands:
```bash
make setup          # Reinstall development tools
```

### Python Version Issues
If you need a specific Python version:
```bash
make setup-python-versions  # Install all required Python versions
```

### Clean Slate
If things are broken:
```bash
make clean          # Remove all generated files
make deep-clean     # Remove everything (venvs, caches, artifacts)
make init           # Reinstall everything
```

### Known Issues
See `python/.env.versions` for notes on current tool version constraints and tracking issues.

## Version Management

Tool versions are centrally managed in `python/.env.versions`:
- Both local development and CI use these same versions
- To update versions, modify `.env.versions` and run `make setup`

## CI/CD

GitHub Actions workflows automatically:
- Source `.env.versions` for consistent tool versions
- Run tests across all DBR versions
- Enforce the same linting and type checking as local development