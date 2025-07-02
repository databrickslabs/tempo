# Setup instructions

We use `tox` and `pyenv` to manage developing and testing `tempo` across multiple Python versions and PySpark versions.
`tox` is a testing tool that helps you automate and standardize testing in Python across multiple environments.
`pyenv` allows you to manage multiple versions of Python on your computer and easily switch between them.

## Pyenv setup

Since `tox` supports creating virtual environments using multiple Python versions, it is recommended to use `pyenv` to manage Python versions on your computer.
`pyenv` does not install via `pip` on all platforms, so see the [pyenv documentation](https://github.com/pyenv/pyenv#installation) for installation instructions. 
Be sure to carefully follow the instructions to configure your shell environment to use `pyenv`, otherwise subsequent commands will not work as intended.

Use `pyenv` to install the following Python versions for testing.
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

This allows `tox` to create virtual environments using any of the Python versions listed in the `.python-version` file.

## Tox setup

Install tox:
```bash
pip install -U tox
```

A brief description of each managed `tox` environment can be found by running `tox list` or in the `tox.ini` file.

## Create a development environment
Each development environment is roughly associated with a DBR LTS version. A list of base environments can be found in your `tox.ini` file, but the number specified below is only the version number.
Run the following command in your terminal to create a virtual environment (you can replace "your_dbr_number" with something like 154, 143, etc.):
```bash
make venv DBR={your_dbr_number}
```

## Environments we test
Use the following to run tests for a given environment. 
```bash
make test DBR={your_dbr_number}
```

If you would like to run tests for all environments, use
```bash
make test-all
```

### Run additional checks locally
`tox` has special environments for additional checks that must be performed as part of the PR process. These include formatting, linting, type checking, etc.
These environments are also defined in the `tox.ini`file and skip installing dependencies listed in the `requirements.txt` file and building the distribution when those are not required . They can be specified using the `-e` flag:
* lint
* type-check
* build-dist
* build-docs
* coverage-report

To run an individual check, use
```bash
make lint
make type-check
make {name_of_environment}
```

To run all checks at once, run
```bash
make all-local
```
# Code style & Standards

The tempo project abides by [`black`](https://black.readthedocs.io/en/stable/index.html) formatting standards, 
as well as using [`flake8`](https://flake8.pycqa.org/en/latest/) and [`mypy`](https://mypy.readthedocs.io/en/stable/) 
to check for effective code style, type-checking and common bad practices.
To test your code against these standards, run the following command:
```bash
tox -e lint, type-check
```
To have `black` automatically format your code, run the following command:
```bash
tox -e format
```

In addition, we apply some project-specific standards:

## Module imports

We organize import statements at the top of each module in the following order, each section being separated by a blank line:
1. Standard Python library imports
2. Third-party library imports
3. PySpark library imports
4. Tempo library imports

Within each section, imports are sorted alphabetically. While it is acceptable to directly import classes and some functions that are
going to be commonly used, for the sake of readability, it is generally preferred to import a package with an alias and then use the alias
to reference the package's classes and functions. 

When importing `pyspark.sql.functions`, we use the convention to alias this package as `sfn`, which is both distinctive and short.
