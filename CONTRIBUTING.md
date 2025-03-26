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
pyenv install 3.8 3.9 3.10
```

You will probably want to set one of these versions as your global Python version. This will be the version of Python that is used when you run `python` commands in your terminal.
For example, to set Python 3.9 as your global Python version, run the following command:
```bash
pyenv global 3.10
```

Within the `tempo/python` folder, run the below command to create a `.python-version` file that will tell `pyenv` which Python version to use when running commands in this directory:
```bash
pyenv local 3.8 3.9 3.10
```

This allows `tox` to create virtual environments using any of the Python versions listed in the `.python-version` file.

## Tox setup

Install tox:
```bash
pip install -U tox
```

A brief description of each managed `tox` environment can be found by running `tox list` or in the `tox.ini` file.

## Create a development environment
Run the following command in your terminal to create a virtual environment in the `.venv` folder:
```bash
tox --devenv .venv -e {environment-name}
```
The `—devenv` flag tells `tox` to create a development environment, and `.venv` is the folder where the virtual environment will be created. The `environment-name` is a reference to the environments that are listed in the `tox.ini` file. In general, these reflect various LTS versions of DBR. 

## Environments we test
The environments we test against are defined within the `tox.ini` file, and the requirements for those environments are stored in `python/tests/requirements`. The makeup of these environments is inspired by the [Databricks Runtime](https://docs.databricks.com/en/release-notes/runtime/index.html#) (hence the naming convention), but it's important to note that developing Databricks is **not** a requirement. We're simply  mimicking some of the different runtime versions because (a) we recognize that much of the user base uses `tempo` on Databricks and (b) it saves development time spent trying to build out test environments with different versions of Python and PySpark from scratch.

## Run tests locally for one or more environments
You can run tests locally for one or more environments defined enviornments without setting up a development environment first.

### To run tests for a single environment, use the `-e` flag followed by the environment name:
```bash
tox -e {environment-name}
```

### To run tests for multiple environments, specify the environment names separated by commas:
```bash
tox -e {environment-name1, environment-name2, etc.}
```
This will run tests for all listed environments.

### Run additional checks locally
`tox` has special environments for additional checks that must be performed as part of the PR process. These include formatting, linting, type checking, etc.
These environments are also defined in the `tox.ini`file and skip installing dependencies listed in the `requirements.txt` file and building the distribution when those are not required . They can be specified using the `-e` flag:
* lint
* type-check
* build-dist
* build-docs
* coverage-report

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
