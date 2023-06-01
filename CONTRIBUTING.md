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
pyenv install 3.7 3.8 3.9
```

You will probably want to set one of these versions as your global Python version. This will be the version of Python that is used when you run `python` commands in your terminal.
For example, to set Python 3.9 as your global Python version, run the following command:
```bash
pyenv global 3.9
```

Within the `tempo/python` folder, run the below command to create a `.python-version` file that will tell `pyenv` which Python version to use when running commands in this directory:
```bash
pyenv local 3.7 3.8 3.9
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
The `—devenv` flag tells `tox` to create a development environment, and `.venv` is the folder where the virtual environment will be created.
Pre-defined environments can be found within the `tox.ini` file for different Python versions and their corresponding PySpark version. They include:
- py37-pyspark300
- py38-pyspark312
- py38-pyspark321
- py39-pyspark330
- py39-pyspark332

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
	●	format
	●	lint
	●	type-check
	●	coverage-report
