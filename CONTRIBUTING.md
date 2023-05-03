# Contributing to Tempo
We happily welcome contributions to Tempo. 
We use [GitHub Issues](https://github.com/databrickslabs/tempo/issues) to track community reported issues 
and [GitHub Pull Requests](https://github.com/databrickslabs/tempo/pulls) for reviewing and accepting code changes.
If you have found a bug or other issue with the library, please [open an new issue](https://github.com/databrickslabs/tempo/issues/new) to report it.
You can [create a fork of the code](https://github.com/databrickslabs/tempo/fork) to begin your own modifications. 
If you have written a patch for a bug or new feature, please [submit a pull request](https://github.com/databrickslabs/tempo/compare) and we will be happy to review it for inclusion in a future release.

## Tox Setup instructions

`tox` is a testing tool that helps you automate and standardize testing in Python across multiple environments.

`pyenv`that allows you to manage multiple versions of Python on your computer and easily switch between them.

Since `tox` supports creating virtual environments using multiple Python versions, it is recommended to use `pyenv` to manage Python versions on your computer.

Install both tox and pyenv packages:
```bash
pip install -U tox pyenv
pyenv install 3.7 3.8 3.9
```

Within `python` folder, run the below command to create a `.python-version` file that will tell `pyenv` which Python version to use when running commands in this directory:
```bash
pyenv local 3.7 3.8 3.9
```

This allows `tox` to create virtual environments using any of the Python versions listed in the `.python-version` file.

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
