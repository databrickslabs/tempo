Contributing
============

Tempo is a Databricks Labs-maintained project and we welcome contributions from the community.


Getting Started
---------------

1. `Check for open issues <https://github.com/databrickslabs/tempo/issues>`_ or open
   a fresh issue to start a discussion around a feature idea or a bug.
2. Fork the `tempo repository on Github <https://github.com/databrickslabs/tempo>`_.
3. Clone your fork and set up your development environment.


Development Setup
-----------------

Tempo uses `Hatch <https://hatch.pypa.io/>`_ for project management.

**Prerequisites:**

- Python 3.9 or higher
- Hatch (``pip install hatch``)

**Setting up the environment:**

.. code-block:: bash

    # Clone your fork
    git clone https://github.com/YOUR_USERNAME/tempo.git
    cd tempo

    # Create and activate the development environment
    hatch env create

    # Or run commands in the hatch environment
    hatch run python -c "import tempo; print(tempo.__version__)"


Running Tests
-------------

.. code-block:: bash

    # Run all tests
    hatch run test

    # Run specific test file
    hatch run pytest tests/tsdf_tests.py

    # Run with coverage
    hatch run test-cov


Code Style
----------

Tempo uses the following tools for code quality:

**Black** for code formatting:

.. code-block:: bash

    hatch run black python/

**Flake8** for linting:

.. code-block:: bash

    hatch run flake8 python/

**Type hints** are encouraged but not strictly enforced.


Docstrings
----------

Use RST-style docstrings for all public methods:

.. code-block:: python

    def resample(self, freq: str, func: str) -> "TSDF":
        """
        Resample the time series to a different frequency.

        :param freq: Target frequency (e.g., "1 minute", "5 seconds")
        :param func: Aggregation function: "floor", "ceil", "min", "max", "mean"
        :return: A new TSDF with resampled data
        :raises ValueError: If freq is not a valid frequency string
        """


Building Documentation
----------------------

.. code-block:: bash

    # Build HTML documentation
    hatch run docs:build

    # Or manually with Sphinx
    cd docs
    make html

The documentation will be generated in ``docs/_build/html/``.


Pull Request Process
--------------------

1. **Create a branch** for your changes:

   .. code-block:: bash

       git checkout -b feature/my-new-feature

2. **Write tests** that demonstrate the bug fix or feature works correctly.

3. **Format and lint** your code:

   .. code-block:: bash

       hatch run black python/
       hatch run flake8 python/

4. **Run the test suite** to ensure nothing is broken:

   .. code-block:: bash

       hatch run test

5. **Commit your changes** with a clear commit message.

6. **Push to your fork** and create a Pull Request.

7. **Add a changelog entry** to the pull request body describing your changes.

8. **Wait for review** from the maintainers.


What We're Looking For
----------------------

- **Bug fixes** with tests that reproduce the issue
- **New features** that have been discussed in an issue first
- **Documentation improvements** (typo fixes, clarifications, new examples)
- **Performance improvements** with benchmarks
- **Test coverage** improvements


Code of Conduct
---------------

Be respectful and constructive in all interactions. We're all here to make
Tempo better for the community.


Questions?
----------

- Open an issue for questions about contributing
- Contact the team at labs@databricks.com
- See the :doc:`tempo-team` for maintainer information
