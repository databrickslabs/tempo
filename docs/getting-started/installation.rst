Installation
============

This guide covers how to install Tempo and its dependencies.


Requirements
------------

- **Python**: 3.9 or higher
- **Apache Spark**: 3.2 or higher
- **PySpark**: Must match your Spark version


Installing from PyPI
--------------------

The recommended way to install Tempo is from PyPI:

.. code-block:: bash

    pip install dbl-tempo

This will install Tempo and its Python dependencies. Note that PySpark is not included as a dependency
since you typically want to match your cluster's Spark version.


Installing on Databricks
------------------------

On Databricks, you can install Tempo in several ways:

**Cluster Library (Recommended)**

1. Navigate to your cluster's Libraries tab
2. Click "Install New"
3. Select "PyPI" and enter ``dbl-tempo``
4. Click "Install"

**Notebook Magic Command**

.. code-block:: python

    %pip install dbl-tempo

**Init Script**

For clusters that need Tempo available on startup, add to your init script:

.. code-block:: bash

    /databricks/python/bin/pip install dbl-tempo


Installing from Source
----------------------

To install the latest development version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/databrickslabs/tempo.git

.. warning::

    Development versions may contain breaking changes and are not recommended for production use.


Verifying Installation
----------------------

After installation, verify that Tempo is working:

.. code-block:: python

    import tempo
    print(tempo.__version__)

    from tempo import TSDF
    print("Tempo installed successfully!")


Dependencies
------------

Tempo has the following dependencies (automatically installed):

- **pandas** - For data manipulation and display
- **numpy** - For numerical operations

Optional dependencies for development:

- **pytest** - For running tests
- **sphinx** - For building documentation
- **hatch** - For project management


Troubleshooting
---------------

**ImportError: No module named 'tempo'**

Ensure Tempo is installed in the same Python environment as your Spark driver:

.. code-block:: bash

    # Check which Python pip is using
    which pip

    # Install in the correct environment
    /path/to/your/python -m pip install dbl-tempo

**Version Mismatch Errors**

If you see errors about missing methods or parameters, ensure you have the latest version:

.. code-block:: bash

    pip install --upgrade dbl-tempo

**Spark Session Not Found**

Tempo requires an active Spark session. On Databricks, this is automatic. Locally:

.. code-block:: python

    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("Tempo Example") \
        .getOrCreate()
