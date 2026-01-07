
Tempo - Time Series Utilities for Data Teams Using Databricks
==============================================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   getting-started/index
   getting-started/installation
   getting-started/quickstart
   getting-started/migration-v02

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user-guide/index
   user-guide/tsdf-basics
   user-guide/resampling
   user-guide/joins
   user-guide/interpolation
   user-guide/intervals
   user-guide/window-operations

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api-reference/index
   api-reference/tsdf
   api-reference/tsschema
   api-reference/intervals
   api-reference/joins
   api-reference/utilities

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: About

   about/contributing
   about/roadmap
   about/tempo-team


.. image:: https://github.com/databrickslabs/tempo/workflows/build/badge.svg
   :target: https://github.com/databrickslabs/tempo/actions?query=workflow%3Abuild
   :alt: GitHub Actions - CI Build
.. image:: https://codecov.io/gh/databrickslabs/tempo/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/databrickslabs/tempo
   :alt: Codecov
.. image:: https://pepy.tech/badge/dbl-tempo/month
   :target: https://pepy.tech/project/dbl-tempo
   :alt: PyPI Monthly Downloads
.. image:: https://badge.fury.io/py/dbl-tempo.svg
   :target: https://badge.fury.io/py/dbl-tempo
   :alt: PyPI Version
.. image:: https://github.com/databrickslabs/tempo/actions/workflows/docs.yml/badge.svg
   :target: https://databrickslabs.github.io/tempo/
   :alt: Tempo sphinx docs


Tempo makes time series analysis on Apache Spark simple and performant. Built for data teams using Databricks,
it provides intuitive APIs for common time series operations that would otherwise require complex window functions
and joins.

Key Features
------------

- **AS OF Joins** - Join datasets by nearest timestamp with automatic strategy selection
- **Resampling** - Aggregate time series to different frequencies (seconds, minutes, hours, days)
- **Interpolation** - Fill missing values with forward fill, backward fill, linear interpolation, and more
- **Intervals API** - Work with time intervals, detect state changes, and handle overlapping periods
- **Window Operations** - Rolling aggregations and analytics with flexible window specifications
- **Time Filtering** - Slice time series by exact time, ranges, or relative positions (earliest, latest)

Quick Example
-------------

.. code-block:: python

    from tempo import TSDF
    from pyspark.sql import functions as F

    # Create a TSDF from a Spark DataFrame
    tsdf = TSDF(
        df,
        ts_col="event_ts",
        series_ids=["symbol"]  # Partition columns for multiple time series
    )

    # Time-based filtering
    recent = tsdf.latest(100)                    # Last 100 records per series
    window = tsdf.between("2024-01-01", "2024-12-31")

    # Resample to 1-minute intervals
    resampled = tsdf.resample(freq="1 minute", func="mean")

    # AS OF join - find latest quote for each trade
    trades_with_quotes = trades_tsdf.asofJoin(quotes_tsdf, right_prefix="quote")

    # Rolling statistics
    window_spec = tsdf.rowsBetweenWindow(-10, 0)
    with_rolling = tsdf.withColumn(
        "rolling_avg",
        F.avg("price").over(window_spec)
    )


Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install dbl-tempo

Or install the latest development version:

.. code-block:: bash

    pip install git+https://github.com/databrickslabs/tempo.git

**Requirements:** Python 3.9+ and Apache Spark 3.2+

.. note::

   The Scala version of Tempo is deprecated and no longer maintained. All new development is in Python.


Documentation
-------------

.. grid:: 2

    .. grid-item-card:: Getting Started
        :link: getting-started/index
        :link-type: doc

        New to Tempo? Start here for installation instructions and a quick tutorial.

    .. grid-item-card:: User Guide
        :link: user-guide/index
        :link-type: doc

        Learn how to use Tempo's features with detailed examples and explanations.

    .. grid-item-card:: API Reference
        :link: api-reference/index
        :link-type: doc

        Complete API documentation for all classes and methods.

    .. grid-item-card:: Migration Guide
        :link: getting-started/migration-v02
        :link-type: doc

        Upgrading from v0.1? See what's changed and how to migrate your code.


Who Uses Tempo?
---------------

Tempo is `one of the most popular packages <https://pepy.tech/project/dbl-tempo>`_ for Spark-based time series
analysis, with thousands of monthly downloads. It's actively maintained by Databricks Labs engineers and field
experts.


License
-------

Tempo is made available under the Databricks License. For more details, see the
`LICENSE <https://github.com/databrickslabs/tempo/blob/master/LICENSE>`_ file.


Contributing
------------

We welcome contributions! See our :doc:`about/contributing` guide for details on how to get involved.
