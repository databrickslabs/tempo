
tempo - Time Series Utilities for Data Teams Using Databricks
=============================================================

.. toctree::
   :hidden:
   :maxdepth: 3

   Databricks Labs <https://databricks.com/learn/labs>
   user-guide
   reference/index
   contributing
   tempo-team
   future-roadmap



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

The purpose of this project is to make time series manipulation with Spark simpler.
Operations covered under this package include AS OF joins, rolling statistics with user-specified window lengths,
featurization of time series using lagged values, and Delta Lake optimization on time and partition fields.

:ref:`Much of the Spark ecosystem already uses <who-uses>` tempo for time-series analysis and you should too.

Tempo is very easy to use:

.. code-block:: python

    from pyspark.sql.functions import *
    phone_accel_df = spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Phones_accelerometer").withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")).withColumn("x", col("x").cast("double")).withColumn("y", col("y").cast("double")).withColumn("z", col("z").cast("double")).withColumn("event_ts_dbl", col("event_ts").cast("double"))
    from tempo import *
    phone_accel_tsdf = TSDF(phone_accel_df, ts_col="event_ts", partition_cols = ["User"])
    display(phone_accel_tsdf)

Installing
----------

Tempo can be installed with `pip <https://pip.pypa.io>`_

.. code-block:: bash

  $ pip install dbl-tempo

Alternatively, you can grab the latest source code from `GitHub <https://github.com/databrickslabs/tempo>`_:

.. code-block:: bash

  $ pip install -e git+https://github.com/databrickslabs/tempo.git#"egg=dbl-tempo&#subdirectory=python"

Usage
-----

The :doc:`user-guide` is the place to go to learn how to use the library and accomplish common tasks.

The :doc:`reference/index` documentation provides API-level documentation.

.. _who-uses:

Who uses tempo?
-----------------

`tempo is one of the most popular packages on for spark based timeseries analysis. <https://pepy.tech/project/dbl-tempo>`_
and is actively maintained by engineers & field experts with constant addition of new features ranging from time series
pre-processing to time-series analytics & machine learning!


License
-------

tempo is made available under ``databricks`` License. For more details, see
`LICENSE <https://github.com/databrickslabs/tempo/blob/master/LICENSE>`_.

Contributing
------------

We happily welcome contributions, please see :doc:`contributing` for details.
