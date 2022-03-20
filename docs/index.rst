
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
.. image:: https://img.shields.io/badge/docs-ghpages-lemon
   :target: https://tempo.github.io
   :alt: Deployed in github pages with sphinx-action

The purpose of this project is to make time series manipulation with Spark simpler.
Operations covered under this package include AS OF joins, rolling statistics with user-specified window lengths,
featurization of time series using lagged values, and Delta Lake optimization on time and partition fields.

Time Series on Spark & Photon with tempo is highly performant for historical analysis. We are simplifying all of the
common functions to make development more easy.

Tempo is very easy to use:

.. code-block:: python

    from pyspark.sql.functions import *
    phone_accel_df = spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Phones_accelerometer").withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")).withColumn("x", col("x").cast("double")).withColumn("y", col("y").cast("double")).withColumn("z", col("z").cast("double")).withColumn("event_ts_dbl", col("event_ts").cast("double"))
    from tempo import *
    phone_accel_tsdf = TSDF(phone_accel_df, ts_col="event_ts", partition_cols = ["User"])
    display(phone_accel_tsdf)

.. _direct-git-install:

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

Tempo is made available under ``databricks`` License. For more details, see
`LICENSE <https://github.com/databrickslabs/tempo/blob/master/LICENSE>`_.

Docs on Github Pages
--------------------

Tempo's `sphinx documentation is hosted on github pages <https://www.sphinx-doc.org/en/master/tutorial/deploying.html#id5>`_
and has been integrated with the project's github actions.

The documentation is updated and kept in line with the latest version that has been
`published on PyPI <https://pypi.org/project/dbl-tempo/>`_ and as such there is no guarantee that features and
functionality that you get from :ref:`installing tempo directly from the github repo <direct-git-install>` will be
documented here.

For support on those functionalities please feel free to reach out to the :doc:`tempo-team`.


Contributing
------------

We happily welcome contributions, please see :doc:`contributing` for details.
