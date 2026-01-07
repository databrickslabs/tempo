Getting Started
===============

Welcome to Tempo! This section will help you get up and running with time series analysis on Apache Spark.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   migration-v02


What is Tempo?
--------------

Tempo is a library that simplifies time series manipulation on Apache Spark. It wraps Spark DataFrames
with time series-aware operations, making it easy to:

- **Resample** data to different time frequencies
- **Join** datasets by nearest timestamp (AS OF joins)
- **Interpolate** missing values
- **Filter** by time ranges and relative positions
- **Compute** rolling statistics and window aggregations
- **Extract** state intervals from time series data

Tempo is designed for large-scale time series analysis, leveraging Spark's distributed computing
capabilities to process billions of records efficiently.


Next Steps
----------

1. :doc:`installation` - Install Tempo and its dependencies
2. :doc:`quickstart` - Learn the basics with a hands-on tutorial
3. :doc:`migration-v02` - If you're upgrading from v0.1, see what's changed
