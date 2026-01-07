TSDF Basics
===========

The TSDF (Time Series DataFrame) is the core class in Tempo. It wraps a Spark DataFrame
and provides time series-aware operations.


Creating a TSDF
---------------

Basic Constructor
^^^^^^^^^^^^^^^^^

The simplest way to create a TSDF is with the constructor:

.. code-block:: python

    from tempo import TSDF

    tsdf = TSDF(
        df,                        # Spark DataFrame
        ts_col="timestamp",        # Name of the timestamp column
        series_ids=["symbol"]      # Columns identifying each time series
    )

**Parameters:**

- ``df``: A Spark DataFrame containing your time series data
- ``ts_col``: The name of the column containing timestamps
- ``series_ids``: A list of column names that identify independent time series (optional)

If ``series_ids`` is not provided, the entire DataFrame is treated as a single time series.


Multiple Series Identifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can have multiple columns as series identifiers:

.. code-block:: python

    # IoT data with device and sensor identifiers
    tsdf = TSDF(
        df,
        ts_col="event_time",
        series_ids=["device_id", "sensor_type"]
    )

Each unique combination of series identifier values forms an independent time series.


Factory Methods
^^^^^^^^^^^^^^^

Tempo provides factory methods for special cases:

**String Timestamps with Custom Format**

.. code-block:: python

    # When timestamp is stored as a string
    tsdf = TSDF.fromStringTimestamp(
        df,
        ts_col="timestamp_str",
        ts_format="yyyy-MM-dd HH:mm:ss.SSS",
        series_ids=["symbol"]
    )

**Subsequence Column**

When you have a secondary ordering column (e.g., sequence numbers for events at the same timestamp):

.. code-block:: python

    tsdf = TSDF.fromSubsequenceCol(
        df,
        ts_col="event_time",
        subsequence_col="sequence_num",
        series_ids=["symbol"]
    )


TSDF Properties
---------------

TSDF provides several useful properties:

.. code-block:: python

    # The timestamp column name
    tsdf.ts_col  # "timestamp"

    # Series identifier columns
    tsdf.series_ids  # ["symbol"]

    # All column names
    tsdf.columns  # ["symbol", "timestamp", "price", "volume"]

    # Structural columns (series_ids + timestamp)
    tsdf.structural_cols  # ["symbol", "timestamp"]

    # Observational columns (everything else)
    tsdf.observational_cols  # ["price", "volume"]

    # Numeric columns suitable for aggregation
    tsdf.metric_cols  # ["price", "volume"]


Accessing the DataFrame
-----------------------

Access the underlying Spark DataFrame:

.. code-block:: python

    # Get the Spark DataFrame
    spark_df = tsdf.df

    # Use in Spark operations
    spark_df.printSchema()
    spark_df.count()


Time-Based Filtering
--------------------

TSDF provides intuitive methods for filtering by time.

Exact Time
^^^^^^^^^^

.. code-block:: python

    # Records at exactly this timestamp
    tsdf.at("2024-01-15 09:30:00")

Before/After
^^^^^^^^^^^^

.. code-block:: python

    # Records strictly before this time
    tsdf.before("2024-01-15 09:30:00")

    # Records strictly after this time
    tsdf.after("2024-01-15 09:30:00")

    # Records at or before this time
    tsdf.atOrBefore("2024-01-15 09:30:00")

    # Records at or after this time
    tsdf.atOrAfter("2024-01-15 09:30:00")

Time Ranges
^^^^^^^^^^^

.. code-block:: python

    # Records between two times (inclusive by default)
    tsdf.between("2024-01-15 09:00:00", "2024-01-15 10:00:00")

    # Exclusive of endpoints
    tsdf.between("2024-01-15 09:00:00", "2024-01-15 10:00:00", inclusive=False)

Earliest/Latest
^^^^^^^^^^^^^^^

Get the first or last N records per series:

.. code-block:: python

    # Last 10 records per series
    tsdf.latest(10)

    # First 5 records per series
    tsdf.earliest(5)

Relative to Timestamp
^^^^^^^^^^^^^^^^^^^^^

Get records relative to a specific point in time:

.. code-block:: python

    # N records immediately before this timestamp
    tsdf.priorTo("2024-01-15 09:30:00", n=5)

    # N records immediately after this timestamp
    tsdf.subsequentTo("2024-01-15 09:30:00", n=5)


DataFrame-Like Operations
-------------------------

TSDF supports common DataFrame transformations:

Adding/Modifying Columns
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pyspark.sql import functions as F

    # Add a new column
    tsdf = tsdf.withColumn("price_cents", F.col("price") * 100)

    # Rename a column
    tsdf = tsdf.withColumnRenamed("price", "unit_price")

Dropping Columns
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Drop one or more columns
    tsdf = tsdf.drop("unnecessary_col", "another_col")

Selecting Columns
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Select specific columns (returns a new TSDF)
    tsdf = tsdf.select("symbol", "timestamp", "price")

Filtering Rows
^^^^^^^^^^^^^^

.. code-block:: python

    # Filter with a condition
    tsdf = tsdf.where(F.col("price") > 100)


Display and Output
------------------

TSDF provides several ways to view and output data:

.. code-block:: python

    # Show data (like Spark's show())
    tsdf.show()
    tsdf.show(20, truncate=False)

    # Get underlying DataFrame for further processing
    result_df = tsdf.df

    # Write to storage via the DataFrame
    tsdf.df.write.parquet("/path/to/output")


Repartitioning
--------------

Optimize data layout for time series operations:

.. code-block:: python

    # Repartition by series identifiers
    tsdf = tsdf.repartitionBySeries(100)

    # Repartition by time
    tsdf = tsdf.repartitionByTime(100)


Ordering
--------

Apply natural ordering to the TSDF:

.. code-block:: python

    # Order by series_ids, then timestamp
    ordered_tsdf = tsdf.withNaturalOrdering()

    # Reverse order (most recent first)
    ordered_tsdf = tsdf.withNaturalOrdering(reverse=True)


Combining TSDFs
---------------

Combine multiple TSDFs:

.. code-block:: python

    # Union (positional column matching)
    combined = tsdf1.union(tsdf2)

    # Union by column name (handles different column orders)
    combined = tsdf1.unionByName(tsdf2)

    # Allow missing columns
    combined = tsdf1.unionByName(tsdf2, allowMissingColumns=True)
