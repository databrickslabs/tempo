Migration Guide: v0.1 to v0.2
=============================

Tempo v0.2 introduces significant API changes to improve consistency and add new features.
This guide helps you migrate existing code from v0.1 to v0.2.


Summary of Breaking Changes
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Change
     - v0.1 (Old)
     - v0.2 (New)
   * - Partition columns parameter
     - ``partition_cols``
     - ``series_ids``
   * - Partition columns attribute
     - ``tsdf.partitionCols``
     - ``tsdf.series_ids``
   * - Sequence column
     - ``sequence_col`` parameter
     - Removed (use ``fromSubsequenceCol()``)
   * - AS OF join optimization
     - ``sql_join_opt=True``
     - ``strategy='broadcast'``
   * - EMA calculation
     - ``tsdf.EMA()``
     - Removed (use ``rollingAgg()``)
   * - Range statistics
     - ``tsdf.withRangeStats()``
     - Removed (use ``rollingAgg()``)
   * - Grouped statistics
     - ``tsdf.withGroupedStats()``
     - Removed (use ``resample()`` + aggregations)


Constructor Changes
-------------------

The TSDF constructor has been updated:

**v0.1:**

.. code-block:: python

    tsdf = TSDF(
        df,
        ts_col="event_ts",
        partition_cols=["symbol"],
        sequence_col="seq_num"
    )

**v0.2:**

.. code-block:: python

    # Basic usage - partition_cols renamed to series_ids
    tsdf = TSDF(
        df,
        ts_col="event_ts",
        series_ids=["symbol"]
    )

    # For sequence columns, use the factory method
    tsdf = TSDF.fromSubsequenceCol(
        df,
        ts_col="event_ts",
        subsequence_col="seq_num",
        series_ids=["symbol"]
    )


Accessing Partition Columns
---------------------------

**v0.1:**

.. code-block:: python

    cols = tsdf.partitionCols

**v0.2:**

.. code-block:: python

    cols = tsdf.series_ids


AS OF Join Strategy Selection
-----------------------------

The ``sql_join_opt`` parameter has been replaced with a ``strategy`` parameter that offers
more control:

**v0.1:**

.. code-block:: python

    # Use broadcast optimization for small right tables
    result = left.asofJoin(right, sql_join_opt=True)

**v0.2:**

.. code-block:: python

    # Automatic strategy selection (recommended)
    result = left.asofJoin(right)

    # Or explicitly choose a strategy
    result = left.asofJoin(right, strategy='broadcast')  # For small right tables
    result = left.asofJoin(right, strategy='union')      # General purpose
    result = left.asofJoin(right, strategy='skew')       # For skewed data


Removed Methods
---------------

Several feature engineering methods have been removed in v0.2. Here's how to achieve the
same results with the new API:

EMA (Exponential Moving Average)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**v0.1:**

.. code-block:: python

    result = tsdf.EMA("price", window=50)

**v0.2:**

The ``EMA()`` method has been removed. For exponential moving averages, you can use
Spark's native window functions or implement a custom UDF:

.. code-block:: python

    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Simple alternative: use rolling average
    window_spec = tsdf.rowsBetweenWindow(-50, 0)
    result = tsdf.withColumn(
        "rolling_avg",
        F.avg("price").over(window_spec)
    )


withRangeStats (Rolling Statistics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**v0.1:**

.. code-block:: python

    result = tsdf.withRangeStats("price", rangeBackWindowSecs=600)

**v0.2:**

Use the new window builder API:

.. code-block:: python

    from pyspark.sql import functions as F

    # Create a range-based window (600 seconds = 10 minutes back)
    window_spec = tsdf.rangeBetweenWindow(-600, 0)

    # Add statistics
    result = tsdf.withColumn("mean_price", F.avg("price").over(window_spec)) \
                 .withColumn("min_price", F.min("price").over(window_spec)) \
                 .withColumn("max_price", F.max("price").over(window_spec)) \
                 .withColumn("stddev_price", F.stddev("price").over(window_spec))


withGroupedStats
^^^^^^^^^^^^^^^^

**v0.1:**

.. code-block:: python

    result = tsdf.withGroupedStats(metricCols=["price"], freq="1 minute")

**v0.2:**

Use ``resample()`` with aggregations:

.. code-block:: python

    from pyspark.sql import functions as F

    # Resample to 1-minute intervals
    resampled = tsdf.resample(freq="1 minute", func="mean")

    # Or use groupBySeries for custom aggregations
    result = tsdf.df.groupBy(
        *tsdf.series_ids,
        F.window("timestamp", "1 minute")
    ).agg(
        F.min("price").alias("min_price"),
        F.max("price").alias("max_price"),
        F.avg("price").alias("avg_price"),
        F.stddev("price").alias("stddev_price"),
        F.count("*").alias("count")
    )


Interpolation API Changes
-------------------------

The interpolation API has been simplified:

**v0.1:**

.. code-block:: python

    from tempo.interpol import Interpolation

    interp = Interpolation(is_resampled=True)
    result = interp.interpolate(
        tsdf,
        partition_cols=["symbol"],
        target_cols=["price"],
        freq="1 minute",
        ts_col="timestamp",
        func="mean",
        method="ffill"
    )

**v0.2:**

.. code-block:: python

    # Direct method on TSDF - much simpler!
    result = tsdf.interpolate(
        method="ffill",
        target_cols=["price"]
    )

    # Or chain with resample
    result = tsdf.resample(freq="1 minute", func="mean").interpolate(method="linear")


Import Changes
--------------

Some imports have moved:

**v0.1:**

.. code-block:: python

    from tempo.resample import floor, ceiling, checkAllowableFreq
    from tempo.utils import calculate_time_horizon

**v0.2:**

.. code-block:: python

    from tempo.resample_utils import floor, ceiling, checkAllowableFreq
    from tempo.resample import calculate_time_horizon


New Features in v0.2
--------------------

While migrating, take advantage of new features:

**Window Builder API**

.. code-block:: python

    # Create window specifications easily
    rows_window = tsdf.rowsBetweenWindow(-10, 0)    # Last 10 rows
    range_window = tsdf.rangeBetweenWindow(-3600, 0)  # Last hour

**Series-Aware Operations**

.. code-block:: python

    # Group by series identifiers
    tsdf.groupBySeries().agg(F.count("*"))

    # Apply functions per series
    tsdf.aggBySeries(F.sum("volume"))

**Intervals API**

.. code-block:: python

    # Extract state intervals from time series
    intervals = tsdf.extractStateIntervals(condition_col="is_active")

**New TSDF Methods**

.. code-block:: python

    # DataFrame-like operations
    tsdf.withColumn("new_col", F.col("price") * 2)
    tsdf.drop("unnecessary_col")
    tsdf.where(F.col("price") > 100)

    # New filtering
    tsdf.priorTo("2024-01-01", n=5)  # 5 records before timestamp
    tsdf.subsequentTo("2024-01-01", n=5)  # 5 records after

    # Repartitioning
    tsdf.repartitionBySeries(100)


Migration Checklist
-------------------

Use this checklist to ensure a complete migration:

.. code-block:: text

    [ ] Replace partition_cols with series_ids in TSDF constructor
    [ ] Replace tsdf.partitionCols with tsdf.series_ids
    [ ] Remove sequence_col parameter, use fromSubsequenceCol() if needed
    [ ] Replace sql_join_opt=True with strategy='broadcast'
    [ ] Replace EMA() with window functions
    [ ] Replace withRangeStats() with window functions
    [ ] Replace withGroupedStats() with resample()
    [ ] Update interpolation code to use new API
    [ ] Update imports from tempo.resample to tempo.resample_utils where needed
