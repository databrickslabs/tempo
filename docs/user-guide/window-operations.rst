Window Operations
=================

Tempo provides a powerful window builder API for computing rolling aggregations and analytics
over time series data. This replaces the older ``withRangeStats()`` method with more flexible
and composable operations.


Window Builders
---------------

TSDF includes methods to create window specifications:

.. code-block:: python

    # Row-based window: last N rows
    window = tsdf.rowsBetweenWindow(start=-10, end=0)

    # Range-based window: time range in seconds
    window = tsdf.rangeBetweenWindow(start=-3600, end=0)  # Last hour

    # All rows before current (inclusive)
    window = tsdf.allBeforeWindow(inclusive=True)

    # All rows after current (inclusive)
    window = tsdf.allAfterWindow(inclusive=True)

    # Base window (sorted by series_ids, then timestamp)
    window = tsdf.baseWindow()


Row-Based Windows
-----------------

Row-based windows define the window by counting rows:

.. code-block:: python

    from pyspark.sql import functions as F

    # Window of last 10 rows (including current)
    window = tsdf.rowsBetweenWindow(start=-10, end=0)

    # Calculate rolling average
    result = tsdf.withColumn(
        "rolling_avg_10",
        F.avg("price").over(window)
    )

**Parameters:**

- ``start``: Rows before current (negative) or after (positive)
- ``end``: Rows before current (negative) or after (positive)
- ``0`` means current row

.. code-block:: python

    # Last 5 rows only (not including current)
    window = tsdf.rowsBetweenWindow(start=-5, end=-1)

    # Current row and next 3 rows
    window = tsdf.rowsBetweenWindow(start=0, end=3)


Range-Based Windows
-------------------

Range-based windows define the window by time duration:

.. code-block:: python

    # Last 60 seconds (including current)
    window = tsdf.rangeBetweenWindow(start=-60, end=0)

    # Last hour
    window = tsdf.rangeBetweenWindow(start=-3600, end=0)

The values are in seconds (or the native unit of your timestamp if numeric).

.. code-block:: python

    # Calculate 5-minute rolling statistics
    window = tsdf.rangeBetweenWindow(start=-300, end=0)  # 5 minutes = 300 seconds

    result = tsdf.withColumn("avg_5min", F.avg("price").over(window)) \
                 .withColumn("min_5min", F.min("price").over(window)) \
                 .withColumn("max_5min", F.max("price").over(window))


Unbounded Windows
-----------------

For cumulative calculations:

.. code-block:: python

    # All rows up to and including current
    window = tsdf.allBeforeWindow(inclusive=True)

    # Cumulative sum
    result = tsdf.withColumn(
        "cumulative_volume",
        F.sum("volume").over(window)
    )

    # All rows after current
    window = tsdf.allAfterWindow(inclusive=True)


Rolling Aggregations
--------------------

Use ``rollingAgg()`` for a cleaner syntax:

.. code-block:: python

    # Define window
    window = tsdf.rowsBetweenWindow(-10, 0)

    # Apply rolling aggregation
    result = tsdf.rollingAgg(
        window,
        F.avg("price").alias("rolling_avg"),
        F.stddev("price").alias("rolling_std"),
        F.count("*").alias("rolling_count")
    )


Multiple Aggregations
^^^^^^^^^^^^^^^^^^^^^

Compute multiple statistics in one pass:

.. code-block:: python

    window = tsdf.rangeBetweenWindow(-3600, 0)  # 1 hour

    result = tsdf.rollingAgg(
        window,
        F.min("price").alias("min_1h"),
        F.max("price").alias("max_1h"),
        F.avg("price").alias("avg_1h"),
        F.stddev("price").alias("std_1h"),
        F.sum("volume").alias("volume_1h")
    )


Per-Series Behavior
-------------------

Window operations are always performed within each series independently:

.. code-block:: python

    # Windows are scoped to each series
    tsdf = TSDF(df, ts_col="timestamp", series_ids=["symbol"])

    window = tsdf.rowsBetweenWindow(-10, 0)
    result = tsdf.withColumn("rolling_avg", F.avg("price").over(window))

    # AAPL's rolling average only considers AAPL rows
    # GOOG's rolling average only considers GOOG rows


Available Aggregation Functions
-------------------------------

Any Spark SQL window function works with Tempo windows:

**Basic Statistics:**

- ``F.avg()`` - Mean
- ``F.sum()`` - Sum
- ``F.min()`` / ``F.max()`` - Extremes
- ``F.count()`` - Count
- ``F.stddev()`` / ``F.variance()`` - Dispersion

**Positional:**

- ``F.first()`` / ``F.last()`` - First/last value
- ``F.lag()`` / ``F.lead()`` - Previous/next values
- ``F.nth_value()`` - Nth value in window

**Ranking:**

- ``F.row_number()`` - Sequential row number
- ``F.rank()`` / ``F.dense_rank()`` - Rank with ties
- ``F.percent_rank()`` / ``F.ntile()`` - Percentile ranking


Example: Technical Indicators
-----------------------------

Calculate common technical analysis indicators:

.. code-block:: python

    from pyspark.sql import functions as F

    # Stock price data
    tsdf = TSDF(prices_df, ts_col="timestamp", series_ids=["symbol"])

    # Simple Moving Averages
    window_20 = tsdf.rowsBetweenWindow(-19, 0)  # 20 periods
    window_50 = tsdf.rowsBetweenWindow(-49, 0)  # 50 periods

    result = tsdf.withColumn("sma_20", F.avg("close").over(window_20)) \
                 .withColumn("sma_50", F.avg("close").over(window_50))

    # Bollinger Bands (20-period)
    result = result.withColumn("std_20", F.stddev("close").over(window_20)) \
                   .withColumn("upper_band", F.col("sma_20") + 2 * F.col("std_20")) \
                   .withColumn("lower_band", F.col("sma_20") - 2 * F.col("std_20"))

    # Rolling high/low
    window_14 = tsdf.rowsBetweenWindow(-13, 0)  # 14 periods
    result = result.withColumn("high_14", F.max("high").over(window_14)) \
                   .withColumn("low_14", F.min("low").over(window_14))


Example: Event Analytics
------------------------

Analyze event patterns:

.. code-block:: python

    # User activity events
    tsdf = TSDF(events_df, ts_col="event_time", series_ids=["user_id"])

    # Events in last hour
    window_1h = tsdf.rangeBetweenWindow(-3600, 0)

    result = tsdf.withColumn(
        "events_last_hour",
        F.count("*").over(window_1h)
    )

    # Time since last event
    window_prev = tsdf.rowsBetweenWindow(-1, -1)  # Previous row only

    result = result.withColumn(
        "prev_event_time",
        F.first("event_time").over(window_prev)
    ).withColumn(
        "seconds_since_last",
        F.unix_timestamp("event_time") - F.unix_timestamp("prev_event_time")
    )


Reverse Windows
---------------

Process windows in reverse order (most recent first):

.. code-block:: python

    # Reverse window (most recent rows first)
    window = tsdf.rowsBetweenWindow(-10, 0, reverse=True)

    # Base window in reverse
    window = tsdf.baseWindow(reverse=True)


Performance Considerations
--------------------------

1. **Row-based vs range-based**: Row-based windows are generally faster. Use range-based when you specifically need time-based boundaries.

2. **Window size**: Larger windows require more memory. Consider if you really need unbounded windows.

3. **Partitioning**: Windows operate within partitions. Well-partitioned data improves performance:

   .. code-block:: python

       tsdf = tsdf.repartitionBySeries(200)

4. **Caching**: If computing multiple window aggregations, cache the intermediate TSDF:

   .. code-block:: python

       from pyspark import StorageLevel

       tsdf.df.persist(StorageLevel.MEMORY_AND_DISK)
       # Compute multiple window operations
       # ...
       tsdf.df.unpersist()
