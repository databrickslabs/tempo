Intervals API
=============

The Intervals API provides tools for working with time intervals - periods with start and end
timestamps. This is useful for:

- Analyzing sessions, events, or state changes
- Detecting and handling overlapping periods
- Converting point-in-time data to interval data


IntervalsDF
-----------

``IntervalsDF`` is the core class for working with interval data:

.. code-block:: python

    from tempo.intervals import IntervalsDF

    # Create from a DataFrame with start/end columns
    idf = IntervalsDF(
        df,
        start_ts="start_time",
        end_ts="end_time",
        series_ids=["user_id"]
    )

**Parameters:**

- ``df``: A Spark DataFrame containing interval data
- ``start_ts``: Column name for interval start timestamps
- ``end_ts``: Column name for interval end timestamps
- ``series_ids``: Columns identifying independent interval series


Extracting Intervals from TSDF
------------------------------

Convert point-in-time data to intervals using state changes:

.. code-block:: python

    from tempo import TSDF

    # Device state data (point-in-time)
    tsdf = TSDF(df, ts_col="timestamp", series_ids=["device_id"])

    # Extract intervals where device is active
    intervals = tsdf.extractStateIntervals(condition_col="is_active")

This creates intervals for consecutive periods where ``is_active`` is True.


Example: User Sessions
----------------------

.. code-block:: python

    from pyspark.sql import functions as F

    # User activity events
    activity_data = [
        ("user_1", "2024-01-15 09:00:00", True),   # Login
        ("user_1", "2024-01-15 09:15:00", True),   # Active
        ("user_1", "2024-01-15 09:30:00", False),  # Logout
        ("user_1", "2024-01-15 10:00:00", True),   # Login again
        ("user_1", "2024-01-15 10:30:00", False),  # Logout
    ]
    df = spark.createDataFrame(activity_data, ["user_id", "timestamp", "is_active"])
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    tsdf = TSDF(df, ts_col="timestamp", series_ids=["user_id"])

    # Extract active sessions
    sessions = tsdf.extractStateIntervals(condition_col="is_active")

Result:

.. code-block:: text

    +-------+---------------------+---------------------+
    |user_id|start_time           |end_time             |
    +-------+---------------------+---------------------+
    |user_1 |2024-01-15 09:00:00  |2024-01-15 09:30:00  |
    |user_1 |2024-01-15 10:00:00  |2024-01-15 10:30:00  |
    +-------+---------------------+---------------------+


IntervalsDF Properties
----------------------

.. code-block:: python

    # Interval boundary columns
    idf.interval_boundaries  # ["start_time", "end_time"]

    # Structural columns (boundaries + series_ids)
    idf.structural_columns  # ["start_time", "end_time", "user_id"]

    # Observational columns (everything else)
    idf.observational_columns

    # Numeric columns for aggregation
    idf.metric_columns


Creating IntervalsDF Directly
-----------------------------

If your data already has interval columns:

.. code-block:: python

    from tempo.intervals import IntervalsDF

    # Subscription data with start/end dates
    subscriptions_data = [
        ("user_1", "2024-01-01", "2024-06-30", "premium"),
        ("user_1", "2024-07-01", "2024-12-31", "basic"),
        ("user_2", "2024-03-01", "2024-09-30", "premium"),
    ]
    df = spark.createDataFrame(subscriptions_data,
        ["user_id", "start_date", "end_date", "plan"])
    df = df.withColumn("start_date", F.to_date("start_date"))
    df = df.withColumn("end_date", F.to_date("end_date"))

    idf = IntervalsDF(
        df,
        start_ts="start_date",
        end_ts="end_date",
        series_ids=["user_id"]
    )


Factory Methods
---------------

``IntervalsDF`` provides factory methods for special cases:

From Nested Boundaries
^^^^^^^^^^^^^^^^^^^^^^

When boundaries are stored in a struct/array column:

.. code-block:: python

    idf = IntervalsDF.fromNestedBoundariesDF(
        df,
        window_col="time_window",
        series_ids=["device_id"]
    )

From Stacked Metrics
^^^^^^^^^^^^^^^^^^^^

When metrics are in a long format:

.. code-block:: python

    idf = IntervalsDF.fromStackedMetrics(
        df,
        start_ts="start_time",
        end_ts="end_time",
        series_ids=["device_id"],
        metric_name_col="metric_name",
        metric_value_col="metric_value"
    )


Working with Intervals
----------------------

Once you have an ``IntervalsDF``, you can perform various operations:

Accessing Data
^^^^^^^^^^^^^^

.. code-block:: python

    # Get the underlying DataFrame
    spark_df = idf.df

    # Show the intervals
    idf.df.show()

Filtering
^^^^^^^^^

.. code-block:: python

    # Filter to intervals within a date range
    filtered = idf.df.filter(
        (F.col("start_time") >= "2024-01-01") &
        (F.col("end_time") <= "2024-06-30")
    )


Overlap Detection
-----------------

Detect overlapping intervals within each series:

.. code-block:: python

    # Check for overlaps
    # (Detailed overlap API methods depend on your tempo version)

Overlapping intervals occur when:

- One interval starts before another ends
- Intervals share any common time points


Use Cases
---------

Session Analysis
^^^^^^^^^^^^^^^^

Analyze user sessions from event data:

.. code-block:: python

    # Events to sessions
    sessions = events_tsdf.extractStateIntervals(condition_col="is_logged_in")

    # Calculate session durations
    sessions_with_duration = sessions.df.withColumn(
        "duration_minutes",
        (F.unix_timestamp("end_time") - F.unix_timestamp("start_time")) / 60
    )

    # Average session duration per user
    avg_duration = sessions_with_duration.groupBy("user_id").agg(
        F.avg("duration_minutes").alias("avg_session_minutes")
    )


Machine States
^^^^^^^^^^^^^^

Track machine operational states:

.. code-block:: python

    # Extract periods where machine is running
    running_periods = machine_tsdf.extractStateIntervals(condition_col="is_running")

    # Calculate total runtime
    total_runtime = running_periods.df.withColumn(
        "runtime_seconds",
        F.unix_timestamp("end_time") - F.unix_timestamp("start_time")
    ).groupBy("machine_id").agg(
        F.sum("runtime_seconds").alias("total_runtime_seconds")
    )


Subscription Periods
^^^^^^^^^^^^^^^^^^^^

Analyze subscription overlaps and gaps:

.. code-block:: python

    # Create IntervalsDF from subscription data
    subscriptions = IntervalsDF(
        df,
        start_ts="subscription_start",
        end_ts="subscription_end",
        series_ids=["customer_id"]
    )

    # Find customers with concurrent subscriptions
    # (would need overlap detection logic)


Integration with TSDF
---------------------

Intervals and point-in-time data can work together:

.. code-block:: python

    # Start with point-in-time data
    tsdf = TSDF(events_df, ts_col="timestamp", series_ids=["device_id"])

    # Extract intervals based on a condition
    active_intervals = tsdf.extractStateIntervals(condition_col="is_active")

    # Use intervals for analysis
    # Then potentially join back with point data if needed
