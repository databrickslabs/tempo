Intervals
=========

The Intervals module provides classes for working with time intervals (periods with start
and end timestamps).


IntervalsDF
-----------

A wrapper for DataFrames containing interval data.

Constructor
^^^^^^^^^^^

.. code-block:: python

    from tempo.intervals import IntervalsDF

    idf = IntervalsDF(
        df,
        start_ts="start_time",
        end_ts="end_time",
        series_ids=["user_id"]
    )

**Parameters:**

- ``df`` - Spark DataFrame with interval data
- ``start_ts`` - Column name for interval start timestamp
- ``end_ts`` - Column name for interval end timestamp
- ``series_ids`` - Columns identifying independent interval series


Properties
^^^^^^^^^^

.. attribute:: df

    The underlying Spark DataFrame.

.. attribute:: interval_boundaries

    List of boundary column names: [start_ts, end_ts]

.. attribute:: structural_columns

    Structural columns: boundaries + series_ids

.. attribute:: observational_columns

    Non-structural columns

.. attribute:: metric_columns

    Numeric columns available for aggregation


Factory Methods
^^^^^^^^^^^^^^^

.. method:: IntervalsDF.fromNestedBoundariesDF(df, window_col, series_ids)

    Create from a DataFrame with boundaries in a nested column.

    .. code-block:: python

        idf = IntervalsDF.fromNestedBoundariesDF(
            df,
            window_col="time_window",  # Contains start/end
            series_ids=["device_id"]
        )

.. method:: IntervalsDF.fromStackedMetrics(df, start_ts, end_ts, series_ids, ...)

    Create from a DataFrame with stacked/long-format metrics.


Creating Intervals from TSDF
----------------------------

Extract intervals from point-in-time data:

.. code-block:: python

    from tempo import TSDF

    tsdf = TSDF(df, ts_col="timestamp", series_ids=["device_id"])

    # Extract intervals where is_active is True
    intervals = tsdf.extractStateIntervals(condition_col="is_active")

This creates an IntervalsDF where each interval represents a contiguous period
where the condition was True.


Example Usage
-------------

Session Analysis
^^^^^^^^^^^^^^^^

.. code-block:: python

    from tempo import TSDF
    from tempo.intervals import IntervalsDF
    from pyspark.sql import functions as F

    # User login/logout events
    events = spark.createDataFrame([
        ("user_1", "2024-01-15 09:00:00", True),
        ("user_1", "2024-01-15 12:00:00", False),
        ("user_1", "2024-01-15 14:00:00", True),
        ("user_1", "2024-01-15 18:00:00", False),
    ], ["user_id", "event_time", "logged_in"])
    events = events.withColumn("event_time", F.to_timestamp("event_time"))

    tsdf = TSDF(events, ts_col="event_time", series_ids=["user_id"])

    # Extract login sessions
    sessions = tsdf.extractStateIntervals(condition_col="logged_in")

    # Calculate session durations
    sessions.df.withColumn(
        "duration_hours",
        (F.unix_timestamp("end_time") - F.unix_timestamp("start_time")) / 3600
    ).show()


Working with Existing Interval Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Data already has start/end columns
    subscription_data = spark.createDataFrame([
        ("user_1", "2024-01-01", "2024-06-30", "premium"),
        ("user_1", "2024-07-01", "2024-12-31", "basic"),
    ], ["user_id", "start_date", "end_date", "plan"])

    # Convert to dates
    subscription_data = subscription_data \
        .withColumn("start_date", F.to_date("start_date")) \
        .withColumn("end_date", F.to_date("end_date"))

    # Create IntervalsDF
    subscriptions = IntervalsDF(
        subscription_data,
        start_ts="start_date",
        end_ts="end_date",
        series_ids=["user_id"]
    )


Interval Core Classes
---------------------

The intervals module also provides lower-level classes:

Interval
^^^^^^^^

Represents a single time interval:

.. code-block:: python

    from tempo.intervals import Interval

    # Create from pandas Series
    interval = Interval.create(
        data,
        start_field="start_time",
        end_field="end_time",
        series_fields=["device_id"],
        metric_fields=["value"]
    )


Full API Reference
------------------

.. automodule:: tempo.intervals
   :members:
   :undoc-members:
   :show-inheritance:
