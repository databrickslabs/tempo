TSDF
====

The TSDF (Time Series DataFrame) class is the core class in Tempo. It wraps a Spark DataFrame
and provides time series-aware operations.


Overview
--------

TSDF inherits from :class:`~tempo.tsschema.WindowBuilder` and provides:

- Time-based filtering (``at``, ``before``, ``after``, ``between``, etc.)
- Resampling to different frequencies
- AS OF joins with automatic strategy selection
- Interpolation of missing values
- Window operations for rolling analytics
- Integration with the Intervals API


Constructor
-----------

.. code-block:: python

    TSDF(
        df: DataFrame,
        ts_col: Optional[str] = None,
        series_ids: Optional[Collection[str]] = None,
        ts_schema: Optional[TSSchema] = None,
    )

**Parameters:**

- ``df`` - A Spark DataFrame
- ``ts_col`` - Name of the timestamp column
- ``series_ids`` - List of column names identifying independent time series
- ``ts_schema`` - Alternative: provide a TSSchema object instead of ts_col

**Examples:**

.. code-block:: python

    from tempo import TSDF

    # Basic usage
    tsdf = TSDF(df, ts_col="timestamp", series_ids=["symbol"])

    # Single time series (no partitioning)
    tsdf = TSDF(df, ts_col="timestamp")

    # Multiple series identifiers
    tsdf = TSDF(df, ts_col="timestamp", series_ids=["device_id", "sensor_type"])


Factory Methods
---------------

.. method:: TSDF.fromSubsequenceCol(df, ts_col, subsequence_col, series_ids=None)

    Create a TSDF with a subsequence column for ordering events at the same timestamp.

    .. code-block:: python

        tsdf = TSDF.fromSubsequenceCol(
            df,
            ts_col="event_time",
            subsequence_col="sequence_num",
            series_ids=["symbol"]
        )

.. method:: TSDF.fromStringTimestamp(df, ts_col, ts_format, series_ids=None)

    Create a TSDF from a string timestamp column with a specified format.

    .. code-block:: python

        tsdf = TSDF.fromStringTimestamp(
            df,
            ts_col="timestamp_str",
            ts_format="yyyy-MM-dd HH:mm:ss.SSS",
            series_ids=["symbol"]
        )


Properties
----------

.. attribute:: ts_col

    Name of the timestamp column.

.. attribute:: series_ids

    List of series identifier column names.

.. attribute:: columns

    List of all column names in the underlying DataFrame.

.. attribute:: structural_cols

    Structural columns: series_ids + timestamp column.

.. attribute:: observational_cols

    Non-structural columns (data columns).

.. attribute:: metric_cols

    Numeric columns available for aggregation.

.. attribute:: df

    The underlying Spark DataFrame.


Time Filtering Methods
----------------------

.. method:: at(ts)

    Filter to records at exactly the specified timestamp.

.. method:: before(ts)

    Filter to records strictly before the timestamp.

.. method:: after(ts)

    Filter to records strictly after the timestamp.

.. method:: atOrBefore(ts)

    Filter to records at or before the timestamp.

.. method:: atOrAfter(ts)

    Filter to records at or after the timestamp.

.. method:: between(start_ts, end_ts, inclusive=True)

    Filter to records within a time range.

.. method:: earliest(n=1)

    Get the first N records per series.

.. method:: latest(n=1)

    Get the last N records per series.

.. method:: priorTo(ts, n=1)

    Get N records immediately before the timestamp per series.

.. method:: subsequentTo(ts, n=1)

    Get N records immediately after the timestamp per series.


Resampling
----------

.. method:: resample(freq, func, fill=False)

    Resample the time series to a different frequency.

    **Parameters:**

    - ``freq`` - Target frequency (e.g., "1 minute", "5 seconds")
    - ``func`` - Aggregation function: "floor", "ceil", "min", "max", "mean"
    - ``fill`` - If True, create rows for empty time buckets

    **Returns:** TSDF

.. method:: calc_bars(freq)

    Calculate OHLCV bars at the specified frequency.


Interpolation
-------------

.. method:: interpolate(method, target_cols=None)

    Interpolate missing values.

    **Parameters:**

    - ``method`` - Interpolation method: "ffill", "bfill", "linear", "zero", "null"
    - ``target_cols`` - Columns to interpolate (default: all numeric)

    **Returns:** TSDF


AS OF Join
----------

.. method:: asofJoin(right_tsdf, left_prefix=None, right_prefix="right", strategy=None, tolerance=None, ...)

    Perform an AS OF join with another TSDF.

    **Parameters:**

    - ``right_tsdf`` - The TSDF to join with
    - ``left_prefix`` - Prefix for left-side columns
    - ``right_prefix`` - Prefix for right-side columns
    - ``strategy`` - Join strategy: "broadcast", "union", "skew", or None (auto)
    - ``tolerance`` - Maximum time difference in seconds

    **Returns:** TSDF


Window Builder Methods
----------------------

Inherited from :class:`~tempo.tsschema.WindowBuilder`:

.. method:: baseWindow(reverse=False)

    Create a basic sort window partitioned by series_ids, ordered by timestamp.

.. method:: rowsBetweenWindow(start, end, reverse=False)

    Create a row-based window specification.

    **Parameters:**

    - ``start`` - Rows before current (negative) or after (positive)
    - ``end`` - Rows before current (negative) or after (positive)

.. method:: rangeBetweenWindow(start, end, reverse=False)

    Create a range-based (time) window specification.

    **Parameters:**

    - ``start`` - Seconds before current (negative) or after (positive)
    - ``end`` - Seconds before current (negative) or after (positive)

.. method:: allBeforeWindow(inclusive=True)

    Create a window including all rows before current.

.. method:: allAfterWindow(inclusive=True)

    Create a window including all rows after current.


DataFrame Operations
--------------------

.. method:: withColumn(colName, col)

    Add or replace a column.

.. method:: withColumnRenamed(existing, new)

    Rename a column.

.. method:: drop(*cols)

    Drop one or more columns.

.. method:: select(*cols)

    Select specific columns.

.. method:: where(condition)

    Filter rows by a condition.

.. method:: union(other)

    Union with another TSDF (positional).

.. method:: unionByName(other, allowMissingColumns=False)

    Union by column name.


Series Operations
-----------------

.. method:: groupBySeries()

    Group the DataFrame by series identifiers.

.. method:: aggBySeries(*exprs)

    Aggregate per series.

.. method:: applyToSeries(func)

    Apply a function to each series.


Intervals
---------

.. method:: extractStateIntervals(condition_col)

    Extract intervals where a condition is True.

    **Returns:** IntervalsDF


Utility Methods
---------------

.. method:: show(n=20, truncate=True)

    Display the TSDF contents.

.. method:: withNaturalOrdering(reverse=False)

    Apply natural ordering (by series_ids, then timestamp).

.. method:: repartitionBySeries(numPartitions)

    Repartition by series identifiers.

.. method:: repartitionByTime(numPartitions)

    Repartition by timestamp.


Full API Reference
------------------

.. automodule:: tempo.tsdf
   :members:
   :undoc-members:
   :show-inheritance:
