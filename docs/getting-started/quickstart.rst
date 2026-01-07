Quickstart
==========

This tutorial will get you started with Tempo in about 5 minutes. You'll learn how to create
a TSDF (Time Series DataFrame), perform common operations, and understand the core concepts.


Creating Your First TSDF
------------------------

A TSDF wraps a Spark DataFrame and adds time series-aware functionality. To create one,
you need to specify which column contains timestamps and optionally which columns identify
different time series.

.. code-block:: python

    from tempo import TSDF
    from pyspark.sql import functions as F

    # Sample data: stock prices for multiple symbols
    data = [
        ("AAPL", "2024-01-01 09:30:00", 185.50, 1000),
        ("AAPL", "2024-01-01 09:31:00", 185.75, 1500),
        ("AAPL", "2024-01-01 09:32:00", 185.60, 800),
        ("GOOG", "2024-01-01 09:30:00", 140.25, 500),
        ("GOOG", "2024-01-01 09:31:00", 140.50, 750),
        ("GOOG", "2024-01-01 09:32:00", 140.30, 600),
    ]

    df = spark.createDataFrame(data, ["symbol", "timestamp", "price", "volume"])
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    # Create a TSDF
    tsdf = TSDF(
        df,
        ts_col="timestamp",       # The timestamp column
        series_ids=["symbol"]     # Columns that identify each time series
    )

    # Display the TSDF
    tsdf.show()

The ``series_ids`` parameter is key to understanding Tempo. It tells Tempo that your DataFrame
contains multiple independent time series (one per symbol in this case). Operations like
``latest()`` or ``resample()`` will be applied to each series independently.


Time-Based Filtering
--------------------

Tempo provides intuitive methods to filter time series data:

.. code-block:: python

    # Get the latest 5 records per series
    tsdf.latest(5).show()

    # Get the earliest 3 records per series
    tsdf.earliest(3).show()

    # Filter to a specific time
    tsdf.at("2024-01-01 09:31:00").show()

    # Filter to a time range
    tsdf.between("2024-01-01 09:30:00", "2024-01-01 09:31:00").show()

    # Get records before a time
    tsdf.before("2024-01-01 09:31:00").show()

    # Get records at or after a time
    tsdf.atOrAfter("2024-01-01 09:31:00").show()


Resampling
----------

Resample your time series to a different frequency:

.. code-block:: python

    # Resample to 5-minute intervals using mean aggregation
    resampled = tsdf.resample(freq="5 minutes", func="mean")
    resampled.show()

    # Available aggregation functions: 'floor', 'ceil', 'min', 'max', 'mean'

The ``freq`` parameter accepts patterns like:

- ``"1 second"``, ``"30 seconds"``
- ``"1 minute"``, ``"5 minutes"``
- ``"1 hour"``, ``"4 hours"``
- ``"1 day"``


AS OF Joins
-----------

AS OF joins are one of Tempo's most powerful features. They join two time series by finding
the closest timestamp match:

.. code-block:: python

    # Create a trades TSDF
    trades_data = [
        ("AAPL", "2024-01-01 09:30:15", 100),
        ("AAPL", "2024-01-01 09:31:30", 150),
    ]
    trades_df = spark.createDataFrame(trades_data, ["symbol", "trade_time", "quantity"])
    trades_df = trades_df.withColumn("trade_time", F.to_timestamp("trade_time"))
    trades_tsdf = TSDF(trades_df, ts_col="trade_time", series_ids=["symbol"])

    # Create a quotes TSDF
    quotes_data = [
        ("AAPL", "2024-01-01 09:30:00", 185.50, 185.55),
        ("AAPL", "2024-01-01 09:30:30", 185.60, 185.65),
        ("AAPL", "2024-01-01 09:31:00", 185.70, 185.75),
    ]
    quotes_df = spark.createDataFrame(quotes_data, ["symbol", "quote_time", "bid", "ask"])
    quotes_df = quotes_df.withColumn("quote_time", F.to_timestamp("quote_time"))
    quotes_tsdf = TSDF(quotes_df, ts_col="quote_time", series_ids=["symbol"])

    # Join trades with the most recent quote
    joined = trades_tsdf.asofJoin(quotes_tsdf, right_prefix="quote")
    joined.show()

This finds, for each trade, the most recent quote that occurred before or at the trade time.


Interpolation
-------------

Fill missing values in your time series:

.. code-block:: python

    # After resampling, you may have missing intervals
    # Use interpolation to fill them

    # Forward fill - carry the last known value forward
    filled = resampled.interpolate(method="ffill")

    # Linear interpolation
    filled = resampled.interpolate(method="linear")

    # Backward fill
    filled = resampled.interpolate(method="bfill")


Accessing the Underlying DataFrame
----------------------------------

A TSDF wraps a Spark DataFrame. You can access it anytime:

.. code-block:: python

    # Get the underlying Spark DataFrame
    spark_df = tsdf.df

    # Use standard Spark operations
    spark_df.filter(F.col("price") > 185).show()

    # Or chain Tempo operations with Spark operations
    result = tsdf.latest(10).df.filter(F.col("volume") > 1000)


Key Properties
--------------

TSDF provides useful properties to inspect your time series:

.. code-block:: python

    # Get the timestamp column name
    print(tsdf.ts_col)

    # Get the series identifier columns
    print(tsdf.series_ids)

    # Get all column names
    print(tsdf.columns)

    # Get numeric columns available for aggregation
    print(tsdf.metric_cols)


Next Steps
----------

Now that you understand the basics, explore:

- :doc:`../user-guide/tsdf-basics` - Deep dive into TSDF creation and properties
- :doc:`../user-guide/joins` - Advanced AS OF join techniques
- :doc:`../user-guide/intervals` - Working with time intervals
- :doc:`../user-guide/window-operations` - Rolling aggregations and analytics
