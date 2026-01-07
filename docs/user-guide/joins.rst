AS OF Joins
===========

AS OF joins are one of Tempo's most powerful features. They join two time series by matching
each row in the left dataset with the most recent row from the right dataset that occurred
at or before the left row's timestamp.

This is essential for:

- Joining trades with the most recent quote
- Enriching events with the latest state/status
- Combining irregularly sampled time series


Basic AS OF Join
----------------

.. code-block:: python

    # Join left with the most recent matching right row
    result = left_tsdf.asofJoin(right_tsdf, right_prefix="right")

**Parameters:**

- ``right_tsdf``: The TSDF to join with
- ``right_prefix``: Prefix added to right-side columns to avoid name collisions


Example: Trades and Quotes
--------------------------

A classic use case is joining trades with quotes:

.. code-block:: python

    from tempo import TSDF
    from pyspark.sql import functions as F

    # Trades dataset
    trades = spark.createDataFrame([
        ("AAPL", "2024-01-15 09:30:15.123", 185.50, 100),
        ("AAPL", "2024-01-15 09:30:45.456", 185.75, 200),
        ("AAPL", "2024-01-15 09:31:30.789", 185.60, 150),
    ], ["symbol", "trade_time", "price", "quantity"])
    trades = trades.withColumn("trade_time", F.to_timestamp("trade_time"))
    trades_tsdf = TSDF(trades, ts_col="trade_time", series_ids=["symbol"])

    # Quotes dataset (more frequent updates)
    quotes = spark.createDataFrame([
        ("AAPL", "2024-01-15 09:30:00.000", 185.45, 185.50),
        ("AAPL", "2024-01-15 09:30:10.000", 185.48, 185.52),
        ("AAPL", "2024-01-15 09:30:30.000", 185.50, 185.55),
        ("AAPL", "2024-01-15 09:31:00.000", 185.55, 185.60),
    ], ["symbol", "quote_time", "bid", "ask"])
    quotes = quotes.withColumn("quote_time", F.to_timestamp("quote_time"))
    quotes_tsdf = TSDF(quotes, ts_col="quote_time", series_ids=["symbol"])

    # Join: for each trade, find the most recent quote
    result = trades_tsdf.asofJoin(quotes_tsdf, right_prefix="quote")
    result.show()

Result:

.. code-block:: text

    +------+------------------------+------+--------+------------------------+----------+----------+
    |symbol|trade_time              |price |quantity|quote_quote_time        |quote_bid |quote_ask |
    +------+------------------------+------+--------+------------------------+----------+----------+
    |AAPL  |2024-01-15 09:30:15.123 |185.50|100     |2024-01-15 09:30:10.000 |185.48    |185.52    |
    |AAPL  |2024-01-15 09:30:45.456 |185.75|200     |2024-01-15 09:30:30.000 |185.50    |185.55    |
    |AAPL  |2024-01-15 09:31:30.789 |185.60|150     |2024-01-15 09:31:00.000 |185.55    |185.60    |
    +------+------------------------+------+--------+------------------------+----------+----------+


Join Strategies
---------------

Tempo automatically selects the best join strategy, but you can override this:

.. code-block:: python

    # Automatic selection (recommended)
    result = left.asofJoin(right, right_prefix="r")

    # Explicit strategy selection
    result = left.asofJoin(right, right_prefix="r", strategy="broadcast")
    result = left.asofJoin(right, right_prefix="r", strategy="union")
    result = left.asofJoin(right, right_prefix="r", strategy="skew")


**Strategy options:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - When to Use
   * - ``"broadcast"``
     - When the right dataset is small (< 30MB). Broadcasts right to all executors.
   * - ``"union"``
     - General-purpose strategy. Good default for most cases.
   * - ``"skew"``
     - When data is skewed (some series have much more data than others). Uses time-based partitioning.


Handling Skewed Data
--------------------

For skewed datasets, the ``skew`` strategy partitions by time windows:

.. code-block:: python

    result = left.asofJoin(
        right,
        right_prefix="r",
        strategy="skew",
        tsPartitionVal=10,   # Partition into 10-second buckets
        fraction=0.1         # Overlap fraction between partitions
    )

**Parameters:**

- ``tsPartitionVal``: Size of time partitions in seconds
- ``fraction``: Overlap between adjacent partitions (0.0 to 1.0)


Tolerance
---------

Limit how far back to look for a match:

.. code-block:: python

    # Only match quotes within 60 seconds of the trade
    result = trades.asofJoin(
        quotes,
        right_prefix="quote",
        tolerance=60  # Maximum time difference in seconds
    )

If no quote exists within the tolerance window, the right columns will be null.


Handling Nulls
--------------

Control how null values are handled:

.. code-block:: python

    # Skip null timestamps in the right dataset
    result = left.asofJoin(right, right_prefix="r", skipNulls=True)

    # Suppress warnings about null values
    result = left.asofJoin(right, right_prefix="r", suppress_null_warning=True)


Column Prefixes
---------------

Use prefixes to distinguish columns:

.. code-block:: python

    # Prefix right columns
    result = left.asofJoin(right, right_prefix="quote")

    # Prefix left columns too
    result = left.asofJoin(right, left_prefix="trade", right_prefix="quote")

After the join:

- Left columns: ``trade_timestamp``, ``trade_price``, etc.
- Right columns: ``quote_timestamp``, ``quote_bid``, etc.


Multi-Column Series IDs
-----------------------

AS OF joins work with multiple series identifiers:

.. code-block:: python

    # Both TSDFs have the same series identifiers
    left = TSDF(df1, ts_col="ts", series_ids=["symbol", "exchange"])
    right = TSDF(df2, ts_col="ts", series_ids=["symbol", "exchange"])

    # Join matches within each (symbol, exchange) combination
    result = left.asofJoin(right, right_prefix="r")


Performance Tips
----------------

1. **Use broadcast for small right tables**: If your right dataset fits in memory, use ``strategy="broadcast"`` for best performance.

2. **Filter before joining**: Reduce data size before the join:

   .. code-block:: python

       left_filtered = left.between("2024-01-15", "2024-01-16")
       result = left_filtered.asofJoin(right, right_prefix="r")

3. **Repartition for large joins**: Ensure data is well-distributed:

   .. code-block:: python

       left = left.repartitionBySeries(200)
       right = right.repartitionBySeries(200)
       result = left.asofJoin(right, right_prefix="r")

4. **Use tolerance**: If you only need recent matches, set a tolerance to prune the search space:

   .. code-block:: python

       result = left.asofJoin(right, right_prefix="r", tolerance=3600)  # 1 hour
