Joins
=====

The joins module provides AS OF join strategies for time series data.


Overview
--------

AS OF joins match rows from two time series by finding, for each left row, the most recent
right row that occurred at or before the left timestamp.

Tempo automatically selects the best join strategy, but you can override this for
specific use cases.


Strategy Selection
------------------

.. function:: tempo.joins.choose_as_of_join_strategy(left_tsdf, right_tsdf)

    Automatically select the best join strategy based on data characteristics.

    **Returns:** An instance of AsOfJoiner

Usage:

.. code-block:: python

    # Automatic selection (recommended)
    result = left.asofJoin(right, right_prefix="r")

    # Manual strategy selection
    result = left.asofJoin(right, right_prefix="r", strategy="broadcast")


Available Strategies
--------------------

BroadcastAsOfJoiner
^^^^^^^^^^^^^^^^^^^

Best for small right datasets that fit in memory (< 30MB).

.. code-block:: python

    result = left.asofJoin(right, right_prefix="r", strategy="broadcast")

**How it works:**

1. Broadcasts the right dataset to all executors
2. For each left row, searches the broadcast data for the latest matching right row
3. Very efficient when right dataset is small

**When to use:**

- Right dataset is small (< 30MB)
- Many left rows need to match against the same right data


UnionSortFilterAsOfJoiner
^^^^^^^^^^^^^^^^^^^^^^^^^

General-purpose strategy suitable for most cases.

.. code-block:: python

    result = left.asofJoin(right, right_prefix="r", strategy="union")

**How it works:**

1. Unions left and right datasets with a source marker
2. Sorts by partition keys and timestamp
3. Uses window functions to find the latest right row for each left row
4. Filters to keep only left rows with their matched right data

**When to use:**

- Default choice for medium to large datasets
- When neither broadcast nor skew is appropriate


SkewAsOfJoiner
^^^^^^^^^^^^^^

Optimized for skewed data where some series have much more data than others.

.. code-block:: python

    result = left.asofJoin(
        right,
        right_prefix="r",
        strategy="skew",
        tsPartitionVal=10,  # Time partition size in seconds
        fraction=0.1        # Overlap fraction
    )

**How it works:**

1. Partitions data into time buckets
2. Adds overlap between adjacent buckets to ensure correct matches at boundaries
3. Processes each bucket independently

**When to use:**

- Data is heavily skewed (some series have much more data)
- Time-based partitioning can improve parallelism

**Parameters:**

- ``tsPartitionVal`` - Size of time partitions in seconds
- ``fraction`` - Overlap fraction between partitions (0.0 to 1.0)


AsOfJoiner Base Class
---------------------

All strategies inherit from the abstract AsOfJoiner class:

.. code-block:: python

    from tempo.joins import AsOfJoiner

Abstract method that subclasses implement:

.. method:: join(left_tsdf, right_tsdf, left_prefix, right_prefix, ...)

    Perform the AS OF join.

    **Parameters:**

    - ``left_tsdf`` - Left TSDF
    - ``right_tsdf`` - Right TSDF
    - ``left_prefix`` - Prefix for left columns
    - ``right_prefix`` - Prefix for right columns
    - Additional strategy-specific parameters

    **Returns:** TSDF


Join Parameters
---------------

Common parameters for ``asofJoin()``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``right_tsdf``
     - The TSDF to join with
   * - ``left_prefix``
     - Prefix added to left columns (optional)
   * - ``right_prefix``
     - Prefix added to right columns (default: "right")
   * - ``strategy``
     - Join strategy: "broadcast", "union", "skew", or None (auto)
   * - ``tolerance``
     - Maximum time difference in seconds (optional)
   * - ``skipNulls``
     - Skip null timestamps in right dataset (default: True)
   * - ``suppress_null_warning``
     - Suppress warnings about nulls (default: False)


Skew-specific parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``tsPartitionVal``
     - Time partition size in seconds
   * - ``fraction``
     - Overlap fraction between partitions


Example: Strategy Comparison
----------------------------

.. code-block:: python

    from tempo import TSDF

    # Small right dataset - use broadcast
    trades = TSDF(trades_df, ts_col="trade_time", series_ids=["symbol"])
    reference = TSDF(ref_df, ts_col="ref_time", series_ids=["symbol"])  # Small

    result = trades.asofJoin(reference, right_prefix="ref", strategy="broadcast")

    # Large datasets - use union (auto-selected)
    quotes = TSDF(quotes_df, ts_col="quote_time", series_ids=["symbol"])  # Large

    result = trades.asofJoin(quotes, right_prefix="quote")  # Auto-selects union

    # Skewed data - use skew
    result = trades.asofJoin(
        quotes,
        right_prefix="quote",
        strategy="skew",
        tsPartitionVal=60,  # 1-minute buckets
        fraction=0.1
    )


Full API Reference
------------------

.. automodule:: tempo.joins
   :members:
   :undoc-members:
   :show-inheritance:
