TSSchema
========

The TSSchema module provides classes for defining time series schema metadata,
including timestamp indexes and window builder functionality.


TSSchema
--------

Defines the structure of a time series, including the timestamp index and series identifiers.

.. code-block:: python

    from tempo.tsschema import TSSchema, SimpleTSIndex

    # Create a schema
    schema = TSSchema(
        ts_idx=SimpleTSIndex("timestamp"),
        series_ids=["symbol"]
    )


TSIndex Classes
---------------

TSIndex classes define how the timestamp column is interpreted.

SimpleTSIndex
^^^^^^^^^^^^^

For timestamp columns that can be used directly:

.. code-block:: python

    from tempo.tsschema import SimpleTSIndex

    # Direct timestamp column reference
    ts_idx = SimpleTSIndex("event_time")

ParsedTSIndex
^^^^^^^^^^^^^

For string timestamps that need parsing:

.. code-block:: python

    from tempo.tsschema import ParsedTSIndex

    # String timestamp with format
    ts_idx = ParsedTSIndex(
        colname="timestamp_str",
        ts_format="yyyy-MM-dd HH:mm:ss.SSS"
    )

SubsequenceTSIndex
^^^^^^^^^^^^^^^^^^

For timestamps with a secondary ordering column:

.. code-block:: python

    from tempo.tsschema import SubsequenceTSIndex

    # Timestamp with sequence number for sub-ordering
    ts_idx = SubsequenceTSIndex(
        ts_col="event_time",
        subsequence_col="seq_num"
    )


WindowBuilder
-------------

Abstract base class that TSDF inherits from. Provides window specification methods.

Methods
^^^^^^^

.. method:: baseWindow(reverse=False)

    Create a basic partition-and-sort window.

    **Parameters:**

    - ``reverse`` - If True, sort in descending order

    **Returns:** WindowSpec

.. method:: rowsBetweenWindow(start, end, reverse=False)

    Create a window defined by row counts.

    **Parameters:**

    - ``start`` - Number of rows before current (negative) or after (positive)
    - ``end`` - Number of rows before current (negative) or after (positive)
    - ``reverse`` - Sort order

    **Returns:** WindowSpec

    **Example:**

    .. code-block:: python

        # Last 10 rows including current
        window = tsdf.rowsBetweenWindow(-9, 0)

        # Previous 5 rows (not including current)
        window = tsdf.rowsBetweenWindow(-5, -1)

.. method:: rangeBetweenWindow(start, end, reverse=False)

    Create a window defined by time range.

    **Parameters:**

    - ``start`` - Seconds before current (negative) or after (positive)
    - ``end`` - Seconds before current (negative) or after (positive)
    - ``reverse`` - Sort order

    **Returns:** WindowSpec

    **Example:**

    .. code-block:: python

        # Last 60 seconds
        window = tsdf.rangeBetweenWindow(-60, 0)

        # Last hour
        window = tsdf.rangeBetweenWindow(-3600, 0)

.. method:: allBeforeWindow(inclusive=True)

    Create a window of all rows before current.

    **Example:**

    .. code-block:: python

        # Cumulative sum
        window = tsdf.allBeforeWindow()
        cumsum = F.sum("value").over(window)

.. method:: allAfterWindow(inclusive=True)

    Create a window of all rows after current.


Constants
---------

.. data:: DEFAULT_TIMESTAMP_FORMAT

    Default format for parsing timestamp strings:

    ``"yyyy-MM-dd HH:mm:ss[.[SSSSSS][SSSSS][SSSS][SSS][SS][S]]"``


Utility Functions
-----------------

.. function:: is_time_format(format_str)

    Check if a string is a valid timestamp format.

.. function:: sub_seconds_precision_digits(ts_format)

    Get the number of sub-second precision digits in a format.


Full API Reference
------------------

.. automodule:: tempo.tsschema
   :members:
   :undoc-members:
   :show-inheritance:
