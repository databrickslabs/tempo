Utilities
=========

Tempo provides several utility modules for common operations.


Resample Utilities
------------------

The ``resample_utils`` module contains functions and constants used by the resampling API.

.. code-block:: python

    from tempo.resample_utils import (
        floor, ceiling, min, max, average,
        ALLOWED_FREQ_KEYS,
        checkAllowableFreq,
    )


Aggregation Functions
^^^^^^^^^^^^^^^^^^^^^

.. function:: floor(df, ts_col, freq)

    Returns the earliest value by timestamp within each time bucket.

.. function:: ceiling(df, ts_col, freq)

    Returns the latest value by timestamp within each time bucket.

.. function:: min(df, ts_col, freq)

    Returns the minimum value within each time bucket.

.. function:: max(df, ts_col, freq)

    Returns the maximum value within each time bucket.

.. function:: average(df, ts_col, freq)

    Returns the mean value within each time bucket.


Frequency Validation
^^^^^^^^^^^^^^^^^^^^

.. data:: ALLOWED_FREQ_KEYS

    Set of allowed frequency keys:

    - ``"sec"``, ``"second"``, ``"seconds"``
    - ``"min"``, ``"minute"``, ``"minutes"``
    - ``"hr"``, ``"hour"``, ``"hours"``
    - ``"day"``, ``"days"``

.. function:: checkAllowableFreq(freq)

    Validate and parse a frequency string.

    **Parameters:**

    - ``freq`` - Frequency string (e.g., "1 minute", "30 seconds")

    **Returns:** Tuple of (count: int, unit: str)

    **Example:**

    .. code-block:: python

        count, unit = checkAllowableFreq("5 minutes")
        # count = 5, unit = "min"


Time Units
----------

The ``timeunit`` module provides time unit definitions.

.. code-block:: python

    from tempo.timeunit import TimeUnit, StandardTimeUnits


TimeUnit
^^^^^^^^

A named tuple representing a time unit:

.. code-block:: python

    TimeUnit = namedtuple("TimeUnit", ["name", "approx_seconds"])

StandardTimeUnits
^^^^^^^^^^^^^^^^^

Pre-defined standard time units:

.. code-block:: python

    from tempo.timeunit import StandardTimeUnits

    StandardTimeUnits.SECONDS      # TimeUnit("seconds", 1)
    StandardTimeUnits.MINUTES      # TimeUnit("minutes", 60)
    StandardTimeUnits.HOURS        # TimeUnit("hours", 3600)
    StandardTimeUnits.DAYS         # TimeUnit("days", 86400)
    StandardTimeUnits.WEEKS        # TimeUnit("weeks", 604800)
    StandardTimeUnits.MONTHS       # TimeUnit("months", 2592000)  # Approximate
    StandardTimeUnits.YEARS        # TimeUnit("years", 31536000)  # Approximate

    # Sub-second units
    StandardTimeUnits.MILLISECONDS # TimeUnit("milliseconds", 0.001)
    StandardTimeUnits.MICROSECONDS # TimeUnit("microseconds", 0.000001)
    StandardTimeUnits.NANOSECONDS  # TimeUnit("nanoseconds", 0.000000001)


Interpolation Functions
-----------------------

The ``interpol`` module provides interpolation functions.

.. code-block:: python

    from tempo.interpol import (
        interpolate,
        forward_fill,
        backward_fill,
        zero_fill,
    )


interpolate
^^^^^^^^^^^

.. function:: interpolate(tsdf, target_cols, method, leading_margin=0, lagging_margin=0)

    Interpolate missing values in a TSDF.

    **Parameters:**

    - ``tsdf`` - The TSDF to interpolate
    - ``target_cols`` - Columns to interpolate
    - ``method`` - Interpolation function (forward_fill, backward_fill, zero_fill)
    - ``leading_margin`` - Margin for leading nulls
    - ``lagging_margin`` - Margin for trailing nulls

    **Returns:** TSDF


Fill Methods
^^^^^^^^^^^^

.. function:: forward_fill

    Fill missing values with the previous known value.

.. function:: backward_fill

    Fill missing values with the next known value.

.. function:: zero_fill

    Fill missing values with zero.

**Example:**

.. code-block:: python

    from tempo.interpol import interpolate, forward_fill

    result = interpolate(
        tsdf,
        target_cols=["price", "volume"],
        method=forward_fill
    )


Type Definitions
----------------

The ``typing`` module provides type definitions for Tempo.

.. code-block:: python

    from tempo.typing import (
        ColumnOrName,
        PandasMapIterFunction,
        PandasGroupedMapFunction,
    )


ColumnOrName
^^^^^^^^^^^^

Union type for column specifications:

.. code-block:: python

    ColumnOrName = Union[Column, str]

Allows functions to accept either a column name string or a Spark Column object.


PandasMapIterFunction
^^^^^^^^^^^^^^^^^^^^^

Type for functions used with ``mapInPandas``:

.. code-block:: python

    PandasMapIterFunction = Callable[
        [Iterator[pd.DataFrame]],
        Iterator[pd.DataFrame]
    ]


PandasGroupedMapFunction
^^^^^^^^^^^^^^^^^^^^^^^^

Type for functions used with grouped pandas operations:

.. code-block:: python

    PandasGroupedMapFunction = Callable[
        [pd.DataFrame],
        pd.DataFrame
    ]


Utils Module
------------

The ``utils`` module provides general utility functions.

.. function:: get_display_df(tsdf, k)

    Get a DataFrame suitable for display with the latest k rows per series.

    **Parameters:**

    - ``tsdf`` - The TSDF
    - ``k`` - Number of rows to display

    **Returns:** DataFrame

.. function:: time_range(spark, start_time, end_time, step_size=None, num_intervals=None, ts_colname="ts")

    Generate a DataFrame with a time range.

    **Parameters:**

    - ``spark`` - SparkSession
    - ``start_time`` - Start of the range
    - ``end_time`` - End of the range
    - ``step_size`` - Time between points (timedelta)
    - ``num_intervals`` - Alternative: number of intervals
    - ``ts_colname`` - Name for the timestamp column

    **Returns:** DataFrame

    **Example:**

    .. code-block:: python

        from tempo.utils import time_range
        from datetime import datetime, timedelta

        time_df = time_range(
            spark,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            step_size=timedelta(hours=1),
            ts_colname="timestamp"
        )


Full API Reference
------------------

resample_utils
^^^^^^^^^^^^^^

.. automodule:: tempo.resample_utils
   :members:
   :undoc-members:

timeunit
^^^^^^^^

.. automodule:: tempo.timeunit
   :members:
   :undoc-members:

interpol
^^^^^^^^

.. automodule:: tempo.interpol
   :members:
   :undoc-members:

utils
^^^^^

.. automodule:: tempo.utils
   :members:
   :undoc-members:
