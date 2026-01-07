Interpolation
=============

Interpolation fills missing or null values in time series data. This is essential when:

- Resampling creates gaps in the time grid
- Sensors have missing readings
- You need a regular time series for analysis or modeling


Basic Interpolation
-------------------

Use the ``interpolate()`` method:

.. code-block:: python

    # Forward fill missing values
    filled = tsdf.interpolate(method="ffill")

**Parameters:**

- ``method``: The interpolation method to use
- ``target_cols``: Specific columns to interpolate (optional, defaults to all numeric columns)


Interpolation Methods
---------------------

Tempo supports several interpolation methods:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description
   * - ``"ffill"``
     - Forward fill: Carry the last known value forward
   * - ``"bfill"``
     - Backward fill: Use the next known value
   * - ``"linear"``
     - Linear interpolation between known values
   * - ``"zero"``
     - Fill with zeros
   * - ``"null"``
     - Fill with nulls (explicit)


Forward Fill
^^^^^^^^^^^^

Carries the last known value forward until a new value is observed:

.. code-block:: python

    filled = tsdf.interpolate(method="ffill")

.. code-block:: text

    Before:  [1.0, null, null, 2.0, null]
    After:   [1.0, 1.0,  1.0,  2.0, 2.0 ]


Backward Fill
^^^^^^^^^^^^^

Uses the next known value to fill gaps:

.. code-block:: python

    filled = tsdf.interpolate(method="bfill")

.. code-block:: text

    Before:  [1.0, null, null, 2.0, null]
    After:   [1.0, 2.0,  2.0,  2.0, null]

Note: Trailing nulls remain null since there's no "next" value.


Linear Interpolation
^^^^^^^^^^^^^^^^^^^^

Calculates values proportionally between known points:

.. code-block:: python

    filled = tsdf.interpolate(method="linear")

.. code-block:: text

    Before:  [1.0, null, null, 4.0]
    After:   [1.0, 2.0,  3.0,  4.0]


Zero Fill
^^^^^^^^^

Replaces missing values with zero:

.. code-block:: python

    filled = tsdf.interpolate(method="zero")

.. code-block:: text

    Before:  [1.0, null, null, 2.0]
    After:   [1.0, 0.0,  0.0,  2.0]


Targeting Specific Columns
--------------------------

By default, interpolation applies to all numeric columns. To target specific columns:

.. code-block:: python

    # Only interpolate price and volume columns
    filled = tsdf.interpolate(
        method="ffill",
        target_cols=["price", "volume"]
    )


Chaining with Resample
----------------------

A common pattern is to resample data and then interpolate gaps:

.. code-block:: python

    # Resample to 1-minute intervals
    # This may create gaps where no data existed
    resampled = tsdf.resample(freq="1 minute", func="mean")

    # Fill the gaps with forward fill
    filled = resampled.interpolate(method="ffill")

    # Or chain in one expression
    result = tsdf.resample(freq="1 minute", func="mean").interpolate(method="ffill")


When using ``interpolate()`` after ``resample()``, you don't need to specify frequency
parameters again - the TSDF already knows its time structure.


Standalone Interpolation
------------------------

You can also use interpolation without resampling:

.. code-block:: python

    # Interpolate nulls in existing data
    filled = tsdf.interpolate(
        method="linear",
        target_cols=["temperature"]
    )


Per-Series Behavior
-------------------

Interpolation is performed independently for each time series (as defined by ``series_ids``):

.. code-block:: python

    # If series_ids=["symbol"], each symbol is interpolated independently
    tsdf = TSDF(df, ts_col="timestamp", series_ids=["symbol"])
    filled = tsdf.interpolate(method="ffill")

This means a forward fill for AAPL won't use values from GOOG.


Handling Edge Cases
-------------------

Leading Nulls
^^^^^^^^^^^^^

Forward fill can't fill values before the first known value:

.. code-block:: python

    # Leading nulls remain null with ffill
    # Before: [null, null, 1.0, 2.0]
    # After:  [null, null, 1.0, 2.0]

Consider using backward fill for leading nulls, or combining methods.


Trailing Nulls
^^^^^^^^^^^^^^

Backward fill can't fill values after the last known value:

.. code-block:: python

    # Trailing nulls remain null with bfill
    # Before: [1.0, 2.0, null, null]
    # After:  [1.0, 2.0, null, null]


All Nulls
^^^^^^^^^

If a column has no non-null values for a series, interpolation can't fill it:

.. code-block:: python

    # If price is all null for a series, it remains all null after interpolation


Supported Data Types
--------------------

Interpolation works with numeric column types:

- ``Integer`` / ``BigInt``
- ``Float`` / ``Double``
- ``Decimal``

Non-numeric columns are left unchanged.


Example: Sensor Data
--------------------

A complete example filling gaps in sensor data:

.. code-block:: python

    from tempo import TSDF
    from pyspark.sql import functions as F

    # Sensor data with missing readings
    data = [
        ("sensor_1", "2024-01-15 10:00:00", 22.5),
        ("sensor_1", "2024-01-15 10:01:00", None),  # Missing
        ("sensor_1", "2024-01-15 10:02:00", None),  # Missing
        ("sensor_1", "2024-01-15 10:03:00", 23.1),
        ("sensor_1", "2024-01-15 10:04:00", 23.4),
        ("sensor_1", "2024-01-15 10:05:00", None),  # Missing
    ]
    df = spark.createDataFrame(data, ["sensor_id", "timestamp", "temperature"])
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

    tsdf = TSDF(df, ts_col="timestamp", series_ids=["sensor_id"])

    # Option 1: Forward fill
    ffilled = tsdf.interpolate(method="ffill")
    # Result: [22.5, 22.5, 22.5, 23.1, 23.4, 23.4]

    # Option 2: Linear interpolation
    linear = tsdf.interpolate(method="linear")
    # Result: [22.5, 22.7, 22.9, 23.1, 23.4, null]

    # Option 3: Zero fill (for sparse data)
    zero = tsdf.interpolate(method="zero")
    # Result: [22.5, 0.0, 0.0, 23.1, 23.4, 0.0]
