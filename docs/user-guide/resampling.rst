Resampling
==========

Resampling aggregates time series data to a different frequency. This is useful for:

- Reducing data granularity (e.g., tick data to minute bars)
- Aligning multiple time series to a common frequency
- Preparing data for interpolation


Basic Resampling
----------------

Use the ``resample()`` method:

.. code-block:: python

    # Resample to 1-minute intervals using mean aggregation
    resampled = tsdf.resample(freq="1 minute", func="mean")

**Parameters:**

- ``freq``: The target frequency (see frequency patterns below)
- ``func``: The aggregation function to use


Frequency Patterns
------------------

Tempo supports flexible frequency specifications:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pattern
     - Examples
   * - Seconds
     - ``"1 second"``, ``"30 seconds"``, ``"sec"``
   * - Minutes
     - ``"1 minute"``, ``"5 minutes"``, ``"min"``
   * - Hours
     - ``"1 hour"``, ``"4 hours"``
   * - Days
     - ``"1 day"``, ``"day"``

You can use singular or plural forms, and abbreviations for common units.


Aggregation Functions
---------------------

The ``func`` parameter determines how values within each time bucket are aggregated:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Function
     - Description
   * - ``"floor"``
     - Returns the earliest value by timestamp within each bucket
   * - ``"ceil"``
     - Returns the latest value by timestamp within each bucket
   * - ``"min"``
     - Returns the minimum value (ignores timestamp ordering)
   * - ``"max"``
     - Returns the maximum value (ignores timestamp ordering)
   * - ``"mean"``
     - Returns the average value

Example with different functions:

.. code-block:: python

    # Get the first value in each minute
    first_per_minute = tsdf.resample(freq="1 minute", func="floor")

    # Get the last value in each minute
    last_per_minute = tsdf.resample(freq="1 minute", func="ceil")

    # Get the average value in each minute
    avg_per_minute = tsdf.resample(freq="1 minute", func="mean")


OHLCV Bars
----------

For financial data, create OHLC (Open, High, Low, Close) bars:

.. code-block:: python

    # Calculate OHLCV bars at minute frequency
    bars = tsdf.calc_bars(freq="1 minute")

This creates:

- ``open``: First value in the period
- ``high``: Maximum value
- ``low``: Minimum value
- ``close``: Last value in the period
- ``volume``: Sum of volume (if volume column exists)


Upsampling with Fill
--------------------

When resampling to a higher frequency, you may want to fill missing intervals:

.. code-block:: python

    # Resample and fill missing intervals
    resampled = tsdf.resample(freq="1 minute", func="mean", fill=True)

The ``fill=True`` parameter creates rows for time intervals that have no data,
with null values for the metric columns.


Chaining with Interpolation
---------------------------

A common pattern is to resample and then interpolate missing values:

.. code-block:: python

    # Resample to 30-second intervals and fill missing values
    result = tsdf.resample(freq="30 seconds", func="mean").interpolate(method="ffill")

See :doc:`interpolation` for more details on fill methods.


Visualization Example
---------------------

After resampling, you can visualize the data:

.. code-block:: python

    import plotly.express as px

    # Resample to minute intervals
    resampled = tsdf.resample(freq="1 minute", func="mean")

    # Convert to pandas for plotting
    pdf = resampled.df.filter(
        F.col("timestamp").cast("date") == "2024-01-15"
    ).toPandas()

    # Create line chart
    fig = px.line(
        pdf,
        x="timestamp",
        y="price",
        color="symbol",
        title="Price by Symbol (1-minute intervals)"
    )
    fig.show()


Per-Series Behavior
-------------------

Resampling is always performed within each series independently. If you have multiple
series identifiers, each combination gets its own resampled time series:

.. code-block:: python

    # Data with two series: AAPL and GOOG
    # After resampling, you get separate minute bars for each

    resampled = tsdf.resample(freq="1 minute", func="mean")
    # Result has AAPL bars and GOOG bars, each with their own time grid


Working with the Result
-----------------------

Resampling returns a new TSDF, so you can chain additional operations:

.. code-block:: python

    result = (
        tsdf
        .resample(freq="5 minutes", func="mean")
        .latest(100)  # Last 100 5-minute bars per series
        .df  # Get underlying DataFrame
    )
