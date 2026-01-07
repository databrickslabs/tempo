User Guide
==========

This guide provides detailed documentation on Tempo's features with practical examples.
Start with the basics and progress to more advanced topics.

.. toctree::
   :maxdepth: 2

   tsdf-basics
   resampling
   joins
   interpolation
   intervals
   window-operations


Overview
--------

Tempo's functionality is built around the **TSDF** (Time Series DataFrame) class, which wraps
a Spark DataFrame and provides time series-aware operations. The key concepts are:

**Series Identifiers**
    Columns that partition your data into independent time series. For example, if you have
    stock data for multiple symbols, ``symbol`` would be a series identifier.

**Timestamp Column**
    The column that contains timestamps for ordering and time-based operations.

**Observational Columns**
    Data columns that contain measurements or values at each timestamp (prices, volumes, etc.).


Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Topic
     - Key Operations
   * - :doc:`tsdf-basics`
     - TSDF creation, properties, time filtering (``at``, ``between``, ``latest``)
   * - :doc:`resampling`
     - ``resample()``, frequency specifications, aggregation functions
   * - :doc:`joins`
     - ``asofJoin()``, strategy selection, prefix handling
   * - :doc:`interpolation`
     - ``interpolate()``, fill methods (ffill, bfill, linear)
   * - :doc:`intervals`
     - ``IntervalsDF``, ``extractStateIntervals()``, overlap handling
   * - :doc:`window-operations`
     - Window builders, ``rollingAgg()``, ``rollingApply()``
