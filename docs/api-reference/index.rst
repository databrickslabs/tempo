API Reference
=============

This section provides detailed API documentation for all Tempo classes and functions.

.. toctree::
   :maxdepth: 2

   tsdf
   tsschema
   intervals
   joins
   utilities


Core Classes
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`tempo.tsdf.TSDF`
     - Time Series DataFrame - the main class for time series operations
   * - :class:`tempo.tsschema.TSSchema`
     - Schema definition for time series structure
   * - :class:`tempo.intervals.IntervalsDF`
     - DataFrame wrapper for interval/period data


Join Strategies
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`tempo.joins.BroadcastAsOfJoiner`
     - AS OF join strategy for small right datasets
   * - :class:`tempo.joins.UnionSortFilterAsOfJoiner`
     - General-purpose AS OF join strategy
   * - :class:`tempo.joins.SkewAsOfJoiner`
     - AS OF join strategy for skewed data


Utilities
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - :mod:`tempo.resample_utils`
     - Resampling utility functions
   * - :mod:`tempo.timeunit`
     - Time unit definitions
   * - :mod:`tempo.interpol`
     - Interpolation functions
