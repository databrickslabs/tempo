Roadmap
=======

This document outlines Tempo's development history and future plans.


Released in v0.2
----------------

The following features were introduced in v0.2:

**New API Architecture**

- ``series_ids`` parameter replaces ``partition_cols`` for clearer naming
- TSSchema and TSIndex classes for flexible timestamp handling
- TSDF now inherits from WindowBuilder for composable window operations

**Window Builder API**

- ``rowsBetweenWindow()`` for row-count based windows
- ``rangeBetweenWindow()`` for time-range based windows
- ``allBeforeWindow()`` and ``allAfterWindow()`` for unbounded windows
- ``baseWindow()`` for basic sort windows

**AS OF Join Strategies**

- ``BroadcastAsOfJoiner`` for small right datasets
- ``UnionSortFilterAsOfJoiner`` as the general-purpose default
- ``SkewAsOfJoiner`` for handling skewed data
- Automatic strategy selection via ``choose_as_of_join_strategy()``

**Intervals API**

- ``IntervalsDF`` class for working with time intervals
- ``extractStateIntervals()`` to convert point-in-time data to intervals
- Overlap detection and handling

**Enhanced TSDF Methods**

- DataFrame-like operations: ``withColumn()``, ``drop()``, ``select()``, ``where()``
- Series operations: ``groupBySeries()``, ``aggBySeries()``, ``applyToSeries()``
- Improved time filtering: ``priorTo()``, ``subsequentTo()``
- Repartitioning: ``repartitionBySeries()``, ``repartitionByTime()``

**Improved Interpolation**

- Function-based API replacing the class-based approach
- Cleaner integration with resampling via method chaining


In Development
--------------

Features currently being worked on:

**Documentation**

- Comprehensive user guides for all features
- Migration guide for v0.1 users
- Expanded API reference with examples

**Performance Improvements**

- Optimizations for large-scale joins
- Better partition management


Planned Features
----------------

The following features are planned for future releases:

**Streaming Support**

- Native support for Spark Structured Streaming
- Incremental updates for time series aggregations

**ML Integration**

- Integration with popular forecasting libraries:
  - Prophet
  - statsmodels
  - TensorFlow / PyTorch time series models
- Feature engineering for ML pipelines

**Advanced Analytics**

- Time series decomposition (trend, seasonality, residuals)
- Anomaly detection
- Change point detection

**Windowing Enhancements**

- Natural language window expressions (e.g., "last 7 days")
- Calendar-aware windows (business days, trading hours)

**Additional Interpolation Methods**

- Spline interpolation
- Polynomial interpolation
- Custom interpolation functions


Contributing to the Roadmap
---------------------------

We welcome input on the roadmap! If you have feature requests or suggestions:

1. Check existing `GitHub Issues <https://github.com/databrickslabs/tempo/issues>`_
2. Open a new issue with the "enhancement" label
3. Discuss in the issue before starting implementation

See :doc:`contributing` for details on how to contribute code.
