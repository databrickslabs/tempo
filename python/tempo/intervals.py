"""Time series interval analysis with overlap detection and resolution."""
from __future__ import annotations

from abc import abstractmethod, ABC
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import cached_property
from typing import Optional, Iterable, Callable, Sequence, Dict, Type, List, Union, TypeVar, Protocol, Generic

import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    # NumericType is a non-public object, so we shouldn't import it directly
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
    BooleanType,
    StructField,
)
from pyspark.sql.window import Window, WindowSpec


# Configuration
# -------------

class ErrorMessages:
    """Centralized error message definitions for consistent error handling"""
    INVALID_SERIES_IDS = "series_ids must be an Iterable or comma separated string of column names, got {}"
    NO_RESOLVER = "No resolver registered for overlap type: {}"
    RESOLUTION_FAILED = "Resolution failed for {}: {}"
    RESOLVER_REGISTERED = "Resolver already registered for {}"
    INTERVAL_INDICES = "Expected indices of interval elements to be equivalent"
    METRIC_COLUMNS_LENGTH = "Metric columns must have the same length"


# Type Definitions
# ---------------

IntervalData = TypeVar('IntervalData', bound=pd.Series)
IntervalBoundary = Union[str, int, float, pd.Timestamp, datetime]
MetricValue = Union[int, float, bool]
T = TypeVar('T')


# Abstract Base Classes & Interfaces
# ---------------------------------
# Abstract classes defining core interfaces and operations


class MetricOperation(Protocol):
    """Protocol defining required behavior for metric operations"""

    def normalize(self, value: MetricValue) -> MetricValue: ...

    def merge(self, value1: MetricValue, value2: MetricValue) -> MetricValue: ...


class MetricNormalizer(ABC):
    @abstractmethod
    def normalize(self, interval: Interval) -> pd.Series:
        pass


class OverlapChecker(ABC):
    """Abstract base class for overlap checking strategies"""
    @abstractmethod
    def check(self, interval: Interval, other: Interval) -> bool:
        pass


@dataclass
class OverlapResult:
    type: OverlapType
    details: Optional[Dict] = None


class OverlapResolver(ABC):
    @abstractmethod
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        pass


@dataclass
class ResolutionResult:
    """Represents the result of interval resolution"""
    resolved_intervals: List[pd.Series]
    metadata: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)


class MetricMerger(ABC):
    """Abstract base class defining metric merging strategy"""

    @abstractmethod
    def merge(self, interval: Interval, other: Interval) -> pd.Series:
        """Merge metrics from two intervals according to strategy"""
        pass

    @staticmethod
    def _validate_metric_columns(interval: Interval, other: Interval) -> None:
        """Validate that metric columns are aligned between intervals"""
        if len(interval.metric_fields) != len(other.metric_fields):
            raise ValueError(ErrorMessages.METRIC_COLUMNS_LENGTH)


@dataclass
class BoundaryConverter(Generic[T]):
    """
    Handles conversion between user-provided boundary types and internal pd.Timestamp.
    Maintains original format for converting back to user format.
    """
    to_timestamp: Callable[[T], pd.Timestamp]
    from_timestamp: Callable[[pd.Timestamp], T]
    original_type: type
    original_format: Optional[str] = None  # Store original string format if applicable

    @classmethod
    def for_type(cls, sample_value: IntervalBoundary) -> 'BoundaryConverter':
        """Factory method to create appropriate converter based on input type"""
        if isinstance(sample_value, str):
            return cls(
                to_timestamp=lambda x: pd.Timestamp(x),
                from_timestamp=lambda x: x.strftime(infer_datetime_format(sample_value)),
                original_type=str,
                original_format=sample_value  # Store complete original string
            )
        elif isinstance(sample_value, (int, float)) or sample_value is None:
            return cls(
                to_timestamp=lambda x: pd.Timestamp(x, unit='s'),
                from_timestamp=lambda x: x.timestamp(),
                original_type=type(sample_value)
            )
        elif isinstance(sample_value, datetime):
            return cls(
                to_timestamp=lambda x: pd.Timestamp(x),
                from_timestamp=lambda x: x.to_pydatetime(),
                original_type=datetime
            )
        elif isinstance(sample_value, pd.Timestamp):
            return cls(
                to_timestamp=lambda x: x,
                from_timestamp=lambda x: x,
                original_type=pd.Timestamp
            )
        else:
            raise ValueError(f"Unsupported boundary type: {type(sample_value)}")


@dataclass
class BoundaryValue:
    """
    Wrapper class that maintains both internal timestamp and original format.
    """
    _timestamp: pd.Timestamp
    _converter: BoundaryConverter

    @classmethod
    def from_user_value(cls, value: IntervalBoundary) -> BoundaryValue:
        converter = BoundaryConverter.for_type(value)
        return cls(
            _timestamp=converter.to_timestamp(value),
            _converter=converter
        )

    @property
    def internal_value(self) -> pd.Timestamp:
        """Get the internal timestamp representation"""
        return self._timestamp

    def to_user_value(self) -> IntervalBoundary:
        """Convert back to the original user format"""
        return self._converter.from_timestamp(self._timestamp)

    def __eq__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp == other._timestamp

    def __lt__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp < other._timestamp

    def __le__(self, other: 'BoundaryValue') -> bool:
        return self._timestamp <= other._timestamp


# Core Interval Classes
# ----------------------
# Main classes for interval data handling and DataFrame operations

class IntervalsDF:
    """
    A wrapper over a Spark DataFrame that enables parallel computations over time-series metric snapshots.

    This class provides functionality for handling time intervals defined by start and end timestamps,
    along with associated series identifiers and metrics. It handles operations like interval merging,
    overlap detection, and metric aggregation.

    Key Components:
    -------------
    - Series: List of columns used for summarization
    - Metrics: List of columns to analyze
    - Start/End Timestamps: Define the interval boundaries (can be epoch or TimestampType)

    Notes:
    -----
    - Complex Python objects cannot be referenced by UDFs due to PyArrow's data type limitations.
      See: https://arrow.apache.org/docs/python/api/datatypes.html
    - Nested iterations require checking that records haven't been previously added to
      global_disjoint_df by another loop to prevent duplicates
    - When processing overlapping intervals, resolution assumes two intervals cannot
      simultaneously report different values for the same metric unless using a
      data type supporting multiple elements

    Examples:
    --------
    >>> df = spark.createDataFrame(
    ...     [["2020-08-01 00:00:09", "2020-08-01 00:00:14", "v1", 5, 0]],
    ...     "start_ts STRING, end_ts STRING, series_1 STRING, metric_1 INT, metric_2 INT",
    ... )
    >>> idf = IntervalsDF(df, "start_ts", "end_ts", ["series_1"])
    >>> idf.df.collect()
    [Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:14', series_1='v1', metric_1=5, metric_2=0)]

    Todo:
    ----
    - Create IntervalsSchema class to validate data types and column existence
    - Check elements of series and identifiers to ensure all are str
    - Check if start_ts, end_ts, and the elements of series and identifiers can be of type col
    """

    def __init__(
            self,
            df: DataFrame,
            start_ts: str,
            end_ts: str,
            series_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """Constructor for IntervalsDF managing time intervals and metrics."""

        self.df = df

        self.start_ts = start_ts
        self.end_ts = end_ts

        self._validate_series_ids(series_ids)

    def _validate_series_ids(self, series_ids):
        if not series_ids:
            self.series_ids = []
        elif isinstance(series_ids, str):
            series_ids = series_ids.split(",")
            self.series_ids = [s.strip() for s in series_ids]
        elif isinstance(series_ids, Iterable):
            self.series_ids = list(series_ids)
        else:
            raise ValueError(
                ErrorMessages.INVALID_SERIES_IDS.format(type(series_ids))
            )

    @cached_property
    def interval_boundaries(self) -> list[str]:
        return [self.start_ts, self.end_ts]

    @cached_property
    def structural_columns(self) -> list[str]:
        return self.interval_boundaries + self.series_ids

    @cached_property
    def observational_columns(self) -> list[str]:
        return list(set(self.df.columns) - set(self.structural_columns))

    @cached_property
    def metric_columns(self) -> list[str]:
        return [col.name for col in self.df.schema.fields if is_metric_col(col)]

    @cached_property
    def window(self) -> WindowSpec:
        return Window.partitionBy(*self.series_ids).orderBy(*self.interval_boundaries)

    @classmethod
    def fromStackedMetrics(
            cls,
            df: DataFrame,
            start_ts: str,
            end_ts: str,
            series: list[str],
            metrics_name_col: str,
            metrics_value_col: str,
            metric_names: Optional[list[str]] = None,
    ) -> "IntervalsDF":
        """
        Returns a new :class:`IntervalsDF` with metrics of the current DataFrame
        pivoted by start and end timestamp and series.

        There are two versions of `fromStackedMetrics`. One that requires the caller
        to specify the list of distinct metric names to pivot on, and one that does
        not. The latter is more concise but less efficient, because Spark needs to
        first compute the list of distinct metric names internally.

        :param df: :class:`DataFrame` to wrap with :class:`IntervalsDF`
        :type df: `DataFrame`_
        :param start_ts: Name of the column which denotes interval start
        :type start_ts: str
        :param end_ts: Name of the column which denotes interval end
        :type end_ts: str
        :param series: column names
        :type series: list[str]
        :param metrics_name_col: column name
        :type metrics_name_col: str
        :param metrics_value_col: column name
        :type metrics_value_col: str
        :param metric_names: List of metric names that will be translated to
            columns in the output :class:`IntervalsDF`.
        :type metric_names: list[str], optional
        :return: A new :class:`IntervalsDF` with a column and respective
            values per distinct metric in `metrics_name_col`.

        :Example:

        .. code-block::

            df = spark.createDataFrame(
                [["2020-08-01 00:00:09", "2020-08-01 00:00:14", "v1", "metric_1", 5],
                 ["2020-08-01 00:00:09", "2020-08-01 00:00:11", "v1", "metric_2", 0]],
                "start_ts STRING, end_ts STRING, series_1 STRING, metric_name STRING, metric_value INT",
            )

            # With distinct metric names specified

            idf = IntervalsDF.fromStackedMetrics(
                df, "start_ts", "end_ts", ["series_1"], "metric_name", "metric_value", ["metric_1", "metric_2"],
            )
            idf.df.collect()
            [Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:14', series_1='v1', metric_1=5, metric_2=null),
             Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:11', series_1='v1', metric_1=null, metric_2=0)]

            # Or without specifying metric names (less efficient)

            idf = IntervalsDF.fromStackedMetrics(df, "start_ts", "end_ts", ["series_1"], "metric_name", "metric_value")
            idf.df.collect()
            [Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:14', series_1='v1', metric_1=5, metric_2=null),
             Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:11', series_1='v1', metric_1=null, metric_2=0)]

        .. _`DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html

        """

        if not isinstance(series, list):
            raise ValueError

        df = (
            df.groupBy(start_ts, end_ts, *series)
            .pivot(metrics_name_col, values=metric_names)
            .max(metrics_value_col)
        )

        return cls(df, start_ts, end_ts, series)

    def make_disjoint(self) -> "IntervalsDF":
        """
        Returns a new :class:`IntervalsDF` where metrics of overlapping time intervals
        are correlated and merged prior to constructing new time interval boundaries.

        The following examples demonstrate each type of overlap case and its resolution:

        1. No Overlap:
            Input:
                Row1: start='2020-01-01', end='2020-01-05', metric=10
                Row2: start='2020-01-06', end='2020-01-10', metric=20
            Output: Same as input (no changes needed)

        2. Boundary Equal (exact same interval):
            Input:
                Row1: start='2020-01-01', end='2020-01-05', metric1=10, metric2=null
                Row2: start='2020-01-01', end='2020-01-05', metric1=null, metric2=20
            Output:
                Row1: start='2020-01-01', end='2020-01-05', metric1=10, metric2=20

        3. Common Start:
            Input:
                Row1: start='2020-01-01', end='2020-01-05', metric=10
                Row2: start='2020-01-01', end='2020-01-07', metric=20
            Output:
                Row1: start='2020-01-01', end='2020-01-05', metric=10
                Row2: start='2020-01-05', end='2020-01-07', metric=20

        4. Common End:
            Input:
                Row1: start='2020-01-01', end='2020-01-07', metric=10
                Row2: start='2020-01-03', end='2020-01-07', metric=20
            Output:
                Row1: start='2020-01-01', end='2020-01-03', metric=10
                Row2: start='2020-01-03', end='2020-01-07', metric=20

        5. Interval Contained:
            Input:
                Row1: start='2020-01-01', end='2020-01-07', metric=10
                Row2: start='2020-01-03', end='2020-01-05', metric=20
            Output:
                Row1: start='2020-01-01', end='2020-01-03', metric=10
                Row2: start='2020-01-03', end='2020-01-05', metric=20
                Row3: start='2020-01-05', end='2020-01-07', metric=10

        6. Partial Overlap:
            Input:
                Row1: start='2020-01-01', end='2020-01-05', metric=10
                Row2: start='2020-01-03', end='2020-01-07', metric=20
            Output:
                Row1: start='2020-01-01', end='2020-01-03', metric=10
                Row2: start='2020-01-03', end='2020-01-05', metric=20
                Row3: start='2020-01-05', end='2020-01-07', metric=20

        7. Metrics Equivalent (overlapping intervals with same metrics):
            Input:
                Row1: start='2020-01-01', end='2020-01-05', metric=10
                Row2: start='2020-01-03', end='2020-01-07', metric=10
            Output:
                Row1: start='2020-01-01', end='2020-01-07', metric=10

        Returns:
            IntervalsDF: A new IntervalsDF containing disjoint time intervals

        Note:
            The resolution process assumes that two overlapping intervals cannot
            simultaneously report different values for the same metric unless recorded
            in a data type which supports multiple elements.
        """
        start_ts = self.start_ts
        end_ts = self.end_ts
        series_ids = self.series_ids

        disjoint_df = self.df.groupby(self.series_ids).applyInPandas(
            func=make_disjoint_wrap(
                self.start_ts,
                self.end_ts,
                self.series_ids,
                self.metric_columns,
            ),
            schema=self.df.schema,
        )

        return IntervalsDF(
            disjoint_df,
            start_ts,
            end_ts,
            series_ids,
        )

    def union(self, other: "IntervalsDF") -> "IntervalsDF":
        """
        Returns a new :class:`IntervalsDF` containing union of rows in this and another
        :class:`IntervalsDF`.

        This is equivalent to UNION ALL in SQL. To do a SQL-style set union
        (that does deduplication of elements), use this function followed by
        distinct().

        Also, as standard in SQL, this function resolves columns by position
        (not by name).

        Based on `pyspark.sql.DataFrame.union`_.

        :param other: :class:`IntervalsDF` to `union`
        :type other: :class:`IntervalsDF`
        :return: A new :class:`IntervalsDF` containing union of rows in this
            and `other`

        .. _`pyspark.sql.DataFrame.union`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.union.html

        """

        if not isinstance(other, IntervalsDF):
            raise TypeError

        return IntervalsDF(
            self.df.union(other.df), self.start_ts, self.end_ts, self.series_ids
        )

    def unionByName(self, other: "IntervalsDF") -> "IntervalsDF":
        """
        Returns a new :class:`IntervalsDF` containing union of rows in this
        and another :class:`IntervalsDF`.

        This is different from both UNION ALL and UNION DISTINCT in SQL. To do
        a SQL-style set union (that does deduplication of elements), use this
        function followed by distinct().

        Based on `pyspark.sql.DataFrame.unionByName`_; however,
        `allowMissingColumns` is not supported.

        :param other: :class:`IntervalsDF` to `unionByName`
        :type other: :class:`IntervalsDF`
        :return: A new :class:`IntervalsDF` containing union of rows in this
            and `other`

        .. _`pyspark.sql.DataFrame.unionByName`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.unionByName.html

        """

        if not isinstance(other, IntervalsDF):
            raise TypeError

        return IntervalsDF(
            self.df.unionByName(other.df),
            self.start_ts,
            self.end_ts,
            self.series_ids,
        )

    def toDF(self, stack: bool = False) -> DataFrame:
        """
        Returns a new `Spark DataFrame`_ converted from :class:`IntervalsDF`.

        There are two versions of `toDF`. One that will output columns as they exist
        in :class:`IntervalsDF` and, one that will stack metric columns into
        `metric_names` and `metric_values` columns populated with their respective
        values. The latter can be thought of as the inverse of
        :meth:`fromStackedMetrics`.

        Based on `pyspark.sql.DataFrame.toDF`_.

        :param stack: How to handle metric columns in the conversion to a `DataFrame`
        :type stack: bool, optional
        :return:

        .. _`Spark DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html
        .. _`pyspark.sql.DataFrame.toDF`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toDF.html
        .. _`STACK`: https://spark.apache.org/docs/latest/api/sql/index.html#stack

        """

        _metric_name, _metric_value = "metric_name", "metric_value"

        if stack:
            n_cols = len(self.metric_columns)
            metric_cols_expr = ",".join(
                tuple(f"'{col}', {col}" for col in self.metric_columns)
            )

            stack_expr = (
                f"STACK({n_cols}, {metric_cols_expr}) AS ({_metric_name}, {_metric_value})"
            )

            return self.df.select(
                *self.interval_boundaries,
                *self.series_ids,
                f.expr(stack_expr),
            ).dropna(subset=_metric_value)

        else:
            return self.df


@dataclass
class IntervalBoundaries:
    _start: BoundaryValue
    _end: BoundaryValue

    @classmethod
    def create(
            cls,
            start: Union[IntervalBoundary, BoundaryValue],
            end: Union[IntervalBoundary, BoundaryValue],
    ) -> 'IntervalBoundaries':
        # Convert only if not already a BoundaryValue
        start_boundary = start if isinstance(start, BoundaryValue) else BoundaryValue.from_user_value(start)
        end_boundary = end if isinstance(end, BoundaryValue) else BoundaryValue.from_user_value(end)

        return cls(
            _start=start_boundary,
            _end=end_boundary
        )


    @property
    def start(self) -> IntervalBoundary:
        """Get start boundary in user format"""
        return self._start.to_user_value()

    @property
    def end(self) -> IntervalBoundary:
        """Get end boundary in user format"""
        return self._end.to_user_value()

    @property
    def internal_start(self) -> pd.Timestamp:
        """Get internal timestamp representation of start"""
        return self._start

    @property
    def internal_end(self) -> pd.Timestamp:
        """Get internal timestamp representation of end"""
        return self._end


# This class handles the mapping between user-provided field names and our internal structure
class _BoundaryAccessor:
    """
    Manages access to interval boundaries using user-provided field names.

    This class serves as an adapter between the user's data structure (which uses
    field names to access values) and our internal representation (which uses
    proper objects). It maintains the flexibility of user-defined fields while
    providing a clean interface for the rest of our code.
    """

    def __init__(self, start_field: str, end_field: str):
        self.start_field = start_field
        self.end_field = end_field

    def get_boundaries(self, data: pd.Series) -> IntervalBoundaries:
        """Extract boundary values from data using configured field names"""
        return IntervalBoundaries.create(
            start=data[self.start_field],
            end=data[self.end_field],
        )

    def set_boundaries(self, data: pd.Series, boundaries: IntervalBoundaries) -> pd.Series:
        """Update data with boundary values using configured field names"""
        data = data.copy()
        data[self.start_field] = boundaries.start
        data[self.end_field] = boundaries.end
        return data


class Interval:
    """
    Represents a single point in a time series with start/end boundaries and associated metrics.

    An Interval is the fundamental unit of time series data in this system, containing:
    - Time boundaries (start and end timestamps)
    - Series identifiers (dimensional values that identify this series)
    - Metric values (measurements or observations for this time point)

    The class provides operations for:
    - Validating interval data and structure
    - Comparing and manipulating time boundaries
    - Managing metric data
    - Checking relationships with other intervals
    """

    @classmethod
    def create(
            cls,
            data: pd.Series,
            start_field: str,
            end_field: str,
            series_fields: Optional[Sequence[str]] = None,
            metric_fields: Optional[Sequence[str]] = None,
    ) -> 'Interval':
        """
        Creates a new Interval instance from a pandas Series and field mappings.

        This is the preferred way to construct an Interval, as it handles all the
        internal setup of field mappings and data validation.
        """
        # Create the accessor internally - users never need to know about it
        boundary_accessor = _BoundaryAccessor(start_field, end_field)

        if start_field is None or end_field is None:
            raise ValueError("start_field and end_field cannot be None")

        return cls(
            data=data,
            boundary_accessor=boundary_accessor,
            series_fields=series_fields,
            metric_fields=metric_fields
        )

    def __init__(
            self,
            data: pd.Series,
            boundary_accessor: _BoundaryAccessor,
            series_fields: Optional[Sequence[str]] = None,
            metric_fields: Optional[Sequence[str]] = None,
    ):
        """
        Initialize an interval with its data and metadata.

        Args:
            data: Series containing the interval's data
            boundary_accessor: Accessor for interval boundaries
            series_fields: Names of columns that identify the series
            metric_fields: Names of columns containing metric values
        """
        # self._validate_initialization(data, start_ts, end_ts, series_ids, metric_columns)

        self.data = data
        self.boundary_accessor = boundary_accessor
        self.series_fields = series_fields or []
        self.metric_fields = metric_fields or []

        # Internal cached representation for clean access
        self._boundaries = self.boundary_accessor.get_boundaries(data)

    # Validation Methods
    # -----------------

    def _validate_initialization(
            self,
            data: pd.Series,
            series_ids: Optional[Sequence[str]],
            metric_columns: Optional[Sequence[str]]
    ) -> None:
        """Validates all components during initialization"""
        self._validate_data(data)
        self._validate_series_ids(series_ids)
        self._validate_metric_columns(metric_columns)

    @property
    def start(self) -> IntervalBoundary:
        """Returns the start timestamp of the interval"""
        return self._boundaries.internal_start

    @property
    def start_field(self):
        return self.boundary_accessor.start_field

    @property
    def end(self) -> IntervalBoundary:
        """Returns the end timestamp of the interval"""
        return self._boundaries.internal_end

    @property
    def end_field(self) -> str:

        return self.boundary_accessor.end_field

    @property
    def boundaries(self) -> tuple[IntervalBoundary, IntervalBoundary]:
        """Returns the start and end timestamps as a Series"""
        return self._boundaries.internal_start, self._boundaries.internal_end

    def _validate_data(self, data: pd.Series) -> None:
        """Validates the basic data structure"""
        if not isinstance(data, pd.Series):
            raise InvalidDataTypeError("Data must be a pandas Series")
        if data.empty:
            raise EmptyIntervalError("Data cannot be empty")


    def _validate_series_ids(self, series_ids: Optional[Sequence[str]]) -> None:
        """Validates series identifier columns"""
        if series_ids is not None:
            if not isinstance(series_ids, Sequence) or isinstance(series_ids, str):
                raise InvalidMetricColumnError("series_ids must be a sequence")
            if not all(isinstance(col, str) for col in series_ids):
                raise InvalidMetricColumnError("All series_ids must be strings")

    def _validate_metric_columns(self, metric_columns: Optional[Sequence[str]]) -> None:
        """Validates metric column names"""
        if metric_columns is not None:
            if not isinstance(metric_columns, Sequence) or isinstance(metric_columns, str):
                raise InvalidMetricColumnError("metric_columns must be a sequence")
            if not all(isinstance(col, str) for col in metric_columns):
                raise InvalidMetricColumnError("All metric_columns must be strings")

    # Time Operations
    # --------------

    def update_start(self, new_start: IntervalBoundary) -> 'Interval':
        """
        Creates a new interval with updated start time while maintaining
        the user's field mapping structure
        """
        new_boundaries = IntervalBoundaries.create(start=new_start, end=self.end)
        new_data = self.boundary_accessor.set_boundaries(self.data, new_boundaries)

        return Interval(
            new_data,
            self.boundary_accessor,
            self.series_fields,
            self.metric_fields
        )

    def update_end(self, new_end: Union[IntervalBoundary, BoundaryValue]) -> 'Interval':
        """
        Creates a new interval with updated end time while maintaining
        the user's field mapping structure
        """
        new_boundaries = IntervalBoundaries.create(start=self.start, end=new_end)
        new_data = self.boundary_accessor.set_boundaries(self.data, new_boundaries)

        return Interval(
            new_data,
            self.boundary_accessor,
            self.series_fields,
            self.metric_fields
        )

    # Interval Relationships
    # ---------------------

    def contains(self, other: 'Interval') -> bool:
        """Checks if this interval fully contains another interval"""
        return self.start <= other.start and self.end >= other.start

    def overlaps_with(self, other: 'Interval') -> bool:
        """Checks if this interval overlaps with another interval"""
        return self.start < other.end and self.end > other.start

    # Metric Operations
    # ----------------

    def merge_metrics(self, other: 'Interval') -> pd.Series:
        """Combines metrics between intervals according to merging strategy"""
        merger = NonNullOverwriteMetricMerger()
        return merger.merge(self, other)

    def validate_metrics_alignment(self, other: 'Interval') -> ValidationResult:
        """Validates that metric columns align between intervals"""
        return self._validate_column_alignment(
            self.metric_fields,
            other.metric_fields,
            "Metric columns"
        )

    def validate_series_alignment(self, other: 'Interval') -> ValidationResult:
        """Validates that series identifiers align between intervals"""
        return self._validate_column_alignment(
            self.series_fields,
            other.series_fields,
            "Series IDs"
        )

    def _validate_column_alignment(
            self,
            self_columns: Sequence[str],
            other_columns: Sequence[str],
            column_type: str
    ) -> ValidationResult:
        """Generic validator for column alignment between intervals"""
        if len(self_columns) != len(other_columns):
            raise InvalidMetricColumnError(
                f"{column_type} count mismatch: {len(self_columns)} vs {len(other_columns)}"
            )
        if set(self_columns) != set(other_columns):
            raise InvalidMetricColumnError(
                f"{column_type} don't match: {self_columns} vs {other_columns}"
            )
        return ValidationResult(is_valid=True)


# Validation Classes
# ------------------
# Classes handling data validation and error checking


class IntervalValidationError(Exception):
    """Base exception for interval validation errors"""
    pass


class EmptyIntervalError(IntervalValidationError):
    """Raised when interval data is empty"""
    pass


class InvalidDataTypeError(IntervalValidationError):
    """Raised when data is not of expected type"""
    pass


class InvalidTimestampError(IntervalValidationError):
    """Raised when timestamps are invalid"""
    pass


class InvalidMetricColumnError(IntervalValidationError):
    """Raised when metric columns are invalid"""
    pass


class IntervalValidator:
    """Validates interval data and properties"""

    @staticmethod
    def validate_data(data: pd.Series) -> ValidationResult:
        if data.empty:
            raise EmptyIntervalError("Data must not be empty")

        if not isinstance(data, pd.Series):
            raise InvalidDataTypeError("Expected data to be a Pandas Series")

        return ValidationResult(is_valid=True)

    @staticmethod
    def _validate_columns(columns: Optional[Sequence[str]], column_type: str) -> ValidationResult:
        if columns is None:
            return ValidationResult(is_valid=True)

        if not (isinstance(columns, Sequence) and not isinstance(columns, str)):
            raise InvalidMetricColumnError(f"{column_type} must be a sequence")

        if not all(isinstance(col, str) for col in columns):
            raise InvalidMetricColumnError(f"All {column_type} must be of type str")

        if len(set(columns)) != len(columns):
            raise InvalidMetricColumnError(f"Duplicate {column_type} found")

        return ValidationResult(is_valid=True)

    @staticmethod
    def validate_series_id_columns(series_ids: Optional[Sequence[str]]) -> ValidationResult:
        return IntervalValidator._validate_columns(series_ids, "series ID columns")

    @staticmethod
    def validate_metric_columns(metric_columns: Optional[Sequence[str]]) -> ValidationResult:
        return IntervalValidator._validate_columns(metric_columns, "metric columns")


@dataclass
class ValidationResult:
    is_valid: bool
    message: Optional[str] = None


# Metric Processing
# -----------------
# Classes for handling metric operations and transformations


class NullMetricNormalizer(MetricNormalizer):
    def normalize(self, interval: Interval) -> pd.Series:
        return interval.data[interval.metric_fields].fillna(pd.NA)


class NonNullOverwriteMetricMerger(MetricMerger):
    """Implementation that overwrites with non-null values from other interval"""

    def merge(self, interval: Interval, other: Interval) -> pd.Series:
        self._validate_metric_columns(interval, other)

        # Create a copy of interval's data
        merged_data = interval.data.copy()

        # Overwrite with non-NaN values from other
        for interval_col, other_col in zip(interval.metric_fields, other.metric_fields):
            merged_data[interval_col] = (
                other.data[other_col]
                if pd.notna(other.data[other_col])
                else merged_data[interval_col]
            )

        return merged_data


# Overlap Detection & Resolution
# -----------------------------
# Classes handling overlap detection and resolution logic

# TODO: The main interval relationship types in interval algebra:
#
# Before (b): A ends before B starts
# Meets (m): A ends exactly when B starts
# Overlaps (o): A starts before B and they overlap, but A ends before B
# Starts (s): A and B start together, but A ends before B
# During (d): A starts after B and ends before B
# Finishes (f): A starts after B but they end together
# Equals (e): A and B have the same start and end
class OverlapType(Enum):
    """Enumeration of possible interval overlap types"""
    BOUNDARY_EQUAL = auto()
    COMMON_END = auto()
    COMMON_START = auto()
    INTERVAL_CONTAINED = auto()
    INTERVAL_ENDS_FIRST = auto()
    INTERVAL_STARTS_FIRST = auto()
    METRICS_EQUIVALENT = auto()
    NO_OVERLAP = auto()
    OTHER_CONTAINED = auto()
    OTHER_ENDS_FIRST = auto()
    OTHER_STARTS_FIRST = auto()
    PARTIAL_OVERLAP = auto()


class NoOverlapChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.end < other.start or interval.start > other.end


class MetricsEquivalentChecker(OverlapChecker):
    def __init__(self, normalizer: MetricNormalizer = NullMetricNormalizer()):
        self.normalizer = normalizer

    def check(self, interval: Interval, other: Interval) -> bool:
        interval_metrics = self.normalizer.normalize(interval)
        other_metrics = self.normalizer.normalize(other)
        return interval_metrics.equals(other_metrics)


class IntervalStartsFirstChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.start < other.start


class OtherStartsFirstChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return other.start < interval.start


class IntervalEndsFirstChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.end < other.end


class OtherEndsFirstChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return other.end < interval.end


class IntervalContainedChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return OtherStartsFirstChecker().check(interval, other) and IntervalEndsFirstChecker().check(interval, other)


class OtherContainedChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return IntervalStartsFirstChecker().check(interval, other) and OtherEndsFirstChecker().check(interval, other)


class StartBoundaryEqualityChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.start == other.start


class EndBoundaryEqualityChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.end == other.end


class BoundaryEqualityChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return (
                StartBoundaryEqualityChecker().check(interval, other)
                and EndBoundaryEqualityChecker().check(interval, other)
        )


class CommonStartChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return (
                StartBoundaryEqualityChecker().check(interval, other)
                and not EndBoundaryEqualityChecker().check(interval, other)
        )


class CommonEndChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return (
                not StartBoundaryEqualityChecker().check(interval, other)
                and EndBoundaryEqualityChecker().check(interval, other)
        )


class PartialOverlapChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return (
                IntervalStartsFirstChecker().check(interval, other)
                and IntervalEndsFirstChecker().check(interval, other)
        )


class OverlapDetector:
    def __init__(
            self,
            interval: Interval,
            other: Interval,
    ) -> None:
        self.interval = interval
        self.other = other
        self._checkers = self._initialize_overlap_checkers()

    def detect_overlap_type(self) -> Optional[OverlapResult]:
        """Returns the most specific type of overlap found"""
        for overlap_type, checker in self._checkers.items():
            if checker.check(self.interval, self.other):
                return OverlapResult(overlap_type)
        return None

    def _initialize_overlap_checkers(self) -> Dict[OverlapType, OverlapChecker]:
        """
        Initializes ordered checkers from most specific to most general.
        To add new checks:
        1. Create new OverlapChecker implementation
        2. Add new type to OverlapType enum
        3. Add checker instance here in appropriate order
        """
        return OrderedDict([
            (OverlapType.NO_OVERLAP, NoOverlapChecker()),
            (OverlapType.METRICS_EQUIVALENT, MetricsEquivalentChecker()),
            (OverlapType.INTERVAL_CONTAINED, IntervalContainedChecker()),
            (OverlapType.OTHER_CONTAINED, OtherContainedChecker()),
            (OverlapType.COMMON_START, CommonStartChecker()),
            (OverlapType.COMMON_END, CommonEndChecker()),
            (OverlapType.BOUNDARY_EQUAL, BoundaryEqualityChecker()),
            (OverlapType.PARTIAL_OVERLAP, PartialOverlapChecker())
        ])

    def register_checker(self, overlap_type: OverlapType, checker: Type[OverlapChecker]) -> None:
        """Adds a new checker type at runtime"""
        self._checkers[overlap_type] = checker()


class NoOverlapResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        return [interval.data, other.data]


class EquivalentMetricsResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        return [interval.update_end(other.end).data]


class ContainedResolverBase(OverlapResolver):
    """Base class for resolving contained intervals"""

    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        resolved = [
            # First interval - update end boundary
            interval.update_end(other.start).data,
            # Middle interval - merge metrics
            self._merge_metrics(interval, other),
            # Last interval - update start boundary
            interval.update_start(other.end).data]

        return resolved

    @abstractmethod
    def _merge_metrics(self, interval: Interval, other: Interval) -> pd.Series:
        """Define how metrics should be merged based on containment type"""
        pass


class IntervalContainedResolver(ContainedResolverBase):
    def _merge_metrics(self, interval: Interval, other: Interval) -> pd.Series:
        return NonNullOverwriteMetricMerger().merge(interval, other)


class OtherContainedResolver(ContainedResolverBase):
    def _merge_metrics(self, interval: Interval, other: Interval) -> pd.Series:
        return NonNullOverwriteMetricMerger().merge(other, interval)


class CommonStartResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        resolved = []

        if IntervalEndsFirstChecker().check(interval, other):
            # 1)
            resolved_series = NonNullOverwriteMetricMerger().merge(interval, other)

            resolved.append(resolved_series)

            # 2)
            resolved_series = other.update_start(interval.end).data

            resolved.append(resolved_series)

        else:
            # 1)
            resolved_series = NonNullOverwriteMetricMerger().merge(other, interval)

            resolved.append(resolved_series)

            # 2)
            resolved_series = interval.update_start(other.end).data

            resolved.append(resolved_series)

        return resolved


class CommonEndResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        resolved = []

        if IntervalStartsFirstChecker().check(interval, other):
            # 1)
            resolved_series = interval.update_end(other.start).data

            resolved.append(resolved_series)

            # 2)
            resolved_series = NonNullOverwriteMetricMerger().merge(other, interval)

            resolved.append(resolved_series)

        return resolved


class BoundaryEqualityResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        return [NonNullOverwriteMetricMerger().merge(interval, other)]


class PartialOverlapResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        resolved = []
        # 1)
        resolved_series = interval.update_end(other.start).data

        resolved.append(resolved_series)

        # 2)
        updated_series = other.update_end(interval.end).data
        resolved_series = NonNullOverwriteMetricMerger().merge(
            Interval.create(
                updated_series,
                other.start_field,
                other.end_field,
                series_fields=other.series_fields,
                metric_fields=other.metric_fields,
            ),
            interval,
        )

        resolved.append(resolved_series)

        # 3)
        resolved_series = other.update_start(interval.end).data

        resolved.append(resolved_series)

        return resolved


class ResolutionManager:
    """
    Handles the coordination of overlap resolution between intervals using registered resolution strategies
    """

    def __init__(self):
        self._resolvers = self._init_resolvers()

    def _init_resolvers(self) -> Dict[OverlapType, OverlapResolver]:
        """
        Initializes mapping of overlap types to their resolvers.
        Order matches overlap detection hierarchy.
        """
        return OrderedDict([
            (OverlapType.BOUNDARY_EQUAL, BoundaryEqualityResolver()),
            (OverlapType.COMMON_END, CommonEndResolver()),
            (OverlapType.COMMON_START, CommonStartResolver()),
            (OverlapType.INTERVAL_CONTAINED, IntervalContainedResolver()),
            (OverlapType.METRICS_EQUIVALENT, EquivalentMetricsResolver()),
            (OverlapType.NO_OVERLAP, NoOverlapResolver()),
            (OverlapType.OTHER_CONTAINED, OtherContainedResolver()),
            (OverlapType.PARTIAL_OVERLAP, PartialOverlapResolver()),
        ])

    def resolve(self, overlap_type: OverlapType, interval: Interval, other: Interval) -> ResolutionResult:
        """
        Resolves overlap between intervals using appropriate strategy

        Args:
            overlap_type: Type of overlap detected
            interval: First interval
            other: Second interval

        Returns:
            ResolutionResult containing resolved intervals and metadata

        Raises:
            ValueError: If no resolver exists for overlap type
        """
        resolver = self._resolvers.get(overlap_type)
        if not resolver:
            raise ValueError(ErrorMessages.NO_RESOLVER.format(overlap_type))

        try:
            resolved = resolver.resolve(interval, other)
            return ResolutionResult(resolved_intervals=resolved)
        except Exception as e:
            raise ValueError(ErrorMessages.RESOLUTION_FAILED.format(overlap_type, str(e)))

    def register_resolver(self, overlap_type: OverlapType, resolver: Type[OverlapResolver]) -> None:
        """Registers a new resolver for given overlap type"""
        if overlap_type in self._resolvers:
            raise ValueError(ErrorMessages.RESOLVER_REGISTERED.format(overlap_type))
        self._resolvers[overlap_type] = resolver()


# Utility Classes
# ---------------
# Supporting utility classes and helpers


class IntervalsUtils:
    def __init__(
            self,
            intervals: pd.DataFrame,
    ):
        """
        Initialize IntervalsUtils with a DataFrame and interval properties.

        intervals_set (pd.DataFrame): Input DataFrame containing interval data, with start and end timestamps.
        interval (Interval): An object representing the reference interval, containing the following:
            - data (pd.Series): A Series representing the interval's attributes.
            - start_ts (str): Column name for the start timestamp.
            - end_ts (str): Column name for the end timestamp.
        """
        self.intervals = intervals
        self._disjoint_set = pd.DataFrame()

    @property
    def disjoint_set(self):
        return self._disjoint_set

    @disjoint_set.setter
    def disjoint_set(self, value: pd.DataFrame):
        self._disjoint_set = value

    def _calculate_all_overlaps(self, interval):
        max_start = "_MAX_START_TS"
        max_end = "_MAX_END_TS"

        # Check if input DataFrame or interval data is empty; return an empty DataFrame if true.
        if self.intervals.empty or interval.data.empty:
            # return in_pdf
            return pd.DataFrame()

        intervals_copy = self.intervals.copy()

        # Calculate the latest possible start timestamp for overlap comparison.
        intervals_copy[max_start] = intervals_copy[
            interval.start_field
        ].where(
            intervals_copy[interval.start_field]
            >= interval.data[interval.start_field],
            interval.data[interval.start_field],
        )

        # Calculate the earliest possible end timestamp for overlap comparison.
        intervals_copy[max_end] = intervals_copy[
            interval.end_field
        ].where(
            intervals_copy[interval.end_field] <= interval.data[interval.end_field],
            interval.data[interval.end_field],
        )

        # https://www.baeldung.com/cs/finding-all-overlapping-intervals
        intervals_copy = intervals_copy[
            intervals_copy[max_start] < intervals_copy[max_end]
            ]

        # Remove intermediate columns used for interval overlap calculation.
        cols_to_drop = [max_start, max_end]
        intervals_copy = intervals_copy.drop(columns=cols_to_drop)

        return intervals_copy

    def find_overlaps(
            self,
            interval: Interval
    ) -> pd.DataFrame:

        all_overlaps = self._calculate_all_overlaps(interval)

        # Remove rows that are identical to `interval.data`
        remove_with_row_mask = ~(
                all_overlaps.isna().eq(interval.data.isna())
                & all_overlaps.eq(interval.data).fillna(False)
        ).all(axis=1)

        deduplicated_overlaps = all_overlaps[remove_with_row_mask]

        return deduplicated_overlaps

    def add_as_disjoint(self, interval: Interval) -> pd.DataFrame:
        """
        returns a disjoint set consisting of the given interval, made disjoint with those already in `disjoint_set`
        """

        if self.disjoint_set is None or self.disjoint_set.empty:
            return pd.DataFrame([interval.data])

        overlapping_subset_df = IntervalsUtils(self.disjoint_set).find_overlaps(interval)

        # if there are no overlaps, add the interval to disjoint_set
        if overlapping_subset_df.empty:
            element_wise_comparison = self.disjoint_set.copy().fillna(pd.NA) == interval.data.fillna(
                pd.NA).values

            row_wise_comparison = element_wise_comparison.all(axis=1)
            # NB: because of the nested iterations, we need to check that the
            # record hasn't already been added to `global_disjoint_df` by another loop
            if row_wise_comparison.any():
                return self.disjoint_set
            else:
                return pd.concat((self.disjoint_set, pd.DataFrame([interval.data])))

        # identify all intervals which do not overlap with the given interval to
        # concatenate them to the disjoint set after resolving overlaps
        non_overlapping_subset_df = self.disjoint_set[

            ~self.disjoint_set.set_index(keys=[interval.start_field, interval.end_field]).index.isin(
                overlapping_subset_df.set_index(keys=[interval.start_field, interval.end_field]).index
            )
        ]

        # Avoid a call to `resolve_all_overlaps` if there is only one to resolve
        multiple_to_resolve = len(overlapping_subset_df.index) > 1

        # If every record overlaps, no need to handle non-overlaps
        only_overlaps_present = len(self.disjoint_set.index) == len(overlapping_subset_df.index)

        # Resolve the interval against all the existing, overlapping intervals
        # `multiple_to_resolve` is used to avoid unnecessary calls to `resolve_all_overlaps`
        # `only_overlaps_present` is used to avoid unnecessary calls to `pd.concat`
        if not multiple_to_resolve and only_overlaps_present:
            resolver = IntervalTransformer(
                interval=Interval.create(
                    interval.data,
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                ),
                other=Interval.create(
                    overlapping_subset_df.iloc[0],
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                ),
            )
            return pd.DataFrame(
                resolver.resolve_overlap()
            )

        if multiple_to_resolve and only_overlaps_present:
            return IntervalsUtils(overlapping_subset_df).resolve_all_overlaps(interval)

        if not multiple_to_resolve and not only_overlaps_present:
            resolver = IntervalTransformer(
                interval=Interval.create(
                    interval.data,
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                ),
                other=Interval.create(
                    overlapping_subset_df.iloc[0],
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                ),
            )
            return pd.concat(
                (
                    pd.DataFrame(
                        resolver.resolve_overlap()
                    ),
                    non_overlapping_subset_df,
                ),
            )

        if multiple_to_resolve and not only_overlaps_present:
            return pd.concat(
                (
                    IntervalsUtils(overlapping_subset_df).resolve_all_overlaps(interval),
                    non_overlapping_subset_df,
                ),
            )

        # if we get here, something went wrong
        raise NotImplementedError("Interval resolution not implemented")

    def resolve_all_overlaps(
            self,
            interval: Interval,
    ) -> pd.DataFrame:
        """
        resolve the interval `x` against all overlapping intervals in `overlapping`,
        returning a set of disjoint intervals with the same spans
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
        """

        first_row = Interval.create(
            self.intervals.iloc[0],
            interval.start_field,
            interval.end_field,
            interval.series_fields,
            interval.metric_fields,
        )
        resolver = IntervalTransformer(interval, first_row)
        initial_intervals = resolver.resolve_overlap()
        disjoint_intervals = pd.DataFrame(initial_intervals)

        def resolve_and_add(row):
            row_interval = Interval.create(
                row,
                interval.start_field,
                interval.end_field,
                interval.series_fields,
                interval.metric_fields,
            )
            local_resolver = IntervalTransformer(interval, row_interval)
            resolved_intervals = local_resolver.resolve_overlap()
            for interval_data in resolved_intervals:
                interval_inner = Interval.create(
                    interval_data,
                    interval.start_field,
                    interval.end_field,
                    interval.series_fields,
                    interval.metric_fields,
                )
                nonlocal disjoint_intervals
                local_interval_utils = IntervalsUtils(disjoint_intervals)
                local_interval_utils.disjoint_set = disjoint_intervals
                disjoint_intervals = local_interval_utils.add_as_disjoint(interval_inner)

        self.intervals.iloc[1:].apply(resolve_and_add, axis=1)

        return disjoint_intervals


# TODO: does it make sense to move this to inside of the Interval class?
class IntervalTransformer:
    def __init__(
            self,
            interval: Interval,
            other: Interval,
    ) -> None:
        """
        Initialize OverlapResolver with two intervals, ensuring the one with the earlier start comes first.

        Args:
            interval (Interval): The first interval.
            other (Interval): The second interval.
        """

        # Ensure intervals are ordered by start time
        if interval.start <= other.start:
            self.interval = interval
            self.other = other
        else:
            self.interval = other
            self.other = interval

        self.overlap_detector = OverlapDetector(self.interval, self.other)
        self.resolution_manager = ResolutionManager()

    def validate_intervals(self):

        if set(self.interval.data.index) != set(self.other.data.index):
            raise ValueError(ErrorMessages.INTERVAL_INDICES)

    def resolve_overlap(  # TODO: need to implement proper metric merging
            #  -> for now, can just take non-null values from both intervals
            self
    ) -> list[pd.Series]:
        """
        resolve overlaps between the two given intervals,
        splitting them as necessary into some set of disjoint intervals
        """

        self.validate_intervals()

        overlap_result = self.overlap_detector.detect_overlap_type()
        if overlap_result is None:
            raise NotImplementedError("Unable to determine overlap type")

        resolution_result = self.resolution_manager.resolve(
            overlap_result.type,
            self.interval,
            self.other
        )

        return resolution_result.resolved_intervals

    def merge_metrics(
            self,
            new_interval: Optional[Interval] = None,
            new_other: Optional[Interval] = None,
            metric_merge_method: bool = False,
    ) -> pd.Series:
        """
        Return a merged set of metrics from `interval` and `other`.

        Parameters:
            metric_merge_method (bool): If True, non-NaN metrics from `other` will overwrite metrics in `interval`.

        Returns:
            pd.Series: A merged set of metrics as a Pandas Series.

        Raises:
            ValueError: If `interval.metric_columns` and `other.metric_columns` are not aligned.
            :param metric_merge_method:
            :param new_other:
            :param new_interval:
        """
        # Determine which intervals to use
        interval_to_use = new_interval if new_interval else self.interval
        other_to_use = new_other if new_other else self.other

        # Validate that metric columns align
        if len(interval_to_use.metric_fields) != len(other_to_use.metric_fields):
            raise ValueError(
                ErrorMessages.METRIC_COLUMNS_LENGTH
            )

        # Create a copy of interval's data
        merged_interval = interval_to_use.data.copy()

        if metric_merge_method:
            for interval_metric_col, other_metric_col in zip(
                    interval_to_use.metric_fields, other_to_use.metric_fields
            ):
                # Overwrite with non-NaN values from `other`
                merged_interval[interval_metric_col] = (
                    other_to_use.data[other_metric_col]
                    if pd.notna(other_to_use.data[other_metric_col])
                    else merged_interval[interval_metric_col]
                )

        return merged_interval


# Utility Functions
# -----------------

def is_metric_col(col: StructField) -> bool:
    return isinstance(
        col.dataType,
        (
            ByteType,
            ShortType,
            IntegerType,
            LongType,
            FloatType,
            DoubleType,
            DecimalType,
        ),
    ) or isinstance(col.dataType, BooleanType)


def make_disjoint_wrap(
        start_field: str,
        end_field: str,
        series_fields: Sequence[str],
        metric_fields: Sequence[str],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Returns a Pandas UDF for resolving overlapping intervals into disjoint intervals."""

    def make_disjoint_inner(pdf: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a grouped Pandas DataFrame to resolve overlapping intervals into disjoint intervals.

        Args:
            pdf (pd.DataFrame): Pandas DataFrame grouped by Spark.

        Returns:
            pd.DataFrame: Disjoint intervals as a Pandas DataFrame.
        """
        # Handle empty input DataFrame explicitly
        if pdf.empty:
            return pdf

        # Ensure intervals are sorted by start and end timestamps
        sorted_intervals = pdf.sort_values(by=[start_field, end_field]).reset_index(drop=True)

        # Initialize an empty DataFrame to store disjoint intervals
        disjoint_intervals = pd.DataFrame(columns=pdf.columns)

        # Define a function to process each row and update disjoint intervals
        def process_row(row):
            nonlocal disjoint_intervals
            interval = Interval.create(row, start_field, end_field, series_fields, metric_fields)
            local_utils = IntervalsUtils(disjoint_intervals)
            local_utils.disjoint_set = disjoint_intervals
            disjoint_intervals = local_utils.add_as_disjoint(interval)

        # Apply the processing function to each row
        sorted_intervals.apply(process_row, axis=1)

        return disjoint_intervals

    return make_disjoint_inner


def infer_datetime_format(date_string: str) -> str:
    """
    Extracts the exact format from a sample datetime string.
    This preserves the exact format of the input string.
    """
    # Replace all date/time components with their format codes
    replacements = [
        # Order matters - replace longer patterns first
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}[-+]\d{4}', '%Y-%m-%dT%H:%M:%S.%f%z'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}', '%Y-%m-%dT%H:%M:%S.%f'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[-+]\d{4}', '%Y-%m-%dT%H:%M:%S%z'),
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '%Y-%m-%dT%H:%M:%S'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}', '%Y-%m-%d %H:%M:%S.%f'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '%Y-%m-%d %H:%M:%S'),
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', '%Y-%m-%d %H:%M'),  # Added this format
        (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),
        (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', '%m/%d/%Y %H:%M:%S'),
        (r'\d{2}/\d{2}/\d{4}', '%m/%d/%Y'),
    ]

    for pattern, repl in replacements:
        if pd.Timestamp(date_string).strftime(repl) == date_string:
            return repl

    # If no exact match found, parse it with pandas and get its format
    return '%Y-%m-%d %H:%M:%S'  # Default format if no match
