from __future__ import annotations

from abc import abstractmethod, ABC
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Optional, Iterable, Callable, Sequence, Dict, Type

import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    # NB: NumericType is a non-public object, so we shouldn't import it directly
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


class IntervalsDF:
    """
    This object is the main wrapper over a `Spark DataFrame`_ which allows a
    user to parallelize computations over snapshots of metrics for intervals
    of time defined by a start and end timestamp and various dimensions.

    The required dimensions are `series` (list of columns by which to
    summarize), `metrics` (list of columns to analyze), `start_ts` (timestamp
    column), and `end_ts` (timestamp column). `start_ts` and `end_ts` can be
    epoch or TimestampType.

    .. _`Spark DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html

    """

    def __init__(
            self,
            df: DataFrame,
            start_ts: str,
            end_ts: str,
            series_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """
         Constructor for :class:`IntervalsDF`.

        :param df:
        :type df: `DataFrame`_
        :param start_ts:
        :type start_ts: str
        :param end_ts:
        :type end_ts: str
        :param series_ids:
        :type series_ids: list[str]
        :rtype: None

        :Example:

        .. code-block:

            df = spark.createDataFrame(
                [["2020-08-01 00:00:09", "2020-08-01 00:00:14", "v1", 5, 0]],
                "start_ts STRING, end_ts STRING, series_1 STRING, metric_1 INT, metric_2 INT",
            )
            idf = IntervalsDF(df, "start_ts", "end_ts", ["series_1"])
            idf.df.collect()
            [Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:14', series_1='v1', metric_1=5, metric_2=0)]

        .. _`DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html

        .. todo::
            - create IntervalsSchema class to validate data types and column
                existence
            - check elements of series and identifiers to ensure all are str
            - check if start_ts, end_ts, and the elements of series and
                identifiers can be of type col

        """

        self.df = df

        self.start_ts = start_ts
        self.end_ts = end_ts

        if not series_ids:
            self.series_ids = []
        elif isinstance(series_ids, str):
            series_ids = series_ids.split(",")
            self.series_ids = [s.strip() for s in series_ids]
        elif isinstance(series_ids, Iterable):
            self.series_ids = list(series_ids)
        else:
            raise ValueError(
                f"series_ids must be an Iterable or comma seperated string"
                f" of column names, instead got {type(series_ids)}"
            )

        # self.make_disjoint = MakeDisjointBuilder()

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

        .. todo::
            - check elements of identifiers to ensure all are str
            - check if start_ts, end_ts, and the elements of series and
                identifiers can be of type col

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
        are correlated and merged prior to constructing new time interval boundaries (
        start and end timestamp) so that all intervals are disjoint.

        The merge process assumes that two overlapping intervals cannot simultaneously
        report two different values for the same metric unless recorded in a data type
        which supports multiple elements (such as ArrayType, etc.).

        This is often used after :meth:`fromStackedMetrics` to reduce the number of
        metrics with `null` values and helps when constructing filter predicates to
        retrieve specific metric values across all instances.

        :return: A new :class:`IntervalsDF` containing disjoint time intervals

        :Example:

        .. code-block::

            df = spark.createDataFrame(
                [["2020-08-01 00:00:10", "2020-08-01 00:00:14", "v1", 5, null],
                 ["2020-08-01 00:00:09", "2020-08-01 00:00:11", "v1", null, 0]],
                "start_ts STRING, end_ts STRING, series_1 STRING, metric_1 STRING, metric_2 INT",
            )
            idf = IntervalsDF(df, "start_ts", "end_ts", ["series_1"], ["metric_1", "metric_2"])
            idf.disjoint().df.collect()
            [Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:10', series_1='v1', metric_1=null, metric_2=0),
             Row(start_ts='2020-08-01 00:00:10', end_ts='2020-08-01 00:00:11', series_1='v1', metric_1=5, metric_2=0),
             Row(start_ts='2020-08-01 00:00:11', end_ts='2020-08-01 00:00:14', series_1='v1', metric_1=5, metric_2=null)]

        """
        # NB: creating local copies of class and instance attributes to be
        # referenced by UDF because complex python objects, like classes,
        # is not possible with PyArrow's supported data types
        # https://arrow.apache.org/docs/python/api/datatypes.html
        local_start_ts = self.start_ts
        local_end_ts = self.end_ts
        local_series_ids = self.series_ids

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
            local_start_ts,
            local_end_ts,
            local_series_ids,
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


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None


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
    def validate_timestamps(data: pd.Series, start_ts: str, end_ts: str) -> ValidationResult:
        if not isinstance(data[start_ts], (str, pd.Timestamp)):
            raise InvalidTimestampError(
                f"start_ts must be string or Timestamp, got {type(data[start_ts])}"
            )

        if not isinstance(data[end_ts], (str, pd.Timestamp)):
            raise InvalidTimestampError(
                f"end_ts must be string or Timestamp, got {type(data[end_ts])}"
            )

        if data[start_ts] > data[end_ts]:
            raise InvalidTimestampError(
                f"start_ts ({data[start_ts]}) must be <= end_ts ({data[end_ts]})"
            )

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
class OverlapResult:
    type: OverlapType
    details: Optional[Dict] = None


class OverlapDetector:
    def __init__(
            self,
            interval: Interval,
            other: Interval,
    ) -> None:
        self.interval = interval
        self.other = other
        self._checkers = self._init_checkers()

    def detect_overlap_type(self) -> Optional[OverlapResult]:
        """Returns the most specific type of overlap found"""
        for overlap_type, checker in self._checkers.items():
            if checker.check(self.interval, self.other):
                return OverlapResult(overlap_type)
        return None

    def _init_checkers(self) -> Dict[OverlapType, OverlapChecker]:
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




class Interval:
    def __init__(
            self,
            data: pd.Series,
            start_ts: Optional[str] = None,
            end_ts: Optional[str] = None,
            series_ids: Optional[Sequence[str]] = None,
            metric_columns: Optional[Sequence[str]] = None,
    ):

        self.validator = IntervalValidator()

        # Validate all components
        self.validator.validate_data(data)
        self.validator.validate_timestamps(data, start_ts, end_ts)
        self.validator.validate_series_id_columns(series_ids)
        self.validator.validate_metric_columns(metric_columns)

        self.data = data
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.series_ids = series_ids
        self.metric_columns = metric_columns

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

    def validate_series_alignment(self, other: 'Interval') -> ValidationResult:
        return self._validate_column_alignment(
            self.series_ids,
            other.series_ids,
            "Series IDs"
        )

    def validate_metrics_alignment(self, other: 'Interval') -> ValidationResult:
        return self._validate_column_alignment(
            self.metric_columns,
            other.metric_columns,
            "Metric columns"
        )

    @property
    def boundaries(self):
        return self.start_ts, self.end_ts

    def update_start_boundary(self, update_value: str) -> 'Interval':
        updated_data = self.data.copy()
        updated_data[self.start_ts] = update_value
        return Interval(updated_data, self.start_ts, self.end_ts, self.metric_columns)

    def update_end_boundary(self, update_value: str) -> 'Interval':
        updated_data = self.data.copy()
        updated_data[self.end_ts] = update_value
        return Interval(updated_data, self.start_ts, self.end_ts, self.metric_columns)


class MetricNormalizer(ABC):
    @abstractmethod
    def normalize(self, interval: Interval) -> pd.Series:
        pass


class NullMetricNormalizer(MetricNormalizer):
    def normalize(self, interval: Interval) -> pd.Series:
        return interval.data[interval.metric_columns].fillna(pd.NA)


class OverlapChecker(ABC):
    """Abstract base class for overlap checking strategies"""

    @abstractmethod
    def check(self, interval: Interval, other: Interval) -> bool:
        pass


class NoOverlapChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return (
                interval.data[interval.end_ts] < other.data[other.start_ts]
                or interval.data[interval.start_ts] > other.data[other.end_ts]
        )


class MetricsEquivalentChecker(OverlapChecker):
    def __init__(self, normalizer: MetricNormalizer = NullMetricNormalizer()):
        self.normalizer = normalizer

    def check(self, interval: Interval, other: Interval) -> bool:
        interval_metrics = self.normalizer.normalize(interval)
        other_metrics = self.normalizer.normalize(other)
        return interval_metrics.equals(other_metrics)


class IntervalStartsFirstChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.data[interval.start_ts] < other.data[other.start_ts]


class OtherStartsFirstChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return other.data[other.start_ts] < interval.data[interval.start_ts]


class IntervalEndsFirstChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.data[interval.end_ts] < other.data[other.end_ts]


class OtherEndsFirstChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return other.data[other.end_ts] < interval.data[interval.end_ts]


class IntervalContainedChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return OtherStartsFirstChecker().check(interval, other) and IntervalEndsFirstChecker().check(interval, other)


class OtherContainedChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return IntervalStartsFirstChecker().check(interval, other) and OtherEndsFirstChecker().check(interval, other)


class StartBoundaryEqualityChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.data[interval.start_ts] == other.data[other.start_ts]


class EndBoundaryEqualityChecker(OverlapChecker):
    def check(self, interval: Interval, other: Interval) -> bool:
        return interval.data[interval.end_ts] == other.data[other.end_ts]


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

class OverlapResolver(ABC):
    @abstractmethod
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        pass


class NoOverlapResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        return [interval.data, other.data]


class EquivalentMetricsResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        return [interval.update_end_boundary(other.data[other.end_ts]).data]


class ContainedResolverBase(OverlapResolver):
    """Base class for resolving contained intervals"""

    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        resolved = [
            # First interval - update end boundary
            interval.update_end_boundary(other.data[other.start_ts]).data,
            # Middle interval - merge metrics
            self._merge_metrics(interval, other),
            # Last interval - update start boundary
            interval.update_start_boundary(other.data[other.end_ts]).data]

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
            resolved_series = other.update_start_boundary(interval.data[interval.end_ts]).data

            resolved.append(resolved_series)

        else:
            # 1)
            resolved_series = NonNullOverwriteMetricMerger().merge(other, interval)

            resolved.append(resolved_series)

            # 2)
            resolved_series = interval.update_start_boundary(other.data[other.end_ts]).data

            resolved.append(resolved_series)

        return resolved


class CommonEndResolver(OverlapResolver):
    def resolve(self, interval: Interval, other: Interval) -> list[pd.Series]:
        resolved = []

        if IntervalStartsFirstChecker().check(interval, other):
            # 1)
            resolved_series = interval.update_end_boundary(other.data[other.start_ts]).data

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
        resolved_series = interval.update_end_boundary(other.data[other.start_ts]).data

        resolved.append(resolved_series)

        # 2)
        updated_series = other.update_end_boundary(interval.data[interval.end_ts]).data
        resolved_series = NonNullOverwriteMetricMerger().merge(
            Interval(
                updated_series,
                other.start_ts,
                other.end_ts,
                metric_columns=other.metric_columns,
            ),
            interval,
        )

        resolved.append(resolved_series)

        # 3)
        resolved_series = other.update_start_boundary(interval.data[interval.end_ts]).data

        resolved.append(resolved_series)

        return resolved


class MetricMerger(ABC):
    """Abstract base class defining metric merging strategy"""

    @abstractmethod
    def merge(self, interval: Interval, other: Interval) -> pd.Series:
        """Merge metrics from two intervals according to strategy"""
        pass

    @staticmethod
    def _validate_metric_columns(interval: Interval, other: Interval) -> None:
        """Validate that metric columns are aligned between intervals"""
        if len(interval.metric_columns) != len(other.metric_columns):
            raise ValueError("Metric columns must have the same length.")


class NonNullOverwriteMetricMerger(MetricMerger):
    """Implementation that overwrites with non-null values from other interval"""

    def merge(self, interval: Interval, other: Interval) -> pd.Series:
        self._validate_metric_columns(interval, other)

        # Create a copy of interval's data
        merged_data = interval.data.copy()

        # Overwrite with non-NaN values from other
        for interval_col, other_col in zip(interval.metric_columns, other.metric_columns):
            merged_data[interval_col] = (
                other.data[other_col]
                if pd.notna(other.data[other_col])
                else merged_data[interval_col]
            )

        return merged_data


# Here are the main interval relationship types in interval algebra:
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
        if interval.data[interval.start_ts] <= other.data[other.start_ts]:
            self.interval = interval
            self.other = other
        else:
            self.interval = other
            self.other = interval

        self.overlap_detector = OverlapDetector(self.interval, self.other)

        self.resolvers = {

            OverlapType.BOUNDARY_EQUAL: BoundaryEqualityResolver(),
            OverlapType.COMMON_END: CommonEndResolver(),
            OverlapType.COMMON_START: CommonStartResolver(),
            OverlapType.INTERVAL_CONTAINED: IntervalContainedResolver(),
            OverlapType.METRICS_EQUIVALENT: EquivalentMetricsResolver(),
            OverlapType.NO_OVERLAP: NoOverlapResolver(),
            OverlapType.OTHER_CONTAINED: OtherContainedResolver(),
            OverlapType.PARTIAL_OVERLAP: PartialOverlapResolver(),


        }

    def validate_intervals(self):

        if set(self.interval.data.index) != set(self.other.data.index):
            raise ValueError("Expected indices of interval elements to be equivalent.")

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

        resolver = self.resolvers.get(overlap_result.type)
        if resolver is None:
            raise NotImplementedError(f"No resolver for overlap type {overlap_result.type}")

        return resolver.resolve(self.interval, self.other)

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
        """
        # Determine which intervals to use
        interval_to_use = new_interval if new_interval else self.interval
        other_to_use = new_other if new_other else self.other

        # Validate that metric columns align
        if len(interval_to_use.metric_columns) != len(other_to_use.metric_columns):
            raise ValueError(
                "Metric columns must have the same length."
            )

        # Create a copy of interval's data
        merged_interval = interval_to_use.data.copy()

        if metric_merge_method:
            for interval_metric_col, other_metric_col in zip(
                    interval_to_use.metric_columns, other_to_use.metric_columns
            ):
                # Overwrite with non-NaN values from `other`
                merged_interval[interval_metric_col] = (
                    other_to_use.data[other_metric_col]
                    if pd.notna(other_to_use.data[other_metric_col])
                    else merged_interval[interval_metric_col]
                )

        return merged_interval


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
            interval.start_ts
        ].where(
            intervals_copy[interval.start_ts]
            >= interval.data[interval.start_ts],
            interval.data[interval.start_ts],
        )

        # Calculate the earliest possible end timestamp for overlap comparison.
        intervals_copy[max_end] = intervals_copy[
            interval.end_ts
        ].where(
            intervals_copy[interval.end_ts] <= interval.data[interval.end_ts],
            interval.data[interval.end_ts],
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
            element_wise_comparison = self.disjoint_set.copy().fillna("¯\\_(ツ)_/¯") == interval.data.fillna(
                "¯\\_(ツ)_/¯").values

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
            ~self.disjoint_set.set_index(*interval.boundaries).index.isin(
                overlapping_subset_df.set_index(*interval.boundaries).index
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
                interval=Interval(
                    interval.data,
                    interval.start_ts,
                    interval.end_ts,
                    interval.series_ids,
                    interval.metric_columns,
                ),
                other=Interval(
                    overlapping_subset_df.iloc[0],
                    interval.start_ts,
                    interval.end_ts,
                    interval.series_ids,
                    interval.metric_columns,
                ),
            )
            return pd.DataFrame(
                resolver.resolve_overlap()
            )

        if multiple_to_resolve and only_overlaps_present:
            return IntervalsUtils(overlapping_subset_df).resolve_all_overlaps(interval)

        if not multiple_to_resolve and not only_overlaps_present:
            resolver = IntervalTransformer(
                interval=Interval(
                    interval.data,
                    interval.start_ts,
                    interval.end_ts,
                    interval.series_ids,
                    interval.metric_columns,
                ),
                other=Interval(
                    overlapping_subset_df.iloc[0],
                    interval.start_ts,
                    interval.end_ts,
                    interval.series_ids,
                    interval.metric_columns,
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

        first_row = Interval(
            self.intervals.iloc[0],
            interval.start_ts,
            interval.end_ts,
            interval.series_ids,
            interval.metric_columns,
        )
        resolver = IntervalTransformer(interval, first_row)
        initial_intervals = resolver.resolve_overlap()
        disjoint_intervals = pd.DataFrame(initial_intervals)

        def resolve_and_add(row):
            row_interval = Interval(
                row,
                interval.start_ts,
                interval.end_ts,
                interval.series_ids,
                interval.metric_columns,
            )
            local_resolver = IntervalTransformer(interval, row_interval)
            resolved_intervals = local_resolver.resolve_overlap()
            for interval_data in resolved_intervals:
                interval_inner = Interval(
                    interval_data,
                    interval.start_ts,
                    interval.end_ts,
                    interval.series_ids,
                    interval.metric_columns,
                )
                nonlocal disjoint_intervals
                local_interval_utils = IntervalsUtils(disjoint_intervals)
                local_interval_utils.disjoint_set = disjoint_intervals
                disjoint_intervals = local_interval_utils.add_as_disjoint(interval_inner)

        self.intervals.iloc[1:].apply(resolve_and_add, axis=1)

        return disjoint_intervals


def make_disjoint_wrap(
        start_ts: str,
        end_ts: str,
        series_ids: Sequence[str],
        metric_columns: Sequence[str],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
     Creates a Pandas UDF to process overlapping intervals and resolve them into disjoint intervals.

     Parameters:
         start_ts (str): Column name for the start timestamp.
         end_ts (str): Column name for the end timestamp.
         series_ids (Sequence[str]): Columns identifying unique series.
         metric_columns (Sequence[str]): Metric columns to process.

     Returns:
         Callable[[pd.DataFrame], pd.DataFrame]: Function to apply on grouped Pandas DataFrames in Spark.
     """

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
        sorted_intervals = pdf.sort_values(by=[start_ts, end_ts]).reset_index(drop=True)

        # Initialize an empty DataFrame to store disjoint intervals
        disjoint_intervals = pd.DataFrame(columns=pdf.columns)

        # Define a function to process each row and update disjoint intervals
        def process_row(row):
            nonlocal disjoint_intervals
            interval = Interval(row, start_ts, end_ts, series_ids, metric_columns)
            local_utils = IntervalsUtils(disjoint_intervals)
            local_utils.disjoint_set = disjoint_intervals
            disjoint_intervals = local_utils.add_as_disjoint(interval)

        # Apply the processing function to each row
        sorted_intervals.apply(process_row, axis=1)

        return disjoint_intervals

    return make_disjoint_inner
