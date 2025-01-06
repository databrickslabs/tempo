from __future__ import annotations

from functools import cached_property
from typing import Optional, Iterable, Callable, Sequence

import pandas as pd
import pyspark.sql.functions as f
from pandas import Timestamp
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


class Interval:
    def __init__(
            self,
            data: pd.Series,
            start_ts: Optional[str] = None,
            end_ts: Optional[str] = None,
            series_ids: Optional[Sequence[str]] = None,
            metric_columns: Optional[Sequence[str]] = None,
    ) -> None:

        if data.empty:
            raise ValueError("Data must not be empty.")
        self.data = data
        # Validate that start_ts and end_ts are strings
        if not isinstance(data[start_ts], (str, Timestamp)) or not isinstance(data[end_ts], (str, Timestamp)):
            raise TypeError(
                "start_ts and end_ts values must both be timestamp formatted strings or pandas.Timestamp type."
            )

        self.start_ts = start_ts
        self.end_ts = end_ts

        if isinstance(series_ids, str):
            series_ids = series_ids.split(",")
            series_ids = [s.strip() for s in series_ids]
        elif isinstance(metric_columns, Sequence):
            series_ids = series_ids
        elif series_ids is None:
            series_ids = []
        else:
            raise ValueError(
                "series_ids must be a Sequence or comma seperated string"
                f" of column names, instead got {type(series_ids)}"
            )

        if series_ids is None:
            self.series_ids: Sequence[str] = []
        else:
            self.series_ids: Sequence[str] = series_ids

        if isinstance(metric_columns, str):
            metric_columns = metric_columns.split(",")
            metric_columns = [s.strip() for s in metric_columns]
        elif isinstance(metric_columns, Sequence):
            metric_columns = metric_columns
        elif metric_columns is None:
            metric_columns = []
        else:
            raise ValueError(
                "metric_columns must be a Sequence or comma seperated string"
                f" of column names, instead got {type(metric_columns)}"
            )

        self.metric_columns = metric_columns

        self.validate_metric_columns()

    @property
    def boundaries(self):
        return self.start_ts, self.end_ts

    def validate_metric_columns(self) -> None:
        if not all(isinstance(col, str) for col in self.metric_columns):
            raise ValueError("All metric columns must be of type str")


class OverlapResolver:
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

        # TODO: get tests to pass but I need to duplicate all methods to handle scnearios for interval and other
        #  instead of swapping intervals. ie (interval_starts_first & other_starts_first). swapping intervals will be confusing for the user
        #  and is actually making the code more complicated than it needs to be

        # Ensure intervals are ordered by start time
        if interval.data[interval.start_ts] <= other.data[other.start_ts]:
            self.interval = interval
            self.other = other
        else:
            self.interval = other
            self.other = interval

    def validate_intervals(self):

        if set(self.interval.data.index) != set(self.other.data.index):
            raise ValueError("Expected indices of interval elements to be equivalent.")

    def check_no_overlap(self) -> bool:
        """
        Return True if `interval` and `other` do not overlap.

        Notes:
            - Validates that both intervals have non-NaN start and end timestamps.
            - Intervals do not overlap if one ends before the other starts or starts after the other ends.
        """

        # Check for non-overlapping intervals
        return (
                self.interval.data[self.interval.end_ts] < self.other.data[self.other.start_ts]
                or self.interval.data[self.interval.start_ts] > self.other.data[self.other.end_ts]
        )

    def resolve_no_overlap(self) -> list[pd.Series]:
        return [self.interval.data, self.other.data]

    def check_interval_starts_first(self) -> bool:
        """
        Return True if `interval` starts before `other` starts.

        Notes:
            - Raises ValueError if any NaN values are present in the start timestamps.
        """

        interval_starts_first = self.interval.data[self.interval.start_ts] < self.other.data[self.other.start_ts]
        return interval_starts_first

    def check_other_starts_first(self) -> bool:
        """
        Return True if `interval` starts before `other` starts.

        Notes:
            - Raises ValueError if any NaN values are present in the start timestamps.
        """

        other_starts_first = self.other.data[self.other.start_ts] < self.interval.data[self.interval.start_ts]
        return other_starts_first

    def check_interval_ends_first(self) -> bool:
        """
        Return True if `interval` ends before `other` starts.

        Notes:
            - Raises ValueError if any NaN values are present in the start timestamps.
        """

        interval_ends_first = self.interval.data[self.interval.end_ts] < self.other.data[self.other.end_ts]
        return interval_ends_first

    def check_other_ends_first(self) -> bool:
        """
        Return True if `interval` ends before `other` starts.

        Notes:
            - Raises ValueError if any NaN values are present in the start timestamps.
        """

        other_ends_first = self.other.data[self.other.end_ts] < self.interval.data[self.interval.end_ts]
        return other_ends_first

    def check_interval_contained(self) -> bool:
        """
        Return True if `interval` is fully contained within `other`.

        Notes:
            - Validates that neither interval has NaN values in start or end timestamps.
            - Uses `does_interval_start_before` and `does_interval_end_before` to determine containment.
        """

        # Check containment
        other_starts_first = self.check_other_starts_first()
        interval_ends_first = self.check_interval_ends_first()
        interval_contained = other_starts_first and interval_ends_first
        return interval_contained

    def check_other_contained(self) -> bool:
        """
        Return True if `interval` is fully contained within `other`.

        Notes:
            - Validates that neither interval has NaN values in start or end timestamps.
            - Uses `does_interval_start_before` and `does_interval_end_before` to determine containment.
        """

        # Check containment
        interval_starts_first = self.check_interval_starts_first()
        other_ends_first = self.check_other_ends_first()
        other_contained = interval_starts_first and other_ends_first
        return other_contained

    def resolve_interval_contained(self) -> list[pd.Series]:
        resolved_intervals = []
        # 1)
        resolved_series = update_interval_boundary(
            interval=self.interval.data,
            boundary_key=self.interval.end_ts,
            update_value=self.other.data[self.other.start_ts],
        )

        resolved_intervals.append(resolved_series)

        resolved_series = merge_metric_columns_of_intervals(
            interval=self.other,
            other=self.interval,
            metric_merge_method=True,
        )

        resolved_intervals.append(resolved_series)

        # 2)
        resolved_series = update_interval_boundary(
            interval=self.interval.data,
            boundary_key=self.interval.start_ts,
            update_value=self.other.data[self.other.end_ts],
        )

        resolved_intervals.append(resolved_series)

        return resolved_intervals

    def resolve_other_contained(self) -> list[pd.Series]:
        resolved_intervals = []
        # 1)
        resolved_series = update_interval_boundary(
            interval=self.interval.data,
            boundary_key=self.interval.end_ts,
            update_value=self.other.data[self.other.start_ts],
        )

        resolved_intervals.append(resolved_series)

        resolved_series = merge_metric_columns_of_intervals(
            interval=self.other,
            other=self.interval,
            metric_merge_method=True,
        )

        resolved_intervals.append(resolved_series)

        # 2)
        resolved_series = update_interval_boundary(
            interval=self.interval.data,
            boundary_key=self.interval.start_ts,
            update_value=self.other.data[self.other.end_ts],
        )

        resolved_intervals.append(resolved_series)

        return resolved_intervals

    def check_starts_equivalent(self) -> bool:
        """
        Return True if `interval` and `other` share the same start boundary.

        Notes:
            - Validates that both intervals have non-NaN start timestamps.
            - Compares start timestamps for exact equality.
        """

        # Check for shared start boundary
        return self.interval.data[self.interval.start_ts] == self.other.data[self.other.start_ts]

    def resolve_starts_equivalent(self):
        resolved_intervals = []

        if self.check_interval_ends_first():
            # 1)
            resolved_series = merge_metric_columns_of_intervals(
                interval=self.interval,
                other=self.other,
                metric_merge_method=True,
            )

            resolved_intervals.append(resolved_series)

            # 2)
            resolved_series = update_interval_boundary(
                interval=self.other.data,
                boundary_key=self.other.start_ts,
                update_value=self.interval.data[self.interval.end_ts],
            )

            resolved_intervals.append(resolved_series)

        else:
            # 1)
            resolved_series = merge_metric_columns_of_intervals(
                interval=self.other,
                other=self.interval,
                metric_merge_method=True,
            )

            resolved_intervals.append(resolved_series)

            # 2)
            resolved_series = update_interval_boundary(
                interval=self.interval.data,
                boundary_key=self.interval.start_ts,
                update_value=self.other.data[self.other.end_ts],
            )

            resolved_intervals.append(resolved_series)

        return resolved_intervals

    def check_ends_equivalent(self) -> bool:
        """
        Return True if `interval` and `other` share the same end boundary.

        Notes:
            - Validates that both intervals have non-NaN end timestamps.
            - Compares end timestamps for exact equality.
        """

        # Check for shared end boundary
        return self.interval.data[self.interval.end_ts] == self.other.data[self.other.end_ts]

    def resolve_ends_equivalent(self):
        resolved_intervals = []

        if self.check_interval_starts_first():
            # 1)
            resolved_series = update_interval_boundary(
                interval=self.interval.data,
                boundary_key=self.interval.end_ts,
                update_value=self.other.data[self.other.start_ts],
            )

            resolved_intervals.append(resolved_series)

            # 2)
            resolved_series = merge_metric_columns_of_intervals(
                interval=self.other,
                other=self.interval,
                metric_merge_method=True,
            )

            resolved_intervals.append(resolved_series)

        return resolved_intervals

    def check_boundaries_equivalent(self) -> bool:
        """
        Return True if `interval` is equivalent to `other`.

        Notes:
            - Validates that neither interval has NaN values in start or end timestamps.
            - Compares both the start and end boundaries for equality.
        """

        return self.check_starts_equivalent() and self.check_ends_equivalent()

    def resolve_boundaries_equivalent(self):
        resolved_intervals = []

        resolved_series = merge_metric_columns_of_intervals(
            interval=self.interval,
            other=self.other,
            metric_merge_method=True,
        )

        resolved_intervals.append(resolved_series)

        return resolved_intervals

    def check_metrics_equivalent(self) -> bool:
        """
        Return True if `interval` and `other` have identical metric columns.

        Notes:
            - Handles missing values explicitly by filling with `pd.NA`.
            - Assumes `interval.metric_columns` and `other.metric_columns` are aligned.
        """
        # Fill missing values with `pd.NA` for consistent handling of nulls
        interval_data_filled = self.interval.data[self.interval.metric_columns].fillna(pd.NA)
        other_data_filled = self.other.data[self.other.metric_columns].fillna(pd.NA)

        # Compare metric columns for equality
        return interval_data_filled.equals(other_data_filled)

    def resolve_metrics_equivalent(self) -> pd.Series:
        return update_interval_boundary(
            interval=self.interval.data,
            boundary_key=self.interval.end_ts,
            update_value=self.other.data[self.other.end_ts],
        )

    def resolve_interval_partial_overlap(self):
        resolved_intervals = []
        # 1)
        resolved_series = update_interval_boundary(
            interval=self.interval.data,
            boundary_key=self.interval.end_ts,
            update_value=self.other.data[self.other.start_ts],
        )

        resolved_intervals.append(resolved_series)

        # 2)
        updated_series = update_interval_boundary(
            interval=self.other.data,
            boundary_key=self.other.end_ts,
            update_value=self.interval.data[self.interval.end_ts],
        )
        resolved_series = merge_metric_columns_of_intervals(
            interval=Interval(
                updated_series,
                self.other.start_ts,
                self.other.end_ts,
                metric_columns=self.other.metric_columns,
            ),
            other=self.interval,
            metric_merge_method=True,
        )

        resolved_intervals.append(resolved_series)

        # 3)
        resolved_series = update_interval_boundary(
            interval=self.other.data,
            boundary_key=self.other.start_ts,
            update_value=self.interval.data[self.interval.end_ts],
        )

        resolved_intervals.append(resolved_series)

        return resolved_intervals

    def resolve_overlap(  # TODO: need to implement proper metric merging
            #  -> for now, can just take non-null values from both intervals
            self
    ) -> list[pd.Series]:
        """
        resolve overlaps between the two given intervals,
        splitting them as necessary into some set of disjoint intervals
        """

        self.validate_intervals()

        resolved_intervals = list()

        # intervals_do_not_overlap(interval, other, ...) is True
        #
        # Results in 2 disjoint intervals
        # 1) A.start, A.end, A.metric_columns
        # 2) B.start, B.end, B.metric_columns

        if self.check_no_overlap():
            return self.resolve_no_overlap()

        # intervals_have_equivalent_metric_columns(interval, other, metric_columns) is True
        #
        # Results in 1 disjoint interval
        # 1) A.start, B.end, A.metric_columns

        if self.check_metrics_equivalent():
            resolved_intervals.append(
                self.resolve_metrics_equivalent()
            )

            return resolved_intervals

        # interval_is_contained_by(interval=other, other=interval, ...) is True
        #
        # Results in 3 disjoint intervals
        # 1) A.start, B.start, A.metric_columns
        # 2) B.start, B.end, merge(A.metric_columns, B.metric_columns)
        # 3) B.end, A.end, A.metric_columns

        # TODO: missing this condition because other is contained in interval
        if self.check_interval_contained():
            resolved_intervals.extend(self.resolve_interval_contained())

            return resolved_intervals

        if self.check_other_contained():
            resolved_intervals.extend(self.resolve_other_contained())

            return resolved_intervals

        # A shares a common start with B, a different end boundary
        # - A.start = B.start & A.end != B.end
        #
        # Results in 2 disjoint intervals
        # - if A.end < B.end
        #     1) A.start, A.end, merge(A.metric_columns, B.metric_columns)
        #     2) A.end, B.end, B.metric_columns
        # - if A.end > B.end
        #     1) B.start, B.end, merge(A.metric_columns, B.metric_columns)
        #     2) B.end, A.end, A.metric_columns

        if self.check_starts_equivalent() and not self.check_ends_equivalent():
            resolved_intervals.extend(
                self.resolve_starts_equivalent()
            )

            return resolved_intervals

        # A shares a common end with B, a different start boundary
        # - A.start != B.start & A.end = B.end
        #
        # Results in 2 disjoint intervals
        # - if A.start < B.start
        #     1) A.start, B.start, A.metric_columns
        #     1) B.start, B.end, merge(A.metric_columns, B.metric_columns)
        # - if A.start > B.start
        #     1) B.start, A.end, B.metric_columns
        #     2) A.start, A.end, merge(A.metric_columns, B.metric_columns)

        if not self.check_starts_equivalent(
        ) and self.check_ends_equivalent(
        ):
            resolved_intervals.extend(
                self.resolve_ends_equivalent()
            )

            return resolved_intervals

        # A and B share a common start and end boundary, making them equivalent.
        # - A.start = B.start & A.end = B.end
        #
        # Results in 1 disjoint interval
        # 1) A.start, A.end, merge(A.metric_columns, B.metric_columns)

        if self.check_boundaries_equivalent():
            resolved_intervals.extend(
                self.resolve_boundaries_equivalent()
            )

            return resolved_intervals

        # Interval A starts first and A partially overlaps B
        # - A.start < B.start & A.end < B.end
        #
        # Results in 3 disjoint intervals
        # 1) A.start, B.start, A.metric_columns
        # 2) B.start, A.end, merge(A.metric_columns, B.metric_columns)
        # 3) A.end, B.end, B.metric_columns

        if self.check_interval_starts_first(
        ) and self.check_interval_ends_first(
        ):
            resolved_intervals.extend(self.resolve_interval_partial_overlap())

            return resolved_intervals

        raise NotImplementedError("Interval resolution not implemented")


def identify_interval_overlaps(
        overlapping_intervals: pd.DataFrame,
        interval: Interval,
) -> pd.DataFrame:
    """
    Identify overlapping intervals in a Pandas DataFrame.

    This function filters the input DataFrame (`in_pdf`) to return rows representing
    intervals that overlap with a specified `interval`. The overlap is determined based on
    the start and end timestamps of each interval. The input interval itself is excluded
    from the results.

    Parameters:
        overlapping_intervals (pd.DataFrame): Input DataFrame containing interval data, with start and end timestamps.
        interval (Interval): An object representing the reference interval, containing the following:
            - data (pd.Series): A Series representing the interval's attributes.
            - start_ts (str): Column name for the start timestamp.
            - end_ts (str): Column name for the end timestamp.

    Returns:
        pd.DataFrame: A subset of the input DataFrame, containing only rows with overlapping intervals.
    """

    max_start = "_MAX_START_TS"
    max_end = "_MAX_END_TS"

    # Check if input DataFrame or interval data is empty; return an empty DataFrame if true.
    if overlapping_intervals.empty or interval.data.empty:
        # return in_pdf
        return pd.DataFrame()

    overlapping_intervals_copy = overlapping_intervals.copy()

    # Calculate the latest possible start timestamp for overlap comparison.
    overlapping_intervals_copy[max_start] = overlapping_intervals_copy[
        interval.start_ts
    ].where(
        overlapping_intervals_copy[interval.start_ts]
        >= interval.data[interval.start_ts],
        interval.data[interval.start_ts],
    )

    # Calculate the earliest possible end timestamp for overlap comparison.
    overlapping_intervals_copy[max_end] = overlapping_intervals_copy[
        interval.end_ts
    ].where(
        overlapping_intervals_copy[interval.end_ts] <= interval.data[interval.end_ts],
        interval.data[interval.end_ts],
    )

    # https://www.baeldung.com/cs/finding-all-overlapping-intervals
    overlapping_intervals_copy = overlapping_intervals_copy[
        overlapping_intervals_copy[max_start] < overlapping_intervals_copy[max_end]
        ]

    # Remove intermediate columns used for interval overlap calculation.
    cols_to_drop = [max_start, max_end]
    overlapping_intervals_copy = overlapping_intervals_copy.drop(columns=cols_to_drop)

    # Remove rows that are identical to `interval.data`
    remove_with_row_mask = ~(
            overlapping_intervals_copy.isna().eq(interval.data.isna())
            & overlapping_intervals_copy.eq(interval.data).fillna(False)
    ).all(axis=1)

    overlapping_intervals_copy = overlapping_intervals_copy[remove_with_row_mask]

    return overlapping_intervals_copy


def update_interval_boundary(
        *,
        interval: pd.Series,
        boundary_key: str,
        update_value: str,
) -> pd.Series:
    """
    Return a new copy of `interval` with the specified boundary updated.

    Parameters:
        interval (pd.Series): A Pandas Series representing the interval data.
        boundary_key (str): The key of the boundary to update (e.g., "start" or "end").
        update_value (str): The new value for the specified boundary.

    Returns:
        pd.Series: A copy of the interval with the updated boundary.

    Raises:
        KeyError: If `boundary_key` does not exist in the interval.
        TypeError: If `interval` is not a Pandas Series.
    """
    if not isinstance(interval, pd.Series):
        raise TypeError("Expected `interval` to be a Pandas Series.")

    if boundary_key not in interval.keys():
        raise KeyError(
            f"boundary_key '{boundary_key}' must exist in {list(interval.keys())}"
        )

    updated_interval = interval.copy()
    updated_interval[boundary_key] = update_value

    return updated_interval


def merge_metric_columns_of_intervals(
        interval: Interval,
        other: Interval,
        metric_merge_method: bool = False,
) -> pd.Series:
    """
    Return a merged set of metrics from `interval` and `other`.

    Parameters:
        interval (Interval): The primary interval whose metrics are to be updated.
        other (Interval): The secondary interval providing metrics to merge.
        metric_merge_method (bool): If True, non-NaN metrics from `other` will overwrite metrics in `interval`.

    Returns:
        pd.Series: A merged set of metrics as a Pandas Series.

    Raises:
        ValueError: If `interval.metric_columns` and `other.metric_columns` are not aligned.
    """
    # Validate that metric columns align
    if len(interval.metric_columns) != len(other.metric_columns):
        raise ValueError(
            "Metric columns of `interval` and `other` must have the same length."
        )

    # Create a copy of interval's data
    merged_interval = interval.data.copy()

    if metric_merge_method:
        for interval_metric_col, other_metric_col in zip(
                interval.metric_columns, other.metric_columns
        ):
            # Overwrite with non-NaN values from `other`
            merged_interval[interval_metric_col] = (
                other.data[other_metric_col]
                if pd.notna(other.data[other_metric_col])
                else merged_interval[interval_metric_col]
            )

    return merged_interval

def resolve_all_overlaps(
        interval: Interval,
        overlaps: pd.DataFrame,
) -> pd.DataFrame:
    """
    resolve the interval `x` against all overlapping intervals in `overlapping`,
    returning a set of disjoint intervals with the same spans
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
    """

    first_row = Interval(
        overlaps.iloc[0],
            interval.start_ts,
            interval.end_ts,
            interval.series_ids,
            interval.metric_columns,
    )
    resolver = OverlapResolver(interval, first_row)
    initial_intervals = resolver.resolve_overlap()
    local_disjoint_df = pd.DataFrame(initial_intervals)

    def resolve_and_add(row):
        row_interval = Interval(
                row,
                interval.start_ts,
                interval.end_ts,
                interval.series_ids,
                interval.metric_columns,
        )
        resolver = OverlapResolver(interval, row_interval)
        resolved_intervals = resolver.resolve_overlap()
        for interval_data in resolved_intervals:
            interval_inner = Interval(
                interval_data,
                interval.start_ts,
                interval.end_ts,
                interval.series_ids,
                interval.metric_columns,
            )
            nonlocal local_disjoint_df
            local_disjoint_df = add_as_disjoint(interval_inner, local_disjoint_df)

    overlaps.iloc[1:].apply(resolve_and_add, axis=1)

    return local_disjoint_df


def add_as_disjoint(interval: Interval, disjoint_set: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    returns a disjoint set consisting of the given interval, made disjoint with those already in `disjoint_set`
    """

    if disjoint_set is None or disjoint_set.empty:
        return pd.DataFrame([interval.data])

    overlapping_subset_df = identify_interval_overlaps(
        overlapping_intervals=disjoint_set, interval=interval
    )

    # if there are no overlaps, add the interval to disjoint_set
    if overlapping_subset_df.empty:
        element_wise_comparison = disjoint_set.fillna("¯\\_(ツ)_/¯") == interval.data.fillna("¯\\_(ツ)_/¯").values

        row_wise_comparison = element_wise_comparison.all(axis=1)
        # NB: because of the nested iterations, we need to check that the
        # record hasn't already been added to `global_disjoint_df` by another loop
        if row_wise_comparison.any():
            return disjoint_set
        else:
            return pd.concat((disjoint_set, pd.DataFrame([interval.data])))

    # identify all intervals which do not overlap with the given interval to
    # concatenate them to the disjoint set after resolving overlaps
    non_overlapping_subset_df = disjoint_set[
        ~disjoint_set.set_index(*interval.boundaries).index.isin(
            overlapping_subset_df.set_index(*interval.boundaries).index
        )
    ]

    # Avoid a call to `resolve_all_overlaps` if there is only one to resolve
    multiple_to_resolve = len(overlapping_subset_df.index) > 1

    # If every record overlaps, no need to handle non-overlaps
    only_overlaps_present = len(disjoint_set.index) == len(overlapping_subset_df.index)

    # Resolve the interval against all the existing, overlapping intervals
    # `multiple_to_resolve` is used to avoid unnecessary calls to `resolve_all_overlaps`
    # `only_overlaps_present` is used to avoid unnecessary calls to `pd.concat`
    if not multiple_to_resolve and only_overlaps_present:
        resolver = OverlapResolver(
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
        return resolve_all_overlaps(
            interval,
            overlapping_subset_df,
        )

    if not multiple_to_resolve and not only_overlaps_present:
        resolver = OverlapResolver(
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
                resolve_all_overlaps(
                    interval,
                    overlapping_subset_df,
                ),
                non_overlapping_subset_df,
            ),
        )

    # if we get here, something went wrong
    raise NotImplementedError("Interval resolution not implemented")

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
            disjoint_intervals = add_as_disjoint(interval, disjoint_intervals)

        # Apply the processing function to each row
        sorted_intervals.apply(process_row, axis=1)

        return disjoint_intervals

    return make_disjoint_inner
