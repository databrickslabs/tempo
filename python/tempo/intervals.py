from __future__ import annotations

from itertools import islice
from typing import Optional, Iterable, cast, Any, Callable
from functools import cached_property

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
import pyspark.sql.functions as f
from pyspark.sql.window import Window, WindowSpec

import numpy as np
import pandas as pd


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

        if stack:
            n_cols = len(self.metric_columns)
            metric_cols_expr = ",".join(
                tuple(f"'{col}', {col}" for col in self.metric_columns)
            )

            stack_expr = (
                f"STACK({n_cols}, {metric_cols_expr}) AS (metric_name, metric_value)"
            )

            return self.df.select(
                *self.interval_boundaries,
                *self.series_ids,
                f.expr(stack_expr),
            ).dropna(subset="metric_value")

        else:
            return self.df


def identify_interval_overlaps(
    in_pdf: pd.DataFrame,
    with_row: pd.Series,
    interval_start_ts: str,
    interval_end_ts: str,
) -> pd.DataFrame:
    """
    return the subset of rows in DataFrame `in_pdf` that overlap with row `with_row`
    """

    if in_pdf.empty or with_row.empty:
        # return in_pdf
        return pd.DataFrame()

    local_in_pdf = in_pdf.copy()

    # https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column
    local_in_pdf["max_start_timestamp"] = [
        _ if _ >= with_row[interval_start_ts] else with_row[interval_start_ts]
        for _ in local_in_pdf[interval_start_ts]
    ]

    local_in_pdf["min_end_timestamp"] = [
        _ if _ <= with_row[interval_end_ts] else with_row[interval_end_ts]
        for _ in local_in_pdf[interval_end_ts]
    ]

    # https://www.baeldung.com/cs/finding-all-overlapping-intervals
    local_in_pdf = local_in_pdf[
        local_in_pdf["max_start_timestamp"] < local_in_pdf["min_end_timestamp"]
    ]

    local_in_pdf = local_in_pdf.drop(
        columns=["max_start_timestamp", "min_end_timestamp"]
    )

    # NB: with_row will always be included in the subset because with_row
    #     is identical to with_row. This step is to remove it from subset.
    remove_with_row_mask = ~(
        local_in_pdf.fillna("¯\\_(ツ)_/¯") == np.array(with_row.fillna("¯\\_(ツ)_/¯"))
    ).all(1)
    local_in_pdf = local_in_pdf[remove_with_row_mask]

    return local_in_pdf


def check_for_nan_values(to_check: Any) -> bool:
    """
    return True if there are any NaN values in `to_check`
    """
    if isinstance(to_check, pd.Series):
        return to_check.isna().any()
    elif isinstance(to_check, pd.DataFrame):
        return to_check.isna().any().any()
    elif isinstance(to_check, np.ndarray):
        return bool(np.isnan(to_check).any())
    elif isinstance(to_check, (np.generic, float)):
        return np.isnan(to_check)
    else:
        return to_check is None


def interval_starts_before(
    *,
    interval: pd.Series,
    other: pd.Series,
    interval_start_ts: str,
    other_start_ts: Optional[str] = None,
) -> bool:
    """
    return True if interval_a starts before interval_b starts
    """

    if other_start_ts is None:
        other_start_ts = interval_start_ts

    if check_for_nan_values(interval[interval_start_ts]) or check_for_nan_values(
        other[other_start_ts]
    ):
        raise ValueError("interval and other cannot contain NaN values for timestamps")

    return interval[interval_start_ts] < other[other_start_ts]


def interval_ends_before(
    *,
    interval: pd.Series,
    other: pd.Series,
    interval_end_ts: str,
    other_end_ts: Optional[str] = None,
) -> bool:
    """
    return True if interval_a ends before interval_b ends
    """

    if other_end_ts is None:
        other_end_ts = interval_end_ts

    if check_for_nan_values(interval[interval_end_ts]) or check_for_nan_values(
        other[other_end_ts]
    ):
        raise ValueError("interval and other cannot contain NaN values for timestamps")

    return interval[interval_end_ts] < other[other_end_ts]


def interval_is_contained_by(
    *,
    interval: pd.Series,
    other: pd.Series,
    interval_start_ts: str,
    interval_end_ts: str,
    other_start_ts: Optional[str] = None,
    other_end_ts: Optional[str] = None,
) -> bool:
    """
    return True if interval is contained in other
    """

    if other_start_ts is None:
        other_start_ts = interval_start_ts

    if other_end_ts is None:
        other_end_ts = interval_end_ts

    if (
        check_for_nan_values(interval[interval_start_ts])
        or check_for_nan_values(interval[interval_end_ts])
        or check_for_nan_values(other[other_start_ts])
        or check_for_nan_values(other[other_end_ts])
    ):
        raise ValueError("interval and other cannot contain NaN values for timestamps")

    return interval_starts_before(
        interval=other,
        other=interval,
        interval_start_ts=other_start_ts,
        other_start_ts=interval_start_ts,
    ) and interval_ends_before(
        interval=interval,
        other=other,
        interval_end_ts=interval_end_ts,
        other_end_ts=other_end_ts,
    )


def intervals_share_start_boundary(
    interval: pd.Series,
    other: pd.Series,
    interval_start_ts: str,
    other_start_ts: Optional[str] = None,
) -> bool:
    """
    return True if interval_a and interval_b share a start boundary
    """

    if other_start_ts is None:
        other_start_ts = interval_start_ts

    if check_for_nan_values(interval[interval_start_ts]) or check_for_nan_values(
        other[other_start_ts]
    ):
        raise ValueError("interval and other cannot contain NaN values for timestamps")

    return interval[interval_start_ts] == other[other_start_ts]


def intervals_share_end_boundary(
    interval: pd.Series,
    other: pd.Series,
    interval_end_ts: str,
    other_end_ts: Optional[str] = None,
) -> bool:
    """
    return True if interval_a and interval_b share an end boundary
    """

    if other_end_ts is None:
        other_end_ts = interval_end_ts

    if check_for_nan_values(interval[interval_end_ts]) or check_for_nan_values(
        other[other_end_ts]
    ):
        raise ValueError("interval and other cannot contain NaN values for timestamps")

    return interval[interval_end_ts] == other[other_end_ts]


def intervals_boundaries_are_equivalent(
    interval: pd.Series,
    other: pd.Series,
    interval_start_ts: str,
    interval_end_ts: str,
    other_start_ts: Optional[str] = None,
    other_end_ts: Optional[str] = None,
) -> bool:
    """
    return True if interval_a is equivalent to interval_b
    """

    if other_start_ts is None:
        other_start_ts = interval_start_ts

    if other_end_ts is None:
        other_end_ts = interval_end_ts

    if (
        check_for_nan_values(interval[interval_start_ts])
        or check_for_nan_values(interval[interval_end_ts])
        or check_for_nan_values(other[other_start_ts])
        or check_for_nan_values(other[other_end_ts])
    ):
        raise ValueError("interval and other cannot contain NaN values for timestamps")

    return intervals_share_start_boundary(
        interval,
        other,
        interval_start_ts=interval_start_ts,
        other_start_ts=other_start_ts,
    ) and intervals_share_end_boundary(
        interval,
        other,
        interval_end_ts=interval_end_ts,
        other_end_ts=other_end_ts,
    )


def intervals_have_equivalent_metric_columns(
    interval_a: pd.Series,
    interval_b: pd.Series,
    metric_columns: Iterable[str],
) -> bool:
    """
    return True if interval_a and interval_b have identical metrics
    """
    if isinstance(metric_columns, str):
        metric_columns = metric_columns.split(",")
        metric_columns = [s.strip() for s in metric_columns]
    elif isinstance(metric_columns, Iterable):
        metric_columns = list(metric_columns)
    else:
        raise ValueError(
            f"series_ids must be an Iterable or comma seperated string"
            f" of column names, instead got {type(metric_columns)}"
        )

    interval_a = interval_a.copy().fillna("¯\\_(ツ)_/¯")
    interval_b = interval_b.copy().fillna("¯\\_(ツ)_/¯")
    return all(
        interval_a[metric_col] == interval_b[metric_col]
        for metric_col in metric_columns
    )


def intervals_do_not_overlap(
    *,
    interval: pd.Series,
    other: pd.Series,
    interval_start_ts: str,
    interval_end_ts: str,
    other_start_ts: Optional[str] = None,
    other_end_ts: Optional[str] = None,
) -> bool:
    if other_start_ts is None:
        other_start_ts = interval_start_ts

    if other_end_ts is None:
        other_end_ts = interval_end_ts

    if (
        check_for_nan_values(interval[interval_start_ts])
        or check_for_nan_values(interval[interval_end_ts])
        or check_for_nan_values(other[other_start_ts])
        or check_for_nan_values(other[other_end_ts])
    ):
        raise ValueError("interval and other cannot contain NaN values for timestamps")

    return (
        interval[interval_end_ts] < other[other_start_ts]
        or interval[interval_start_ts] > other[other_end_ts]
    )


def update_interval_boundary(
    *,
    interval: pd.Series,
    boundary_to_update: str,
    update_value: str,
) -> pd.Series:
    """
    return new copy of interval with start or end time updated using update_value
    """
    if boundary_to_update not in (interval_keys := interval.keys()):
        raise KeyError(f"boundary_to_update must exist in of {interval_keys}")

    updated_interval = interval.copy()
    updated_interval[boundary_to_update] = update_value

    return updated_interval


def merge_metric_columns_of_intervals(
    *,
    main_interval: pd.Series,
    child_interval: pd.Series,
    metric_columns: Iterable[str],
    metric_merge_method: bool = False,
) -> pd.Series:
    """
    return the merged metrics of interval_a and interval_b
    """

    if isinstance(metric_columns, str):
        metric_columns = metric_columns.split(",")
        metric_columns = [s.strip() for s in metric_columns]
    elif isinstance(metric_columns, Iterable):
        metric_columns = list(metric_columns)
    else:
        raise ValueError(
            f"series_ids must be an Iterable or comma seperated string"
            f" of column names, instead got {type(metric_columns)}"
        )

    merged_interval = main_interval.copy()

    if metric_merge_method:
        for metric_col in metric_columns:
            if pd.notna(child_interval[metric_col]):
                merged_interval[metric_col] = child_interval[metric_col]

    return merged_interval


def resolve_overlap(  # TODO: need to implement proper metric merging
    #  -> for now, can just take non-null values from both intervals
    interval: pd.Series,
    other: pd.Series,
    interval_start_ts: str,
    interval_end_ts: str,
    series_ids: Iterable[str],
    metric_columns: Iterable[str],
    other_start_ts: Optional[str] = None,
    other_end_ts: Optional[str] = None,
) -> list[pd.Series]:
    """
    resolve overlaps between the two given intervals,
    splitting them as necessary into some set of disjoint intervals
    """

    if other_start_ts is None:
        try:
            _ = other[interval_start_ts]
            other_start_ts = interval_start_ts
        except KeyError:
            raise ValueError(
                f"`other_start_ts` must be set or equivalent to `interval_start_ts`, got {other_start_ts}"
            )

    if other_end_ts is None:
        try:
            _ = other[interval_end_ts]
            other_end_ts = interval_end_ts
        except KeyError:
            raise ValueError(
                f"`other_end_ts` must be set or equivalent to `interval_end_ts`, got {other_end_ts}"
            )

    if (
        check_for_nan_values(interval[interval_start_ts])
        or check_for_nan_values(interval[interval_end_ts])
        or check_for_nan_values(other[other_start_ts])
        or check_for_nan_values(other[other_end_ts])
    ):
        raise ValueError("interval and other cannot contain NaN values for timestamps")

    interval_index = set(interval.index).difference(
        (
            interval_start_ts,
            interval_end_ts,
        )
    )
    other_index = set(other.index).difference(
        (
            other_start_ts,
            other_end_ts,
        )
    )

    if not interval_index == other_index:
        raise ValueError("Expected indices of pd.Series elements to be equivalent.")

    for arg in (series_ids, metric_columns):
        if isinstance(arg, str):
            arg = [s.strip() for s in arg.split(",")]
        elif isinstance(arg, Iterable):
            arg = list(arg)
        else:
            raise ValueError(
                f"{arg} must be an Iterable or comma seperated string"
                f" of column names, instead got {type(arg)}"
            )

    series_ids = cast(list[str], series_ids)
    metric_columns = cast(list[str], metric_columns)

    resolved_intervals = list()

    # NB: Checking order of intervals in terms of start time allows
    # us to remove all cases where b precedes a because the interval
    # which opens sooner can always be set to a
    if interval[interval_start_ts] > other[other_start_ts]:
        interval, other = other, interval

    # intervals_do_not_overlap(interval, other, ...) is True
    #
    # Results in 2 disjoint intervals
    # 1) A.start, A.end, A.metric_columns
    # 2) B.start, B.end, B.metric_columns

    if intervals_do_not_overlap(
        interval=interval,
        other=other,
        interval_start_ts=interval_start_ts,
        interval_end_ts=interval_end_ts,
        other_start_ts=other_start_ts,
        other_end_ts=other_end_ts,
    ):
        return [interval, other]

    # intervals_have_equivalent_metric_columns(interval, other, metric_columns) is True
    #
    # Results in 1 disjoint interval
    # 1) A.start, B.end, A.metric_columns

    if intervals_have_equivalent_metric_columns(interval, other, metric_columns):
        resolved_series = update_interval_boundary(
            interval=interval,
            boundary_to_update=interval_end_ts,
            update_value=other[other_end_ts],
        )

        resolved_intervals.append(resolved_series)

        return resolved_intervals

    # interval_is_contained_by(interval=other, other=interval, ...) is True
    #
    # Results in 3 disjoint intervals
    # 1) A.start, B.start, A.metric_columns
    # 2) B.start, B.end, merge(A.metric_columns, B.metric_columns)
    # 3) B.end, A.end, A.metric_columns

    if interval_is_contained_by(
        interval=other,
        other=interval,
        interval_start_ts=other_start_ts,
        interval_end_ts=other_end_ts,
        other_start_ts=interval_start_ts,
        other_end_ts=interval_end_ts,
    ):
        # 1)
        resolved_series = update_interval_boundary(
            interval=interval,
            boundary_to_update=interval_end_ts,
            update_value=other[other_start_ts],
        )

        resolved_intervals.append(resolved_series)

        # 2)
        resolved_series = merge_metric_columns_of_intervals(
            main_interval=other,
            child_interval=interval,
            metric_columns=metric_columns,
            metric_merge_method=True,
        )

        resolved_intervals.append(resolved_series)

        # 3)
        resolved_series = update_interval_boundary(
            interval=interval,
            boundary_to_update=interval_start_ts,
            update_value=other[other_end_ts],
        )

        resolved_intervals.append(resolved_series)

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

    if intervals_share_start_boundary(
        interval,
        other,
        interval_start_ts=interval_start_ts,
        other_start_ts=other_start_ts,
    ) and not intervals_share_end_boundary(
        interval, other, interval_end_ts=interval_end_ts
    ):
        if interval_ends_before(
            interval=interval,
            other=other,
            interval_end_ts=interval_end_ts,
            other_end_ts=other_end_ts,
        ):
            # 1)
            resolved_series = merge_metric_columns_of_intervals(
                main_interval=interval,
                child_interval=other,
                metric_columns=metric_columns,
                metric_merge_method=True,
            )

            resolved_intervals.append(resolved_series)

            # 2)
            resolved_series = update_interval_boundary(
                interval=other,
                boundary_to_update=other_start_ts,
                update_value=interval[interval_end_ts],
            )

            resolved_intervals.append(resolved_series)

        else:
            # 1)
            resolved_series = merge_metric_columns_of_intervals(
                main_interval=other,
                child_interval=interval,
                metric_columns=metric_columns,
                metric_merge_method=True,
            )

            resolved_intervals.append(resolved_series)

            # 2)
            resolved_series = update_interval_boundary(
                interval=interval,
                boundary_to_update=interval_start_ts,
                update_value=other[other_end_ts],
            )

            resolved_intervals.append(resolved_series)

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

    if not intervals_share_start_boundary(
        interval,
        other,
        interval_start_ts=interval_start_ts,
        other_start_ts=other_start_ts,
    ) and intervals_share_end_boundary(
        interval,
        other,
        interval_end_ts=interval_end_ts,
        other_end_ts=other_end_ts,
    ):
        if interval_starts_before(
            interval=interval,
            other=other,
            interval_start_ts=interval_start_ts,
            other_start_ts=other_start_ts,
        ):
            # 1)
            resolved_series = update_interval_boundary(
                interval=interval,
                boundary_to_update=interval_end_ts,
                update_value=other[other_start_ts],
            )

            resolved_intervals.append(resolved_series)

            # 2)
            resolved_series = merge_metric_columns_of_intervals(
                main_interval=other,
                child_interval=interval,
                metric_columns=metric_columns,
                metric_merge_method=True,
            )

            resolved_intervals.append(resolved_series)

        return resolved_intervals

    # A and B share a common start and end boundary, making them equivalent.
    # - A.start = B.start & A.end = B.end
    #
    # Results in 1 disjoint interval
    # 1) A.start, A.end, merge(A.metric_columns, B.metric_columns)

    if intervals_boundaries_are_equivalent(
        interval,
        other,
        interval_start_ts=interval_start_ts,
        interval_end_ts=interval_end_ts,
        other_start_ts=other_start_ts,
        other_end_ts=other_end_ts,
    ):
        resolved_series = merge_metric_columns_of_intervals(
            main_interval=interval,
            child_interval=other,
            metric_columns=metric_columns,
            metric_merge_method=True,
        )

        resolved_intervals.append(resolved_series)

        return resolved_intervals

    # Interval A starts first and A partially overlaps B
    # - A.start < B.start & A.end < B.end
    #
    # Results in 3 disjoint intervals
    # 1) A.start, B.start, A.metric_columns
    # 2) B.start, A.end, merge(A.metric_columns, B.metric_columns)
    # 3) A.end, B.end, B.metric_columns

    if interval_starts_before(
        interval=interval,
        other=other,
        interval_start_ts=interval_start_ts,
        other_start_ts=other_start_ts,
    ) and interval_ends_before(
        interval=interval,
        other=other,
        interval_end_ts=interval_end_ts,
        other_end_ts=other_end_ts,
    ):
        # 1)
        resolved_series = update_interval_boundary(
            interval=interval,
            boundary_to_update=interval_end_ts,
            update_value=other[other_start_ts],
        )

        resolved_intervals.append(resolved_series)

        # 2)
        updated_series = update_interval_boundary(
            interval=other,
            boundary_to_update=other_end_ts,
            update_value=interval[interval_end_ts],
        )
        resolved_series = merge_metric_columns_of_intervals(
            main_interval=updated_series,
            child_interval=interval,
            metric_columns=metric_columns,
            metric_merge_method=True,
        )

        resolved_intervals.append(resolved_series)

        # 3)
        resolved_series = update_interval_boundary(
            interval=other,
            boundary_to_update=other_start_ts,
            update_value=interval[interval_end_ts],
        )

        resolved_intervals.append(resolved_series)

        return resolved_intervals

    raise NotImplementedError("Interval resolution not implemented")


def resolve_all_overlaps(
    with_row: pd.Series,
    overlaps: pd.DataFrame,
    with_row_start_ts: str,
    with_row_end_ts: str,
    series_ids: Iterable[str],
    metric_columns: Iterable[str],
    overlap_start_ts: Optional[str] = None,
    overlap_end_ts: Optional[str] = None,
) -> pd.DataFrame:
    """
    resolve the interval `x` against all overlapping intervals in `overlapping`,
    returning a set of disjoint intervals with the same spans
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
    """

    if overlap_start_ts is None:
        try:
            _ = overlaps[with_row_start_ts]
            overlap_start_ts = with_row_start_ts
        except KeyError:
            raise ValueError(
                f"`overlaps_start_ts` must be set or equivalent to `with_row_start_ts`, got {overlap_start_ts}"
            )

    if overlap_end_ts is None:
        try:
            _ = overlaps[with_row_end_ts]
            overlap_end_ts = with_row_end_ts
        except KeyError:
            raise ValueError(
                f"`overlaps_end_ts` must be set or equivalent to `with_row_end_ts`, got {overlap_end_ts}"
            )

    for arg in (series_ids, metric_columns):
        if isinstance(arg, str):
            arg = [s.strip() for s in arg.split(",")]
        elif isinstance(arg, Iterable):
            arg = list(arg)
        else:
            raise ValueError(
                f"{arg} must be an Iterable or comma seperated string"
                f" of column names, instead got {type(arg)}"
            )

    series_ids = cast(list[str], series_ids)
    metric_columns = cast(list[str], metric_columns)

    first_row = overlaps.iloc[0]
    initial_intervals = resolve_overlap(
        with_row,
        first_row,
        with_row_start_ts,
        with_row_end_ts,
        series_ids,
        metric_columns,
        overlap_start_ts,
        overlap_end_ts,
    )
    local_disjoint_df = pd.DataFrame(initial_intervals)

    # NB: using `itertools.islice` to build a generator that skips the first
    # row of overlaps
    for _, row in islice(overlaps.iterrows(), 1, None):
        resolved_intervals = resolve_overlap(
            with_row,
            row,
            with_row_start_ts,
            with_row_end_ts,
            series_ids,
            metric_columns,
            overlap_start_ts,
            overlap_end_ts,
        )
        for interval in resolved_intervals:
            local_disjoint_df = add_as_disjoint(
                interval,
                local_disjoint_df,
                (overlap_start_ts, overlap_end_ts),
                series_ids,
                metric_columns,
            )

    return local_disjoint_df


def add_as_disjoint(
    interval: pd.Series,
    disjoint_set: Optional[pd.DataFrame],
    interval_boundaries: Iterable[str],
    series_ids: Iterable[str],
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    """
    returns a disjoint set consisting of the given interval, made disjoint with those already in `disjoint_set`
    """

    if isinstance(interval_boundaries, str):
        _ = interval_boundaries.split(",")
        interval_boundaries = [s.strip() for s in _]
    elif isinstance(interval_boundaries, Iterable):
        interval_boundaries = list(interval_boundaries)
    else:
        raise ValueError(
            f"series_ids must be an Iterable or comma seperated string"
            f" of column names, instead got {type(interval_boundaries)}"
        )

    if len(interval_boundaries) != 2:
        raise ValueError(
            f"interval_boundaries must be an Iterable of length 2, instead got {len(interval_boundaries)}"
        )

    start_ts, end_ts = interval_boundaries

    for arg in (series_ids, metric_columns):
        if isinstance(arg, str):
            arg = [s.strip() for s in arg.split(",")]
        elif isinstance(arg, Iterable):
            arg = list(arg)
        else:
            raise ValueError(
                f"{arg} must be an Iterable or comma seperated string"
                f" of column names, instead got {type(arg)}"
            )

    series_ids = cast(list[str], series_ids)
    metric_columns = cast(list[str], metric_columns)

    if disjoint_set is None:
        return pd.DataFrame([interval])

    if disjoint_set.empty:
        return pd.DataFrame([interval])

    overlapping_subset_df = identify_interval_overlaps(
        in_pdf=disjoint_set,
        with_row=interval,
        interval_start_ts=start_ts,
        interval_end_ts=end_ts,
    )

    # if there are no overlaps, add the interval to disjoint_set
    if overlapping_subset_df.empty:
        element_wise_comparison = (
            disjoint_set.fillna("¯\\_(ツ)_/¯") == interval.fillna("¯\\_(ツ)_/¯").values
        )
        row_wise_comparison = element_wise_comparison.all(axis=1)
        # NB: because of the nested iterations, we need to check that the
        # record hasn't already been added to `global_disjoint_df` by another loop
        if row_wise_comparison.any():
            return disjoint_set
        else:
            return pd.concat((disjoint_set, pd.DataFrame([interval])))

    # identify all intervals which do not overlap with the given interval to
    # concatenate them to the disjoint set after resolving overlaps
    non_overlapping_subset_df = disjoint_set[
        ~disjoint_set.set_index(interval_boundaries).index.isin(
            overlapping_subset_df.set_index(interval_boundaries).index
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
        return pd.DataFrame(
            resolve_overlap(
                interval,
                overlapping_subset_df.iloc[0],
                start_ts,
                end_ts,
                series_ids,
                metric_columns,
            )
        )

    if multiple_to_resolve and only_overlaps_present:
        return resolve_all_overlaps(
            interval,
            overlapping_subset_df,
            start_ts,
            end_ts,
            series_ids,
            metric_columns,
        )

    if not multiple_to_resolve and not only_overlaps_present:
        return pd.concat(
            (
                pd.DataFrame(
                    resolve_overlap(
                        interval,
                        overlapping_subset_df.iloc[0],
                        start_ts,
                        end_ts,
                        series_ids,
                        metric_columns,
                    )
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
                    start_ts,
                    end_ts,
                    series_ids,
                    metric_columns,
                ),
                non_overlapping_subset_df,
            ),
        )

    # if we get here, something went wrong
    raise NotImplementedError


def make_disjoint_wrap(
    start_ts: str,
    end_ts: str,
    series_ids: Iterable[str],
    metric_columns: Iterable[str],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def make_disjoint_inner(
        pdf: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        function will process all intervals in the input, and break down overlapping intervals into a fully disjoint set
        https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-and-then-filling-it
        https://stackoverflow.com/questions/55478191/list-of-series-to-dataframe
        https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.append.html
        """

        global_disjoint_df = pd.DataFrame(columns=pdf.columns)

        sorted_pdf = pdf.sort_values([start_ts, end_ts])

        for _, row in sorted_pdf.iterrows():
            global_disjoint_df = add_as_disjoint(
                row,
                global_disjoint_df,
                (start_ts, end_ts),
                series_ids,
                metric_columns,
            )

        return global_disjoint_df

    return make_disjoint_inner
