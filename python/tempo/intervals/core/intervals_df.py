from functools import cached_property
from typing import Optional, Iterable

import pyspark.sql.functions as f
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window, WindowSpec

from tempo.intervals.core.exceptions import ErrorMessages
from tempo.intervals.spark.functions import make_disjoint_wrap, is_metric_col


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
        :param start_ts: Name of the column which denotes interval._start
        :type start_ts: str
        :param end_ts: Name of the column which denotes interval._end
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
