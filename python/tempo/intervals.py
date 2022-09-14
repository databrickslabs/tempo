from __future__ import annotations

from typing import Optional, Sequence
from typing import cast as typing_cast

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as f
from pyspark.sql.window import Window


class IntervalsDF:
    """
    This object is the main wrapper over a `Spark DataFrame`_ which allows a
    user to parallelize computations over snapshots of metrics for intervals
    of time defined by a start and end timestamp and various dimensions.

    The required dimensions are `series` (list of columns by which to summarize),
    `metrics` (list of columns to analyze), `start_ts` (timestamp column), and
    `end_ts` (timestamp column). `start_ts` and `end_ts` can be epoch or TimestampType.

    .. _`Spark DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html

    """

    def __init__(
        self,
        df: DataFrame,
        start_ts: str,
        end_ts: str,
        series: list[str],
        metrics: list[str],
    ) -> None:
        """
        :class:`IntervalsDF` Constructor

        :param df:
        :type df: `DataFrame`_
        :param start_ts:
        :type start_ts: str
        :param end_ts:
        :type end_ts: str
        :param series:
        :type series: list[str]
        :param metrics:
        :type metrics: list[str]
        :rtype: None

        :Example:

        .. code-block:

            df = spark.createDataFrame(
                [["2020-08-01 00:00:09", "2020-08-01 00:00:14", "v1", 5, 0]],
                "start_ts STRING, end_ts STRING, series_1 STRING, metric_1 INT, metric_2 INT",
            )
            idf = IntervalsDF(df, "start_ts", "end_ts", ["series_1"], ["metric_1", "metric_2"])
            idf.df.collect()
            [Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:14', series_1='v1', metric_1=5, metric_2=0)]

        .. _`DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html

        .. todo::
            create IntervalsSchema class to validate data types and column existence
            check elements of series and identifers to ensure all are str
            check if start_ts, end_ts, and the elements of series and identifiers can be of type col

        """

        self.start_ts = start_ts
        self.end_ts = end_ts

        self._interval_boundaries = [
            self.start_ts,
            self.end_ts,
        ]

        if not series or not isinstance(series, list):
            raise ValueError
        else:
            self.series_cols = series

        if not metrics or not isinstance(metrics, list):
            raise ValueError
        else:
            self.metric_cols = metrics

        self._window = Window.partitionBy(*self.series_cols).orderBy(
            *self._interval_boundaries
        )

        self.df = df

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
        Pivots metrics of the current DataFrame by start and end timestamp and identifiers.
        There are two versions of :meth:`fromStackedMetrics`. One that requires the caller
        to specify the list of distinct metric names to pivot on, and one that does not. The
        latter is more concise but less efficient, because Spark needs to first compute the
        list of distinct metric names internally.

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
        :param metric_names: List of metric names that will be translated to columns in the output :class:`IntervalsDF`.
        :type metric_names: list[str], optional
        :return: A new :class:`IntervalsDF` with a column and respective values per distinct metric in `metrics_name_col`.

        :Example:

        .. code-block::

            df = spark.createDataFrame(
                 [["2020-08-01 00:00:09", "2020-08-01 00:00:14", "v1", "metric_1", 5],
                 ["2020-08-01 00:00:09", "2020-08-01 00:00:11", "v1", "metric_2", 0]],
                "start_ts STRING, end_ts STRING, series_1 STRING, metric_name STRING, metric_value INT",
            )

            # With distinct metric names specified

            idf = IntervalsDF.fromStackedMetrics(df,"start_ts","end_ts",["series_1"],"metric_name","metric_value",["metric_1", "metric_2"])
            idf.df.collect()
            [Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:14', series_1='v1', metric_1=5, metric_2=null),
             Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:11', series_1='v1', metric_1=null, metric_2=0)]

            # Or without specifying metric names (less efficient)

            idf = IntervalsDF.fromStackedMetrics(df,"start_ts","end_ts",["series_1"],"metric_name","metric_value")
            idf.df.collect()
            [Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:14', series_1='v1', metric_1=5, metric_2=null),
             Row(start_ts='2020-08-01 00:00:09', end_ts='2020-08-01 00:00:11', series_1='v1', metric_1=null, metric_2=0)]

        .. _`DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html

        .. todo::
            check elements of identifiers to ensure all are str
            check if start_ts, end_ts, and the elements of series and identifiers can be of type col
            new test case for metric_names

        """

        if not isinstance(series, list):
            raise ValueError

        df = (
            df.groupBy(start_ts, end_ts, *series)
            .pivot(metrics_name_col, values=metric_names)  # type: ignore
            .max(metrics_value_col)
        )

        metric_cols = list(
            col for col in df.columns if col not in (start_ts, end_ts, *series)
        )

        return cls(df, start_ts, end_ts, series, metric_cols)

    def __get_adjacent_rows(self, df: DataFrame) -> DataFrame:
        """
        Returns a new `Spark DataFrame`_ containing columns from applying the
        `lead`_ and `lag`_ window functions.


        .. _`Spark DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html
        .. `lead`_: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.lead.html#pyspark.sql.functions.lead
        .. `lag`_: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.lag.html#pyspark.sql.functions.lag

        .. todo:
            should column names generated here be created at class initialization, saved as attributes then iterated on here?
             this would allow easier reuse throughout code

        """
        for c in self._interval_boundaries + self.metric_cols:
            df = df.withColumn(
                f"_lead_1_{c}",
                f.lead(c, 1).over(self._window),
            ).withColumn(
                f"_lag_1_{c}",
                f.lag(c, 1).over(self._window),
            )

        return df

    def __identify_subset_intervals(self, df: DataFrame) -> tuple[DataFrame, str]:
        """
        Returns a new `Spark DataFrame`_ containing a boolean column if the
        current interval is a subset of the previous interval and the name
        of this column for future use

        .. _`Spark DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html

        .. todo:
            should subset_indicator be defined here or as an attribute for easier reuse across code?

        """

        subset_indicator = "_lag_1_is_subset"

        df = df.withColumn(
            subset_indicator,
            (f.col(f"_lag_1_{self.start_ts}") <= f.col(self.start_ts))
            & (f.col(f"_lag_1_{self.end_ts}") >= f.col(self.end_ts)),
        )

        # NB: the first record cannot be a subset of the previous and
        #     lag will return null for this record with no default set
        df = df.fillna(
            False,
            subset=[subset_indicator],
        )

        return df, subset_indicator

    def __identify_overlaps(self, df: DataFrame) -> tuple[DataFrame, list[str]]:
        """
        Returns a new `Spark DataFrame`_ containing boolean columns if the
        current interval overlaps with the previous or next interval and a list
        with the names of these columns for future use

        .. _`Spark DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html

        .. todo:
            should overlap_indicators be defined here or as an attribute for easier reuse across code?

        """

        overlap_indicators: list[str] = []

        # identify overlaps for each interval boundary
        # NB: between is inclusive so not used here and
        #     we don't care if the intervals match at the boundaries
        for ts in self._interval_boundaries:
            df = df.withColumn(
                f"_lead_1_{ts}_overlaps",
                (f.col(f"_lead_1_{ts}") > f.col(self.start_ts))
                & (f.col(f"_lead_1_{ts}") < f.col(self.end_ts)),
            ).withColumn(
                f"_lag_1_{ts}_overlaps",
                (f.col(f"_lag_1_{ts}") > f.col(self.start_ts))
                & (f.col(f"_lag_1_{ts}") < f.col(self.end_ts)),
            )

            overlap_indicators.extend(
                (
                    f"_lead_1_{ts}_overlaps",
                    f"_lag_1_{ts}_overlaps",
                )
            )

        # NB: the first and last record cannot be a subset of the previous and next respectively.
        #     lag will return null for this record with no default set.
        df = df.fillna(
            False,
            subset=overlap_indicators,
        )

        return df, overlap_indicators

    def __merge_adjacent_subset_and_superset(
        self, df: DataFrame, subset_indicator: Optional[str]
    ) -> DataFrame:
        """
        Returns a new `Spark DataFrame`_ where a subset and it's adjacent superset,
        identified by `subset_indicator` are merged together.

        We assume that a metric cannot simultaneously have two values in the same
        interval (unless captured in a structure such as ArrayType, etc) so `coalesce`_
        is used to merge metrics when overlaps exist. Priority in the coalesce is
        given to the metrics of the current record, ie it is listed as the first argumnt.

        .. _`Spark DataFrame`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html
        .. _`coalesce`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.coalesce.html

        .. todo:
            should subset_indicator be defined here or as an attribute for easier reuse across code?

        """

        if not subset_indicator:
            df, subset_indicator = self.__identify_subset_intervals(df)

        for c in self.metric_cols:
            df = df.withColumn(
                c,
                f.when(
                    f.col(subset_indicator), f.coalesce(f.col(c), f"_lag_1_{c}")
                ).otherwise(f.col(c)),
            )

        return df

    def __merge_adjacent_overlaps(
        self, df: DataFrame, how: str, overlap_indicators: Optional[list[str]]
    ) -> DataFrame:
        """

        :rtype: DataFrame
        :param df:
        :param how:
        :param overlap_indicators:
        :return:
        """
        if not (how == "left" or how == "right"):
            raise ValueError
        elif how == "left":
            new_boundary_col = self.end_ts
            new_boundary_val = f"_lead_1_{self.start_ts}"
        else:
            new_boundary_col = self.start_ts
            new_boundary_val = f"_lead_1_{self.end_ts}"

        if not overlap_indicators:
            df, overlap_indicators = self.__identify_overlaps(df)

        superset_interval_when_case = (
            # TODO : can replace with subset_indicator?
            f"WHEN _lead_1_{self.start_ts}_overlaps "
            f"AND _lead_1_{self.end_ts}_overlaps "
            f"THEN {new_boundary_val} "
        )

        overlap_interval_when_cases = tuple(
            f"WHEN {c} THEN {c.replace('_overlaps', '')} " for c in overlap_indicators
        )

        new_interval_boundaries = (
            "CASE "
            + superset_interval_when_case
            + " ".join(overlap_interval_when_cases)
            + "ELSE null END "
        )

        df = df.withColumn(
            new_boundary_col,
            f.expr(new_interval_boundaries),
        )

        if how == "left":

            for c in self.metric_cols:
                df = df.withColumn(
                    f"{c}",
                    f.when(
                        f.col(f"_lead_1_{self.start_ts}_overlaps"),
                        f.col(c),
                    )
                    .when(
                        f.col(f"_lead_1_{self.end_ts}_overlaps"),
                        f.coalesce(f.col(c), f.col(f"_lead_1_{c}")),
                    )
                    .when(
                        f.col(f"_lag_1_{self.start_ts}_overlaps"),
                        f.col(c),
                    )
                    .when(
                        f.col(f"_lag_1_{self.end_ts}_overlaps"),
                        f.coalesce(f.col(c), f.col(f"_lag_1_{c}")),
                    ),
                )

        else:

            for c in self.metric_cols:
                df = df.withColumn(
                    f"{c}",
                    f.when(
                        f.col(f"_lead_1_{self.start_ts}_overlaps")
                        & f.col(f"_lead_1_{self.end_ts}_overlaps"),
                        f.col(c),
                    )
                    .when(
                        f.col(f"_lead_1_{self.start_ts}_overlaps"),
                        f.coalesce(f.col(c), f.col(f"_lead_1_{c}")),
                    )
                    .when(f.col(f"_lead_1_{self.end_ts}_overlaps"), f.col(c))
                    .when(
                        f.col(f"_lag_1_{self.start_ts}_overlaps"),
                        f.coalesce(f.col(c), f.col(f"_lag_1_{c}")),
                    )
                    .when(f.col(f"_lag_1_{self.end_ts}_overlaps"), f.col(c)),
                )

        return df

    def __merge_equal_intervals(self, df: DataFrame) -> DataFrame:
        """

        :param df:
        :return:
        :rtype: DataFrame
        """
        merge_expr = tuple(f.max(c).alias(c) for c in self.metric_cols)

        return df.groupBy(*self._interval_boundaries, *self.series_cols).agg(
            *merge_expr
        )

    def disjoint(self) -> "IntervalsDF":
        """

        :return:
        """
        df = self.df

        df = self.__get_adjacent_rows(df)

        (df, subset_indicator) = self.__identify_subset_intervals(df)

        subset_df = df.filter(f.col(subset_indicator))

        subset_df = self.__merge_adjacent_subset_and_superset(
            subset_df, subset_indicator
        )

        subset_df = subset_df.select(
            *self._interval_boundaries, *self.series_cols, *self.metric_cols
        )

        non_subset_df = df.filter(~f.col(subset_indicator))

        (non_subset_df, overlap_indicators) = self.__identify_overlaps(non_subset_df)

        overlaps_predicate = " OR ".join(tuple(col for col in overlap_indicators))

        overlaps_df = non_subset_df.filter(overlaps_predicate)

        disjoint_predicate = f"NOT({overlaps_predicate})"

        # extract intervals that are already disjoint
        disjoint_df = non_subset_df.filter(disjoint_predicate).select(
            *self._interval_boundaries, *self.series_cols, *self.metric_cols
        )

        left_overlaps_df = self.__merge_adjacent_overlaps(
            overlaps_df, "left", overlap_indicators
        )

        left_overlaps_df = left_overlaps_df.select(
            *self._interval_boundaries, *self.series_cols, *self.metric_cols
        )

        right_overlaps_df = self.__merge_adjacent_overlaps(
            overlaps_df, "right", overlap_indicators
        )

        right_overlaps_df = right_overlaps_df.select(
            *self._interval_boundaries, *self.series_cols, *self.metric_cols
        )

        unioned_df = (
            subset_df.unionByName(disjoint_df)
            .unionByName(left_overlaps_df)
            .unionByName(right_overlaps_df)
        )

        disjoint_df = self.__merge_equal_intervals(unioned_df)

        return IntervalsDF(
            disjoint_df,
            self.start_ts,
            self.end_ts,
            self.series_cols,
            self.metric_cols,
        )

    def union(self, other: "IntervalsDF") -> "IntervalsDF":
        """

        :param other:
        :return:
        """
        if not isinstance(other, IntervalsDF):
            raise TypeError

        return IntervalsDF(
            self.df.union(other.df),
            self.start_ts,
            self.end_ts,
            self.series_cols,
            self.metric_cols,
        )

    def unionByName(self, other: "IntervalsDF") -> "IntervalsDF":
        """

        :param other:
        :return:
        """
        if not isinstance(other, IntervalsDF):
            raise TypeError

        return IntervalsDF(
            self.df.unionByName(other.df),
            self.start_ts,
            self.end_ts,
            self.series_cols,
            self.metric_cols,
        )

    def toDF(self, stack: bool = False) -> DataFrame:
        """

        :param stack:
        :return:
        """
        if stack:

            n_cols = len(self.metric_cols)
            metric_cols_expr = ",".join(
                tuple(f"'{col}', {col}" for col in self.metric_cols)
            )

            stack_expr = (
                f"STACK({n_cols}, {metric_cols_expr}) AS (metric_name, metric_value)"
            )

            return self.df.select(
                *self._interval_boundaries, *self.series_cols, f.expr(stack_expr)
            ).dropna(subset="metric_value")

        else:
            return self.df
