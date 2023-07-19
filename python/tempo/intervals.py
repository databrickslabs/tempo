from __future__ import annotations

from typing import Optional, Iterable, cast
from functools import cached_property

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import NumericType, BooleanType, StructField
import pyspark.sql.functions as f
from pyspark.sql.window import Window

import numpy as np
import pandas as pd


def is_metric_col(col: StructField) -> bool:
    return isinstance(col.dataType, NumericType) or isinstance(
        col.dataType, BooleanType
    )


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

    # TODO: https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    # TODO: try this one too: https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.functions.pandas_udf.html
    def __init__(
            self, df: DataFrame, start_ts: str, end_ts: str, series_ids: list[str] = None
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
            idf = IntervalsDF(df, "start_ts", "end_ts", ["series_1"], ["metric_1", "metric_2"])
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
        elif isinstance(series_ids, list):
            self.series_ids = series_ids
        else:
            raise ValueError(
                f"series_ids must be a list of column names, instead got {type(series_ids)}"
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
    def window(self):
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
            .pivot(metrics_name_col, values=metric_names)  # type: ignore
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
        to retrieve specific metric values across all instances.

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
        local_df = self.df
        local_start_ts = self.start_ts
        local_end_ts = self.end_ts
        local_series_ids = self.series_ids
        local_metric_columns = self.metric_columns

        def identify_overlaps(
                in_pdf: pd.DataFrame,
                with_row: pd.Series,
        ) -> pd.DataFrame:
            """
            return the subset of rows in DataFrame `in_pdf` that overlap with row `with_row`
            """

            if in_pdf.empty or with_row.empty:
                return in_pdf

            local_in_pdf = in_pdf.copy()  # TODO: do we need to make a copy?

            # https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column
            local_in_pdf["max_start_timestamp"] = [
                _ if _ >= with_row[local_start_ts] else with_row[local_start_ts]
                for _ in local_in_pdf[local_start_ts]
            ]

            local_in_pdf["min_end_timestamp"] = [
                _ if _ <= with_row[local_end_ts] else with_row[local_end_ts]
                for _ in local_in_pdf[local_end_ts]
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
                    local_in_pdf.fillna("¯\\_(ツ)_/¯")
                    == np.array(with_row.fillna("¯\\_(ツ)_/¯"))
            ).all(1)
            local_in_pdf = local_in_pdf[remove_with_row_mask]

            return local_in_pdf

        def interval_starts_before(
                *, interval: pd.Series, starts_before: pd.Series
        ) -> bool:
            """
            return True if interval_a starts before interval_b starts
            """
            return interval[local_start_ts] < starts_before[local_start_ts]

        def interval_ends_before(
                *, interval: pd.Series, ends_before: pd.Series
        ) -> bool:
            """
            return True if interval_a ends before interval_b ends
            """
            return interval[local_end_ts] < ends_before[local_end_ts]

        def interval_is_contained(
                *, interval: pd.Series, in_interval: pd.Series
        ) -> bool:
            """
            return True if interval is contained in in_interval
            """
            return interval_starts_before(
                interval=in_interval, starts_before=interval
            ) and interval_ends_before(interval=interval, ends_before=in_interval)

        def intervals_share_start_boundary(
                interval_a: pd.Series, interval_b: pd.Series
        ) -> bool:
            """
            return True if interval_a and interval_b share a start boundary
            """
            return interval_a[local_start_ts] == interval_b[local_start_ts]

        def intervals_share_end_boundary(
                interval_a: pd.Series, interval_b: pd.Series
        ) -> bool:
            """
            return True if interval_a and interval_b share an end boundary
            """
            return interval_a[local_end_ts] == interval_b[local_end_ts]

        def intervals_are_equivalent(
                interval_a: pd.Series, interval_b: pd.Series
        ) -> bool:
            """
            return True if interval_a is equivalent to interval_b
            """
            return intervals_share_start_boundary(
                interval_a, interval_b
            ) and intervals_share_end_boundary(interval_a, interval_b)

        def intervals_have_equivalent_metric_columns(
                interval_a: pd.Series,
                interval_b: pd.Series,
                metric_columns: Iterable[str],
        ) -> bool:
            """
            return True if interval_a and interval_b have identical metrics
            """
            interval_a = interval_a.copy().fillna("¯\\_(ツ)_/¯")
            interval_b = interval_b.copy().fillna("¯\\_(ツ)_/¯")
            return all(
                interval_a[metric_col] == interval_b[metric_col]
                for metric_col in metric_columns
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
            updated_interval = interval.copy()
            updated_interval[boundary_to_update] = update_value

            return updated_interval

        def merge_metric_columns_of_intervals(
                *,
                main_interval: pd.Series,
                child_interval: pd.Series,
                metric_columns: Iterable[str] = None,
                metric_merge_method: bool = False,
        ) -> pd.Series:
            """
            return the merged metrics of interval_a and interval_b
            """

            if metric_columns is None:
                metric_columns = local_metric_columns

            merged_interval = main_interval.copy()
            if metric_merge_method:
                for metric_col in metric_columns:
                    if pd.notna(child_interval[metric_col]):
                        merged_interval[metric_col] = child_interval[metric_col]

            return merged_interval

        def resolve_overlap(  # TODO: need to implement proper metric merging
                #  -> for now, can just take non-null values from both intervals
                interval_a: pd.Series,
                interval_b: pd.Series,
                series_ids: Iterable[str] = None,
                metric_columns: Iterable[str] = None,
                resolved_intervals: list[pd.Series] = None,
        ) -> list[pd.Series]:
            """
            resolve overlaps between the two given intervals,
            splitting them as necessary into some set of disjoint intervals
            """
            
            if series_ids is None:
                series_ids = local_series_ids
                
            if metric_columns is None:
                metric_columns = local_metric_columns

            if not (interval_a.index == interval_b.index).all():
                raise ValueError(
                    "Expected indices of pd.Series elements to be equivalent."
                )

            if resolved_intervals is None:
                resolved_intervals = list()

            # NB: Checking order of intervals in terms of start time allows
            # us to remove all cases where b precedes a because the interval
            # which opens sooner can always be set to a
            if interval_a[local_start_ts] > interval_b[local_start_ts]:
                interval_a, interval_b = interval_b, interval_a

            # intervals_have_identical_metric_columns(interval_a, interval_b, metric_columns) is True
            #
            # Results in 1 disjoint interval
            # 1) A.start, B.end, A.metric_columns

            if intervals_have_equivalent_metric_columns(interval_a, interval_b, metric_columns):
                resolved_series = update_interval_boundary(
                    interval=interval_a,
                    boundary_to_update=local_end_ts,
                    update_value=interval_b[local_end_ts],
                )

                resolved_intervals.append(resolved_series)

                return resolved_intervals

            # interval_is_contained(interval=interval_b, in_interval=interval_a) is True
            #
            # Results in 3 disjoint intervals
            # 1) A.start, B.start, A.metric_columns
            # 2) B.start, B.end, merge(A.metric_columns, B.metric_columns)
            # 3) B.end, A.end, A.metric_columns

            if interval_is_contained(interval=interval_b, in_interval=interval_a):
                # 1)
                resolved_series = update_interval_boundary(
                    interval=interval_a,
                    boundary_to_update=local_end_ts,
                    update_value=interval_b[local_start_ts],
                )

                resolved_intervals.append(resolved_series)

                # 2)
                resolved_series = merge_metric_columns_of_intervals(
                    main_interval=interval_b,
                    child_interval=interval_a,
                    metric_columns=metric_columns,
                    metric_merge_method=True,
                )

                resolved_intervals.append(resolved_series)

                # 3)
                resolved_series = update_interval_boundary(
                    interval=interval_a,
                    boundary_to_update=local_start_ts,
                    update_value=interval_b[local_end_ts],
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
                    interval_a, interval_b
            ) and not intervals_share_end_boundary(interval_a, interval_b):
                if interval_ends_before(interval=interval_a, ends_before=interval_b):
                    # 1)
                    resolved_series = merge_metric_columns_of_intervals(
                        main_interval=interval_a,
                        child_interval=interval_b,
                        metric_columns=metric_columns,
                        metric_merge_method=True,
                    )

                    resolved_intervals.append(resolved_series)

                    # 2)
                    resolved_series = update_interval_boundary(
                        interval=interval_b,
                        boundary_to_update=local_start_ts,
                        update_value=interval_a[local_end_ts],
                    )

                    resolved_intervals.append(resolved_series)

                else:
                    # 1)
                    resolved_series = merge_metric_columns_of_intervals(
                        main_interval=interval_b,
                        child_interval=interval_a,
                        metric_columns=metric_columns,
                        metric_merge_method=True,
                    )

                    resolved_intervals.append(resolved_series)

                    # 2)
                    resolved_series = update_interval_boundary(
                        interval=interval_a,
                        boundary_to_update=local_start_ts,
                        update_value=interval_b[local_end_ts],
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
                    interval_a, interval_b
            ) and intervals_share_end_boundary(interval_a, interval_b):
                if interval_starts_before(
                        interval=interval_a, starts_before=interval_b
                ):
                    # 1)
                    resolved_series = update_interval_boundary(
                        interval=interval_a,
                        boundary_to_update=local_end_ts,
                        update_value=interval_b[local_start_ts],
                    )

                    resolved_intervals.append(resolved_series)

                    # 2)
                    resolved_series = merge_metric_columns_of_intervals(
                        main_interval=interval_b,
                        child_interval=interval_a,
                        metric_columns=metric_columns,
                        metric_merge_method=True,

                    )

                    resolved_intervals.append(resolved_series)

                else:  # TODO: THINK WE CAN REMOVE THIS BECAUSE WE CHECK FOR ORDERING AT BEGINNING OF FUNCTION
                    # 1)
                    resolved_series = update_interval_boundary(
                        interval=interval_b,
                        boundary_to_update=local_end_ts,
                        update_value=interval_a[local_start_ts],
                    )

                    resolved_intervals.append(resolved_series)

                    # 2)
                    resolved_series = merge_metric_columns_of_intervals(
                        main_interval=interval_a,
                        child_interval=interval_b,
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

            if intervals_are_equivalent(interval_a, interval_b):
                resolved_series = merge_metric_columns_of_intervals(
                    main_interval=interval_a,
                    child_interval=interval_b,
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
                    interval=interval_a, starts_before=interval_b
            ) and interval_ends_before(interval=interval_a, ends_before=interval_b):
                # 1)
                resolved_series = update_interval_boundary(
                    interval=interval_a,
                    boundary_to_update=local_end_ts,
                    update_value=interval_b[local_start_ts],
                )

                resolved_intervals.append(resolved_series)

                # 2)
                updated_series = update_interval_boundary(
                    interval=interval_b,
                    boundary_to_update=local_end_ts,
                    update_value=interval_a[local_end_ts],
                )
                resolved_series = merge_metric_columns_of_intervals(
                    main_interval=updated_series,
                    child_interval=interval_a,
                    metric_columns=metric_columns,
                    metric_merge_method=True,
                )

                resolved_intervals.append(resolved_series)

                # 3)
                resolved_series = update_interval_boundary(
                    interval=interval_b,
                    boundary_to_update=local_start_ts,
                    update_value=interval_a[local_end_ts],
                )

                resolved_intervals.append(resolved_series)

                return resolved_intervals

        # pd_df_1 = resolve_overlap(pd_series_1, pd_series_2)
        # print("pd_df_1:", pd_df)
        # pd_df_2 = resolve_overlap(pd_series_2, pd_series_1)
        # print("pd_df_1:", pd_df)

        def resolve_all_overlaps(
                with_row: pd.Series,
                overlaps: pd.DataFrame,
                series_ids: Iterable[str] = None,
                metric_columns: Iterable[str] = None,
        ) -> pd.DataFrame:
            """
            resolve the interval `x` against all overlapping intervals in `overlapping`,
            returning a set of disjoint intervals with the same spans
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
            """

            local_disjoint_df = None

            if series_ids is None:
                series_ids = local_series_ids

            if metric_columns is None:
                metric_columns = local_metric_columns

            for _, row in overlaps.iterrows():
                resolved_intervals = resolve_overlap(with_row, row, series_ids, metric_columns)
                if local_disjoint_df is None:
                    local_disjoint_df = pd.DataFrame(resolved_intervals)
                    continue
                for interval in resolved_intervals:
                    local_disjoint_df = add_as_disjoint(interval, local_disjoint_df)

            return local_disjoint_df

        def add_as_disjoint(
                interval: pd.Series,
                disjoint_set: pd.DataFrame,
                interval_boundaries: list[str] = None,
        ) -> pd.DataFrame:
            """
            returns a disjoint set consisting of the given interval, made disjoint with those already in `disjoint_set`
            """
            print("interval", "-----", interval, sep="\n")
            if interval_boundaries is None:
                interval_boundaries = [local_start_ts, local_end_ts]

            if disjoint_set is None:
                return pd.DataFrame([interval])

            if disjoint_set.empty:
                return pd.DataFrame([interval])

            # if there are values in disjoint_set, resolve the interval against all the values present in disjoint_set
            # and replace disjoint_set with the result
            overlapping_subset_df = identify_overlaps(
                in_pdf=disjoint_set,
                with_row=interval,
            )
            print("overlapping_subset_df", "-----", overlapping_subset_df, sep="\n")

            non_overlapping_subset_df = disjoint_set[
                ~disjoint_set.set_index(interval_boundaries).index.isin(
                    overlapping_subset_df.set_index(interval_boundaries).index
                )
            ]  # TODO: PULL INTO A FUNCTION
            print("non_overlapping_subset_df", "-----", non_overlapping_subset_df, sep="\n")

            # if overlapping_subset_df.empty and not non_overlapping_subset_df.empty:
            #     return pd.concat((disjoint_set, non_overlapping_subset_df))

            if overlapping_subset_df.empty:
                element_wise_comparison = disjoint_set.fillna("¯\\_(ツ)_/¯") == interval.fillna("¯\\_(ツ)_/¯").values
                print("elementwise_comparison", "-----", element_wise_comparison, sep="\n")
                row_wise_comparison = element_wise_comparison.all(axis=1)
                print("rowwise_comparison", "-----", row_wise_comparison, sep="\n")
                if row_wise_comparison.any():
                    return disjoint_set
                else:
                    return pd.concat((disjoint_set, pd.DataFrame([interval])))

            multiple_to_resolve = len(overlapping_subset_df.index) > 1
            only_overlaps_present = len(disjoint_set.index) == len(
                overlapping_subset_df.index
            )

            # If every record overlaps, no need to check and handle non-overlaps
            if not multiple_to_resolve and only_overlaps_present:
                return pd.DataFrame(
                    resolve_overlap(interval, overlapping_subset_df.iloc[0])
                )

            if multiple_to_resolve and only_overlaps_present:
                return resolve_all_overlaps(interval, overlapping_subset_df)

            if not multiple_to_resolve and not only_overlaps_present:
                return pd.concat(
                    (
                        pd.DataFrame(
                            resolve_overlap(interval, overlapping_subset_df.iloc[0])
                        ),
                        non_overlapping_subset_df,
                    ),
                )

            if multiple_to_resolve and not only_overlaps_present:
                return pd.concat(
                    (
                        resolve_all_overlaps(interval, overlapping_subset_df),
                        non_overlapping_subset_df,
                    ),
                )

        def make_disjoint_inner(pdf: pd.DataFrame) -> pd.DataFrame:
            """
            function will process all intervals in the input, and break down overlapping intervals into a fully disjoint set
            https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-and-then-filling-it
            https://stackoverflow.com/questions/55478191/list-of-series-to-dataframe
            https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.append.html
            """

            global_disjoint_df = pd.DataFrame(columns=pdf.columns)

            sorted_pdf = pdf.sort_values([local_start_ts, local_end_ts])

            for _, row in sorted_pdf.iterrows():
                global_disjoint_df = add_as_disjoint(row, global_disjoint_df)

            return global_disjoint_df

        disjoint_df = local_df.groupby(self.series_ids).applyInPandas(
                func=make_disjoint_inner, schema=self.df.schema
            )
        # https://issues.apache.org/jira/browse/SPARK-34544
        # pdf = local_df.toPandas()
        # pd_df = cast(pd.DataFrame, pdf)
        # final_result = make_disjoint_inner(pd_df)

        # return final_result

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
            self.df.unionByName(other.df), self.start_ts, self.end_ts, self.series_ids,
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

