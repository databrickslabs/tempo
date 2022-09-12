from __future__ import annotations

from typing import Optional

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as f
from pyspark.sql.window import Window


class IntervalsDF:
    def __init__(
        self,
        df: DataFrame,
        start_ts: str,
        end_ts: str,
        series: list[str],
        metrics: list[str],
    ):
        # TODO: validate data types
        # TODO: validate cols exist
        #  wait until refactor (v0.2 to see if this is exposed in meaningful way from TSDF)
        self.start_ts = start_ts
        self.end_ts = end_ts

        self.interval_boundaries = [
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
            *self.interval_boundaries
        )

        self.df = df

    @classmethod
    def fromStackedMetrics(
        cls,
        df: DataFrame,
        start_ts: str,
        end_ts: str,
        identifiers: list[str],
        metrics_name_col: str,
        metrics_value_col: str,
    ):
        if not isinstance(identifiers, list):
            raise ValueError

        df = (
            df.groupBy(start_ts, end_ts, *identifiers)
            .pivot(metrics_name_col)
            .max(metrics_value_col)
        )

        metric_cols = list(
            col for col in df.columns if col not in (start_ts, end_ts, *identifiers)
        )

        return cls(df, start_ts, end_ts, identifiers, metric_cols)

    def __get_adjacent_rows(self, df: DataFrame) -> DataFrame:

        for c in self.interval_boundaries + self.metric_cols:
            df = df.withColumn(
                f"_lead_1_{c}",
                f.lead(c, 1).over(self._window),
            ).withColumn(
                f"_lag_1_{c}",
                f.lag(c, 1).over(self._window),
            )

        return df

    def __identify_subset_intervals(self, df: DataFrame) -> tuple[DataFrame, str]:

        subset_indicator = "_lag_1_is_subset"

        df = df.withColumn(
            subset_indicator,
            (f.col(f"_lag_1_{self.start_ts}") <= f.col(self.start_ts))
            & (f.col(f"_lag_1_{self.end_ts}") >= f.col(self.end_ts)),
        )

        df = df.fillna(
            False,
            subset=[subset_indicator],
        )

        return df, subset_indicator

    def __identify_overlaps(self, df: DataFrame) -> tuple[DataFrame, list[str]]:

        overlap_indicators: list[str] = []

        # identify overlaps for each interval boundary
        for ts in self.interval_boundaries:
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

        # null values will have no overlap and not be subset
        df = df.fillna(
            False,
            subset=overlap_indicators,
        )

        return df, overlap_indicators

    def __merge_adjacent_subset_and_superset(
        self, df: DataFrame, subset_indicator: Optional[str]
    ) -> DataFrame:

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
    ):

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

        merge_expr = tuple(f.max(c).alias(c) for c in self.metric_cols)

        return df.groupBy(*self.interval_boundaries, *self.series_cols).agg(
            *merge_expr
        )

    def disjoint(self) -> "IntervalsDF":

        df = self.df

        df = self.__get_adjacent_rows(df)

        (df, subset_indicator) = self.__identify_subset_intervals(df)

        subset_df = df.filter(f.col(subset_indicator))

        subset_df = self.__merge_adjacent_subset_and_superset(
            subset_df, subset_indicator
        )

        subset_df = subset_df.select(
            *self.interval_boundaries, *self.series_cols, *self.metric_cols
        )

        non_subset_df = df.filter(~f.col(subset_indicator))

        (non_subset_df, overlap_indicators) = self.__identify_overlaps(non_subset_df)

        overlaps_predicate = " OR ".join(tuple(col for col in overlap_indicators))

        overlaps_df = non_subset_df.filter(overlaps_predicate)

        disjoint_predicate = f"NOT({overlaps_predicate})"

        # extract intervals that are already disjoint
        disjoint_df = non_subset_df.filter(disjoint_predicate).select(
            *self.interval_boundaries, *self.series_cols, *self.metric_cols
        )

        left_overlaps_df = self.__merge_adjacent_overlaps(
            overlaps_df, "left", overlap_indicators
        )

        left_overlaps_df = left_overlaps_df.select(
            *self.interval_boundaries, *self.series_cols, *self.metric_cols
        )

        right_overlaps_df = self.__merge_adjacent_overlaps(
            overlaps_df, "right", overlap_indicators
        )

        right_overlaps_df = right_overlaps_df.select(
            *self.interval_boundaries, *self.series_cols, *self.metric_cols
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

        if stack:

            n_cols = len(self.metric_cols)
            metric_cols_expr = ",".join(
                tuple(f"'{col}', {col}" for col in self.metric_cols)
            )

            stack_expr = (
                f"STACK({n_cols}, {metric_cols_expr}) AS (metric_name, metric_value)"
            )

            return self.df.select(
                *self.interval_boundaries, *self.series_cols, f.expr(stack_expr)
            ).dropna(subset="metric_value")

        else:
            return self.df
