from __future__ import annotations

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as f
from pyspark.sql.window import Window


class IntervalsDF:
    def __init__(
        self,
        df: DataFrame,
        start_ts: str,
        end_ts: str,
        identifiers: list[str],
        series: list[str],
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

        if not identifiers or not isinstance(identifiers, list):
            raise ValueError
        else:
            self.identifiers = identifiers

        if not series or not isinstance(series, list):
            raise ValueError
        else:
            self.series_ids = series

        self._window = Window.partitionBy(*self.identifiers).orderBy(
            *self.interval_boundaries
        )

        self.df = df

    @classmethod
    def fromStackedSeries(
        cls,
        df: DataFrame,
        start_ts: str,
        end_ts: str,
        identifiers: list[str],
        series_name_col: str,
        series_value_col: str,
    ):
        if not isinstance(identifiers, list):
            raise ValueError

        df = (
            df.groupBy(start_ts, end_ts, *identifiers)
            .pivot(series_name_col)
            .max(series_value_col)
        )

        series_ids = list(
            col for col in df.columns if col not in (start_ts, end_ts, *identifiers)
        )

        return cls(df, start_ts, end_ts, identifiers, series_ids)

    def __get_adjacent_rows(self, df: DataFrame) -> DataFrame:
        for c in self.interval_boundaries + self.series_ids:
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

        return_df = df.fillna(
            False,
            subset=[subset_indicator],
        )

        return return_df, subset_indicator

    def __identify_overlap_intervals(
        self, df: DataFrame
    ) -> tuple[DataFrame, list[str]]:

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
        return_df = df.fillna(
            False,
            subset=overlap_indicators,
        )

        return return_df, overlap_indicators

    def __merge_adjacent_subset_and_superset(
        self, df: DataFrame, subset_indicator: str
    ) -> DataFrame:

        for c in self.series_ids:
            df = df.withColumn(
                c,
                f.when(
                    f.col(subset_indicator), f.coalesce(f.col(c), f"_lag_1_{c}")
                ).otherwise(f.col(c)),
            )

        return df

    def __merge_adjacent_overlaps(self, df: DataFrame, how: str):

        if not (how == "left" or how == "right"):
            raise ValueError

        if how == "left":
            # create left section for disjoint intervals
            left_overlaps_df = df

            left_overlaps_df = left_overlaps_df.withColumn(
                "end_ts",
                (
                    f.when(
                        f.col(f"_lead_1_{self.start_ts}_overlaps")
                        & f.col(f"_lead_1_{self.end_ts}_overlaps"),
                        f.col(f"_lead_1_{self.start_ts}"),
                    )
                    .when(
                        f.col(f"_lead_1_{self.start_ts}_overlaps"),
                        f.col(f"_lead_1_{self.start_ts}"),
                    )
                    .when(
                        f.col(f"_lead_1_{self.end_ts}_overlaps"),
                        f.col(f"_lead_1_{self.end_ts}"),
                    )
                    .when(
                        f.col(f"_lag_1_{self.start_ts}_overlaps"),
                        f.col(f"_lag_1_{self.start_ts}"),
                    )
                    .when(
                        f.col(f"_lag_1_{self.end_ts}_overlaps"),
                        f.col(f"_lag_1_{self.end_ts}"),
                    )
                    .otherwise(None)
                ),
            )

            for c in self.series_ids:
                left_overlaps_df = left_overlaps_df.withColumn(
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

            return left_overlaps_df

        else:
            # create right section for disjoint intervals
            right_overlaps_df = df

            right_overlaps_df = right_overlaps_df.withColumn(
                f"{self.start_ts}",
                (
                    f.when(
                        f.col(f"_lead_1_{self.start_ts}_overlaps")
                        & f.col(f"_lead_1_{self.end_ts}_overlaps"),
                        f.col(f"_lead_1_{self.end_ts}"),
                    )
                    .when(
                        f.col(f"_lead_1_{self.start_ts}_overlaps"),
                        f.col(f"_lead_1_{self.start_ts}"),
                    )
                    .when(
                        f.col(f"_lead_1_{self.end_ts}_overlaps"),
                        f.col(f"_lead_1_{self.end_ts}"),
                    )
                    .when(
                        f.col(f"_lag_1_{self.start_ts}_overlaps"),
                        f.col(f"_lag_1_{self.start_ts}"),
                    )
                    .when(
                        f.col(f"_lag_1_{self.end_ts}_overlaps"),
                        f.col(f"_lag_1_{self.end_ts}"),
                    )
                    .otherwise(None)
                ),
            )

            for c in self.series_ids:
                right_overlaps_df = right_overlaps_df.withColumn(
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

            return right_overlaps_df

    def __merge_equal_intervals(self, df: DataFrame) -> DataFrame:

        merge_expr = tuple(
            f.max(c).alias(c) for c in self.series_ids
        )

        return_df = df.groupBy(*self.interval_boundaries, *self.identifiers).agg(
            *merge_expr
        )

        return return_df

    def disjoint(self) -> "IntervalsDF":

        df = self.df

        df = self.__get_adjacent_rows(df)

        (df, subset_indicator) = self.__identify_subset_intervals(df)

        subset_df = df.filter(f.col(subset_indicator))

        subset_df = self.__merge_adjacent_subset_and_superset(
            subset_df, subset_indicator
        )

        subset_df = subset_df.select(
            *self.interval_boundaries, *self.identifiers, *self.series_ids
        )

        non_subset_df = df.filter(~f.col(subset_indicator))

        (non_subset_df, overlap_indicators) = self.__identify_overlap_intervals(
            non_subset_df
        )

        overlaps_predicate = " OR ".join(tuple(col for col in overlap_indicators))

        overlaps_df = non_subset_df.filter(overlaps_predicate)

        disjoint_predicate = f"NOT({overlaps_predicate})"

        # extract intervals that are already disjoint
        disjoint_df = non_subset_df.filter(disjoint_predicate).select(
            *self.interval_boundaries, *self.identifiers, *self.series_ids
        )

        left_overlaps_df = self.__merge_adjacent_overlaps(overlaps_df, "left")

        left_overlaps_df = left_overlaps_df.select(
            *self.interval_boundaries, *self.identifiers, *self.series_ids
        )

        right_overlaps_df = self.__merge_adjacent_overlaps(overlaps_df, "right")

        right_overlaps_df = right_overlaps_df.select(
            *self.interval_boundaries, *self.identifiers, *self.series_ids
        )

        unioned_df = (
            subset_df.unionByName(disjoint_df)
            .unionByName(left_overlaps_df)
            .unionByName(right_overlaps_df)
        )

        return_df = self.__merge_equal_intervals(unioned_df)

        return IntervalsDF(
            return_df,
            self.start_ts,
            self.end_ts,
            self.identifiers,
            self.series_ids,
        )

    def union(self, other: "IntervalsDF") -> "IntervalsDF":

        if not isinstance(other, IntervalsDF):
            raise TypeError

        return IntervalsDF(
            self.df.union(other.df),
            self.start_ts,
            self.end_ts,
            self.identifiers,
            self.series_ids,
        )

    def unionByName(self, other: "IntervalsDF") -> "IntervalsDF":

        if not isinstance(other, IntervalsDF):
            raise TypeError

        return IntervalsDF(
            self.df.unionByName(other.df),
            self.start_ts,
            self.end_ts,
            self.identifiers,
            self.series_ids,
        )

    def toDF(self, stack: bool = False) -> DataFrame:

        write_df = self.df

        if stack:

            n_cols = len(self.series_ids)
            series_cols_expr = ",".join(
                tuple(f"'{col}', {col}" for col in self.series_ids)
            )

            stack_expr = (
                f"STACK({n_cols}, {series_cols_expr}) AS (series_name, series_value)"
            )

            return write_df.select(
                *self.interval_boundaries, *self.identifiers, f.expr(stack_expr)
            ).dropna(subset="series_value")

        else:
            return write_df
