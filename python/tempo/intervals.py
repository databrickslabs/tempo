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
            series: list[str] = None,
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

        if series is None:
            self.series_ids = series
        elif not series or not isinstance(series, list):
            raise ValueError
        else:
            self.series_ids = series

        self._window = Window.partitionBy(*self.identifiers).orderBy(
            *self.interval_boundaries
        )

        self.df = df

        for c in self.interval_boundaries + self.series_ids:
            self.df = self.df.withColumn(
                f"_lead_1_{c}",
                f.lead(c, 1).over(self._window),
            ).withColumn(
                f"_lag_1_{c}",
                f.lag(c, 1).over(self._window),
            )

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

    def _identify_intersecting_boundaries(self) -> [DataFrame, list[str]]:

        df = self.df
        intersect_indicators = []

        # identify overlaps for each interval boundary
        for ts in self.interval_boundaries:
            df = df.withColumn(
                f"_lead_1_{ts}_intersects",
                (f.col(f"_lead_1_{ts}") > f.col(self.start_ts))
                & (f.col(f"_lead_1_{ts}") < f.col(self.end_ts)),
            ).withColumn(
                f"_lag_1_{ts}_intersects",
                (f.col(f"_lag_1_{ts}") > f.col(self.start_ts))
                & (f.col(f"_lag_1_{ts}") < f.col(self.end_ts)),
            )

            intersect_indicators.extend(
                (
                    f"_lead_1_{ts}_intersects",
                    f"_lag_1_{ts}_intersects",
                )
            )

        # null values will have no overlap
        df = df.fillna(
            False,
            subset=[*intersect_indicators],
        )

        return_df = df

        return df, intersect_indicators

    def subset(self) -> "IntervalsDF":
        ...

    def equal(self) -> "IntervalsDF":
        ...

    def equivalent(self) -> "IntervalsDF":
        ...

    def disjoint(self) -> "IntervalsDF":

        df, intersect_indicators = self._identify_intersecting_boundaries()

        non_disjoint_predicate = " OR ".join(
            tuple(col for col in intersect_indicators)
        )

        non_disjoint_intervals = df.filter(non_disjoint_predicate)

        disjoint_predicate = f"NOT({non_disjoint_predicate})"

        # extract intervals that are already disjoint
        disjoint_intervals_df = df.filter(disjoint_predicate)

        # use prev/next timestamps to construct new interval boundaries
        # TODO : this is the problem for subset test; we don't reach ever `when`
        #  if both start and end are within the interval (ie is subset)
        interval_boundary_logic = (
            f.when(
                f.col(f"_lead_1_{self.start_ts}_intersects"),
                f.col(f"_lead_1_{self.start_ts}"),
            )
            .when(
                f.col(f"_lead_1_{self.end_ts}_intersects"),
                f.col(f"_lead_1_{self.end_ts}"),
            )
            .when(
                f.col(f"_lag_1_{self.start_ts}_intersects"),
                f.col(f"_lag_1_{self.start_ts}"),
            )
            .when(
                f.col(f"_lag_1_{self.end_ts}_intersects"),
                f.col(f"_lag_1_{self.end_ts}"),
            )
            .otherwise(None)
        )

        # create left section for disjoint intervals
        left_interval_sections_df = non_disjoint_intervals

        left_interval_sections_df = left_interval_sections_df.withColumn(
            f"end_ts",
            interval_boundary_logic,
        )

        for c in self.series_ids:
            left_interval_sections_df = left_interval_sections_df.withColumn(
                f"{c}",
                f.when(
                    f.col(f"_lead_1_{self.start_ts}_intersects"),
                    f.col(c),
                )
                .when(
                    f.col(f"_lead_1_{self.end_ts}_intersects"),
                    f.coalesce(f.col(c), f.col(f"_lead_1_{c}")),
                )
                .when(
                    f.col(f"_lag_1_{self.start_ts}_intersects"),
                    f.col(c),
                )
                .when(
                    f.col(f"_lag_1_{self.end_ts}_intersects"),
                    f.coalesce(f.col(c), f.col(f"_lag_1_{c}")),
                ),
            )

        # create right section for disjoint intervals
        right_interval_sections_df = non_disjoint_intervals

        right_interval_sections_df = right_interval_sections_df.withColumn(
            f"{self.start_ts}",
            interval_boundary_logic,
        )

        for c in self.series_ids:
            right_interval_sections_df = right_interval_sections_df.withColumn(
                f"{c}",
                f.when(
                    f.col(f"_lead_1_{self.start_ts}_intersects"),
                    f.coalesce(f.col(c), f.col(f"_lead_1_{c}")),
                )
                .when(f.col(f"_lead_1_{self.end_ts}_intersects"), f.col(c))
                .when(
                    f.col(f"_lag_1_{self.start_ts}_intersects"),
                    f.coalesce(f.col(c), f.col(f"_lag_1_{c}")),
                )
                .when(f.col(f"_lag_1_{self.end_ts}_intersects"), f.col(c)),
            )

        equal_intervals_aggregator_expr = tuple(
            f.max(c).alias(c) for c in self.series_ids
        )

        final_disjoint_df = (
            (
                disjoint_intervals_df.unionByName(
                    left_interval_sections_df
                ).unionByName(right_interval_sections_df)
            )
            .groupBy(*self.interval_boundaries, *self.identifiers)
            .agg(*equal_intervals_aggregator_expr)
        )

        print("df")
        df.show(truncate=False)
        print("disjoint_intervals_df")
        disjoint_intervals_df.show(truncate=False)
        print("left_interval_sections_df")
        left_interval_sections_df.show(truncate=False)
        print("right_interval_sections_df")
        right_interval_sections_df.show(truncate=False)

        return_df = final_disjoint_df

        print("final_disjoint_df")
        final_disjoint_df.show()

        return IntervalsDF(
            return_df,
            self.start_ts,
            self.end_ts,
            self.identifiers,
            self.series_ids,
        )

    def overlap(self) -> "IntervalsDF":
        ...

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
        if stack:

            n_cols = len(self.series_ids)
            series_cols_expr = ",".join(
                tuple(f"'{col}', {col}" for col in self.series_ids)
            )
            other_cols_expr = tuple(
                col for col in self.df.columns if col not in self.series_ids
            )

            stack_expr = (
                f"STACK({n_cols}, {series_cols_expr}) AS (series_name, series_value)"
            )

            return self.df.select(*other_cols_expr, f.expr(stack_expr)).dropna(
                subset="series_value"
            )

        else:
            return self.df
