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
        # TODO: should we convert start_ts & end_ts to timestamp or double type?
        self.df = df
        # TODO: validate cols exist
        #  wait until refactor (v0.2 to see if this is exposed in meaningful way from TSDF)
        self.start_ts = start_ts
        self.end_ts = end_ts

        self.interval_boundaries = (
            self.start_ts,
            self.end_ts,
        )

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

    def equals(self) -> "IntervalsDF":
        ...

    def disjoint(self) -> "IntervalsDF":

        df = self.df

        next_prefix = "next"
        prev_prefix = "prev"

        # create list to store boolean columns indicating overlap
        overlap_bool_indicators = []

        # get previous and next interval timestamps to use for checking overlaps
        for ts in self.interval_boundaries:
            df = df.withColumn(
                f"{next_prefix}_{ts}",
                f.lead(ts, 1).over(self._window),
            ).withColumn(
                f"{prev_prefix}_{ts}",
                f.lag(ts, 1).over(self._window),
            )

            # identify overlaps for each interval boundary
            df = df.withColumn(
                f"{next_prefix}_{ts}_has_overlap",
                (f.col(f"{next_prefix}_{ts}") > f.col(self.start_ts))
                & (f.col(f"{next_prefix}_{ts}") < f.col(self.end_ts)),
            ).withColumn(
                f"{prev_prefix}_{ts}_has_overlap",
                (f.col(f"{prev_prefix}_{ts}") > f.col(self.start_ts))
                & (f.col(f"{prev_prefix}_{ts}") < f.col(self.end_ts)),
            )

            overlap_bool_indicators.extend(
                (
                    f"{next_prefix}_{ts}_has_overlap",
                    f"{prev_prefix}_{ts}_has_overlap",
                )
            )

        # null values will have no overlap
        df = df.fillna(
            False,
            subset=[*overlap_bool_indicators],
        )

        non_disjoint_predicate = " OR ".join(
            tuple(col for col in overlap_bool_indicators)
        )

        non_disjoint_intervals = df.filter(non_disjoint_predicate)

        disjoint_predicate = f"NOT({non_disjoint_predicate})"

        # extract intervals that are already disjoint

        disjoint_intervals_df = df.filter(disjoint_predicate).drop(
            *tuple(
                col
                for col in df.columns
                if col.startswith(next_prefix) or col.startswith(prev_prefix)
            )
        )

        # get previous and next series per interval for constructing disjoint intervals
        for c in self.series_ids:
            non_disjoint_intervals = non_disjoint_intervals.withColumn(
                f"{prev_prefix}_{c}",
                f.when(
                    f.col(f"{prev_prefix}_{self.start_ts}_has_overlap")
                    | f.col(f"{prev_prefix}_{self.end_ts}_has_overlap"),
                    f.lag(c, 1).over(self._window),
                ).otherwise(None),
            ).withColumn(
                f"{next_prefix}_{c}",
                f.when(
                    f.col(f"{next_prefix}_{self.start_ts}_has_overlap")
                    | f.col(f"{next_prefix}_{self.end_ts}_has_overlap"),
                    f.lead(c, 1).over(self._window),
                ).otherwise(None),
            )

        # use prev/next timestamps to construct new interval boundaries
        interval_boundary_logic = (
            f.when(
                f.col(f"{next_prefix}_{self.start_ts}_has_overlap"),
                f.col(f"{next_prefix}_{self.start_ts}"),
            )
            .when(
                f.col(f"{next_prefix}_{self.end_ts}_has_overlap"),
                f.col(f"{next_prefix}_{self.end_ts}"),
            )
            .when(
                f.col(f"{prev_prefix}_{self.start_ts}_has_overlap"),
                f.col(f"{prev_prefix}_{self.start_ts}"),
            )
            .when(
                f.col(f"{prev_prefix}_{self.end_ts}_has_overlap"),
                f.col(f"{prev_prefix}_{self.end_ts}"),
            )
            .otherwise(None)
        )

        # create left section for disjoint intervals

        left_prefix = "left"
        left_interval_sections_df = non_disjoint_intervals

        left_interval_sections_df = left_interval_sections_df.withColumn(
            f"{left_prefix}_end_ts",
            interval_boundary_logic,
        )

        for c in self.series_ids:
            left_interval_sections_df = left_interval_sections_df.withColumn(
                f"{left_prefix}_{c}",
                f.when(
                    f.col(f"{next_prefix}_{self.start_ts}_has_overlap"),
                    f.col(c),
                )
                .when(
                    f.col(f"{next_prefix}_{self.end_ts}_has_overlap"),
                    f.coalesce(f.col(c), f.col(f"{next_prefix}_{c}")),
                )
                .when(
                    f.col(f"{prev_prefix}_{self.start_ts}_has_overlap"),
                    f.col(c),
                )
                .when(
                    f.col(f"{prev_prefix}_{self.end_ts}_has_overlap"),
                    f.coalesce(f.col(c), f.col(f"{prev_prefix}_{c}")),
                ),
            )

        left_cols = left_interval_sections_df.columns

        left_interval_sections_df = left_interval_sections_df.select(
            "start_ts",
            *self.identifiers,
            *tuple(
                f.col(c).alias(c.replace(f"{left_prefix}_", ""))
                for c in left_cols
                if c.startswith(left_prefix)
            ),
        )

        # create right section for disjoint intervals

        right_prefix = "right"
        right_interval_sections_df = non_disjoint_intervals

        right_interval_sections_df = right_interval_sections_df.withColumn(
            f"{right_prefix}_{self.start_ts}",
            interval_boundary_logic,
        )

        for c in self.series_ids:
            right_interval_sections_df = right_interval_sections_df.withColumn(
                f"{right_prefix}_{c}",
                f.when(
                    f.col(f"{next_prefix}_{self.start_ts}_has_overlap"),
                    f.coalesce(f.col(c), f.col(f"{next_prefix}_{c}")),
                )
                .when(f.col(f"{next_prefix}_{self.end_ts}_has_overlap"), f.col(c))
                .when(
                    f.col(f"{prev_prefix}_{self.start_ts}_has_overlap"),
                    f.coalesce(f.col(c), f.col(f"{prev_prefix}_{c}")),
                )
                .when(f.col(f"{prev_prefix}_{self.end_ts}_has_overlap"), f.col(c)),
            )

        right_cols = right_interval_sections_df.columns

        right_interval_sections_df = right_interval_sections_df.select(
            "end_ts",
            *self.identifiers,
            *tuple(
                f.col(c).alias(c.replace(f"{right_prefix}_", ""))
                for c in right_cols
                if c.startswith(right_prefix)
            ),
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

        return_df = final_disjoint_df

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
