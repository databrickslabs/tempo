from __future__ import annotations

from typing import Sequence

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as f
from pyspark.sql.window import Window


class IntervalsDF:
    def __init__(
        self,
        df: DataFrame,
        start_ts: str,
        end_ts: str,
        identifiers: [str | Sequence[str]],
        series: [str | Sequence[str]] = None,
    ):
        # TODO: validate data types
        # TODO: should we convert start_ts & end_ts to timestamp or double type?
        self.df = df
        # TODO: validate cols exist
        #  wait until refactor (v0.2 to see if this is exposed in meaningful way from TSDF)
        self.start_ts = start_ts
        self.end_ts = end_ts

        if not identifiers:
            raise ValueError
        elif isinstance(identifiers, str):
            self.identifiers = (identifiers,)
        else:
            self.identifiers = identifiers

        if not series:
            if series is None:
                self.series_ids = series
            else:
                raise ValueError
        elif isinstance(series, str):
            self.series_ids = (series,)
        else:
            self.series_ids = series

        self._window = Window.partitionBy(*self.identifiers)

    @classmethod
    def fromStackedSeries(
        cls,
        df: DataFrame,
        start_ts: str,
        end_ts: str,
        identifiers: [str | Sequence[str]],
        series_name_col: str,
        series_value_col: str,
    ):
        if isinstance(identifiers, str):
            identifiers = (identifiers,)

        df = (
            df.groupBy(start_ts, end_ts, *identifiers)
            .pivot(series_name_col)
            .max(series_value_col)
        )

        series_ids = tuple(
            col for col in df.columns if col not in (start_ts, end_ts, *identifiers)
        )

        return cls(df, start_ts, end_ts, identifiers, series_ids)

    def disjoint(self) -> "IntervalsDF":

        # TODO : can we change window and use `last`?
        # internal slack thread: https://databricks.slack.com/archives/C0JCGEF17/p1662125947149539
        disjoint_series_window = self._window.orderBy(
            f.col(self.start_ts).desc(), f.col(self.end_ts).desc()
        )
        disjoint_ts_window = self._window.orderBy(self.start_ts, self.end_ts)
        disjoint_prefix = "disjoint_"

        df = self.df

        # TODO: overwrite columns in place here instead or keep for debugging?
        for series_id in self.series_ids:
            df = df.withColumn(
                f"{disjoint_prefix}{series_id}",
                f.coalesce(
                    series_id,
                    f.first(series_id).over(disjoint_series_window),
                ),
            )

        df = df.orderBy(self.start_ts, self.end_ts)

        df = df.withColumn(
            f"{disjoint_prefix}{self.start_ts}",
            f.coalesce(
                f.lag(f.col(self.end_ts), 1).over(disjoint_ts_window),
                self.start_ts,
            ),
        )

        for series_id in self.series_ids:
            df = df.withColumn(series_id, f.col(f"{disjoint_prefix}{series_id}")).drop(
                f"{disjoint_prefix}{series_id}"
            )

        df = df.withColumn(
            self.start_ts, f.col(f"{disjoint_prefix}{self.start_ts}")
        ).drop(f"{disjoint_prefix}{self.start_ts}")

        return IntervalsDF(
            df,
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
