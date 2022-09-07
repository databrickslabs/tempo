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

        self._window = Window.partitionBy(*self.identifiers).orderBy("start_ts", "end_ts")

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

    def disjoint(self) -> "IntervalsDF":

        df = self.df

        ...

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
