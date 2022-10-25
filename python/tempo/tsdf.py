from __future__ import annotations

import logging
import operator
from copy import deepcopy
from functools import reduce, cached_property
from typing import List, Union, Callable, Collection

import numpy as np
import pyspark.sql.functions as Fn
from IPython.core.display import HTML
from IPython.display import display as ipydisplay
from pyspark.sql import SparkSession
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window, WindowSpec
from scipy.fft import fft, fftfreq

import tempo.io as tio
import tempo.resample as rs
from tempo.interpol import Interpolation
from tempo.tsschema import TSIndex, TSSchema, SubSequenceTSIndex
from tempo.utils import (
    ENV_CAN_RENDER_HTML,
    IS_DATABRICKS,
    calculate_time_horizon,
    get_display_df
)

logger = logging.getLogger(__name__)

class TSDFStructureChangeError(Exception):
    """
    Error raised when a user attempts an operation that would alter the structure of a TSDF in a destructive manner.
    """
    __MSG_TEMPLATE: str = """
    The attempted operation ({op}) is not allowed because it would result in altering the structure of the TSDF.
    If you really want to make this change, perform the operation on the underlying DataFrame, then re-create a new TSDF.
    {d}"""

    def __init__(self, operation: str, details: str = None) -> None:
        super().__init__(self.__MSG_TEMPLATE.format(op=operation, d=details))


class IncompatibleTSError(Exception):
    """
    Error raised when an operation is attempted between two incompatible TSDFs.
    """
    __MSG_TEMPLATE: str = """
    The attempted operation ({op}) cannot be performed because the given TSDFs have incompatible structure.
    {d}"""

    def __init__(self, operation: str, details: str = None) -> None:
        super().__init__(self.__MSG_TEMPLATE.format(op=operation, d=details))

class TSDF:
    """
    This object is the main wrapper over a Spark data frame which allows a user to parallelize time series computations on a Spark data frame by various dimensions. The two dimensions required are partition_cols (list of columns by which to summarize) and ts_col (timestamp column, which can be epoch or TimestampType).
    """

    def __init__(
        self,
        df: DataFrame,
        ts_schema: TSSchema = None,
        ts_col: str = None,
        series_ids: Collection[str] = None,
        validate_schema=True
    ) -> None:
        self.df = df
        # construct schema if we don't already have one
        if ts_schema:
            self.ts_schema = ts_schema
        else:
            self.ts_schema = TSSchema.fromDFSchema(self.df.schema, ts_col, series_ids)
        # validate that this schema works for this DataFrame
        if validate_schema:
            self.ts_schema.validate(df.schema)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"""TSDF({id(self)}):
    TS Index: {self.ts_index}
    Series IDs: {self.series_ids}
    Observational Cols: {self.observational_cols}
    DataFrame: {self.df.schema}"""

    def __withTransformedDF(self, new_df: DataFrame) -> "TSDF":
        """
        This helper function will create a new :class:`TSDF` using the current schema, but a new / transformed :class:`DataFrame`

        :param new_df: the new / transformed :class:`DataFrame` to

        :return: a new TSDF object with the transformed DataFrame
        """
        return TSDF(new_df, ts_schema=deepcopy(self.ts_schema), validate_schema=False)

    def __withStandardizedColOrder(self) -> TSDF:
        """
        Standardizes the column ordering as such:
        * series_ids,
        * ts_index,
        * observation columns

        :return: a :class:`TSDF` with the columns reordered into "standard order" (as described above)
        """
        std_ordered_cols = list(self.series_ids) + [self.ts_index.name] + list(self.observational_cols)

        return self.__withTransformedDF(self.df.select(std_ordered_cols))

    @classmethod
    def __makeStructFromCols(cls, df: DataFrame, struct_col_name: str, cols_to_move: List[str]) -> DataFrame:
        """
        Transform a :class:`DataFrame` by moving certain columns into a struct

        :param df: the :class:`DataFrame` to transform
        :param struct_col_name: name of the struct column to create
        :param cols_to_move: name of the columns to move into the struct

        :return: the transformed :class:`DataFrame`
        """
        return df.withColumn(struct_col_name, Fn.struct(cols_to_move)).drop(*cols_to_move)

    # default column name for constructed timeseries index struct columns
    __DEFAULT_TS_IDX_COL = "ts_idx"

    @classmethod
    def fromSubsequenceCol(cls, df: DataFrame, ts_col: str, subsequence_col: str, series_ids: Collection[str] = None) -> "TSDF":
        # construct a struct with the ts_col and subsequence_col
        struct_col_name = cls.__DEFAULT_TS_IDX_COL
        with_subseq_struct_df = cls.__makeStructFromCols(df, struct_col_name, [ts_col, subsequence_col])
        # construct an appropriate TSIndex
        subseq_struct = with_subseq_struct_df.schema[struct_col_name]
        subseq_idx = SubSequenceTSIndex(subseq_struct, ts_col, subsequence_col)
        # construct & return the TSDF with appropriate schema
        return TSDF(with_subseq_struct_df, ts_schema=TSSchema(subseq_idx, series_ids))


    @classmethod
    def fromTimestampString(cls, df: DataFrame, ts_col: str, series_ids: Collection[str] = None, ts_fmt: str = "YYYY-MM-DDThh:mm:ss[.SSSSSS]") -> "TSDF":
        pass

    @classmethod
    def fromDateString(cls, df: DataFrame, ts_col: str, series_ids: Collection[str], date_fmt: str = "YYYY-MM-DD") -> "TSDF ":
        pass

    @property
    def ts_index(self) -> "TSIndex":
        return self.ts_schema.ts_idx

    @property
    def ts_col(self) -> str:
        return self.ts_index.ts_col

    @property
    def columns(self) -> List[str]:
        return self.df.columns

    @property
    def series_ids(self) -> List[str]:
        return self.ts_schema.series_ids

    @property
    def structural_cols(self) -> List[str]:
        return self.ts_schema.structural_columns

    @cached_property
    def observational_cols(self) -> List[str]:
        return self.ts_schema.find_observational_columns(self.df.schema)

    @cached_property
    def metric_cols(self) -> List[str]:
        return self.ts_schema.find_metric_columns(self.df.schema)

    #
    # Helper functions
    #

    # def __add_double_ts(self):
    #     """Add a double (epoch) version of the string timestamp out to nanos"""
    #     self.df = (
    #         self.df.withColumn(
    #             "nanos",
    #             (
    #                 Fn.when(
    #                     Fn.col(self.ts_col).contains("."),
    #                     Fn.concat(Fn.lit("0."), Fn.split(Fn.col(self.ts_col), "\.")[1]),
    #                 ).otherwise(0)
    #             ).cast("double"),
    #         )
    #         .withColumn("long_ts", Fn.col(self.ts_col).cast("timestamp").cast("long"))
    #         .withColumn("double_ts", Fn.col("long_ts") + Fn.col("nanos"))
    #         .drop("nanos")
    #         .drop("long_ts")
    #     )

    # def __validate_ts_string(self, ts_text):
    #     """Validate the format for the string using Regex matching for ts_string"""
    #     import re
    #
    #     ts_pattern = "^\d{4}-\d{2}-\d{2}T| \d{2}:\d{2}:\d{2}\.\d*$"
    #     if re.match(ts_pattern, ts_text) is None:
    #         raise ValueError(
    #             "Incorrect data format, should be YYYY-MM-DD HH:MM:SS[.nnnnnnnn]"
    #         )

    # def __validated_column(self, df, colname):
    #     if type(colname) != str:
    #         raise TypeError(
    #             f"Column names must be of type str; found {type(colname)} instead!"
    #         )
    #     if colname.lower() not in [col.lower() for col in df.columns]:
    #         raise ValueError(f"Column {colname} not found in Dataframe")
    #     return colname

    # def __validated_columns(self, df, colnames):
    #     # if provided a string, treat it as a single column
    #     if type(colnames) == str:
    #         colnames = [colnames]
    #     # otherwise we really should have a list or None
    #     if colnames is None:
    #         colnames = []
    #     elif type(colnames) != list:
    #         raise TypeError(
    #             f"Columns must be of type list, str, or None; found {type(colnames)} instead!"
    #         )
    #     # validate each column
    #     for col in colnames:
    #         self.__validated_column(df, col)
    #     return colnames

    def __checkPartitionCols(self, tsdf_right):
        for left_col, right_col in zip(self.series_ids, tsdf_right.series_ids):
            if left_col != right_col:
                raise ValueError(
                    "left and right dataframe partition columns should have same name in same order"
                )

    def __validateTsColMatch(self, right_tsdf):
        # TODO - can simplify this to get types from schema object
        left_ts_datatype = self.df.select(self.ts_col).dtypes[0][1]
        right_ts_datatype = right_tsdf.df.select(right_tsdf.ts_col).dtypes[0][1]
        if left_ts_datatype != right_ts_datatype:
            raise ValueError(
                "left and right dataframe timestamp index columns should have same type"
            )

    def __addPrefixToColumns(self, col_list, prefix):
        """
        Add prefix to all specified columns.
        """
        if prefix != "":
            prefix = prefix + "_"

        df = reduce(
            lambda df, idx: df.withColumnRenamed(
                col_list[idx], "".join([prefix, col_list[idx]])
            ),
            range(len(col_list)),
            self.df,
        )

        if prefix == "":
            ts_col = self.ts_col
        else:
            ts_col = "".join([prefix, self.ts_col])

        return TSDF(df, ts_col=ts_col, series_ids=self.series_ids)

    def __addColumnsFromOtherDF(self, other_cols):
        """
        Add columns from some other DF as lit(None), as pre-step before union.
        """
        new_df = reduce(
            lambda df, idx: df.withColumn(other_cols[idx], Fn.lit(None)),
            range(len(other_cols)),
            self.df,
        )

        return self.__withTransformedDF(new_df)

    def __combineTSDF(self, ts_df_right, combined_ts_col):
        combined_df = self.df.unionByName(ts_df_right.df).withColumn(
            combined_ts_col, Fn.coalesce(self.ts_col, ts_df_right.ts_col)
        )

        return TSDF(combined_df, ts_col=combined_ts_col, series_ids=self.series_ids)

    def __getLastRightRow(
        self,
        left_ts_col,
        right_cols,
        sequence_col,
        tsPartitionVal,
        ignoreNulls,
        suppress_null_warning,
    ):
        """Get last right value of each right column (inc. right timestamp) for each self.ts_col value

        self.ts_col, which is the combined time-stamp column of both left and right dataframe, is dropped at the end
        since it is no longer used in subsequent methods.
        """
        ptntl_sort_keys = [self.ts_col, "rec_ind", sequence_col]
        sort_keys = [Fn.col(col_name) for col_name in ptntl_sort_keys if col_name]

        window_spec = (
            Window.partitionBy(self.series_ids)
            .orderBy(sort_keys)
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        if ignoreNulls is False:
            if tsPartitionVal is not None:
                raise ValueError(
                    "Disabling null skipping with a partition value is not supported yet."
                )
            df = reduce(
                lambda df, idx: df.withColumn(
                    right_cols[idx],
                    Fn.last(
                        Fn.when(
                            Fn.col("rec_ind") == -1, Fn.struct(right_cols[idx])
                        ).otherwise(None),
                        True,  # ignore nulls because it indicates rows from the left side
                    ).over(window_spec),
                ),
                range(len(right_cols)),
                self.df,
            )
            df = reduce(
                lambda df, idx: df.withColumn(
                    right_cols[idx], Fn.col(right_cols[idx])[right_cols[idx]]
                ),
                range(len(right_cols)),
                df,
            )
        elif tsPartitionVal is None:
            # splitting off the condition as we want different columns in the reduce if implementing the skew AS OF join
            df = reduce(
                lambda df, idx: df.withColumn(
                    right_cols[idx],
                    Fn.last(right_cols[idx], ignoreNulls).over(window_spec),
                ),
                range(len(right_cols)),
                self.df,
            )
        else:
            df = reduce(
                lambda df, idx: df.withColumn(
                    right_cols[idx],
                    Fn.last(right_cols[idx], ignoreNulls).over(window_spec),
                ).withColumn(
                    "non_null_ct" + right_cols[idx],
                    Fn.count(right_cols[idx]).over(window_spec),
                ),
                range(len(right_cols)),
                self.df,
            )

        df = (df.filter(Fn.col(left_ts_col).isNotNull()).drop(self.ts_col)).drop(
            "rec_ind"
        )

        # remove the null_ct stats used to record missing values in partitioned as of join
        if tsPartitionVal is not None:
            for column in df.columns:
                if column.startswith("non_null"):
                    # Avoid collect() calls when explicitly ignoring the warnings about null values due to lookback window.
                    # if setting suppress_null_warning to True and warning logger is enabled for other part of the code,
                    # it would make sense to not log warning in this function while allowing other part of the code to continue to log warning.
                    # So it makes more sense for and than or on this line
                    if not suppress_null_warning and logger.isEnabledFor(
                        logging.WARNING
                    ):
                        any_blank_vals = df.agg({column: "min"}).collect()[0][0] == 0
                        newCol = column.replace("non_null_ct", "")
                        if any_blank_vals:
                            logger.warning(
                                "Column "
                                + newCol
                                + " had no values within the lookback window. Consider using a larger window to avoid missing values. If this is the first record in the data frame, this warning can be ignored."
                            )
                    df = df.drop(column)

        return TSDF(df, ts_col=left_ts_col, series_ids=self.series_ids)

    def __getTimePartitions(self, tsPartitionVal, fraction=0.1):
        """
        Create time-partitions for our data-set. We put our time-stamps into brackets of <tsPartitionVal>. Timestamps
        are rounded down to the nearest <tsPartitionVal> seconds.

        We cast our timestamp column to double instead of using f.unix_timestamp, since it provides more precision.

        Additionally, we make these partitions overlapping by adding a remainder df. This way when calculating the
        last right timestamp we will not end up with nulls for the first left timestamp in each partition.

        TODO: change ts_partition to accomodate for higher precision than seconds.
        """
        partition_df = (
            self.df.withColumn(
                "ts_col_double", Fn.col(self.ts_col).cast("double")
            )  # double is preferred over unix_timestamp
            .withColumn(
                "ts_partition",
                Fn.lit(tsPartitionVal)
                * (Fn.col("ts_col_double") / Fn.lit(tsPartitionVal)).cast("integer"),
            )
            .withColumn(
                "partition_remainder",
                (Fn.col("ts_col_double") - Fn.col("ts_partition"))
                / Fn.lit(tsPartitionVal),
            )
            .withColumn("is_original", Fn.lit(1))
        ).cache()  # cache it because it's used twice.

        # add [1 - fraction] of previous time partition to the next partition.
        remainder_df = (
            partition_df.filter(Fn.col("partition_remainder") >= Fn.lit(1 - fraction))
            .withColumn("ts_partition", Fn.col("ts_partition") + Fn.lit(tsPartitionVal))
            .withColumn("is_original", Fn.lit(0))
        )

        df = partition_df.union(remainder_df).drop(
            "partition_remainder", "ts_col_double"
        )
        return TSDF(df, ts_col=self.ts_col, series_ids=self.series_ids + ["ts_partition"])

    #
    # Slicing & Selection
    #

    def select(self, *cols):
        """
        pyspark.sql.DataFrame.select() method's equivalent for TSDF objects
        Parameters
        ----------
        cols : str or list of strs
        column names (string).
        If one of the column names is '*', that column is expanded to include all columns
        in the current :class:`TSDF`.

        Examples
        --------
        tsdf.select('*').collect()
        [Row(age=2, name='Alice'), Row(age=5, name='Bob')]
        tsdf.select('name', 'age').collect()
        [Row(name='Alice', age=2), Row(name='Bob', age=5)]

        """
        # The columns which will be a mandatory requirement while selecting from TSDFs
        if set(self.structural_cols).issubset(set(cols)):
            return self.__withTransformedDF(self.df.select(*cols))
        else:
            raise TSDFStructureChangeError("select that does not include all structural columns")

    def __slice(self, op: str, target_ts):
        """
        Private method to slice TSDF by time

        :param op: string symbol of the operation to perform
        :type op: str
        :param target_ts: timestamp on which to filter

        :return: a TSDF object containing only those records within the time slice specified
        """
        # quote our timestamp if its a string
        target_expr = f"'{target_ts}'" if isinstance(target_ts, str) else target_ts
        slice_expr = Fn.expr(f"{self.ts_col} {op} {target_expr}")
        sliced_df = self.df.where(slice_expr)
        return self.__withTransformedDF(sliced_df)

    def at(self, ts):
        """
        Select only records at a given time

        :param ts: timestamp of the records to select

        :return: a :class:`~tsdf.TSDF` object containing just the records at the given time
        """
        return self.__slice("==", ts)

    def before(self, ts):
        """
        Select only records before a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records before the given time
        """
        return self.__slice("<", ts)

    def atOrBefore(self, ts):
        """
        Select only records at or before a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records at or before the given time
        """
        return self.__slice("<=", ts)

    def after(self, ts):
        """
        Select only records after a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records after the given time
        """
        return self.__slice(">", ts)

    def atOrAfter(self, ts):
        """
        Select only records at or after a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records at or after the given time
        """
        return self.__slice(">=", ts)

    def between(self, start_ts, end_ts, inclusive=True):
        """
        Select only records in a given range

        :param start_ts: starting time of the range to select
        :param end_ts: ending time of the range to select
        :param inclusive: whether the range is inclusive of the endpoints or not, defaults to True
        :type inclusive: bool

        :return: a :class:`~tsdf.TSDF` object containing just the records within the range specified
        """
        if inclusive:
            return self.atOrAfter(start_ts).atOrBefore(end_ts)
        return self.after(start_ts).before(end_ts)

    def __top_rows_per_series(self, win: WindowSpec, n: int):
        """
        Private method to select just the top n rows per series (as defined by a window ordering)

        :param win: the window on which we order the rows in each series
        :param n: the number of rows to return

        :return: a :class:`~tsdf.TSDF` object containing just the top n rows in each series
        """
        row_num_col = "__row_num"
        prev_records_df = (
            self.df.withColumn(row_num_col, Fn.row_number().over(win))
            .where(Fn.col(row_num_col) <= Fn.lit(n))
            .drop(row_num_col)
        )
        return self.__withTransformedDF(prev_records_df)

    def earliest(self, n: int = 1):
        """
        Select the earliest n records for each series

        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the earliest n records for each series
        """
        prev_window = self.__baseWindow(reverse=False)
        return self.__top_rows_per_series(prev_window, n)

    def latest(self, n: int = 1):
        """
        Select the latest n records for each series

        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the latest n records for each series
        """
        next_window = self.__baseWindow(reverse=True)
        return self.__top_rows_per_series(next_window, n)

    def priorTo(self, ts, n: int = 1):
        """
        Select the n most recent records prior to a given time
        You can think of this like an 'asOf' select - it selects the records as of a particular time

        :param ts: timestamp on which to filter records
        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the n records prior to the given time
        """
        return self.atOrBefore(ts).latest(n)

    def subsequentTo(self, ts, n: int = 1):
        """
        Select the n records subsequent to a give time

        :param ts: timestamp on which to filter records
        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the n records subsequent to the given time
        """
        return self.atOrAfter(ts).earliest(n)

    #
    # Display functions
    #

    def show(self, n=20, k=5, truncate=True, vertical=False):
        """
        pyspark.sql.DataFrame.show() method's equivalent for TSDF objects

        Parameters
        ----------
        n : int, optional
        Number of rows to show.
        truncate : bool or int, optional
        If set to ``True``, truncate strings longer than 20 chars by default.
        If set to a number greater than one, truncates long strings to length ``truncate``
        and align cells right.
        vertical : bool, optional
        If set to ``True``, print output rows vertically (one line
        per column value).

        Example to show usage
        ---------------------
        from pyspark.sql.functions import *

        phone_accel_df = spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Phones_accelerometer").withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")).withColumn("x", col("x").cast("double")).withColumn("y", col("y").cast("double")).withColumn("z", col("z").cast("double")).withColumn("event_ts_dbl", col("event_ts").cast("double"))

        from tempo import *

        phone_accel_tsdf = TSDF(phone_accel_df, ts_col="event_ts", partition_cols = ["User"])

        # Call show method here
        phone_accel_tsdf.show()

        """
        # validate k <= n
        if k > n:
            raise ValueError(f"Parameter k {k} cannot be greater than parameter n {n}")

        if not (IS_DATABRICKS) and ENV_CAN_RENDER_HTML:
            # In Jupyter notebooks, for wide dataframes the below line will enable rendering the output in a scrollable format.
            ipydisplay(HTML("<style>pre { white-space: pre !important; }</style>"))
        get_display_df(self,k=k).show(n, truncate, vertical)

    def describe(self):
        """
        Describe a TSDF object using a global summary across all time series (anywhere from 10 to millions) as well as the standard Spark data frame stats. Missing vals
        Summary
        global - unique time series based on partition columns, min/max times, granularity - lowest precision in the time series timestamp column
        count / mean / stddev / min / max - standard Spark data frame describe() output
        missing_vals_pct - percentage (from 0 to 100) of missing values.
        """
        # extract the double version of the timestamp column to summarize
        double_ts_col = self.ts_col + "_dbl"

        this_df = self.df.withColumn(double_ts_col, Fn.col(self.ts_col).cast("double"))

        # summary missing value percentages
        missing_vals = this_df.select(
            [
                (
                    100
                    * Fn.count(Fn.when(Fn.col(c[0]).isNull(), c[0]))
                    / Fn.count(Fn.lit(1))
                ).alias(c[0])
                for c in this_df.dtypes
                if c[1] != "timestamp"
            ]
        ).select(Fn.lit("missing_vals_pct").alias("summary"), "*")

        # describe stats
        desc_stats = this_df.describe().union(missing_vals)
        unique_ts = this_df.select(*self.series_ids).distinct().count()

        max_ts = this_df.select(Fn.max(Fn.col(self.ts_col)).alias("max_ts")).collect()[0][
            0
        ]
        min_ts = this_df.select(Fn.min(Fn.col(self.ts_col)).alias("max_ts")).collect()[0][
            0
        ]
        gran = this_df.selectExpr(
            """min(case when {0} - cast({0} as integer) > 0 then '1-millis'
                  when {0} % 60 != 0 then '2-seconds'
                  when {0} % 3600 != 0 then '3-minutes'
                  when {0} % 86400 != 0 then '4-hours'
                  else '5-days' end) granularity""".format(
                double_ts_col
            )
        ).collect()[0][0][2:]

        non_summary_cols = [c for c in desc_stats.columns if c != "summary"]

        desc_stats = desc_stats.select(
            Fn.col("summary"),
            Fn.lit(" ").alias("unique_ts_count"),
            Fn.lit(" ").alias("min_ts"),
            Fn.lit(" ").alias("max_ts"),
            Fn.lit(" ").alias("granularity"),
            *non_summary_cols,
        )

        # add in single record with global summary attributes and the previously computed missing value and Spark data frame describe stats
        global_smry_rec = desc_stats.limit(1).select(
            Fn.lit("global").alias("summary"),
            Fn.lit(unique_ts).alias("unique_ts_count"),
            Fn.lit(min_ts).alias("min_ts"),
            Fn.lit(max_ts).alias("max_ts"),
            Fn.lit(gran).alias("granularity"),
            *[Fn.lit(" ").alias(c) for c in non_summary_cols],
        )

        full_smry = global_smry_rec.union(desc_stats)
        full_smry = full_smry.withColumnRenamed(
            "unique_ts_count", "unique_time_series_count"
        )

        try:
            dbutils.fs.ls("/")
            return full_smry
        except Exception:
            return full_smry
            pass

    def __getBytesFromPlan(self, df, spark):
        """
        Internal helper function to obtain how many bytes in memory the Spark data frame is likely to take up. This is an upper bound and is obtained from the plan details in Spark

        Parameters
        :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
        :param spark - Spark session which is used to query the view obtained from the Spark data frame
        """

        df.createOrReplaceTempView("view")
        plan = spark.sql("explain cost select * from view").collect()[0][0]

        import re

        result = (
            re.search(r"sizeInBytes=.*(['\)])", plan, re.MULTILINE)
            .group(0)
            .replace(")", "")
        )
        size = result.split("=")[1].split(" ")[0]
        units = result.split("=")[1].split(" ")[1]

        # perform to MB for threshold check
        if units == "GiB":
            bytes = float(size) * 1024 * 1024 * 1024
        elif units == "MiB":
            bytes = float(size) * 1024 * 1024
        elif units == "KiB":
            bytes = float(size) * 1024
        else:
            bytes = float(size)

        return bytes

    def asofJoin(
        self,
        right_tsdf,
        left_prefix=None,
        right_prefix="right",
        tsPartitionVal=None,
        fraction=0.5,
        skipNulls=True,
        sql_join_opt=False,
        suppress_null_warning=False,
    ):
        """
        Performs an as-of join between two time-series. If a tsPartitionVal is specified, it will do this partitioned by
        time brackets, which can help alleviate skew.

        NOTE: partition cols have to be the same for both Dataframes. We are collecting stats when the WARNING level is
        enabled also.

        Parameters
        :param right_tsdf - right-hand data frame containing columns to merge in
        :param left_prefix - optional prefix for base data frame
        :param right_prefix - optional prefix for right-hand data frame
        :param tsPartitionVal - value to break up each partition into time brackets
        :param fraction - overlap fraction
        :param skipNulls - whether to skip nulls when joining in values
        :param sql_join_opt - if set to True, will use standard Spark SQL join if it is estimated to be efficient
        :param suppress_null_warning - when tsPartitionVal is specified, will collect min of each column and raise warnings about null values, set to True to avoid
        """

        # first block of logic checks whether a standard range join will suffice
        left_df = self.df
        right_df = right_tsdf.df

        spark = SparkSession.builder.getOrCreate()
        left_bytes = self.__getBytesFromPlan(left_df, spark)
        right_bytes = self.__getBytesFromPlan(right_df, spark)

        # choose 30MB as the cutoff for the broadcast
        bytes_threshold = 30 * 1024 * 1024
        if sql_join_opt & (
            (left_bytes < bytes_threshold) | (right_bytes < bytes_threshold)
        ):
            spark.conf.set("spark.databricks.optimizer.rangeJoin.binSize", "60")
            partition_cols = right_tsdf.series_ids
            left_cols = list(set(left_df.columns).difference(set(self.series_ids)))
            right_cols = list(
                set(right_df.columns).difference(set(right_tsdf.series_ids))
            )

            left_prefix = (
                ""
                if ((left_prefix is None) | (left_prefix == ""))
                else left_prefix + "_"
            )
            right_prefix = (
                ""
                if ((right_prefix is None) | (right_prefix == ""))
                else right_prefix + "_"
            )

            w = Window.partitionBy(*partition_cols).orderBy(
                right_prefix + right_tsdf.ts_col
            )

            new_left_ts_col = left_prefix + self.ts_col
            new_left_cols = [Fn.col(c).alias(left_prefix + c) for c in left_cols] + partition_cols
            new_right_cols = [Fn.col(c).alias(right_prefix + c) for c in right_cols] + partition_cols
            quotes_df_w_lag = right_df.select(*new_right_cols).withColumn(
                "lead_" + right_tsdf.ts_col,
                Fn.lead(right_prefix + right_tsdf.ts_col).over(w),
            )
            left_df = left_df.select(*new_left_cols)
            res = (
                left_df.join(quotes_df_w_lag, partition_cols)
                .where(
                    left_df[new_left_ts_col].between(
                        Fn.col(right_prefix + right_tsdf.ts_col),
                        Fn.coalesce(
                            Fn.col("lead_" + right_tsdf.ts_col),
                            Fn.lit("2099-01-01").cast("timestamp"),
                        ),
                    )
                )
                .drop("lead_" + right_tsdf.ts_col)
            )
            return TSDF(res, series_ids=self.series_ids, ts_col=new_left_ts_col)

        # end of block checking to see if standard Spark SQL join will work

        if tsPartitionVal is not None:
            logger.warning(
                "You are using the skew version of the AS OF join. This may result in null values if there are any values outside of the maximum lookback. For maximum efficiency, choose smaller values of maximum lookback, trading off performance and potential blank AS OF values for sparse keys"
            )

        # Check whether partition columns have same name in both dataframes
        self.__checkPartitionCols(right_tsdf)

        # prefix non-partition columns, to avoid duplicated columns.
        left_df = self.df
        right_df = right_tsdf.df

        # validate timestamp datatypes match
        self.__validateTsColMatch(right_tsdf)

        orig_left_col_diff = list(set(left_df.columns).difference(set(self.series_ids)))
        orig_right_col_diff = list(
            set(right_df.columns).difference(set(self.series_ids))
        )

        left_tsdf = (
            (self.__addPrefixToColumns([self.ts_col] + orig_left_col_diff, left_prefix))
            if left_prefix is not None
            else self
        )
        right_tsdf = right_tsdf.__addPrefixToColumns(
            [right_tsdf.ts_col] + orig_right_col_diff, right_prefix
        )

        left_nonpartition_cols = list(
            set(left_tsdf.df.columns).difference(set(self.series_ids))
        )
        right_nonpartition_cols = list(
            set(right_tsdf.df.columns).difference(set(self.series_ids))
        )

        # For both dataframes get all non-partition columns (including ts_col)
        left_columns = [left_tsdf.ts_col] + left_nonpartition_cols
        right_columns = [right_tsdf.ts_col] + right_nonpartition_cols

        # Union both dataframes, and create a combined TS column
        combined_ts_col = "combined_ts"
        combined_df = left_tsdf.__addColumnsFromOtherDF(right_columns).__combineTSDF(
            right_tsdf.__addColumnsFromOtherDF(left_columns), combined_ts_col
        )
        combined_df.df = combined_df.df.withColumn(
            "rec_ind", Fn.when(Fn.col(left_tsdf.ts_col).isNotNull(), 1).otherwise(-1)
        )

        # perform asof join.
        if tsPartitionVal is None:
            seq_col = None
            if isinstance(combined_df.ts_index, SubSequenceTSIndex):
                seq_col = combined_df.ts_index.sub_seq_col
            asofDF = combined_df.__getLastRightRow(
                left_tsdf.ts_col,
                right_columns,
                seq_col,
                tsPartitionVal,
                skipNulls,
                suppress_null_warning,
            )
        else:
            tsPartitionDF = combined_df.__getTimePartitions(
                tsPartitionVal, fraction=fraction
            )
            seq_col = None
            if isinstance(tsPartitionDF.ts_index, SubSequenceTSIndex):
                seq_col = tsPartitionDF.ts_index.sub_seq_col
            asofDF = tsPartitionDF.__getLastRightRow(
                left_tsdf.ts_col,
                right_columns,
                seq_col,
                tsPartitionVal,
                skipNulls,
                suppress_null_warning,
            )

            # Get rid of overlapped data and the extra columns generated from timePartitions
            df = asofDF.df.filter(Fn.col("is_original") == 1).drop(
                "ts_partition", "is_original"
            )

            asofDF = TSDF(df, ts_col=asofDF.ts_col, series_ids=combined_df.series_ids)

        return asofDF

    def __baseWindow(self, reverse=False):
        # The index will determine the appropriate sort order
        w = Window().orderBy(self.ts_index.orderByExpr(reverse))

        # and partitioned by any series IDs
        if self.series_ids:
            w = w.partitionBy([Fn.col(sid) for sid in self.series_ids])
        return w

    def __rowsBetweenWindow(self, rows_from, rows_to, reverse=False):
        return self.__baseWindow(reverse=reverse).rowsBetween(rows_from, rows_to)

    def __rangeBetweenWindow(self, range_from, range_to, reverse=False):
        return ( self.__baseWindow(reverse=reverse)
                     .orderBy(self.ts_index.rangeOrderByExpr(reverse=reverse))
                     .rangeBetween(range_from, range_to ) )

    #
    # Core Transformations
    #

    def withNaturalOrdering(self, reverse: bool = False) -> "TSDF":
        order_expr = [ Fn.col(c) for c in self.series_ids]
        ts_idx_expr = self.ts_index.orderByExpr(reverse)
        if isinstance(ts_idx_expr, list):
            order_expr.extend(ts_idx_expr)
        else:
            order_expr.append(ts_idx_expr)

        return self.__withTransformedDF(self.df.orderBy(order_expr))

    def withColumn(self, colName: str, col: Column) -> "TSDF":
        """
        Returns a new :class:`TSDF` by adding a column or replacing the existing column that has the same name.

        :param colName: the name of the new column (or existing column to be replaced)
        :param col: a :class:`Column` expression for the new column definition
        """
        if colName in self.structural_cols:
            raise TSDFStructureChangeError(f"withColumn on the structural column {colName}.")
        new_df = self.df.withColumn(colName, col)
        return self.__withTransformedDF(new_df)

    def withColumnRenamed(self, existing: str, new: str) -> "TSDF":
        """
        Returns a new :class:`TSDF` with the given column renamed.

        :param existing: name of the existing column to renmame
        :param new: new name for the column
        """

        # create new TSIndex
        new_ts_index = deepcopy(self.ts_index)
        if existing == self.ts_index.name:
            new_ts_index = new_ts_index.renamed(new)

        # and for series ids
        new_series_ids = self.series_ids
        if existing in self.series_ids:
            # replace column name in series
            new_series_ids = self.series_ids
            new_series_ids[new_series_ids.index(existing)] = new

        # rename the column in the underlying DF
        new_df = self.df.withColumnRenamed(existing,new)

        # return new TSDF
        new_schema = TSSchema(new_ts_index, new_series_ids)
        return TSDF(new_df, ts_schema=new_schema)

    def union(self, other: TSDF) -> TSDF:
        # union of the underlying DataFrames
        union_df = self.df.union(other.df)
        return self.__withTransformedDF(union_df)

    def unionByName(self, other: TSDF, allowMissingColumns: bool = False) -> TSDF:
        # union of the underlying DataFrames
        union_df = self.df.unionByName(other.df, allowMissingColumns=allowMissingColumns)
        return self.__withTransformedDF(union_df)

    #
    # utility functions
    #

    def vwap(self, frequency="m", volume_col="volume", price_col="price"):
        # set pre_vwap as self or enrich with the frequency
        pre_vwap = self.df
        if frequency == "m":
            pre_vwap = self.df.withColumn(
                "time_group",
                Fn.concat(
                    Fn.lpad(Fn.hour(Fn.col(self.ts_col)), 2, "0"),
                    Fn.lit(":"),
                    Fn.lpad(Fn.minute(Fn.col(self.ts_col)), 2, "0"),
                ),
            )
        elif frequency == "H":
            pre_vwap = self.df.withColumn(
                "time_group", Fn.concat(Fn.lpad(Fn.hour(Fn.col(self.ts_col)), 2, "0"))
            )
        elif frequency == "D":
            pre_vwap = self.df.withColumn(
                "time_group", Fn.concat(Fn.lpad(Fn.day(Fn.col(self.ts_col)), 2, "0"))
            )

        group_cols = ["time_group"]
        if self.series_ids:
            group_cols.extend(self.series_ids)
        vwapped = (
            pre_vwap.withColumn("dllr_value", Fn.col(price_col) * Fn.col(volume_col))
            .groupby(group_cols)
            .agg(
                sum("dllr_value").alias("dllr_value"),
                sum(volume_col).alias(volume_col),
                max(price_col).alias("_".join(["max", price_col])),
            )
            .withColumn("vwap", Fn.col("dllr_value") / Fn.col(volume_col))
        )

        return self.__withTransformedDF(vwapped)

    def EMA(self, colName, window=30, exp_factor=0.2):
        """
        Constructs an approximate EMA in the fashion of:
        EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) etc, up until window
        TODO: replace case when statement with coalesce
        TODO: add in time partitions functionality (what is the overlap fraction?)
        """

        emaColName = "_".join(["EMA", colName])
        df = self.df.withColumn(emaColName, Fn.lit(0)).orderBy(self.ts_col)
        w = self.__baseWindow()
        # Generate all the lag columns:
        for i in range(window):
            lagColName = "_".join(["lag", colName, str(i)])
            weight = exp_factor * (1 - exp_factor) ** i
            df = df.withColumn(lagColName, weight * Fn.lag(Fn.col(colName), i).over(w))
            df = df.withColumn(
                emaColName,
                Fn.col(emaColName)
                + Fn.when(Fn.col(lagColName).isNull(), Fn.lit(0)).otherwise(
                    Fn.col(lagColName)
                ),
            ).drop(lagColName)
            # Nulls are currently removed

        return self.__withTransformedDF(df)

    def withLookbackFeatures(
        self, featureCols, lookbackWindowSize, exactSize=True, featureColName="features"
    ):
        """
        Creates a 2-D feature tensor suitable for training an ML model to predict current values from the history of
        some set of features. This function creates a new column containing, for each observation, a 2-D array of the values
        of some number of other columns over a trailing "lookback" window from the previous observation up to some maximum
        number of past observations.

        :param featureCols: the names of one or more feature columns to be aggregated into the feature column
        :param lookbackWindowSize: The size of lookback window (in terms of past observations). Must be an integer >= 1
        :param exactSize: If True (the default), then the resulting DataFrame will only include observations where the
          generated feature column contains arrays of length lookbackWindowSize. This implies that it will truncate
          observations that occurred less than lookbackWindowSize from the start of the timeseries. If False, no truncation
          occurs, and the column may contain arrays less than lookbackWindowSize in length.
        :param featureColName: The name of the feature column to be generated. Defaults to "features"
        :return: a DataFrame with a feature column named featureColName containing the lookback feature tensor
        """
        # first, join all featureCols into a single array column
        tempArrayColName = "__TempArrayCol"
        feat_array_tsdf = self.df.withColumn(tempArrayColName, Fn.array(featureCols))

        # construct a lookback array
        lookback_win = self.__rowsBetweenWindow(-lookbackWindowSize, -1)
        lookback_tsdf = feat_array_tsdf.withColumn(
            featureColName, Fn.collect_list(Fn.col(tempArrayColName)).over(lookback_win)
        ).drop(tempArrayColName)

        # make sure only windows of exact size are allowed
        if exactSize:
            return lookback_tsdf.where(Fn.size(featureColName) == lookbackWindowSize)

        return self.__withTransformedDF(lookback_tsdf)

    def withRangeStats(
        self, type="range", colsToSummarize=[], rangeBackWindowSecs=1000
    ):
        """
        Create a wider set of stats based on all numeric columns by default
        Users can choose which columns they want to summarize also. These stats are:
        mean/count/min/max/sum/std deviation/zscore
        :param type - this is created in case we want to extend these stats to lookback over a fixed number of rows instead of ranging over column values
        :param colsToSummarize - list of user-supplied columns to compute stats for. All numeric columns are used if no list is provided
        :param rangeBackWindowSecs - lookback this many seconds in time to summarize all stats. Note this will look back from the floor of the base event timestamp (as opposed to the exact time since we cast to long)
        Assumptions:

        1. The features are summarized over a rolling window that ranges back
        2. The range back window can be specified by the user
        3. Sequence numbers are not yet supported for the sort
        4. There is a cast to long from timestamp so microseconds or more likely breaks down - this could be more easily handled with a string timestamp or sorting the timestamp itself. If using a 'rows preceding' window, this wouldn't be a problem
        """

        # by default summarize all metric columns
        if not colsToSummarize:
            colsToSummarize = self.metric_cols

        # build window
        w = self.__rangeBetweenWindow(-1 * rangeBackWindowSecs, 0)

        # compute column summaries
        selectedCols = self.df.columns
        derivedCols = []
        for metric in colsToSummarize:
            selectedCols.append(Fn.mean(metric).over(w).alias("mean_" + metric))
            selectedCols.append(Fn.count(metric).over(w).alias("count_" + metric))
            selectedCols.append(Fn.min(metric).over(w).alias("min_" + metric))
            selectedCols.append(Fn.max(metric).over(w).alias("max_" + metric))
            selectedCols.append(Fn.sum(metric).over(w).alias("sum_" + metric))
            selectedCols.append(Fn.stddev(metric).over(w).alias("stddev_" + metric))
            derivedCols.append(
                (
                    (Fn.col(metric) - Fn.col("mean_" + metric))
                    / Fn.col("stddev_" + metric)
                ).alias("zscore_" + metric)
            )
        selected_df = self.df.select(*selectedCols)
        summary_df = selected_df.select(*selected_df.columns, *derivedCols).drop(
            "double_ts"
        )

        return self.__withTransformedDF(summary_df)

    def withGroupedStats(self, metricCols=[], freq=None):
        """
        Create a wider set of stats based on all numeric columns by default
        Users can choose which columns they want to summarize also. These stats are:
        mean/count/min/max/sum/std deviation
        :param metricCols - list of user-supplied columns to compute stats for. All numeric columns are used if no list is provided
        :param freq - frequency (provide a string of the form '1 min', '30 seconds' and we interpret the window to use to aggregate
        """

        # identify columns to summarize if not provided
        # these should include all numeric columns that
        # are not the timestamp column and not any of the partition columns
        if not metricCols:
            # columns we should never summarize
            prohibited_cols = [self.ts_col.lower()]
            if self.series_ids:
                prohibited_cols.extend([pc.lower() for pc in self.series_ids])
            # types that can be summarized
            summarizable_types = ["int", "bigint", "float", "double"]
            # filter columns to find summarizable columns
            metricCols = [
                datatype[0]
                for datatype in self.df.dtypes
                if (
                    (datatype[1] in summarizable_types)
                    and (datatype[0].lower() not in prohibited_cols)
                )
            ]

        # build window
        parsed_freq = rs.checkAllowableFreq(freq)
        agg_window = Fn.window(
            Fn.col(self.ts_col),
            "{} {}".format(parsed_freq[0], rs.freq_dict[parsed_freq[1]]),
        )

        # compute column summaries
        selectedCols = []
        for metric in metricCols:
            selectedCols.extend(
                [
                    Fn.mean(Fn.col(metric)).alias("mean_" + metric),
                    Fn.count(Fn.col(metric)).alias("count_" + metric),
                    Fn.min(Fn.col(metric)).alias("min_" + metric),
                    Fn.max(Fn.col(metric)).alias("max_" + metric),
                    Fn.sum(Fn.col(metric)).alias("sum_" + metric),
                    Fn.stddev(Fn.col(metric)).alias("stddev_" + metric),
                ]
            )

        selected_df = self.df.groupBy(self.series_ids + [agg_window]).agg(*selectedCols)
        summary_df = (
            selected_df.select(*selected_df.columns)
            .withColumn(self.ts_col, Fn.col("window").start)
            .drop("window")
        )

        return self.__withTransformedDF(summary_df)

    def write(self, spark, tabName, optimizationCols=None):
        tio.write(self, spark, tabName, optimizationCols)

    def resample(
        self,
        freq,
        func=None,
        metricCols=None,
        prefix=None,
        fill=None,
        perform_checks=True,
    ):
        """
        function to upsample based on frequency and aggregate function similar to pandas
        :param freq: frequency for upsample - valid inputs are "hr", "min", "sec" corresponding to hour, minute, or second
        :param func: function used to aggregate input
        :param metricCols supply a smaller list of numeric columns if the entire set of numeric columns should not be returned for the resample function
        :param prefix - supply a prefix for the newly sampled columns
        :param fill - Boolean - set to True if the desired output should contain filled in gaps (with 0s currently)
        :param perform_checks: calculate time horizon and warnings if True (default is True)
        :return: TSDF object with sample data using aggregate function
        """
        rs.validateFuncExists(func)

        # Throw warning for user to validate that the expected number of output rows is valid.
        if fill is True and perform_checks is True:
            calculate_time_horizon(self.df, self.ts_col, freq, self.series_ids)

        enriched_df: DataFrame = rs.aggregate(
            self, freq, func, metricCols, prefix, fill
        )
        return _ResampledTSDF(
            enriched_df,
            ts_col=self.ts_col,
            series_ids=self.series_ids,
            freq=freq,
            func=func,
        )

    def interpolate(
        self,
        freq: str,
        func: str,
        method: str,
        target_cols: List[str] = None,
        ts_col: str = None,
        series_ids: List[str] = None,
        show_interpolated: bool = False,
        perform_checks: bool = True,
    ):
        """
        Function to interpolate based on frequency, aggregation, and fill similar to pandas. Data will first be aggregated using resample, then missing values
        will be filled based on the fill calculation.

        :param freq: frequency for upsample - valid inputs are "hr", "min", "sec" corresponding to hour, minute, or second
        :param func: function used to aggregate input
        :param method: function used to fill missing values e.g. linear, null, zero, bfill, ffill
        :param target_cols [optional]: columns that should be interpolated, by default interpolates all numeric columns
        :param ts_col [optional]: specify other ts_col, by default this uses the ts_col within the TSDF object
        :param partition_cols [optional]: specify other partition_cols, by default this uses the partition_cols within the TSDF object
        :param show_interpolated [optional]: if true will include an additional column to show which rows have been fully interpolated.
        :param perform_checks: calculate time horizon and warnings if True (default is True)
        :return: new TSDF object containing interpolated data
        """

        # Set defaults for target columns, timestamp column and partition columns when not provided
        if ts_col is None:
            ts_col = self.ts_col
        if series_ids is None:
            series_ids = self.series_ids
        if target_cols is None:
            prohibited_cols: List[str] = series_ids + [ts_col]
            summarizable_types = ["int", "bigint", "float", "double"]

            # get summarizable find summarizable columns
            target_cols: List[str] = [
                datatype[0]
                for datatype in self.df.dtypes
                if (
                    (datatype[1] in summarizable_types)
                    and (datatype[0].lower() not in prohibited_cols)
                )
            ]

        interpolate_service: Interpolation = Interpolation(is_resampled=False)
        tsdf_input = TSDF(self.df, ts_col=ts_col, series_ids=series_ids)
        interpolated_df: DataFrame = interpolate_service.interpolate(
            tsdf_input,
            ts_col,
            series_ids,
            target_cols,
            freq,
            func,
            method,
            show_interpolated,
            perform_checks,
        )

        return TSDF(interpolated_df, ts_col=ts_col, series_ids=series_ids)

    def calc_bars(tsdf, freq, func=None, metricCols=None, fill=None):

        resample_open = tsdf.resample(
            freq=freq, func="floor", metricCols=metricCols, prefix="open", fill=fill
        )
        resample_low = tsdf.resample(
            freq=freq, func="min", metricCols=metricCols, prefix="low", fill=fill
        )
        resample_high = tsdf.resample(
            freq=freq, func="max", metricCols=metricCols, prefix="high", fill=fill
        )
        resample_close = tsdf.resample(
            freq=freq, func="ceil", metricCols=metricCols, prefix="close", fill=fill
        )

        join_cols = resample_open.series_ids + [resample_open.ts_col]
        bars = (
            resample_open.df.join(resample_high.df, join_cols)
            .join(resample_low.df, join_cols)
            .join(resample_close.df, join_cols)
        )
        non_part_cols = set(set(bars.columns) - set(resample_open.series_ids)) - set(
            [resample_open.ts_col]
        )
        sel_and_sort = (
            resample_open.series_ids + [resample_open.ts_col] + sorted(non_part_cols)
        )
        bars = bars.select(sel_and_sort)

        return TSDF(bars, ts_col=resample_open.ts_col, series_ids=resample_open.series_ids)

    def fourier_transform(self, timestep, valueCol):
        """
        Function to fourier transform the time series to its frequency domain representation.
        :param timestep: timestep value to be used for getting the frequency scale
        :param valueCol: name of the time domain data column which will be transformed
        """

        def tempo_fourier_util(pdf):
            """
            This method is a vanilla python logic implementing fourier transform on a numpy array using the scipy module.
            This method is meant to be called from Tempo TSDF as a pandas function API on Spark
            """
            select_cols = list(pdf.columns)
            pdf.sort_values(by=["tpoints"], inplace=True, ascending=True)
            y = np.array(pdf["tdval"])
            tran = fft(y)
            r = tran.real
            i = tran.imag
            pdf["ft_real"] = r
            pdf["ft_imag"] = i
            N = tran.shape
            xf = fftfreq(N[0], timestep)
            pdf["freq"] = xf
            return pdf[select_cols + ["freq", "ft_real", "ft_imag"]]

        # valueCol = self.__validated_column(self.df, valueCol)
        data = self.df

        if self.series_ids == []:
            data = data.withColumn("dummy_group", Fn.lit("dummy_val"))
            data = (
                data.select(Fn.col("dummy_group"), self.ts_col, Fn.col(valueCol))
                .withColumn("tdval", Fn.col(valueCol))
                .withColumn("tpoints", Fn.col(self.ts_col))
            )
            return_schema = ",".join(
                [f"{i[0]} {i[1]}" for i in data.dtypes]
                + ["freq double", "ft_real double", "ft_imag double"]
            )
            result = data.groupBy("dummy_group").applyInPandas(
                tempo_fourier_util, return_schema
            )
            result = result.drop("dummy_group", "tdval", "tpoints")
        else:
            group_cols = self.series_ids
            data = (
                data.select(*group_cols, self.ts_col, Fn.col(valueCol))
                .withColumn("tdval", Fn.col(valueCol))
                .withColumn("tpoints", Fn.col(self.ts_col))
            )
            return_schema = ",".join(
                [f"{i[0]} {i[1]}" for i in data.dtypes]
                + ["freq double", "ft_real double", "ft_imag double"]
            )
            result = data.groupBy(*group_cols).applyInPandas(
                tempo_fourier_util, return_schema
            )
            result = result.drop("tdval", "tpoints")

        return self.__withTransformedDF(result)

    def extractStateIntervals(
        self,
        *metric_cols: str,
        state_definition: Union[str, Callable[[Column, Column], Column]] = "=",
    ) -> DataFrame:
        """
        Extracts intervals from a :class:`~tsdf.TSDF` based on some notion of "state", as defined by the :param
        state_definition: parameter. The state definition consists of a comparison operation between the current and
        previous values of a metric. If the comparison operation evaluates to true across all metric columns,
        then we consider both points to be in the same "state". Changes of state occur when the comparison operator
        returns false for any given metric column. So, the default state definition ('=') entails that intervals of
        time wherein the metrics all remained constant. A state definition of '>=' would extract intervals wherein
        the metrics were all monotonically increasing.

        :param: metric_cols: the set of metric columns to evaluate for state changes
        :param: state_definition: the comparison function used to evaluate individual metrics for state changes.
        Either a string, giving a standard PySpark column comparison operation, or a binary function with the
        signature: `(x1: Column, x2: Column) -> Column` where the returned column expression evaluates to a
        :class:`~pyspark.sql.types.BooleanType`

        :return: a :class:`~pyspark.sql.DataFrame` object containing the resulting intervals
        """

        # https://spark.apache.org/docs/latest/sql-ref-null-semantics.html#comparison-operators-
        def null_safe_equals(col1: Column, col2: Column) -> Column:
            return (
                Fn.when(col1.isNull() & col2.isNull(), True)
                .when(col1.isNull() | col2.isNull(), False)
                .otherwise(operator.eq(col1, col2))
            )

        operator_dict = {
            # https://spark.apache.org/docs/latest/api/sql/#_2
            "!=": operator.ne,
            # https://spark.apache.org/docs/latest/api/sql/#_11
            "<>": operator.ne,
            # https://spark.apache.org/docs/latest/api/sql/#_8
            "<": operator.lt,
            # https://spark.apache.org/docs/latest/api/sql/#_9
            "<=": operator.le,
            # https://spark.apache.org/docs/latest/api/sql/#_10
            "<=>": null_safe_equals,
            # https://spark.apache.org/docs/latest/api/sql/#_12
            "=": operator.eq,
            # https://spark.apache.org/docs/latest/api/sql/#_13
            "==": operator.eq,
            # https://spark.apache.org/docs/latest/api/sql/#_14
            ">": operator.gt,
            # https://spark.apache.org/docs/latest/api/sql/#_15
            ">=": operator.ge,
        }

        # Validate state definition and construct state comparison function
        if type(state_definition) is str:
            if state_definition not in operator_dict.keys():
                raise ValueError(
                    f"Invalid comparison operator for `state_definition` argument: {state_definition}."
                )

            def state_comparison_fn(a, b):
                return operator_dict[state_definition](a, b)

        elif callable(state_definition):
            state_comparison_fn = state_definition

        else:
            raise TypeError(
                f"The `state_definition` argument can be of type `str` or `callable`, "
                f"but received value of type {type(state_definition)}"
            )

        w = self.__baseWindow()

        data = self.df

        # Get previous timestamp to identify start time of the interval
        data = data.withColumn(
            "previous_ts",
            Fn.lag(Fn.col(self.ts_col), offset=1).over(w),
        )

        # Determine state intervals using user-provided the state comparison function
        # The comparison occurs on the current and previous record per metric column
        temp_metric_compare_cols = []
        for mc in metric_cols:
            temp_metric_compare_col = f"__{mc}_compare"
            data = data.withColumn(
                temp_metric_compare_col,
                state_comparison_fn(Fn.col(mc), Fn.lag(Fn.col(mc), 1).over(w)),
            )
            temp_metric_compare_cols.append(temp_metric_compare_col)

        # Remove first record which will have no state change
        # and produces `null` for all state comparisons
        data = data.filter(Fn.col("previous_ts").isNotNull())

        # Each state comparison should return True if state remained constant
        data = data.withColumn(
            "state_change", Fn.array_contains(Fn.array(*temp_metric_compare_cols), False)
        )

        # Count the distinct state changes to get the unique intervals
        data = data.withColumn(
            "state_incrementer",
            Fn.sum(Fn.col("state_change").cast("int")).over(w),
        ).filter(~Fn.col("state_change"))

        # Find the start and end timestamp of the interval
        result = (
            data.groupBy(*self.series_ids, "state_incrementer")
            .agg(
                Fn.min("previous_ts").alias("start_ts"),
                Fn.max(self.ts_col).alias("end_ts"),
            )
            .drop("state_incrementer")
        )

        return result


class _ResampledTSDF(TSDF):
    def __init__(
        self,
        df,
        ts_col="event_ts",
        series_ids=None,
        freq=None,
        func=None,
    ):
        super(_ResampledTSDF, self).__init__(df, ts_col=ts_col, series_ids=series_ids)
        self.__freq = freq
        self.__func = func

    def interpolate(
        self,
        method: str,
        target_cols: List[str] = None,
        show_interpolated: bool = False,
        perform_checks: bool = True,
    ):
        """
        Function to interpolate based on frequency, aggregation, and fill similar to pandas. This method requires an already sampled data set in order to use.

        :param method: function used to fill missing values e.g. linear, null, zero, bfill, ffill
        :param target_cols [optional]: columns that should be interpolated, by default interpolates all numeric columns
        :param show_interpolated [optional]: if true will include an additional column to show which rows have been fully interpolated.
        :param perform_checks: calculate time horizon and warnings if True (default is True)
        :return: new TSDF object containing interpolated data
        """

        # Set defaults for target columns, timestamp column and partition columns when not provided
        if target_cols is None:
            prohibited_cols: List[str] = self.series_ids + [self.ts_col]
            summarizable_types = ["int", "bigint", "float", "double"]

            # get summarizable find summarizable columns
            target_cols: List[str] = [
                datatype[0]
                for datatype in self.df.dtypes
                if (
                    (datatype[1] in summarizable_types)
                    and (datatype[0].lower() not in prohibited_cols)
                )
            ]

        interpolate_service: Interpolation = Interpolation(is_resampled=True)
        tsdf_input = TSDF(self.df, ts_col=self.ts_col, series_ids=self.series_ids)
        interpolated_df = interpolate_service.interpolate(
            tsdf=tsdf_input,
            ts_col=self.ts_col,
            series_ids=self.series_ids,
            target_cols=target_cols,
            freq=self.__freq,
            func=self.__func,
            method=method,
            show_interpolated=show_interpolated,
            perform_checks=perform_checks,
        )

        return TSDF(interpolated_df, ts_col=self.ts_col, series_ids=self.series_ids)
