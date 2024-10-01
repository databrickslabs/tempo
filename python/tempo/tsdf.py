from __future__ import annotations

import logging
import operator
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import pyspark.sql.functions as sfn
from IPython.core.display import HTML  # type: ignore
from IPython.display import display as ipydisplay  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType, TimestampType
from pyspark.sql.window import Window, WindowSpec
from scipy.fft import fft, fftfreq  # type: ignore

import tempo.interpol as t_interpolation
import tempo.io as t_io
import tempo.resample as t_resample
import tempo.utils as t_utils

logger = logging.getLogger(__name__)


class TSDF:
    """
    This object is the main wrapper over a Spark data frame which allows a user to parallelize time series computations on a Spark data frame by various dimensions. The two dimensions required are partition_cols (list of columns by which to summarize) and ts_col (timestamp column, which can be epoch or TimestampType).
    """

    def __init__(
        self,
        df: DataFrame,
        ts_col: str = "event_ts",
        partition_cols: Optional[list[str]] = None,
        sequence_col: Optional[str] = None,
    ):
        """
        Constructor
        :param df:
        :param ts_col:
        :param partition_cols:
        :sequence_col every tsdf allows for a tie-breaker secondary sort key
        """
        self.ts_col = self.__validated_column(df, ts_col)
        self.partitionCols = (
            []
            if partition_cols is None
            else self.__validated_columns(df, partition_cols.copy())
        )

        self.df = df
        self.sequence_col = "" if sequence_col is None else sequence_col

        # Add customized check for string type for the timestamp.
        # If we see a string, we will proactively created a double
        # version of the string timestamp for sorting purposes and
        # rename to ts_col

        # TODO : we validate the string is of a specific format. Spark will
        # convert a valid formatted timestamp string to timestamp type so
        # this if clause seems unneeded. Perhaps we should check for non-valid
        # Timestamp string matching then do some pattern matching to extract
        # the time stamp.
        if isinstance(df.schema[ts_col].dataType, StringType):  # pragma: no cover
            sample_ts = df.select(ts_col).limit(1).collect()[0][0]
            self.__validate_ts_string(sample_ts)
            self.df = (
                self.__add_double_ts()
                .drop(self.ts_col)
                .withColumnRenamed("double_ts", self.ts_col)
            )

        """
    Make sure DF is ordered by its respective ts_col and partition columns.
    """

    #
    # Helper functions
    #

    @staticmethod
    def parse_nanos_timestamp(
        df: DataFrame,
        str_ts_col: str,
        ts_fmt: str = "yyyy-MM-dd HH:mm:ss",
        double_ts_col: Optional[str] = None,
        parsed_ts_col: Optional[str] = None,
    ) -> DataFrame:
        """
        Parse a string timestamp column with nanosecond precision into a double timestamp column.

        :param df: DataFrame containing the string timestamp column
        :param str_ts_col: Name of the string timestamp column
        :param ts_fmt: Format of the string timestamp column (default: "yyyy-MM-dd HH:mm:ss")
        :param double_ts_col: Name of the double timestamp column to create, if None
                the source string column will be overwritten
        :param parsed_ts_col: Name of the parsed timestamp column to create, if None
                no parsed timestamp column will be kept

        :return: DataFrame with the double timestamp column
        """

        # add a parsed timestamp column if requested
        src_df = (
            df.withColumn(parsed_ts_col, sfn.to_timestamp(sfn.col(str_ts_col), ts_fmt))
            if parsed_ts_col
            else df
        )

        return (
            src_df.withColumn(
                "nanos",
                sfn.when(
                    sfn.col(str_ts_col).contains("."),
                    sfn.concat(sfn.lit("0."), sfn.split(sfn.col(str_ts_col), r"\.")[1]),
                )
                .otherwise(0)
                .cast("double"),
            )
            .withColumn("long_ts", sfn.unix_timestamp(str_ts_col, ts_fmt))
            .withColumn(
                (double_ts_col or str_ts_col), sfn.col("long_ts") + sfn.col("nanos")
            )
        )

    def __add_double_ts(self) -> DataFrame:
        """Add a double (epoch) version of the string timestamp out to nanos"""
        return (
            self.df.withColumn(
                "nanos",
                (
                    sfn.when(
                        sfn.col(self.ts_col).contains("."),
                        sfn.concat(
                            sfn.lit("0."),
                            sfn.split(sfn.col(self.ts_col), r"\.")[1],
                        ),
                    ).otherwise(0)
                ).cast("double"),
            )
            .withColumn("long_ts", sfn.col(self.ts_col).cast("timestamp").cast("long"))
            .withColumn("double_ts", sfn.col("long_ts") + sfn.col("nanos"))
            .drop("nanos")
            .drop("long_ts")
        )

    @staticmethod
    def __validate_ts_string(ts_text: str) -> None:
        """Validate the format for the string using Regex matching for ts_string"""
        import re

        ts_pattern = r"^(\d{4}-\d{2}-\d{2}[T| ]\d{2}:\d{2}:\d{2})(\.\d+)?$"
        if re.match(ts_pattern, ts_text) is None:
            raise ValueError(
                "Incorrect data format, should be YYYY-MM-DD HH:MM:SS[.nnnnnnnn]"
            )

    @staticmethod
    def __validated_column(df: DataFrame, colname: str) -> str:
        if not isinstance(colname, str):
            raise TypeError(
                f"Column names must be of type str; found {type(colname)} instead!"
            )
        if colname.lower() not in [col.lower() for col in df.columns]:
            raise ValueError(f"Column {colname} not found in Dataframe")
        return colname

    def __validated_columns(
        self, df: DataFrame, colnames: Optional[Union[str, List[str]]]
    ) -> List[str]:
        # if provided a string, treat it as a single column
        if isinstance(colnames, str):
            colnames = [colnames]
        # otherwise we really should have a list or None
        elif colnames is None:
            colnames = []
        elif not isinstance(colnames, list):
            raise TypeError(
                f"Columns must be of type list, str, or None; found {type(colnames)} instead!"
            )
        # validate each column
        for col in colnames:
            self.__validated_column(df, col)
        return colnames

    def __checkPartitionCols(self, tsdf_right: "TSDF") -> None:
        for left_col, right_col in zip(self.partitionCols, tsdf_right.partitionCols):
            if left_col != right_col:
                raise ValueError(
                    "left and right dataframe partition columns should have same name in same order"
                )

    def __validateTsColMatch(self, right_tsdf: "TSDF") -> None:
        left_ts_datatype = self.df.select(self.ts_col).dtypes[0][1]
        right_ts_datatype = right_tsdf.df.select(self.ts_col).dtypes[0][1]
        if left_ts_datatype != right_ts_datatype:
            raise ValueError(
                "left and right dataframe timestamp index columns should have same type"
            )

    def __addPrefixToColumns(self, col_list: list[str], prefix: str) -> "TSDF":
        """
        Add prefix to all specified columns.
        """
        # no-op if no prefix
        if not prefix:
            return self

        # build a column rename map
        col_map = {col: "_".join([prefix, col]) for col in col_list}
        # TODO - In the future (when Spark 3.4+ is standard) we should implement batch rename using:
        # df = self.df.withColumnsRenamed(col_map)

        # build a list of column expressions to rename columns in a select
        select_exprs = [
            sfn.col(col).alias(col_map[col]) if col in col_map else sfn.col(col)
            for col in self.df.columns
        ]
        # select the renamed columns
        renamed_df = self.df.select(*select_exprs)

        # find the structural columns
        ts_col = col_map.get(self.ts_col, self.ts_col)
        partition_cols = [col_map.get(c, c) for c in self.partitionCols]
        sequence_col = col_map.get(self.sequence_col, self.sequence_col)
        return TSDF(renamed_df, ts_col, partition_cols, sequence_col=sequence_col)

    def __addColumnsFromOtherDF(self, other_cols: Sequence[str]) -> "TSDF":
        """
        Add columns from some other DF as lit(None), as pre-step before union.
        """

        # build a list of column expressions to rename columns in a select
        current_cols = [sfn.col(col) for col in self.df.columns]
        new_cols = [sfn.lit(None).alias(col) for col in other_cols]
        new_df = self.df.select(current_cols + new_cols)

        return TSDF(new_df, self.ts_col, self.partitionCols)

    def __combineTSDF(self, ts_df_right: "TSDF", combined_ts_col: str) -> "TSDF":
        combined_df = self.df.unionByName(ts_df_right.df).withColumn(
            combined_ts_col, sfn.coalesce(self.ts_col, ts_df_right.ts_col)
        )

        return TSDF(combined_df, combined_ts_col, self.partitionCols)

    def __getLastRightRow(
        self,
        left_ts_col: str,
        right_cols: list[str],
        sequence_col: str,
        tsPartitionVal: Optional[int],
        ignoreNulls: bool,
        suppress_null_warning: bool,
    ) -> "TSDF":
        """Get last right value of each right column (inc. right timestamp) for each self.ts_col value

        self.ts_col, which is the combined time-stamp column of both left and right dataframe, is dropped at the end
        since it is no longer used in subsequent methods.
        """
        ptntl_sort_keys = [self.ts_col, "rec_ind"]
        if sequence_col:
            ptntl_sort_keys.append(sequence_col)

        sort_keys = [
            sfn.col(col_name) for col_name in ptntl_sort_keys if col_name != ""
        ]

        window_spec = (
            Window.partitionBy(self.partitionCols)
            .orderBy(sort_keys)
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        # generate expressions to find the last value of each right-hand column
        if ignoreNulls is False:
            if tsPartitionVal is not None:
                raise ValueError(
                    "Disabling null skipping with a partition value is not supported yet."
                )
            mod_right_cols = [
                sfn.last(
                    sfn.when(sfn.col("rec_ind") == -1, sfn.struct(col)).otherwise(None),
                    True,
                )
                .over(window_spec)[col]
                .alias(col)
                for col in right_cols
            ]
        elif tsPartitionVal is None:
            mod_right_cols = [
                sfn.last(col, ignoreNulls).over(window_spec).alias(col)
                for col in right_cols
            ]
        else:
            mod_right_cols = [
                sfn.last(col, ignoreNulls).over(window_spec).alias(col)
                for col in right_cols
            ]
            # non-null count columns, these will be dropped below
            mod_right_cols += [
                sfn.count(col).over(window_spec).alias("non_null_ct" + col)
                for col in right_cols
            ]

        # select the left-hand side columns, and the modified right-hand side columns
        non_right_cols = list(set(self.df.columns) - set(right_cols))
        df = self.df.select(non_right_cols + mod_right_cols)
        # drop the null left-hand side rows
        df = (df.filter(sfn.col(left_ts_col).isNotNull()).drop(self.ts_col)).drop(
            "rec_ind"
        )

        # remove the null_ct stats used to record missing values in partitioned as of join
        if tsPartitionVal is not None:
            for column in df.columns:
                if column.startswith("non_null"):
                    # Avoid collect() calls when explicitly ignoring the warnings about null values due to lookback
                    # window. if setting suppress_null_warning to True and warning logger is enabled for other part
                    # of the code, it would make sense to not log warning in this function while allowing other part
                    # of the code to continue to log warning. So it makes more sense for and than or on this line
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

        return TSDF(df, left_ts_col, self.partitionCols)

    def __getTimePartitions(self, tsPartitionVal: int, fraction: float = 0.1) -> "TSDF":
        """
        Create time-partitions for our data-set. We put our time-stamps into brackets of <tsPartitionVal>. Timestamps
        are rounded down to the nearest <tsPartitionVal> seconds.

        We cast our timestamp column to double instead of using f.unix_timestamp, since it provides more precision.

        Additionally, we make these partitions overlapping by adding a remainder df. This way when calculating the
        last right timestamp we will not end up with nulls for the first left timestamp in each partition.

        TODO: change ts_partition to accommodate for higher precision than seconds.
        """
        partition_df = (
            self.df.withColumn(
                "ts_col_double", sfn.col(self.ts_col).cast("double")
            )  # double is preferred over unix_timestamp
            .withColumn(
                "ts_partition",
                sfn.lit(tsPartitionVal)
                * (sfn.col("ts_col_double") / sfn.lit(tsPartitionVal)).cast("integer"),
            )
            .withColumn(
                "partition_remainder",
                (sfn.col("ts_col_double") - sfn.col("ts_partition"))
                / sfn.lit(tsPartitionVal),
            )
            .withColumn("is_original", sfn.lit(1))
        ).cache()  # cache it because it's used twice.

        # add [1 - fraction] of previous time partition to the next partition.
        remainder_df = (
            partition_df.filter(sfn.col("partition_remainder") >= sfn.lit(1 - fraction))
            .withColumn(
                "ts_partition", sfn.col("ts_partition") + sfn.lit(tsPartitionVal)
            )
            .withColumn("is_original", sfn.lit(0))
        )

        df = partition_df.union(remainder_df).drop(
            "partition_remainder", "ts_col_double"
        )
        return TSDF(df, self.ts_col, self.partitionCols + ["ts_partition"])

    #
    # Slicing & Selection
    #

    def select(self, *cols: Union[str, List[str]]) -> "TSDF":
        """
        pyspark.sql.DataFrame.select() method's equivalent for TSDF objects

        :param cols: str or list of strs column names (string). If one of the column names is '*', that
            column is expanded to include all columns in the current :class:`TSDF`.

        ## Examples
        .. code-block:: python
        tsdf.select('*').collect()
        [Row(age=2, name='Alice'), Row(age=5, name='Bob')]
        tsdf.select('name', 'age').collect()
        [Row(name='Alice', age=2), Row(name='Bob', age=5)]

        """

        # The columns which will be a mandatory requirement while selecting from TSDFs
        seq_col_stub = [] if bool(self.sequence_col) is False else [self.sequence_col]
        mandatory_cols = [self.ts_col] + self.partitionCols + seq_col_stub
        if set(mandatory_cols).issubset(set(cols)):
            return TSDF(
                self.df.select(*cols),
                self.ts_col,
                self.partitionCols,
                self.sequence_col,
            )
        else:
            raise Exception(
                "In TSDF's select statement original ts_col, partitionCols and seq_col_stub(optional) must be present"
            )

    def __slice(self, op: str, target_ts: Union[str, int]) -> "TSDF":
        """
        Private method to slice TSDF by time

        :param op: string symbol of the operation to perform
        :type op: str
        :param target_ts: timestamp on which to filter

        :return: a TSDF object containing only those records within the time slice specified
        """
        # quote our timestamp if its a string
        target_expr = f"'{target_ts}'" if isinstance(target_ts, str) else target_ts
        slice_expr = sfn.expr(f"{self.ts_col} {op} {target_expr}")
        sliced_df = self.df.where(slice_expr)
        return TSDF(
            sliced_df,
            ts_col=self.ts_col,
            partition_cols=self.partitionCols,
            sequence_col=self.sequence_col,
        )

    def at(self, ts: Union[str, int]) -> "TSDF":
        """
        Select only records at a given time

        :param ts: timestamp of the records to select

        :return: a :class:`~tsdf.TSDF` object containing just the records at the given time
        """
        return self.__slice("==", ts)

    def before(self, ts: Union[str, int]) -> "TSDF":
        """
        Select only records before a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records before the given time
        """
        return self.__slice("<", ts)

    def atOrBefore(self, ts: Union[str, int]) -> "TSDF":
        """
        Select only records at or before a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records at or before the given time
        """
        return self.__slice("<=", ts)

    def after(self, ts: Union[str, int]) -> "TSDF":
        """
        Select only records after a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records after the given time
        """
        return self.__slice(">", ts)

    def atOrAfter(self, ts: Union[str, int]) -> "TSDF":
        """
        Select only records at or after a given time

        :param ts: timestamp on which to filter records

        :return: a :class:`~tsdf.TSDF` object containing just the records at or after the given time
        """
        return self.__slice(">=", ts)

    def between(
        self, start_ts: Union[str, int], end_ts: Union[str, int], inclusive: bool = True
    ) -> "TSDF":
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

    def __top_rows_per_series(self, win: WindowSpec, n: int) -> "TSDF":
        """
        Private method to select just the top n rows per series (as defined by a window ordering)

        :param win: the window on which we order the rows in each series
        :param n: the number of rows to return

        :return: a :class:`~tsdf.TSDF` object containing just the top n rows in each series
        """
        row_num_col = "__row_num"
        prev_records_df = (
            self.df.withColumn(row_num_col, sfn.row_number().over(win))
            .where(sfn.col(row_num_col) <= sfn.lit(n))
            .drop(row_num_col)
        )
        return TSDF(
            prev_records_df,
            ts_col=self.ts_col,
            partition_cols=self.partitionCols,
            sequence_col=self.sequence_col,
        )

    def earliest(self, n: int = 1) -> "TSDF":
        """
        Select the earliest n records for each series

        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the earliest n records for each series
        """
        prev_window = self.__baseWindow(reverse=False)
        return self.__top_rows_per_series(prev_window, n)

    def latest(self, n: int = 1) -> "TSDF":
        """
        Select the latest n records for each series

        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the latest n records for each series
        """
        next_window = self.__baseWindow(reverse=True)
        return self.__top_rows_per_series(next_window, n)

    def priorTo(self, ts: Union[str, int], n: int = 1) -> "TSDF":
        """
        Select the n most recent records prior to a given time
        You can think of this like an 'asOf' select - it selects the records as of a particular time

        :param ts: timestamp on which to filter records
        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the n records prior to the given time
        """
        return self.atOrBefore(ts).latest(n)

    def subsequentTo(self, ts: Union[str, int], n: int = 1) -> "TSDF":
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

    def show(
        self, n: int = 20, k: int = 5, truncate: bool = True, vertical: bool = False
    ) -> None:
        """
        pyspark.sql.DataFrame.show() method's equivalent for TSDF objects

        :param n: Number of rows to show. (default: 20)
        :param truncate: If set to True, truncate strings longer than 20 chars by default.
            If set to a number greater than one, truncates long strings to length truncate
            and align cells right.
        :param vertical: If set to True, print output rows vertically (one line per column value).

        ## Example to show usage:
        .. code-block:: python
        from pyspark.sql.functions import *

        phone_accel_df = spark.read.format("csv").option("header", "true").load("dbfs:/home/tempo/Phones_accelerometer") \n
            .withColumn("event_ts", (col("Arrival_Time").cast("double")/1000).cast("timestamp")) \n
            .withColumn("x", col("x").cast("double")) \n
            .withColumn("y", col("y").cast("double")) \n
            .withColumn("z", col("z").cast("double")) \n
            .withColumn("event_ts_dbl", col("event_ts").cast("double"))

        from tempo import *

        phone_accel_tsdf = TSDF(phone_accel_df, ts_col="event_ts", partition_cols = ["User"])

        # Call show method here
        phone_accel_tsdf.show()
        """

        # validate k <= n
        if k > n:
            raise ValueError(f"Parameter k {k} cannot be greater than parameter n {n}")

        if not t_utils.IS_DATABRICKS and t_utils.ENV_CAN_RENDER_HTML:
            # In Jupyter notebooks, for wide dataframes the below line will enable
            # rendering the output in a scrollable format.
            ipydisplay(
                HTML("<style>pre { white-space: pre !important; }</style>")
            )  # pragma: no cover
        t_utils.get_display_df(self, k).show(n, truncate, vertical)

    def describe(self) -> DataFrame:
        """
        Describe a TSDF object using a global summary across all time series (anywhere from 10 to millions) as well as the standard Spark data frame stats. Missing vals
        Summary
        global - unique time series based on partition columns, min/max times, granularity - lowest precision in the time series timestamp column
        count / mean / stddev / min / max - standard Spark data frame describe() output
        missing_vals_pct - percentage (from 0 to 100) of missing values.
        """
        # extract the double version of the timestamp column to summarize
        double_ts_col = self.ts_col + "_dbl"

        this_df = self.df.withColumn(double_ts_col, sfn.col(self.ts_col).cast("double"))

        # summary missing value percentages
        missing_vals = this_df.select(
            [
                (
                    100
                    * sfn.count(sfn.when(sfn.col(c[0]).isNull(), c[0]))
                    / sfn.count(sfn.lit(1))
                ).alias(c[0])
                for c in this_df.dtypes
                if c[1] != "timestamp"
            ]
        ).select(sfn.lit("missing_vals_pct").alias("summary"), "*")

        # describe stats
        desc_stats = this_df.describe().union(missing_vals)
        unique_ts = this_df.select(*self.partitionCols).distinct().count()

        max_ts = this_df.select(
            sfn.max(sfn.col(self.ts_col)).alias("max_ts")
        ).collect()[0][0]
        min_ts = this_df.select(
            sfn.min(sfn.col(self.ts_col)).alias("max_ts")
        ).collect()[0][0]
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
            sfn.col("summary"),
            sfn.lit(" ").alias("unique_ts_count"),
            sfn.lit(" ").alias("min_ts"),
            sfn.lit(" ").alias("max_ts"),
            sfn.lit(" ").alias("granularity"),
            *non_summary_cols,
        )

        # add in single record with global summary attributes and the previously computed missing value and Spark data frame describe stats
        global_smry_rec = desc_stats.limit(1).select(
            sfn.lit("global").alias("summary"),
            sfn.lit(unique_ts).alias("unique_ts_count"),
            sfn.lit(min_ts).alias("min_ts"),
            sfn.lit(max_ts).alias("max_ts"),
            sfn.lit(gran).alias("granularity"),
            *[sfn.lit(" ").alias(c) for c in non_summary_cols],
        )

        full_smry = global_smry_rec.union(desc_stats)
        full_smry = full_smry.withColumnRenamed(
            "unique_ts_count", "unique_time_series_count"
        )

        try:  # pragma: no cover
            dbutils.fs.ls("/")  # type: ignore
            return full_smry
        # TODO: Can we raise something other than generic Exception?
        #  perhaps refactor to check for IS_DATABRICKS
        except Exception:
            return full_smry

    def __getSparkPlan(self, df: DataFrame, spark: SparkSession) -> str:
        """
        Internal helper function to obtain the Spark plan for the input data frame

        Parameters
        :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
        :param spark - Spark session which is used to query the view obtained from the Spark data frame
        """

        df.createOrReplaceTempView("view")
        plan = spark.sql("explain cost select * from view").collect()[0][0]

        return plan

    def __getBytesFromPlan(self, df: DataFrame, spark: SparkSession) -> float:
        """
        Internal helper function to obtain how many bytes in memory the Spark data
        frame is likely to take up. This is an upper bound and is obtained from the
        plan details in Spark

        Parameters
        :param df - input Spark data frame - the AS OF join has 2 data frames; this will be called for each
        :param spark - Spark session which is used to query the view obtained from the Spark data frame
        """

        plan = self.__getSparkPlan(df, spark)

        import re

        search_result = re.search(r"sizeInBytes=.*(['\)])", plan, re.MULTILINE)
        if search_result is not None:
            result = search_result.group(0).replace(")", "")
        else:
            raise ValueError("Unable to obtain sizeInBytes from Spark plan")

        size = result.split("=")[1].split(" ")[0]
        units = result.split("=")[1].split(" ")[1]

        # perform to MB for threshold check
        if units == "GiB":
            plan_bytes = float(size) * 1024 * 1024 * 1024
        elif units == "MiB":
            plan_bytes = float(size) * 1024 * 1024
        elif units == "KiB":
            plan_bytes = float(size) * 1024
        else:
            plan_bytes = float(size)

        return plan_bytes

    def asofJoin(
        self,
        right_tsdf: "TSDF",
        left_prefix: Optional[str] = None,
        right_prefix: str = "right",
        tsPartitionVal: Optional[int] = None,
        fraction: float = 0.5,
        skipNulls: bool = True,
        sql_join_opt: bool = False,
        suppress_null_warning: bool = False,
        tolerance: Optional[int] = None,
    ) -> "TSDF":
        """
        Performs an as-of join between two time-series. If a tsPartitionVal is
        specified, it will do this partitioned by time brackets, which can help alleviate skew.

        NOTE: partition cols have to be the same for both Dataframes. We are
        collecting stats when the WARNING level is enabled also.

        Parameters
        :param right_tsdf - right-hand data frame containing columns to merge in
        :param left_prefix - optional prefix for base data frame
        :param right_prefix - optional prefix for right-hand data frame
        :param tsPartitionVal - value to break up each partition into time brackets
        :param fraction - overlap fraction
        :param skipNulls - whether to skip nulls when joining in values
        :param sql_join_opt - if set to True, will use standard Spark SQL join if it is estimated to be efficient
        :param suppress_null_warning - when tsPartitionVal is specified, will collect min of each column and raise warnings about null values, set to True to avoid
        :param tolerance - only join values within this tolerance range (inclusive), expressed in number of seconds as a double
        """

        # first block of logic checks whether a standard range join will suffice
        left_df = self.df
        right_df = right_tsdf.df

        # test if the broadcast join will be efficient
        if sql_join_opt:
            spark = SparkSession.builder.getOrCreate()
            left_bytes = self.__getBytesFromPlan(left_df, spark)
            right_bytes = self.__getBytesFromPlan(right_df, spark)

            # choose 30MB as the cutoff for the broadcast
            bytes_threshold = 30 * 1024 * 1024
            if (left_bytes < bytes_threshold) or (right_bytes < bytes_threshold):
                spark.conf.set("spark.databricks.optimizer.rangeJoin.binSize", 60)
                partition_cols = right_tsdf.partitionCols
                left_cols = list(set(left_df.columns) - set(self.partitionCols))
                right_cols = list(set(right_df.columns) - set(right_tsdf.partitionCols))

                left_prefix = left_prefix + "_" if left_prefix else ""
                right_prefix = right_prefix + "_" if right_prefix else ""

                w = Window.partitionBy(*partition_cols).orderBy(
                    right_prefix + right_tsdf.ts_col
                )

                new_left_ts_col = left_prefix + self.ts_col
                new_left_cols = [
                    sfn.col(c).alias(left_prefix + c) for c in left_cols
                ] + partition_cols
                new_right_cols = [
                    sfn.col(c).alias(right_prefix + c) for c in right_cols
                ] + partition_cols
                quotes_df_w_lag = right_df.select(*new_right_cols).withColumn(
                    "lead_" + right_tsdf.ts_col,
                    sfn.lead(right_prefix + right_tsdf.ts_col).over(w),
                )
                left_df = left_df.select(*new_left_cols)
                res = (
                    left_df.join(quotes_df_w_lag, partition_cols)
                    .where(
                        left_df[new_left_ts_col].between(
                            sfn.col(right_prefix + right_tsdf.ts_col),
                            sfn.coalesce(
                                sfn.col("lead_" + right_tsdf.ts_col),
                                sfn.lit("2099-01-01").cast("timestamp"),
                            ),
                        )
                    )
                    .drop("lead_" + right_tsdf.ts_col)
                )
                return TSDF(
                    res, partition_cols=self.partitionCols, ts_col=new_left_ts_col
                )

        # end of block checking to see if standard Spark SQL join will work

        if tsPartitionVal is not None:
            logger.warning(
                "You are using the skew version of the AS OF join. This may result in null values if there are any "
                "values outside of the maximum lookback. For maximum efficiency, choose smaller values of maximum "
                "lookback, trading off performance and potential blank AS OF values for sparse keys"
            )

        # Check whether partition columns have same name in both dataframes
        self.__checkPartitionCols(right_tsdf)

        # prefix non-partition columns, to avoid duplicated columns.
        left_df = self.df
        right_df = right_tsdf.df

        # validate timestamp datatypes match
        self.__validateTsColMatch(right_tsdf)

        orig_left_col_diff = list(
            set(left_df.columns).difference(set(self.partitionCols))
        )
        orig_right_col_diff = list(
            set(right_df.columns).difference(set(self.partitionCols))
        )

        left_tsdf = (
            (self.__addPrefixToColumns([self.ts_col] + orig_left_col_diff, left_prefix))
            if left_prefix is not None
            else self
        )
        right_tsdf = right_tsdf.__addPrefixToColumns(
            [right_tsdf.ts_col] + orig_right_col_diff, right_prefix
        )

        left_columns = list(
            set(left_tsdf.df.columns).difference(set(self.partitionCols))
        )
        right_columns = list(
            set(right_tsdf.df.columns).difference(set(self.partitionCols))
        )

        # Union both dataframes, and create a combined TS column
        combined_ts_col = "combined_ts"
        combined_df = left_tsdf.__addColumnsFromOtherDF(right_columns).__combineTSDF(
            right_tsdf.__addColumnsFromOtherDF(left_columns), combined_ts_col
        )
        combined_df.df = combined_df.df.withColumn(
            "rec_ind",
            sfn.when(sfn.col(left_tsdf.ts_col).isNotNull(), 1).otherwise(-1),
        )

        # perform asof join.
        if tsPartitionVal is None:
            asofDF = combined_df.__getLastRightRow(
                left_tsdf.ts_col,
                right_columns,
                right_tsdf.sequence_col,
                tsPartitionVal,
                skipNulls,
                suppress_null_warning,
            )
        else:
            tsPartitionDF = combined_df.__getTimePartitions(
                tsPartitionVal, fraction=fraction
            )
            asofDF = tsPartitionDF.__getLastRightRow(
                left_tsdf.ts_col,
                right_columns,
                right_tsdf.sequence_col,
                tsPartitionVal,
                skipNulls,
                suppress_null_warning,
            )

            # Get rid of overlapped data and the extra columns generated from timePartitions
            df = asofDF.df.filter(sfn.col("is_original") == 1).drop(
                "ts_partition", "is_original"
            )

            asofDF = TSDF(df, asofDF.ts_col, combined_df.partitionCols)

        if tolerance is not None:
            df = asofDF.df
            left_ts_col = left_tsdf.ts_col
            right_ts_col = right_tsdf.ts_col
            tolerance_condition = (
                df[left_ts_col].cast("double") - df[right_ts_col].cast("double")
                > tolerance
            )

            for right_col in right_columns:
                # First set right non-timestamp columns to null for rows outside of tolerance band
                if right_col != right_ts_col:
                    df = df.withColumn(
                        right_col,
                        sfn.when(tolerance_condition, sfn.lit(None)).otherwise(
                            df[right_col]
                        ),
                    )

            # Finally, set right timestamp column to null for rows outside of tolerance band
            df = df.withColumn(
                right_ts_col,
                sfn.when(tolerance_condition, sfn.lit(None)).otherwise(
                    df[right_ts_col]
                ),
            )
            asofDF.df = df

        return asofDF

    def __baseWindow(
        self, sort_col: Optional[str] = None, reverse: bool = False
    ) -> WindowSpec:
        # figure out our sorting columns
        primary_sort_col = self.ts_col if not sort_col else sort_col
        sort_cols = (
            [primary_sort_col, self.sequence_col]
            if self.sequence_col
            else [primary_sort_col]
        )

        # are we ordering forwards (default) or reveresed?
        col_fn = sfn.col
        if reverse:
            col_fn = lambda colname: sfn.col(colname).desc()  # noqa E731

        # our window will be sorted on our sort_cols in the appropriate direction
        w = Window().orderBy([col_fn(col) for col in sort_cols])
        # and partitioned by any series IDs
        if self.partitionCols:
            w = w.partitionBy([sfn.col(elem) for elem in self.partitionCols])
        return w

    def __rangeBetweenWindow(
        self,
        range_from: int,
        range_to: int,
        sort_col: Optional[str] = None,
        reverse: bool = False,
    ) -> WindowSpec:
        return self.__baseWindow(sort_col=sort_col, reverse=reverse).rangeBetween(
            range_from, range_to
        )

    def __rowsBetweenWindow(
        self,
        rows_from: int,
        rows_to: int,
        reverse: bool = False,
    ) -> WindowSpec:
        return self.__baseWindow(reverse=reverse).rowsBetween(rows_from, rows_to)

    def withPartitionCols(self, partitionCols: list[str]) -> "TSDF":
        """
        Sets certain columns of the TSDF as partition columns. Partition columns are those that differentiate distinct timeseries
        from each other.
        :param partitionCols: a list of columns used to partition distinct timeseries
        :return: a TSDF object with the given partition columns
        """
        return TSDF(self.df, self.ts_col, partitionCols)

    def vwap(
        self,
        frequency: str = "m",
        volume_col: str = "volume",
        price_col: str = "price",
    ) -> "TSDF":
        # set pre_vwap as self or enrich with the frequency
        pre_vwap = self.df
        if frequency == "m":
            pre_vwap = self.df.withColumn(
                "time_group",
                sfn.concat(
                    sfn.lpad(sfn.hour(sfn.col(self.ts_col)), 2, "0"),
                    sfn.lit(":"),
                    sfn.lpad(sfn.minute(sfn.col(self.ts_col)), 2, "0"),
                ),
            )
        elif frequency == "H":
            pre_vwap = self.df.withColumn(
                "time_group",
                sfn.concat(sfn.lpad(sfn.hour(sfn.col(self.ts_col)), 2, "0")),
            )
        elif frequency == "D":
            pre_vwap = self.df.withColumn(
                "time_group",
                sfn.concat(sfn.lpad(sfn.day(sfn.col(self.ts_col)), 2, "0")),
            )

        group_cols = ["time_group"]
        if self.partitionCols:
            group_cols.extend(self.partitionCols)
        vwapped = (
            pre_vwap.withColumn("dllr_value", sfn.col(price_col) * sfn.col(volume_col))
            .groupby(group_cols)
            .agg(
                sfn.sum("dllr_value").alias("dllr_value"),
                sfn.sum(volume_col).alias(volume_col),
                sfn.max(price_col).alias("_".join(["max", price_col])),
            )
            .withColumn("vwap", sfn.col("dllr_value") / sfn.col(volume_col))
        )

        return TSDF(vwapped, self.ts_col, self.partitionCols)

    def EMA(self, colName: str, window: int = 30, exp_factor: float = 0.2) -> "TSDF":
        """
        Constructs an approximate EMA in the fashion of:
        EMA = e * lag(col,0) + e * (1 - e) * lag(col, 1) + e * (1 - e)^2 * lag(col, 2) etc, up until window
        TODO: replace case when statement with coalesce
        TODO: add in time partitions functionality (what is the overlap fraction?)
        """

        emaColName = "_".join(["EMA", colName])
        df = self.df.withColumn(emaColName, sfn.lit(0)).orderBy(self.ts_col)
        w = self.__baseWindow()
        # Generate all the lag columns:
        for i in range(window):
            lagColName = "_".join(["lag", colName, str(i)])
            weight = exp_factor * (1 - exp_factor) ** i
            df = df.withColumn(
                lagColName, weight * sfn.lag(sfn.col(colName), i).over(w)
            )
            df = df.withColumn(
                emaColName,
                sfn.col(emaColName)
                + sfn.when(sfn.col(lagColName).isNull(), sfn.lit(0)).otherwise(
                    sfn.col(lagColName)
                ),
            ).drop(lagColName)
            # Nulls are currently removed

        return TSDF(df, self.ts_col, self.partitionCols)

    def withLookbackFeatures(
        self,
        featureCols: List[str],
        lookbackWindowSize: int,
        exactSize: bool = True,
        featureColName: str = "features",
    ) -> Union[DataFrame | "TSDF"]:
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
        feat_array_tsdf = self.df.withColumn(tempArrayColName, sfn.array(featureCols))

        # construct a lookback array
        lookback_win = self.__rowsBetweenWindow(-lookbackWindowSize, -1)
        lookback_tsdf = feat_array_tsdf.withColumn(
            featureColName,
            sfn.collect_list(sfn.col(tempArrayColName)).over(lookback_win),
        ).drop(tempArrayColName)

        # make sure only windows of exact size are allowed
        if exactSize:
            return lookback_tsdf.where(sfn.size(featureColName) == lookbackWindowSize)

        return TSDF(lookback_tsdf, self.ts_col, self.partitionCols)

    def withRangeStats(
        self,
        type: str = "range",
        colsToSummarize: Optional[List[Column]] = None,
        rangeBackWindowSecs: int = 1000,
    ) -> "TSDF":
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

        # identify columns to summarize if not provided
        # these should include all numeric columns that
        # are not the timestamp column and not any of the partition columns
        if colsToSummarize is None:
            # columns we should never summarize
            prohibited_cols = [self.ts_col.lower()]
            if self.partitionCols:
                prohibited_cols.extend([pc.lower() for pc in self.partitionCols])
            # types that can be summarized
            summarizable_types = ["int", "bigint", "float", "double"]
            # filter columns to find summarizable columns
            colsToSummarize = [
                datatype[0]
                for datatype in self.df.dtypes
                if (
                    (datatype[1] in summarizable_types)
                    and (datatype[0].lower() not in prohibited_cols)
                )
            ]

        # build window
        if isinstance(self.df.schema[self.ts_col].dataType, TimestampType):
            self.df = self.__add_double_ts()
            prohibited_cols.extend(["double_ts"])
            w = self.__rangeBetweenWindow(
                -1 * rangeBackWindowSecs, 0, sort_col="double_ts"
            )
        else:
            w = self.__rangeBetweenWindow(-1 * rangeBackWindowSecs, 0)

        # compute column summaries
        selectedCols = self.df.columns
        derivedCols = []
        for metric in colsToSummarize:
            selectedCols.append(sfn.mean(metric).over(w).alias("mean_" + metric))
            selectedCols.append(sfn.count(metric).over(w).alias("count_" + metric))
            selectedCols.append(sfn.min(metric).over(w).alias("min_" + metric))
            selectedCols.append(sfn.max(metric).over(w).alias("max_" + metric))
            selectedCols.append(sfn.sum(metric).over(w).alias("sum_" + metric))
            selectedCols.append(sfn.stddev(metric).over(w).alias("stddev_" + metric))
            derivedCols.append(
                (
                    (sfn.col(metric) - sfn.col("mean_" + metric))
                    / sfn.col("stddev_" + metric)
                ).alias("zscore_" + metric)
            )
        selected_df = self.df.select(*selectedCols)
        summary_df = selected_df.select(*selected_df.columns, *derivedCols).drop(
            "double_ts"
        )

        return TSDF(summary_df, self.ts_col, self.partitionCols)

    def withGroupedStats(
        self,
        metricCols: Optional[List[str]] = None,
        freq: Optional[str] = None,
    ) -> "TSDF":
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
        if metricCols is None:
            # columns we should never summarize
            prohibited_cols = [self.ts_col.lower()]
            if self.partitionCols:
                prohibited_cols.extend([pc.lower() for pc in self.partitionCols])
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
        parsed_freq = t_resample.checkAllowableFreq(freq)
        period, unit = parsed_freq[0], parsed_freq[1]
        agg_window = sfn.window(
            sfn.col(self.ts_col),
            "{} {}".format(
                period, t_resample.freq_dict[unit]  # type: ignore[literal-required]
            ),
        )

        # compute column summaries
        selectedCols = []
        for metric in metricCols:
            selectedCols.extend(
                [
                    sfn.mean(sfn.col(metric)).alias("mean_" + metric),
                    sfn.count(sfn.col(metric)).alias("count_" + metric),
                    sfn.min(sfn.col(metric)).alias("min_" + metric),
                    sfn.max(sfn.col(metric)).alias("max_" + metric),
                    sfn.sum(sfn.col(metric)).alias("sum_" + metric),
                    sfn.stddev(sfn.col(metric)).alias("stddev_" + metric),
                ]
            )

        selected_df = self.df.groupBy(self.partitionCols + [agg_window]).agg(
            *selectedCols
        )
        summary_df = (
            selected_df.select(*selected_df.columns)
            .withColumn(self.ts_col, sfn.col("window").start)
            .drop("window")
        )

        return TSDF(summary_df, self.ts_col, self.partitionCols)

    def write(
        self,
        spark: SparkSession,
        tabName: str,
        optimizationCols: Optional[List[str]] = None,
    ) -> None:
        t_io.write(self, spark, tabName, optimizationCols)

    def resample(
        self,
        freq: str,
        func: Union[Callable | str],
        metricCols: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        fill: Optional[bool] = None,
        perform_checks: bool = True,
    ) -> "TSDF":
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
        t_resample.validateFuncExists(func)

        # Throw warning for user to validate that the expected number of output rows is valid.
        if fill is True and perform_checks is True:
            t_utils.calculate_time_horizon(
                self.df, self.ts_col, freq, self.partitionCols
            )

        enriched_df: DataFrame = t_resample.aggregate(
            self, freq, func, metricCols, prefix, fill
        )
        return _ResampledTSDF(
            enriched_df,
            ts_col=self.ts_col,
            partition_cols=self.partitionCols,
            freq=freq,
            func=func,
        )

    def interpolate(
        self,
        method: str,
        freq: Optional[str] = None,
        func: Optional[Union[Callable | str]] = None,
        target_cols: Optional[List[str]] = None,
        ts_col: Optional[str] = None,
        partition_cols: Optional[List[str]] = None,
        show_interpolated: bool = False,
        perform_checks: bool = True,
    ) -> "TSDF":
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
        if freq is None:
            raise ValueError("freq must be provided")
        if func is None:
            raise ValueError("func must be provided")
        if ts_col is None:
            ts_col = self.ts_col
        if partition_cols is None:
            partition_cols = self.partitionCols
        if target_cols is None:
            prohibited_cols: List[str] = partition_cols + [ts_col]
            summarizable_types = ["int", "bigint", "float", "double"]

            # get summarizable find summarizable columns
            target_cols = [
                datatype[0]
                for datatype in self.df.dtypes
                if (
                    (datatype[1] in summarizable_types)
                    and (datatype[0].lower() not in prohibited_cols)
                )
            ]

        interpolate_service = t_interpolation.Interpolation(is_resampled=False)
        tsdf_input = TSDF(self.df, ts_col=ts_col, partition_cols=partition_cols)
        interpolated_df: DataFrame = interpolate_service.interpolate(
            tsdf_input,
            ts_col,
            partition_cols,
            target_cols,
            freq,
            func,
            method,
            show_interpolated,
            perform_checks,
        )

        return TSDF(interpolated_df, ts_col=ts_col, partition_cols=partition_cols)

    def calc_bars(
        tsdf,
        freq: str,
        metricCols: Optional[List[str]] = None,
        fill: Optional[bool] = None,
    ) -> "TSDF":
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

        join_cols = resample_open.partitionCols + [resample_open.ts_col]
        bars = (
            resample_open.df.join(resample_high.df, join_cols)
            .join(resample_low.df, join_cols)
            .join(resample_close.df, join_cols)
        )
        non_part_cols = set(set(bars.columns) - set(resample_open.partitionCols)) - set(
            [resample_open.ts_col]
        )
        sel_and_sort = (
            resample_open.partitionCols + [resample_open.ts_col] + sorted(non_part_cols)
        )
        bars = bars.select(sel_and_sort)

        return TSDF(bars, resample_open.ts_col, resample_open.partitionCols)

    def fourier_transform(
        self, timestep: Union[int | float | complex], valueCol: str
    ) -> "TSDF":
        """
        Function to fourier transform the time series to its frequency domain representation.
        :param timestep: timestep value to be used for getting the frequency scale
        :param valueCol: name of the time domain data column which will be transformed
        """

        def tempo_fourier_util(
            pdf: pd.DataFrame,
        ) -> pd.DataFrame:
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

        valueCol = self.__validated_column(self.df, valueCol)
        data = self.df
        if self.sequence_col:
            if self.partitionCols == []:
                data = data.withColumn("dummy_group", sfn.lit("dummy_val"))
                data = (
                    data.select(
                        sfn.col("dummy_group"),
                        self.ts_col,
                        self.sequence_col,
                        sfn.col(valueCol),
                    )
                    .withColumn("tdval", sfn.col(valueCol))
                    .withColumn("tpoints", sfn.col(self.ts_col))
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
                group_cols = self.partitionCols
                data = (
                    data.select(
                        *group_cols,
                        self.ts_col,
                        self.sequence_col,
                        sfn.col(valueCol),
                    )
                    .withColumn("tdval", sfn.col(valueCol))
                    .withColumn("tpoints", sfn.col(self.ts_col))
                )
                return_schema = ",".join(
                    [f"{i[0]} {i[1]}" for i in data.dtypes]
                    + ["freq double", "ft_real double", "ft_imag double"]
                )
                result = data.groupBy(*group_cols).applyInPandas(
                    tempo_fourier_util, return_schema
                )
                result = result.drop("tdval", "tpoints")
        else:
            if self.partitionCols == []:
                data = data.withColumn("dummy_group", sfn.lit("dummy_val"))
                data = (
                    data.select(sfn.col("dummy_group"), self.ts_col, sfn.col(valueCol))
                    .withColumn("tdval", sfn.col(valueCol))
                    .withColumn("tpoints", sfn.col(self.ts_col))
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
                group_cols = self.partitionCols
                data = (
                    data.select(*group_cols, self.ts_col, sfn.col(valueCol))
                    .withColumn("tdval", sfn.col(valueCol))
                    .withColumn("tpoints", sfn.col(self.ts_col))
                )
                return_schema = ",".join(
                    [f"{i[0]} {i[1]}" for i in data.dtypes]
                    + ["freq double", "ft_real double", "ft_imag double"]
                )
                result = data.groupBy(*group_cols).applyInPandas(
                    tempo_fourier_util, return_schema
                )
                result = result.drop("tdval", "tpoints")

        return TSDF(result, self.ts_col, self.partitionCols, self.sequence_col)

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
                sfn.when(col1.isNull() & col2.isNull(), True)
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

            def state_comparison_fn(a: CT, b: CT) -> Callable[[Column, Column], Column]:
                return operator_dict[state_definition](a, b)

        elif callable(state_definition):
            state_comparison_fn = state_definition  # type: ignore

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
            sfn.lag(sfn.col(self.ts_col), offset=1).over(w),
        )

        # Determine state intervals using user-provided the state comparison function
        # The comparison occurs on the current and previous record per metric column
        temp_metric_compare_cols = []
        for mc in metric_cols:
            temp_metric_compare_col = f"__{mc}_compare"
            data = data.withColumn(
                temp_metric_compare_col,
                state_comparison_fn(sfn.col(mc), sfn.lag(sfn.col(mc), 1).over(w)),
            )
            temp_metric_compare_cols.append(temp_metric_compare_col)

        # Remove first record which will have no state change
        # and produces `null` for all state comparisons
        data = data.filter(sfn.col("previous_ts").isNotNull())

        # Each state comparison should return True if state remained constant
        data = data.withColumn(
            "state_change",
            sfn.array_contains(sfn.array(*temp_metric_compare_cols), False),
        )

        # Count the distinct state changes to get the unique intervals
        data = data.withColumn(
            "state_incrementer",
            sfn.sum(sfn.col("state_change").cast("int")).over(w),
        ).filter(~sfn.col("state_change"))

        # Find the start and end timestamp of the interval
        result = (
            data.groupBy(*self.partitionCols, "state_incrementer")
            .agg(
                sfn.min("previous_ts").alias("start_ts"),
                sfn.max(self.ts_col).alias("end_ts"),
            )
            .drop("state_incrementer")
        )

        return result


class _ResampledTSDF(TSDF):
    def __init__(
        self,
        df: DataFrame,
        freq: str,
        func: Union[Callable | str],
        ts_col: str = "event_ts",
        partition_cols: Optional[List[str]] = None,
        sequence_col: Optional[str] = None,
    ):
        super(_ResampledTSDF, self).__init__(df, ts_col, partition_cols, sequence_col)
        self.__freq = freq
        self.__func = func

    def interpolate(
        self,
        method: str,
        freq: Optional[str] = None,
        func: Optional[Union[Callable | str]] = None,
        target_cols: Optional[List[str]] = None,
        ts_col: Optional[str] = None,
        partition_cols: Optional[List[str]] = None,
        show_interpolated: bool = False,
        perform_checks: bool = True,
    ) -> "TSDF":
        """
        Function to interpolate based on frequency, aggregation, and fill similar to pandas. This method requires an already sampled data set in order to use.

        :param method: function used to fill missing values e.g. linear, null, zero, bfill, ffill
        :param target_cols [optional]: columns that should be interpolated, by default interpolates all numeric columns
        :param show_interpolated [optional]: if true will include an additional column to show which rows have been fully interpolated.
        :param perform_checks: calculate time horizon and warnings if True (default is True)
        :return: new TSDF object containing interpolated data
        """

        if freq is None:
            freq = self.__freq

        if func is None:
            func = self.__func

        if ts_col is None:
            ts_col = self.ts_col

        if partition_cols is None:
            partition_cols = self.partitionCols

        # Set defaults for target columns, timestamp column and partition columns when not provided
        if target_cols is None:
            prohibited_cols: List[str] = self.partitionCols + [self.ts_col]
            summarizable_types = ["int", "bigint", "float", "double"]

            # get summarizable find summarizable columns
            target_cols = [
                datatype[0]
                for datatype in self.df.dtypes
                if (
                    (datatype[1] in summarizable_types)
                    and (datatype[0].lower() not in prohibited_cols)
                )
            ]

        interpolate_service = t_interpolation.Interpolation(is_resampled=True)
        tsdf_input = TSDF(
            self.df, ts_col=self.ts_col, partition_cols=self.partitionCols
        )
        interpolated_df = interpolate_service.interpolate(
            tsdf=tsdf_input,
            ts_col=self.ts_col,
            partition_cols=self.partitionCols,
            target_cols=target_cols,
            freq=freq,
            func=func,
            method=method,
            show_interpolated=show_interpolated,
            perform_checks=perform_checks,
        )

        return TSDF(
            interpolated_df, ts_col=self.ts_col, partition_cols=self.partitionCols
        )


class Comparable(metaclass=ABCMeta):
    """For typing functions generated by operator_dict"""

    @abstractmethod
    def __ne__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __le__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __gt__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __ge__(self, other: Any) -> bool:
        pass


CT = TypeVar("CT", bound=Comparable)
