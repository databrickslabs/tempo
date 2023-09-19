from __future__ import annotations

import copy
import logging
import operator
from abc import ABCMeta, abstractmethod
from functools import cached_property, reduce
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, TypeVar, \
    Union, \
    cast, overload

import pyspark.sql.functions as sfn
from IPython.core.display import HTML
from IPython.display import display as ipydisplay
from pyspark.sql import GroupedData, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import DataType, StructType
from pyspark.sql._typing import ColumnOrName
from pyspark.sql.window import Window, WindowSpec

import tempo.interpol as t_interpolation
import tempo.io as t_io
import tempo.resample as t_resample
import tempo.utils as t_utils
from tempo.intervals import IntervalsDF
from tempo.tsschema import CompositeTSIndex, TSIndex, TSSchema, WindowBuilder

logger = logging.getLogger(__name__)

# default column name for constructed timeseries index struct columns
DEFAULT_TS_IDX_COL = "ts_idx"

def makeStructFromCols(df: DataFrame,
                       struct_col_name: str,
                       cols_to_move: Collection[str]) -> DataFrame:
    """
    Transform a :class:`DataFrame` by moving certain columns into a struct

    :param df: the :class:`DataFrame` to transform
    :param struct_col_name: name of the struct column to create
    :param cols_to_move: name of the columns to move into the struct

    :return: the transformed :class:`DataFrame`
    """
    return (df.withColumn(struct_col_name,
                          sfn.struct(*cols_to_move))
              .drop(*cols_to_move))

class TSDF(WindowBuilder):
    """
    This object is the main wrapper over a Spark data frame which allows a user to parallelize time series computations on a Spark data frame by various dimensions. The two dimensions required are partition_cols (list of columns by which to summarize) and ts_col (timestamp column, which can be epoch or TimestampType).
    """

    def __init__(self,
                 df: DataFrame,
                 ts_schema: Optional[TSSchema] = None,
                 ts_col: Optional[str] = None,
                 series_ids: Optional[Collection[str]] = None) -> None:
        self.df = df
        # construct schema if we don't already have one
        if ts_schema:
            self.ts_schema = ts_schema
        else:
            self.ts_schema = TSSchema.fromDFSchema(self.df.schema, ts_col, series_ids)
        # validate that this schema works for this DataFrame
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
        return TSDF(new_df, ts_schema=copy.deepcopy(self.ts_schema))

    def __withStandardizedColOrder(self) -> TSDF:
        """
        Standardizes the column ordering as such:
        * series_ids,
        * ts_index,
        * observation columns

        :return: a :class:`TSDF` with the columns reordered into "standard order" (as described above)
        """
        std_ordered_cols = (
            list(self.series_ids) + [self.ts_index.colname] + list(self.observational_cols)
        )

        return self.__withTransformedDF(self.df.select(std_ordered_cols))

    @classmethod
    def makeCompositeIndexTSDF(
        cls,
        df: DataFrame,
        ts_index_cols: Collection[str],
        series_ids: Collection[str] = None,
        other_index_cols: Collection[str] = None,
        composite_index_name: str = DEFAULT_TS_IDX_COL) -> "TSDF":
        """
        Construct a TSDF with a composite index from the specified columns
        """
        # move all the index columns into a struct
        all_composite_cols = set(ts_index_cols).union(set(other_index_cols or []))
        with_struct_df = makeStructFromCols(df, composite_index_name, all_composite_cols)
        # construct an appropriate TSIndex
        ts_idx_struct = with_struct_df.schema[DEFAULT_TS_IDX_COL]
        comp_idx = CompositeTSIndex(ts_idx_struct, *ts_index_cols)
        # construct & return the TSDF with appropriate schema
        return TSDF(with_struct_df, ts_schema=TSSchema(comp_idx, series_ids))

    @classmethod
    def fromSubsequenceCol(
        cls,
        df: DataFrame,
        ts_col: str,
        subsequence_col: str,
        series_ids: Collection[str] = None,
    ) -> "TSDF":
        return cls.makeCompositeIndexTSDF(df, [ts_col, subsequence_col], series_ids)

    @classmethod
    def fromTimestampString(
        cls,
        df: DataFrame,
        ts_col: str,
        series_ids: Collection[str] = None,
        ts_fmt: str = "YYYY-MM-DDThh:mm:ss[.SSSSSS]",
    ) -> "TSDF":
        pass

    @classmethod
    def fromDateString(
        cls,
        df: DataFrame,
        ts_col: str,
        series_ids: Collection[str],
        date_fmt: str = "YYYY-MM-DD",
    ) -> "TSDF ":
        pass

    @property
    def ts_index(self) -> "TSIndex":
        return self.ts_schema.ts_idx

    @property
    def ts_col(self) -> str:
        return self.ts_index.colname

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

    def __addPrefixToColumns(self, col_list: list[str], prefix: str) -> "TSDF":
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

    def __addColumnsFromOtherDF(self, other_cols: Sequence[str]) -> "TSDF":
        """
        Add columns from some other DF as lit(None), as pre-step before union.
        """
        new_df = reduce(
            lambda df, idx: df.withColumn(other_cols[idx], sfn.lit(None)),
            range(len(other_cols)),
            self.df,
        )

        return self.__withTransformedDF(new_df)

    def __combineTSDF(self, ts_df_right: "TSDF", combined_ts_col: str) -> "TSDF":
        combined_df = self.df.unionByName(ts_df_right.df).withColumn(
            combined_ts_col, sfn.coalesce(self.ts_col, ts_df_right.ts_col)
        )

        return TSDF(combined_df, ts_col=combined_ts_col, series_ids=self.series_ids)

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
        ptntl_sort_keys = [self.ts_col, "rec_ind", sequence_col]
        sort_keys = [sfn.col(col_name) for col_name in ptntl_sort_keys if col_name]

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
                    sfn.last(
                        sfn.when(
                            sfn.col("rec_ind") == -1, sfn.struct(right_cols[idx])
                        ).otherwise(None),
                        True,  # ignore nulls because it indicates rows from the left side
                    ).over(window_spec),
                ),
                range(len(right_cols)),
                self.df,
            )
            df = reduce(
                lambda df, idx: df.withColumn(
                    right_cols[idx], sfn.col(right_cols[idx])[right_cols[idx]]
                ),
                range(len(right_cols)),
                df,
            )
        elif tsPartitionVal is None:
            # splitting off the condition as we want different columns in the reduce if implementing the skew AS OF join
            df = reduce(
                lambda df, idx: df.withColumn(
                    right_cols[idx],
                    sfn.last(right_cols[idx], ignoreNulls).over(window_spec),
                ),
                range(len(right_cols)),
                self.df,
            )
        else:
            df = reduce(
                lambda df, idx: df.withColumn(
                    right_cols[idx],
                    sfn.last(right_cols[idx], ignoreNulls).over(window_spec),
                ).withColumn(
                    "non_null_ct" + right_cols[idx],
                    sfn.count(right_cols[idx]).over(window_spec),
                ),
                range(len(right_cols)),
                self.df,
            )

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

        return TSDF(df, ts_col=left_ts_col, series_ids=self.series_ids)

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
        return TSDF(df, ts_col=self.ts_col, series_ids=self.series_ids + ["ts_partition"])

    #
    # Slicing & Selection
    #

    def select(self, *cols: Union[str, Column]) -> TSDF:
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
        selected_df = self.df.select(*cols)
        return self.__withTransformedDF(selected_df)

    def where(self, condition: ColumnOrName) -> "TSDF":
        """
        Selects rows using the given condition.

        :param condition: a :class:`Column` of :class:`types.BooleanType` or a string of SQL expression.

        :return: a new :class:`TSDF` object
        :rtype: :class:`TSDF`
        """
        where_df = self.df.where(condition)
        return self.__withTransformedDF(where_df)

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
        return self.__withTransformedDF(sliced_df)

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
        return self.__withTransformedDF(prev_records_df)

    def earliest(self, n: int = 1) -> "TSDF":
        """
        Select the earliest n records for each series

        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the earliest n records for each series
        """
        prev_window = self.baseWindow(reverse=False)
        return self.__top_rows_per_series(prev_window, n)

    def latest(self, n: int = 1) -> "TSDF":
        """
        Select the latest n records for each series

        :param n: number of records to select (default is 1)

        :return: a :class:`~tsdf.TSDF` object containing the latest n records for each series
        """
        next_window = self.baseWindow(reverse=True)
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
        unique_ts = this_df.select(*self.series_ids).distinct().count()

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
                # TODO - call broadcast join here


        # end of block checking to see if standard Spark SQL join will work

        if tsPartitionVal is not None:
            logger.warning(
                "You are using the skew version of the AS OF join. This may result in null values if there are any "
                "values outside of the maximum lookback. For maximum efficiency, choose smaller values of maximum "
                "lookback, trading off performance and potential blank AS OF values for sparse keys"
            )

        # Check whether partition columns have same name in both dataframes
        # self.__checkPartitionCols(right_tsdf)
        #
        # # prefix non-partition columns, to avoid duplicated columns.
        # left_df = self.df
        # right_df = right_tsdf.df
        #
        # # validate timestamp datatypes match
        # self.__validateTsColMatch(right_tsdf)
        #
        # orig_left_col_diff = list(
        #     set(left_df.columns) - set(self.series_ids)
        # )
        # orig_right_col_diff = list(
        #     set(right_df.columns) - set(self.series_ids)
        # )
        #
        # left_tsdf = (
        #     (self.__addPrefixToColumns([self.ts_col] + orig_left_col_diff, left_prefix))
        #     if left_prefix is not None
        #     else self
        # )
        # right_tsdf = right_tsdf.__addPrefixToColumns(
        #     [right_tsdf.ts_col] + orig_right_col_diff, right_prefix
        # )
        #
        # left_nonpartition_cols = list(
        #     set(left_tsdf.df.columns) - set(self.series_ids)
        # )
        # right_nonpartition_cols = list(
        #     set(right_tsdf.df.columns) - set(self.series_ids)
        # )

        # For both dataframes get all non-partition columns (including ts_col)
        # left_columxxmns = [left_tsdf.ts_col] + left_nonpartition_cols
        # right_columns = [right_tsdf.ts_col] + right_nonpartition_cols
        #
        # # Union both dataframes, and create a combined TS column
        # combined_ts_col = "combined_ts"
        # combined_df = left_tsdf.__addColumnsFromOtherDF(right_columns).__combineTSDF(
        #     right_tsdf.__addColumnsFromOtherDF(left_columns), combined_ts_col
        # )
        # combined_df.df = combined_df.df.withColumn(
        #     "rec_ind",
        #     sfn.when(sfn.col(left_tsdf.ts_col).isNotNull(), 1).otherwise(-1),
        # )

        # perform asof join.
        if tsPartitionVal is None:
            seq_col = None
            if isinstance(combined_df.ts_index, CompositeTSIndex):
                seq_col = cast(CompositeTSIndex, combined_df.ts_index).ts_component(1)
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
            if isinstance(tsPartitionDF.ts_index, CompositeTSIndex):
                seq_col = cast(CompositeTSIndex, tsPartitionDF.ts_index).ts_component(1)
            asofDF = tsPartitionDF.__getLastRightRow(
                left_tsdf.ts_col,
                right_columns,
                seq_col,
                tsPartitionVal,
                skipNulls,
                suppress_null_warning,
            )

            # Get rid of overlapped data and the extra columns generated from timePartitions
            df = asofDF.df.filter(sfn.col("is_original") == 1).drop(
                "ts_partition", "is_original"
            )

            asofDF = TSDF(df, ts_col=asofDF.ts_col, series_ids=combined_df.series_ids)

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

    def baseWindow(self, reverse: bool = False) -> WindowSpec:
        return self.ts_schema.baseWindow(reverse=reverse)

    def rowsBetweenWindow(self,
                          start: int,
                          end: int,
                          reverse: bool = False) -> WindowSpec:
        return self.ts_schema.rowsBetweenWindow(start, end, reverse=reverse)

    def rangeBetweenWindow(self,
                           start: int,
                           end: int,
                           reverse: bool = False) -> WindowSpec:
        return self.ts_schema.rangeBetweenWindow(start, end, reverse=reverse)

    #
    # Core Transformations
    #

    def withNaturalOrdering(self, reverse: bool = False) -> "TSDF":
        order_expr = [sfn.col(c) for c in self.series_ids]
        ts_idx_expr = self.ts_index.orderByExpr(reverse)
        if isinstance(ts_idx_expr, list):
            order_expr.extend(ts_idx_expr)
        else:
            order_expr.append(ts_idx_expr)

        return self.__withTransformedDF(self.df.orderBy(order_expr))

    def withColumn(self, colName: str, col: Column) -> "TSDF":
        """
        Returns a new :class:`TSDF` by adding a column or replacing the
        existing column that has the same name.

        :param colName: the name of the new column (or existing column to be replaced)
        :param col: a :class:`Column` expression for the new column definition
        """
        new_df = self.df.withColumn(colName, col)
        return self.__withTransformedDF(new_df)

    def withColumnRenamed(self, existing: str, new: str) -> "TSDF":
        """
        Returns a new :class:`TSDF` with the given column renamed.

        :param existing: name of the existing column to renmame
        :param new: new name for the column
        """

        # create new TSIndex
        new_ts_index = copy.deepcopy(self.ts_index)
        if existing == self.ts_index.colname:
            new_ts_index = new_ts_index.renamed(new)

        # and for series ids
        new_series_ids = self.series_ids
        if existing in self.series_ids:
            # replace column name in series
            new_series_ids = self.series_ids
            new_series_ids[new_series_ids.index(existing)] = new

        # rename the column in the underlying DF
        new_df = self.df.withColumnRenamed(existing, new)

        # return new TSDF
        new_schema = TSSchema(new_ts_index, new_series_ids)
        return TSDF(new_df, ts_schema=new_schema)

    def withColumnTypeChanged(self, colName: str, newType: Union[DataType, str]):
        """

        :param colName:
        :param newType:
        :return:
        """
        new_df = self.df.withColumn(colName, sfn.col(colName).cast(newType))
        return self.__withTransformedDF(new_df)

    @overload
    def drop(self, cols: "ColumnOrName") -> "TSDF":
        ...

    @overload
    def drop(self, *cols: str) -> "TSDF":
        ...

    def drop(self, *cols: "ColumnOrName") -> "TSDF":

        """
        Returns a new :class:`TSDF` that drops the specified column.

        :param cols: name of the column to drop

        :return: new :class:`TSDF` with the column dropped
        :rtype: TSDF
        """
        dropped_df = self.df.drop(*cols)
        return self.__withTransformedDF(dropped_df)

    def mapInPandas(self,
                    func: "PandasMapIterFunction",
                    schema: Union[StructType, str]) -> TSDF:
        """

        :param func:
        :param schema:
        :return:
        """
        mapped_df = self.df.mapInPandas(func,schema)
        return self.__withTransformedDF(mapped_df)

    def union(self, other: TSDF) -> TSDF:
        # union of the underlying DataFrames
        union_df = self.df.union(other.df)
        return self.__withTransformedDF(union_df)

    def unionByName(self, other: TSDF, allowMissingColumns: bool = False) -> TSDF:
        # union of the underlying DataFrames
        union_df = self.df.unionByName(
            other.df, allowMissingColumns=allowMissingColumns
        )
        return self.__withTransformedDF(union_df)

    #
    # Rolling (Windowed) Transformations
    #

    def rollingAgg(self,
                   window: WindowSpec,
                   *exprs: Union[Column, Dict[str, str]]) -> TSDF:
        """

        :param window:
        :param exprs:
        :return:
        """
        roll_agg_tsdf = self
        if len(exprs) == 1 and isinstance(exprs[0], dict):
            # dict
            for input_col in exprs.keys():
                expr_str = exprs[input_col]
                new_col_name = f"{expr_str}({input_col})"
                roll_agg_tsdf = roll_agg_tsdf.withColumn(new_col_name,
                                                         sfn.expr(expr_str).over(window))
        else:
            # Columns
            assert all(isinstance(c, Column) for c in exprs), \
                "all exprs should be Column"
            for expr in exprs:
                new_col_name = f"{expr}"
                roll_agg_tsdf = roll_agg_tsdf.withColumn(new_col_name,
                                                         expr.over(window))

        return roll_agg_tsdf

    def rollingApply(self,
                     outputCol: str,
                     window: WindowSpec,
                     func: "PandasGroupedMapFunction",
                     schema: Union[StructType, str],
                     *inputCols: Union[str, Column]) -> TSDF:
        """

        :param outputCol:
        :param window:
        :param func:
        :param schema:
        :param inputCols:
        :return:
        """
        inputCols = [sfn.col(col) for col in inputCols if not isinstance(col, Column)]
        pd_udf = sfn.pandas_udf(func, schema)
        return self.withColumn(outputCol, pd_udf(*inputCols).over(window))

    #
    # Aggregations
    #

    ## Aggregations across series and time

    def summarize(self, *cols: Optional[Union[str, List[str]]]) -> GroupedData:
        """
        Groups the underlying :class:`DataFrame` such that the user can compute
        aggregations over the given columns.
        If no columns are specified, all metric columns will be assumed.

        :param cols: columns to summarize. If none are given,
        then all the `metric_cols` will be used
        :type cols: str or List[str]
        :return: a :class:`GroupedData` object that can be used for
        summarizing columns/metrics across all observations from all series
        :rtype: :class:`GroupedData`
        """
        if cols is None or len(cols) < 1:
            cols = self.metric_cols
        return self.df.select(cols).groupBy()

    def agg(self, *exprs: Union[Column, Dict[str, str]]) -> DataFrame:
        """

        :param exprs:
        :return:
        """
        return self.df.agg(exprs)

    def describe(self, *cols: Optional[Union[str, List[str]]]) -> DataFrame:
        """

        :param cols:
        :return:
        """
        if cols is None or len(cols) < 1:
            cols = self.metric_cols
        return self.df.describe(cols)

    def metricSummary(self, *statistics: str) -> DataFrame:
        """

        :param statistics:
        :return:
        """
        return self.df.select(self.metric_cols).summary(statistics)

    ## Aggregations by series

    def groupBySeries(self) -> GroupedData:
        """
        Groups the underlying :class:`DataFrame` by the series IDs

        :return: a :class:`GroupedData` object that can be used for
        aggregating within Series
        :rtype: :class:`GroupedData`
        """
        return self.df.groupBy(self.series_ids)

    def aggBySeries(self, *exprs: Union[Column, Dict[str, str]]) -> DataFrame:
        """
        Compute aggregates of each series.

        :param exprs: a dict mapping from column name (string) to aggregate functions (string),
        or a list of :class:`Column`.
        :return: a :class:`DataFrame` of the resulting aggregates
        :rtype: :class:`DataFrame`
        """
        return self.groupBySeries().agg(exprs)

    def applyToSeries(self,
                      func: "PandasGroupedMapFunction",
                      schema: Union[StructType, str]) -> DataFrame:
        """
        Maps each series using a pandas udf and returns the result as a `DataFrame`.

        The function should take a `pandas.DataFrame` and return another
        `pandas.DataFrame`. Alternatively, the user can pass a function that takes
        a tuple of the grouping key(s) and a `pandas.DataFrame`.
        For each group, all columns are passed together as a `pandas.DataFrame`
        to the user-function and the returned `pandas.DataFrame` are combined as a
        :class:`DataFrame`.

        The `schema` should be a :class:`StructType` describing the schema of the returned
        `pandas.DataFrame`. The column labels of the returned `pandas.DataFrame` must either match
        the field names in the defined schema if specified as strings, or match the
        field data types by position if not strings, e.g. integer indices.
        The length of the returned `pandas.DataFrame` can be arbitrary.

        :param func: a Python native function that takes a `pandas.DataFrame` and outputs a
        `pandas.DataFrame`, or that takes one tuple (grouping keys) and a
        `pandas.DataFrame` and outputs a `pandas.DataFrame`.
        :type func: function
        :param schema: the return type of the `func` in PySpark. The value can be either a
        :class:`pyspark.sql.types.DataType` object or a DDL-formatted type string.
        :type schema: :class:`pyspark.sql.types.DataType` or str
        :return: a :class:`pyspark.sql.DataFrame` (of the given schema)
        containing the results of applying the given function per series
        :rtype: :class:`pyspark.sql.DataFrame`
        """
        return self.groupBySeries().applyInPandas(func, schema)

    ### Cyclical Aggregtion

    def groupByCycles(self,
                      length: str,
                      period: Optional[str] = None,
                      offset: Optional[str] = None,
                      bySeries: bool = True) -> GroupedData:
        """

        :param length:
        :param period:
        :param offset:
        :param bySeries:
        :return:
        """
        # build our set of grouping columns
        if bySeries:
            grouping_cols = [sfn.col(series_col) for series_col in self.series_ids]
        else:
            grouping_cols = []
        grouping_cols.append(sfn.window(timeColumn=self.ts_col,
                                        windowDuration=length,
                                        slideDuration=period,
                                        startTime=offset))

        # return the DataFrame grouped accordingly
        return self.df.groupBy(grouping_cols)

    def aggByCycles(self,
                    length: str,
                    *exprs: Union[Column, Dict[str, str]],
                    period: Optional[str] = None,
                    offset: Optional[str] = None,
                    bySeries: bool = True) -> IntervalsDF:
        """

        :param length:
        :param exprs:
        :param period:
        :param offset:
        :param bySeries:
        :return:
        """
        # build aggregated DataFrame
        agged_df = self.groupByCycles(length, period, offset, bySeries).agg(exprs)

        # if we have aggregated over series, we return a TSDF without series
        if bySeries:
            return IntervalsDF.fromNestedBoundariesDF(agged_df,
                                                      "window",
                                                      self.series_ids)
        else:
            return IntervalsDF.fromNestedBoundariesDF(agged_df, "window")

    def applyToCycles(self,
                      length: str,
                      func: "PandasGroupedMapFunction",
                      schema: Union[StructType, str],
                      period: Optional[str] = None,
                      offset: Optional[str] = None,
                      bySeries: bool = True) -> IntervalsDF:
        """

        :param length:
        :param func:
        :param schema:
        :param period:
        :param offset:
        :param bySeries:
        :return:
        """
        # apply function to get DataFrame of results
        applied_df = self.groupByCycles(length, period, offset, bySeries)\
            .applyInPandas(func, schema)

        # if we have applied over series, we return a TSDF without series
        if bySeries:
            return IntervalsDF.fromNestedBoundariesDF(applied_df,
                                                      "window",
                                                      self.series_ids)
        else:
            return IntervalsDF.fromNestedBoundariesDF(applied_df, "window")

    #
    # utility functions
    #

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
            t_utils.calculate_time_horizon(self, freq)

        enriched_df: DataFrame = t_resample.aggregate(
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
        method: str,
        freq: Optional[str] = None,
        func: Optional[Union[Callable | str]] = None,
        target_cols: Optional[List[str]] = None,
        ts_col: Optional[str] = None,
        series_ids: Optional[List[str]] = None,
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
        if series_ids is None:
            series_ids = self.series_ids
        if target_cols is None:
            prohibited_cols: List[str] = series_ids + [ts_col]
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
                return operator_dict[state_definition](a, b)  # type: ignore

        elif callable(state_definition):
            state_comparison_fn = state_definition  # type: ignore

        else:
            raise TypeError(
                f"The `state_definition` argument can be of type `str` or `callable`, "
                f"but received value of type {type(state_definition)}"
            )

        w = self.baseWindow()

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
            data.groupBy(*self.series_ids, "state_incrementer")
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
        series_ids: Optional[List[str]] = None
    ):
        super(_ResampledTSDF, self).__init__(df, ts_col, series_ids)
        self.__freq = freq
        self.__func = func

    def interpolate(
        self,
        method: str,
        freq: Optional[str] = None,
        func: Optional[Union[Callable | str]] = None,
        target_cols: Optional[List[str]] = None,
        ts_col: Optional[str] = None,
        series_ids: Optional[List[str]] = None,
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

        if series_ids is None:
            partition_cols = self.series_ids

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

        interpolate_service = t_interpolation.Interpolation(is_resampled=True)
        tsdf_input = TSDF(self.df, ts_col=self.ts_col, series_ids=self.series_ids)
        interpolated_df = interpolate_service.interpolate(
            tsdf=tsdf_input,
            ts_col=self.ts_col,
            series_ids=self.series_ids,
            target_cols=target_cols,
            freq=freq,
            func=func,
            method=method,
            show_interpolated=show_interpolated,
            perform_checks=perform_checks,
        )

        return TSDF(interpolated_df, ts_col=self.ts_col, series_ids=self.series_ids)


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